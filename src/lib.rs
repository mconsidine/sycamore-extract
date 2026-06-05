//! Fast star centroid extraction for plate solving, tuned for the Raspberry Pi
//! Zero 2 W (Cortex-A53). Exposes `detect_stars(image, ...)` to Python via PyO3
//! with zero-copy access to a C-contiguous uint8 numpy image.
//!
//! The 1-D and 2-D gate tests are closely derived from CedarDetect
//! (https://github.com/smroid/cedar-detect, Apache-2.0) by Steven Rosenthal.
//! This file is an Apache-2.0 derivative work.
//!
//! Pipeline:
//!   1. Cheap noise estimate from dark midline cuts.
//!   2. Parallel (rayon) per-row-band scan: cache-line-sampled row-min prefilter
//!      gates a 7-pixel integer "gate" test producing candidates. The pixel-wise
//!      threshold prefilter is autovectorized by default and can be switched to
//!      an explicit NEON path via `use_neon=True`.
//!   3. Union-find blob assembly over vertically-adjacent candidates.
//!   4. 2-D gate (size / edge / perimeter uniformity / sigma over background
//!      using a perimeter-derived local noise estimate).
//!   5. Background-subtracted separable projection centroid with parabolic
//!      sub-pixel interpolation. With bin=2 and `centroid_full_res=True`,
//!      centroiding is performed on the full-resolution image for sub-pixel
//!      precision (CedarDetect's design).

// Module-level clippy allow. The pyo3 `#[pyfunction]` macro expands to code
// that includes an implicit conversion on the PyErr return path; clippy
// flags this as `useless_conversion` even though the conversion lives inside
// macro-generated code we don't write. Per-function `#[allow]` annotations
// don't reliably reach the warning site when compiling test targets, so we
// scope the suppression at module level.
#![allow(clippy::useless_conversion)]

use numpy::PyReadonlyArray2;
// PyUntypedArrayMethods is required for `.shape()` on PyReadonlyArray; numpy
// 0.22 moved this method behind an explicit trait import. Without it: E0599
// "no method named `shape` found".
use numpy::PyUntypedArrayMethods;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::HashMap;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

const GATE_HALF: usize = 3; // 7-pixel gate => 3 pixels of context each side.

/// Background-floor estimation strategy used by the prefilter and by the 2-D
/// gate's perimeter check.
///
/// `RowPercentile` (default): take the 25th percentile of cache-line-sampled
///   pixels per row. Cheap; robust to partial-black rows and dust shadows.
///   Adequate for clean dark-sky frames with mild vignetting.
///
/// `LineMedian`: compute the true median of each row via a 256-entry histogram
///   pass. Slightly more expensive but normalizes per-row offset noise and
///   vertical brightness gradients (vignetting). Equivalent to olive-solve's
///   FastBgSubMode::LineMedian. Recommended for frames near twilight, with
///   light pollution gradients, or with strong vertical vignetting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BgMode {
    RowPercentile,
    LineMedian,
}

// =========================================================================
// Morphological white top-hat (separable, O(n) per row/column)
// =========================================================================
//
// White top-hat = image − morphological_opening(image), where opening is
// erosion followed by dilation with a flat (box) structuring element of
// the given radius. The opening removes smooth low-frequency structure
// (gradients, vignetting, sky-glow) while leaving high-frequency objects
// (stars) intact. Detection then runs on the residual (near-zero flat
// background with star peaks above it), and centroiding still uses the
// original pixel values for sub-pixel accuracy.
//
// Algorithm: prefix/suffix running min/max within blocks of size w=2r+1
// (van Herk / Gil-Werman). O(n) per 1-D pass, O(1) per pixel. Separable:
// apply horizontally (morph_h), then vertically via transpose → morph_h →
// transpose (morph_v). Four passes total per white-top-hat call.

/// Sliding window minimum or maximum of a 1-D slice.
/// Window size = 2*radius+1. Boundary elements use a reduced (clamped) window.
/// O(n) via the van Herk / Gil-Werman prefix–suffix block trick with neutral
/// padding so boundary windows work correctly without special-casing.
fn extreme_1d(src: &[u8], radius: usize, is_max: bool) -> Vec<u8> {
    let n = src.len();
    if n == 0 {
        return Vec::new();
    }
    let w = 2 * radius + 1;
    let neutral = if is_max { 0u8 } else { 255u8 };

    // Pad by `radius` neutral elements on each side so every window in the
    // padded array is exactly w wide (no boundary asymmetry).
    let pn = n + 2 * radius;
    let mut padded = vec![neutral; pn];
    padded[radius..radius + n].copy_from_slice(src);

    // Prefix: left-to-right running extreme, reset at each block boundary.
    let mut prefix = vec![neutral; pn];
    for i in 0..pn {
        prefix[i] = if i % w == 0 {
            padded[i]
        } else if is_max {
            prefix[i - 1].max(padded[i])
        } else {
            prefix[i - 1].min(padded[i])
        };
    }

    // Suffix: right-to-left running extreme, reset at each block boundary.
    let mut suffix = vec![neutral; pn];
    for i in (0..pn).rev() {
        suffix[i] = if i % w == w - 1 || i == pn - 1 {
            padded[i]
        } else if is_max {
            suffix[i + 1].max(padded[i])
        } else {
            suffix[i + 1].min(padded[i])
        };
    }

    // Output: window [j, j+2r] in padded space maps to output[j] (original index j).
    (0..n)
        .map(|j| {
            let lo = j;
            let hi = j + 2 * radius;
            if is_max {
                suffix[lo].max(prefix[hi])
            } else {
                suffix[lo].min(prefix[hi])
            }
        })
        .collect()
}

/// Apply extreme_1d along every row of a 2-D image (horizontal morphology pass).
fn morph_h(data: &[u8], w: usize, h: usize, radius: usize, is_max: bool) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row_out)| {
        let row_in = &data[y * w..y * w + w];
        row_out.copy_from_slice(&extreme_1d(row_in, radius, is_max));
    });
    out
}

/// Transpose a row-major 2-D image. Returns (transposed_data, new_w, new_h).
/// Each output row (= one input column) is filled independently, so this is
/// safe to parallelise without any synchronisation.
fn transpose(data: &[u8], src_w: usize, src_h: usize) -> (Vec<u8>, usize, usize) {
    let mut out = vec![0u8; src_w * src_h];
    out.par_chunks_mut(src_h).enumerate().for_each(|(x, col_out)| {
        for y in 0..src_h {
            col_out[y] = data[y * src_w + x];
        }
    });
    (out, src_h, src_w)
}

/// Apply extreme_1d along every column of a 2-D image (vertical morphology pass).
/// Implemented as: transpose → morph_h → transpose back.
fn morph_v(data: &[u8], w: usize, h: usize, radius: usize, is_max: bool) -> Vec<u8> {
    let (t, tw, th) = transpose(data, w, h);
    let r = morph_h(&t, tw, th, radius, is_max);
    let (out, _, _) = transpose(&r, tw, th);
    out
}

/// Morphological white top-hat transform: image − opening(image).
/// `opening` = dilate(erode(image)) with a separable flat structuring element
/// of the given radius. Removes broad low-frequency background (gradients,
/// vignetting, sky-glow) while preserving point-source stars.
fn white_tophat(data: &[u8], w: usize, h: usize, radius: usize) -> Vec<u8> {
    let eroded = morph_v(&morph_h(data, w, h, radius, false), w, h, radius, false);
    let opened = morph_v(&morph_h(&eroded, w, h, radius, true), w, h, radius, true);
    data.iter()
        .zip(opened.iter())
        .map(|(&a, &b)| a.saturating_sub(b))
        .collect()
}

#[derive(Copy, Clone)]
struct Candidate {
    x: u32,
    y: u32,
}

struct Star {
    x: f64,
    y: f64,
    brightness: f64,
    peak: u8,
}

// =========================================================================
// 1-D gate: matched filter
// =========================================================================
//
// Standard signal-detection construction. Convolve the 7-pixel window with a
// discrete approximation of the expected stellar PSF (a Gaussian, sigma=1.5,
// FWHM ~3.5 px tuned to representative HQ Camera + finder lens frames), and
// accept pixels whose response exceeds a noise-scaled threshold. This is the
// optimal linear detector for a known-shape pulse in additive Gaussian white
// noise (North 1943, Turin 1960, Van Trees 1968).
//
// Kernel derivation (Python):
//   raw    = exp(-x^2/(2*1.5^2)) for x in -3..3
//          = [0.1353, 0.4111, 0.8007, 1.0, 0.8007, 0.4111, 0.1353]
//   zm     = raw - mean(raw)              (mean-removed Gaussian)
//   scale  = 60 / zm[center]              (target center coefficient ~60)
//   k      = round(zm * scale), symmetrized
//          = [-50, -15, 35, 60, 35, -15, -50]   sum = 0, symmetric
//
// k sums to 0, so a uniform background contributes 0 to the response — no
// separate background-subtraction step required. The response is the dot
// product <window, k>, an i32. For pure white noise of std `noise`, the
// response has std `noise * ||k||_2`. Threshold is `sigma * noise * ||k||_2`.
//
// Kernel norm:
//   sqrt(50^2 + 15^2 + 35^2 + 60^2 + 35^2 + 15^2 + 50^2)
//   = sqrt(11500) ~= 107.24
//
// Empirical behavior on real frames: the matched filter is somewhat
// conservative — its threshold derivation assumes pure Gaussian noise, but
// real-sky frames have correlated noise structure that adds variance to
// the matched-filter response. Users on real-sky data may need to lower
// sigma (typically by 1-2) below the conventional sigma=8 default to detect
// faint stars near the noise floor. Field validation is encouraged before
// committing to a specific default sigma for new sensor/optics combinations.
// See ARCHITECTURE.md for the calibration history.

// Local-maximum suppression: avoid double-claiming neighbors that are part
// of the same star. Three-pixel local-max test with deterministic tie-breaks
// so the same star always picks the same representative pixel.
#[inline(always)]
fn gate_1d(g: &[u8], mf_thresh: i32) -> bool {
    let g0 = g[0] as i32;
    let g1 = g[1] as i32;
    let g2 = g[2] as i32;
    let g3 = g[3] as i32;
    let g4 = g[4] as i32;
    let g5 = g[5] as i32;
    let g6 = g[6] as i32;

    // Dot product with kernel [-50, -15, 35, 60, 35, -15, -50].
    let response = -50 * (g0 + g6) - 15 * (g1 + g5) + 35 * (g2 + g4) + 60 * g3;
    if response < mf_thresh {
        return false;
    }
    // Local-maximum suppression in raw pixel space.
    if g2 > g3 || g3 < g4 {
        return false;
    }
    // Deterministic tie-breaks for flat-topped peaks.
    if g2 == g3 && g1 > g4 {
        return false;
    }
    if g3 == g4 && g2 <= g5 {
        return false;
    }
    true
}

// Compute the matched-filter threshold from sigma/noise. Kept as a function
// so the caller can compute it once per band, not per pixel.
#[inline]
fn mf_threshold(sigma: f64, noise: f64) -> i32 {
    // Kernel L2 norm: ~107.24 (sigma=1.5 Gaussian, see gate_1d derivation).
    // We use 107 (slight conservative round) to avoid f64 math in the hot path.
    let t = sigma * noise * 107.0 + 0.5;
    t.clamp(1.0, i32::MAX as f64) as i32
}

// =========================================================================
// Threshold prefilter: scalar (autovectorized by LLVM) vs explicit NEON
// =========================================================================
// Both fill `hits` with x-coords where row[x] >= thresh, restricted to
// [GATE_HALF, width-GATE_HALF). The center loop in scan_band is data-dependent
// (only on threshold-hit pixels do we descend into the 7-pixel gate), so the
// SIMD-friendly part is purely the byte-wise compare. The hand-coded NEON
// version stops trusting the autovectorizer and uses 16 bytes per `cmhs`.

#[inline]
fn threshold_scan_scalar(row: &[u8], thresh: u8, hits: &mut Vec<u32>) {
    let n = row.len();
    if n < 2 * GATE_HALF + 1 {
        return;
    }
    // We want explicit x indices to push to `hits`; the clippy-suggested
    // iter().enumerate() rewrite is the same shape but less clear here.
    #[allow(clippy::needless_range_loop)]
    for x in GATE_HALF..(n - GATE_HALF) {
        if row[x] >= thresh {
            hits.push(x as u32);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn threshold_scan_neon(row: &[u8], thresh: u8, hits: &mut Vec<u32>) {
    let n = row.len();
    if n < 2 * GATE_HALF + 1 {
        return;
    }
    let start = GATE_HALF;
    let end = n - GATE_HALF;

    // We need x in [start, end). Process 16 bytes at a time aligned to byte
    // offsets within the row. The vector compares row[i..i+16] >= thresh and
    // emits hit indices in raster order. Pixels outside [start, end) are masked
    // out by adjusting the bookends.
    let vthr = vdupq_n_u8(thresh);
    let mut i = start;

    // Bulk: process 16-byte groups that fit entirely inside [start, end).
    let bulk_end = if end >= 16 { end - 16 } else { start };
    while i <= bulk_end {
        let v = vld1q_u8(row.as_ptr().add(i));
        // cmhs: unsigned >= compare. Each lane: 0xFF if >=, else 0x00.
        let cmp = vcgeq_u8(v, vthr);

        // Extract a 16-bit mask of which lanes hit. Standard trick: shrink by
        // shift-right-by-7 + horizontal narrow so each lane becomes a single
        // bit, then move to a GPR.
        let shr = vshrq_n_u8(cmp, 7);
        // Pairwise-add into a u16x8, then narrow into a u8x8 whose 8 lanes
        // each carry the OR of two original lanes; do it twice to fold 16->1.
        let p1 = vpaddlq_u8(shr); // u16x8 each = sum of 2 lanes
        let p2 = vpaddlq_u16(p1); // u32x4 each = sum of 4 lanes
        let p3 = vpaddlq_u32(p2); // u64x2 each = sum of 8 lanes
        let lo = vgetq_lane_u64::<0>(p3);
        let hi = vgetq_lane_u64::<1>(p3);
        if (lo | hi) != 0 {
            // Some lane hit; fall back to a tiny scalar pass over this 16-byte
            // window to record exact x-indices in order. The pass is short and
            // branch-predictor-friendly because the test it repeats was just
            // proven true overall.
            for k in 0..16 {
                if *row.get_unchecked(i + k) >= thresh {
                    hits.push((i + k) as u32);
                }
            }
        }
        i += 16;
    }

    // Tail: scalar finish for the last <16 elements up to `end`.
    while i < end {
        if *row.get_unchecked(i) >= thresh {
            hits.push(i as u32);
        }
        i += 1;
    }
}

// Scan one band of rows [y0, y1) producing candidates in raster order.
// Compute one median value per row, via a 256-entry histogram. Used by
// BgMode::LineMedian as the row background floor. Parallelizable: each row
// is independent and histogram fits in L1.
fn compute_row_medians(data: &[u8], width: usize, height: usize) -> Vec<u8> {
    use rayon::prelude::*;
    let mut out = vec![0u8; height];
    out.par_iter_mut().enumerate().for_each(|(y, slot)| {
        let row = &data[y * width..y * width + width];
        let mut hist = [0u32; 256];
        for &p in row {
            hist[p as usize] += 1;
        }
        let target = (width as u32).div_ceil(2);
        let mut acc = 0u32;
        for (v, &c) in hist.iter().enumerate() {
            acc += c;
            if acc >= target {
                *slot = v as u8;
                break;
            }
        }
    });
    out
}

/// Compute per-column median of a 2-D uint8 image. Returns a Vec<u8> of length `width`.
fn compute_col_medians(data: &[u8], width: usize, height: usize) -> Vec<u8> {
    let (t, tw, th) = transpose(data, width, height);
    compute_row_medians(&t, tw, th)
}

/// Subtract a per-row floor from every pixel (saturating). Returns a new image.
fn subtract_row_floor(data: &[u8], width: usize, height: usize, floors: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; width * height];
    out.par_chunks_mut(width).enumerate().for_each(|(y, row_out)| {
        let row_in = &data[y * width..y * width + width];
        let f = floors[y];
        for (o, &p) in row_out.iter_mut().zip(row_in.iter()) {
            *o = p.saturating_sub(f);
        }
    });
    out
}

/// Subtract a per-column floor from every pixel (saturating). Returns a new image.
fn subtract_col_floor(data: &[u8], width: usize, height: usize, floors: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; width * height];
    out.par_chunks_mut(width).enumerate().for_each(|(y, row_out)| {
        let row_in = &data[y * width..y * width + width];
        for (x, (o, &p)) in row_out.iter_mut().zip(row_in.iter()).enumerate() {
            *o = p.saturating_sub(floors[x]);
        }
    });
    out
}

/// Bilinear block-median background subtraction.
///
/// Divides the image into `block_size × block_size` tiles. For each tile the
/// median pixel value is computed. A smooth background surface is reconstructed
/// via bilinear interpolation between tile centers and subtracted (saturating).
/// Removes large-scale 2-D gradients (vignetting, sky-glow, light pollution)
/// while preserving point sources.
///
/// `block_size` must comfortably exceed the largest expected star radius so no
/// star flux is captured in the tile median. Typical values: 32 (bin=2), 64
/// (bin=1). Parallelised: per-tile medians and per-row interpolation fan out.
fn block_percentile_bg(data: &[u8], width: usize, height: usize, block_size: usize) -> Vec<u8> {
    let bs = block_size.max(2);
    let nx = width.div_ceil(bs);
    let ny = height.div_ceil(bs);

    let block_medians: Vec<u8> = (0..nx * ny)
        .into_par_iter()
        .map(|bi| {
            let bx = bi % nx;
            let by = bi / nx;
            let x0 = bx * bs;
            let y0 = by * bs;
            let x1 = (x0 + bs).min(width);
            let y1 = (y0 + bs).min(height);
            let mut vals: Vec<u8> = Vec::with_capacity(bs * bs);
            for y in y0..y1 {
                vals.extend_from_slice(&data[y * width + x0..y * width + x1]);
            }
            vals.sort_unstable();
            vals[vals.len() / 2]
        })
        .collect();

    let half_bs = bs as f32 / 2.0;
    let mut out = vec![0u8; width * height];
    out.par_chunks_mut(width).enumerate().for_each(|(y, row_out)| {
        let row_in = &data[y * width..y * width + width];
        let by_f = (y as f32 - half_bs) / bs as f32;
        let by0 = (by_f.floor() as isize).max(0).min(ny as isize - 1) as usize;
        let by1 = (by0 + 1).min(ny - 1);
        let fy = (by_f - by0 as f32).clamp(0.0, 1.0);
        for (x, (o, &pixel)) in row_out.iter_mut().zip(row_in.iter()).enumerate() {
            let bx_f = (x as f32 - half_bs) / bs as f32;
            let bx0 = (bx_f.floor() as isize).max(0).min(nx as isize - 1) as usize;
            let bx1 = (bx0 + 1).min(nx - 1);
            let fx = (bx_f - bx0 as f32).clamp(0.0, 1.0);
            let m00 = block_medians[by0 * nx + bx0] as f32;
            let m10 = block_medians[by0 * nx + bx1] as f32;
            let m01 = block_medians[by1 * nx + bx0] as f32;
            let m11 = block_medians[by1 * nx + bx1] as f32;
            let bg = m00 * (1.0 - fx) * (1.0 - fy)
                + m10 * fx * (1.0 - fy)
                + m01 * (1.0 - fx) * fy
                + m11 * fx * fy;
            *o = pixel.saturating_sub(bg as u8);
        }
    });
    out
}

/// Apply spatial background preprocessing, returning Some(corrected_image) or
/// None if no preprocessing is needed.
enum SpatialBg {
    None,
    TopHat(usize),
    ColPercentile,
    RowColPercentile,
    BlockPercentile(usize),
}

fn apply_spatial_bg(img: &[u8], w: usize, h: usize, mode: &SpatialBg) -> Option<Vec<u8>> {
    match mode {
        SpatialBg::None => None,
        SpatialBg::TopHat(r) => Some(white_tophat(img, w, h, *r)),
        SpatialBg::ColPercentile => {
            let cols = compute_col_medians(img, w, h);
            Some(subtract_col_floor(img, w, h, &cols))
        }
        SpatialBg::RowColPercentile => {
            // Row correction first (dominant CMOS readout artifact), then column.
            let rows = compute_row_medians(img, w, h);
            let row_corrected = subtract_row_floor(img, w, h, &rows);
            let cols = compute_col_medians(&row_corrected, w, h);
            Some(subtract_col_floor(&row_corrected, w, h, &cols))
        }
        SpatialBg::BlockPercentile(bs) => Some(block_percentile_bg(img, w, h, *bs)),
    }
}

// Scan one band of rows [y0, y1) producing candidates in raster order.
// `row_floors`, if Some, supplies a precomputed per-row background floor
// (e.g. from LineMedian). If None, the in-loop 25th-percentile of cache-line
// samples is used.
// `sn2` (≈ sigma*noise) sets the cheap prefilter cutoff above the row floor
// — every pixel above it gets the full matched-filter evaluation.
// `mf_thresh` is the precomputed matched-filter response threshold.
#[allow(clippy::too_many_arguments)]
fn scan_band(
    data: &[u8],
    width: usize,
    y0: usize,
    y1: usize,
    sn2: i32,
    use_neon: bool,
    row_floors: Option<&[u8]>,
    mf_thresh: i32,
) -> Vec<Candidate> {
    let half = (sn2 / 2).clamp(0, 255) as u8;
    let mut out = Vec::new();
    let mut hits: Vec<u32> = Vec::with_capacity(64);

    for y in y0..y1 {
        let row = &data[y * width..y * width + width];

        // Row background floor:
        //   - If row_floors is provided (LineMedian path), use the precomputed
        //     per-row median.
        //   - Otherwise, take a 25th-percentile of cache-line-sampled pixels
        //     (RowPercentile path). Robust to partial-black rows and dust
        //     shadows; cheaper than a full median.
        let row_floor = if let Some(rf) = row_floors {
            rf[y]
        } else {
            let mut sbuf: [u8; 64] = [0; 64];
            let mut sn = 0usize;
            let mut i = 0;
            while i < width && sn < 64 {
                sbuf[sn] = row[i];
                sn += 1;
                i += 64;
            }
            if sn == 0 {
                0u8
            } else {
                sbuf[..sn].sort_unstable();
                sbuf[sn / 4]
            }
        };
        let thresh = row_floor.saturating_add(half);

        hits.clear();
        if use_neon {
            #[cfg(target_arch = "aarch64")]
            unsafe {
                threshold_scan_neon(row, thresh, &mut hits);
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                threshold_scan_scalar(row, thresh, &mut hits);
            }
        } else {
            threshold_scan_scalar(row, thresh, &mut hits);
        }

        // For each pixel that cleared the prefilter, run the matched-filter gate.
        for &x in &hits {
            let x = x as usize;
            if x < GATE_HALF || x + GATE_HALF >= width {
                continue;
            }
            let g = &row[x - GATE_HALF..x + GATE_HALF + 1];
            if gate_1d(g, mf_thresh) {
                out.push(Candidate {
                    x: x as u32,
                    y: y as u32,
                });
            }
        }
    }
    out
}

// =========================================================================
// Blob assembly (union-find over vertically-adjacent candidates)
// =========================================================================
struct Dsu {
    parent: Vec<usize>,
}
impl Dsu {
    fn new(n: usize) -> Self {
        Dsu {
            parent: (0..n).collect(),
        }
    }
    fn find(&mut self, mut a: usize) -> usize {
        while self.parent[a] != a {
            self.parent[a] = self.parent[self.parent[a]]; // path halving
            a = self.parent[a];
        }
        a
    }
    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.parent[ra] = rb;
        }
    }
}

fn form_blobs(cands: &[Candidate], height: usize) -> Vec<Vec<usize>> {
    let n = cands.len();
    let mut dsu = Dsu::new(n);
    let mut by_row: Vec<Vec<usize>> = vec![Vec::new(); height];
    for (i, c) in cands.iter().enumerate() {
        by_row[c.y as usize].push(i);
    }
    for y in 1..height {
        let (prev, cur) = (&by_row[y - 1], &by_row[y]);
        for &ci in cur {
            let cx = cands[ci].x as i64;
            for &pi in prev {
                let px = cands[pi].x as i64;
                if px < cx - 3 {
                    continue;
                }
                if px > cx + 3 {
                    break;
                }
                dsu.union(ci, pi);
            }
        }
    }
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let r = dsu.find(i);
        groups.entry(r).or_default().push(i);
    }
    groups.into_values().collect()
}

// =========================================================================
// Box / ring statistics
// =========================================================================
#[inline]
fn box_mean(data: &[u8], w: usize, x0: usize, y0: usize, bw: usize, bh: usize) -> f64 {
    let mut s = 0u64;
    for y in y0..y0 + bh {
        let row = &data[y * w + x0..y * w + x0 + bw];
        for &p in row {
            s += p as u64;
        }
    }
    s as f64 / (bw * bh) as f64
}

// mean, min, max, stddev over the 1-pixel-thick border of a box.
fn ring_stats(
    data: &[u8],
    w: usize,
    x0: usize,
    y0: usize,
    bw: usize,
    bh: usize,
) -> (f64, u8, u8, f64) {
    let mut sum = 0f64;
    let mut sumsq = 0f64;
    let mut n = 0f64;
    let mut mn = 255u8;
    let mut mx = 0u8;
    let mut acc = |p: u8| {
        let pf = p as f64;
        sum += pf;
        sumsq += pf * pf;
        n += 1.0;
        mn = mn.min(p);
        mx = mx.max(p);
    };
    for x in x0..x0 + bw {
        acc(data[y0 * w + x]);
        acc(data[(y0 + bh - 1) * w + x]);
    }
    for y in (y0 + 1)..(y0 + bh - 1) {
        acc(data[y * w + x0]);
        acc(data[y * w + x0 + bw - 1]);
    }
    let mean = sum / n;
    let var = (sumsq / n - mean * mean).max(0.0);
    (mean, mn, mx, var.sqrt())
}

// 3-point parabolic peak of a 1-D projection. Returns sub-pixel index.
fn parabolic_peak(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut pi = 0usize;
    let mut pv = v[0];
    for (i, &x) in v.iter().enumerate() {
        if x > pv {
            pv = x;
            pi = i;
        }
    }
    if pi == 0 || pi == v.len() - 1 {
        return pi as f64;
    }
    let (a, b, c) = (v[pi - 1], v[pi], v[pi + 1]);
    let denom = a - 2.0 * b + c;
    if denom.abs() < 1e-12 {
        return pi as f64;
    }
    pi as f64 + (0.5 * (a - c) / denom).clamp(-0.5, 0.5)
}

// =========================================================================
// 2-D gate + centroid
// =========================================================================
// `det_img`/`det_w`/`det_h` is the image used for detection (possibly binned).
// `cent_img`/`cent_w`/`cent_h` is the image used for centroiding (full-res when
// we want sub-pixel precision after bin=2 detection). `scale` maps detection
// coords to centroid coords (1 if no binning, 2 if detect-on-binned/centroid-
// on-full-res).
#[allow(clippy::too_many_arguments)]
fn gate_2d(
    cands: &[Candidate],
    blob: &[usize],
    det_img: &[u8],
    det_w: usize,
    det_h: usize,
    cent_img: &[u8],
    cent_w: usize,
    cent_h: usize,
    scale: usize,
    noise: f64,
    sigma: f64,
    max_size: usize,
    max_axis_ratio: f64,
) -> Option<Star> {
    let mut x_min = u32::MAX;
    let mut x_max = 0u32;
    let mut y_min = u32::MAX;
    let mut y_max = 0u32;
    for &i in blob {
        let c = cands[i];
        x_min = x_min.min(c.x);
        x_max = x_max.max(c.x);
        y_min = y_min.min(c.y);
        y_max = y_max.max(c.y);
    }
    let (cx0, cy0) = (x_min as usize, y_min as usize);
    let cw = (x_max - x_min) as usize + 1;
    let ch = (y_max - y_min) as usize + 1;

    if cw > max_size || ch > max_size {
        return None; // extended object
    }
    if cx0 < 3 || cy0 < 3 || cx0 + cw + 3 > det_w || cy0 + ch + 3 > det_h {
        return None;
    }

    // Detection-image gates use the detection image (binned or not).
    let core_mean = box_mean(det_img, det_w, cx0, cy0, cw, ch);
    let (bg_det, p_min, p_max, p_std) =
        ring_stats(det_img, det_w, cx0 - 3, cy0 - 3, cw + 6, ch + 6);

    if (p_max as f64 - p_min as f64) > 3.0 * sigma * noise {
        return None; // perimeter not uniform
    }
    let local_noise = noise.max(p_std);
    if core_mean - bg_det < sigma * local_noise {
        return None; // core not bright enough above local background
    }

    // ---- Centroid on the centroid image, in its coordinates.
    // Map the margin box from detection coords to centroid coords.
    let mx0 = (cx0 - 2) * scale;
    let my0 = (cy0 - 2) * scale;
    let mw = (cw + 4) * scale;
    let mh = (ch + 4) * scale;
    if mx0 + mw > cent_w || my0 + mh > cent_h {
        return None; // shouldn't happen given the 3-px detection ring check
    }

    // Background for centroiding: re-estimate on the centroid image perimeter
    // for sub-pixel correctness.
    let (bg_cent, _, _, _) = ring_stats(
        cent_img,
        cent_w,
        mx0.saturating_sub(1),
        my0.saturating_sub(1),
        mw + 2,
        mh + 2,
    );

    let mut hproj = vec![0f64; mw];
    let mut vproj = vec![0f64; mh];
    let mut brightness = 0f64;
    let mut peak = 0u8;
    for yy in 0..mh {
        let row = &cent_img[(my0 + yy) * cent_w..(my0 + yy) * cent_w + cent_w];
        for xx in 0..mw {
            let p = row[mx0 + xx];
            peak = peak.max(p);
            let val = (p as f64 - bg_cent).max(0.0);
            hproj[xx] += val;
            vproj[yy] += val;
            brightness += val;
        }
    }

    // Cheap axis-ratio rejection. Derive the second moments of the centroid
    // box from the already-computed separable projections. This catches trails
    // (long in one direction, short in the other) and grossly bloomed stars
    // without needing a full 2-D moment pass.
    //
    // Note: this is a strict subset of olive-solve's eigendecomposition
    // (we don't compute m2_xy, the off-diagonal moment, because projections
    // don't give it cheaply). For a finder this is plenty; for astrometry it
    // would not be.
    if brightness > 0.0 && max_axis_ratio.is_finite() && max_axis_ratio > 1.0 {
        let inv_b = 1.0 / brightness;
        let mut m1x = 0f64;
        for (i, &v) in hproj.iter().enumerate() {
            m1x += (i as f64) * v;
        }
        m1x *= inv_b;
        let mut m1y = 0f64;
        for (j, &v) in vproj.iter().enumerate() {
            m1y += (j as f64) * v;
        }
        m1y *= inv_b;
        let mut var_x = 0f64;
        for (i, &v) in hproj.iter().enumerate() {
            let d = i as f64 - m1x;
            var_x += d * d * v;
        }
        var_x *= inv_b;
        let mut var_y = 0f64;
        for (j, &v) in vproj.iter().enumerate() {
            let d = j as f64 - m1y;
            var_y += d * d * v;
        }
        var_y *= inv_b;
        let major2 = var_x.max(var_y);
        let minor2 = var_x.min(var_y).max(1e-6);
        // (major/minor)^2 > threshold^2; avoid sqrt.
        if major2 / minor2 > max_axis_ratio * max_axis_ratio {
            return None;
        }
    }

    let cx = mx0 as f64 + parabolic_peak(&hproj) + 0.5;
    let cy = my0 as f64 + parabolic_peak(&vproj) + 0.5;

    Some(Star {
        x: cx,
        y: cy,
        brightness,
        peak,
    })
}

// =========================================================================
// Noise estimate
// =========================================================================
// Robust noise estimator: sample several 2-D patches scattered across the
// frame, drop ones contaminated by bright sources, compute median-absolute-
// deviation (MAD) on each survivor, then take the median MAD * 1.4826 as the
// noise sigma. This is robust to vignetting, partial-black bars from frame
// shifts, dust shadows, and bright interlopers in any single patch.
fn estimate_noise(data: &[u8], w: usize, h: usize) -> f64 {
    const MIN_NOISE: f64 = 0.5;
    if w < 64 || h < 64 {
        return MIN_NOISE;
    }
    // 3x3 grid of 32x32 patches placed in the middle ~80% of the frame; this
    // avoids the edges (vignetting, sensor masking) without being so central
    // that a single bright object dominates.
    let patch = 32usize;
    let xs = [w / 5, w / 2 - patch / 2, 4 * w / 5 - patch];
    let ys = [h / 5, h / 2 - patch / 2, 4 * h / 5 - patch];
    let mut mads: Vec<f64> = Vec::with_capacity(9);

    for &py in &ys {
        for &px in &xs {
            if px + patch > w || py + patch > h {
                continue;
            }
            // Collect patch into a small buffer (32*32 = 1024 u8s).
            let mut buf: [u8; 1024] = [0; 1024];
            for r in 0..patch {
                let src = &data[(py + r) * w + px..(py + r) * w + px + patch];
                buf[r * patch..r * patch + patch].copy_from_slice(src);
            }
            let n = patch * patch;
            // Mean & stddev for a cheap contamination check.
            let mut sum = 0u32;
            for &p in &buf[..n] {
                sum += p as u32;
            }
            let mean = sum as f64 / n as f64;
            let mut ss = 0f64;
            for &p in &buf[..n] {
                ss += (p as f64 - mean).powi(2);
            }
            let std = (ss / n as f64).sqrt();
            // Skip patches that look heavily contaminated (a star or hot spot
            // in the patch). Empirically: if max > mean + 6*std the patch has
            // a bright outlier; skip and let other patches speak.
            let mut pmax = 0u8;
            for &p in &buf[..n] {
                if p > pmax {
                    pmax = p;
                }
            }
            if pmax as f64 > mean + 6.0 * std.max(1.0) {
                continue;
            }
            // MAD: median(|x - median(x)|). u8 -> 256-bin histogram for an
            // O(n) median; on 1024 samples this is plenty fast.
            let mut hist = [0u32; 256];
            for &p in &buf[..n] {
                hist[p as usize] += 1;
            }
            let half = (n / 2) as u32;
            let mut acc = 0u32;
            let mut median = 0u8;
            for (v, &c) in hist.iter().enumerate() {
                acc += c;
                if acc >= half {
                    median = v as u8;
                    break;
                }
            }
            // Histogram of |x - median|.
            let mut dhist = [0u32; 256];
            for &p in &buf[..n] {
                let d = (p as i32 - median as i32).unsigned_abs() as usize;
                dhist[d] += 1;
            }
            acc = 0;
            let mut mad = 0u8;
            for (v, &c) in dhist.iter().enumerate() {
                acc += c;
                if acc >= half {
                    mad = v as u8;
                    break;
                }
            }
            mads.push(mad as f64);
        }
    }
    if mads.is_empty() {
        return MIN_NOISE;
    }
    mads.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let med_mad = mads[mads.len() / 2];
    // 1.4826 * MAD converts MAD to an estimate of the Gaussian sigma.
    (1.4826 * med_mad).max(MIN_NOISE)
}

// =========================================================================
// 2x2 mean bin
// =========================================================================
fn bin2x2_mean(data: &[u8], w: usize, h: usize) -> (Vec<u8>, usize, usize) {
    let wb = w / 2;
    let hb = h / 2;
    let mut out = vec![0u8; wb * hb];
    for by in 0..hb {
        let r0 = &data[(2 * by) * w..(2 * by) * w + w];
        let r1 = &data[(2 * by + 1) * w..(2 * by + 1) * w + w];
        let orow = &mut out[by * wb..by * wb + wb];
        for bx in 0..wb {
            let s = r0[2 * bx] as u32
                + r0[2 * bx + 1] as u32
                + r1[2 * bx] as u32
                + r1[2 * bx + 1] as u32;
            orow[bx] = ((s + 2) / 4) as u8;
        }
    }
    (out, wb, hb)
}

// =========================================================================
// Top-level detection
// =========================================================================
#[allow(clippy::too_many_arguments)]
fn detect(
    det_img: &[u8],
    det_w: usize,
    det_h: usize,
    cent_img: &[u8],
    cent_w: usize,
    cent_h: usize,
    scale: usize,
    sigma: f64,
    noise: f64,
    use_neon: bool,
    bg_mode: BgMode,
    max_axis_ratio: f64,
) -> Vec<Star> {
    if det_w < 7 || det_h < 7 {
        return Vec::new();
    }
    let sn2 = ((2.0 * sigma * noise + 0.5) as i32).max(2);
    let mf_thresh = mf_threshold(sigma, noise);

    // LineMedian: precompute one median per row in parallel. This is the
    // equivalent of olive-solve's FastBgSubMode::LineMedian and handles:
    //   - per-row offset noise (some CMOS sensors, notably IMX296mono),
    //   - vertical brightness gradients from vignetting/light pollution,
    //   - black-bar artifacts (median is unaffected by up to 49% bad pixels
    //     per row, vs the row-percentile's ~20%).
    let row_floors: Option<Vec<u8>> = match bg_mode {
        BgMode::LineMedian => Some(compute_row_medians(det_img, det_w, det_h)),
        BgMode::RowPercentile => None,
    };

    // Parallel row-band scan.
    let n_bands = rayon::current_num_threads().max(1);
    let band_rows = det_h.div_ceil(n_bands);
    let banded: Vec<Vec<Candidate>> = (0..n_bands)
        .into_par_iter()
        .map(|b| {
            let y0 = b * band_rows;
            let y1 = ((b + 1) * band_rows).min(det_h);
            if y0 >= y1 {
                Vec::new()
            } else {
                scan_band(
                    det_img,
                    det_w,
                    y0,
                    y1,
                    sn2,
                    use_neon,
                    row_floors.as_deref(),
                    mf_thresh,
                )
            }
        })
        .collect();
    let cands: Vec<Candidate> = banded.into_iter().flatten().collect();
    if cands.is_empty() {
        return Vec::new();
    }

    let max_size = (det_w / 100).max(3);
    let blobs = form_blobs(&cands, det_h);

    let mut stars: Vec<Star> = blobs
        .par_iter()
        .filter_map(|blob| {
            gate_2d(
                &cands,
                blob,
                det_img,
                det_w,
                det_h,
                cent_img,
                cent_w,
                cent_h,
                scale,
                noise,
                sigma,
                max_size,
                max_axis_ratio,
            )
        })
        .collect();

    stars.sort_by(|a, b| {
        b.brightness
            .partial_cmp(&a.brightness)
            .unwrap_or(Ordering::Equal)
    });
    stars
}

// =========================================================================
// Python entry point
// =========================================================================

// A dedicated rayon thread pool with a small, bounded number of threads. The
// default (2 on a Pi Zero 2W) is a UX choice: it keeps the other two cores
// available for camera capture, SkySafari I/O, IMU, web handlers, etc., so a
// detection burst doesn't preempt everything else on the system.
//
// Built lazily on first use; size is read from STAR_DETECT_THREADS env var or
// can be overridden at runtime via `star_detect.set_num_threads(n)`.
use std::sync::OnceLock;
static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();

fn default_threads() -> usize {
    if let Ok(s) = std::env::var("STAR_DETECT_THREADS") {
        if let Ok(n) = s.parse::<usize>() {
            if n >= 1 {
                return n.min(64);
            }
        }
    }
    2 // sensible default for Pi Zero 2W
}

fn get_pool() -> &'static rayon::ThreadPool {
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(default_threads())
            .thread_name(|i| format!("star_detect-{}", i))
            .build()
            .expect("failed to build star_detect rayon pool")
    })
}

/// Override the number of worker threads used by detect_stars. Must be called
/// before the first detect_stars call (the pool is initialized lazily on first
/// use and is immutable afterward). Raises ValueError if called too late.
#[pyfunction]
fn set_num_threads(n: usize) -> PyResult<()> {
    if POOL.get().is_some() {
        return Err(PyValueError::new_err(
            "set_num_threads must be called before the first detect_stars() call",
        ));
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n.clamp(1, 64))
        .thread_name(|i| format!("star_detect-{}", i))
        .build()
        .map_err(|e| PyValueError::new_err(format!("rayon pool build failed: {e}")))?;
    POOL.set(pool)
        .map_err(|_| PyValueError::new_err("thread pool already initialized"))?;
    Ok(())
}

/// Detect stars in a grayscale uint8 image.
///
/// Args:
///   image: 2-D C-contiguous numpy uint8 array (height, width).
///   sigma: detection threshold in noise sigmas (typical 5-10; default 8).
///          The matched filter is calibrated for sigma=8 on representative
///          HQ Camera frames. On real-sky data with lots of correlated noise
///          structure, you may need to lower sigma to 6-7 to catch faint
///          stars near the noise floor. Field validation is encouraged.
///   noise: optional precomputed noise level; estimated if None.
///   bin:   1 (full res) or 2 (detect on a 2x2-binned image for speed). Centroids
///          are returned in full-resolution pixel coordinates regardless.
///   centroid_full_res: when bin=2, perform centroiding on the full-resolution
///          image. Slightly more work per star, but recovers the sub-pixel
///          precision that scaling-from-binned loses. No effect when bin=1.
///   use_neon: when true and built for aarch64, use the explicit NEON threshold
///          prefilter. No-op on non-aarch64 targets.
///   bg_mode: background subtraction strategy. Options:
///     "row_percentile" (default) — cheapest; per-row 25th-percentile floor
///       estimated inline per row. Works well on dark-sky frames with mild
///       horizontal gradients.
///     "line_median" — true per-row median (256-bin histogram, parallel);
///       more robust to per-row offset noise and vertical gradients.
///     "top_hat" — morphological white top-hat (image − opening) before
///       detection; removes 2-D vignetting/sky-glow. Centroids are taken on
///       the original image. Controlled by tophat_radius (default 12).
///     "column_percentile" — subtracts per-column median before detection,
///       removing vertical gradients (lens falloff, sky-glow top-to-bottom).
///       Scan-band RowPercentile handles residual horizontal variation.
///     "row_column_percentile" — row median subtraction followed by column
///       median subtraction on the residual; separable 2-D background removal
///       at ~2× the cost of line_median. Good all-round substitute for top_hat
///       when the gradient is approximately separable.
///     "block_percentile" — bilinear interpolation of per-tile medians;
///       removes non-separable 2-D gradients (vignetting, nebulosity, light
///       pollution) at a fraction of top_hat's cost. Tile size controlled by
///       bg_block_size (default 32 at bin=2; scale up for bin=1).
///   max_axis_ratio: optional cap on detected blob elongation. A trail or
///          satellite streak has axis_ratio >> 1; an in-focus star is ~1.
///          Default Inf (no filtering). Recommend 3-5 for a finder.
///   tophat_radius: structuring-element radius for the white top-hat (pixels,
///          default 0 = use 12). Must exceed the largest star radius. Only
///          used when bg_mode="top_hat".
///   bg_block_size: tile side length in pixels for block_percentile mode
///          (default 0 = use 32). Must comfortably exceed the largest star
///          radius. Typical: 32 (bin=2 detection image), 64 (bin=1).
///
/// Concurrency:
///   The image is copied into a Rust-owned buffer up front; the GIL is then
///   released for the duration of the compute, so other Python threads (e.g.
///   the SkySafari handler, IMU reader, web handlers) can run concurrently.
///   Internal parallelism is bounded by the configured thread pool (default 2).
///
/// Returns: list of (x, y, brightness, peak_value), brightest first.
///          (0.5, 0.5) is the center of the top-left pixel.
#[pyfunction]
#[pyo3(signature = (
    image,
    sigma=8.0,
    noise=None,
    bin=1,
    centroid_full_res=true,
    use_neon=false,
    bg_mode="row_percentile",
    max_axis_ratio=f64::INFINITY,
    tophat_radius=0u32,
    bg_block_size=0u32,
))]
#[allow(clippy::too_many_arguments)]
fn detect_stars(
    py: Python<'_>,
    image: PyReadonlyArray2<u8>,
    sigma: f64,
    noise: Option<f64>,
    bin: u32,
    centroid_full_res: bool,
    use_neon: bool,
    bg_mode: &str,
    max_axis_ratio: f64,
    tophat_radius: u32,
    bg_block_size: u32,
) -> PyResult<Vec<(f64, f64, f64, i64)>> {
    let shape = image.shape();
    let (h, w) = (shape[0], shape[1]);
    if shape.len() != 2 {
        return Err(PyValueError::new_err("image must be 2-D"));
    }

    let (bg, spatial) = match bg_mode.to_ascii_lowercase().as_str() {
        "row_percentile" | "rowpercentile" | "percentile" | "default" => {
            (BgMode::RowPercentile, SpatialBg::None)
        }
        "line_median" | "linemedian" | "row_median" => (BgMode::LineMedian, SpatialBg::None),
        "top_hat" | "tophat" | "white_tophat" => {
            let r = if tophat_radius == 0 { 12 } else { tophat_radius as usize };
            (BgMode::RowPercentile, SpatialBg::TopHat(r))
        }
        "column_percentile" | "col_percentile" | "colpercentile" => {
            (BgMode::RowPercentile, SpatialBg::ColPercentile)
        }
        "row_column_percentile" | "rowcolumnpercentile" | "row_col_percentile" => {
            (BgMode::RowPercentile, SpatialBg::RowColPercentile)
        }
        "block_percentile" | "blockpercentile" | "block_median" => {
            let bs = if bg_block_size == 0 { 32 } else { bg_block_size as usize };
            (BgMode::RowPercentile, SpatialBg::BlockPercentile(bs))
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "bg_mode must be one of 'row_percentile', 'line_median', 'top_hat', \
                 'column_percentile', 'row_column_percentile', 'block_percentile'; got '{other}'"
            )))
        }
    };

    // Copy the image into a Rust-owned buffer so we can drop the GIL.
    // ~0.7 MB on a 0.73 MP frame; ~1 ms memcpy on the Zero 2W. The win is that
    // every other Python thread can run during the multi-millisecond compute.
    let owned: Vec<u8> = image
        .as_slice()
        .map_err(|_| {
            PyValueError::new_err("image must be C-contiguous uint8; use np.ascontiguousarray")
        })?
        .to_vec();

    let stars = py
        .allow_threads(|| {
            let pool = get_pool();
            pool.install(|| match bin {
                1 => {
                    let preproc = apply_spatial_bg(&owned, w, h, &spatial);
                    let det: &[u8] = preproc.as_deref().unwrap_or(&owned);
                    let nz = noise.unwrap_or_else(|| estimate_noise(det, w, h));
                    Ok(detect(det, w, h, &owned, w, h, 1, sigma, nz, use_neon, bg, max_axis_ratio))
                }
                2 => {
                    let (b, wb, hb) = bin2x2_mean(&owned, w, h);
                    let preproc = apply_spatial_bg(&b, wb, hb, &spatial);
                    let det: &[u8] = preproc.as_deref().unwrap_or(&b);
                    let nz = noise.unwrap_or_else(|| estimate_noise(det, wb, hb));
                    let result = if centroid_full_res {
                        detect(det, wb, hb, &owned, w, h, 2, sigma, nz, use_neon, bg, max_axis_ratio)
                    } else {
                        detect(det, wb, hb, &b, wb, hb, 1, sigma, nz, use_neon, bg, max_axis_ratio)
                            .into_iter()
                            .map(|s| Star { x: s.x * 2.0, y: s.y * 2.0, ..s })
                            .collect()
                    };
                    Ok(result)
                }
                _ => Err("bin must be 1 or 2"),
            })
        })
        .map_err(PyValueError::new_err)?;

    Ok(stars
        .into_iter()
        .map(|s| (s.x, s.y, s.brightness, s.peak as i64))
        .collect())
}

/// Detect stars using a pre-computed per-row background floor and noise sigma.
///
/// This is the steady-state "cached" entry point intended for use inside a
/// finder's tracking-state loop. A background worker periodically computes a
/// high-quality row-offset model (median across N stacked frames) plus a noise
/// sigma estimate, and this function consumes them directly — no per-frame
/// noise estimation, no per-row median computation. The result is detection
/// that is both faster and *more accurate* than per-frame estimation, because
/// the background model was computed over temporal data.
///
/// Args:
///   image:       2-D C-contiguous numpy uint8 array (height, width).
///   row_offsets: 1-D numpy uint8 array of length `height`. The per-row
///                background floor in image pixel units. Typically the median
///                value of each row in the dark-sky reference, optionally
///                after dark-frame subtraction.
///   noise:       Pre-computed noise sigma in pixel units (typically from a
///                MAD on the same temporal stack used to build row_offsets).
///   sigma:       Detection threshold in noise sigmas (default 8).
///   bin:         1 or 2. With bin=2, row_offsets must be of length height/2
///                (one entry per row of the downsampled image).
///   centroid_full_res, use_neon, max_axis_ratio: same as detect_stars.
///   tophat_radius: when > 0, apply white top-hat to the detection image
///          before using the cached row_floors. The cached noise estimate is
///          still used for thresholding. Centroids are taken on the original
///          (pre-tophat) image. Default 0 (disabled).
///
/// The caller is responsible for refreshing row_offsets and noise when the
/// scene changes (slew, moon angle shift, twilight progression). If the
/// cached state is stale, detected counts will drop and false positives may
/// rise; the application's state machine should fall back to detect_stars()
/// in those regimes.
///
/// Returns: list of (x, y, brightness, peak), brightest first.
#[pyfunction]
#[pyo3(signature = (
    image,
    row_offsets,
    noise,
    sigma=8.0,
    bin=1,
    centroid_full_res=true,
    use_neon=false,
    max_axis_ratio=f64::INFINITY,
    tophat_radius=0u32,
))]
#[allow(clippy::too_many_arguments)]
fn detect_stars_with_cache(
    py: Python<'_>,
    image: PyReadonlyArray2<u8>,
    row_offsets: numpy::PyReadonlyArray1<u8>,
    noise: f64,
    sigma: f64,
    bin: u32,
    centroid_full_res: bool,
    use_neon: bool,
    max_axis_ratio: f64,
    tophat_radius: u32,
) -> PyResult<Vec<(f64, f64, f64, i64)>> {
    let shape = image.shape();
    let (h, w) = (shape[0], shape[1]);
    if shape.len() != 2 {
        return Err(PyValueError::new_err("image must be 2-D"));
    }

    let expected_rows = match bin {
        1 => h,
        2 => h / 2,
        _ => return Err(PyValueError::new_err("bin must be 1 or 2")),
    };
    let rof_slice = row_offsets
        .as_slice()
        .map_err(|_| PyValueError::new_err("row_offsets must be C-contiguous uint8"))?;
    if rof_slice.len() != expected_rows {
        return Err(PyValueError::new_err(format!(
            "row_offsets has length {} but {} were expected for bin={}",
            rof_slice.len(),
            expected_rows,
            bin
        )));
    }

    // Copy both inputs so we can drop the GIL.
    let owned: Vec<u8> = image
        .as_slice()
        .map_err(|_| {
            PyValueError::new_err("image must be C-contiguous uint8; use np.ascontiguousarray")
        })?
        .to_vec();
    let owned_rof: Vec<u8> = rof_slice.to_vec();

    let tophat_r = tophat_radius as usize;

    let stars = py
        .allow_threads(|| {
            let pool = get_pool();
            pool.install(|| match bin {
                1 => {
                    let tophat_buf: Option<Vec<u8>> = if tophat_r > 0 {
                        Some(white_tophat(&owned, w, h, tophat_r))
                    } else {
                        None
                    };
                    let det: &[u8] = tophat_buf.as_deref().unwrap_or(&owned);
                    // With tophat, the detection image background is ~0; use zero
                    // row_floors so the threshold is sigma*noise (not inflated by
                    // the cached original-space background level).
                    let zero_floors: Vec<u8>;
                    let floors: &[u8] = if tophat_r > 0 {
                        zero_floors = vec![0u8; owned_rof.len()];
                        &zero_floors
                    } else {
                        &owned_rof
                    };
                    Ok(detect_cached(
                        det,
                        w,
                        h,
                        &owned,
                        w,
                        h,
                        1,
                        sigma,
                        noise,
                        use_neon,
                        floors,
                        max_axis_ratio,
                    ))
                }
                2 => {
                    let (b, wb, hb) = bin2x2_mean(&owned, w, h);
                    let tophat_buf: Option<Vec<u8>> = if tophat_r > 0 {
                        Some(white_tophat(&b, wb, hb, tophat_r))
                    } else {
                        None
                    };
                    let det: &[u8] = tophat_buf.as_deref().unwrap_or(&b);
                    let zero_floors: Vec<u8>;
                    let floors: &[u8] = if tophat_r > 0 {
                        zero_floors = vec![0u8; owned_rof.len()];
                        &zero_floors
                    } else {
                        &owned_rof
                    };
                    let result = if centroid_full_res {
                        detect_cached(
                            det,
                            wb,
                            hb,
                            &owned,
                            w,
                            h,
                            2,
                            sigma,
                            noise,
                            use_neon,
                            floors,
                            max_axis_ratio,
                        )
                    } else {
                        let in_binned = detect_cached(
                            det,
                            wb,
                            hb,
                            &b,
                            wb,
                            hb,
                            1,
                            sigma,
                            noise,
                            use_neon,
                            floors,
                            max_axis_ratio,
                        );
                        in_binned
                            .into_iter()
                            .map(|s| Star {
                                x: s.x * 2.0,
                                y: s.y * 2.0,
                                brightness: s.brightness,
                                peak: s.peak,
                            })
                            .collect()
                    };
                    Ok(result)
                }
                _ => Err("bin must be 1 or 2"),
            })
        })
        .map_err(PyValueError::new_err)?;

    Ok(stars
        .into_iter()
        .map(|s| (s.x, s.y, s.brightness, s.peak as i64))
        .collect())
}

// Like detect(), but consumes a pre-computed row_floors slice and skips the
// noise estimator (caller supplies `noise` directly).
#[allow(clippy::too_many_arguments)]
fn detect_cached(
    det_img: &[u8],
    det_w: usize,
    det_h: usize,
    cent_img: &[u8],
    cent_w: usize,
    cent_h: usize,
    scale: usize,
    sigma: f64,
    noise: f64,
    use_neon: bool,
    row_floors: &[u8],
    max_axis_ratio: f64,
) -> Vec<Star> {
    if det_w < 7 || det_h < 7 {
        return Vec::new();
    }
    let sn2 = ((2.0 * sigma * noise + 0.5) as i32).max(2);
    let mf_thresh = mf_threshold(sigma, noise);

    let n_bands = rayon::current_num_threads().max(1);
    let band_rows = det_h.div_ceil(n_bands);
    let banded: Vec<Vec<Candidate>> = (0..n_bands)
        .into_par_iter()
        .map(|b| {
            let y0 = b * band_rows;
            let y1 = ((b + 1) * band_rows).min(det_h);
            if y0 >= y1 {
                Vec::new()
            } else {
                scan_band(
                    det_img,
                    det_w,
                    y0,
                    y1,
                    sn2,
                    use_neon,
                    Some(row_floors),
                    mf_thresh,
                )
            }
        })
        .collect();
    let cands: Vec<Candidate> = banded.into_iter().flatten().collect();
    if cands.is_empty() {
        return Vec::new();
    }

    let max_size = (det_w / 100).max(3);
    let blobs = form_blobs(&cands, det_h);

    let mut stars: Vec<Star> = blobs
        .par_iter()
        .filter_map(|blob| {
            gate_2d(
                &cands,
                blob,
                det_img,
                det_w,
                det_h,
                cent_img,
                cent_w,
                cent_h,
                scale,
                noise,
                sigma,
                max_size,
                max_axis_ratio,
            )
        })
        .collect();
    stars.sort_by(|a, b| {
        b.brightness
            .partial_cmp(&a.brightness)
            .unwrap_or(Ordering::Equal)
    });
    stars
}

/// Compute the per-row median of a uint8 image. Returns a 1-D uint8 array of
/// length `height`. This is exposed so the background worker can compute it
/// in parallel (rayon-backed) without re-implementing the histogram in Python.
///
/// Combined with stacking multiple frames in Python (e.g. np.median across the
/// time axis), this gives a robust temporal row-offset model.
#[pyfunction]
fn compute_row_medians_py<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<u8>,
) -> PyResult<Bound<'py, numpy::PyArray1<u8>>> {
    let shape = image.shape();
    let (h, w) = (shape[0], shape[1]);
    let data: Vec<u8> = image
        .as_slice()
        .map_err(|_| PyValueError::new_err("image must be C-contiguous uint8"))?
        .to_vec();
    let medians = py.allow_threads(|| {
        let pool = get_pool();
        pool.install(|| compute_row_medians(&data, w, h))
    });
    Ok(numpy::PyArray1::from_vec_bound(py, medians))
}

#[pymodule]
fn star_detect(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(detect_stars, m)?)?;
    m.add_function(wrap_pyfunction!(detect_stars_with_cache, m)?)?;
    m.add_function(wrap_pyfunction!(compute_row_medians_py, m)?)?;
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    Ok(())
}
