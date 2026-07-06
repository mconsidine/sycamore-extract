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
//!      gates an integer matched-filter "gate" test producing candidates. The
//!      gate kernel is a runtime-generated Gaussian (width set by `kernel_sigma`,
//!      default 1.5 = the legacy 7-tap kernel). The pixel-wise threshold
//!      prefilter is autovectorized by default and can be switched to an
//!      explicit NEON path via `use_neon=True`.
//!   3. Union-find blob assembly over vertically-adjacent candidates.
//!   4. 2-D gate (size / edge / perimeter uniformity / sigma over background
//!      using a perimeter-derived local noise estimate; full 2-D second-moment
//!      axis-ratio rejection when `max_axis_ratio` is finite).
//!   5. Background-subtracted separable projection centroid with parabolic
//!      sub-pixel interpolation. With bin>1 and `centroid_full_res=True`,
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

// Default matched-filter gate half-width: 7-pixel gate => 3 pixels of context
// each side. The runtime-tunable kernel (see `MatchedKernel`) generalizes this;
// GATE_HALF remains the half-width for the default sigma=1.5 kernel and is used
// by the standalone threshold prefilter, whose border margin is independent of
// the gate width (it only needs >=1 px of context so the gate window is valid).
const GATE_HALF: usize = 3; // 7-pixel gate => 3 pixels of context each side.

// Maximum matched-filter kernel half-width (sigma=4.0 => ceil(2*4)=8, clamped to
// 7 => 15 taps). Bounds stack buffers in the gate hot path.
const MAX_KERNEL_HALF: usize = 7;

// =========================================================================
// Runtime-tunable matched-filter kernel (Feature 1)
// =========================================================================
//
// The 1-D gate convolves a window with a discrete, mean-subtracted Gaussian
// kernel and thresholds the response. Historically the kernel was the hardcoded
// constant `[-50, -15, 35, 60, 35, -15, -50]` (sigma=1.5, 7 taps). It is now
// generated at runtime from `kernel_sigma` so the gate can be widened for
// bloated PSFs (poor seeing, heavy defocus) without recompiling.
//
// Generation (matches the documented derivation; see gate_1d comment):
//   h     = ceil(2*sigma) clamped to [3, 7]                (half-width in px)
//   raw   = exp(-x^2 / (2*sigma^2)) for x in -h..=h         (Gaussian samples)
//   zm    = raw - mean(raw)                                 (mean-removed => DC=0)
//   scale = 60 / zm[center]                                 (center coeff -> 60)
//   k     = round(zm * scale) as i32                        (integer taps)
//   then enforce sum(k) == 0 exactly by a symmetric residual adjustment on the
//   outermost taps (the kernel is symmetric, so any rounding residual is split
//   evenly across the two ends; an odd remainder lands on the leftmost tap).
//
// At sigma=1.5 this reproduces the exact historical constant kernel (verified
// by unit test `kernel_sigma_1_5_matches_legacy`), so default-path results are
// bit-identical to pre-0.12 behavior.
#[derive(Clone)]
struct MatchedKernel {
    taps: Vec<i32>,
    half: usize,
    // L2 norm used to scale the response threshold. For the default sigma=1.5
    // kernel this is forced to exactly 107.0 (the historical hardcoded value,
    // a slight conservative round of sqrt(11500) ~= 107.238) so default results
    // do not shift; for all other kernels it is the true computed ||k||_2.
    l2: f64,
}

fn generate_matched_kernel(sigma: f64) -> MatchedKernel {
    let h = (2.0 * sigma).ceil() as usize;
    let h = h.clamp(3, MAX_KERNEL_HALF);
    let n = 2 * h + 1;

    let raw: Vec<f64> = (0..n)
        .map(|i| {
            let x = i as f64 - h as f64;
            (-(x * x) / (2.0 * sigma * sigma)).exp()
        })
        .collect();
    let mean = raw.iter().sum::<f64>() / n as f64;
    let zm: Vec<f64> = raw.iter().map(|r| r - mean).collect();
    let center = zm[h];
    let scale = 60.0 / center;
    let mut taps: Vec<i32> = zm.iter().map(|z| (z * scale).round() as i32).collect();

    // Enforce sum == 0 by a symmetric residual adjustment on the outermost taps.
    let residual: i32 = taps.iter().sum();
    if residual != 0 {
        let half = residual / 2;
        taps[0] -= half;
        taps[n - 1] -= half;
        let rem = residual - 2 * half; // -1, 0, or +1 (sign of residual)
        if rem != 0 {
            taps[0] -= rem;
        }
    }
    debug_assert_eq!(taps.iter().sum::<i32>(), 0, "kernel must be DC-free");

    // L2 norm. Preserve exact 107.0 for the default kernel so default results
    // don't shift (the old hot path used 107.0, not sqrt(11500)).
    const DEFAULT_KERNEL: [i32; 7] = [-50, -15, 35, 60, 35, -15, -50];
    let l2 = if taps == DEFAULT_KERNEL {
        107.0
    } else {
        taps.iter().map(|&t| (t * t) as f64).sum::<f64>().sqrt()
    };

    MatchedKernel { taps, half: h, l2 }
}

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
///
/// `ZeroFloor`: skip floor estimation entirely (fixed floor of 0). For use
///   after a spatial preprocessor (e.g. `SpatialBg::TopHat`) has already
///   flattened the image to a near-zero background — computing a fresh
///   per-row percentile on top of that would just be re-measuring ~0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BgMode {
    RowPercentile,
    LineMedian,
    ZeroFloor,
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
fn extreme_1d_into(src: &[u8], radius: usize, is_max: bool, s: &mut MorphScratch, out: &mut [u8]) {
    let n = src.len();
    if n == 0 {
        return;
    }
    let w = 2 * radius + 1;
    let neutral = if is_max { 0u8 } else { 255u8 };
    let pn = n + 2 * radius;

    // Reuse caller-owned scratch: grow once per thread, then only the small
    // pad regions are re-filled per row. The prefix/suffix buffers are fully
    // overwritten below, so they need no clearing at all.
    if s.padded.len() < pn {
        s.padded.resize(pn, 0);
        s.prefix.resize(pn, 0);
        s.suffix.resize(pn, 0);
    }
    let padded = &mut s.padded[..pn];
    padded[..radius].fill(neutral);
    padded[radius..radius + n].copy_from_slice(src);
    padded[radius + n..].fill(neutral);

    // Prefix: left-to-right running extreme, reset at each block boundary.
    let prefix = &mut s.prefix[..pn];
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
    let suffix = &mut s.suffix[..pn];
    for i in (0..pn).rev() {
        suffix[i] = if i % w == w - 1 || i == pn - 1 {
            padded[i]
        } else if is_max {
            suffix[i + 1].max(padded[i])
        } else {
            suffix[i + 1].min(padded[i])
        };
    }

    // Output: window [j, j+2r] in padded space maps to output[j].
    for (j, o) in out.iter_mut().enumerate() {
        *o = if is_max {
            suffix[j].max(prefix[j + 2 * radius])
        } else {
            suffix[j].min(prefix[j + 2 * radius])
        };
    }
}

/// Per-thread scratch for the van Herk passes. One set of buffers per rayon
/// worker (via for_each_init) instead of three fresh Vecs per row — the
/// allocation churn was the dominant constant factor in white_tophat.
#[derive(Default)]
struct MorphScratch {
    padded: Vec<u8>,
    prefix: Vec<u8>,
    suffix: Vec<u8>,
}

/// Apply extreme_1d along every row of a 2-D image (horizontal morphology pass).
fn morph_h(data: &[u8], w: usize, h: usize, radius: usize, is_max: bool) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    out.par_chunks_mut(w).enumerate().for_each_init(
        MorphScratch::default,
        |scratch, (y, row_out)| {
            extreme_1d_into(&data[y * w..y * w + w], radius, is_max, scratch, row_out);
        },
    );
    out
}

/// Transpose a row-major 2-D image. Returns (transposed_data, new_w, new_h).
/// Each output row (= one input column) is filled independently, so this is
/// safe to parallelise without any synchronisation.
fn transpose(data: &[u8], src_w: usize, src_h: usize) -> (Vec<u8>, usize, usize) {
    // Tiled (32x32) so both the source and destination lines of a tile stay
    // resident in L1; the naive strided version walked the whole source image
    // once per output row and was memory-latency bound on the A53.
    const TILE: usize = 32;
    let mut out = vec![0u8; src_w * src_h];
    // Output is src_w rows of length src_h; one strip = TILE output rows
    // (= TILE source columns), filled independently per rayon job.
    out.par_chunks_mut(TILE * src_h)
        .enumerate()
        .for_each(|(strip, chunk)| {
            let x0 = strip * TILE;
            let x1 = (x0 + TILE).min(src_w);
            for y0 in (0..src_h).step_by(TILE) {
                let y1 = (y0 + TILE).min(src_h);
                for x in x0..x1 {
                    let orow = &mut chunk[(x - x0) * src_h..(x - x0) * src_h + src_h];
                    for y in y0..y1 {
                        orow[y] = data[y * src_w + x];
                    }
                }
            }
        });
    (out, src_h, src_w)
}

/// Morphological white top-hat transform: image − opening(image).
/// `opening` = dilate(erode(image)) with a separable flat structuring element
/// of the given radius. Removes broad low-frequency background (gradients,
/// vignetting, sky-glow) while preserving point-source stars.
fn white_tophat(data: &[u8], w: usize, h: usize, radius: usize) -> Vec<u8> {
    // Separable opening with the two middle transposes fused away. The H and V
    // components of erosion commute (likewise dilation), and erosion fully
    // precedes dilation, so:
    //   minH -> T -> minH (==minV) -> maxH (==maxV) -> T -> maxH
    // is the same opening with 2 transposes instead of 4.
    let e1 = morph_h(data, w, h, radius, false);
    let (t, tw, th) = transpose(&e1, w, h);
    let e2 = morph_h(&t, tw, th, radius, false); // erosion complete (transposed)
    let d1 = morph_h(&e2, tw, th, radius, true); // dilation, vertical component
    let (back, bw, bh) = transpose(&d1, tw, th);
    debug_assert_eq!((bw, bh), (w, h));
    let opened = morph_h(&back, bw, bh, radius, true); // dilation, horizontal

    // Final top-hat subtraction, parallel by row (was a serial full-image pass).
    let mut out = vec![0u8; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let a = &data[y * w..y * w + w];
        let o = &opened[y * w..y * w + w];
        for (r, (&av, &ov)) in row.iter_mut().zip(a.iter().zip(o.iter())) {
            *r = av.saturating_sub(ov);
        }
    });
    out
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
//
// `g` is the (2*half+1)-pixel window; `kernel.taps` has the same length. The
// matched-filter response is the dot product <g, kernel> (an i32). The
// local-max test and tie-breaks operate on the three central raw pixels
// (g[half-1], g[half], g[half+1]) exactly as in the original 7-tap gate, which
// is independent of the kernel width.
#[inline(always)]
fn gate_1d(g: &[u8], kernel: &MatchedKernel, mf_thresh: i32) -> bool {
    let half = kernel.half;
    let mut response = 0i32;
    for (gi, &k) in g.iter().zip(kernel.taps.iter()) {
        response += (*gi as i32) * k;
    }
    if response < mf_thresh {
        return false;
    }
    // Local-maximum suppression in raw pixel space (three central pixels).
    let c1 = g[half - 1] as i32; // left of center
    let c = g[half] as i32; // center
    let c2 = g[half + 1] as i32; // right of center
    if c1 > c || c < c2 {
        return false;
    }
    // Deterministic tie-breaks for flat-topped peaks. The original 7-tap gate
    // compared the next-out neighbors (g1 vs g4, g2 vs g5); generalized, those
    // are g[half-2] vs g[half+1] and g[half-1] vs g[half+2].
    if c1 == c && (g[half - 2] as i32) > c2 {
        return false;
    }
    if c == c2 && c1 <= (g[half + 2] as i32) {
        return false;
    }
    true
}

// Compute the matched-filter threshold from sigma/noise. Kept as a function
// so the caller can compute it once per band, not per pixel. Uses the kernel's
// L2 norm: for the default sigma=1.5 kernel this is exactly 107.0 (preserving
// pre-0.12 behavior); for wider kernels it is the true ||k||_2.
#[inline]
fn mf_threshold(sigma: f64, noise: f64, kernel: &MatchedKernel) -> i32 {
    let t = sigma * noise * kernel.l2 + 0.5;
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
    out.par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row_out)| {
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
    out.par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row_out)| {
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
    let (block_medians, nx, ny) = compute_block_medians(data, width, height, bs);
    subtract_block_grid(data, width, height, bs, &block_medians, nx, ny)
}

/// Compute the per-tile median grid for `block_percentile`, exposed so it can be
/// cached temporally (Feature 5). Returns (grid, nx, ny) where the grid is
/// row-major (ny rows of nx tiles). Same 256-bin counting median as
/// `block_percentile_bg`, parallel over tiles.
fn compute_block_medians(
    data: &[u8],
    width: usize,
    height: usize,
    block_size: usize,
) -> (Vec<u8>, usize, usize) {
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
            let mut hist = [0u32; 256];
            for y in y0..y1 {
                for &px in &data[y * width + x0..y * width + x1] {
                    hist[px as usize] += 1;
                }
            }
            let count = ((y1 - y0) * (x1 - x0)) as u32;
            let target = count / 2 + 1;
            let mut acc = 0u32;
            let mut med = 0u8;
            for (v, &c) in hist.iter().enumerate() {
                acc += c;
                if acc >= target {
                    med = v as u8;
                    break;
                }
            }
            med
        })
        .collect();
    (block_medians, nx, ny)
}

/// Subtract a bilinearly-interpolated block-median background grid (saturating).
/// Shared by `block_percentile_bg` (per-frame) and the cached block path.
/// Per-pixel saturating subtraction of a full background image (same length as
/// `data`). The bg_image cached path: the temporal median stack IS the model,
/// so no interpolation is needed — one pass, autovectorizes on A53.
fn subtract_image(data: &[u8], bg: &[u8]) -> Vec<u8> {
    data.iter()
        .zip(bg.iter())
        .map(|(&p, &b)| p.saturating_sub(b))
        .collect()
}

fn subtract_block_grid(
    data: &[u8],
    width: usize,
    height: usize,
    block_size: usize,
    block_medians: &[u8],
    nx: usize,
    ny: usize,
) -> Vec<u8> {
    let bs = block_size.max(2);
    let half_bs = bs as f32 / 2.0;
    let mut out = vec![0u8; width * height];
    out.par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row_out)| {
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
    UniformMean(usize),
}

/// 2-D sliding-window mean background subtraction via summed-area table.
///
/// Each pixel's background estimate is the mean of the `filter_size × filter_size`
/// window centered on it (smaller windows at the boundary). Uses an integral
/// image (SAT) so the per-pixel cost is O(1) regardless of window size, giving
/// O(W×H) total. Equivalent to `scipy.ndimage.uniform_filter(image, filter_size)`
/// followed by subtraction — reproduces the default tetra3 / olive-solve background.
///
/// `filter_size` must be odd and > 0 for a symmetric window; even values work
/// but the center pixel falls one step off-center (same as scipy's behavior).
fn uniform_mean_bg(data: &[u8], width: usize, height: usize, filter_size: usize) -> Vec<u8> {
    let w = width;
    let h = height;
    let half = filter_size / 2;

    // Build a summed-area table (SAT) with a 1-pixel border of zeros.
    // sat[(y+1)*(w+1)+(x+1)] = sum of data[0..=y, 0..=x].
    // Maximum value: 255 * 960 * 760 ≈ 185 M  — fits in u32.
    let mut sat = vec![0u32; (w + 1) * (h + 1)];
    for y in 0..h {
        let mut row_sum = 0u32;
        for x in 0..w {
            row_sum += data[y * w + x] as u32;
            sat[(y + 1) * (w + 1) + (x + 1)] = row_sum + sat[y * (w + 1) + (x + 1)];
        }
    }

    // Subtract per-pixel window mean (parallelised by row).
    let mut out = vec![0u8; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row_out)| {
        let y0 = y.saturating_sub(half);
        let y1 = (y + half + 1).min(h);
        for (x, o) in row_out.iter_mut().enumerate() {
            let x0 = x.saturating_sub(half);
            let x1 = (x + half + 1).min(w);
            let area = ((y1 - y0) * (x1 - x0)) as u32;
            // SAT rectangle query: sum(y0..y1, x0..x1).
            let sum = sat[y1 * (w + 1) + x1] + sat[y0 * (w + 1) + x0]
                - sat[y0 * (w + 1) + x1]
                - sat[y1 * (w + 1) + x0];
            let mean = (sum / area) as u8;
            *o = data[y * w + x].saturating_sub(mean);
        }
    });
    out
}

/// Global RMS noise estimator: sqrt(mean(pixel²)) over the entire image.
///
/// Used in place of the MAD-based `estimate_noise` when `noise_mode="global_rms"`.
/// This matches olive-solve's `SigmaMode::GlobalRootSquare` and the original
/// Python tetra3's default. It is faster than MAD but less robust — a bright
/// nebula or strong gradient will inflate the estimate and raise the detection
/// threshold, so it should only be used on already background-subtracted images.
fn estimate_noise_global_rms(data: &[u8], w: usize, h: usize) -> f64 {
    let n = (w * h) as u64;
    if n == 0 {
        return 0.5;
    }
    let sum_sq: u64 = data.iter().map(|&p| (p as u64) * (p as u64)).sum();
    ((sum_sq as f64 / n as f64).sqrt()).max(0.5)
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
        SpatialBg::UniformMean(fs) => Some(uniform_mean_bg(img, w, h, *fs)),
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
    kernel: &MatchedKernel,
    mf_thresh: i32,
) -> Vec<Candidate> {
    let gate_half = kernel.half;
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
        // The gate window spans [x-gate_half, x+gate_half]; gate_half generalizes
        // the old fixed 3 to the runtime kernel half-width.
        for &x in &hits {
            let x = x as usize;
            if x < gate_half || x + gate_half >= width {
                continue;
            }
            let g = &row[x - gate_half..x + gate_half + 1];
            if gate_1d(g, kernel, mf_thresh) {
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

// Squared axis ratio (lambda_max / lambda_min) of the symmetric 2x2 covariance
// [[vx, cxy], [cxy, vy]], via closed-form eigenvalues
//   lambda = t/2 +/- sqrt((t/2)^2 - d),  t = vx+vy (trace), d = vx*vy - cxy^2.
// Returning the squared ratio lets callers compare against max_axis_ratio^2
// without a sqrt. The off-diagonal term cxy is what lets this detect
// diagonally-elongated trails that a var_x/var_y-only test cannot see.
#[inline]
fn cov2x2_axis_ratio2(vx: f64, vy: f64, cxy: f64) -> f64 {
    let half_t = 0.5 * (vx + vy);
    let det = vx * vy - cxy * cxy;
    let disc = (half_t * half_t - det).max(0.0).sqrt();
    let lam_max = half_t + disc;
    let lam_min = (half_t - disc).max(1e-6);
    lam_max / lam_min
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
    local_noise: bool,
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
    // Feature 3: perimeter-derived local noise inflation (concept inspired by
    // cedar-detect, Apache-2.0; independent implementation — no code copied).
    // The acceptance test compares core brightness above background against
    // sigma * effective_noise, where effective_noise = max(global, ring_spread).
    // In cluttered/noisy neighborhoods (moon halo, clouds, foreground glow) the
    // ring's own scatter (p_std) raises the local bar and suppresses false
    // positives; on clean sky p_std < global noise so behavior is unchanged.
    // `local_noise=false` disables it (uses the global noise only) for A/B.
    let effective_noise = if local_noise { noise.max(p_std) } else { noise };
    if core_mean - bg_det < sigma * effective_noise {
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

    // Feature 2: full 2-D second-moment axis-ratio rejection. The axis ratio is
    // sqrt(lambda_max / lambda_min) of the 2x2 covariance
    //   [[var_x, cov_xy], [cov_xy, var_y]]
    // computed (background-subtracted, negative-clamped weights) over the
    // centroid box. Including the off-diagonal moment cov_xy lets this catch
    // *diagonally* elongated trails (satellite / aircraft streaks) that the old
    // separable var_x/var_y-only test could not see. Closed-form eigenvalues of
    // a symmetric 2x2 are used (no iterative solve).
    //
    // Cost discipline: this requires a single extra 2-D accumulation pass over
    // the (small, per-blob) box. It is skipped entirely when max_axis_ratio is
    // infinite (the common default), so the hot path pays nothing for it.
    if brightness > 0.0 && max_axis_ratio.is_finite() && max_axis_ratio > 1.0 {
        let inv_b = 1.0 / brightness;
        // First moments (centroid within the box) from the cheap projections.
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

        // Second moments via one 2-D pass over the box (same weights as the
        // projections: background-subtracted, negative-clamped).
        let mut var_x = 0f64;
        let mut var_y = 0f64;
        let mut cov_xy = 0f64;
        for yy in 0..mh {
            let row = &cent_img[(my0 + yy) * cent_w..(my0 + yy) * cent_w + cent_w];
            let dy = yy as f64 - m1y;
            for xx in 0..mw {
                let val = (row[mx0 + xx] as f64 - bg_cent).max(0.0);
                if val > 0.0 {
                    let dx = xx as f64 - m1x;
                    var_x += dx * dx * val;
                    var_y += dy * dy * val;
                    cov_xy += dx * dy * val;
                }
            }
        }
        var_x *= inv_b;
        var_y *= inv_b;
        cov_xy *= inv_b;

        // axis_ratio = sqrt(lam_max/lam_min) of [[var_x, cov_xy],[cov_xy, var_y]];
        // reject if it exceeds max_axis_ratio (see cov2x2_axis_ratio2).
        if cov2x2_axis_ratio2(var_x, var_y, cov_xy) > max_axis_ratio * max_axis_ratio {
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

// 4x4 mean bin = two cascaded 2x2 mean bins. Used by bin=4 (escape hatch for
// badly defocused / oversampled stars). Coordinates map x4 to full-res.
fn bin4x4_mean(data: &[u8], w: usize, h: usize) -> (Vec<u8>, usize, usize) {
    let (b2, w2, h2) = bin2x2_mean(data, w, h);
    bin2x2_mean(&b2, w2, h2)
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
    kernel: &MatchedKernel,
    local_noise: bool,
) -> Vec<Star> {
    if det_w < 7 || det_h < 7 {
        return Vec::new();
    }
    let sn2 = ((2.0 * sigma * noise + 0.5) as i32).max(2);
    let mf_thresh = mf_threshold(sigma, noise, kernel);

    // LineMedian: precompute one median per row in parallel. This is the
    // equivalent of olive-solve's FastBgSubMode::LineMedian and handles:
    //   - per-row offset noise (some CMOS sensors, notably IMX296mono),
    //   - vertical brightness gradients from vignetting/light pollution,
    //   - black-bar artifacts (median is unaffected by up to 49% bad pixels
    //     per row, vs the row-percentile's ~20%).
    let row_floors: Option<Vec<u8>> = match bg_mode {
        BgMode::LineMedian => Some(compute_row_medians(det_img, det_w, det_h)),
        BgMode::RowPercentile => None,
        BgMode::ZeroFloor => Some(vec![0u8; det_h]),
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
                    kernel,
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
                local_noise,
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
///          Values below 0.5 are clamped to 0.5.
///          The matched filter is calibrated for sigma=8 on representative
///          HQ Camera frames. On real-sky data with lots of correlated noise
///          structure, you may need to lower sigma to 6-7 to catch faint
///          stars near the noise floor. Field validation is encouraged.
///   noise: optional precomputed noise level; estimated if None.
///   bin:   1 (full res), 2 (2x2-binned detection for speed), or 4 (4x4-binned,
///          two cascaded 2x2 mean bins — the escape hatch for badly defocused /
///          oversampled stars). Centroids are returned in full-resolution pixel
///          coordinates regardless.
///   centroid_full_res: when bin>1, perform centroiding on the full-resolution
///          image. Slightly more work per star, but recovers the sub-pixel
///          precision that scaling-from-binned loses. No effect when bin=1.
///   use_neon: when true and built for aarch64, use the explicit NEON threshold
///          prefilter. No-op on non-aarch64 targets.
///   bg_mode: background subtraction strategy. Options:
///     "uniform_mean" — 2-D sliding-window mean via summed-area table; O(1)
///       per pixel regardless of window size. Reproduces the default tetra3 /
///       olive-solve background (their filtsize=25). Window controlled by
///       uniform_filter_size (default 25). Pair with noise_mode="global_rms"
///       to exactly replicate the tetra3 pipeline.
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
///          Default Inf (no filtering). Recommend 3-5 for a finder. When finite,
///          rejection uses the full 2x2 second-moment covariance (including the
///          off-diagonal m2_xy term) so diagonal trails are caught too.
///   local_noise: when True (default), inflate the per-blob acceptance noise to
///          max(global_noise, perimeter_ring_spread). This raises the bar in
///          cluttered/noisy neighborhoods (moon halo, clouds, foreground glow)
///          and suppresses false positives there; clean-sky behavior is
///          unchanged (ring spread < global noise). Set False for A/B testing.
///   kernel_sigma: Gaussian sigma (px) of the matched-filter kernel, range
///          1.0-4.0 (ValueError outside). Default 1.5 reproduces the historical
///          7-tap kernel exactly (bit-identical results). Larger values widen
///          the kernel (sigma=2.5 -> 11 taps, sigma=4.0 -> 15 taps) to better
///          match bloated PSFs from poor seeing or heavy defocus.
///   tophat_radius: structuring-element radius for the white top-hat (pixels,
///          default 0 = use 12). Must exceed the largest star radius. Only
///          used when bg_mode="top_hat".
///   bg_block_size: tile side length in pixels for block_percentile mode
///          (default 0 = use 32). Must comfortably exceed the largest star
///          radius. Typical: 32 (bin=2 detection image), 64 (bin=1).
///   uniform_filter_size: sliding-window side length for uniform_mean mode
///          (default 0 = use 25, matching tetra3/olive-solve's filtsize=25).
///          Must exceed the largest star PSF diameter.
///   noise_mode: how to estimate the noise sigma when `noise` is not given.
///          "mad" (default) — median absolute deviation on scattered patches;
///            robust to gradients, stars, and hot pixels. Recommended for all
///            modes except uniform_mean.
///          "global_rms" — sqrt(mean(pixel²)) over the whole (preprocessed)
///            image; matches olive-solve's GlobalRootSquare and the original
///            Python tetra3 default. Faster but inflated by residual structure.
///            Use with uniform_mean for an exact tetra3-compatible pipeline.
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
    uniform_filter_size=0u32,
    noise_mode="mad",
    kernel_sigma=1.5,
    local_noise=true,
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
    uniform_filter_size: u32,
    noise_mode: &str,
    kernel_sigma: f64,
    local_noise: bool,
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
            let r = if tophat_radius == 0 {
                12
            } else {
                tophat_radius as usize
            };
            // TopHat already flattens the image to a near-zero background
            // (image - opening(image)); a fresh per-row percentile floor on
            // top of that would just be re-measuring ~0. Mirrors the explicit
            // zero-floor override already used in the cached row-offset path.
            (BgMode::ZeroFloor, SpatialBg::TopHat(r))
        }
        "column_percentile" | "col_percentile" | "colpercentile" => {
            (BgMode::RowPercentile, SpatialBg::ColPercentile)
        }
        "row_column_percentile" | "rowcolumnpercentile" | "row_col_percentile" => {
            (BgMode::RowPercentile, SpatialBg::RowColPercentile)
        }
        "block_percentile" | "blockpercentile" | "block_median" => {
            let bs = if bg_block_size == 0 {
                32
            } else {
                bg_block_size as usize
            };
            (BgMode::RowPercentile, SpatialBg::BlockPercentile(bs))
        }
        "uniform_mean" | "uniformmean" | "local_mean" | "box_filter" => {
            let fs = if uniform_filter_size == 0 {
                25
            } else {
                uniform_filter_size as usize
            };
            (BgMode::RowPercentile, SpatialBg::UniformMean(fs))
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "bg_mode must be one of 'row_percentile', 'line_median', 'top_hat', \
                 'column_percentile', 'row_column_percentile', 'block_percentile', \
                 'uniform_mean'; got '{other}'"
            )))
        }
    };

    // Guard against pathological thresholds: sigma near 0 makes the matched-
    // filter threshold ~1 ADU, every noise pixel becomes a candidate, and a
    // frame can take hundreds of ms while emitting a flood of false stars.
    // (diofinder exposes a 0-20 sigma range; 0.5 is the useful floor.)
    let sigma = sigma.max(0.5);
    let use_global_rms = matches!(
        noise_mode.to_ascii_lowercase().as_str(),
        "global_rms" | "globalrms" | "rms" | "root_square" | "global_root_square"
    );

    // Validate and build the matched-filter kernel (Feature 1). Range 1.0-4.0.
    if !(1.0..=4.0).contains(&kernel_sigma) {
        return Err(PyValueError::new_err(format!(
            "kernel_sigma must be in [1.0, 4.0]; got {kernel_sigma}"
        )));
    }
    let kernel = generate_matched_kernel(kernel_sigma);

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
                    let nz = noise.unwrap_or_else(|| {
                        if use_global_rms {
                            estimate_noise_global_rms(det, w, h)
                        } else {
                            estimate_noise(det, w, h)
                        }
                    });
                    Ok(detect(
                        det,
                        w,
                        h,
                        &owned,
                        w,
                        h,
                        1,
                        sigma,
                        nz,
                        use_neon,
                        bg,
                        max_axis_ratio,
                        &kernel,
                        local_noise,
                    ))
                }
                2 | 4 => {
                    // bin=2: one 2x2 mean bin (scale 2). bin=4: two cascaded 2x2
                    // mean bins (scale 4) — the escape hatch for badly defocused
                    // or oversampled stars. Noise/background estimation runs on
                    // the binned image; with centroid_full_res the centroid is
                    // taken on the full-res image with coords mapped x{scale}.
                    let scale = bin as usize;
                    let (b, wb, hb) = if bin == 2 {
                        bin2x2_mean(&owned, w, h)
                    } else {
                        bin4x4_mean(&owned, w, h)
                    };
                    let preproc = apply_spatial_bg(&b, wb, hb, &spatial);
                    let det: &[u8] = preproc.as_deref().unwrap_or(&b);
                    let nz = noise.unwrap_or_else(|| {
                        if use_global_rms {
                            estimate_noise_global_rms(det, wb, hb)
                        } else {
                            estimate_noise(det, wb, hb)
                        }
                    });
                    let result = if centroid_full_res {
                        detect(
                            det,
                            wb,
                            hb,
                            &owned,
                            w,
                            h,
                            scale,
                            sigma,
                            nz,
                            use_neon,
                            bg,
                            max_axis_ratio,
                            &kernel,
                            local_noise,
                        )
                    } else {
                        let sc = scale as f64;
                        detect(
                            det,
                            wb,
                            hb,
                            &b,
                            wb,
                            hb,
                            1,
                            sigma,
                            nz,
                            use_neon,
                            bg,
                            max_axis_ratio,
                            &kernel,
                            local_noise,
                        )
                        .into_iter()
                        .map(|s| Star {
                            x: s.x * sc,
                            y: s.y * sc,
                            ..s
                        })
                        .collect()
                    };
                    Ok(result)
                }
                _ => Err("bin must be 1, 2, or 4"),
            })
        })
        .map_err(PyValueError::new_err)?;

    Ok(stars
        .into_iter()
        .map(|s| (s.x, s.y, s.brightness, s.peak as i64))
        .collect())
}

/// Detect stars using a pre-computed background model and noise sigma.
///
/// This is the steady-state "cached" entry point intended for use inside a
/// finder's tracking-state loop. A background worker periodically computes a
/// high-quality background model (median across N stacked frames) plus a noise
/// sigma estimate, and this function consumes it directly — no per-frame
/// noise estimation, no per-row median computation. The result is detection
/// that is both faster and *more accurate* than per-frame estimation, because
/// the background model was computed over temporal data.
///
/// Three cached background models are supported (exactly one must be supplied):
///
///   * `row_offsets` — a 1-D per-row floor (the original cache model). Cheapest;
///     handles per-row offset noise and vertical gradients.
///   * `block_offsets` — a 2-D grid of per-tile medians (from
///     `compute_block_medians_py`). Bilinearly interpolated and subtracted, this
///     removes *non-separable* 2-D gradients (vignetting, sky-glow, light
///     pollution) — the first time 2-D background correction composes with the
///     temporal cache. After subtraction, detection uses the standard inline
///     row-percentile floor on the corrected image.
///   * `bg_image` — the full per-pixel background at detection resolution
///     (typically the temporal median stack itself, binned). Subtracted
///     directly, this is the most complete model: it removes gradients,
///     vignetting, sky-glow AND per-pixel fixed-pattern structure (warm
///     pixels, amp glow) in one pass — everything the temporal stack knows.
///     After subtraction, detection uses the standard inline row-percentile
///     floor on the (near-flat) corrected image, like the block path.
///
/// Args:
///   image:       2-D C-contiguous numpy uint8 array (height, width).
///   row_offsets: optional 1-D numpy uint8 array of length `height // bin`. The
///                per-row background floor in (binned) image pixel units.
///                Positional for backward compatibility. Pass None to use
///                block_offsets instead.
///   noise:       Pre-computed noise sigma in pixel units (typically from a
///                MAD on the same temporal stack used to build the model).
///   sigma:       Detection threshold in noise sigmas (default 8). Values
///                below 0.5 are clamped to 0.5.
///   bin:         1, 2, or 4. The cached model length must match the binned
///                detection image (`height // bin` rows for row_offsets;
///                `compute_block_medians_py` run at the same binning for
///                block_offsets).
///   centroid_full_res, use_neon, max_axis_ratio, kernel_sigma, local_noise:
///                same as detect_stars.
///   tophat_radius: when > 0, apply white top-hat to the detection image before
///                using the cached row_floors. Only valid with row_offsets.
///                Centroids are taken on the original image. Default 0.
///   block_offsets: optional 2-D numpy uint8 grid of per-tile medians (keyword-
///                only). Mutually exclusive with row_offsets / bg_image.
///   block_size:  tile side length (px) that produced block_offsets (default 32).
///   bg_image:    optional 2-D numpy uint8 per-pixel background at the binned
///                detection resolution (height // bin, width // bin), keyword-
///                only. Mutually exclusive with row_offsets / block_offsets.
///
/// The caller is responsible for refreshing the model and noise when the scene
/// changes (slew, moon angle shift, twilight progression).
///
/// Returns: list of (x, y, brightness, peak), brightest first.
#[pyfunction]
#[pyo3(signature = (
    image,
    row_offsets=None,
    noise=0.0,
    sigma=8.0,
    bin=1,
    centroid_full_res=true,
    use_neon=false,
    max_axis_ratio=f64::INFINITY,
    tophat_radius=0u32,
    kernel_sigma=1.5,
    local_noise=true,
    *,
    block_offsets=None,
    block_size=32u32,
    bg_image=None,
))]
#[allow(clippy::too_many_arguments)]
fn detect_stars_with_cache(
    py: Python<'_>,
    image: PyReadonlyArray2<u8>,
    row_offsets: Option<numpy::PyReadonlyArray1<u8>>,
    noise: f64,
    sigma: f64,
    bin: u32,
    centroid_full_res: bool,
    use_neon: bool,
    max_axis_ratio: f64,
    tophat_radius: u32,
    kernel_sigma: f64,
    local_noise: bool,
    block_offsets: Option<numpy::PyReadonlyArray2<u8>>,
    block_size: u32,
    bg_image: Option<PyReadonlyArray2<u8>>,
) -> PyResult<Vec<(f64, f64, f64, i64)>> {
    let shape = image.shape();
    let (h, w) = (shape[0], shape[1]);
    if shape.len() != 2 {
        return Err(PyValueError::new_err("image must be 2-D"));
    }
    if !matches!(bin, 1 | 2 | 4) {
        return Err(PyValueError::new_err("bin must be 1, 2, or 4"));
    }
    let scale = bin as usize;

    // Exactly one of row_offsets / block_offsets / bg_image must be given.
    let n_models =
        row_offsets.is_some() as u8 + block_offsets.is_some() as u8 + bg_image.is_some() as u8;
    if n_models != 1 {
        return Err(PyValueError::new_err(
            "exactly one of row_offsets / block_offsets / bg_image must be supplied",
        ));
    }
    if !(1.0..=4.0).contains(&kernel_sigma) {
        return Err(PyValueError::new_err(format!(
            "kernel_sigma must be in [1.0, 4.0]; got {kernel_sigma}"
        )));
    }
    let kernel = generate_matched_kernel(kernel_sigma);

    // Binned detection-image dimensions (the model is in this space).
    let wb = w / scale;
    let hb = h / scale;

    // Validate and copy whichever cached model was supplied.
    let owned_rof: Option<Vec<u8>> = match &row_offsets {
        Some(ro) => {
            let s = ro
                .as_slice()
                .map_err(|_| PyValueError::new_err("row_offsets must be C-contiguous uint8"))?;
            if s.len() != hb {
                return Err(PyValueError::new_err(format!(
                    "row_offsets has length {} but {} were expected for bin={}",
                    s.len(),
                    hb,
                    bin
                )));
            }
            Some(s.to_vec())
        }
        None => None,
    };

    let block_grid: Option<(Vec<u8>, usize, usize)> = match &block_offsets {
        Some(bo) => {
            let bshape = bo.shape();
            if bshape.len() != 2 {
                return Err(PyValueError::new_err("block_offsets must be 2-D"));
            }
            let (gh, gw) = (bshape[0], bshape[1]);
            let bs = if block_size == 0 {
                32
            } else {
                block_size as usize
            };
            // The grid must match what compute_block_medians produces on the
            // binned detection image at this block_size.
            let exp_gw = wb.div_ceil(bs.max(2));
            let exp_gh = hb.div_ceil(bs.max(2));
            if gw != exp_gw || gh != exp_gh {
                return Err(PyValueError::new_err(format!(
                    "block_offsets shape ({gh}, {gw}) does not match expected \
                     ({exp_gh}, {exp_gw}) for bin={bin}, block_size={bs}"
                )));
            }
            let s = bo
                .as_slice()
                .map_err(|_| PyValueError::new_err("block_offsets must be C-contiguous uint8"))?;
            Some((s.to_vec(), gw, gh))
        }
        None => None,
    };

    // Validate and copy the full per-pixel background model, if supplied. It
    // must be at the binned detection resolution (the same space the row and
    // block models live in).
    let owned_bg: Option<Vec<u8>> = match &bg_image {
        Some(bg) => {
            let bshape = bg.shape();
            if bshape.len() != 2 || bshape[0] != hb || bshape[1] != wb {
                return Err(PyValueError::new_err(format!(
                    "bg_image shape ({}, {}) does not match the binned \
                     detection resolution ({hb}, {wb}) for bin={bin}",
                    bshape[0],
                    bshape.get(1).copied().unwrap_or(0),
                )));
            }
            let s = bg
                .as_slice()
                .map_err(|_| PyValueError::new_err("bg_image must be C-contiguous uint8"))?;
            Some(s.to_vec())
        }
        None => None,
    };

    if (block_grid.is_some() || owned_bg.is_some()) && tophat_radius > 0 {
        return Err(PyValueError::new_err(
            "tophat_radius is only supported with row_offsets",
        ));
    }

    // Copy the image so we can drop the GIL.
    let owned: Vec<u8> = image
        .as_slice()
        .map_err(|_| {
            PyValueError::new_err("image must be C-contiguous uint8; use np.ascontiguousarray")
        })?
        .to_vec();

    // Guard against pathological thresholds (see detect_stars).
    let sigma = sigma.max(0.5);
    let tophat_r = tophat_radius as usize;
    let bs = if block_size == 0 {
        32
    } else {
        block_size as usize
    };

    let stars = py
        .allow_threads(|| {
            let pool = get_pool();
            pool.install(|| {
                // Bin the input to the detection-image resolution (scale 1/2/4).
                let (binned, dw, dh): (Vec<u8>, usize, usize) = match bin {
                    1 => (Vec::new(), w, h),
                    2 => {
                        let (b, bw, bh) = bin2x2_mean(&owned, w, h);
                        (b, bw, bh)
                    }
                    _ => {
                        let (b, bw, bh) = bin4x4_mean(&owned, w, h);
                        (b, bw, bh)
                    }
                };
                // detection-source slice (binned or original).
                let det_src: &[u8] = if bin == 1 { &owned } else { &binned };

                // centroid image + its dims + coord scale.
                let (cent_img, cw, ch, cscale): (&[u8], usize, usize, usize) = if bin == 1 {
                    (&owned, w, h, 1)
                } else if centroid_full_res {
                    (&owned, w, h, scale)
                } else {
                    (det_src, dw, dh, 1)
                };

                let mut result = if let Some(bg) = &owned_bg {
                    // ---- Full per-pixel cached path. Subtract the temporal
                    // background image directly (gradients, vignetting AND
                    // fixed-pattern structure in one pass). Unlike the
                    // block-grid path, this model is a per-pixel (not per-tile)
                    // temporal median, so it resolves row-scale structure the
                    // block grid can't -- verified (multi-seed synthetic A/B,
                    // including adversarial per-row bias / fast row oscillation)
                    // to leave no measurable residual, so the trailing floor
                    // scan is skipped entirely (BgMode::ZeroFloor) rather than
                    // re-measuring an already-flat image.
                    let corrected = subtract_image(det_src, bg);
                    detect(
                        &corrected,
                        dw,
                        dh,
                        cent_img,
                        cw,
                        ch,
                        cscale,
                        sigma,
                        noise,
                        use_neon,
                        BgMode::ZeroFloor,
                        max_axis_ratio,
                        &kernel,
                        local_noise,
                    )
                } else if let Some((grid, gnx, gny)) = &block_grid {
                    // ---- Block-grid cached path. Subtract the bilinearly-
                    // interpolated 2-D background, then run the standard
                    // detection with the inline RowPercentile floor (same as the
                    // per-frame spatial modes). The supplied `noise` is used.
                    let corrected = subtract_block_grid(det_src, dw, dh, bs, grid, *gnx, *gny);
                    detect(
                        &corrected,
                        dw,
                        dh,
                        cent_img,
                        cw,
                        ch,
                        cscale,
                        sigma,
                        noise,
                        use_neon,
                        BgMode::RowPercentile,
                        max_axis_ratio,
                        &kernel,
                        local_noise,
                    )
                } else {
                    // ---- Row-offset cached path (original behavior).
                    let rof = owned_rof.as_ref().unwrap();
                    let tophat_buf: Option<Vec<u8>> = if tophat_r > 0 {
                        Some(white_tophat(det_src, dw, dh, tophat_r))
                    } else {
                        None
                    };
                    let det: &[u8] = tophat_buf.as_deref().unwrap_or(det_src);
                    // With tophat, the detection image background is ~0; use zero
                    // row_floors so the threshold is sigma*noise.
                    let zero_floors: Vec<u8>;
                    let floors: &[u8] = if tophat_r > 0 {
                        zero_floors = vec![0u8; rof.len()];
                        &zero_floors
                    } else {
                        rof
                    };
                    detect_cached(
                        det,
                        dw,
                        dh,
                        cent_img,
                        cw,
                        ch,
                        cscale,
                        sigma,
                        noise,
                        use_neon,
                        floors,
                        max_axis_ratio,
                        &kernel,
                        local_noise,
                    )
                };

                // When centroiding on the binned image (centroid_full_res=false,
                // bin>1), map coords back to full-res.
                if bin > 1 && !centroid_full_res {
                    let sc = scale as f64;
                    for s in result.iter_mut() {
                        s.x *= sc;
                        s.y *= sc;
                    }
                }
                Ok::<Vec<Star>, &'static str>(result)
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
    kernel: &MatchedKernel,
    local_noise: bool,
) -> Vec<Star> {
    if det_w < 7 || det_h < 7 {
        return Vec::new();
    }
    let sn2 = ((2.0 * sigma * noise + 0.5) as i32).max(2);
    let mf_thresh = mf_threshold(sigma, noise, kernel);

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
                    kernel,
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
                local_noise,
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

/// Compute the per-tile median grid of a uint8 image (Feature 5). Returns a 2-D
/// uint8 array of shape (grid_h, grid_w) where grid_w = ceil(width/block_size)
/// and grid_h = ceil(height/block_size). This is the 2-D analogue of
/// `compute_row_medians_py`: the background worker can median-stack frames in
/// Python, then call this on the (binned) temporal-median frame to build a
/// cached 2-D background model for `detect_stars_with_cache(block_offsets=...)`.
///
/// Run it at the same binning and `block_size` you will pass to
/// `detect_stars_with_cache` so the grid dimensions line up.
#[pyfunction]
#[pyo3(signature = (image, block_size=32u32))]
fn compute_block_medians_py<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<u8>,
    block_size: u32,
) -> PyResult<Bound<'py, numpy::PyArray2<u8>>> {
    let shape = image.shape();
    let (h, w) = (shape[0], shape[1]);
    let data: Vec<u8> = image
        .as_slice()
        .map_err(|_| PyValueError::new_err("image must be C-contiguous uint8"))?
        .to_vec();
    let bs = if block_size == 0 {
        32
    } else {
        block_size as usize
    };
    let (grid, nx, ny) = py.allow_threads(|| {
        let pool = get_pool();
        pool.install(|| compute_block_medians(&data, w, h, bs))
    });
    let arr = numpy::ndarray::Array2::from_shape_vec((ny, nx), grid)
        .map_err(|e| PyValueError::new_err(format!("grid reshape failed: {e}")))?;
    Ok(numpy::PyArray2::from_owned_array_bound(py, arr))
}

// =========================================================================
// ROI (window-list) detection — tracking-mode fast path (v0.14.0)
// =========================================================================
//
// The downstream diofinder tracking mode detects stars in ~48 px windows
// around predicted positions. Before this entry point it sliced numpy windows
// in Python and called the detector once per window — one GIL round-trip and
// one thread-pool entry per window. `detect_stars_roi` takes the full frame
// plus an (N, 4) window list, copies the frame once, releases the GIL once,
// and fans the windows out across the existing bounded rayon pool.
//
// Per-window pipeline (bin is always 1 — windows are far too small to bin):
//   background: per-row true median inside the window (`line_median`; the
//     inline cache-line-sampled row-percentile the full-frame bin=1 path uses
//     degenerates to ~1 sample per row at window widths, so the median is the
//     correct floor here — computed with the same `compute_row_medians` the
//     LineMedian path uses);
//   noise: MAD over the whole window (robust at 48x48 = 2304 px; the
//     full-frame patch-grid estimator requires >=64 px sides);
//   then the standard scan_band -> form_blobs -> gate_2d chain, keeping only
//   the brightest surviving star per window.

/// Minimum usable window side length (px) after clamping. Smaller windows are
/// skipped: gate_2d needs a 3-px context ring on each side, so below ~8 px
/// there is no interior left to detect in.
const ROI_MIN_SIDE: usize = 8;

/// Defensively clamp a caller-supplied window (x0, y0, x1, y1; x1/y1
/// exclusive) to the image bounds. Returns None if the clamped window is
/// degenerate (inverted, empty, or under ROI_MIN_SIDE on either side).
fn clamp_window(
    x0: i64,
    y0: i64,
    x1: i64,
    y1: i64,
    w: usize,
    h: usize,
) -> Option<(usize, usize, usize, usize)> {
    let cx0 = x0.clamp(0, w as i64) as usize;
    let cy0 = y0.clamp(0, h as i64) as usize;
    let cx1 = x1.clamp(0, w as i64) as usize;
    let cy1 = y1.clamp(0, h as i64) as usize;
    if cx1 <= cx0 || cy1 <= cy0 {
        return None;
    }
    if cx1 - cx0 < ROI_MIN_SIDE || cy1 - cy0 < ROI_MIN_SIDE {
        return None;
    }
    Some((cx0, cy0, cx1, cy1))
}

/// MAD noise estimate over an entire (small) buffer: 1.4826 * median(|x -
/// median(x)|), floored at 0.5. The window analogue of the patch-based
/// `estimate_noise` (which needs >=64 px sides and would return the floor for
/// every tracking window). Two O(n) histogram passes, no allocation.
fn estimate_noise_mad_window(data: &[u8]) -> f64 {
    const MIN_NOISE: f64 = 0.5;
    let n = data.len();
    if n == 0 {
        return MIN_NOISE;
    }
    let mut hist = [0u32; 256];
    for &p in data {
        hist[p as usize] += 1;
    }
    let target = (n as u32).div_ceil(2);
    let mut acc = 0u32;
    let mut median = 0u8;
    for (v, &c) in hist.iter().enumerate() {
        acc += c;
        if acc >= target {
            median = v as u8;
            break;
        }
    }
    let mut dhist = [0u32; 256];
    for &p in data {
        dhist[(p as i32 - median as i32).unsigned_abs() as usize] += 1;
    }
    acc = 0;
    let mut mad = 0u8;
    for (v, &c) in dhist.iter().enumerate() {
        acc += c;
        if acc >= target {
            mad = v as u8;
            break;
        }
    }
    (1.4826 * mad as f64).max(MIN_NOISE)
}

/// Detect the single brightest star in one contiguous window buffer
/// (window-local coordinates; the caller adds the window offset back).
/// Reuses the full-frame internals: `compute_row_medians` (line_median floor),
/// `scan_band`, `form_blobs`, `gate_2d` at scale 1 (bin=1, detection image ==
/// centroid image). Returns None when nothing passes the gates.
fn detect_roi_window(
    win: &[u8],
    ww: usize,
    wh: usize,
    sigma: f64,
    kernel: &MatchedKernel,
    local_noise: bool,
    max_axis_ratio: f64,
) -> Option<Star> {
    // The gate window must fit inside the row (wide kernels need more width).
    if ww < 2 * kernel.half + 1 || wh < 7 {
        return None;
    }
    let noise = estimate_noise_mad_window(win);
    let sn2 = ((2.0 * sigma * noise + 0.5) as i32).max(2);
    let mf_thresh = mf_threshold(sigma, noise, kernel);
    let floors = compute_row_medians(win, ww, wh);
    let cands = scan_band(win, ww, 0, wh, sn2, false, Some(&floors), kernel, mf_thresh);
    if cands.is_empty() {
        return None;
    }
    // Extended-object cap. The full-frame formula (det_w/100) would give the
    // 3-px minimum for any window and reject a bright bin=1 star whose blob
    // spans ~5 rows, so scale with the window instead: a point source in a
    // tracking window is comfortably under a quarter of the window side.
    let max_size = (ww.max(wh) / 4).max(3);
    let blobs = form_blobs(&cands, wh);
    let mut best: Option<Star> = None;
    for blob in &blobs {
        if let Some(s) = gate_2d(
            &cands,
            blob,
            win,
            ww,
            wh,
            win,
            ww,
            wh,
            1,
            noise,
            sigma,
            local_noise,
            max_size,
            max_axis_ratio,
        ) {
            if best.as_ref().is_none_or(|b| s.brightness > b.brightness) {
                best = Some(s);
            }
        }
    }
    best
}

/// Extract an (N, 4) int32 or int64 numpy window list into Vec<[i64; 4]>.
fn extract_windows_array(windows: &Bound<'_, PyAny>) -> PyResult<Vec<[i64; 4]>> {
    fn rows<T: Copy + Into<i64>>(s: &[T]) -> Vec<[i64; 4]> {
        s.chunks_exact(4)
            .map(|c| [c[0].into(), c[1].into(), c[2].into(), c[3].into()])
            .collect()
    }
    if let Ok(a) = windows.extract::<numpy::PyReadonlyArray2<i64>>() {
        if a.shape()[1] != 4 {
            return Err(PyValueError::new_err(
                "windows must have shape (N, 4): x0, y0, x1, y1 (exclusive)",
            ));
        }
        let s = a
            .as_slice()
            .map_err(|_| PyValueError::new_err("windows must be C-contiguous"))?;
        return Ok(rows(s));
    }
    if let Ok(a) = windows.extract::<numpy::PyReadonlyArray2<i32>>() {
        if a.shape()[1] != 4 {
            return Err(PyValueError::new_err(
                "windows must have shape (N, 4): x0, y0, x1, y1 (exclusive)",
            ));
        }
        let s = a
            .as_slice()
            .map_err(|_| PyValueError::new_err("windows must be C-contiguous"))?;
        return Ok(rows(s));
    }
    Err(PyValueError::new_err(
        "windows must be a 2-D numpy int32 or int64 array of shape (N, 4): \
         x0, y0, x1, y1 (exclusive)",
    ))
}

/// Detect stars in N small windows of a full frame with ONE call.
///
/// The tracking-mode fast path: instead of slicing numpy windows in Python and
/// calling `detect_stars` once per window (one GIL round-trip each), pass the
/// full frame plus the window list. The frame is copied once, the GIL is
/// released once, and the windows fan out across the bounded thread pool.
///
/// Args:
///   image:   2-D C-contiguous numpy uint8 array (height, width) — the FULL
///            frame, not pre-sliced windows.
///   windows: 2-D numpy int32 or int64 array of shape (N, 4), each row
///            (x0, y0, x1, y1) with x1/y1 EXCLUSIVE. Windows are defensively
///            re-clamped to the image bounds; windows degenerate after
///            clamping (< 8 px on a side, inverted, or empty) are skipped.
///   sigma:   detection threshold in noise sigmas (same as detect_stars;
///            values below 0.5 are clamped to 0.5).
///   kernel_sigma: matched-filter kernel width, 1.0-4.0 (same as detect_stars).
///   local_noise:  same as detect_stars.
///   max_axis_ratio: same semantics as detect_stars (default Inf = off).
///
/// Per-window pipeline (bin is always 1 — a 48-px window must not be binned):
/// per-row median background floor (line_median), whole-window MAD noise,
/// then the standard matched-filter gate / blob / 2-D gate chain. At most ONE
/// star — the brightest surviving detection — is returned per window.
///
/// Probe `getattr(star_detect, "HAS_ROI", False)` before calling (native
/// functions don't support inspect.signature).
///
/// Returns: list of (x, y, brightness, peak), brightest first, in FULL-FRAME
///          coordinates (window offset added back); (0.5, 0.5) is the center
///          of the frame's top-left pixel, exactly as detect_stars.
#[pyfunction]
#[pyo3(signature = (
    image,
    windows,
    sigma=8.0,
    kernel_sigma=1.5,
    local_noise=true,
    max_axis_ratio=f64::INFINITY,
))]
fn detect_stars_roi(
    py: Python<'_>,
    image: PyReadonlyArray2<u8>,
    windows: &Bound<'_, PyAny>,
    sigma: f64,
    kernel_sigma: f64,
    local_noise: bool,
    max_axis_ratio: f64,
) -> PyResult<Vec<(f64, f64, f64, i64)>> {
    let shape = image.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err("image must be 2-D"));
    }
    let (h, w) = (shape[0], shape[1]);

    let win_list = extract_windows_array(windows)?;

    if !(1.0..=4.0).contains(&kernel_sigma) {
        return Err(PyValueError::new_err(format!(
            "kernel_sigma must be in [1.0, 4.0]; got {kernel_sigma}"
        )));
    }
    let kernel = generate_matched_kernel(kernel_sigma);
    let sigma = sigma.max(0.5);

    // Copy the image once so we can drop the GIL (the existing pattern).
    let owned: Vec<u8> = image
        .as_slice()
        .map_err(|_| {
            PyValueError::new_err("image must be C-contiguous uint8; use np.ascontiguousarray")
        })?
        .to_vec();

    let mut stars: Vec<Star> = py.allow_threads(|| {
        let pool = get_pool();
        pool.install(|| {
            win_list
                .par_iter()
                .filter_map(|&[x0, y0, x1, y1]| {
                    let (cx0, cy0, cx1, cy1) = clamp_window(x0, y0, x1, y1, w, h)?;
                    let ww = cx1 - cx0;
                    let wh = cy1 - cy0;
                    // Contiguous window copy (a few KB; keeps the row math of
                    // the shared internals simple and cache-friendly).
                    let mut win = Vec::with_capacity(ww * wh);
                    for y in cy0..cy1 {
                        win.extend_from_slice(&owned[y * w + cx0..y * w + cx1]);
                    }
                    detect_roi_window(&win, ww, wh, sigma, &kernel, local_noise, max_axis_ratio)
                        .map(|s| Star {
                            x: s.x + cx0 as f64,
                            y: s.y + cy0 as f64,
                            ..s
                        })
                })
                .collect()
        })
    });

    stars.sort_by(|a, b| {
        b.brightness
            .partial_cmp(&a.brightness)
            .unwrap_or(Ordering::Equal)
    });

    Ok(stars
        .into_iter()
        .map(|s| (s.x, s.y, s.brightness, s.peak as i64))
        .collect())
}

#[pymodule]
fn star_detect(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(detect_stars, m)?)?;
    m.add_function(wrap_pyfunction!(detect_stars_with_cache, m)?)?;
    m.add_function(wrap_pyfunction!(detect_stars_roi, m)?)?;
    m.add_function(wrap_pyfunction!(compute_row_medians_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_block_medians_py, m)?)?;
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    // Capability flags for downstream probes (native functions don't support
    // inspect.signature, so consumers check getattr(star_detect,
    // "HAS_BG_IMAGE", False) before passing bg_image=, and
    // getattr(star_detect, "HAS_ROI", False) before calling detect_stars_roi).
    m.add("HAS_BG_IMAGE", true)?;
    m.add("HAS_ROI", true)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg_image(w: usize, h: usize, seed: u64) -> Vec<u8> {
        let mut s = seed;
        (0..w * h)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s >> 33) as u8
            })
            .collect()
    }

    #[test]
    fn clamp_window_defensive() {
        // In-bounds window passes through unchanged.
        assert_eq!(
            clamp_window(10, 20, 58, 68, 400, 300),
            Some((10, 20, 58, 68))
        );
        // Out-of-bounds coordinates are clamped to the image, not rejected,
        // as long as >= ROI_MIN_SIDE survives on each side.
        assert_eq!(
            clamp_window(-20, -5, 30, 40, 400, 300),
            Some((0, 0, 30, 40))
        );
        assert_eq!(
            clamp_window(380, 280, 500, 500, 400, 300),
            Some((380, 280, 400, 300))
        );
        // Degenerate after clamping (< ROI_MIN_SIDE on a side): skipped.
        assert_eq!(clamp_window(395, 0, 500, 48, 400, 300), None); // 5 px wide
        assert_eq!(clamp_window(0, 0, 48, 7, 400, 300), None); // 7 px tall

        // Inverted / empty / fully outside: skipped.
        assert_eq!(clamp_window(50, 50, 10, 90, 400, 300), None);
        assert_eq!(clamp_window(50, 50, 50, 90, 400, 300), None);
        assert_eq!(clamp_window(500, 500, 600, 600, 400, 300), None);
        assert_eq!(clamp_window(-100, -100, -10, -10, 400, 300), None);
    }

    #[test]
    fn mad_window_noise_estimate() {
        // Flat window: MAD = 0 -> floored at 0.5.
        assert_eq!(estimate_noise_mad_window(&vec![20u8; 48 * 48]), 0.5);
        assert_eq!(estimate_noise_mad_window(&[]), 0.5);
        // Equal thirds of {16, 20, 24}: median = 20, deviations are 0 (1/3)
        // and 4 (2/3), so MAD = 4 -> 1.4826 * 4.
        let alt: Vec<u8> = (0..48 * 48).map(|i| 16 + 4 * (i % 3) as u8).collect();
        let n = estimate_noise_mad_window(&alt);
        assert!((n - 1.4826 * 4.0).abs() < 1e-9, "n = {n}");
    }

    /// Synthetic Gaussian star on a flat background, window-local coords.
    fn star_window(ww: usize, wh: usize, bg: u8, cx: f64, cy: f64, amp: f64, psf: f64) -> Vec<u8> {
        (0..ww * wh)
            .map(|i| {
                let x = (i % ww) as f64;
                let y = (i / ww) as f64;
                let d2 = (x - cx).powi(2) + (y - cy).powi(2);
                let v = bg as f64 + amp * (-d2 / (2.0 * psf * psf)).exp();
                v.round().clamp(0.0, 255.0) as u8
            })
            .collect()
    }

    #[test]
    fn roi_window_finds_single_star() {
        let kernel = generate_matched_kernel(1.5);
        // Star centered on pixel (24, 24) -> expected centroid (24.5, 24.5)
        // in the (0.5, 0.5)-center-of-pixel convention.
        let win = star_window(48, 48, 20, 24.0, 24.0, 130.0, 1.0);
        let s = detect_roi_window(&win, 48, 48, 8.0, &kernel, true, f64::INFINITY)
            .expect("bright star must be detected");
        assert!((s.x - 24.5).abs() < 0.5, "x = {}", s.x);
        assert!((s.y - 24.5).abs() < 0.5, "y = {}", s.y);
        assert!(s.peak >= 140);
    }

    #[test]
    fn roi_window_empty_returns_none() {
        let kernel = generate_matched_kernel(1.5);
        // Flat background: nothing to detect.
        let win = vec![20u8; 48 * 48];
        assert!(detect_roi_window(&win, 48, 48, 8.0, &kernel, true, f64::INFINITY).is_none());
        // Too narrow for the gate window: skipped, not panicking.
        assert!(
            detect_roi_window(&win[..6 * 48], 6, 48, 8.0, &kernel, true, f64::INFINITY).is_none()
        );
    }

    #[test]
    fn roi_window_picks_brightest_of_two() {
        let kernel = generate_matched_kernel(1.5);
        // Two well-separated stars in one window; only the brighter returns.
        let mut win = star_window(48, 48, 20, 14.0, 14.0, 130.0, 1.0);
        let faint = star_window(48, 48, 0, 34.0, 34.0, 60.0, 1.0);
        for (w, f) in win.iter_mut().zip(faint.iter()) {
            *w = w.saturating_add(*f);
        }
        let s = detect_roi_window(&win, 48, 48, 8.0, &kernel, true, f64::INFINITY)
            .expect("must detect");
        assert!(
            (s.x - 14.5).abs() < 0.5 && (s.y - 14.5).abs() < 0.5,
            "picked ({}, {})",
            s.x,
            s.y
        );
    }

    #[test]
    fn subtract_image_saturates_and_flattens() {
        // A gradient background subtracted from gradient+star leaves only the
        // star; pixels where bg >= data saturate at 0 (never wrap).
        let w = 8usize;
        let h = 4usize;
        let bg: Vec<u8> = (0..w * h).map(|i| (i % w) as u8 * 10).collect();
        let mut data = bg.clone();
        data[2 * w + 3] = data[2 * w + 3].saturating_add(50); // the "star"
        data[w + 1] = 0; // dead pixel below the background
        let out = subtract_image(&data, &bg);
        assert_eq!(out[2 * w + 3], 50);
        assert_eq!(out[w + 1], 0); // saturated, not wrapped
        let residue: u32 = out
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != 2 * w + 3)
            .map(|(_, &v)| v as u32)
            .sum();
        assert_eq!(residue, 0);
    }

    /// Naive O(n*r) sliding-window extreme with neutral padding — reference
    /// for the van Herk implementation.
    fn naive_extreme_1d(src: &[u8], radius: usize, is_max: bool) -> Vec<u8> {
        let n = src.len();
        let neutral = if is_max { 0u8 } else { 255u8 };
        (0..n)
            .map(|j| {
                let lo = j.saturating_sub(radius);
                let hi = (j + radius + 1).min(n);
                let mut acc = neutral;
                for &v in &src[lo..hi] {
                    acc = if is_max { acc.max(v) } else { acc.min(v) };
                }
                acc
            })
            .collect()
    }

    #[test]
    fn kernel_sigma_1_5_matches_legacy() {
        // REQUIRED: sigma=1.5 must reproduce the historical hardcoded kernel
        // exactly, so default-path results are bit-identical to pre-0.12.
        let k = generate_matched_kernel(1.5);
        assert_eq!(k.taps, vec![-50, -15, 35, 60, 35, -15, -50]);
        assert_eq!(k.half, 3);
        // L2 norm forced to exactly 107.0 for the default kernel.
        assert_eq!(k.l2, 107.0);
        // mf_threshold must equal the old hardcoded-107.0 computation.
        let want = (8.0 * 2.5 * 107.0 + 0.5) as i32;
        assert_eq!(mf_threshold(8.0, 2.5, &k), want);
    }

    #[test]
    fn axis_ratio_2d_moments() {
        // Round blob: var_x==var_y, cov==0 => ratio 1.
        assert!((cov2x2_axis_ratio2(2.0, 2.0, 0.0) - 1.0).abs() < 1e-9);
        // Axis-aligned elongation 4:1 in variance => sqrt ratio 2 => ratio^2 4.
        assert!((cov2x2_axis_ratio2(4.0, 1.0, 0.0) - 4.0).abs() < 1e-9);
        // Diagonal trail: equal diagonal variances but strong off-diagonal.
        // [[3,2.9],[2.9,3]] -> lam = 3 +/- 2.9 => 5.9 / 0.1 = 59. The separable
        // var_x/var_y-only test would see ratio 1 here and MISS the trail; the
        // full 2x2 form correctly flags it.
        let r2 = cov2x2_axis_ratio2(3.0, 3.0, 2.9);
        assert!(r2 > 50.0, "diagonal trail ratio^2 = {r2}");
        // And it would be rejected for any max_axis_ratio up to ~7.6 (sqrt 59).
        assert!(r2 > 3.0 * 3.0);
    }

    #[test]
    fn kernel_generation_widths_and_dc() {
        // Half-width clamp and tap counts, and DC-free (sum==0) for all sigma.
        for (sigma, exp_taps) in [
            (1.0, 7usize),
            (1.5, 7),
            (2.0, 9),
            (2.5, 11),
            (3.0, 13),
            (4.0, 15),
        ] {
            let k = generate_matched_kernel(sigma);
            assert_eq!(k.taps.len(), exp_taps, "sigma={sigma}");
            assert_eq!(k.taps.iter().sum::<i32>(), 0, "DC-free sigma={sigma}");
            assert_eq!(k.half, exp_taps / 2);
            // Symmetric kernel.
            for i in 0..k.taps.len() {
                assert_eq!(k.taps[i], k.taps[k.taps.len() - 1 - i], "sym sigma={sigma}");
            }
            // Center coefficient rounds to 60.
            assert_eq!(k.taps[k.half], 60, "center sigma={sigma}");
            // Non-default kernels use true L2 norm (not 107.0).
            if sigma != 1.5 {
                let true_l2 = (k.taps.iter().map(|&t| (t * t) as f64).sum::<f64>()).sqrt();
                assert!((k.l2 - true_l2).abs() < 1e-9, "l2 sigma={sigma}");
            }
        }
    }

    #[test]
    fn gate_1d_default_kernel_matches_hardcoded() {
        // The generalized gate_1d with the sigma=1.5 kernel must produce the
        // same response/decision as the old hardcoded dot product on a window.
        let k = generate_matched_kernel(1.5);
        // A clean peak window.
        let g = [10u8, 20, 80, 200, 80, 20, 10];
        let resp = -50 * (g[0] as i32 + g[6] as i32) - 15 * (g[1] as i32 + g[5] as i32)
            + 35 * (g[2] as i32 + g[4] as i32)
            + 60 * g[3] as i32;
        let thr = resp; // threshold exactly at response: passes (>=)
        assert!(gate_1d(&g, &k, thr));
        assert!(!gate_1d(&g, &k, thr + 1)); // just above response: fails
    }

    #[test]
    fn block_medians_grid_matches_block_percentile_internal() {
        // compute_block_medians must produce the same grid the per-frame
        // block_percentile_bg uses internally (subtracting it reproduces it).
        for &(w, h, bs) in &[(96usize, 64usize, 32usize), (100, 70, 16)] {
            let img = lcg_image(w, h, (w * h + bs) as u64);
            let (grid, nx, ny) = compute_block_medians(&img, w, h, bs);
            let via_grid = subtract_block_grid(&img, w, h, bs, &grid, nx, ny);
            assert_eq!(
                via_grid,
                block_percentile_bg(&img, w, h, bs),
                "{w}x{h} bs={bs}"
            );
        }
    }

    #[test]
    fn van_herk_matches_naive() {
        for &n in &[1usize, 7, 31, 64, 257] {
            for &r in &[1usize, 3, 12, 30] {
                for &is_max in &[false, true] {
                    let src = lcg_image(n, 1, (n * 31 + r) as u64);
                    let mut s = MorphScratch::default();
                    let mut out = vec![0u8; n];
                    extreme_1d_into(&src, r, is_max, &mut s, &mut out);
                    assert_eq!(
                        out,
                        naive_extreme_1d(&src, r, is_max),
                        "n={n} r={r} is_max={is_max}"
                    );
                }
            }
        }
    }

    #[test]
    fn scratch_reuse_across_calls_is_clean() {
        // Same scratch across different radii/polarity must not leak state.
        let mut s = MorphScratch::default();
        let a = lcg_image(101, 1, 7);
        let mut out = vec![0u8; 101];
        extreme_1d_into(&a, 30, true, &mut s, &mut out); // grow buffers large
        extreme_1d_into(&a, 3, false, &mut s, &mut out); // then small window
        assert_eq!(out, naive_extreme_1d(&a, 3, false));
    }

    #[test]
    fn transpose_matches_naive_and_roundtrips() {
        for &(w, h) in &[(1usize, 1usize), (37, 53), (64, 64), (33, 95), (97, 31)] {
            let img = lcg_image(w, h, (w * 1000 + h) as u64);
            let (t, tw, th) = transpose(&img, w, h);
            assert_eq!((tw, th), (h, w));
            for y in 0..h {
                for x in 0..w {
                    assert_eq!(t[x * h + y], img[y * w + x], "({x},{y}) in {w}x{h}");
                }
            }
            let (back, bw, bh) = transpose(&t, tw, th);
            assert_eq!((bw, bh), (w, h));
            assert_eq!(back, img);
        }
    }

    /// Reference top-hat: the pre-fusion 4-transpose pipeline built from the
    /// naive 1-D extreme. Byte-identical output is required.
    fn reference_tophat(data: &[u8], w: usize, h: usize, radius: usize) -> Vec<u8> {
        let row_op = |img: &[u8], iw: usize, ih: usize, is_max: bool| -> Vec<u8> {
            let mut out = vec![0u8; iw * ih];
            for y in 0..ih {
                out[y * iw..y * iw + iw].copy_from_slice(&naive_extreme_1d(
                    &img[y * iw..y * iw + iw],
                    radius,
                    is_max,
                ));
            }
            out
        };
        let tr = |img: &[u8], iw: usize, ih: usize| -> Vec<u8> {
            let mut out = vec![0u8; iw * ih];
            for y in 0..ih {
                for x in 0..iw {
                    out[x * ih + y] = img[y * iw + x];
                }
            }
            out
        };
        let col_op = |img: &[u8], iw: usize, ih: usize, is_max: bool| -> Vec<u8> {
            tr(&row_op(&tr(img, iw, ih), ih, iw, is_max), ih, iw)
        };
        let eroded = col_op(&row_op(data, w, h, false), w, h, false);
        let opened = col_op(&row_op(&eroded, w, h, true), w, h, true);
        data.iter()
            .zip(opened.iter())
            .map(|(&a, &b)| a.saturating_sub(b))
            .collect()
    }

    #[test]
    fn tophat_fused_matches_reference() {
        for &(w, h, r) in &[(64usize, 48usize, 5usize), (95, 33, 12), (40, 40, 3)] {
            let img = lcg_image(w, h, (w + h + r) as u64);
            assert_eq!(
                white_tophat(&img, w, h, r),
                reference_tophat(&img, w, h, r),
                "{w}x{h} r={r}"
            );
        }
    }

    #[test]
    fn block_median_matches_sorted_reference() {
        // The histogram median must select exactly what vals.sort();vals[len/2]
        // selected before. Compare full background outputs.
        let sorted_block_bg = |data: &[u8], width: usize, height: usize, bs: usize| -> Vec<u8> {
            let nx = width.div_ceil(bs);
            let ny = height.div_ceil(bs);
            let meds: Vec<u8> = (0..nx * ny)
                .map(|bi| {
                    let (bx, by) = (bi % nx, bi / nx);
                    let (x0, y0) = (bx * bs, by * bs);
                    let (x1, y1) = ((x0 + bs).min(width), (y0 + bs).min(height));
                    let mut vals: Vec<u8> = Vec::new();
                    for y in y0..y1 {
                        vals.extend_from_slice(&data[y * width + x0..y * width + x1]);
                    }
                    vals.sort_unstable();
                    vals[vals.len() / 2]
                })
                .collect();
            meds
        };
        for &(w, h, bs) in &[(96usize, 64usize, 32usize), (100, 70, 16), (33, 33, 32)] {
            let img = lcg_image(w, h, (w * h) as u64);
            let want = sorted_block_bg(&img, w, h, bs);
            // Re-derive the medians the production path computes by probing
            // block centers of the subtracted image: instead, recompute via the
            // same internal expression — simplest is to compare the medians by
            // reconstructing them from block_percentile_bg with a flat image
            // delta trick. Direct check: run production tile-median by calling
            // block_percentile_bg on an image and comparing against a
            // sort-reference reimplementation of the WHOLE function.
            let bg_ref = {
                let nx = w.div_ceil(bs);
                let ny = h.div_ceil(bs);
                let meds = want;
                let half = bs as f32 / 2.0;
                let mut out = vec![0u8; w * h];
                for y in 0..h {
                    let by_f = (y as f32 - half) / bs as f32;
                    let by0 = (by_f.floor() as isize).max(0).min(ny as isize - 1) as usize;
                    let by1 = (by0 + 1).min(ny - 1);
                    let fy = (by_f - by0 as f32).clamp(0.0, 1.0);
                    for x in 0..w {
                        let bx_f = (x as f32 - half) / bs as f32;
                        let bx0 = (bx_f.floor() as isize).max(0).min(nx as isize - 1) as usize;
                        let bx1 = (bx0 + 1).min(nx - 1);
                        let fx = (bx_f - bx0 as f32).clamp(0.0, 1.0);
                        let m00 = meds[by0 * nx + bx0] as f32;
                        let m10 = meds[by0 * nx + bx1] as f32;
                        let m01 = meds[by1 * nx + bx0] as f32;
                        let m11 = meds[by1 * nx + bx1] as f32;
                        let bg = m00 * (1.0 - fx) * (1.0 - fy)
                            + m10 * fx * (1.0 - fy)
                            + m01 * (1.0 - fx) * fy
                            + m11 * fx * fy;
                        out[y * w + x] = img[y * w + x].saturating_sub(bg as u8);
                    }
                }
                out
            };
            assert_eq!(
                block_percentile_bg(&img, w, h, bs),
                bg_ref,
                "{w}x{h} bs={bs}"
            );
        }
    }
}
