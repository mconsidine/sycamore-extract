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
// 1-D gate
// =========================================================================
// Two variants, selectable at the call site via GateMode:
//
//   Cedar: the original 7-pixel heuristic from cedar-detect (Apache-2.0).
//     A pixel is a candidate if its value rises above local background by a
//     sigma*noise margin AND is a local maximum within its 7-pixel neighborhood
//     AND has a roughly uniform background on both sides. Branch-heavy,
//     ordered for selectivity (cheapest, most-rejecting tests first). Battle-
//     tested across years of real-sky frames in the PiFinder ecosystem.
//
//   MatchedFilter: standard signal-detection construction. Convolve the 7-pixel
//     window with a discrete approximation of the expected stellar PSF (a tight
//     Gaussian), and accept pixels whose response exceeds a threshold matched
//     to maintain the same false-positive rate as the cedar gate. This is the
//     optimal linear detector for a known-shape pulse in additive Gaussian
//     noise (Van Trees, _Detection, Estimation, and Modulation Theory_, 1968;
//     classical matched-filter result going back to North 1943 and Turin 1960).
//     Implementation-wise: fewer branches, cleaner SIMD story, and the kernel
//     coefficients are tunable for sensor-specific PSF. Performance on real
//     finder frames is an open question; see tests/ab_gates.py.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateMode {
    /// 7-pixel cedar-detect heuristic. The default.
    Cedar,
    /// Matched filter with a Gaussian-shaped 7-tap kernel. Experimental.
    MatchedFilter,
}

fn parse_gate_mode(s: &str) -> PyResult<GateMode> {
    match s.to_ascii_lowercase().as_str() {
        "cedar" | "default" | "heuristic" => Ok(GateMode::Cedar),
        "matched_filter" | "matchedfilter" | "mf" | "matched" => Ok(GateMode::MatchedFilter),
        other => Err(PyValueError::new_err(format!(
            "gate_mode must be 'cedar' or 'matched_filter', got '{other}'"
        ))),
    }
}

// Cedar gate: gate = [lb, lm, l, c, r, rm, rb]; returns true if center `c` is
// a candidate.
#[inline(always)]
fn gate_1d_cedar(g: &[u8], sn2: i32, sn3: i32) -> bool {
    let lb = g[0] as i32;
    let rb = g[6] as i32;
    let lm = g[1] as i32;
    let l = g[2] as i32;
    let c = g[3] as i32;
    let r = g[4] as i32;
    let rm = g[5] as i32;

    if c + c - (lb + rb) < sn2 {
        return false; // center not above local background by sigma*noise
    }
    if l > c || c < r {
        return false; // not >= immediate neighbors
    }
    if lm >= c || c <= rm {
        return false; // not strictly brighter than margins
    }
    if l == c && lm > r {
        return false; // tie-break: left owns this candidate
    }
    if c == r && l <= rm {
        return false; // tie-break: right owns this candidate
    }
    if (lb - rb).abs() > sn3 {
        return false; // borders not ~uniform
    }
    true
}

// Matched filter response, integer arithmetic.
//
// Kernel: a Gaussian sampled at integer offsets sigma=1.5 (FWHM ~3.5 px),
// scaled and mean-removed so DC (background) cancels exactly. The sigma=1.5
// width was chosen to match the actual PSF FWHM seen on representative
// Raspberry Pi HQ Camera + finder lens frames; a tighter kernel (sigma=1.0)
// systematically rejected real, well-sampled stars whose energy extended
// past the inner 3 pixels. See tests/inspect_disagreement.py for the
// calibration procedure.
//
// Derivation (Python):
//   raw    = exp(-x^2/(2*1.5^2)) for x in -3..3
//          = [0.1353, 0.4111, 0.8007, 1.0, 0.8007, 0.4111, 0.1353]
//   zm     = raw - mean(raw)              (mean-removed Gaussian)
//   scale  = 60 / zm[center]              (target center coefficient ~60)
//   k      = round(zm * scale), symmetrized
//          = [-50, -15, 35, 60, 35, -15, -50]   sum = 0, symmetric
//
// k now sums to 0, so a uniform background contributes 0 to the response —
// no separate background-subtraction step required. The response is the dot
// product <window, k>, an i32. For pure white noise of std `noise`, the
// response has std `noise * ||k||_2`. We set the threshold so the false-
// positive rate matches the cedar gate's "sigma*noise above local background"
// criterion.
//
// The kernel norm:
//   sqrt(50^2 + 15^2 + 35^2 + 60^2 + 35^2 + 15^2 + 50^2)
//   = sqrt(2500+225+1225+3600+1225+225+2500) = sqrt(11500) ~= 107.24
//
// Caller passes a precomputed threshold (`mf_thresh` = sigma * noise * 107).
// Local-maximum test is kept (3-pixel) to avoid double-claiming adjacent
// candidates from the same star — without it the matched filter spreads
// detections over 2-3 pixels per star.
//
// EMPIRICAL NOTE: At the same nominal sigma value, this matched filter is
// MORE CONSERVATIVE than the cedar gate. On real-sky HQ Camera frames, MF at
// sigma=8 detects roughly the same star count as cedar at sigma=9-10. The
// threshold derivation assumes Gaussian white noise in the response; real
// frames have lower-frequency structure (residual background gradients,
// sensor patterns) that adds variance to the matched-filter response
// specifically. Cedar's local-max heuristic isn't sensitive to that
// structure; matched filter is.
//
// This is not a bug — it's a different point in the detection-theory
// tradeoff space. Users switching gate modes should expect to lower sigma
// by 1-2 to maintain the same detected count. The matched filter's
// advantage is a more principled false-positive rate on truly Gaussian
// backgrounds; cedar's advantage is more permissive detection on real
// frames with their characteristic structured noise.
#[inline(always)]
fn gate_1d_mf(g: &[u8], mf_thresh: i32) -> bool {
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
    // Local-maximum suppression: don't double-claim neighbors that are part of
    // the same star. We approximate this via the raw pixel values, same as
    // cedar — if `c` isn't the brightest of {l, c, r} we let the neighbor win.
    if g2 > g3 || g3 < g4 {
        return false;
    }
    // Tie-breaks identical to cedar's, for consistency when the two modes are
    // compared on the same frame.
    if g2 == g3 && g1 > g4 {
        return false;
    }
    if g3 == g4 && g2 <= g5 {
        return false;
    }
    true
}

// Compute the matched-filter threshold from sigma/noise. Kept as a function
// so the caller can compute once per band, not per pixel.
#[inline]
fn mf_threshold(sigma: f64, noise: f64) -> i32 {
    // Kernel L2 norm: ~107.24 (sigma=1.5 Gaussian, see gate_1d_mf derivation).
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

// Scan one band of rows [y0, y1) producing candidates in raster order.
// `row_floors`, if Some, supplies a precomputed per-row background floor
// (e.g. from LineMedian). If None, the in-loop 25th-percentile of cache-line
// samples is used.
// `gate_mode` selects between the cedar-detect heuristic gate and the
// matched-filter alternative. `mf_thresh` is the precomputed threshold for
// the matched filter (ignored when gate_mode is Cedar).
#[allow(clippy::too_many_arguments)]
fn scan_band(
    data: &[u8],
    width: usize,
    y0: usize,
    y1: usize,
    sn2: i32,
    sn3: i32,
    use_neon: bool,
    row_floors: Option<&[u8]>,
    gate_mode: GateMode,
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

        // For each pixel that cleared the prefilter, run the full 7-pixel gate.
        for &x in &hits {
            let x = x as usize;
            if x < GATE_HALF || x + GATE_HALF >= width {
                continue;
            }
            let g = &row[x - GATE_HALF..x + GATE_HALF + 1];
            let pass = match gate_mode {
                GateMode::Cedar => gate_1d_cedar(g, sn2, sn3),
                GateMode::MatchedFilter => gate_1d_mf(g, mf_thresh),
            };
            if pass {
                out.push(Candidate { x: x as u32, y: y as u32 });
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
        Dsu { parent: (0..n).collect() }
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
fn ring_stats(data: &[u8], w: usize, x0: usize, y0: usize, bw: usize, bh: usize) -> (f64, u8, u8, f64) {
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
    let (bg_cent, _, _, _) =
        ring_stats(cent_img, cent_w, mx0.saturating_sub(1), my0.saturating_sub(1), mw + 2, mh + 2);

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

    Some(Star { x: cx, y: cy, brightness, peak })
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
            let s = r0[2 * bx] as u32 + r0[2 * bx + 1] as u32
                  + r1[2 * bx] as u32 + r1[2 * bx + 1] as u32;
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
    gate_mode: GateMode,
) -> Vec<Star> {
    if det_w < 7 || det_h < 7 {
        return Vec::new();
    }
    let sn2 = ((2.0 * sigma * noise + 0.5) as i32).max(2);
    let sn3 = ((3.0 * sigma * noise + 0.5) as i32).max(3);
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
                    sn3,
                    use_neon,
                    row_floors.as_deref(),
                    gate_mode,
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

    stars.sort_by(|a, b| b.brightness.partial_cmp(&a.brightness).unwrap_or(Ordering::Equal));
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
///   noise: optional precomputed noise level; estimated if None.
///   bin:   1 (full res) or 2 (detect on a 2x2-binned image for speed). Centroids
///          are returned in full-resolution pixel coordinates regardless.
///   centroid_full_res: when bin=2, perform centroiding on the full-resolution
///          image. Slightly more work per star, but recovers the sub-pixel
///          precision that scaling-from-binned loses. No effect when bin=1.
///   use_neon: when true and built for aarch64, use the explicit NEON threshold
///          prefilter. No-op on non-aarch64 targets.
///   bg_mode: "row_percentile" (default) or "line_median". RowPercentile is
///          the cheapest mode and works well on dark-sky frames with mild
///          vignetting. LineMedian computes a true per-row median (via 256-bin
///          histogram, parallel across rows) and is more robust to per-row
///          offset noise, vertical brightness gradients, and twilight/light-
///          pollution backgrounds. Equivalent to olive-solve's
///          FastBgSubMode::LineMedian.
///   max_axis_ratio: optional cap on detected blob elongation. A trail or
///          satellite streak has axis_ratio >> 1; an in-focus star is ~1.
///          Default Inf (no filtering). Recommend 3-5 for a finder.
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
    gate_mode="cedar",
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
    gate_mode: &str,
) -> PyResult<Vec<(f64, f64, f64, i64)>> {
    let shape = image.shape();
    let (h, w) = (shape[0], shape[1]);
    if shape.len() != 2 {
        return Err(PyValueError::new_err("image must be 2-D"));
    }

    let bg = match bg_mode.to_ascii_lowercase().as_str() {
        "row_percentile" | "rowpercentile" | "percentile" | "default" => BgMode::RowPercentile,
        "line_median" | "linemedian" | "row_median" => BgMode::LineMedian,
        other => {
            return Err(PyValueError::new_err(format!(
                "bg_mode must be 'row_percentile' or 'line_median', got '{other}'"
            )))
        }
    };
    let gate = parse_gate_mode(gate_mode)?;

    // Copy the image into a Rust-owned buffer so we can drop the GIL.
    // ~0.7 MB on a 0.73 MP frame; ~1 ms memcpy on the Zero 2W. The win is that
    // every other Python thread can run during the multi-millisecond compute.
    let owned: Vec<u8> = image
        .as_slice()
        .map_err(|_| PyValueError::new_err(
            "image must be C-contiguous uint8; use np.ascontiguousarray",
        ))?
        .to_vec();

    let stars = py.allow_threads(|| {
        let pool = get_pool();
        pool.install(|| match bin {
            1 => {
                let nz = noise.unwrap_or_else(|| estimate_noise(&owned, w, h));
                Ok(detect(
                    &owned, w, h, &owned, w, h, 1, sigma, nz, use_neon, bg, max_axis_ratio, gate,
                ))
            }
            2 => {
                let (b, wb, hb) = bin2x2_mean(&owned, w, h);
                let nz = noise.unwrap_or_else(|| estimate_noise(&b, wb, hb));
                let result = if centroid_full_res {
                    detect(
                        &b, wb, hb, &owned, w, h, 2, sigma, nz, use_neon, bg, max_axis_ratio, gate,
                    )
                } else {
                    let in_binned = detect(
                        &b, wb, hb, &b, wb, hb, 1, sigma, nz, use_neon, bg, max_axis_ratio, gate,
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
    gate_mode="cedar",
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
    gate_mode: &str,
) -> PyResult<Vec<(f64, f64, f64, i64)>> {
    let shape = image.shape();
    let (h, w) = (shape[0], shape[1]);
    if shape.len() != 2 {
        return Err(PyValueError::new_err("image must be 2-D"));
    }
    let gate = parse_gate_mode(gate_mode)?;

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
        .map_err(|_| PyValueError::new_err(
            "image must be C-contiguous uint8; use np.ascontiguousarray",
        ))?
        .to_vec();
    let owned_rof: Vec<u8> = rof_slice.to_vec();

    let stars = py.allow_threads(|| {
        let pool = get_pool();
        pool.install(|| match bin {
            1 => Ok(detect_cached(
                &owned, w, h, &owned, w, h, 1, sigma, noise, use_neon,
                &owned_rof, max_axis_ratio, gate,
            )),
            2 => {
                let (b, wb, hb) = bin2x2_mean(&owned, w, h);
                let result = if centroid_full_res {
                    detect_cached(
                        &b, wb, hb, &owned, w, h, 2, sigma, noise, use_neon,
                        &owned_rof, max_axis_ratio, gate,
                    )
                } else {
                    let in_binned = detect_cached(
                        &b, wb, hb, &b, wb, hb, 1, sigma, noise, use_neon,
                        &owned_rof, max_axis_ratio, gate,
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
    gate_mode: GateMode,
) -> Vec<Star> {
    if det_w < 7 || det_h < 7 {
        return Vec::new();
    }
    let sn2 = ((2.0 * sigma * noise + 0.5) as i32).max(2);
    let sn3 = ((3.0 * sigma * noise + 0.5) as i32).max(3);
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
                    det_img, det_w, y0, y1, sn2, sn3, use_neon,
                    Some(row_floors), gate_mode, mf_thresh,
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
                &cands, blob, det_img, det_w, det_h, cent_img, cent_w, cent_h,
                scale, noise, sigma, max_size, max_axis_ratio,
            )
        })
        .collect();
    stars.sort_by(|a, b| b.brightness.partial_cmp(&a.brightness).unwrap_or(Ordering::Equal));
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
