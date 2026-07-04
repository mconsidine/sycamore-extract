# Changelog

## 0.14.0

### Added
- **ROI detection API: `detect_stars_roi(image, windows, sigma=8.0,
  kernel_sigma=1.5, local_noise=True, max_axis_ratio=inf)`** — the
  tracking-mode fast path. Takes the FULL frame plus a 2-D numpy int32/int64
  window list of shape (N, 4) (`x0, y0, x1, y1`, exclusive) and detects stars
  in all N windows with ONE call: the frame is copied once, the GIL is
  released once, and the windows fan out across the existing bounded rayon
  pool (previously the downstream diofinder tracking mode sliced numpy
  windows in Python and paid one GIL round-trip per window). Windows are
  defensively re-clamped to the image bounds; windows degenerate after
  clamping (< 8 px on a side, inverted, or empty) are skipped. Per-window
  pipeline is always bin=1: per-row median background floor (`line_median` —
  the inline cache-line-sampled row percentile degenerates at ~48 px widths),
  whole-window MAD noise (the patch-grid estimator needs ≥ 64 px sides), then
  the standard matched-filter gate / blob / 2-D gate chain, reusing the
  full-frame internals. Returns at most ONE star per window (the brightest
  surviving detection), `[(x, y, brightness, peak), ...]` brightest-first in
  FULL-FRAME coordinates with the usual (0.5, 0.5)-center-of-pixel
  convention.
- **`HAS_ROI = True` capability flag** (module-level, next to
  `HAS_BG_IMAGE`) — probe `getattr(star_detect, "HAS_ROI", False)` before
  calling, per the consumer contract (native functions defeat
  `inspect.signature`).
- Off-device validation in `tests/test_roi.py` (pytest, synthetic frames):
  full-frame coordinate mapping within 1 px of truth, empty windows return
  nothing, out-of-bounds windows clamp without panicking, int32/int64 window
  arrays both accepted, one-star-per-window cap.

## 0.13.0

*(entry backfilled — this release shipped without a changelog entry)*

### Added
- **`bg_image` cached background** for `detect_stars_with_cache`
  (keyword-only, mutually exclusive with `row_offsets` / `block_offsets`):
  the full per-pixel background at the binned detection resolution
  (typically the temporal median stack itself), subtracted directly —
  removes gradients, vignetting AND per-pixel fixed-pattern structure in one
  pass. Capability flag `HAS_BG_IMAGE = True`.

## 0.12.0

Robustness across seeing conditions (bloated PSFs, 2-D gradients, trails) for
the downstream diofinder finder. Five additive features; **default-parameter
behavior is unchanged and bit-identical to 0.11.x** (verified by comparing the
0.11.2 and 0.12.0 wheels on synthetic frames: row_percentile / line_median /
block_percentile outputs match exactly under a canonical sort). New costs are
gated behind non-default parameters only.

### Added
- **Runtime-tunable matched-filter kernel** via `kernel_sigma` (float, default
  1.5, valid range 1.0–4.0, ValueError outside) on `detect_stars` and
  `detect_stars_with_cache`. The integer Gaussian kernel is generated at runtime
  (half-width `ceil(2*sigma)` clamped to [3,7] → 7/9/11/13/15 taps; mean-
  subtracted; center coeff rounds to 60; sum forced to 0 by a symmetric residual
  adjustment on the outermost taps). `kernel_sigma=1.5` reproduces the historical
  hardcoded kernel `[-50,-15,35,60,35,-15,-50]` exactly (unit-tested), and its
  threshold L2 norm is pinned to 107.0 so default results don't shift; wider
  kernels use their true ||k||₂. Widen the kernel for bloated PSFs from poor
  seeing or heavy defocus.
- **Full 2-D second-moment trail rejection.** When `max_axis_ratio` is finite,
  `gate_2d` now computes the off-diagonal moment `m2_xy` (one extra 2-D pass
  over the per-blob box) and rejects on `sqrt(λ_max/λ_min)` of the full 2×2
  covariance. This catches *diagonally*-elongated satellite/aircraft trails the
  old separable var_x/var_y-only approximation could not see. Skipped entirely
  when `max_axis_ratio` is infinite (the default), so the hot path is unchanged.
- **Perimeter-derived local noise inflation** (concept inspired by cedar-detect,
  Apache-2.0; independent implementation, no code copied). The per-blob
  acceptance test now uses `effective_noise = max(global_noise, ring_spread)`,
  where `ring_spread` is the stddev of the blob's perimeter ring. This raises
  the detection bar in cluttered/noisy neighborhoods (moon halo, clouds,
  foreground glow) and suppresses false positives there; on clean sky
  `ring_spread < global noise` so behavior is unchanged. New `local_noise` bool
  (default True) on both entry points disables it for A/B testing. Applied at
  the 2-D stage only — the 1-D scan is untouched.
- **`bin=4`** on `detect_stars` and `detect_stars_with_cache`: two cascaded 2×2
  mean bins (the escape hatch for badly defocused / oversampled stars). With
  `centroid_full_res=True`, centroids are taken on the full-res image with
  coordinates mapped ×4; noise/background estimation runs on the binned image.
- **2-D block-grid temporal cache.** New `compute_block_medians_py(image,
  block_size=32) -> (grid_h, grid_w) uint8` returns the per-tile median grid
  (reusing the `block_percentile` internals). `detect_stars_with_cache` now
  accepts the cache as either `row_offsets` (1-D, positional, backward-
  compatible) **or** a keyword-only `block_offsets` (2-D grid) + `block_size`;
  exactly one must be supplied (ValueError otherwise). The block path subtracts
  the bilinearly-interpolated 2-D background then runs the standard scan with the
  inline row-percentile floor — making 2-D background correction compose with
  the temporal cache for the first time (the "rejected: full 2-D BlockMedian
  cache" item in ARCHITECTURE.md is now implemented). `examples/bg_cache.py`
  gained a `model="block"` variant.

### Notes
- The block-cache path produces output bit-identical to per-frame
  `bg_mode="block_percentile"` on a static frame with the same grid + noise
  (unit-tested at the Rust level; verified end-to-end in the Python smoke test).

## 0.11.2

- Switch the Python extension to PyO3 abi3 (`abi3-py38`): one wheel now
  covers every CPython >= 3.8 instead of per-interpreter cp311/cp312/cp313
  builds. No functional changes.
- Release workflow builds the single abi3 wheel once (the per-Python matrix
  produced identical wheels) and hyphenated versions (e.g. `v0.12.0-test`)
  publish as prereleases, excluded from "latest".

Note: the wheel attached to the v0.11.2 GitHub release was built before the
version field was bumped and is internally versioned 0.11.1; the code is
identical.

## 0.11.1

Performance pass for the Pi Zero 2W hot paths (output is byte-identical to
0.11.0 — verified by parity unit tests against reference implementations):

- `white_tophat`: per-thread scratch reuse in the van Herk passes (was 3 Vec
  allocations per row per pass), middle transposes fused away (2 transposes per
  call instead of 4), and the final subtraction parallelised. Expected ~2x
  faster top_hat at bin=2.
- `transpose`: tiled 32x32 (cache-friendly on Cortex-A53) instead of naive
  strided.
- `block_percentile`: per-tile median via 256-bin counting (no per-tile
  allocation or sort), selecting the same element as the previous sort.
- `detect_stars` / `detect_stars_with_cache`: sigma is clamped to >= 0.5 — a
  sigma near 0 made every noise pixel a candidate (multi-hundred-ms frames and
  a flood of false stars).
- New `#[cfg(test)]` parity suite (`cargo test`): van Herk vs naive window
  extreme, scratch-reuse cleanliness, tiled transpose vs naive + roundtrip,
  fused top-hat vs the pre-fusion 4-transpose pipeline, histogram tile median
  vs the sorted reference.
- CLAUDE.md: refreshed the thread-pool decision (downstream diofinder now runs
  3 threads on dedicated cores) and removed a stale duplicate gate-mode
  decision left from a pre-0.9.0 merge.


Notable changes per release. Format follows [Keep a Changelog](https://keepachangelog.com/),
versions follow [Semantic Versioning](https://semver.org/).

## [0.8.0] — 2026-06-01

### Changed
- **`gate_mode` default changed from `"cedar"` to `"matched_filter"`** in both
  `detect_stars` and `detect_stars_with_cache`. Callers that relied on the
  implicit default will now use the matched filter; pass `gate_mode="cedar"`
  explicitly to restore the previous behaviour.
- `"default"` alias in `parse_gate_mode` remapped from Cedar → MatchedFilter to
  stay consistent with the new default.
- Version jump from 0.6.x to 0.8.0 avoids collision with 0.7.x tags on the
  `matched-filter-only` branch.

### Notes
- Both gates remain present and fully supported; this is a default-only change.
- Empirical sensitivity difference is unchanged: matched_filter at sigma=8
  detects roughly the same star count as cedar at sigma=9-10. Callers
  switching from cedar may want to lower sigma by 1-2 to maintain star count.
  See `ARCHITECTURE.md` and tests/ab_gates.py for the calibration evidence.
## [0.7.0] — 2026-05-31 — matched-filter-only branch

### Why
This branch establishes sycamore-extract's algorithmic independence from
cedar-detect. The matched filter is independently derived from classical
signal-detection theory (North/Turin/Van Trees) and was implemented from
that lineage, not from cedar source code. With the cedar-derived gate
removed, no cedar source code is retained.

### Practical impact
- The default sigma is unchanged at 8.0. This is the conventional astronomy
  threshold; lowering the default would mask the real behavior change.
  Document your sigma choice if you want reproducibility.

### Removed
- `tests/ab_gates.py` (compared the two gates; only one gate now).
- `tests/inspect_disagreement.py` (visualized gate disagreements; same).
- `tests/prototype_mf_bg.py` (dead-end exploration; documented in ARCHITECTURE.md).
- `tests/spectral_diagnostic.py` (same; had known methodology issues).

### Retained (and still useful)
- `tests/bench.py` — single-extractor timing.
- `tests/bench_pipeline.py` — sycamore vs olive-solve's extractor.
- `tests/backend_speed_test.py` — end-to-end backend comparison.

### Note
- Versioned as 0.7.0 (not 1.0.0) to signal the breaking change in the API
  without yet claiming validated field stability. 1.0.0 is reserved for
  after under-the-sky validation confirms the matched-filter-only behavior
  is good across real conditions (twilight, moon-near-FoV, star-rich
  fields). Until then this branch is "ready for field testing" rather than
  "stable for production."

## [0.6.2] — 2026-05-30

### Documentation
- Documented the empirical finding that `matched_filter` is structurally
  more conservative than `cedar` at the same nominal sigma. On HQ Camera
  frames, MF at sigma=8 ≈ cedar at sigma=9-10 in star count. The threshold
  derivation assumes pure Gaussian noise; real frames have correlated
  structure that perturbs the MF response. This is a feature of the
  detector, not a bug.
- Updated `tests/ab_gates.py` docstring with expected output shape (non-zero
  `cedar_only`, near-zero `mf_only`, `rank_corr > 0.99`).
- Added "Two 1-D gate algorithms" section to `ARCHITECTURE.md` capturing
  the four-step investigation that led to the v0.6.1 kernel widening
  and the final sensitivity finding.
- Documented the bg-subtraction "third detector" prototype as a rejected
  alternative in `ARCHITECTURE.md`. The hypothesis (matched filter's
  conservatism comes from spatial structure that wide-stencil subtraction
  would whiten) was tested via Python prototype and disconfirmed: bg
  subtraction either did nothing or made detection worse, depending on
  the frame. The remaining gap between cedar and matched filter lives in
  the pixel-level noise distribution, not in spatial structure.

## [0.6.1] — 2026-05-30

### Changed
- Matched-filter kernel widened from sigma=1.0 to sigma=1.5
  (`[-50, -15, 35, 60, 35, -15, -50]`) to match the actual PSF FWHM
  seen on representative Raspberry Pi HQ Camera frames. The tighter
  kernel was systematically rejecting real, well-sampled stars.
- Matched-filter threshold constant updated from 109 to 107 to reflect
  the new kernel's L2 norm (107.24).

## [0.6.0] — 2026-05-30

### Added
- `gate_mode` parameter on `detect_stars` and `detect_stars_with_cache`.
  Two options: `"cedar"` (default; the cedar-detect heuristic) and
  `"matched_filter"` (experimental Gaussian matched filter).
- `tests/ab_gates.py` — A/B comparison harness with sigma sweep.
- `tests/inspect_disagreement.py` — visualize stars where the two gates
  disagree, annotated PNG output for eyeball inspection.

## [0.5.0] — 2026-05-30

### Added
- `detect_stars_with_cache(image, row_offsets, noise, ...)` — steady-state
  entry that consumes pre-computed background state and skips per-frame
  estimation. Designed for use with the background-cache worker in
  `examples/bg_cache.py`.
- `compute_row_medians_py(image)` — Python-callable per-row median helper,
  parallel-histogram backed.
- `examples/bg_cache.py` — `BackgroundCache` class with worker thread,
  state machine (WARMING_UP / STEADY / SLEWING), and atomic publish.
- `CLAUDE.md`, `ARCHITECTURE.md` — design context for Claude Code and
  future maintainers.

### Changed
- `examples/finder_loop.py` updated to route detection through the
  background cache, with slew-driven invalidation.

## [0.4.0] — 2026-05-29

### Added
- `bg_mode` parameter on `detect_stars`. `"row_percentile"` (default; v0.3
  behavior) and `"line_median"` (per-row true median via 256-bin histogram,
  parallel across rows). Equivalent to olive-solve's `FastBgSubMode::LineMedian`.
- `max_axis_ratio` parameter for rejecting trails and bloomed stars,
  computed cheaply from existing projections.
- `tests/bench_pipeline.py` rewritten against olive-solve's real Python API
  (`tetra3_py.Tetra3(db).get_centroids_from_image_fast`).

## [0.3.0] — 2026-05-29

### Added
- Bounded rayon thread pool (default 2 threads, configurable via
  `STAR_DETECT_THREADS` env var or `set_num_threads(n)`). On a 4-core Pi
  Zero 2 W with libcamera + IMU + SkySafari + web all competing,
  unbounded parallelism causes UX jitter. The 2-thread cap trades ~50%
  per-frame speed for keeping other workers responsive.
- GIL release during compute via `py.allow_threads`. Image is copied once
  (~1 ms) before release. Other Python threads run concurrently with
  detection.

### Fixed
- Replaced single-row noise estimator with 9-patch MAD median (more robust
  to vignetting, partial-black bars, and bright contamination).
- Replaced `row_min` over cache-line samples with row 25th-percentile.
  The min collapsed to zero on any row overlapping a black bar or dead
  column, flooding the prefilter; the percentile is robust to ~20% bad
  pixels per row.

## [0.2.0] — 2026-05-29

### Added
- Explicit NEON threshold prefilter via `use_neon=True` (uses
  `std::arch::aarch64::vcgeq_u8`). Alternative to LLVM autovectorization;
  autovec usually does fine on the A53.
- `bin=2` parameter for 2x2-binned detection. `centroid_full_res=True`
  centroids on the full-res image for sub-pixel precision (CedarDetect's
  design).

## [0.1.0] — 2026-05-29

Initial release. PyO3 + maturin extension wrapping a Rust port of the
cedar-detect detection pipeline: cache-line-sampled row floor, 7-pixel
1-D gate, candidate connected-component blob assembly, 2-D gate with
perimeter-derived local noise, background-subtracted separable
projections with parabolic peak interpolation.
