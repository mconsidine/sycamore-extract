# Changelog

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
