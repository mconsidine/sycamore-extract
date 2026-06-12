# CLAUDE.md — handoff for Claude Code

This file documents the design intent, constraints, and accumulated decisions
for **sycamore-extract**. Claude Code reads it on session start; treat
everything here as in-context guidance for any task in this repo.

The internal Python package name is `star_detect`; the repo is named
`sycamore-extract`. This is intentional — the name mismatch is a deliberate
choice, not an oversight.

## What this project is

`sycamore-extract` provides `star_detect`, a Rust extension module (PyO3)
that extracts star centroids from grayscale 8-bit camera frames, for use in
a Raspberry Pi Zero 2 W electronic finder. It is **only an extractor** —
plate solving is delegated to [olive-solve](https://github.com/mconsidine/olive-solve)
(specifically the `olive` branch of the diofinder repo, which uses
olive-solve's solver). The accuracy target is **"within a pixel or two of
the field-of-view center."** This is a finder, not astrometry. Speed and
robustness matter more than sub-pixel precision.

The downstream consumer is the diofinder repo's `olive` branch. This crate
is intended for evaluation alongside (not as a replacement for) olive-solve's
own `FastExtractor`. See `tests/bench_pipeline.py` for the head-to-head.

## Hardware and runtime context

- **Pi Zero 2 W:** 4× Cortex-A53 cores, 512 MB RAM, aarch64. Memory-bandwidth
  bound, not compute bound, for image-processing tasks. NEON SIMD available.
- **Sensor:** IMX477 (Pi HQ Camera). 8-bit grayscale processing path; the
  finder doesn't need the 12-bit range.
- **OS:** Pi OS Bookworm 64-bit, Python 3.13 in a venv at `~/.venvs/finder`.
- **Other processes on the box:** libcamera/picamera2 capture, IMU read,
  SkySafari LX200 TCP server, self-hosted web pages. There is **no display**;
  the finder reports pose via SkySafari and the web pages.

## Architectural decisions (load-bearing)

These were debated and chosen for reasons. Ask before reverting any of them.

1. **Bounded thread pool, not unlimited rayon.** Default 2 threads (configurable
   via `STAR_DETECT_THREADS` env var or `star_detect.set_num_threads(n)`).
   The cap exists so a detection burst can't starve other workers on a shared
   box. NOTE: the downstream diofinder daemon now pins its solver process to
   three dedicated cores (CPUs 1-3, with comms/webui on CPU 0) and calls
   `set_num_threads(3)` — on that deployment the 2-thread rationale no longer
   applies and 3 is correct. The library default stays 2 for unknown callers.

2. **GIL is released during detection.** The image is copied (~1 ms for 0.73
   MP) before `py.allow_threads(|| ...)`. This lets the SkySafari handler,
   IMU reader, and web threads run concurrently with detection. The copy is
   worth it.

3. **Detection is event-driven, not per-frame.** The finder loop captures at
   the camera's native rate but only triggers detection on "stale" (>500 ms
   since last solve), "moved" (IMU drift > threshold), or explicit user
   request. See `examples/finder_loop.py`.

4. **Tracking mode uses ROI windows, not full frames.** After a confident solve,
   subsequent detections use 48×48-pixel windows around predicted star
   positions. Full-frame detection is only for cold (lost-in-space) solves.

5. **Background estimation is "analytic threaded," not per-frame.** A worker
   thread maintains a temporally-averaged per-row background model. Detection
   in steady state uses the cached model via `detect_stars_with_cache(...)`;
   it falls back to per-frame `bg_mode="line_median"` during slew or warmup.
   See `examples/bg_cache.py`. This is genuinely a better architecture than
   per-frame estimation and was deliberately chosen.

6. **Olive-solve does plate solving.** `star_detect` will not get a solver
   added. The right division of labor is: `star_detect` produces centroids
   fast; olive-solve consumes them. Both link into the same Python process
   in the downstream diofinder app.

7. **Single 1-D gate: matched filter.** Standard signal-detection
   construction: convolve the window with a Gaussian-shaped kernel and threshold
   the response against `sigma * noise * ||k||_2`. Implementation is integer
   arithmetic; mean-zero kernel cancels DC, so no per-pixel bg subtraction
   is required. Since v0.12.0 the kernel is generated at runtime from
   `kernel_sigma` (default 1.5, range 1.0–4.0); the default reproduces the
   historical hardcoded `[-50,-15,35,60,35,-15,-50]` kernel exactly (and pins
   the threshold norm to 107.0), so default behavior is unchanged. Larger
   `kernel_sigma` widens the kernel (7→15 taps) for bloated PSFs.

   **Empirical behavior**: somewhat conservative on real-sky frames
   because the threshold derivation assumes pure Gaussian noise but
   real frames have correlated structure. Users on field data may need
   to lower sigma by 1-2 from the conventional sigma=8 default to detect
   faint stars near the noise floor. See ARCHITECTURE.md for the
   calibration history that led to this design choice.

   Since v0.9.0 the crate contains *only* the matched filter — the
   `gate_mode` parameter was removed (passing it raises TypeError).
   The v0.6.x two-gate history lives in git history if ever needed.

## Things deliberately not done

- **`uniform_mean` + `noise_mode="global_rms"` added in v0.11.0.** This is the
  exact tetra3 / olive-solve default pipeline: 2-D sliding-window mean background
  (summed-area table, O(1) per pixel) + global RMS noise estimate. Use
  `uniform_filter_size` (default 25) and `noise_mode="global_rms"`. The SAT
  building step is serial O(N) but fast; subtraction is parallelised by row.
- **BlockMedian is now `block_percentile`.** Added in v0.10.0 as
  `bg_mode="block_percentile"` with `bg_block_size` (default 32 for bin=2
  detection images). Uses bilinear interpolation of per-tile medians — same
  algorithm as tetra3rs's `estimate_local_background`. Also added
  `column_percentile` and `row_column_percentile` (separable 2-D removal) and
  parallelised the `transpose` helper used by `white_tophat`.
- **Full 2-D moment computation in `gate_2d` (added in v0.12.0).** The concrete
  failure case arrived: diagonal satellite/aircraft trails in a finder that now
  enables `max_axis_ratio`. The old separable var_x/var_y-only axis ratio is
  blind to diagonal elongation (a 45° streak can have var_x ≈ var_y yet a huge
  off-diagonal moment). When `max_axis_ratio` is finite, `gate_2d` now computes
  the off-diagonal moment `m2_xy` in one extra 2-D pass over the per-blob box and
  rejects on `sqrt(λ_max/λ_min)` of the full 2×2 covariance (closed-form
  eigenvalues, `cov2x2_axis_ratio2`). The pass is skipped when `max_axis_ratio`
  is infinite (the default), so the clean-sky hot path is unchanged.
- **No 16-bit pixel path.** The finder explicitly works in `u8`. Olive-solve
  has `u16`/`f32` paths if you ever need them; `star_detect` does not and
  shouldn't.
- **No solver, no database loading, no FITS handling.** Out of scope.

## File map

- `src/lib.rs` — the extension. All algorithms here.
- `Cargo.toml`, `.cargo/config.toml` — Rust build config; pinned to Cortex-A53.
- `scripts/build_native.sh` — build on the Pi directly.
- `scripts/build_cross.sh` — cross-build from x86 (default Python is 3.11;
  override with `PYVER=3.13` to match Pi OS Bookworm).
- `tests/bench.py` — single-extractor timing harness.
- `tests/ab_background.py` — A/B background compensation across bg_mode / bin /
  gate / temporal cache on saved frames (counts, timing, centroid agreement).
- `tests/bench_pipeline.py` — A/B against olive-solve's `FastExtractor`.
- `examples/finder_loop.py` — control-flow sketch for the full finder.
- `examples/bg_cache.py` — the background-cache state machine and worker.
- `ARCHITECTURE.md` — longer-form design notes and rejected alternatives.

## Public API (Python)

```python
import star_detect

# Bounded thread pool. Call before first detect_stars.
star_detect.set_num_threads(2)

# Standard per-frame detection.
stars = star_detect.detect_stars(
    image_u8,                  # 2-D C-contiguous numpy uint8 (H, W)
    sigma=8.0,                 # threshold in noise sigmas
    bin=2,                     # 1=full-res, 2 or 4 = 2x2 / 4x4-binned detection
    centroid_full_res=True,    # if bin>1, centroid on full-res image
    bg_mode="row_percentile",  # "line_median", "top_hat", "column_percentile",
                               # "row_column_percentile", "block_percentile",
                               # "uniform_mean"
    max_axis_ratio=4.0,        # reject trails (full 2-D moments); default inf
    use_neon=False,            # explicit NEON prefilter (autovec is usually fine)
    kernel_sigma=1.5,          # matched-filter kernel width, 1.0-4.0 (1.5=legacy)
    local_noise=True,          # inflate per-blob noise to ring spread (A/B: False)
)
# -> [(x, y, brightness, peak), ...] brightest-first, (0.5, 0.5)=center of pixel (0,0)

# Steady-state cached detection. Supply EXACTLY ONE of row_offsets / block_offsets.
stars = star_detect.detect_stars_with_cache(
    image_u8,
    row_offsets_u8,             # 1-D uint8, len == image height // bin (positional)
    noise=2.5,                  # precomputed sigma
    sigma=8.0,
    bin=2,                      # 1, 2, or 4
    max_axis_ratio=4.0,
    kernel_sigma=1.5,
    local_noise=True,
)
# 2-D cached background variant (keyword-only block_offsets, mutually exclusive
# with row_offsets):
stars = star_detect.detect_stars_with_cache(
    image_u8, None, 2.5, bin=2,
    block_offsets=grid_u8,      # 2-D uint8 (grid_h, grid_w) from compute_block_medians_py
    block_size=32,
)

# Background-worker helpers (parallel-histogram backed):
medians = star_detect.compute_row_medians_py(image_u8)        # 1-D per-row median
grid    = star_detect.compute_block_medians_py(image_u8, 32)  # 2-D per-tile medians
```

## How to build

Cross-build from the x86 dev box (faster):
```
PYVER=3.13 ./scripts/build_cross.sh
scp target/wheels/star_detect-*-cp313-*.whl pi:
# On the Pi:
mv star_detect-*-manylinux_*_aarch64.whl star_detect-...-linux_aarch64.whl
pip install ~/star_detect-...-linux_aarch64.whl
```

Native on the Pi (slow first compile, fine for incremental):
```
./scripts/build_native.sh
```

## Common gotchas (recorded from prior debugging)

- `cp311` wheel won't install in `cp313` venv. Match `PYVER` to the Pi's
  Python.
- `manylinux_2_34_aarch64` wheel tag is rejected by older pip; rename to
  `linux_aarch64` after cross-build (the binary works fine; the tag was a
  conservative glibc claim by maturin).
- `numpy::PyArrayMethods` and `PyUntypedArrayMethods` trait imports moved in
  numpy 0.22. Both are imported at the top of `src/lib.rs`.
- LTO + 512 MB RAM = OOM during link. Native builds may need 2 GB swap
  added via a swapfile (Bookworm doesn't ship `dphys-swapfile`):
  `sudo fallocate -l 2G /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile`.
- The `bg_cache` worker must not rebuild while `slewing` is true; the resulting
  model would be from blurred frames.

## When changing the hot path

1. Run `tests/bench.py` on `/var/lib/efinder/test{1,2,3}.png` before and
   after. test1/test2 are typical dark-sky frames; test3 is a synthetic
   partial-black-bar case used to catch regressions in the row-floor and
   noise estimators.
2. The p50 target on test1/test2 with `bin=2` is **under 6 ms** on the Zero
   2 W. Going above that without a robustness reason is a regression.
3. For accuracy regressions, run `tests/bench_pipeline.py` against
   olive-solve and check the `dx_px` centroid-agreement column; should stay
   well under 1 pixel mean offset.

## When in doubt

Prefer the question to the assumption. The history of this project includes
several "obvious" guesses about Python API shapes, file locations, and
sensor behavior that turned out wrong on contact with reality. If a tool
result is ambiguous, ask before patching past it.
