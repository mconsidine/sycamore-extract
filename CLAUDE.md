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
   On a 4-core Pi running camera + IMU + SkySafari + web + solver, claiming all
   4 cores during detection causes UX jitter. The 2-thread cap trades ~50%
   per-frame speed for keeping other workers responsive. Wall-clock detection
   is still fast (5–10 ms p50 at 0.73 MP).

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
   construction: convolve the 7-pixel window with a Gaussian-shaped kernel
   (sigma=1.5, calibrated against HQ Camera typical PSF) and threshold the
   response against `sigma * noise * ||k||_2`. Implementation is integer
   arithmetic; mean-zero kernel cancels DC, so no per-pixel bg subtraction
   is required.

   **Empirical behavior**: somewhat conservative on real-sky frames
   because the threshold derivation assumes pure Gaussian noise but
   real frames have correlated structure. Users on field data may need
   to lower sigma by 1-2 from the conventional sigma=8 default to detect
   faint stars near the noise floor. See ARCHITECTURE.md for the
   calibration history that led to this design choice.

   This branch contains *only* the matched filter. v0.6.x had a
   selectable two-gate design (cedar-detect-derived heuristic + matched
   filter) — that history is preserved on `main` for users who want it,
   but this branch establishes algorithmic independence from cedar-detect.

## Things deliberately not done

- **No BlockMedian background.** LineMedian + the cached background worker
  together cover the same ground at less per-frame cost. If you find a real
  case the cache + LineMedian can't handle, propose BlockMedian then.
- **No full 2-D moment computation in `gate_2d`.** The current `max_axis_ratio`
  filter derives a coarse axis ratio from separable projections; it can't see
  diagonally-elongated trails as cleanly as olive-solve's eigendecomposition.
  For a static-FOV finder this is acceptable. Don't change it without a
  concrete failure case.
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
    bin=2,                     # 1=full-res, 2=2x2-binned detection
    centroid_full_res=True,    # if bin=2, centroid on full-res image
    bg_mode="row_percentile",  # or "line_median"
    max_axis_ratio=4.0,        # reject trails; default inf
    use_neon=False,            # explicit NEON prefilter (autovec is usually fine)
)
# -> [(x, y, brightness, peak), ...] brightest-first, (0.5, 0.5)=center of pixel (0,0)

# Steady-state cached detection (requires pre-computed background).
stars = star_detect.detect_stars_with_cache(
    image_u8,
    row_offsets_u8,            # 1-D uint8, len == image height // bin
    noise=2.5,                 # precomputed sigma
    sigma=8.0,
    bin=2,
    max_axis_ratio=4.0,
)

# Helper for the background worker: per-row median, parallel-histogram backed.
medians = star_detect.compute_row_medians_py(image_u8)
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
