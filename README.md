# sycamore-extract

Fast star centroid extraction for plate solving on Raspberry Pi Zero 2 W class
hardware. A native Python extension (PyO3 + maturin), tuned for the Cortex-A53
and intended for use as the extractor stage of a finder pipeline whose solver
is olive-solve or tetra3rs.

The internal Python package name is `star_detect`. The repository is named
`sycamore-extract`.

## Status

Experimental. Tested against a small set of frames from a Raspberry Pi HQ
Camera (IMX477) at 0.73 MP. The architectural design is settled; the
specific gate algorithm, background-cache cadence, and other parameters
are still being measured.

## Design intent

This is a **finder-only extractor**. It is not astrometry.

- Centroids only need to be accurate to within a pixel or two of the FoV
  center after plate solving. Sub-pixel precision is a non-goal.
- Speed under contention matters more than peak speed. The finder runs
  alongside libcamera capture, an LX200 server, a web UI, and an IMU reader
  on a 4-core Pi; the extractor is bounded to 2 rayon threads by default so
  the other workers stay responsive.
- The GIL is released during compute so other Python threads run.
- A background cache (`examples/bg_cache.py`) maintains a temporally
  averaged background model in a worker thread, so per-frame detection
  consumes pre-computed state instead of re-estimating noise and row
  offsets every frame. See `ARCHITECTURE.md`.

## 1-D gate: matched filter

The 7-pixel "is this pixel a star candidate" gate is the heart of the
detector. This branch uses a standard signal-detection construction: a
discrete Gaussian kernel (sigma=1.5, calibrated against representative
HQ Camera + finder lens frames) convolved against the 7-pixel window, with
the response thresholded against `sigma * noise * ||k||_2`. Mathematically
the optimal linear detector for a known PSF in additive Gaussian noise
(North 1943, Turin 1960, Van Trees 1968).

The matched filter is somewhat conservative on real-sky frames — its
threshold derivation assumes Gaussian white noise, but real frames have
correlated noise structure that the filter is sensitive to. If you're
seeing fewer stars detected than expected, lower `sigma` by 1-2 from
the conventional default. See `ARCHITECTURE.md` for the calibration
history.

## About this branch

This is the `matched-filter-only` branch (v0.7.x). The `main` branch
(v0.6.x) ships a selectable two-gate design: cedar-detect-derived
heuristic plus matched filter. This branch ships only the matched
filter, with no cedar-derived code. See `CHANGELOG.md` and
`ARCHITECTURE.md` for the rationale.

## Attribution

This crate is Apache-2.0 licensed. The matched-filter detector was
independently implemented from textbook signal-detection theory. The
overall pipeline structure (prefilter → 1-D gate → blob assembly →
2-D gate → centroid) was informed by study of cedar-detect; that
influence is acknowledged in `NOTICE`. No cedar-detect source code is
retained in this branch.

## Building

The Rust extension cross-builds for the Pi from an x86 host (faster) or
builds natively on the Pi (slower). See `scripts/build_cross.sh` and
`scripts/build_native.sh`.

```
# On x86 host: cross-build a wheel for the Pi (Python 3.13 venv)
PYVER=3.13 ./scripts/build_cross.sh

# Copy wheel to Pi, rename to remove maturin's conservative glibc tag:
scp target/wheels/star_detect-*-cp313-*aarch64*.whl pi:
ssh pi
mv star_detect-*-manylinux_*_aarch64.whl star_detect-...-linux_aarch64.whl
pip install ~/star_detect-...-linux_aarch64.whl
```

Building natively on the Zero 2 W takes 15-25 minutes; you'll likely need
a 2 GB swapfile for the LTO link step.

## Use from Python

```python
import star_detect

# Bounded thread pool. Call before first detect_stars().
star_detect.set_num_threads(2)

# Per-frame detection.
stars = star_detect.detect_stars(
    image_u8,                  # 2-D C-contiguous numpy uint8 (H, W)
    sigma=8.0,
    bin=2,                     # 1=full-res, 2=2x2-binned detection
    centroid_full_res=True,    # if bin=2, centroid on full-res image
    bg_mode="row_percentile",  # or "line_median"
    max_axis_ratio=4.0,
    use_neon=False,
)
# -> [(x, y, brightness, peak), ...] brightest-first

# Cached detection (steady-state, background pre-computed elsewhere).
stars = star_detect.detect_stars_with_cache(
    image_u8, row_offsets_u8, noise=2.5,
    sigma=8.0, bin=2, max_axis_ratio=4.0,
)

# Helper exposed for the background worker.
medians = star_detect.compute_row_medians_py(image_u8)
```

## Benchmarks

- `tests/bench.py` — single-extractor timing across a directory of frames.
- `tests/bench_pipeline.py` — comparison against olive-solve's `FastExtractor`
  if installed.
- `tests/backend_speed_test.py` — end-to-end backend comparison
  (extract + solve + LX200 formatting).

## Files

- `src/lib.rs` — extension. All algorithms here.
- `Cargo.toml`, `.cargo/config.toml` — Rust build config; pinned to Cortex-A53.
- `examples/finder_loop.py` — control-flow sketch for a full finder.
- `examples/bg_cache.py` — background-cache state machine and worker.
- `tests/` — benchmarks (see above).
- `scripts/` — build helpers.
- `CLAUDE.md` — context for Claude Code sessions.
- `ARCHITECTURE.md` — long-form design rationale.
- `NOTICE` — third-party attributions.
- `CHANGELOG.md` — release-by-release change history.

## Releases

Every tagged version (`vX.Y.Z`) triggers GitHub Actions to cross-build
aarch64 wheels for Python 3.11, 3.12, and 3.13 and attach them to a
GitHub Release. Pi-side install is then just `pip install <url>` — no
Rust toolchain needed on the Pi, no compile time.

## Integration with diofinder

This is the extractor half of an electronic finder. The companion solver is
[olive-solve](https://github.com/mconsidine/olive-solve), used downstream
by [diofinder](https://github.com/mconsidine/diofinder)'s `olive` branch.

The intended integration pattern: diofinder's `install.sh` pulls a
pre-built wheel from this repo's GitHub Releases and installs it into the
device's Python venv. No compile on the Pi.

```bash
# In diofinder's install.sh, for the Pi Zero 2 W aarch64 / Python 3.13 case:
SYCAMORE_VERSION="v0.6.2"
pip install "https://github.com/mconsidine/sycamore-extract/releases/download/${SYCAMORE_VERSION}/star_detect-${SYCAMORE_VERSION#v}-cp313-cp313-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"
```

Pinning a specific version (rather than tracking `main`) means diofinder's
behavior is reproducible — bumping the sycamore-extract version is an
explicit, audit-trail-leaving step in diofinder.

For development, override the version with a local editable install:

```bash
git clone https://github.com/mconsidine/sycamore-extract ~/sycamore-extract
cd ~/sycamore-extract
pip install -e .
```

That gives you a path-mounted install where edits to `src/lib.rs` only need
a `maturin develop --release` to take effect, without rebuilding a wheel.

