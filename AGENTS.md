# AGENTS.md — AI-agent contributor guide for sycamore-extract

Model-agnostic guide for any AI coding agent (or human) picking up this repo
cold. Read in this order:

1. This file — workflow, release mechanics, consumer contract, backlog.
2. `CLAUDE.md` — the design-intent handoff (load-bearing architectural
   decisions and things deliberately NOT done). Despite the name it applies
   to any reader. **Ask before reverting anything it lists as deliberate.**
3. `ARCHITECTURE.md` — longer-form design notes and rejected alternatives.
4. The ecosystem-level guide: `mconsidine/diofinder` → `AGENTS.md` (how this
   wheel is consumed, released, and diagnosed in the field).

---

## 1. Identity and role

Repo name **sycamore-extract**, Python module **`star_detect`** — the
mismatch is intentional (documented in CLAUDE.md). It is a Rust (PyO3, abi3)
extension that extracts star centroids from 8-bit grayscale frames on a
Raspberry Pi Zero 2W. It is **only an extractor**: no solver, no FITS, no
16-bit path (all deliberate — see CLAUDE.md).

The production consumer is diofinder's `bg_cache.py` + `solver_proc.py`.
Default branch: `main`; work via branch → PR → squash-merge.

## 2. The consumer contract (do not break)

diofinder installs whatever wheel the **latest GitHub release** carries and
capability-probes features instead of version-checking. Rules that follow:

1. **Capability flags are the API for new features.** Native functions do
   not support `inspect.signature`, so a new keyword argument is invisible
   to consumers unless you also export a module-level flag
   (`HAS_BG_IMAGE = True` is the precedent; `hasattr` on functions covers
   new entry points). Ship the flag in the same release as the feature.
2. **Coordinate/output conventions are frozen**: `detect_stars` returns
   `[(x, y, brightness, peak), ...]`, brightest-first, `(0.5, 0.5)` = center
   of pixel (0,0). diofinder swaps to (row, col) downstream; changing the
   order or tuple shape breaks every consumer silently.
3. **`detect_stars_with_cache` model args are exactly-one-of**:
   `row_offsets` (positional-friendly) / `block_offsets` / `bg_image`
   (keyword-only). Validation errors, not silent acceptance, on misuse.
   `bg_image` must be the BINNED (h//bin, w//bin) uint8 median stack.
4. **u8 frames only**; `bin` ∈ {1, 2, 4}; unsupported/legacy parameters must
   raise `TypeError` (the removed `gate_mode` is the precedent), not be
   silently ignored.
5. Removing/renaming anything public requires a coordinated diofinder
   release; additive keyword-only evolution otherwise.

Version history (what a fielded wheel might be):
0.9 matched-filter-only + `top_hat`; 0.10 `block_percentile`/`column`/
`row_column`; 0.11 `uniform_mean` + `noise_mode="global_rms"`; 0.12
`kernel_sigma`, `local_noise`, cached `block_offsets`, full 2-D moment trail
rejection; **0.13 (current)** `bg_image` per-pixel cached background +
`HAS_BG_IMAGE`.

## 3. Build, test, performance gates

```bash
# Local wheel (host arch; also what diofinder's offline replay uses):
python3 -m pip wheel . --no-deps -w wheels/

# Cross-build for the Pi (Bookworm = Python 3.13):
PYVER=3.13 ./scripts/build_cross.sh     # see CLAUDE.md for the wheel-tag rename gotcha

cargo test
```

Hot-path changes MUST run the benchmarks in `tests/bench.py` before/after on
the reference frames (see CLAUDE.md "When changing the hot path"):
**p50 under 6 ms at bin=2 on the Zero 2W** on test1/test2 is the regression
gate; `tests/bench_pipeline.py` checks centroid agreement (< 1 px mean)
against olive-solve's extractor; `tests/ab_background.py` A/Bs background
modes. Update `CHANGELOG.md` per release.

## 4. Releasing

Version lives in `Cargo.toml`. The `build` workflow builds the aarch64 abi3
wheel and publishes the GitHub release. Two triggers:

- tag push `vX.Y.Z`, or
- **workflow_dispatch** with input `version: vX.Y.Z` — the reliable path
  from environments whose git proxy rejects tag pushes (known failure mode).
  A `-suffix` version builds without releasing.

**A release is immediately fleet-visible**: diofinder image builds and OTA
updates pull the latest release (pin via diofinder's `SYCAMORE_TAG`).
Validate risky changes against real frames first — diofinder debug bundles
replayed offline are the best corpus.

## 5. Backlog

*(empty as of v0.13.0 — planned items shipped: block cache 0.12, bg_image
0.13)*

Optional future item, take only with diofinder coordination:
**ROI detection API** — diofinder's tracking mode currently slices numpy
windows itself and runs the detector per window because there is no native
ROI entry point. A `detect_stars_roi(image, windows, ...)` that runs the
matched filter over a window list in one call (one GIL release, one thread
fan-out) would cut tracking-mode overhead. Additive + capability flag
(`HAS_ROI`), per §2.

When you complete or add a task, update this section in the same PR — this
file is the durable task queue across sessions and agent frameworks.

## 6. Handoff protocol

Tests + benchmarks green before commit; PR → squash-merge to `main`;
CHANGELOG updated; if the public API grew, name the new capability flag in
the PR body and release the wheel BEFORE the diofinder release that uses it.
