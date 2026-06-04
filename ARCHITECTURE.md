# ARCHITECTURE.md — design rationale

This is the long-form companion to `CLAUDE.md`. It records *why* each design
decision was made and what alternatives were considered. If a future change
proposal seems to relitigate one of these, read the relevant section first.

## About this branch: matched-filter-only

This `matched-filter-only` branch (v0.7.0+) ships only the matched-filter
1-D gate. The cedar-detect-derived heuristic gate, which was selectable as
an option in v0.6.x on `main`, has been removed entirely.

The motivation is forward-looking risk management around cedar-detect's
algorithmic lineage, rather than any technical shortcoming. The matched
filter was independently derived from classical signal-detection theory
(North/Turin/Van Trees) and contains no cedar source code. With the cedar
heuristic gone, this branch establishes algorithmic independence.

The pipeline structure (prefilter → 1-D gate → blob assembly → 2-D gate
→ centroid) was informed by study of cedar-detect — that's true and worth
acknowledging; see `NOTICE`. But pipeline architecture is not the same
thing as source code lineage. This branch's *code* is independent.

The history of why both gates existed, what was learned by comparing them,
and what was tried to close the sensitivity gap is preserved below in the
"Rejected alternatives" section. That work happened on `main` and informs
why the matched filter is what it is today; deleting it from history would
be dishonest about the design process.

Practical consequence: detection is somewhat more conservative than v0.6.x
with the cedar default. Users seeing fewer stars detected than expected
should lower `sigma` by 1-2 from the conventional sigma=8 default.

## The two-call division (extractor + solver)

The finder pipeline is split into two separate Python calls:
`star_detect.detect_stars(image) -> centroids` and
`tetra3_py.Tetra3.solve_from_centroids(centroids) -> pose`.

This was chosen over an integrated single call because:

1. The two stages have very different optimization profiles. Detection is
   memory-bandwidth bound on a Zero 2 W; solving is compute bound on a tuned
   hash database. They benefit from different code, different parallelism,
   different sensor inputs.
2. Olive-solve's Rust solver is sub-millisecond on the Zero 2 W with an
   FOV-tuned database. There's no plausible win from reimplementing it.
3. The split lets us swap one half without touching the other. If a better
   solver appears, we use it; if a faster detector appears, we use it.

## Why Rust, not C

Both compile through LLVM to roughly identical assembly for the inner loops.
The choice was made on the periphery:

- **PyO3 + maturin** gives zero-copy numpy access and direct `pip install`,
  with no manual `setup.py` or ABI dance.
- **Rayon** makes the bounded-thread-pool model a 5-line change. Doing the
  same correctly in C requires hand-rolled pthread code with manual aliasing
  reasoning.
- **The borrow checker** caught two real bugs during development of the
  parallel candidate collection.
- **`std::arch::aarch64`** gives explicit NEON intrinsics if the autovectorizer
  underperforms; in practice it does fine.

## The "analytic threading" background model

This is the most consequential architectural decision. Rather than estimate
background per frame (what cedar-detect and olive-solve do), a worker thread
maintains a cached background model from temporally-stacked frames, and
detection consumes the cache directly.

The intuition: the per-frame budget should buy reactive throughput. Idle
moments should buy *better state* for the next reactive frame. Per-frame
estimation forces a tradeoff — better background means slower detection —
and that tradeoff disappears when the expensive work happens in idle time.

What this gives us that per-frame estimation can't:

- **Temporal averaging.** N stacked frames cut noise in the estimate by √N.
  Per-frame estimation only ever sees one frame's noise.
- **Free hot-pixel detection.** A pixel that's bright across all N stacked
  frames is hot; one that comes and goes is a star or transient. The worker
  builds a hot-pixel mask alongside the background as a side effect.
- **Adaptive cadence.** The worker yields when the system is busy; it works
  harder when the system is idle. The UX-preserving behavior is structural.

What it costs:

- **State invalidation.** Slewing makes the cache wrong. We need a clear
  "we moved enough that the cached background is stale" signal — currently
  IMU angular distance from the pose at cache build. Below threshold:
  STEADY. Above: SLEWING; fall back to per-frame `bg_mode="line_median"`.
- **Two code paths in detection.** Cached and uncached. Both need to stay
  correct. Both are tested by `bench.py`.
- **Memory for the stack.** ~5–6 MB for 8 frames at 0.73 MP. Fits comfortably
  in 512 MB but not free.

### Rejected: rebuild the background continuously, no state machine

This would be simpler but defeats the entire point. Continuous rebuild = work
per frame = no different from olive-solve. The "build only when steady"
property is the lever.

### Rejected: full 2-D BlockMedian background instead of row offsets

The current cached model is per-row. A 2-D background map (e.g. 32×24 grid,
bilinearly interpolated) would handle horizontal as well as vertical
gradients. Reasons we don't yet:

- For an IMX477 with a typical finder lens, the vignetting is primarily
  radial but the worst component is still vertical (the LineMedian model
  catches most of it).
- BlockMedian doubles the worker complexity (grid + interpolation) and the
  detection hot path needs to interpolate two grid neighbors per pixel
  instead of indexing one array.
- It's a clear next step if data shows it's needed. The cache API can grow
  to include a 2-D map alongside the per-row offsets without breaking the
  existing call shape.

### Rejected: pure dark-frame subtraction

A dark-frame (lens-cap-on) calibration handles hot pixels, amp glow, and
sensor bias, but not vignetting or sky gradients. It's complementary to
the cached background, not a replacement. The right move is to do both:
subtract a dark frame at capture, then run cached-background detection on
the result. Not yet wired in; should be straightforward when needed.

## Why ROI tracking, not always full-frame

After a confident solve, we know where every catalog star *should* be in the
frame. Re-detecting them across the full frame is wasted work — detection
inside small windows around predicted positions is roughly an order of
magnitude faster and just as accurate.

This was chosen over:

- **Detect every frame at full res** — too slow, dominates the budget.
- **Detect every Nth frame, propagate IMU between** — works but the IMU has
  ~0.1°/min drift, so we'd still need periodic full-frame re-anchoring.
  ROI re-detection at every cycle keeps the centroid measurements fresh.

Caveat: ROI mode falls back to full-frame on detection failure (e.g. clouds
moved across the FoV). The state machine catches this via confidence decay.

## Threading model

Five threads. All do one thing each.

- **Camera**: writes the latest frame to a single-slot last-wins buffer.
  Never queues; we don't care about stale frames.
- **IMU**: integrates attitude at ~100 Hz, posts to `bg_cache` and triggers
  detection on drift.
- **Detect/solve**: pulls trigger events, reads the latest frame, dispatches
  to ROI or full-frame detection, calls the solver, updates pose.
- **SkySafari LX200 server**: pure I/O, answers from pose + IMU extrapolation.
- **Web server**: pure I/O.
- (**bg_cache worker**: maintains the background model in idle time.)

Detection thread is the only one that runs CPU-heavy work. By bounding
star_detect to 2 rayon threads, we guarantee 2 cores are always free for
the I/O threads regardless of frame complexity. This matters more than
shaving milliseconds off detection.

## Coordinate conventions

- `star_detect` returns `(x, y, ...)`. `(0.5, 0.5)` is the center of the
  top-left pixel.
- `tetra3` family expects `(y, x)`. We swap at the boundary.
- IMU quaternions are `(w, x, y, z)` scalar-first, body-frame, unit norm.
- Pose `(ra, dec, roll)` is radians.

## Two 1-D gate algorithms

The 7-pixel "is this pixel a star candidate" gate is the heart of the
detector. Two variants are implemented and live as runtime-selectable
modes.

**`cedar`** is the heuristic from cedar-detect: integer comparisons,
ordered for branch selectivity, local-max within a 3-pixel window with
uniform-background sanity checks. Battle-tested across years of real-sky
frames in the PiFinder ecosystem.

**`matched_filter`** is the standard signal-detection construction:
convolve the 7-pixel window with a discrete Gaussian kernel (sigma=1.5,
FWHM ~3.5 px), threshold against `sigma * noise * kernel_L2_norm`.
Mathematically the optimal *linear* detector for a known-shape pulse in
additive Gaussian white noise (North 1943, Turin 1960, Van Trees 1968).
Independently derived; not inherited from cedar.

### Why both exist

The empirical finding from this project: at the same nominal sigma value,
the two gates produce *different effective false-positive rates*.
Matched filter is more conservative — at sigma=8 it detects roughly the
same star count as cedar at sigma=9-10 on representative HQ Camera
frames. This held up through three rounds of investigation:

1. Initial run showed cedar finding 8 stars matched filter rejected,
   matched filter finding 0 cedar rejected. Strong one-directional
   disagreement.
2. Sigma sweep showed the gap roughly constant in count (6-19 stars)
   across sigma 4-12, ruling out a simple threshold-calibration offset.
3. Visual inspection at sigma=8 showed all 8 disagreeing stars were
   real, well-sampled PSFs. The matched-filter kernel was widened from
   sigma=1.0 to sigma=1.5 to better match the actual PSF, recovering
   some — but not all — of the missing stars.
4. Visual inspection at sigma=11 showed the residual 8 stars were
   bright, well-formed real stars indistinguishable from those both
   gates accept. No flat tops, no saturation, no obvious feature.

This is a structural property of the two detectors. The matched
filter's theoretical optimality assumes Gaussian white noise; real
frames have correlated structure (residual gradients, sensor patterns)
that adds variance to the matched-filter response specifically. Cedar's
local-max heuristic isn't sensitive to that structure.

Neither gate is "wrong." They are different points in detector design
space, internally consistent but not interchangeable. Cedar is the
default because it's been validated on more sky; matched filter exists
because it's a credible alternative for users who want a principled
false-positive rate on Gaussian-noise inputs, or who want a clean break
from cedar's algorithmic lineage.

### Rejected: tune matched_filter until it matches cedar

Done implicitly by the kernel widening (sigma 1.0 → 1.5). It closed
some of the gap but not all. Further tuning would mean fighting the
inherent characteristic of the matched filter rather than benefiting
from it. If a user wants cedar's count, they should use cedar.

### Rejected: matched_filter on wide-stencil-bg-subtracted input

We hypothesized the matched filter's conservatism came from
spatially-correlated background structure (vignetting + sky gradients)
that violated the filter's white-noise assumption. The fix would be:
subtract a wide-stencil local mean before applying the matched filter,
restoring the noise-whitening assumption.

Tested via `tests/prototype_mf_bg.py` (since removed from the
canonical tree; the prototype was a Python wrapper, no Rust changes).
Result was unambiguous and disconfirming:

- On the cleanest test frame, plain matched filter matched cedar's
  star count (100/100). After bg subtraction, the count dropped to
  52, missing 49 stars cedar found. The "fix" made the detector
  meaningfully worse.
- On frames with mild structure (test2, test3), bg subtraction made
  no detectable difference at any sigma from 4 to 12. mf_n and
  mfbg_n were identical; mfbg_miss exactly equaled mf_miss.

The conclusion: the matched filter's conservatism is *not* primarily
caused by correlated background structure. Whatever causes the gap
(probably noise-distribution non-Gaussianity at the per-pixel level,
or shot-noise/read-noise interaction at low signal levels) is
something bg subtraction doesn't address.

The wider lesson worth recording: adding redundant correction steps
to a robust estimator pipeline doesn't monotonically improve it.
The cedar gate's row-floor + 2-D gate logic already handles slow
gradients well; adding a wide-stencil pre-subtraction just adds
variance to its threshold estimators without removing anything the
gate didn't already handle.

If a future maintainer is tempted to revisit this: the next experiment
should be a pixel-value distribution analysis of a star-free patch
(check whether the noise is actually Gaussian), not more bg-subtraction
work. But this work has diminishing returns relative to the existing
detector quality, and was deliberately shelved.



Measured on a Pi Zero 2 W with the IMX477 finder camera at 0.73 MP:

- **Per-frame detection, bin=2, row_percentile (uncached)**: ≤ 6 ms p50.
- **Per-frame detection, bin=2, line_median**: ≤ 9 ms p50.
- **Cached detection, `detect_stars_with_cache`**: ≤ 4 ms p50.
- **ROI detection (12 windows × 48 px)**: ≤ 1.5 ms total.
- **Cold solve (olive-solve, FOV-tuned DB)**: < 50 ms p50.
- **Steady-state pose update cadence**: 5 Hz minimum, 10 Hz target.

A regression in any of these without a robustness justification is a bug.

## What we know we don't know

- **NEON autovectorization on the A53.** The `build_native.sh` script dumps
  asm and counts `ld1 ... .16b` instructions inside `scan_band`. If autovec
  fails on a future toolchain version, switch to `use_neon=True`.
- **Sensor-specific row noise.** IMX296mono has well-documented per-row
  offset noise; IMX477 has it less severely but it's real. LineMedian
  handles it; the row_percentile path may be more sensitive to it than
  we've measured.
- **Worst-case frame at moon-near-FoV.** None of the test frames in
  `/var/lib/efinder/` capture this. Real-sky validation needed.
