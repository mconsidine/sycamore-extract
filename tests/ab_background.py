#!/usr/bin/env python3
"""
A/B background-compensation harness for star_detect.

Runs the same saved frame(s) through several background / binning / gate
configurations and reports, per configuration:

    - star count
    - detection latency (min / p50 / p95 / max ms over --runs)
    - centroid agreement vs the chosen baseline (matched count + mean dx px)

Every knob can be toggled from the command line so configurations can be
compared head-to-head ("A/B") on real saved frames:

    --bg-modes   line_median,row_percentile     (per-frame background floor)
    --bins       1,2                             (full-res vs 2x2-binned)
    --gates      matched_filter,cedar            (1-D gate algorithm)
    --cache                                      (also test the temporal cache)

The temporal cache configuration mimics the production background worker: it
median-stacks ALL provided frames into one background model (per-row median +
MAD noise) and runs detect_stars_with_cache against each frame. Give it a
burst of consecutive frames from the same pointing for a realistic result.

Examples
--------
    # Default sweep (line_median vs row_percentile, bin=1) on a directory:
    python3 tests/ab_background.py /var/lib/efinder/captures/

    # Full matrix incl. bin=2 and the temporal cache, vs a row_percentile
    # baseline, 50 timing runs each:
    python3 tests/ab_background.py test1.png test2.png test3.png \\
        --bg-modes line_median,row_percentile --bins 1,2 --cache \\
        --runs 50 --baseline row_percentile/bin1

    # Sweep sigma to see how each background mode trades count vs noise:
    python3 tests/ab_background.py frames/ --sigma-sweep
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

try:
    from PIL import Image
except ImportError:
    print("Need pillow: pip install pillow", file=sys.stderr)
    sys.exit(1)

import star_detect


# ── frame / timing helpers ──────────────────────────────────────────────────
def load_gray_u8(path):
    img = Image.open(path).convert("L")
    return np.ascontiguousarray(np.asarray(img, dtype=np.uint8))


def expand_paths(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
                out.extend(sorted(glob.glob(os.path.join(p, ext))))
        else:
            out.extend(sorted(glob.glob(p)))
    return out


def time_calls(fn, runs):
    fn()  # warm-up (not timed)
    ts = []
    result = None
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        result = fn()
        ts.append((time.perf_counter_ns() - t0) / 1e6)
    ts.sort()
    return ts[0], ts[len(ts) // 2], ts[min(len(ts) - 1, int(len(ts) * 0.95))], ts[-1], result


def centroid_match(a, b, tol_px=1.0):
    """Greedy nearest-neighbour match; returns (matched, mean_offset_px)."""
    if not a or not b:
        return 0, float("nan")
    A = np.array([(x, y) for (x, y, *_) in a])
    B = np.array([(x, y) for (x, y, *_) in b])
    used, offs = set(), []
    for ax, ay in A:
        d2 = (B[:, 0] - ax) ** 2 + (B[:, 1] - ay) ** 2
        for j in np.argsort(d2):
            if j in used:
                continue
            if d2[j] <= tol_px * tol_px:
                used.add(int(j))
                offs.append(float(np.sqrt(d2[j])))
            break
    if not offs:
        return 0, float("nan")
    return len(offs), sum(offs) / len(offs)


# ── temporal background model (mirrors efinder/bg_cache.py) ──────────────────
def build_cache_model(frames, bin):
    """Median-stack frames -> (row_offsets uint8, noise) at detection res."""
    time_med = np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)
    h, w = time_med.shape
    if bin == 2:
        tm = time_med[: (h // 2) * 2, : (w // 2) * 2]
        time_med = tm.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3)).astype(np.uint8)
    hd, wd = time_med.shape
    row_offsets = np.ascontiguousarray(
        star_detect.compute_row_medians_py(time_med), dtype=np.uint8)
    patch = time_med[hd // 3:2 * hd // 3, wd // 3:2 * wd // 3].astype(np.float32).ravel()
    mad = np.median(np.abs(patch - np.median(patch)))
    return row_offsets, max(0.5, 1.4826 * float(mad))


# ── configuration model ──────────────────────────────────────────────────────
class Config:
    def __init__(self, name, kind, bin, sigma, max_axis_ratio, gate, bg_mode=None):
        self.name = name
        self.kind = kind          # "frame" or "cache"
        self.bin = bin
        self.sigma = sigma
        self.max_axis_ratio = max_axis_ratio
        self.gate = gate
        self.bg_mode = bg_mode    # frame kind only
        self._model_cache = {}    # bin -> (row_offsets, noise) for cache kind

    def caller(self, img, all_frames):
        if self.kind == "frame":
            def call():
                return star_detect.detect_stars(
                    img, sigma=self.sigma, bin=self.bin,
                    centroid_full_res=True, bg_mode=self.bg_mode,
                    gate_mode=self.gate, max_axis_ratio=self.max_axis_ratio)
            return call
        # cache kind: build (once) a model from the whole frame set
        if self.bin not in self._model_cache:
            self._model_cache[self.bin] = build_cache_model(all_frames, self.bin)
        row_offsets, noise = self._model_cache[self.bin]

        def call():
            return star_detect.detect_stars_with_cache(
                img, row_offsets, noise, sigma=self.sigma, bin=self.bin,
                centroid_full_res=True, gate_mode=self.gate,
                max_axis_ratio=self.max_axis_ratio)
        return call


def build_configs(args):
    bg_modes = [m.strip() for m in args.bg_modes.split(",") if m.strip()]
    bins = [int(b) for b in args.bins.split(",") if b.strip()]
    gates = [g.strip() for g in args.gates.split(",") if g.strip()]
    mar = float("inf") if args.max_axis_ratio <= 1.0 else args.max_axis_ratio
    cfgs = []
    for gate in gates:
        for bin in bins:
            for bg in bg_modes:
                gtag = "" if gate == "matched_filter" else f"/{gate}"
                cfgs.append(Config(
                    f"{bg}/bin{bin}{gtag}", "frame", bin, args.sigma, mar, gate, bg))
            if args.cache:
                gtag = "" if gate == "matched_filter" else f"/{gate}"
                cfgs.append(Config(
                    f"cache/bin{bin}{gtag}", "cache", bin, args.sigma, mar, gate))
    return cfgs


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="Saved frame(s) or director(ies).")
    ap.add_argument("--bg-modes", default="line_median,row_percentile",
                    help="Comma list of per-frame bg modes (default both).")
    ap.add_argument("--bins", default="1", help="Comma list of bins, e.g. 1,2.")
    ap.add_argument("--gates", default="matched_filter",
                    help="Comma list of gate modes (matched_filter,cedar).")
    ap.add_argument("--cache", action="store_true",
                    help="Also test the temporal cache (built from all frames).")
    ap.add_argument("--sigma", type=float, default=8.0)
    ap.add_argument("--max-axis-ratio", type=float, default=0.0,
                    help="Trail rejection; <=1 disables (default off).")
    ap.add_argument("--runs", type=int, default=20, help="Timing reps per frame.")
    ap.add_argument("--threads", type=int, default=2,
                    help="star_detect worker threads (default 2, the Pi setting).")
    ap.add_argument("--baseline", default=None,
                    help="Config name to compare centroids against "
                         "(default: first config).")
    ap.add_argument("--sigma-sweep", action="store_true",
                    help="Sweep sigma 4-12 per config (star count + p50 only).")
    args = ap.parse_args()

    try:
        star_detect.set_num_threads(args.threads)
    except Exception:
        pass  # already initialised in this interpreter

    paths = expand_paths(args.paths)
    if not paths:
        print("No images found.", file=sys.stderr)
        sys.exit(2)
    frames = [load_gray_u8(p) for p in paths]
    # Cache modelling assumes a common shape; warn if mixed.
    shapes = {f.shape for f in frames}
    if len(shapes) > 1 and args.cache:
        print(f"WARN: mixed frame shapes {shapes}; cache built from first shape only",
              file=sys.stderr)

    cfgs = build_configs(args)
    print(f"frames={len(frames)}  runs={args.runs}  sigma={args.sigma}  "
          f"threads={args.threads}  configs={len(cfgs)}")

    if args.sigma_sweep:
        _sigma_sweep(cfgs, frames, paths)
        return

    baseline = args.baseline or cfgs[0].name
    print(f"baseline (centroid agreement) = {baseline}\n")

    for path, img in zip(paths, frames):
        same_shape = [f for f in frames if f.shape == img.shape]
        print(f"== {os.path.basename(path)}  {img.shape[1]}x{img.shape[0]}  "
              f"peak={int(img.max())}  mean={img.mean():.1f} ==")
        print(f"  {'config':<26}{'stars':>6}{'min':>8}{'p50':>8}{'p95':>8}"
              f"{'max':>8}{'matched':>9}{'dx_px':>9}")
        base_stars = None
        for c in cfgs:
            mn, p50, p95, mx, stars = time_calls(c.caller(img, same_shape), args.runs)
            if c.name == baseline:
                base_stars = stars
            matched, dx = ("", "")
            if base_stars is not None and c.name != baseline:
                m, off = centroid_match(stars, base_stars, tol_px=1.0)
                matched, dx = str(m), (f"{off:.3f}" if m else "n/a")
            print(f"  {c.name:<26}{len(stars):>6}{mn:>8.2f}{p50:>8.2f}"
                  f"{p95:>8.2f}{mx:>8.2f}{matched:>9}{dx:>9}")
        print()


def _sigma_sweep(cfgs, frames, paths):
    sigmas = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0]
    for path, img in zip(paths, frames):
        same_shape = [f for f in frames if f.shape == img.shape]
        print(f"\n== {os.path.basename(path)} ==  (star count / p50 ms per sigma)")
        print(f"  {'config':<26}" + "".join(f"{s:>10.0f}" for s in sigmas))
        for c in cfgs:
            cells = []
            for s in sigmas:
                c.sigma = s
                c._model_cache.clear()
                _, p50, _, _, stars = time_calls(c.caller(img, same_shape), max(5, 8))
                cells.append(f"{len(stars)}/{p50:.0f}")
            print(f"  {c.name:<26}" + "".join(f"{cell:>10}" for cell in cells))


if __name__ == "__main__":
    main()
