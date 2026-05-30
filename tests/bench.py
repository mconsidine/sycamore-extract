#!/usr/bin/env python3
"""
Benchmark star_detect against cedar-detect (if installed) on the same frames.

Usage:
    python3 bench.py path/to/frame.png [more.png ...]
    python3 bench.py path/to/dir_of_pngs/
    python3 bench.py --runs 100 --bin 2 frame.png

Reports per-image:
    n stars, p50/p95/max latency (ms) over --runs iterations,
    and (if cedar-detect-py is installed) the same for cedar
    plus a centroid agreement diff (how close the centroids match).
"""
import argparse
import glob
import os
import statistics
import sys
import time

import numpy as np

try:
    from PIL import Image
except ImportError:
    print("Need pillow: pip install pillow", file=sys.stderr)
    sys.exit(1)

import star_detect

try:
    # Steven Rosenthal also publishes Python bindings; package name on PyPI is
    # `cedar-detect-py`. Import is best-effort.
    import cedar_detect_py as cedar
    HAVE_CEDAR = True
except Exception:
    HAVE_CEDAR = False


def load_gray_u8(path):
    img = Image.open(path).convert("L")
    return np.ascontiguousarray(np.asarray(img, dtype=np.uint8))


def time_calls(fn, runs):
    # Warmup once; record `runs` calls.
    fn()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        result = fn()
        ts.append((time.perf_counter_ns() - t0) / 1e6)  # ms
    ts.sort()
    p50 = ts[len(ts) // 2]
    p95 = ts[int(len(ts) * 0.95)]
    return ts[0], p50, p95, max(ts), result


def centroid_match(a, b, tol_px=1.0):
    """
    Greedy nearest-neighbor match between two centroid lists.
    Returns (matched, mean_offset_px) where matched <= min(len(a), len(b)).
    """
    if not a or not b:
        return 0, float("nan")
    A = np.array([(x, y) for (x, y, *_) in a])
    B = np.array([(x, y) for (x, y, *_) in b])
    used_b = set()
    offsets = []
    for ax, ay in A:
        d2 = (B[:, 0] - ax) ** 2 + (B[:, 1] - ay) ** 2
        order = np.argsort(d2)
        for j in order:
            if j in used_b:
                continue
            if d2[j] <= tol_px * tol_px:
                used_b.add(int(j))
                offsets.append(float(np.sqrt(d2[j])))
            break
    if not offsets:
        return 0, float("nan")
    return len(offsets), sum(offsets) / len(offsets)


def expand_paths(args_paths):
    out = []
    for p in args_paths:
        if os.path.isdir(p):
            out.extend(sorted(glob.glob(os.path.join(p, "*.png"))))
            out.extend(sorted(glob.glob(os.path.join(p, "*.jpg"))))
            out.extend(sorted(glob.glob(os.path.join(p, "*.tif"))))
        else:
            out.extend(sorted(glob.glob(p)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="Image files or directories.")
    ap.add_argument("--runs", type=int, default=20, help="Iterations per image.")
    ap.add_argument("--sigma", type=float, default=8.0)
    ap.add_argument("--bin", type=int, default=1, choices=[1, 2])
    ap.add_argument("--use-neon", action="store_true",
                    help="Force explicit NEON path in star_detect (if built with it).")
    args = ap.parse_args()

    paths = expand_paths(args.paths)
    if not paths:
        print("No images found.", file=sys.stderr)
        sys.exit(2)

    kwargs = dict(sigma=args.sigma, bin=args.bin)
    if args.use_neon:
        kwargs["use_neon"] = True  # supported once step 3 is built in

    print(f"runs/image={args.runs}  sigma={args.sigma}  bin={args.bin}  "
          f"use_neon={args.use_neon}  cedar={'yes' if HAVE_CEDAR else 'no'}")
    print()
    header = ("file", "MP", "stars", "min", "p50", "p95", "max")
    if HAVE_CEDAR:
        header += ("cedar p50", "cedar p95", "matched", "mean_dx_px")
    print("\t".join(header))

    for path in paths:
        img = load_gray_u8(path)
        h, w = img.shape
        mp = (w * h) / 1e6

        def call_us():
            return star_detect.detect_stars(img, **kwargs)

        mn, p50, p95, mx, stars_us = time_calls(call_us, args.runs)

        row = [os.path.basename(path), f"{mp:.2f}", str(len(stars_us)),
               f"{mn:.2f}", f"{p50:.2f}", f"{p95:.2f}", f"{mx:.2f}"]

        if HAVE_CEDAR:
            # cedar-detect-py API differs slightly between releases; this is
            # the common form. Adapt if your installed version differs.
            def call_cedar():
                noise = cedar.estimate_noise_from_image(img)
                stars, *_ = cedar.get_stars_from_image(
                    img, noise, args.sigma, False,
                    binning=args.bin if args.bin > 1 else 1,
                    detect_hot_pixels=False,
                    return_binned_image=False,
                )
                return [(s.centroid_x, s.centroid_y, s.brightness, s.peak_value)
                        for s in stars]

            _, c_p50, c_p95, _, stars_cd = time_calls(call_cedar, args.runs)
            matched, mean_off = centroid_match(stars_us, stars_cd, tol_px=1.0)
            row += [f"{c_p50:.2f}", f"{c_p95:.2f}", str(matched),
                    f"{mean_off:.3f}" if matched else "n/a"]

        print("\t".join(row))


if __name__ == "__main__":
    main()
