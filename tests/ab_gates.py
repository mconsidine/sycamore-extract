#!/usr/bin/env python3
"""
A/B compare the two 1-D gates in star_detect on the same frames:
  - cedar:           cedar-detect's 7-pixel heuristic (the default)
  - matched_filter:  experimental Gaussian-kernel matched filter

What this measures:
  - Star counts per mode.
  - Per-mode timing distribution (p50, p95, max).
  - Centroid agreement (how many stars matched within tol, mean offset).
  - Cedar-only and MF-only stars (the *disagreement* — usually the most
    interesting column; reveals what each gate is picking up that the other
    isn't).
  - Brightness-rank correlation across matched stars (do they agree on which
    are bright?).

Expected output pattern:
  cedar_only will be NON-ZERO across all sigmas (typically 5-20 stars). This
  is not a bug or a tuning issue — it reflects a structural difference in
  sensitivity. The matched filter is more conservative at any given sigma
  because its threshold derivation assumes pure Gaussian noise, while real
  frames have correlated structure that perturbs the matched-filter response.
  Cedar's local-max heuristic isn't sensitive to that structure.

  mf_only is typically near zero. Matched filter almost never finds a star
  cedar misses.

  rank_corr should be > 0.99 — when the gates agree on a star, they centroid
  and rank-by-brightness it essentially identically.

Usage:
    python3 tests/ab_gates.py path/to/frame.png [more.png ...]
    python3 tests/ab_gates.py path/to/dir/
    python3 tests/ab_gates.py --runs 50 --bin 2 --bg-mode line_median /var/lib/efinder/

Recommended sweep across sigma values to see where the gates diverge:
    python3 tests/ab_gates.py --sigma-sweep /var/lib/efinder/test1.png

Use tests/inspect_disagreement.py to see *which* stars each gate is missing.
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


def load_gray_u8(path):
    img = Image.open(path).convert("L")
    return np.ascontiguousarray(np.asarray(img, dtype=np.uint8))


def time_calls(fn, runs):
    """Returns (min, p50, p95, max) ms, and the last result."""
    fn()  # warmup
    ts = []
    last = None
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        last = fn()
        ts.append((time.perf_counter_ns() - t0) / 1e6)
    ts.sort()
    n = len(ts)
    return (ts[0], ts[n // 2], ts[int(n * 0.95)], ts[-1]), last


def centroid_diff(a, b, tol_px=1.5):
    """
    Compare two centroid lists. Each is [(x, y, brightness, peak), ...].

    Returns dict with:
      n_a, n_b:        total counts
      matched:         star pairs within tol_px (greedy nearest neighbor)
      only_a, only_b:  stars unique to each list
      mean_offset_px:  mean distance between matched centroids
      rank_corr:       Spearman rank correlation of brightness on matched stars
                       (1.0 = perfect agreement on ordering, 0 = no relationship)
    """
    if not a and not b:
        return dict(n_a=0, n_b=0, matched=0, only_a=0, only_b=0,
                    mean_offset_px=float("nan"), rank_corr=float("nan"))
    A = np.array([(x, y) for (x, y, *_) in a]) if a else np.zeros((0, 2))
    B = np.array([(x, y) for (x, y, *_) in b]) if b else np.zeros((0, 2))
    bA = np.array([br for (_x, _y, br, _p) in a]) if a else np.zeros((0,))
    bB = np.array([br for (_x, _y, br, _p) in b]) if b else np.zeros((0,))

    used_b = set()
    matches = []  # (i_a, j_b, dist_px)
    for i, (ax, ay) in enumerate(A):
        if len(B) == 0:
            break
        d2 = (B[:, 0] - ax) ** 2 + (B[:, 1] - ay) ** 2
        for j in np.argsort(d2):
            j = int(j)
            if j in used_b:
                continue
            if d2[j] <= tol_px * tol_px:
                used_b.add(j)
                matches.append((i, j, float(np.sqrt(d2[j]))))
            break

    matched = len(matches)
    only_a = len(A) - matched
    only_b = len(B) - matched
    mean_offset_px = (
        float(np.mean([m[2] for m in matches])) if matched else float("nan")
    )

    # Spearman rank correlation on the matched stars' brightnesses.
    if matched >= 3:
        ia = np.array([m[0] for m in matches])
        ib = np.array([m[1] for m in matches])
        ra = np.argsort(np.argsort(-bA[ia]))   # rank 0=brightest
        rb = np.argsort(np.argsort(-bB[ib]))
        # Pearson on ranks == Spearman.
        if np.std(ra) > 0 and np.std(rb) > 0:
            rank_corr = float(np.corrcoef(ra, rb)[0, 1])
        else:
            rank_corr = float("nan")
    else:
        rank_corr = float("nan")

    return dict(
        n_a=len(A), n_b=len(B), matched=matched,
        only_a=only_a, only_b=only_b,
        mean_offset_px=mean_offset_px, rank_corr=rank_corr,
    )


def expand_paths(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            for ext in ("png", "jpg", "jpeg", "tif", "tiff"):
                out.extend(sorted(glob.glob(os.path.join(p, f"*.{ext}"))))
        else:
            out.extend(sorted(glob.glob(p)))
    return out


def run_one(img, sigma, common_kwargs, runs):
    """Run both gates on one image; return a row dict."""
    cedar_kwargs = dict(common_kwargs, sigma=sigma, gate_mode="cedar")
    mf_kwargs = dict(common_kwargs, sigma=sigma, gate_mode="matched_filter")

    def call_cedar():
        return star_detect.detect_stars(img, **cedar_kwargs)

    def call_mf():
        return star_detect.detect_stars(img, **mf_kwargs)

    (_, c_p50, c_p95, _), cedar_stars = time_calls(call_cedar, runs)
    (_, m_p50, m_p95, _), mf_stars = time_calls(call_mf, runs)
    diff = centroid_diff(cedar_stars, mf_stars, tol_px=1.5)
    return {
        "sigma": sigma,
        "c_n": diff["n_a"], "m_n": diff["n_b"],
        "c_p50": c_p50, "c_p95": c_p95,
        "m_p50": m_p50, "m_p95": m_p95,
        "match": diff["matched"],
        "cedar_only": diff["only_a"], "mf_only": diff["only_b"],
        "dx_px": diff["mean_offset_px"],
        "rank_corr": diff["rank_corr"],
    }


def fmt_row(d, cols):
    out = []
    for k in cols:
        v = d[k]
        if isinstance(v, float):
            if v != v:  # NaN
                out.append("n/a")
            elif k in ("dx_px", "rank_corr"):
                out.append(f"{v:.3f}")
            else:
                out.append(f"{v:.2f}")
        else:
            out.append(str(v))
    return "\t".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--sigma", type=float, default=8.0)
    ap.add_argument("--bin", type=int, default=1, choices=[1, 2])
    ap.add_argument("--use-neon", action="store_true")
    ap.add_argument("--bg-mode", default="row_percentile",
                    choices=["row_percentile", "line_median"])
    ap.add_argument("--max-axis-ratio", type=float, default=float("inf"))
    ap.add_argument("--sigma-sweep", action="store_true",
                    help="For each image, sweep sigma 4..12 in steps of 1.")
    args = ap.parse_args()

    paths = expand_paths(args.paths)
    if not paths:
        print("No images found.", file=sys.stderr)
        sys.exit(2)

    common = dict(
        bin=args.bin, use_neon=args.use_neon, bg_mode=args.bg_mode,
        max_axis_ratio=args.max_axis_ratio,
    )

    sigmas = list(range(4, 13)) if args.sigma_sweep else [args.sigma]

    print(f"runs/image={args.runs}  bin={args.bin}  bg_mode={args.bg_mode}  "
          f"max_axis_ratio={args.max_axis_ratio}")
    print()

    cols = ["sigma", "c_n", "m_n", "c_p50", "c_p95", "m_p50", "m_p95",
            "match", "cedar_only", "mf_only", "dx_px", "rank_corr"]
    header_line = "\t".join(cols)

    for path in paths:
        img = load_gray_u8(path)
        h, w = img.shape
        mp = (w * h) / 1e6
        print(f"### {os.path.basename(path)}  ({h}x{w}, {mp:.2f} MP)")
        print(header_line)
        for sigma in sigmas:
            row = run_one(img, sigma, common, args.runs)
            print(fmt_row(row, cols))
        print()


if __name__ == "__main__":
    main()
