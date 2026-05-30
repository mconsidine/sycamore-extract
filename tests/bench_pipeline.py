#!/usr/bin/env python3
"""
Benchmark star_detect against olive-solve's FastExtractor on the same frames.

Uses the actual olive-solve Python API:
    tetra3_py.Tetra3(db_path).get_centroids_from_image_fast(img_u8, **kwargs)
which returns an (N, 2) ndarray of (y, x) centroids, brightest first.

Usage:
    python3 tests/bench_pipeline.py path/to/frames/
    python3 tests/bench_pipeline.py --runs 50 --bin 2 --bg-mode line_median frames/
    python3 tests/bench_pipeline.py --database db.npz --solve frames/

Notes:
- The olive-solve solver needs an .npz database (built offline on a workstation
  via cedar-solve's tetra3.generate_database). Pass --database PATH to enable
  the solver column. Pass --solve to actually time it on centroids from each
  extractor (otherwise we just print n centroids).
- For star_detect, --bg-mode can be 'row_percentile' (default) or 'line_median'.
- For olive-solve, --olive-bg-mode mirrors that: 'line_median', 'global_mean',
  'global_median', or 'none'.
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

# olive-solve's Python module is named `tetra3_py` (per its Cargo.toml).
HAVE_OLIVE = False
olive_import_err = None
try:
    import tetra3_py as olive
    HAVE_OLIVE = True
except Exception as e:
    olive_import_err = e


def load_gray_u8(path):
    img = Image.open(path).convert("L")
    return np.ascontiguousarray(np.asarray(img, dtype=np.uint8))


def time_calls(fn, runs):
    """Returns (min, p50, p95, max) in ms, and the last result."""
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


def sd_to_yx(stars):
    """star_detect returns [(x, y, brightness, peak), ...]; tetra3 wants (y, x)."""
    return [(y, x) for (x, y, _b, _p) in stars]


def olive_to_yx(centroids):
    """olive-solve returns an (N, 2) ndarray of (y, x). Normalize to list."""
    if centroids is None:
        return []
    if isinstance(centroids, np.ndarray) and centroids.ndim == 2 and centroids.shape[1] >= 2:
        return [(float(r[0]), float(r[1])) for r in centroids]
    out = []
    for row in centroids:
        try:
            out.append((float(row[0]), float(row[1])))
        except Exception:
            pass
    return out


def centroid_match(a_yx, b_yx, tol_px=1.5):
    """Greedy NN match between two centroid lists (both (y, x))."""
    if not a_yx or not b_yx:
        return 0, float("nan")
    A = np.asarray(a_yx, dtype=np.float64)
    B = np.asarray(b_yx, dtype=np.float64)
    used = set()
    offs = []
    for ay, ax in A:
        d2 = (B[:, 0] - ay) ** 2 + (B[:, 1] - ax) ** 2
        for j in np.argsort(d2):
            j = int(j)
            if j in used:
                continue
            if d2[j] <= tol_px * tol_px:
                used.add(j)
                offs.append(float(np.sqrt(d2[j])))
            break
    if not offs:
        return 0, float("nan")
    return len(offs), sum(offs) / len(offs)


def expand_paths(args_paths):
    out = []
    for p in args_paths:
        if os.path.isdir(p):
            for ext in ("png", "jpg", "jpeg", "tif", "tiff"):
                out.extend(sorted(glob.glob(os.path.join(p, f"*.{ext}"))))
        else:
            out.extend(sorted(glob.glob(p)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--sigma", type=float, default=8.0)
    ap.add_argument("--bin", type=int, default=1, choices=[1, 2])
    ap.add_argument("--use-neon", action="store_true")
    ap.add_argument("--bg-mode", default="row_percentile",
                    choices=["row_percentile", "line_median"],
                    help="star_detect background-floor strategy.")
    ap.add_argument("--max-axis-ratio", type=float, default=float("inf"),
                    help="star_detect: reject elongated blobs (trails). "
                         "Default Inf; recommend 3-5 for a finder.")
    ap.add_argument("--olive-bg-mode", default="line_median",
                    choices=["line_median", "global_mean", "global_median", "none"],
                    help="olive-solve FastBgSubMode (passed as bg_sub_mode).")
    ap.add_argument("--olive-sigma-mode", default="global_root_square",
                    choices=["global_root_square", "global_median_abs"])
    ap.add_argument("--database", default=None,
                    help="olive-solve .npz database path. Required for --solve "
                         "and required to construct the Tetra3 object even if "
                         "only the extractor is used (the constructor needs it).")
    ap.add_argument("--solve", action="store_true",
                    help="Also time the solver on centroids from each extractor.")
    args = ap.parse_args()

    paths = expand_paths(args.paths)
    if not paths:
        print("No images found.", file=sys.stderr)
        sys.exit(2)

    # Construct the olive Tetra3 instance if available + database given.
    olive_t3 = None
    if HAVE_OLIVE and args.database:
        try:
            olive_t3 = olive.Tetra3(args.database)
        except Exception as e:
            print(f"olive Tetra3 construction failed: {e}", file=sys.stderr)
            olive_t3 = None
    elif HAVE_OLIVE and not args.database:
        print("note: olive-solve is installed but no --database was given, so "
              "olive's extractor cannot be constructed (its Tetra3 ctor "
              "requires a db path). Pass --database to enable the A/B.",
              file=sys.stderr)
    elif not HAVE_OLIVE:
        print(f"note: olive-solve (tetra3_py) not importable: {olive_import_err}",
              file=sys.stderr)

    print(f"runs/image={args.runs}  sigma={args.sigma}  bin={args.bin}  "
          f"use_neon={args.use_neon}  sd_bg={args.bg_mode}  "
          f"olive={'yes' if olive_t3 else 'no'}  "
          f"olive_bg={args.olive_bg_mode}  solve={args.solve}")
    print()

    header = ["file", "MP", "sd_n", "sd_p50", "sd_p95"]
    if olive_t3 is not None:
        header += ["ol_n", "ol_p50", "ol_p95", "match", "dx_px"]
    if args.solve and olive_t3 is not None:
        header += ["solve_sd", "solve_ol"]
    print("\t".join(header))

    sd_kwargs = dict(
        sigma=args.sigma,
        bin=args.bin,
        use_neon=args.use_neon,
        bg_mode=args.bg_mode,
        max_axis_ratio=args.max_axis_ratio,
    )
    olive_kwargs = dict(
        sigma=float(args.sigma),
        bg_sub_mode=args.olive_bg_mode if args.olive_bg_mode != "none" else None,
        sigma_mode=args.olive_sigma_mode,
        downsample=args.bin if args.bin > 1 else None,
    )
    olive_kwargs = {k: v for k, v in olive_kwargs.items() if v is not None}

    for path in paths:
        img = load_gray_u8(path)
        h, w = img.shape
        mp = (w * h) / 1e6

        # ---- star_detect timing
        def call_sd():
            return star_detect.detect_stars(img, **sd_kwargs)

        (_, sd_p50, sd_p95, _), sd_stars = time_calls(call_sd, args.runs)
        row = [os.path.basename(path), f"{mp:.2f}", str(len(sd_stars)),
               f"{sd_p50:.2f}", f"{sd_p95:.2f}"]

        # ---- olive extractor timing
        ol_centroids = None
        if olive_t3 is not None:
            def call_ol():
                return olive_t3.get_centroids_from_image_fast(img, **olive_kwargs)

            try:
                (_, ol_p50, ol_p95, _), ol_centroids_raw = time_calls(call_ol, args.runs)
                ol_centroids = olive_to_yx(ol_centroids_raw)
                sd_yx = sd_to_yx(sd_stars)
                matched, mean_off = centroid_match(sd_yx, ol_centroids, tol_px=1.5)
                row += [str(len(ol_centroids)), f"{ol_p50:.2f}", f"{ol_p95:.2f}",
                        str(matched), f"{mean_off:.3f}" if matched else "n/a"]
            except Exception as e:
                row += ["err", "err", "err", "n/a", "n/a"]
                print(f"  olive extractor failed on {path}: {e}", file=sys.stderr)

        # ---- solver timing on centroids from each extractor
        if args.solve and olive_t3 is not None:
            runs = max(5, args.runs // 4)

            sd_yx_arr = np.array(sd_to_yx(sd_stars)[:30], dtype=np.float64)
            def call_solve_sd():
                if len(sd_yx_arr) < 4:
                    return None
                return olive_t3.solve_from_centroids(sd_yx_arr, (float(h), float(w)))

            try:
                (_, ssd_p50, _, _), _ = time_calls(call_solve_sd, runs)
                row.append(f"{ssd_p50:.2f}")
            except Exception as e:
                row.append("err")
                print(f"  solve(sd centroids) failed: {e}", file=sys.stderr)

            if ol_centroids and len(ol_centroids) >= 4:
                ol_yx_arr = np.array(ol_centroids[:30], dtype=np.float64)
                def call_solve_ol():
                    return olive_t3.solve_from_centroids(ol_yx_arr, (float(h), float(w)))

                try:
                    (_, sol_p50, _, _), _ = time_calls(call_solve_ol, runs)
                    row.append(f"{sol_p50:.2f}")
                except Exception as e:
                    row.append("err")
                    print(f"  solve(olive centroids) failed: {e}", file=sys.stderr)
            else:
                row.append("n/a")

        print("\t".join(row))


if __name__ == "__main__":
    main()
