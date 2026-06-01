#!/usr/bin/env python3
"""
End-to-end latency comparison of finder backends on static test frames.

What this measures
------------------
For each (backend, frame) combination:
  - extract_ms:    time from image_u8 to centroid list
  - solve_ms:      time from centroid list to (RA, Dec, roll)
  - lx200_ms:      time to format the LX200 response string (≈ 5 µs)
  - end_to_end_ms: extract + solve + lx200

What this does NOT measure
--------------------------
  - Camera capture / DMA
  - The decide-when-to-detect cadence logic
  - IMU extrapolation between solves
  - Actual socket I/O to SkySafari
  - Performance under network or other system contention

So treat the numbers as "compute pipeline latency", not "what the user
will feel". The latter requires under-the-sky testing.

Backends compared
-----------------
  olive       olive-solve's FastExtractor + olive-solve's solver
  sycamore    sycamore-extract (star_detect) + olive-solve's solver
  cedar       cedar-detect (if installed) + olive-solve's solver

Sycamore is timed in two modes: "cold" (first call on each frame, no
bg_cache) and "warm" (after 10 priming frames have populated the cache).
The warm mode is the realistic deployed behavior.

Methodology notes
-----------------
  - Each backend imports lazily inside the runner to avoid first-import
    cost leaking into the timed loop.
  - Each backend warms up with 5 throwaway calls before measurement.
  - Order: olive, sycamore, [cedar], olive_drift. If olive_drift's p50 is
    >20% higher than the first olive's p50, the test is invalid — most
    likely the Pi thermally throttled. The script will print a warning.
  - Stop diofinder before running: `sudo systemctl stop efinder` or
    equivalent. Other CPU consumers contaminate the measurement.

Usage
-----
    python3 backend_speed_test.py \\
        --database /var/lib/efinder/default_database.npz \\
        /var/lib/efinder/test.png /var/lib/efinder/test2.png

    python3 backend_speed_test.py --runs 100 --warmup-frames 10 ...

    python3 backend_speed_test.py --backends olive,sycamore ...   # skip cedar

The database path is required because the solver needs it (and olive's
constructor takes it).
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from typing import Callable, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image
except ImportError:
    print("Need pillow: pip install pillow", file=sys.stderr)
    sys.exit(1)


# --- Image I/O ------------------------------------------------------------

def load_gray_u8(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.ascontiguousarray(np.asarray(img, dtype=np.uint8))


# --- Backend abstractions -------------------------------------------------
#
# Each backend exposes:
#   .name           identifier for output
#   .extract(img)   -> list of (y, x) tuples in image coordinates
#   .solve(yx, shape)  -> (ra_rad, dec_rad, roll_rad) or None on failure
#
# The solver is the same olive-solve instance across all backends, so the
# difference between rows is purely the extractor (and any centroid-format
# overhead, which we capture in extract time).

class OliveBackend:
    name = "olive"

    def __init__(self, db_path: str):
        import tetra3 as olive
        self._t3 = olive.Tetra3(database_path=db_path)
        # Solver kwargs we'll pass on each solve call; tune if needed.
        self._solve_kwargs = {}

    def extract(self, img: np.ndarray) -> List[Tuple[float, float]]:
        # olive-solve's get_centroids_from_image_fast returns an (N, 2) array
        # of (y, x). It also returns auxiliary outputs in some versions — we
        # take the first element if a tuple is returned.
        result = self._t3.get_centroids_from_image_fast(
            img, sigma=8.0, bg_sub_mode="local_mean",
            sigma_mode="global_root_square",
        )
        if isinstance(result, tuple):
            centroids = result[0]
        else:
            centroids = result
        return [(float(c[0]), float(c[1])) for c in centroids]

    def solve(self, yx: List[Tuple[float, float]], shape) -> Optional[Tuple[float, float, float]]:
        if len(yx) < 4:
            return None
        arr = np.array(yx, dtype=np.float64)
        try:
            result = self._t3.solve_from_centroids(arr, (float(shape[0]), float(shape[1])))
        except Exception:
            return None
        if result is None:
            return None
        # The solver returns a dict-like with ra/dec/roll keys, or a tuple,
        # depending on the version. Normalize both.
        if isinstance(result, dict):
            ra = result.get("ra") or result.get("RA")
            dec = result.get("dec") or result.get("Dec")
            roll = result.get("roll") or result.get("Roll") or 0.0
            if ra is None or dec is None:
                return None
            return float(ra), float(dec), float(roll)
        try:
            return float(result[0]), float(result[1]), float(result[2])
        except (TypeError, IndexError):
            return None


class SycamoreBackend:
    """Sycamore-extract (star_detect) for extraction, olive-solve for solving."""
    name = "sycamore"

    def __init__(self, db_path: str, use_cache: bool = False):
        import star_detect
        import tetra3 as olive
        self._sd = star_detect
        self._t3 = olive.Tetra3(database_path=db_path)
        self._use_cache = use_cache
        self._cache = None
        if use_cache:
            # Lazy-import the bg_cache helper; it's pure Python.
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            try:
                from bg_cache import BackgroundCache
                self._cache = BackgroundCache(bin=2, stack_size=4,
                                              refresh_interval_s=5.0)
                self._cache.start()
            except ImportError:
                print("WARNING: bg_cache.py not found; sycamore-warm will fall back "
                      "to non-cached path.", file=sys.stderr)
                self._use_cache = False

    @property
    def name(self):
        return "sycamore_warm" if self._use_cache else "sycamore_cold"

    def prime(self, img: np.ndarray, n_frames: int = 10):
        """Feed N frames into the bg_cache to populate it. No-op if not using cache."""
        if not self._use_cache or self._cache is None:
            return
        for _ in range(n_frames):
            self._cache.submit_frame(img)
        # Wait a beat for the worker to consume frames and build the model.
        for _ in range(20):
            time.sleep(0.05)
            from bg_cache import CacheState
            if self._cache.state() == CacheState.STEADY:
                return
        # If we get here the cache never reached STEADY; the warm path will
        # silently fall back to line_median.

    def extract(self, img: np.ndarray) -> List[Tuple[float, float]]:
        if self._use_cache and self._cache is not None:
            stars = self._cache.detect(img, sigma=8.0, max_axis_ratio=4.0)
        else:
            stars = self._sd.detect_stars(
                img, sigma=8.0, bin=2, bg_mode="line_median",
                max_axis_ratio=4.0,
            )
        # detect_stars returns (x, y, brightness, peak); convert to (y, x).
        return [(float(y), float(x)) for (x, y, _b, _p) in stars]

    def solve(self, yx, shape):
        if len(yx) < 4:
            return None
        arr = np.array(yx, dtype=np.float64)
        try:
            result = self._t3.solve_from_centroids(arr, (float(shape[0]), float(shape[1])))
        except Exception:
            return None
        if result is None:
            return None
        if isinstance(result, dict):
            ra = result.get("ra") or result.get("RA")
            dec = result.get("dec") or result.get("Dec")
            roll = result.get("roll") or result.get("Roll") or 0.0
            if ra is None or dec is None:
                return None
            return float(ra), float(dec), float(roll)
        try:
            return float(result[0]), float(result[1]), float(result[2])
        except (TypeError, IndexError):
            return None


class CedarBackend:
    """cedar-detect for extraction, olive-solve for solving.

    NOTE: cedar-detect's Python binding has changed across versions. The
    extract() implementation below is a best guess based on common usage.
    If it fails on first call, run this once to find the real API:

        python3 -c "import cedar_detect_py as cd; help(cd)"

    and adjust the body of extract() accordingly.
    """
    name = "cedar"

    def __init__(self, db_path: str):
        import cedar_detect_py
        import tetra3 as olive
        self._cd = cedar_detect_py
        self._t3 = olive.Tetra3(database_path=db_path)

    def extract(self, img: np.ndarray) -> List[Tuple[float, float]]:
        # YOU MAY NEED TO ADJUST THIS. Likely API surface:
        result = self._cd.get_stars_from_image(img, sigma_value=8.0)
        # Different versions return: list of star objects with .centroid_x/.y,
        # OR an (N, 2) ndarray of (y, x), OR a list of (y, x) tuples.
        # Handle the common cases:
        out = []
        for s in result:
            if hasattr(s, "centroid_y") and hasattr(s, "centroid_x"):
                out.append((float(s.centroid_y), float(s.centroid_x)))
            elif hasattr(s, "y") and hasattr(s, "x"):
                out.append((float(s.y), float(s.x)))
            else:
                # Fallback: assume iterable of (y, x) or (x, y, ...) — guess (y, x).
                try:
                    out.append((float(s[0]), float(s[1])))
                except Exception:
                    pass
        return out

    def solve(self, yx, shape):
        if len(yx) < 4:
            return None
        arr = np.array(yx, dtype=np.float64)
        try:
            result = self._t3.solve_from_centroids(arr, (float(shape[0]), float(shape[1])))
        except Exception:
            return None
        if result is None:
            return None
        if isinstance(result, dict):
            ra = result.get("ra") or result.get("RA")
            dec = result.get("dec") or result.get("Dec")
            roll = result.get("roll") or result.get("Roll") or 0.0
            if ra is None or dec is None:
                return None
            return float(ra), float(dec), float(roll)
        try:
            return float(result[0]), float(result[1]), float(result[2])
        except (TypeError, IndexError):
            return None


# --- LX200 formatting -----------------------------------------------------
#
# What SkySafari actually sees from a finder is a sequence of :GR# / :GD#
# responses. We time the formatting of a single (RA, Dec) pair into the
# LX200 strings; that's the "send to SkySafari" leg minus actual I/O.

import math

def format_lx200(ra_rad: float, dec_rad: float) -> Tuple[bytes, bytes]:
    h = math.degrees(ra_rad) / 15.0
    hh = int(h)
    mm = int((h - hh) * 60)
    ss = int((((h - hh) * 60) - mm) * 60)
    ra_str = f"{hh:02d}:{mm:02d}:{ss:02d}#".encode()

    d = math.degrees(dec_rad)
    sign = "+" if d >= 0 else "-"
    d = abs(d)
    dd = int(d)
    mm2 = int((d - dd) * 60)
    ss2 = int((((d - dd) * 60) - mm2) * 60)
    dec_str = f"{sign}{dd:02d}*{mm2:02d}:{ss2:02d}#".encode()
    return ra_str, dec_str


# --- Timing harness -------------------------------------------------------

def percentiles(ts_ms: List[float]) -> Tuple[float, float, float, float]:
    ts = sorted(ts_ms)
    n = len(ts)
    return ts[0], ts[n // 2], ts[int(n * 0.95)], ts[-1]


def time_pipeline(backend, img: np.ndarray, runs: int) -> dict:
    """Run extract -> solve -> format `runs` times. Return per-stage stats."""
    # Warmup
    for _ in range(5):
        yx = backend.extract(img)
        if len(yx) >= 4:
            r = backend.solve(yx, img.shape)
            if r is not None:
                format_lx200(r[0], r[1])

    extract_ts, solve_ts, lx_ts, e2e_ts = [], [], [], []
    n_solved = 0
    n_centroids_last = 0

    for _ in range(runs):
        t0 = time.perf_counter_ns()
        yx = backend.extract(img)
        t1 = time.perf_counter_ns()
        result = backend.solve(yx, img.shape) if len(yx) >= 4 else None
        t2 = time.perf_counter_ns()
        if result is not None:
            n_solved += 1
            format_lx200(result[0], result[1])
        t3 = time.perf_counter_ns()

        extract_ts.append((t1 - t0) / 1e6)
        solve_ts.append((t2 - t1) / 1e6)
        lx_ts.append((t3 - t2) / 1e6)
        e2e_ts.append((t3 - t0) / 1e6)
        n_centroids_last = len(yx)

    return {
        "name": backend.name,
        "extract": percentiles(extract_ts),
        "solve": percentiles(solve_ts),
        "lx200": percentiles(lx_ts),
        "e2e": percentiles(e2e_ts),
        "n_centroids": n_centroids_last,
        "n_solved": n_solved,
        "runs": runs,
    }


def fmt_stats(d: dict) -> str:
    e_min, e_p50, e_p95, e_max = d["extract"]
    s_min, s_p50, s_p95, s_max = d["solve"]
    t_min, t_p50, t_p95, t_max = d["e2e"]
    return (
        f"  {d['name']:18s}  "
        f"extract p50/p95 {e_p50:6.2f}/{e_p95:6.2f}  "
        f"solve p50/p95 {s_p50:6.2f}/{s_p95:6.2f}  "
        f"e2e p50/p95 {t_p50:6.2f}/{t_p95:6.2f}  "
        f"n={d['n_centroids']:3d}  solved={d['n_solved']}/{d['runs']}"
    )


# --- Optional thermal log ---------------------------------------------------

def read_temp_c() -> Optional[float]:
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        return None


def read_freq_mhz() -> Optional[float]:
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq") as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        return None


# --- Driver ----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", nargs="+", help="Test image files.")
    ap.add_argument("--database", required=True,
                    help="Path to olive-solve .npz database.")
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup-frames", type=int, default=10,
                    help="Frames to feed sycamore's bg_cache before warm timing.")
    ap.add_argument("--backends", default="olive,sycamore_cold,sycamore_warm,cedar",
                    help="Comma-separated list. Skip what isn't available.")
    ap.add_argument("--cooldown-s", type=float, default=2.0,
                    help="Pause between backends to let CPU temperature settle.")
    args = ap.parse_args()

    requested = [b.strip() for b in args.backends.split(",") if b.strip()]
    for path in args.frames:
        if not os.path.exists(path):
            print(f"Frame not found: {path}", file=sys.stderr)
            sys.exit(2)
    if not os.path.exists(args.database):
        print(f"Database not found: {args.database}", file=sys.stderr)
        sys.exit(2)

    # Load all frames once.
    frames = [(os.path.basename(p), load_gray_u8(p)) for p in args.frames]

    # Build backends lazily so import errors are clear.
    backends = []
    if "olive" in requested:
        try:
            backends.append(("olive", OliveBackend(args.database)))
        except Exception as e:
            print(f"SKIP olive: {e}", file=sys.stderr)
    if "sycamore_cold" in requested:
        try:
            backends.append(("sycamore_cold", SycamoreBackend(args.database, use_cache=False)))
        except Exception as e:
            print(f"SKIP sycamore_cold: {e}", file=sys.stderr)
    if "sycamore_warm" in requested:
        try:
            sc = SycamoreBackend(args.database, use_cache=True)
            backends.append(("sycamore_warm", sc))
        except Exception as e:
            print(f"SKIP sycamore_warm: {e}", file=sys.stderr)
    if "cedar" in requested:
        try:
            backends.append(("cedar", CedarBackend(args.database)))
        except Exception as e:
            print(f"SKIP cedar: {e}", file=sys.stderr)

    if not backends:
        print("No backends available. Check imports.", file=sys.stderr)
        sys.exit(3)

    # Drift check disabled to keep memory footprint within Pi Zero 2W limits.
    drift_check = None

    print(f"runs/measurement={args.runs}  frames={len(frames)}  "
          f"backends={[b[0] for b in backends]}")
    t_start, f_start = read_temp_c(), read_freq_mhz()
    if t_start is not None:
        print(f"temp_start={t_start:.1f}C  freq_start={f_start:.0f}MHz")
    print()

    initial_olive_e2e = None

    for fname, img in frames:
        print(f"### {fname}  ({img.shape[0]}x{img.shape[1]})")
        for label, backend in backends:
            # Prime sycamore_warm before measurement
            if label == "sycamore_warm" and hasattr(backend, "prime"):
                backend.prime(img, n_frames=args.warmup_frames)

            stats = time_pipeline(backend, img, args.runs)
            print(fmt_stats(stats))

            if label == "olive" and initial_olive_e2e is None:
                initial_olive_e2e = stats["e2e"][1]  # p50

            t_now, f_now = read_temp_c(), read_freq_mhz()
            if t_now is not None:
                print(f"    [temp={t_now:.1f}C  freq={f_now:.0f}MHz]")
            time.sleep(args.cooldown_s)

        # Drift check on the last frame only.
        if drift_check is not None and fname == frames[-1][0]:
            label, backend = drift_check
            stats = time_pipeline(backend, img, args.runs)
            print(fmt_stats(stats))
            drift_p50 = stats["e2e"][1]
            if initial_olive_e2e is not None:
                drift_pct = (drift_p50 - initial_olive_e2e) / initial_olive_e2e * 100
                if abs(drift_pct) > 20:
                    print(f"  ⚠  DRIFT WARNING: olive p50 changed {drift_pct:+.1f}% "
                          f"during the test ({initial_olive_e2e:.2f} → {drift_p50:.2f} ms).")
                    print(f"  ⚠  Likely thermal throttling; results are unreliable.")
                else:
                    print(f"  drift OK: olive p50 changed {drift_pct:+.1f}% "
                          f"({initial_olive_e2e:.2f} → {drift_p50:.2f} ms)")
        print()

    t_end, f_end = read_temp_c(), read_freq_mhz()
    if t_end is not None:
        print(f"temp_end={t_end:.1f}C  freq_end={f_end:.0f}MHz  "
              f"(Δtemp = {t_end - t_start:+.1f}C)")


if __name__ == "__main__":
    main()
