#!/usr/bin/env python3
"""
Finder loop sketch — event-driven cadence, ROI tracking, concurrent I/O.

This is a SKETCH, not production code. It shows the control-flow shape; the
camera/IMU/solver/SkySafari/web hooks are stubs you replace with your real
implementations. The point is the *cadence and threading*, not the details.

Architectural premise:
  - star_detect releases the GIL during compute (v0.3+).
  - Detection cadence is driven by events (moved, stale, manual), NOT by frame
    rate. We don't solve every frame; we solve when it matters.
  - Tracking mode replaces full-frame detection with small ROI windows around
    predicted star positions. Cold lost-in-space is the only time we go full
    frame.
  - All I/O lives on dedicated threads so a slow solve doesn't stall responses
    to SkySafari or the web pages.
"""
import time
import math
import threading
from dataclasses import dataclass
from queue import Queue, Empty
import os

import numpy as np
import star_detect
from bg_cache import BackgroundCache

# Reserve cores for I/O. Done before first detect_stars() call. Adjust if you
# run more or fewer parallel processes on the Pi.
star_detect.set_num_threads(2)

# Background cache: maintains a temporally-averaged row-offset model in a
# worker thread, used by detection when the mount is steady. Falls back to
# per-frame LineMedian when warming up or slewing.
bg_cache = BackgroundCache(
    bin=2,
    stack_size=8,
    refresh_interval_s=5.0,
    slew_threshold_deg=0.5,
)


# ---------- shared state ---------------------------------------------------
@dataclass
class Pose:
    ra: float          # radians
    dec: float         # radians
    roll: float        # radians
    epoch: float       # monotonic seconds when this pose was valid
    confidence: float  # 0..1; 0 = lost, 1 = just solved cleanly


pose_lock = threading.Lock()
current_pose = Pose(0, 0, 0, 0.0, 0.0)

# Detection trigger: workers post requests here; the detect/solve worker reads.
trigger_q: "Queue[str]" = Queue(maxsize=4)

# Latest frame buffer + its capture timestamp. Camera thread writes; detect
# worker reads. Single-slot, last-wins (no backlog).
frame_lock = threading.Lock()
latest_frame = None        # np.ndarray uint8 (h, w)
latest_frame_t = 0.0


# ---------- camera thread --------------------------------------------------
def camera_thread(stop_evt: threading.Event):
    """
    Continuously captures frames. Always overwrites the single slot — we never
    queue stale frames. libcamera/picamera2 already does most of its work on
    the ISP/GPU, so this thread is light on CPU.
    """
    global latest_frame, latest_frame_t
    while not stop_evt.is_set():
        frame = capture_one()         # <-- your picamera2 call here
        ts = time.monotonic()
        with frame_lock:
            latest_frame = frame
            latest_frame_t = ts
        # If we haven't solved in a while, nudge the detector.
        if ts - last_solve_ts() > 0.5:
            trigger("stale")


def capture_one() -> np.ndarray:
    raise NotImplementedError("plug in picamera2 here")


# ---------- IMU thread -----------------------------------------------------
imu_lock = threading.Lock()
imu_quat = (1.0, 0.0, 0.0, 0.0)
imu_quat_t = 0.0
imu_quat_at_solve = (1.0, 0.0, 0.0, 0.0)  # IMU reading when last solve was good


def imu_thread(stop_evt: threading.Event):
    """
    High-rate (~100 Hz) attitude integration. This thread is the *only* thing
    that determines pose between solves. Pin to a dedicated core for jitter:
       os.sched_setaffinity(0, {0})
    """
    global imu_quat, imu_quat_t
    while not stop_evt.is_set():
        q = read_imu_quat()           # <-- your BNO085/ICM-20948 call here
        ts = time.monotonic()
        with imu_lock:
            imu_quat = q
            imu_quat_t = ts
        # Inform the background cache so it can invalidate on slew.
        bg_cache.note_motion(q)
        # If we've drifted significantly since the last solve, ask for a new one.
        if angular_distance(q, imu_quat_at_solve) > math.radians(0.5):
            trigger("moved")
        time.sleep(0.01)


def read_imu_quat():
    raise NotImplementedError("plug in your IMU here")


def angular_distance(q1, q2) -> float:
    """Smallest rotation between two unit quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    dot = abs(w1 * w2 + x1 * x2 + y1 * y2 + z1 * z2)
    return 2.0 * math.acos(min(1.0, dot))


# ---------- detect + solve worker ------------------------------------------
last_solve_ts_lock = threading.Lock()
_last_solve_ts = 0.0


def last_solve_ts() -> float:
    with last_solve_ts_lock:
        return _last_solve_ts


def set_last_solve_ts(t: float):
    global _last_solve_ts
    with last_solve_ts_lock:
        _last_solve_ts = t


def trigger(reason: str):
    """Request a detect+solve. Idempotent: dropping is fine (queue full)."""
    try:
        trigger_q.put_nowait(reason)
    except Exception:
        pass


def detect_solve_thread(stop_evt: threading.Event, solver):
    """
    Pulls trigger requests, takes the latest frame, runs detect + solve.
    Two modes:
      - Tracking: confidence high, recent solve, IMU prediction available.
        Detect within small ROIs around predicted star positions. Fast.
      - Cold:     confidence low, or no recent solve. Detect full frame.
        Slow but unavoidable when lost.
    """
    while not stop_evt.is_set():
        try:
            reason = trigger_q.get(timeout=0.05)
        except Empty:
            continue

        with frame_lock:
            frame = latest_frame
            frame_t = latest_frame_t
        if frame is None:
            continue

        # Decide tracking vs cold based on current confidence.
        with pose_lock:
            conf = current_pose.confidence
            age = time.monotonic() - current_pose.epoch
        tracking = conf > 0.6 and age < 1.0

        t0 = time.perf_counter()
        if tracking:
            # ROI mode: predict star positions from last solve + IMU delta and
            # only run detect_stars on small windows. Order of magnitude faster.
            with imu_lock:
                imu_now = imu_quat
            predicted = predict_star_positions(current_pose, imu_now, frame.shape)
            centroids = []
            for (cx, cy) in predicted[:30]:
                roi = extract_roi(frame, cx, cy, half=24)
                if roi is None:
                    continue
                local_stars = star_detect.detect_stars(roi, sigma=6.0, bin=1)
                # Best star in this ROI, offset to global coords.
                if local_stars:
                    lx, ly, br, pk = local_stars[0]
                    centroids.append((cx - 24 + lx, cy - 24 + ly, br))
        else:
            # Cold/wide mode: full-frame detection. Routes through the bg cache:
            #   - STEADY -> detect_stars_with_cache (precomputed row offsets)
            #   - WARMING_UP / SLEWING -> detect_stars(bg_mode='line_median')
            # The cache also wants every frame fed to it so its temporal stack
            # stays current. Cheap (just an append).
            bg_cache.submit_frame(frame)
            stars = bg_cache.detect(frame, sigma=8.0, max_axis_ratio=4.0)
            centroids = [(x, y, br) for (x, y, br, pk) in stars[:30]]
            cache_state = bg_cache.state().name

        t_detect = time.perf_counter() - t0

        if not centroids:
            with pose_lock:
                current_pose.confidence = max(0.0, current_pose.confidence - 0.2)
            continue

        t1 = time.perf_counter()
        result = solver.solve(centroids, frame.shape)
        t_solve = time.perf_counter() - t1

        if result is not None:
            ra, dec, roll = result
            with pose_lock:
                current_pose.ra = ra
                current_pose.dec = dec
                current_pose.roll = roll
                current_pose.epoch = frame_t
                current_pose.confidence = 1.0 if not tracking else min(1.0, conf + 0.1)
            with imu_lock:
                globals()["imu_quat_at_solve"] = imu_quat
            set_last_solve_ts(time.monotonic())
            print(f"[{reason}] {'TRACK' if tracking else 'COLD '} "
                  f"detect {t_detect*1e3:.1f}ms solve {t_solve*1e3:.1f}ms "
                  f"n={len(centroids)}")
        else:
            with pose_lock:
                current_pose.confidence = max(0.0, current_pose.confidence - 0.3)


def predict_star_positions(pose, imu_now, frame_shape):
    """Apply IMU delta to last solve to predict where bright stars should
    currently be. Returns list of (x, y) pixel centers."""
    raise NotImplementedError("project catalog stars given pose+IMU delta")


def extract_roi(frame, cx, cy, half=24):
    h, w = frame.shape
    cx, cy = int(cx), int(cy)
    if cx - half < 0 or cy - half < 0 or cx + half > w or cy + half > h:
        return None
    return np.ascontiguousarray(frame[cy - half:cy + half, cx - half:cx + half])


# ---------- SkySafari LX200 server -----------------------------------------
def skysafari_thread(stop_evt: threading.Event, port=4030):
    """
    Listens for SkySafari LX200-protocol queries and answers from current_pose,
    using IMU to extrapolate between solves. Pure I/O, no CPU contention.
    """
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", port))
    s.listen(1)
    s.settimeout(0.5)
    while not stop_evt.is_set():
        try:
            conn, _ = s.accept()
        except socket.timeout:
            continue
        with conn:
            handle_lx200(conn, stop_evt)
    s.close()


def handle_lx200(conn, stop_evt):
    """Parse :GR# / :GD# / :Q# / etc. and respond from latest IMU-extrapolated pose."""
    buf = b""
    while not stop_evt.is_set():
        try:
            data = conn.recv(64)
        except Exception:
            return
        if not data:
            return
        buf += data
        while b"#" in buf:
            cmd, _, buf = buf.partition(b"#")
            resp = lx200_response(cmd)
            if resp is not None:
                conn.sendall(resp)


def lx200_response(cmd: bytes):
    # Compute current pose: take last-solved pose, apply IMU delta since then.
    with pose_lock:
        p = current_pose
    with imu_lock:
        q_now = imu_quat
        q_solve = imu_quat_at_solve
    # delta_q = q_now * q_solve^-1, applied to (ra, dec, roll)... [details omitted]
    ra_now, dec_now = extrapolate(p, q_solve, q_now)
    if cmd == b":GR":
        return format_lx200_ra(ra_now)
    if cmd == b":GD":
        return format_lx200_dec(dec_now)
    return None


def extrapolate(pose, q_solve, q_now):
    """Replace with proper quaternion rotation; this stub returns the solve."""
    return pose.ra, pose.dec


def format_lx200_ra(ra):
    h = math.degrees(ra) / 15.0
    hh = int(h); mm = int((h - hh) * 60); ss = int((((h - hh) * 60) - mm) * 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}#".encode()


def format_lx200_dec(dec):
    d = math.degrees(dec)
    sign = "+" if d >= 0 else "-"
    d = abs(d)
    dd = int(d); mm = int((d - dd) * 60); ss = int((((d - dd) * 60) - mm) * 60)
    return f"{sign}{dd:02d}*{mm:02d}:{ss:02d}#".encode()


# ---------- web thread (self-hosted status / config pages) -----------------
def web_thread(stop_evt: threading.Event, port=8080):
    """Tiny HTTP server for status pages. Pure I/O; doesn't contend for CPU."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            with pose_lock:
                p = current_pose
            body = (
                f"RA  {math.degrees(p.ra):.4f}\n"
                f"Dec {math.degrees(p.dec):.4f}\n"
                f"conf {p.confidence:.2f} age {time.monotonic()-p.epoch:.2f}s\n"
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    httpd = HTTPServer(("0.0.0.0", port), Handler)
    httpd.timeout = 0.5
    while not stop_evt.is_set():
        httpd.handle_request()


# ---------- bring it all up ------------------------------------------------
class StubSolver:
    """Replace with your real solver (olive-solve / cedar-solve / etc)."""
    def solve(self, centroids, shape):
        if len(centroids) < 4:
            return None
        return (0.0, 0.0, 0.0)


def main():
    stop = threading.Event()
    solver = StubSolver()  # <-- replace
    bg_cache.start()
    threads = [
        threading.Thread(target=camera_thread, args=(stop,), name="cam", daemon=True),
        threading.Thread(target=imu_thread, args=(stop,), name="imu", daemon=True),
        threading.Thread(target=detect_solve_thread, args=(stop, solver), name="solve", daemon=True),
        threading.Thread(target=skysafari_thread, args=(stop,), name="sky", daemon=True),
        threading.Thread(target=web_thread, args=(stop,), name="web", daemon=True),
    ]
    # Optional: pin the latency-critical threads to specific cores.
    # On the Pi: cores 0-1 for I/O, 2-3 for star_detect's rayon workers.
    # NB: thread pinning has to be done from within each thread; see
    #     os.sched_setaffinity(0, {core}) inside each thread entry.
    for t in threads:
        t.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop.set()
        bg_cache.stop()
        for t in threads:
            t.join(timeout=2)


if __name__ == "__main__":
    main()
