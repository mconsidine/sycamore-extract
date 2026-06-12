"""
Background cache + state machine for the finder.

Maintains a temporally-averaged background model + noise sigma in a worker
thread, publishes it atomically, and exposes:

    cache.detect(image_u8, sigma=8.0, max_axis_ratio=4.0, ...)

The model is either a 1-D per-row floor (``model="row"``, default; cheapest,
vertical gradients) or a 2-D per-tile median grid (``model="block"``, added
with star_detect v0.12; non-separable 2-D gradients — moon glow, light
pollution). The block variant is the first time 2-D background correction
composes with the temporal cache.

which routes to either:
  - star_detect.detect_stars_with_cache(...)  when the cache is fresh (STEADY)
  - star_detect.detect_stars(..., bg_mode='line_median')  otherwise

The worker pulls captured frames from a ring buffer, accumulates a stack, and
on each refresh:
  1. median-stacks N frames along time axis (robust to meteors/aircraft),
  2. computes per-row median across the stacked frame,
  3. estimates noise sigma from the same stack,
  4. atomically swaps the new model in.

Slew invalidation is driven by IMU angular distance: call `cache.note_motion(q)`
on each IMU update; if the integrated angle since the cached pose exceeds
`slew_threshold_deg`, the state drops to SLEWING and detection falls back to
per-frame LineMedian until the worker rebuilds.

This module is the application-side counterpart to star_detect v0.5+.
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

import numpy as np
import star_detect


class CacheState(Enum):
    WARMING_UP = auto()  # haven't yet collected enough frames
    STEADY = auto()      # cache is fresh, use detect_stars_with_cache
    SLEWING = auto()     # IMU says we moved; cache is stale until rebuilt


@dataclass(frozen=True)
class BgModel:
    """Snapshot of the cached background. Frozen so swaps are atomic.

    Carries EITHER a 1-D per-row floor (``row_offsets``) OR a 2-D per-tile
    median grid (``block_offsets``), depending on the cache's ``model`` mode.
    The unused one is None.
    """
    noise: float              # scalar pixel units
    h: int                    # full-res image height that produced it
    w: int                    # full-res image width that produced it
    bin: int                  # 1, 2, or 4; det image = h // bin rows
    epoch: float              # time.monotonic() when built
    n_frames: int             # how many frames went into the temporal stack
    row_offsets: Optional[np.ndarray] = None    # uint8 (h_det,) — row model
    block_offsets: Optional[np.ndarray] = None  # uint8 (grid_h, grid_w) — block model
    block_size: int = 32                        # tile size for the block model
    pose_quat: Optional[Tuple[float, float, float, float]] = None


class BackgroundCache:
    """
    Application-side cache. Thread-safe: workers can update while detection
    reads via an atomic reference swap (Python attribute writes are atomic at
    the bytecode level; we don't need a lock around `self._model`).
    """

    def __init__(
        self,
        bin: int = 2,
        stack_size: int = 8,
        refresh_interval_s: float = 5.0,
        slew_threshold_deg: float = 0.5,
        max_age_s: float = 60.0,
        model: str = "row",       # "row" (1-D per-row floor) or "block" (2-D grid)
        block_size: int = 32,     # tile size for the block model (bin=2 default)
    ):
        if model not in ("row", "block"):
            raise ValueError("model must be 'row' or 'block'")
        self.bin = bin
        self.model = model
        self.block_size = block_size
        self.stack_size = stack_size
        self.refresh_interval_s = refresh_interval_s
        self.slew_threshold_deg = math.radians(slew_threshold_deg)
        self.max_age_s = max_age_s

        # The cache. None when WARMING_UP.
        self._model: Optional[BgModel] = None

        # Frame ring buffer for temporal stacking. Worker reads; capture writes.
        self._frame_buf: "deque[np.ndarray]" = deque(maxlen=stack_size)
        self._frame_buf_lock = threading.Lock()

        # Forced rebuild flag (set on slew, cleared by worker).
        self._needs_rebuild = threading.Event()
        # Worker control.
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None

        # IMU bookkeeping.
        self._last_imu_quat: Optional[Tuple[float, float, float, float]] = None
        self._slewing = False

    # ----- lifecycle -------------------------------------------------------
    def start(self):
        if self._worker is not None:
            return
        self._worker = threading.Thread(
            target=self._worker_loop, name="bg-cache", daemon=True
        )
        self._worker.start()

    def stop(self):
        self._stop.set()
        if self._worker:
            self._worker.join(timeout=2)

    # ----- producer side ---------------------------------------------------
    def submit_frame(self, frame_u8: np.ndarray):
        """Capture thread calls this on every frame."""
        if frame_u8.dtype != np.uint8:
            raise TypeError("frame_u8 must be uint8")
        with self._frame_buf_lock:
            self._frame_buf.append(frame_u8)

    def note_motion(self, quat: Tuple[float, float, float, float]):
        """IMU thread calls this on each attitude update."""
        model = self._model
        if model is None or model.pose_quat is None:
            self._last_imu_quat = quat
            return
        # Angular distance from the pose at last refresh.
        ang = _angular_distance(quat, model.pose_quat)
        if ang > self.slew_threshold_deg and not self._slewing:
            self._slewing = True
            self._needs_rebuild.set()
        elif ang <= self.slew_threshold_deg and self._slewing:
            # Stopped moving; signal worker to refresh ASAP.
            self._slewing = False
            self._needs_rebuild.set()
        self._last_imu_quat = quat

    # ----- consumer side ---------------------------------------------------
    def state(self) -> CacheState:
        m = self._model
        if m is None:
            return CacheState.WARMING_UP
        if self._slewing:
            return CacheState.SLEWING
        if time.monotonic() - m.epoch > self.max_age_s:
            return CacheState.SLEWING  # treat too-old as needs-rebuild
        return CacheState.STEADY

    def detect(
        self,
        image_u8: np.ndarray,
        sigma: float = 8.0,
        max_axis_ratio: float = float("inf"),
        use_neon: bool = False,
    ):
        """
        Single entry point for the finder loop. Routes to the cached path when
        possible, falls back to per-frame line_median otherwise.

        Returns list of (x, y, brightness, peak), brightest first, in
        full-resolution pixel coordinates.
        """
        st = self.state()
        m = self._model
        if st is CacheState.STEADY and m is not None and m.h == image_u8.shape[0] \
                and m.w == image_u8.shape[1] and m.bin == self.bin:
            if m.block_offsets is not None:
                # 2-D block-grid cached path: composes the temporal cache with
                # non-separable 2-D background removal. row_offsets stays None.
                return star_detect.detect_stars_with_cache(
                    image_u8,
                    None,
                    m.noise,
                    sigma=sigma,
                    bin=self.bin,
                    use_neon=use_neon,
                    max_axis_ratio=max_axis_ratio,
                    block_offsets=m.block_offsets,
                    block_size=m.block_size,
                )
            return star_detect.detect_stars_with_cache(
                image_u8,
                m.row_offsets,
                m.noise,
                sigma=sigma,
                bin=self.bin,
                use_neon=use_neon,
                max_axis_ratio=max_axis_ratio,
            )
        # Fallback: per-frame line_median. Slower but self-contained.
        return star_detect.detect_stars(
            image_u8,
            sigma=sigma,
            bin=self.bin,
            bg_mode="line_median",
            use_neon=use_neon,
            max_axis_ratio=max_axis_ratio,
        )

    # ----- worker ----------------------------------------------------------
    def _worker_loop(self):
        last_build = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            need = (
                self._needs_rebuild.is_set()
                or (self._model is None and self._frame_count() >= self.stack_size)
                or (now - last_build > self.refresh_interval_s
                    and self._frame_count() >= self.stack_size
                    and not self._slewing)
            )
            if not need:
                # Sleep a bit; we don't need to spin.
                self._stop.wait(0.25)
                continue
            # Don't try to rebuild while the mount is slewing — frames are blurred
            # and the resulting model would be garbage.
            if self._slewing:
                self._stop.wait(0.25)
                continue

            stack = self._snapshot_stack()
            if len(stack) < max(2, self.stack_size // 2):
                self._stop.wait(0.25)
                continue
            try:
                model = self._build_model(stack)
                self._model = model         # atomic publish
                last_build = now
                self._needs_rebuild.clear()
            except Exception as e:
                # Don't let a bad frame kill the worker.
                print(f"[bg-cache] build failed: {e}")
                self._stop.wait(0.5)

    def _frame_count(self) -> int:
        with self._frame_buf_lock:
            return len(self._frame_buf)

    def _snapshot_stack(self) -> list:
        with self._frame_buf_lock:
            return list(self._frame_buf)

    def _build_model(self, frames: list) -> BgModel:
        """
        Build the cached model from a list of recent frames.
          1. median across time (rejects single-frame transients: aircraft, meteors).
          2. optional bin (we work in the detection-image space).
          3. per-row median of the temporally-medianed frame.
          4. noise sigma from MAD on a dark patch of the same.
        """
        h, w = frames[0].shape
        # 1. Time-median. np.median on uint8 → float, cast back to uint8.
        stack = np.stack(frames, axis=0)
        time_med = np.median(stack, axis=0).astype(np.uint8)

        # 2. Optional binning to match the detection-image resolution.
        if self.bin == 2:
            tm = time_med[: (h // 2) * 2, : (w // 2) * 2]
            time_med = (
                tm.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3)).astype(np.uint8)
            )
            h_det, w_det = time_med.shape
        else:
            h_det, w_det = h, w

        # 3. Background model from the native helper (parallel histograms).
        #    "row":   1-D per-row median floor (cheap; vertical gradients).
        #    "block": 2-D per-tile median grid (non-separable 2-D gradients;
        #             composes 2-D background correction with the temporal cache).
        if self.model == "block":
            row_offsets = None
            block_offsets = star_detect.compute_block_medians_py(
                time_med, block_size=self.block_size
            )
        else:
            row_offsets = star_detect.compute_row_medians_py(time_med)
            block_offsets = None

        # 4. Noise sigma: MAD on a dark patch in the middle quarter.
        patch = time_med[h_det // 3 : 2 * h_det // 3, w_det // 3 : 2 * w_det // 3]
        flat = patch.astype(np.float32).ravel()
        med = np.median(flat)
        mad = np.median(np.abs(flat - med))
        noise = max(0.5, 1.4826 * float(mad))

        return BgModel(
            row_offsets=row_offsets,
            block_offsets=block_offsets,
            block_size=self.block_size,
            noise=noise,
            h=h, w=w, bin=self.bin,
            epoch=time.monotonic(),
            n_frames=len(frames),
            pose_quat=self._last_imu_quat,
        )


def _angular_distance(q1, q2) -> float:
    """Smallest rotation between two unit quaternions, radians."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    dot = abs(w1 * w2 + x1 * x2 + y1 * y2 + z1 * z2)
    return 2.0 * math.acos(min(1.0, dot))
