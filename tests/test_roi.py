"""Off-device validation of detect_stars_roi (the tracking-mode ROI API).

Builds a synthetic frame (flat background + Gaussian noise + Gaussian stars at
known positions) and checks that one detect_stars_roi call over a window list
returns exactly one detection per star window, in FULL-FRAME coordinates,
within 1 px of truth; that empty windows return nothing; that out-of-bounds
windows are clamped defensively instead of panicking; and that the HAS_ROI
capability flag is exported.

Run:  pip install <built wheel>; pytest tests/test_roi.py
"""

import numpy as np
import pytest

star_detect = pytest.importorskip("star_detect")

W, H = 400, 400
BG = 20.0
NOISE_SIGMA = 2.0
STAR_SIGMA = 1.0  # PSF sigma, px
STAR_PEAK = 150.0  # amplitude above background
WIN = 48  # window side, px

# True star centers in the (0.5, 0.5) = center-of-pixel-(0,0) convention.
STARS = [
    (60.0, 80.0),
    (150.5, 60.25),
    (200.0, 210.0),
    (320.75, 120.5),
    (90.25, 330.0),
    (300.0, 340.5),
]

# Empty (starless) windows, well away from every star.
EMPTY_WINDOWS = [(10, 180), (340, 20)]  # top-left corners


def make_frame(rng):
    """Flat background + Gaussian noise + Gaussian stars at STARS."""
    frame = BG + rng.normal(0.0, NOISE_SIGMA, size=(H, W))
    yy, xx = np.mgrid[0:H, 0:W]
    # Pixel (i, j) center is at (j + 0.5, i + 0.5) in the output convention.
    px = xx + 0.5
    py = yy + 0.5
    for sx, sy in STARS:
        d2 = (px - sx) ** 2 + (py - sy) ** 2
        frame += STAR_PEAK * np.exp(-d2 / (2.0 * STAR_SIGMA**2))
    return np.clip(np.round(frame), 0, 255).astype(np.uint8)


def star_windows():
    """WIN x WIN windows centered on the true star positions."""
    rows = []
    for sx, sy in STARS:
        x0 = int(round(sx)) - WIN // 2
        y0 = int(round(sy)) - WIN // 2
        rows.append((x0, y0, x0 + WIN, y0 + WIN))
    return rows


@pytest.fixture(scope="module")
def frame():
    return make_frame(np.random.default_rng(12345))


def test_has_roi_flag():
    assert getattr(star_detect, "HAS_ROI", False) is True


def test_roi_detects_all_stars_full_frame_coords(frame):
    windows = np.array(
        star_windows() + [(x0, y0, x0 + WIN, y0 + WIN) for x0, y0 in EMPTY_WINDOWS],
        dtype=np.int64,
    )
    stars = star_detect.detect_stars_roi(frame, windows, sigma=8.0)

    # Exactly one detection per star window, none from the empty windows.
    assert len(stars) == len(STARS), f"expected {len(STARS)} stars, got {stars}"

    # Every returned centroid must be within 1 px of exactly one truth star,
    # in FULL-FRAME coordinates.
    truths = list(STARS)
    for x, y, brightness, peak in stars:
        dists = [np.hypot(x - sx, y - sy) for sx, sy in truths]
        i = int(np.argmin(dists))
        assert dists[i] <= 1.0, f"({x:.2f}, {y:.2f}) is {dists[i]:.2f} px from truth"
        truths.pop(i)  # each star claimed at most once
        assert brightness > 0.0
        assert 0 < peak <= 255
    assert not truths, f"stars not matched: {truths}"

    # Brightest-first ordering (the frozen output convention).
    b = [s[2] for s in stars]
    assert b == sorted(b, reverse=True)


def test_roi_empty_windows_return_nothing(frame):
    windows = np.array(
        [(x0, y0, x0 + WIN, y0 + WIN) for x0, y0 in EMPTY_WINDOWS], dtype=np.int64
    )
    assert star_detect.detect_stars_roi(frame, windows, sigma=8.0) == []


def test_roi_out_of_bounds_window_clamped_no_panic(frame):
    # Window hanging off every edge, plus one fully outside, plus one
    # degenerate (too small after clamping), plus a valid star window with
    # deliberately out-of-bounds corners: must clamp, not panic, and still
    # detect the star whose window survives clamping.
    sx, sy = STARS[0]
    windows = np.array(
        [
            (-30, -30, 20, 20),  # clamps to 20x20 corner window
            (W - 20, H - 20, W + 100, H + 100),  # clamps to 20x20 corner
            (W + 50, H + 50, W + 90, H + 90),  # fully outside -> skipped
            (5, 5, 11, 60),  # 6 px wide -> degenerate, skipped
            (100, 100, 90, 140),  # inverted -> skipped
            # Star window pushed out of bounds on the left/top: after clamping
            # the star is still inside.
            (int(sx) - WIN, int(sy) - WIN, int(sx) + WIN // 2, int(sy) + WIN // 2),
        ],
        dtype=np.int64,
    )
    stars = star_detect.detect_stars_roi(frame, windows, sigma=8.0)
    assert len(stars) == 1
    x, y = stars[0][0], stars[0][1]
    assert np.hypot(x - sx, y - sy) <= 1.0


def test_roi_accepts_int32_windows(frame):
    windows = np.array(star_windows(), dtype=np.int32)
    stars = star_detect.detect_stars_roi(frame, windows, sigma=8.0)
    assert len(stars) == len(STARS)


def test_roi_rejects_bad_windows_shape(frame):
    with pytest.raises(ValueError):
        star_detect.detect_stars_roi(frame, np.zeros((3, 3), dtype=np.int64))
    with pytest.raises(ValueError):
        star_detect.detect_stars_roi(frame, np.zeros((3, 4), dtype=np.float64))


def test_roi_at_most_one_star_per_window(frame):
    # A window covering two stars must return only the brightest single one.
    (x1, y1), (x2, y2) = STARS[0], STARS[1]
    x0 = int(min(x1, x2)) - 24
    y0 = int(min(y1, y2)) - 24
    x3 = int(max(x1, x2)) + 24
    y3 = int(max(y1, y2)) + 24
    windows = np.array([(x0, y0, x3, y3)], dtype=np.int64)
    stars = star_detect.detect_stars_roi(frame, windows, sigma=8.0)
    assert len(stars) == 1
