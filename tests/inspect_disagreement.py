#!/usr/bin/env python3
"""
Visualize stars where the two gate modes disagree.

Runs detect_stars in both cedar and matched_filter modes on a single image,
finds the stars each gate accepts that the other rejects, and saves a small
neighborhood crop for each one so you can eyeball whether it's a real star,
a hot pixel, or noise.

Outputs:
  out/cedar_only_summary.png   Grid of all stars cedar found but MF rejected.
  out/mf_only_summary.png      Grid of all stars MF found but cedar rejected.
  out/individual/...           One PNG per disagreement, named by (mode, x, y).

The crops use an arcsinh stretch so faint structure is visible without
blowing out bright cores. Each crop has a red crosshair at the detected
centroid and shows brightness + peak in the title strip.

Usage:
    python3 tests/inspect_disagreement.py /var/lib/efinder/test2.png
    python3 tests/inspect_disagreement.py --sigma 8.0 --crop 21 frame.png
    python3 tests/inspect_disagreement.py --out-dir my_run frame.png
"""
import argparse
import os
import sys

import numpy as np

try:
    from PIL import Image, ImageDraw
except ImportError:
    print("Need pillow: pip install pillow", file=sys.stderr)
    sys.exit(1)

import star_detect


def load_gray_u8(path):
    img = Image.open(path).convert("L")
    return np.ascontiguousarray(np.asarray(img, dtype=np.uint8))


def find_disagreements(cedar_stars, mf_stars, tol_px=1.5):
    """
    Greedy match between cedar and MF centroid lists.
    Returns (cedar_only, mf_only) — each a list of (x, y, brightness, peak).
    """
    if not cedar_stars and not mf_stars:
        return [], []
    A = np.array([(x, y) for (x, y, *_) in cedar_stars]) if cedar_stars else np.zeros((0, 2))
    B = np.array([(x, y) for (x, y, *_) in mf_stars]) if mf_stars else np.zeros((0, 2))

    used_b = set()
    matched_a = set()
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
                matched_a.add(i)
            break

    cedar_only = [cedar_stars[i] for i in range(len(cedar_stars)) if i not in matched_a]
    mf_only = [mf_stars[j] for j in range(len(mf_stars)) if j not in used_b]
    return cedar_only, mf_only


def asinh_stretch(arr, beta=None):
    """
    arcsinh stretch for display. arr is float; returns uint8 in [0, 255].
    beta defaults to 10% of the median (the SkySafari-style "sky-subtracted
    asinh" that diofinder uses for its live view).
    """
    arr = arr.astype(np.float64)
    sky = float(np.median(arr))
    residual = arr - sky
    if beta is None:
        beta = max(1.0, sky * 0.1)
    stretched = np.arcsinh(residual / beta)
    lo = float(np.percentile(stretched, 1))
    hi = float(np.percentile(stretched, 99.5))
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((stretched - lo) / (hi - lo), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def crop_neighborhood(image, cx, cy, half=10):
    """Crop a (2*half+1) square around (cx, cy). Pads with zeros if near edge.
    Returns (crop_u8_stretched, (dx, dy)) where (dx, dy) is the centroid
    position within the crop coordinate system."""
    h, w = image.shape
    cxi, cyi = int(round(cx)), int(round(cy))
    size = 2 * half + 1
    out = np.zeros((size, size), dtype=np.uint8)
    x0_src = max(0, cxi - half)
    y0_src = max(0, cyi - half)
    x1_src = min(w, cxi + half + 1)
    y1_src = min(h, cyi + half + 1)
    x0_dst = x0_src - (cxi - half)
    y0_dst = y0_src - (cyi - half)
    x1_dst = x0_dst + (x1_src - x0_src)
    y1_dst = y0_dst + (y1_src - y0_src)
    out[y0_dst:y1_dst, x0_dst:x1_dst] = image[y0_src:y1_src, x0_src:x1_src]
    out_stretched = asinh_stretch(out)
    # centroid position within crop frame (continuous coords)
    dx = cx - (cxi - half)
    dy = cy - (cyi - half)
    return out_stretched, (dx, dy)


def annotate_crop(crop_u8, dx, dy, label, scale=8):
    """Upscale the crop and draw a red crosshair + label."""
    h, w = crop_u8.shape
    img = Image.fromarray(crop_u8, mode="L").convert("RGB")
    img = img.resize((w * scale, h * scale), Image.NEAREST)
    draw = ImageDraw.Draw(img)
    cx_pix = int(dx * scale)
    cy_pix = int(dy * scale)
    # Crosshair (4-pixel gap so the centroid itself isn't covered).
    arm = scale * 3
    gap = scale
    red = (255, 64, 64)
    draw.line([(cx_pix - arm - gap, cy_pix), (cx_pix - gap, cy_pix)], fill=red, width=2)
    draw.line([(cx_pix + gap, cy_pix), (cx_pix + arm + gap, cy_pix)], fill=red, width=2)
    draw.line([(cx_pix, cy_pix - arm - gap), (cx_pix, cy_pix - gap)], fill=red, width=2)
    draw.line([(cx_pix, cy_pix + gap), (cx_pix, cy_pix + arm + gap)], fill=red, width=2)
    # Label strip at top
    draw.rectangle([(0, 0), (w * scale, 16)], fill=(0, 0, 0))
    draw.text((4, 1), label, fill=(220, 220, 220))
    return img


def make_grid(annotated_imgs, cols=4, pad=4, bg=(20, 20, 20)):
    """Tile annotated PIL images into a single summary image."""
    if not annotated_imgs:
        return None
    cell_w = max(im.width for im in annotated_imgs)
    cell_h = max(im.height for im in annotated_imgs)
    rows = (len(annotated_imgs) + cols - 1) // cols
    total_w = cols * cell_w + (cols + 1) * pad
    total_h = rows * cell_h + (rows + 1) * pad
    grid = Image.new("RGB", (total_w, total_h), bg)
    for idx, im in enumerate(annotated_imgs):
        r, c = divmod(idx, cols)
        x = pad + c * (cell_w + pad)
        y = pad + r * (cell_h + pad)
        grid.paste(im, (x, y))
    return grid


def save_disagreements(stars, mode_name, image, out_dir, crop_half):
    """Save individual crops + a summary grid for one mode's disagreements."""
    if not stars:
        return None
    per_dir = os.path.join(out_dir, "individual")
    os.makedirs(per_dir, exist_ok=True)
    annotated = []
    for (x, y, br, pk) in sorted(stars, key=lambda s: -s[2]):
        crop, (dx, dy) = crop_neighborhood(image, x, y, half=crop_half)
        label = f"{mode_name}  ({x:.1f},{y:.1f}) br={br:.0f} pk={pk}"
        ann = annotate_crop(crop, dx, dy, label)
        ann.save(os.path.join(per_dir,
                              f"{mode_name}_{int(round(x)):04d}_{int(round(y)):04d}.png"))
        annotated.append(ann)
    grid = make_grid(annotated)
    if grid is not None:
        grid_path = os.path.join(out_dir, f"{mode_name}_summary.png")
        grid.save(grid_path)
        return grid_path
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="Frame to analyze.")
    ap.add_argument("--sigma", type=float, default=8.0)
    ap.add_argument("--bin", type=int, default=1, choices=[1, 2])
    ap.add_argument("--bg-mode", default="row_percentile",
                    choices=["row_percentile", "line_median"])
    ap.add_argument("--max-axis-ratio", type=float, default=float("inf"))
    ap.add_argument("--tol", type=float, default=1.5,
                    help="Matching tolerance in pixels.")
    ap.add_argument("--crop", type=int, default=10,
                    help="Crop half-size in pixels (window = 2*half+1).")
    ap.add_argument("--out-dir", default="disagreements",
                    help="Directory for output PNGs.")
    args = ap.parse_args()

    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}", file=sys.stderr)
        sys.exit(2)

    img = load_gray_u8(args.image)
    h, w = img.shape
    print(f"image: {os.path.basename(args.image)}  ({h}x{w})")
    print(f"sigma={args.sigma}  bin={args.bin}  bg_mode={args.bg_mode}  tol={args.tol}px")

    common = dict(
        sigma=args.sigma, bin=args.bin, bg_mode=args.bg_mode,
        max_axis_ratio=args.max_axis_ratio,
    )

    cedar_stars = star_detect.detect_stars(img, gate_mode="cedar", **common)
    mf_stars = star_detect.detect_stars(img, gate_mode="matched_filter", **common)
    print(f"cedar: {len(cedar_stars)} stars")
    print(f"mf:    {len(mf_stars)} stars")

    cedar_only, mf_only = find_disagreements(cedar_stars, mf_stars, tol_px=args.tol)
    print(f"cedar_only (cedar found, MF rejected): {len(cedar_only)}")
    print(f"mf_only    (MF found, cedar rejected): {len(mf_only)}")

    os.makedirs(args.out_dir, exist_ok=True)

    co_path = save_disagreements(cedar_only, "cedar_only", img, args.out_dir, args.crop)
    mo_path = save_disagreements(mf_only, "mf_only", img, args.out_dir, args.crop)

    print()
    if co_path:
        print(f"wrote: {co_path}")
    if mo_path:
        print(f"wrote: {mo_path}")
    if not co_path and not mo_path:
        print("No disagreements at this sigma. Try a lower sigma to find more.")
    else:
        print(f"individual crops: {os.path.join(args.out_dir, 'individual')}/")
        print()
        print("What to look for in each crop:")
        print("  - PSF-shaped bump (~2-4 px FWHM)  → likely a real star")
        print("  - single bright pixel, no skirt   → hot pixel")
        print("  - noisy-looking, no clear center  → noise blob (false positive)")


if __name__ == "__main__":
    main()
