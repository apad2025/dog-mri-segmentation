"""
biceps_pipeline.py
==================

DICOM loading + classical detection helpers for muscle segmentation.

This was originally a full SAM (vit_l) batch pipeline; the SAM stage has been
dropped now that segmentation is done by sam2_segmentation.ipynb.
What remains are the pieces that notebook imports:

    load_dicom_images -> foreground_mask -> bone_mask -> crotch_cut -> leg_boxes

The notebook runs this detection sequence on a single middle "seed" slice to
get the per-leg bounding boxes, then lets SAM2 propagate the mask through the
stack.

The thresholds / sizes are copied from the original notebook and may need
tuning per dog.
"""

from pathlib import Path
import numpy as np
import pydicom

from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import (
    remove_small_objects, remove_small_holes, dilation, disk,
)
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes


# ----------------------------------------------------------------------------
# 1. LOADING  (your load_dicom_images, lightly hardened)
# ----------------------------------------------------------------------------
def load_dicom_images(
        folder,
        *,
        n_echo: int = 7,
        n_slices_per_echo: int = 50,
        shape=(192, 192),
        dtype=np.float32,
        mode: str = "magnitude",
        normalize_mag: bool = True,
):
    """Return imgs with shape (n_echo, n_slices_per_echo, H, W).

    Bins each file by its DICOM EchoNumbers tag, exactly like your notebook.
    Normalisation is done per-volume so every slice shares the same intensity
    scale -- important, because your bone threshold (mag < 0.23) and the
    SAM input both assume a stable 0..1 range.
    """
    mode = mode.lower()
    if mode not in {"magnitude", "phase"}:
        raise ValueError("mode must be 'magnitude' or 'phase'")

    folder = Path(folder)
    files = sorted(p for p in folder.iterdir() if p.is_file())
    expected = n_echo * n_slices_per_echo
    if len(files) < expected:
        raise ValueError(f"Found {len(files)} files in {folder.name}, expected {expected}.")

    H, W = shape
    imgs = np.zeros((n_echo, n_slices_per_echo, H, W), dtype=dtype)
    next_slice = np.zeros(n_echo, dtype=int)

    for p in files[:expected]:
        ds = pydicom.dcmread(str(p), force=True)
        img = ds.pixel_array

        en = getattr(ds, "EchoNumbers", None)
        if en is None:
            raise ValueError(f"Missing EchoNumbers in {p.name}")
        echo_idx = int(en) - 1
        if not (0 <= echo_idx < n_echo):
            raise ValueError(f"EchoNumbers={en} out of range in {p.name}")

        slice_idx = next_slice[echo_idx]
        if slice_idx >= n_slices_per_echo:
            raise ValueError(f"Too many slices for echo {en} ({p.name})")
        next_slice[echo_idx] += 1

        if mode == "phase":
            img = img.astype(np.float32, copy=False)
            img = (img * 2 - 4096) * (np.pi / np.max(img))

        imgs[echo_idx, slice_idx] = img

    if mode == "magnitude" and normalize_mag:
        mn, mx = float(imgs.min()), float(imgs.max())
        if mx > mn:
            imgs = (imgs - mn) / (mx - mn)

    return imgs


# ----------------------------------------------------------------------------
# 2. PER-SLICE MASKS  (cells 6 and 10, as functions)
# ----------------------------------------------------------------------------
def foreground_mask(mag, *, smooth_sigma=1.0, min_size=100, hole_area=100):
    """Body vs background via Otsu, then cleaned up."""
    mag_s = gaussian(mag, sigma=smooth_sigma, preserve_range=True)
    thresh = threshold_otsu(mag_s)
    fg = mag > thresh
    fg = remove_small_objects(fg, max_size=min_size, connectivity=1)
    fg = remove_small_holes(fg, max_size=hole_area, connectivity=1)
    return fg


def bone_mask(mag, *, dark_thresh=0.23, area_threshold=15, circularity_min=0.5):
    """Dark, roughly-circular regions = the two femurs."""
    bones = mag < dark_thresh
    lab = label(bones)
    out = np.zeros(mag.shape, dtype=bool)
    for region in regionprops(lab):
        if region.perimeter == 0:
            continue
        circ = (4 * np.pi * region.area) / (region.perimeter ** 2)
        if region.area > area_threshold and circ > circularity_min:
            out[tuple(region.coords.T)] = True  # vectorised vs the per-pixel loop
    return binary_fill_holes(out)


def crotch_cut(fgrnd, bone_grown, *, taper_ratio=0.45, dilation_radius=2):
    """Remove the tapered strip between the two femurs to split left/right legs.

    Returns body_cut (bool) or None if the two bones can't be found cleanly.
    """
    lab = label(bone_grown, connectivity=2)
    props = sorted(regionprops(lab), key=lambda r: r.area, reverse=True)
    if len(props) < 2:
        return None  # <-- was a hard raise; now skip
    props = sorted(props[:2], key=lambda r: r.centroid[1])
    left_bone = lab == props[0].label
    right_bone = lab == props[1].label

    x_left_inner = np.where(left_bone)[1].max() + 2
    x_right_inner = np.where(right_bone)[1].min() - 2
    if x_right_inner <= x_left_inner:
        return None

    ys = np.where(fgrnd)[0]
    if ys.size == 0:
        return None
    y_top, y_bot = ys.min(), ys.max()

    x_center = (x_left_inner + x_right_inner) / 2.0
    halfwidth_top = (x_right_inner - x_left_inner) / 2.0
    strip = np.zeros(fgrnd.shape, dtype=bool)
    height = max(y_bot - y_top, 1)
    for y in range(y_top, y_bot + 1):
        y_norm = (y - y_top) / height
        halfw = halfwidth_top * (1.0 - (1.0 - taper_ratio) * y_norm)
        x1 = int(np.clip(round(x_center - halfw), 0, strip.shape[1] - 1))
        x2 = int(np.clip(round(x_center + halfw), 0, strip.shape[1] - 1))
        if x2 > x1:
            strip[y, x1:x2 + 1] = True

    strip = dilation(strip & fgrnd, disk(dilation_radius))
    return fgrnd & ~strip


def leg_boxes(masked_img, bone_grown):
    """Per-leg bounding boxes for SAM (cell 11), or None if it can't assign two legs.

    Box = inner-top of the femur to the outer-bottom corner of the leg.
    """
    if masked_img.max() <= masked_img.min():
        return None
    bi = masked_img > threshold_otsu(masked_img)
    H, W = masked_img.shape

    lab_leg = label(bi, connectivity=2)
    leg_props = sorted(regionprops(lab_leg), key=lambda r: r.area, reverse=True)
    if len(leg_props) < 2:
        return None
    leg_props = sorted(leg_props[:2], key=lambda r: r.centroid[1])
    left_leg = lab_leg == leg_props[0].label
    right_leg = lab_leg == leg_props[1].label

    lab_bone = label(bone_grown, connectivity=2)
    bone_props = sorted(regionprops(lab_bone), key=lambda r: r.area, reverse=True)
    if len(bone_props) < 2:
        return None
    bone_props = bone_props[:2]

    def top_of_bone(blabel, side):
        ys, xs = np.where(lab_bone == blabel)
        y_top = int(ys.min())
        xs_at_top = xs[ys == y_top]
        x_top = int(xs_at_top.max()) + 5 if side == "left" else int(xs_at_top.min()) - 3
        return x_top, y_top

    def leg_of_point(x, y):
        if 0 <= y < H and 0 <= x < W:
            if left_leg[y, x]:
                return "left"
            if right_leg[y, x]:
                return "right"
        ly, lx = leg_props[0].centroid
        ry, rx = leg_props[1].centroid
        return "left" if (x - lx) ** 2 + (y - ly) ** 2 < (x - rx) ** 2 + (y - ry) ** 2 else "right"

    left_top = right_top = None
    for r in bone_props:
        yb, xb = r.centroid
        side = leg_of_point(int(round(xb)), int(round(yb)))
        if side == "left":
            left_top = top_of_bone(r.label, "left")
        else:
            right_top = top_of_bone(r.label, "right")
    if left_top is None or right_top is None:
        return None

    def outer(leg_mask, side):
        ys, xs = np.where(leg_mask)
        return (int(xs.min()) if side == "left" else int(xs.max())), int(ys.max())

    xL_out, yL_bot = outer(left_leg, "left")
    xR_out, yR_bot = outer(right_leg, "right")

    def order_box(x0, y0, x1, y1):
        return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]

    return [order_box(*left_top, xL_out, yL_bot),
            order_box(*right_top, xR_out, yR_bot)]
