"""
review_masks.py
---------------
Interactive viewer for SAM segmentation results.

Usage:
    python review_masks.py

Supports two mask folder formats:

  Classic  (masks_out, masks_out_sam2):
      Files named  {date}_{series}_mask.npy,  shape (7, 50, H, W).
      DICOM path is inferred from the filename automatically.

  Echo     (new_masks_sam1_test, etc.):
      Files named  bf_masks_echo{e:02d}.npy,  shape (n_slices, H, W) each.
      All echo files are stacked into (n_echoes, n_slices, H, W).
      Set DICOM_SERIES and SLICE_OFFSET below so the viewer aligns correctly.

FOLDERS:
20240709_GRE2D_FATWATER_WAYLON_0012 - good
20240710_GRE2D_FATWATER_SUSHI_0012          range: 13-25
                                            are slices before making the mask incorrect?
20240711-1_GRE2D_FATWATER_APHRODITE_0012
                                            range: 5-37
                                            over-masked onto part of lateralis (slide 25)
20240711-2_GRE2D_FATWATER_SELENE_0012
                                            range: 12-43
                                            bad at 31 and onward
20240923_GRE2D_FATWATER_SUSHI_0012
                                            range: 13-32
                                            masks are all bad, muscle is hard to find in the images
20240924_GRE2D_FATWATER_APHRODITE_0012
                                            range: 8-37
                                            right leg mask is shifted up, bad mask at 30 so need to mask out stomach
20240925_GRE2D_FATWATER_WAYLON_0012
                                            range: 10-37
                                            did not mask right leg at all, sequence is in reverse order
20240926_GRE2D_FATWATER_SELENE_0012
                                            range: 4-30
                                            masking is good overall, examine slide 26 left leg
20250113_GRE2D_FATWATER_WAYLON_0012
                                            range: 8-42
                                            after slide 30 manually adjust
20250115_GRE2D_FATWATER_SELENE_0012
                                            range: 9-39
                                            left mask is bad
20250127_GRE2D_FATWATER_APHRODITE_0012
                                            range: 9-27
                                            masks seem decent
20250129_GRE2D_FATWATER_SUSHI_0012
                                            range: 13-34
                                            masks are bad past 24
20250501_GRE2D_FATWATER_WAYLON_0012
                                            range: 14-49
                                            mask on right leg around 41 gets rough
20250502_GRE2D_FATWATER_SELENE_0012
                                            range: 18-46
                                            after 32 right leg mask fails
20250505_GRE2D_FATWATER_APHRODITE_0012
                                            range: 20-44
20250506_GRE2D_FATWATER_SUSHI_0012
                                            range: 16-35
                                            bad after 24 on the left
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from biceps_pipeline import load_dicom_images


PROJECT_ROOT = Path(__file__).parent
DICOM_ROOT   = PROJECT_ROOT / "DICOM_Files"
MASK_DIR     = PROJECT_ROOT / "masks_out"

# ── Echo-format settings (only used when MASK_DIR contains bf_masks_echo*.npy) ─
# Set DICOM_SERIES to the relative path under DICOM_ROOT, e.g.:
#   "20240709/GRE2D_FATWATER_WAYLON_0012"
# Set SLICE_OFFSET to the first DICOM slice the masks correspond to, e.g. 13.
DICOM_SERIES = "20240709/GRE2D_FATWATER_WAYLON_0012"
SLICE_OFFSET = 0


# ── Detect format and load masks ───────────────────────────────────────────────
classic_files = sorted(MASK_DIR.glob("*_mask.npy"))
echo_files    = sorted(MASK_DIR.glob("bf_masks_echo*.npy"))

if classic_files:
    # ── Classic format: one file per series ──────────────────────────────────
    available = [p.name.replace("_mask.npy", "") for p in classic_files]

    print("Available series:")
    for name in available:
        print(f"  {name}")

    name = input("\nEnter folder name: ").strip()

    if name not in available:
        print(f"\nNot found. Available series:")
        for a in available:
            print(f"  {a}")
        sys.exit(1)

    mask_path = MASK_DIR / f"{name}_mask.npy"
    date, subseries = name.split("_", 1)
    dicom_path = DICOM_ROOT / date / subseries

    if not dicom_path.exists():
        print(f"DICOM folder not found: {dicom_path}")
        sys.exit(1)

    print(f"\nLoading {name} ...")
    masks = np.load(mask_path)                      # (7, 50, H, W)
    imgs  = load_dicom_images(str(dicom_path))      # (7, 50, H, W)
    echo  = 0

elif echo_files:
    # ── Echo format: one file per echo, stack into (n_echoes, n_slices, H, W) ─
    echo_arrays = [np.load(p) for p in echo_files]  # each (n_slices, H, W)
    masks = np.stack(echo_arrays, axis=0)            # (n_echoes, n_slices, H, W)
    name  = MASK_DIR.name
    echo  = 0
    n_slices = masks.shape[1]

    if DICOM_SERIES:
        dicom_path = DICOM_ROOT / DICOM_SERIES
        if not dicom_path.exists():
            print(f"DICOM folder not found: {dicom_path}")
            sys.exit(1)
        print(f"\nLoading {name} ...")
        all_imgs = load_dicom_images(str(dicom_path))   # (7, 50, H, W)
        end = SLICE_OFFSET + n_slices
        imgs = all_imgs[:, SLICE_OFFSET:end, :, :]      # (7, n_slices, H, W)
        if imgs.shape[1] < n_slices:
            print(f"Warning: DICOM only has {all_imgs.shape[1]} slices; "
                  f"SLICE_OFFSET={SLICE_OFFSET} leaves only {imgs.shape[1]} slices.")
    else:
        print(f"\nLoading {name} (no DICOM — set DICOM_SERIES to show originals) ...")
        H, W = masks.shape[2], masks.shape[3]
        imgs  = np.zeros((masks.shape[0], n_slices, H, W), dtype=np.float32)

else:
    print(f"No masks found in {MASK_DIR}\n"
          f"  Expected '*_mask.npy' (classic) or 'bf_masks_echo*.npy' (echo format).")
    sys.exit(1)


# ── Build figure ───────────────────────────────────────────────────────────────
n_slices = masks.shape[1]
IMG_H, IMG_W = imgs.shape[2], imgs.shape[3]

fig, axes = plt.subplots(1, 3, figsize=(13, 5))
plt.subplots_adjust(bottom=0.12, top=0.88)

im_orig    = axes[0].imshow(imgs[echo, 0],  cmap="gray", vmin=0, vmax=1)
im_mask    = axes[1].imshow(masks[echo, 0], cmap="gray", vmin=0, vmax=1)
im_base    = axes[2].imshow(imgs[echo, 0],  cmap="gray", vmin=0, vmax=1)
im_overlay = axes[2].imshow(masks[echo, 0], cmap="Reds", alpha=0.45, vmin=0, vmax=1)

axes[0].set_title("Original")
axes[1].set_title("Mask")
axes[2].set_title("Overlay")
for ax in axes:
    ax.axis("off")

title = fig.suptitle(f"{name}  —  slice 0 / {n_slices - 1}", fontsize=11)
fig.text(0.5, 0.97, "Scroll to zoom  ·  R to reset", ha="center", va="top",
         fontsize=8, color="gray")

# ── Slider ─────────────────────────────────────────────────────────────────────
ax_slider = plt.axes([0.15, 0.03, 0.7, 0.03])
slider    = Slider(ax_slider, "Slice", 0, n_slices - 1, valinit=0, valstep=1)


def update(val):
    s    = int(slider.val)
    img  = imgs[echo, s]
    mask = masks[echo, s]

    im_orig.set_data(img)
    im_mask.set_data(mask)
    im_base.set_data(img)
    im_overlay.set_data(mask)

    title.set_text(f"{name}  —  slice {s} / {n_slices - 1}")
    fig.canvas.draw_idle()


slider.on_changed(update)


def on_scroll(event):
    if event.inaxes not in axes:
        return
    factor = 0.75 if event.button == "up" else 1.33
    cx, cy = event.xdata, event.ydata
    for ax in axes:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim([cx + (x - cx) * factor for x in xlim])
        ax.set_ylim([cy + (y - cy) * factor for y in ylim])
    fig.canvas.draw_idle()


def on_key(event):
    if event.key == "r":
        for ax in axes:
            ax.set_xlim(-0.5, IMG_W - 0.5)
            ax.set_ylim(IMG_H - 0.5, -0.5)
        fig.canvas.draw_idle()


fig.canvas.mpl_connect("scroll_event", on_scroll)
fig.canvas.mpl_connect("key_press_event", on_key)

plt.show()
