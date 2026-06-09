"""
mask_editor.py
--------------
Interactive mask editor for the biceps femoris segmentation pipeline.

Left-drag on the centre (Mask) panel to paint. Edits are mirrored to all 7
echoes so the mask volume stays consistent.

Controls:
  Slice slider    – navigate slices
  Left-drag       – paint (add or subtract depending on mode)
  A               – switch to ADD mode      (cyan brush)
  E               – switch to SUBTRACT/erase mode (red brush)
  [               – decrease brush radius by 1 px
  ]               – increase brush radius by 1 px
  Scroll wheel    – zoom all three panels simultaneously
  R               – reset zoom
  Ctrl+Z          – undo last stroke on the current slice
  Ctrl+S          – save mask (writes to edited_masks/, leaving masks_out/ untouched)
  Q               – quit (prompts if there are unsaved changes)

Usage:
    python mask_editor.py
"""

from pathlib import Path
import shutil
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Circle

from biceps_pipeline import load_dicom_images


PROJECT_ROOT = Path(__file__).parent
DICOM_ROOT = PROJECT_ROOT / "DICOM_Files"
MASK_DIR = PROJECT_ROOT / "masks_out"
EDITED_MASK_DIR = PROJECT_ROOT / "edited_masks"


# ── Backup prompt ─────────────────────────────────────────────────────────────
EDITED_MASK_DIR.mkdir(exist_ok=True)
source_masks = sorted(MASK_DIR.glob("*_mask.npy"))
uncopied = [f for f in source_masks if not (EDITED_MASK_DIR / f.name).exists()]
if uncopied:
    resp = (
        input(
            f"Copy {len(uncopied)} mask(s) from masks_out/ to edited_masks/ "
            f"to keep the originals as a backup? [Y/n]: "
        )
        .strip()
        .lower()
    )
    if resp in ("", "y"):
        for f in uncopied:
            shutil.copy2(f, EDITED_MASK_DIR / f.name)
        print(f"Copied {len(uncopied)} mask(s) to {EDITED_MASK_DIR}")


# ── Series selection ──────────────────────────────────────────────────────────
mask_files = sorted(MASK_DIR.glob("*_mask.npy"))
if not mask_files:
    print(f"No *_mask.npy files found in {MASK_DIR}")
    sys.exit(1)

available = [p.name.replace("_mask.npy", "") for p in mask_files]
print("Available series:")
for i, n in enumerate(available):
    print(f"  [{i:2d}] {n}")

choice = input("\nEnter folder name or index: ").strip()
if choice.isdigit():
    idx = int(choice)
    if not (0 <= idx < len(available)):
        print("Index out of range.")
        sys.exit(1)
    name = available[idx]
elif choice in available:
    name = choice
else:
    print("Not found.")
    sys.exit(1)

edited_path = EDITED_MASK_DIR / f"{name}_mask.npy"
mask_path = edited_path if edited_path.exists() else MASK_DIR / f"{name}_mask.npy"
save_path = edited_path  # always write to edited_masks/
date, subseries = name.split("_", 1)
dicom_path = DICOM_ROOT / date / subseries

if not dicom_path.exists():
    print(f"DICOM folder not found: {dicom_path}")
    sys.exit(1)

print(f"Loading {name} ...")
masks_orig = np.load(mask_path)  # (7, 50, H, W)  bool
imgs = load_dicom_images(str(dicom_path))  # (7, 50, H, W)  float32

masks = masks_orig.copy()
echo = 0
n_echoes = masks.shape[0]
n_slices = masks.shape[1]
IMG_H, IMG_W = imgs.shape[2], imgs.shape[3]

# Per-slice undo stack: each entry is a (n_echoes, H, W) snapshot
undo_stack = [[] for _ in range(n_slices)]

# Mutable editor state
state = {"mode": "add", "brush_r": 5, "painting": False, "cur_slice": 0}
dirty = False  # unsaved changes flag


# ── Figure ────────────────────────────────────────────────────────────────────
# Prevent matplotlib built-ins from shadowing our keys.
plt.rcParams["keymap.save"] = []  # was 's' / 'ctrl+s'
plt.rcParams["keymap.quit"] = []  # was 'q' / 'ctrl+w'

fig = plt.figure(figsize=(15, 6))

ax_orig = fig.add_axes([0.02, 0.14, 0.28, 0.76])
ax_mask = fig.add_axes([0.36, 0.14, 0.28, 0.76])
ax_over = fig.add_axes([0.70, 0.14, 0.28, 0.76])

for ax in (ax_orig, ax_mask, ax_over):
    ax.axis("off")

ax_slider = fig.add_axes([0.15, 0.05, 0.70, 0.03])
slider = Slider(ax_slider, "Slice", 0, n_slices - 1, valinit=0, valstep=1)

title = fig.suptitle("", fontsize=11, y=0.97)
status_txt = fig.text(
    0.5, 0.01, "", ha="center", va="bottom", fontsize=8, fontfamily="monospace"
)

im_orig = ax_orig.imshow(imgs[echo, 0], cmap="gray", vmin=0, vmax=1)
im_mask = ax_mask.imshow(masks[echo, 0].astype("float32"), cmap="gray", vmin=0, vmax=1)
im_base = ax_over.imshow(imgs[echo, 0], cmap="gray", vmin=0, vmax=1)
im_overlay = ax_over.imshow(
    np.ma.masked_where(masks[echo, 0] == 0, masks[echo, 0]), cmap="Reds", alpha=0.5, vmin=0, vmax=1
)

ax_orig.set_title("Original", fontsize=10, pad=4)
ax_mask.set_title("Mask  (edit)", fontsize=10, pad=4)
ax_over.set_title("Overlay", fontsize=10, pad=4)

# Brush cursor shown on the mask panel
brush_circle = Circle(
    (0, 0),
    radius=state["brush_r"],
    fill=False,
    color="cyan",
    linewidth=1.2,
    visible=False,
)
ax_mask.add_patch(brush_circle)

brush_circle_over = Circle(
    (0, 0),
    radius=state["brush_r"],
    fill=False,
    color="cyan",
    linewidth=1.2,
    visible=False,
)
ax_over.add_patch(brush_circle_over)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _status():
    add_mode = state["mode"] == "add"
    color = "cyan" if add_mode else "tomato"
    label = "ADD" if add_mode else "SUBTRACT"
    unsaved = "  *UNSAVED*" if dirty else ""
    status_txt.set_text(
        f"[{label}]  brush={state['brush_r']}px  |  "
        f"A=add  E=erase  [=smaller  ]=larger  Ctrl+Z=undo  Ctrl+S=save  R=zoom reset{unsaved}"
    )
    status_txt.set_color(color if dirty else "black")
    brush_circle.set_color(color)
    brush_circle_over.set_color(color)


def _redraw(s=None):
    if s is None:
        s = state["cur_slice"]
    m = masks[echo, s].astype("float32")
    im_orig.set_data(imgs[echo, s])
    im_mask.set_data(m)
    im_base.set_data(imgs[echo, s])
    im_overlay.set_data(np.ma.masked_where(m == 0, m))
    title.set_text(f"{name}  —  slice {s} / {n_slices - 1}")
    _status()
    fig.canvas.draw_idle()


def _paint(x, y):
    """Apply a circular brush at image-space coordinates (x, y)."""
    global dirty
    s = state["cur_slice"]
    r = state["brush_r"]
    xi, yi = int(round(x)), int(round(y))
    x0, x1 = max(0, xi - r), min(IMG_W, xi + r + 1)
    y0, y1 = max(0, yi - r), min(IMG_H, yi + r + 1)
    if x1 <= x0 or y1 <= y0:
        return
    gx, gy = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
    in_disk = (gx - xi) ** 2 + (gy - yi) ** 2 <= r**2
    rows, cols = np.where(in_disk)
    abs_y = rows + y0
    abs_x = cols + x0
    val = state["mode"] == "add"
    masks[:, s, abs_y, abs_x] = val  # update all echoes at once
    dirty = True
    _redraw(s)


def _save():
    global dirty
    np.save(save_path, masks)
    dirty = False
    print(f"Saved -> {save_path}")
    _status()
    fig.canvas.draw_idle()


# ── Event callbacks ───────────────────────────────────────────────────────────
def on_slider(val):
    state["cur_slice"] = int(val)
    _redraw()


def on_press(event):
    if event.inaxes not in (ax_mask, ax_over) or event.button != 1:
        return
    if event.xdata is None or event.ydata is None:
        return
    s = state["cur_slice"]
    # Push undo snapshot before the first pixel of a stroke
    undo_stack[s].append(masks[:, s].copy())
    if len(undo_stack[s]) > 30:
        undo_stack[s].pop(0)
    state["painting"] = True
    _paint(event.xdata, event.ydata)


def on_release(event):
    state["painting"] = False


def on_motion(event):
    on_mask = event.inaxes == ax_mask and event.xdata is not None
    on_over = event.inaxes == ax_over and event.xdata is not None

    if on_mask or on_over:
        circle = brush_circle if on_mask else brush_circle_over
        other  = brush_circle_over if on_mask else brush_circle
        circle.set_center((event.xdata, event.ydata))
        circle.set_visible(True)
        other.set_visible(False)
        if state["painting"]:
            _paint(event.xdata, event.ydata)
        else:
            fig.canvas.draw_idle()
    else:
        changed = brush_circle.get_visible() or brush_circle_over.get_visible()
        brush_circle.set_visible(False)
        brush_circle_over.set_visible(False)
        if changed:
            fig.canvas.draw_idle()


def on_key(event):
    global dirty
    s = state["cur_slice"]

    if event.key == "a":
        state["mode"] = "add"
        _status()
        fig.canvas.draw_idle()

    elif event.key == "e":
        state["mode"] = "subtract"
        _status()
        fig.canvas.draw_idle()

    elif event.key in ("[", "bracketleft"):
        state["brush_r"] = max(1, state["brush_r"] - 1)
        brush_circle.set_radius(state["brush_r"])
        brush_circle_over.set_radius(state["brush_r"])
        _status()
        fig.canvas.draw_idle()

    elif event.key in ("]", "bracketright"):
        state["brush_r"] = min(60, state["brush_r"] + 1)
        brush_circle.set_radius(state["brush_r"])
        brush_circle_over.set_radius(state["brush_r"])
        _status()
        fig.canvas.draw_idle()

    elif event.key == "ctrl+z":
        if undo_stack[s]:
            masks[:, s] = undo_stack[s].pop()
            dirty = True
            _redraw(s)

    elif event.key == "ctrl+s":
        _save()

    elif event.key == "r":
        for ax in (ax_orig, ax_mask, ax_over):
            ax.set_xlim(-0.5, IMG_W - 0.5)
            ax.set_ylim(IMG_H - 0.5, -0.5)
        fig.canvas.draw_idle()

    elif event.key == "q":
        if dirty:
            resp = (
                input("Unsaved changes. Save before quitting? [y/n/c=cancel]: ")
                .strip()
                .lower()
            )
            if resp == "y":
                _save()
            elif resp == "c":
                return
        plt.close("all")


def on_scroll(event):
    if event.inaxes not in (ax_orig, ax_mask, ax_over):
        return
    factor = 0.75 if event.button == "up" else 1.33
    cx, cy = event.xdata, event.ydata
    for ax in (ax_orig, ax_mask, ax_over):
        xl, yl = ax.get_xlim(), ax.get_ylim()
        ax.set_xlim([cx + (v - cx) * factor for v in xl])
        ax.set_ylim([cy + (v - cy) * factor for v in yl])
    fig.canvas.draw_idle()


# ── Wire events ───────────────────────────────────────────────────────────────
slider.on_changed(on_slider)
fig.canvas.mpl_connect("button_press_event", on_press)
fig.canvas.mpl_connect("button_release_event", on_release)
fig.canvas.mpl_connect("motion_notify_event", on_motion)
fig.canvas.mpl_connect("key_press_event", on_key)
fig.canvas.mpl_connect("scroll_event", on_scroll)

_redraw(0)
plt.show()
