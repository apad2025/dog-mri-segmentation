"""
mask_editor.py
--------------
Interactive ortho-slice mask editor for the biceps femoris segmentation
pipeline -- a viewer with linked XY and XZ panels.

The mask volume is (n_echo, Z, Y, X); every echo shares the same anatomy, so
edits are mirrored to all echoes automatically. The two panels are linked by a
crosshair:

    XY panel   shows imgs[echo, Z, :, :]        (one anatomical slice)
    XZ panel   shows imgs[echo, :, Y, :]         (a cross-section sweeping Z)

The XZ panel is the one your friend used to catch the mask bleeding into the
vastus lateralis around the middle slices -- now you can scrub it deliberately
and even paint directly in it.

Controls
  Left-drag           paint, in whichever panel the cursor is over
  Right-click/drag     move the crosshair (navigate): in XY it picks the XZ
                       row; in XZ it picks the XY slice
  Z slider            step through anatomical slices (XY)
  Y slider            step the XZ cross-section row
  A                   ADD mode        (cyan brush)
  E                   ERASE mode      (red brush)
  [ / ]               brush radius -1 / +1
  Scroll wheel        zoom the panel under the cursor
  R                   reset zoom on both panels
  Ctrl+Z              undo last stroke
  Ctrl+S              save to edited_masks/  (masks_out/ is left untouched)
  Q                   quit (prompts if unsaved)

Usage:
    python mask_editor.py
"""

from pathlib import Path
from collections import deque
import shutil
import sys

import numpy as np
import pydicom
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Circle


PROJECT_ROOT = Path(__file__).parent
DICOM_ROOT = PROJECT_ROOT / "DICOM_Files"
MASK_DIR = PROJECT_ROOT / "masks_out"
EDITED_MASK_DIR = PROJECT_ROOT / "edited_masks"


# ── Lightweight DICOM loader (pydicom only) ───────────────────────────────────
# Kept independent of biceps_pipeline so the editor never drags in skimage /
# scipy / segment_anything just to display the underlying images.
def load_dicom_images(
    folder, *, n_echo=7, n_slices=50, shape=(192, 192), dtype=np.float32
):
    """Return imgs of shape (n_echo, n_slices, H, W), normalized to [0, 1].

    Files are binned by their DICOM EchoNumbers tag, matching the pipeline.
    """
    folder = Path(folder)
    files = sorted(p for p in folder.iterdir() if p.is_file())
    expected = n_echo * n_slices
    if len(files) < expected:
        raise ValueError(
            f"Found {len(files)} files in {folder.name}, expected {expected}."
        )

    H, W = shape
    imgs = np.zeros((n_echo, n_slices, H, W), dtype=dtype)
    next_slice = np.zeros(n_echo, dtype=int)

    for p in files[:expected]:
        ds = pydicom.dcmread(str(p), force=True)
        en = getattr(ds, "EchoNumbers", None)
        if en is None:
            raise ValueError(f"Missing EchoNumbers in {p.name}")
        echo_idx = int(en) - 1
        if not (0 <= echo_idx < n_echo):
            raise ValueError(f"EchoNumbers={en} out of range in {p.name}")
        s = next_slice[echo_idx]
        if s >= n_slices:
            raise ValueError(f"Too many slices for echo {en} ({p.name})")
        imgs[echo_idx, s] = ds.pixel_array
        next_slice[echo_idx] += 1

    mn, mx = float(imgs.min()), float(imgs.max())
    if mx > mn:
        imgs = (imgs - mn) / (mx - mn)
    return imgs


def read_voxel_spacing(folder):
    """Return (sz, sy, sx) voxel spacing in mm from the first DICOM.

    The thigh volume is anisotropic -- in-plane ~2.08 mm but 3 mm between
    slices -- so the XZ panel needs a physical aspect ratio (sz / sx) to avoid
    looking stretched.  Falls back to isotropic 1 mm if tags are missing.
    """
    folder = Path(folder)
    files = sorted(p for p in folder.iterdir() if p.is_file())
    ds = pydicom.dcmread(str(files[0]), force=True)
    py, px = (float(v) for v in getattr(ds, "PixelSpacing", (1.0, 1.0)))
    sz = float(getattr(ds, "SpacingBetweenSlices", getattr(ds, "SliceThickness", 1.0)))
    return sz, py, px


# ── Series selection ──────────────────────────────────────────────────────────
def pick_series():
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
    date, subseries = name.split("_", 1)
    dicom_path = DICOM_ROOT / date / subseries
    if not dicom_path.exists():
        print(f"DICOM folder not found: {dicom_path}")
        sys.exit(1)

    return name, mask_path, edited_path, dicom_path


# ── Blit manager (figure-level, from the matplotlib blitting tutorial) ─────────
class BlitManager:
    """Smooth rendering via blitting, with a cleanly separated foreground.

    The foreground artists (mask overlays + brush cursor) are drawn ONLY through
    blitting, never baked into the cached background.  ``set_animated`` does not
    reliably exclude an AxesImage from a full draw, so we exclude the foreground
    the robust way: hide it while capturing the background, then blit it on top.
    This means the mask is drawn exactly once at its true opacity, and toggling
    its visibility makes it fully disappear.

      capture()  -- background changed (slice/row/zoom/text): redraw + regrab
      update()   -- foreground only changed (paint/cursor/toggle): fast blit
    """

    def __init__(self, canvas, fg_artists):
        self.canvas = canvas
        self._fg = list(fg_artists)
        self._bg = None
        canvas.mpl_connect("resize_event", lambda _e: self.capture())

    def capture(self):
        """Redraw the background with the foreground hidden, then re-blit it."""
        states = [a.get_visible() for a in self._fg]
        for a in self._fg:
            a.set_visible(False)
        self.canvas.draw()
        self._bg = self.canvas.copy_from_bbox(self.canvas.figure.bbox)
        for a, s in zip(self._fg, states):
            a.set_visible(s)
        self.update()

    def update(self):
        """Fast path: restore cached background, draw foreground, blit."""
        if self._bg is None:
            self.capture()
            return
        self.canvas.restore_region(self._bg)
        fig = self.canvas.figure
        for a in self._fg:
            if a.get_visible():
                fig.draw_artist(a)
        self.canvas.blit(fig.bbox)
        self.canvas.flush_events()


# ── Editor ────────────────────────────────────────────────────────────────────
class OrthoMaskEditor:
    def __init__(self, name, mask_path, save_path, dicom_path):
        self.name = name
        self.save_path = save_path

        print(f"Loading {name} ...")
        self.masks = np.load(mask_path)  # (E, Z, Y, X) bool
        self.imgs = load_dicom_images(str(dicom_path))  # (E, Z, Y, X) float

        # Physical aspect for the XZ panel (Z is vertical, X horizontal):
        # each Z step is `sz` mm tall, each X step `sx` mm wide.
        sz, _sy, sx = read_voxel_spacing(dicom_path)
        self.xz_aspect = sz / sx  # ~1.44 for this data

        self.echo = 0
        self.n_echo, self.nZ, self.H, self.W = self.masks.shape

        # Crosshair / cursor state
        self.z = 0  # XY slice (start at the first slice)
        self.y_row = self.H // 2  # XZ cross-section row
        self.x_col = self.W // 2

        self.mode = "add"
        self.brush_r = 5
        self.painting = False
        self.dirty = False
        self.mask_visible = True

        # Undo: each entry is {z_index: slice_copy} captured before a stroke.
        self.undo_stack = deque(maxlen=20)
        self._stroke_backup = None

        self._build_figure()

    # ── figure construction ──────────────────────────────────────────────────
    def _build_figure(self):
        plt.rcParams["keymap.save"] = []
        plt.rcParams["keymap.quit"] = []

        self.fig = plt.figure(figsize=(14, 7))
        self.ax_xy = self.fig.add_axes([0.04, 0.16, 0.44, 0.74])
        self.ax_xz = self.fig.add_axes([0.54, 0.16, 0.44, 0.74])
        for ax in (self.ax_xy, self.ax_xz):
            ax.set_xticks([])
            ax.set_yticks([])
        self.active_panel = self.ax_xy  # arrow keys act on this panel

        # XY panel ------------------------------------------------------------
        self.im_xy_base = self.ax_xy.imshow(
            self.imgs[self.echo, self.z], cmap="gray", vmin=0, vmax=1
        )
        self.im_xy_over = self.ax_xy.imshow(
            self._xy_mask(),
            cmap="Reds",
            alpha=0.5,
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        self.ax_xy.set_title("XY  ·  edit", fontsize=10)

        # XZ panel (rows = Z, cols = X) ---------------------------------------
        self.im_xz_base = self.ax_xz.imshow(
            self._xz_img(), cmap="gray", vmin=0, vmax=1, aspect=self.xz_aspect
        )
        self.im_xz_over = self.ax_xz.imshow(
            self._xz_mask(),
            cmap="Reds",
            alpha=0.5,
            vmin=0,
            vmax=1,
            aspect=self.xz_aspect,
            interpolation="nearest",
        )
        self.ax_xz.set_title("XZ  ·  cross-section", fontsize=10)

        # Crosshairs ----------------------------------------------------------
        ch = dict(color="yellow", lw=0.8, ls="--", alpha=0.7)
        self.xy_hline = self.ax_xy.axhline(self.y_row, **ch)  # XZ row marker
        self.xy_vline = self.ax_xy.axvline(self.x_col, **ch)
        self.xz_hline = self.ax_xz.axhline(self.z, **ch)  # XY slice marker
        self.xz_vline = self.ax_xz.axvline(self.x_col, **ch)

        # Brush cursors (one per panel) ---------------------------------------
        self.brush_xy = Circle(
            (0, 0), self.brush_r, fill=False, color="cyan", lw=1.2, visible=False
        )
        self.brush_xz = Circle(
            (0, 0), self.brush_r, fill=False, color="cyan", lw=1.2, visible=False
        )
        self.ax_xy.add_patch(self.brush_xy)
        self.ax_xz.add_patch(self.brush_xz)

        # Sliders -------------------------------------------------------------
        ax_z = self.fig.add_axes([0.10, 0.075, 0.38, 0.03])
        ax_y = self.fig.add_axes([0.60, 0.075, 0.38, 0.03])
        self.s_z = Slider(ax_z, "Z slice", 0, self.nZ - 1, valinit=self.z, valstep=1)
        self.s_y = Slider(ax_y, "Y row", 0, self.H - 1, valinit=self.y_row, valstep=1)
        self.s_z.on_changed(self._on_z_slider)
        self.s_y.on_changed(self._on_y_slider)

        self._highlight_panels()

        self.title = self.fig.suptitle("", fontsize=11, y=0.97)
        self.status = self.fig.text(
            0.5, 0.015, "", ha="center", va="bottom", fontsize=8, fontfamily="monospace"
        )

        # Animated artists -> handled by the blit manager.
        # Foreground = mask overlays + brush cursors; drawn only via blit.
        self.bm = BlitManager(
            self.fig.canvas,
            [self.im_xy_over, self.im_xz_over, self.brush_xy, self.brush_xz],
        )

        # Events --------------------------------------------------------------
        c = self.fig.canvas
        c.mpl_connect("button_press_event", self._on_press)
        c.mpl_connect("button_release_event", self._on_release)
        c.mpl_connect("motion_notify_event", self._on_motion)
        c.mpl_connect("key_press_event", self._on_key)
        c.mpl_connect("scroll_event", self._on_scroll)

        self._structural_update()

    # ── cross-section helpers ────────────────────────────────────────────────
    def _xy_mask(self):
        m = self.masks[self.echo, self.z].astype("float32")
        return np.ma.masked_where(m == 0, m)

    def _xz_img(self):
        return self.imgs[self.echo, :, self.y_row, :]

    def _xz_mask(self):
        m = self.masks[self.echo, :, self.y_row, :].astype("float32")
        return np.ma.masked_where(m == 0, m)

    # ── drawing ──────────────────────────────────────────────────────────────
    def _structural_update(self):
        """Full redraw: base images / crosshairs / titles changed."""
        self.im_xy_base.set_data(self.imgs[self.echo, self.z])
        self.im_xz_base.set_data(self._xz_img())
        self.im_xy_over.set_data(self._xy_mask())
        self.im_xz_over.set_data(self._xz_mask())

        self.xy_hline.set_ydata([self.y_row, self.y_row])
        self.xy_vline.set_xdata([self.x_col, self.x_col])
        self.xz_hline.set_ydata([self.z, self.z])
        self.xz_vline.set_xdata([self.x_col, self.x_col])

        self.title.set_text(
            f"{self.name}    Z={self.z}/{self.nZ - 1}   Y={self.y_row}/{self.H - 1}"
        )
        self._set_status()
        self.bm.capture()

    def _overlay_update(self):
        """Fast path: only the mask overlays / brush changed."""
        self.im_xy_over.set_data(self._xy_mask())
        self.im_xz_over.set_data(self._xz_mask())
        self.bm.update()

    def _highlight_panels(self):
        """Draw a bright border on the active panel (the arrow-key target)."""
        for ax in (self.ax_xy, self.ax_xz):
            on = ax is self.active_panel
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_color("deepskyblue" if on else "0.4")
                sp.set_linewidth(2.4 if on else 0.8)

    def _set_status(self):
        add = self.mode == "add"
        color = "cyan" if add else "tomato"
        flag = "  *UNSAVED*" if self.dirty else ""
        self.status.set_text(
            f"[{'ADD' if add else 'ERASE'}]  brush={self.brush_r}px   "
            f"L-drag=paint  R-click=move crosshair  Up/Dn=step active panel  "
            f"A/E=add/erase  [ ]=size  V=toggle mask  Ctrl+Z=undo  "
            f"Ctrl+S=save  R=reset  Q=quit{flag}"
        )
        self.status.set_color("red" if self.dirty else "black")
        self.brush_xy.set_color(color)
        self.brush_xz.set_color(color)

    # ── painting ─────────────────────────────────────────────────────────────
    def _backup_slice(self, z):
        if self._stroke_backup is not None and z not in self._stroke_backup:
            self._stroke_backup[z] = self.masks[:, z].copy()

    def _paint_xy(self, x, y):
        xi, yi = int(round(x)), int(round(y))
        r = self.brush_r
        x0, x1 = max(0, xi - r), min(self.W, xi + r + 1)
        y0, y1 = max(0, yi - r), min(self.H, yi + r + 1)
        if x1 <= x0 or y1 <= y0:
            return
        gx, gy = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
        disk = (gx - xi) ** 2 + (gy - yi) ** 2 <= r**2
        self._backup_slice(self.z)
        ys, xs = np.where(disk)
        self.masks[:, self.z, ys + y0, xs + x0] = self.mode == "add"
        self.dirty = True

    def _paint_xz(self, x, z):
        """Paint a disk in (X, Z) at the current Y row -- edits multiple slices."""
        xi, zi = int(round(x)), int(round(z))
        r = self.brush_r
        x0, x1 = max(0, xi - r), min(self.W, xi + r + 1)
        z0, z1 = max(0, zi - r), min(self.nZ, zi + r + 1)
        if x1 <= x0 or z1 <= z0:
            return
        gx, gz = np.meshgrid(np.arange(x0, x1), np.arange(z0, z1))
        disk = (gx - xi) ** 2 + (gz - zi) ** 2 <= r**2
        zs, xs = np.where(disk)
        val = self.mode == "add"
        for zz in range(z0, z1):
            self._backup_slice(zz)
        self.masks[:, zs + z0, self.y_row, xs + x0] = val
        self.dirty = True

    # ── events ───────────────────────────────────────────────────────────────
    def _on_press(self, event):
        if event.inaxes not in (self.ax_xy, self.ax_xz):
            return
        if event.xdata is None or event.ydata is None:
            return

        if event.button == 3:  # right-click -> navigate
            self._navigate(event)
            return
        if event.button != 1:
            return

        self._stroke_backup = {}
        self.painting = event.inaxes
        if event.inaxes is self.ax_xy:
            self._paint_xy(event.xdata, event.ydata)
        else:
            self._paint_xz(event.xdata, event.ydata)
        self._overlay_update()

    def _on_release(self, _event):
        if self.painting and self._stroke_backup:
            self.undo_stack.append(self._stroke_backup)
        self._stroke_backup = None
        self.painting = False
        self._set_status()
        self.bm.capture()

    def _on_motion(self, event):
        ax = event.inaxes
        if ax not in (self.ax_xy, self.ax_xz) or event.xdata is None:
            if self.brush_xy.get_visible() or self.brush_xz.get_visible():
                self.brush_xy.set_visible(False)
                self.brush_xz.set_visible(False)
                self.bm.update()
            return

        if ax is self.ax_xy:
            self.brush_xy.set_center((event.xdata, event.ydata))
            self.brush_xy.set_visible(True)
            self.brush_xz.set_visible(False)
        else:
            self.brush_xz.set_center((event.xdata, event.ydata))
            self.brush_xz.set_visible(True)
            self.brush_xy.set_visible(False)

        # Focus-follows-mouse: entering a panel makes it the arrow-key target.
        if not self.painting and ax is not self.active_panel:
            self.active_panel = ax
            self._highlight_panels()
            self.bm.capture()  # bake new border into background
            return

        if self.painting is self.ax_xy and ax is self.ax_xy:
            self._paint_xy(event.xdata, event.ydata)
            self._overlay_update()
        elif self.painting is self.ax_xz and ax is self.ax_xz:
            self._paint_xz(event.xdata, event.ydata)
            self._overlay_update()
        else:
            self.bm.update()

    def _navigate(self, event):
        """Right-click moves the crosshair, linking the two views."""
        self.x_col = int(round(np.clip(event.xdata, 0, self.W - 1)))
        if event.inaxes is self.ax_xy:
            self.y_row = int(round(np.clip(event.ydata, 0, self.H - 1)))
            self.s_y.set_val(self.y_row)  # triggers structural update
        else:
            self.z = int(round(np.clip(event.ydata, 0, self.nZ - 1)))
            self.s_z.set_val(self.z)
        self._structural_update()

    def _on_z_slider(self, val):
        self.z = int(val)
        self._structural_update()

    def _on_y_slider(self, val):
        self.y_row = int(val)
        self._structural_update()

    def _on_key(self, event):
        if event.key == "a":
            self.mode = "add"
            self._set_status()
            self.bm.capture()
        elif event.key == "e":
            self.mode = "subtract"
            self._set_status()
            self.bm.capture()
        elif event.key in ("up", "down"):
            step = 1 if event.key == "up" else -1
            if self.active_panel is self.ax_xz:
                self.s_y.set_val(int(np.clip(self.y_row + step, 0, self.H - 1)))
            else:
                self.s_z.set_val(int(np.clip(self.z + step, 0, self.nZ - 1)))
        elif event.key in ("[", "bracketleft"):
            self.brush_r = max(1, self.brush_r - 1)
            self._sync_brush()
        elif event.key in ("]", "bracketright"):
            self.brush_r = min(60, self.brush_r + 1)
            self._sync_brush()
        elif event.key == "v":
            self.mask_visible = not self.mask_visible
            self.im_xy_over.set_visible(self.mask_visible)
            self.im_xz_over.set_visible(self.mask_visible)
            self.bm.update()
        elif event.key == "ctrl+z":
            self._undo()
        elif event.key == "ctrl+s":
            self._save()
        elif event.key == "r":
            self.ax_xy.set_xlim(-0.5, self.W - 0.5)
            self.ax_xy.set_ylim(self.H - 0.5, -0.5)
            self.ax_xz.set_xlim(-0.5, self.W - 0.5)
            self.ax_xz.set_ylim(self.nZ - 0.5, -0.5)
            self.bm.capture()
        elif event.key == "q":
            self._quit()

    def _sync_brush(self):
        self.brush_xy.set_radius(self.brush_r)
        self.brush_xz.set_radius(self.brush_r)
        self._set_status()
        self.bm.capture()

    def _on_scroll(self, event):
        ax = event.inaxes
        if ax not in (self.ax_xy, self.ax_xz):
            return
        factor = 0.75 if event.button == "up" else 1.33
        cx, cy = event.xdata, event.ydata
        ax.set_xlim([cx + (v - cx) * factor for v in ax.get_xlim()])
        ax.set_ylim([cy + (v - cy) * factor for v in ax.get_ylim()])
        self.bm.capture()

    # ── actions ──────────────────────────────────────────────────────────────
    def _undo(self):
        if not self.undo_stack:
            return
        backup = self.undo_stack.pop()
        for z, slc in backup.items():
            self.masks[:, z] = slc
        self.dirty = True
        self._structural_update()

    def _save(self):
        np.save(self.save_path, self.masks)
        self.dirty = False
        print(f"Saved -> {self.save_path}")
        self._set_status()
        self.bm.capture()

    def _quit(self):
        if self.dirty:
            resp = (
                input("Unsaved changes. Save before quitting? [y/n/c=cancel]: ")
                .strip()
                .lower()
            )
            if resp == "y":
                self._save()
            elif resp == "c":
                return
        plt.close("all")

    def run(self):
        plt.show()


if __name__ == "__main__":
    name, mask_path, save_path, dicom_path = pick_series()
    OrthoMaskEditor(name, mask_path, save_path, dicom_path).run()
