"""
mask_editor.py
--------------
Interactive ortho-slice mask editor (PyQtGraph).
A viewer with linked XY and XZ panels, GPU-friendly and far more responsive
than the previous matplotlib version (kept as mask_editor_mpl.py).

The mask volume is (n_echo, Z, Y, X); every echo shares the same anatomy, so
edits are mirrored to all echoes automatically. The two panels are linked by a
crosshair:

    XY panel   shows imgs[echo, Z, :, :]         (one anatomical slice)
    XZ panel   shows imgs[echo, :, Y, :]         (a cross-section sweeping Z)

Controls
  Left-drag           paint, in whichever panel the cursor is over
  Right-click/drag     move the crosshair (navigate): in XY it picks the XZ
                       row; in XZ it picks the XY slice
  Middle-drag         pan the panel
  Z slider            step through anatomical slices (XY)
  Y slider            step the XZ cross-section row
  Up/Down             step the active (hovered) panel
  A                   ADD mode        (cyan brush)
  E                   ERASE mode      (red brush)
  [ / ]               brush radius -1 / +1
  Scroll wheel        zoom the panel under the cursor
  R                   reset zoom on both panels
  C                   copy previous Z slice's mask onto the current slice
  V                   toggle mask overlay
  Ctrl+Z              undo last stroke
  Ctrl+S              save to edited_masks/  (masks_out/ is left untouched)
  Q                   quit (prompts if unsaved)

Water/fat-separated series share one mask. A series that has been run through
water_fat_separation/wfs_to_mask_editor.py shows up three times -- the unaltered
series plus its "..._WATER" and "..._FAT" variants -- and you can open any of
them to segment on whichever image is clearest. The anatomy (and therefore the
mask) is identical across the three, so all three edit and save the single base
"<series>_mask.npy"; the per-variant "..._WATER_mask.npy" / "..._FAT_mask.npy"
files are left untouched.

Images and masks are expected to already be 2x Fourier-upscaled on disk by
fourier_upscale.py (run once): the upscaled DICOMs live in
DICOM_Files_upscaled/ and the masks are pixel-doubled in place. Because both
now share the same fine grid, the editor loads them directly with no per-open
upscaling and the overlay aligns 1:1 with the base image. If the upscaled
DICOM mirror is missing, the editor falls back to the original DICOM_Files/.

Usage:
    python mask_editor.py
"""

from pathlib import Path
from collections import deque
import shutil
import sys

import numpy as np
import pydicom
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

pg.setConfigOptions(imageAxisOrder="row-major")


PROJECT_ROOT = Path(__file__).parent
DICOM_ROOT = PROJECT_ROOT / "DICOM_Files"
UPSCALED_DICOM_ROOT = PROJECT_ROOT / "DICOM_Files_upscaled"
MASK_DIR = PROJECT_ROOT / "masks_out"
EDITED_MASK_DIR = PROJECT_ROOT / "edited_masks"

# Two-entry lookup table for the mask overlay: 0 -> transparent, 1 -> red.
MASK_LUT = np.array([[0, 0, 0, 0], [220, 40, 40, 130]], dtype=np.uint8)


# ── Lightweight DICOM loader (pydicom only) ───────────────────────────────────
def load_dicom_images(folder, *, n_echo=7, n_slices=50, dtype=np.float32):
    """Return imgs of shape (n_echo, n_slices, H, W), normalized to [0, 1].

    Files are binned by their DICOM EchoNumbers tag, matching the pipeline. The
    in-plane size (H, W) is read from the data, so the upscaled mirror (384x384)
    and the original DICOMs (192x192) both load without changes here.
    """
    folder = Path(folder)
    files = sorted(p for p in folder.iterdir() if p.is_file())
    expected = n_echo * n_slices
    if len(files) < expected:
        raise ValueError(
            f"Found {len(files)} files in {folder.name}, expected {expected}."
        )

    H, W = pydicom.dcmread(str(files[0]), force=True).pixel_array.shape
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
# Suffixes added by wfs_to_mask_editor.py for the water/fat-separated variants.
WFS_SUFFIXES = ("_WATER", "_FAT")


def canonical_mask_name(name):
    """Map a series variant to the series that owns its editable mask.

    Water/fat separation produces three views of one anatomy -- the unaltered
    series plus ``..._WATER`` / ``..._FAT`` -- and the mask is identical across
    all three. So the variants share the base series' mask: editing any of them
    reads and writes the single ``<base>_mask.npy``, and the per-variant
    ``..._WATER_mask.npy`` / ``..._FAT_mask.npy`` files are never touched.
    """
    for suffix in WFS_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def pick_series():
    EDITED_MASK_DIR.mkdir(exist_ok=True)
    mask_files = sorted(MASK_DIR.glob("*_mask.npy"))
    if not mask_files:
        print(f"No *_mask.npy files found in {MASK_DIR}")
        sys.exit(1)

    # List every base series (those with a mask) and, right after each, any of
    # its water/fat-separated variants. A variant is openable whenever its
    # upscaled DICOM folder exists -- it needs no mask file of its own, since it
    # shares the base series' mask (see canonical_mask_name).
    available = []
    for base in (p.name.replace("_mask.npy", "") for p in mask_files):
        available.append(base)
        date, subseries = base.split("_", 1)
        for suffix in WFS_SUFFIXES:
            if (UPSCALED_DICOM_ROOT / date / f"{subseries}{suffix}").exists():
                available.append(f"{base}{suffix}")

    # Only the canonical (base) masks are editable; back those up to
    # edited_masks/ and leave the redundant water/fat copies alone.
    canonical_files = sorted(
        {MASK_DIR / f"{canonical_mask_name(n)}_mask.npy" for n in available}
    )
    canonical_files = [f for f in canonical_files if f.exists()]
    uncopied = [f for f in canonical_files if not (EDITED_MASK_DIR / f.name).exists()]
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

    print("Available series:")
    for i, n in enumerate(available):
        cn = canonical_mask_name(n)
        note = f"   (shares {cn} mask)" if cn != n else ""
        print(f"  [{i:2d}] {n}{note}")

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

    # The image comes from the chosen variant, but the mask is the shared base
    # one. Fall back to the variant's own mask if the base mask is missing.
    mask_name = canonical_mask_name(name)
    edited_path = EDITED_MASK_DIR / f"{mask_name}_mask.npy"
    if not edited_path.exists() and not (MASK_DIR / f"{mask_name}_mask.npy").exists():
        mask_name = name
        edited_path = EDITED_MASK_DIR / f"{mask_name}_mask.npy"
    mask_path = edited_path if edited_path.exists() else MASK_DIR / f"{mask_name}_mask.npy"
    date, subseries = name.split("_", 1)
    # Prefer the upscaled mirror (fourier_upscale.py); fall back to originals.
    dicom_path = UPSCALED_DICOM_ROOT / date / subseries
    if not dicom_path.exists():
        dicom_path = DICOM_ROOT / date / subseries
        print(
            f"Note: upscaled DICOMs not found; using originals at {dicom_path}.\n"
            f"      Run 'python fourier_upscale.py' for the sharper, "
            f"resolution-matched view."
        )
    if not dicom_path.exists():
        print(f"DICOM folder not found: {dicom_path}")
        sys.exit(1)

    return name, mask_path, edited_path, dicom_path


# ── Painting helper ───────────────────────────────────────────────────────────
def disk_indices(x, y, r, width, height):
    """Index arrays (rows, cols) covering a radius-`r` disk centered at (x, y),
    clipped to a (height, width) grid.  Returns None if fully outside."""
    xi, yi = int(round(x)), int(round(y))
    x0, x1 = max(0, xi - r), min(width, xi + r + 1)
    y0, y1 = max(0, yi - r), min(height, yi + r + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    gx, gy = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
    inside = (gx - xi) ** 2 + (gy - yi) ** 2 <= r**2
    rows, cols = np.nonzero(inside)
    return rows + y0, cols + x0


# ── ViewBox with paint / navigate mouse bindings ──────────────────────────────
class PanelViewBox(pg.ViewBox):
    """ViewBox where left-drag paints and right-click/drag navigates.

    The editor assigns `on_paint(x, y)`, `on_stroke_start()`, `on_stroke_end()`
    and `on_navigate(x, y)` after construction.  Wheel zoom is inherited;
    middle-drag falls through to the default pan behavior.
    """

    def __init__(self):
        super().__init__(invertY=True, enableMenu=False)
        self.on_paint = None
        self.on_stroke_start = None
        self.on_stroke_end = None
        self.on_navigate = None

    def _view_pos(self, ev):
        p = self.mapSceneToView(ev.scenePos())
        return p.x(), p.y()

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            ev.accept()
            self.on_stroke_start()
            self.on_paint(*self._view_pos(ev))
            self.on_stroke_end()
        elif ev.button() == QtCore.Qt.MouseButton.RightButton:
            ev.accept()
            self.on_navigate(*self._view_pos(ev))
        else:
            super().mouseClickEvent(ev)

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            ev.accept()
            if ev.isStart():
                self.on_stroke_start()
            self.on_paint(*self._view_pos(ev))
            if ev.isFinish():
                self.on_stroke_end()
        elif ev.button() == QtCore.Qt.MouseButton.RightButton:
            ev.accept()
            self.on_navigate(*self._view_pos(ev))
        else:
            super().mouseDragEvent(ev, axis=axis)


# ── Panel: one orthogonal view ────────────────────────────────────────────────
class Panel:
    """Base image + mask overlay + crosshair + brush cursor in one ViewBox.

    The editor supplies data accessors (`get_base`, `get_mask`) and a `paint`
    callback, so the panel is agnostic about which mask axes it cuts through;
    `n_rows`/`n_cols` are the panel's extent in mask pixels (vertical axis is
    Y for the XY panel and Z for the XZ panel).
    """

    ACTIVE_PEN = pg.mkPen("deepskyblue", width=2)
    INACTIVE_PEN = pg.mkPen((110, 110, 110), width=1)

    def __init__(
        self,
        vb,
        *,
        get_base,
        get_mask,
        paint,
        n_rows,
        n_cols,
        base_rect,
        aspect=1.0,
        brush_r=5,
    ):
        self.vb = vb
        self.get_base = get_base
        self.get_mask = get_mask
        self.paint = paint
        self.n_rows = n_rows
        self.n_cols = n_cols

        # pyqtgraph's ratio is (pixels per x-unit) / (pixels per y-unit), so a
        # Z step that is physically `aspect` times taller than an X step needs
        # ratio = 1 / aspect.
        vb.setAspectLocked(True, ratio=1.0 / aspect)

        self.im_base = pg.ImageItem()
        vb.addItem(self.im_base)

        # Overlay lives on the native mask grid (pixel centers at integers).
        self.im_over = pg.ImageItem()
        self.im_over.setZValue(1)
        vb.addItem(self.im_over)

        ch_pen = pg.mkPen("yellow", width=1, style=QtCore.Qt.PenStyle.DashLine)
        self.hline = pg.InfiniteLine(angle=0, movable=False, pen=ch_pen)
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=ch_pen)
        for line in (self.hline, self.vline):
            line.setZValue(2)
            vb.addItem(line, ignoreBounds=True)

        self.brush = QtWidgets.QGraphicsEllipseItem()
        self.brush.setZValue(3)
        self.brush.setVisible(False)
        vb.addItem(self.brush, ignoreBounds=True)
        self._brush_r = brush_r
        self._brush_pos = (0.0, 0.0)
        self.set_brush(color="cyan")

        self.refresh()
        # setRect must come after the images are assigned -- on an empty
        # ImageItem it computes the scale against a 1x1 placeholder size.
        self.im_base.setRect(base_rect)
        self.im_over.setRect(QtCore.QRectF(-0.5, -0.5, n_cols, n_rows))
        self.reset_zoom()

    def refresh(self):
        self.im_base.setImage(self.get_base(), autoLevels=False, levels=(0.0, 1.0))
        self.refresh_mask()

    def refresh_mask(self):
        self.im_over.setImage(
            self.get_mask().astype(np.uint8),
            autoLevels=False,
            levels=(0, 1),
            lut=MASK_LUT,
        )

    def set_crosshair(self, x, y):
        self.hline.setPos(y)
        self.vline.setPos(x)

    def reset_zoom(self):
        self.vb.setRange(
            xRange=(-0.5, self.n_cols - 0.5),
            yRange=(-0.5, self.n_rows - 0.5),
            padding=0,
        )

    def set_active(self, on):
        """Bright border marks the active panel (the arrow-key target)."""
        self.vb.setBorder(self.ACTIVE_PEN if on else self.INACTIVE_PEN)

    def set_brush(self, radius=None, color=None):
        if radius is not None:
            self._brush_r = radius
        if color is not None:
            self.brush.setPen(pg.mkPen(color, width=1.5, cosmetic=True))
        self.move_brush(*self._brush_pos)

    def move_brush(self, x, y):
        self._brush_pos = (x, y)
        r = self._brush_r
        self.brush.setRect(QtCore.QRectF(x - r, y - r, 2 * r, 2 * r))


# ── Editor ────────────────────────────────────────────────────────────────────
class OrthoMaskEditor(QtWidgets.QMainWindow):
    def __init__(self, name, mask_path, save_path, dicom_path):
        super().__init__()
        self.name = name
        self.save_path = save_path

        print(f"Loading {name} ...")
        self.masks = np.load(mask_path)  # (E, Z, Y, X) bool
        self.imgs = load_dicom_images(str(dicom_path))  # (E, Z, Y, X) float

        if self.imgs.shape[2:] != self.masks.shape[2:]:
            raise ValueError(
                f"Image grid {self.imgs.shape[2:]} != mask grid "
                f"{self.masks.shape[2:]}. Run 'python fourier_upscale.py' so "
                f"both share the same upscaled grid."
            )

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
        self.painting = None  # Panel being painted, or None
        self.dirty = False
        self.mask_visible = True

        # Undo: each entry is {z_index: slice_copy} captured before a stroke.
        self.undo_stack = deque(maxlen=20)
        self._stroke_backup = None

        self._build_ui()
        self._structural_update()

    # ── UI construction ──────────────────────────────────────────────────────
    def _build_ui(self):
        self.resize(1400, 760)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        self.setCentralWidget(central)

        self.gw = pg.GraphicsLayoutWidget()
        layout.addWidget(self.gw, stretch=1)
        self.gw.addLabel("XY  ·  edit", row=0, col=0)
        self.gw.addLabel("XZ  ·  cross-section", row=0, col=1)

        # Images and masks share one grid (both upscaled on disk), so the base
        # image, overlay, painting, crosshairs and sliders all use the same
        # pixel coordinates -- pixel centers at integers, rect offset by -0.5.
        vb_xy, vb_xz = PanelViewBox(), PanelViewBox()
        self.gw.addItem(vb_xy, row=1, col=0)
        self.gw.addItem(vb_xz, row=1, col=1)

        self.panel_xy = Panel(
            vb_xy,
            get_base=lambda: self.imgs[self.echo, self.z],
            get_mask=lambda: self.masks[self.echo, self.z],
            paint=self._paint_xy,
            n_rows=self.H,
            n_cols=self.W,
            base_rect=QtCore.QRectF(-0.5, -0.5, self.W, self.H),
            brush_r=self.brush_r,
        )
        # XZ panel: rows = Z (never upscaled), cols = X.
        self.panel_xz = Panel(
            vb_xz,
            get_base=lambda: self.imgs[self.echo, :, self.y_row, :],
            get_mask=lambda: self.masks[self.echo, :, self.y_row, :],
            paint=self._paint_xz,
            n_rows=self.nZ,
            n_cols=self.W,
            base_rect=QtCore.QRectF(-0.5, -0.5, self.W, self.nZ),
            aspect=self.xz_aspect,
            brush_r=self.brush_r,
        )
        self.panels = (self.panel_xy, self.panel_xz)
        self.active_panel = self.panel_xy  # arrow keys act on this panel

        for panel in self.panels:
            vb = panel.vb
            vb.on_paint = panel.paint
            vb.on_stroke_start = lambda p=panel: self._stroke_start(p)
            vb.on_stroke_end = self._stroke_end
            vb.on_navigate = lambda x, y, p=panel: self._navigate(p, x, y)

        # Sliders -------------------------------------------------------------
        def make_slider(text, maximum, value, handler):
            box = QtWidgets.QHBoxLayout()
            box.addWidget(QtWidgets.QLabel(text))
            s = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            s.setRange(0, maximum)
            s.setValue(value)
            s.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
            s.valueChanged.connect(handler)
            box.addWidget(s, stretch=1)
            label = QtWidgets.QLabel()
            label.setMinimumWidth(60)
            box.addWidget(label)
            return s, label, box

        sliders = QtWidgets.QHBoxLayout()
        self.s_z, self.lab_z, box_z = make_slider(
            "Z slice", self.nZ - 1, self.z, self._on_z_slider
        )
        self.s_y, self.lab_y, box_y = make_slider(
            "Y row", self.H - 1, self.y_row, self._on_y_slider
        )
        sliders.addLayout(box_z, stretch=1)
        sliders.addSpacing(20)
        sliders.addLayout(box_y, stretch=1)
        layout.addLayout(sliders)

        self.status = QtWidgets.QLabel()
        self.status.setStyleSheet("font-family: monospace; font-size: 8pt;")
        self.status.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.statusBar().addWidget(self.status, 1)  # stretch to fill, center text

        # Brush tracking + focus-follows-mouse across both panels.
        self.gw.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.gw.installEventFilter(self)

        self._highlight_panels()

        # Keyboard ------------------------------------------------------------
        for keys, cb in [
            ("A", lambda: self._set_mode("add")),
            ("E", lambda: self._set_mode("subtract")),
            ("[", lambda: self._set_brush_r(self.brush_r - 1)),
            ("]", lambda: self._set_brush_r(self.brush_r + 1)),
            ("C", self._copy_prev_frame),
            ("V", self._toggle_mask),
            ("R", self._reset_zoom),
            ("Q", self.close),
            ("Ctrl+Z", self._undo),
            ("Ctrl+S", self._save),
            ("Up", lambda: self._step_active(+1)),
            ("Down", lambda: self._step_active(-1)),
        ]:
            QtGui.QShortcut(QtGui.QKeySequence(keys), self).activated.connect(cb)

    def eventFilter(self, obj, event):
        if obj is self.gw and event.type() == QtCore.QEvent.Type.Leave:
            self._hide_brushes()
        return super().eventFilter(obj, event)

    def _panel_at(self, scene_pos):
        """The Panel under the scene position, or None."""
        for p in self.panels:
            if p.vb.sceneBoundingRect().contains(scene_pos):
                return p
        return None

    # ── drawing ──────────────────────────────────────────────────────────────
    def _structural_update(self):
        """Base images / crosshairs / titles changed."""
        for p in self.panels:
            p.refresh()
        self.panel_xy.set_crosshair(self.x_col, self.y_row)  # XZ row marker
        self.panel_xz.set_crosshair(self.x_col, self.z)  # XY slice marker

        self.lab_z.setText(f"{self.z}/{self.nZ - 1}")
        self.lab_y.setText(f"{self.y_row}/{self.H - 1}")
        self._set_status()

    def _overlay_update(self):
        for p in self.panels:
            p.refresh_mask()

    def _highlight_panels(self):
        for p in self.panels:
            p.set_active(p is self.active_panel)

    def _set_status(self):
        add = self.mode == "add"
        flag = "  *UNSAVED*" if self.dirty else ""
        self.setWindowTitle(
            f"{self.name}    Z={self.z}/{self.nZ - 1}   "
            f"Y={self.y_row}/{self.H - 1}{flag}"
        )
        self.status.setText(
            f"[{'ADD' if add else 'ERASE'}]  brush={self.brush_r}px   "
            f"L-drag=paint  R-click=move crosshair  Up/Dn=step active panel  "
            f"A/E=add/erase  [ ]=size  C=copy prev slice  V=toggle mask  "
            f"Ctrl+Z=undo  Ctrl+S=save  R=reset  Q=quit{flag}"
        )
        self.status.setStyleSheet(
            "font-family: monospace; font-size: 8pt; "
            f"color: {'red' if self.dirty else 'palette(window-text)'};"
        )
        for p in self.panels:
            p.set_brush(color="cyan" if add else "tomato")

    # ── painting ─────────────────────────────────────────────────────────────
    def _backup_slice(self, z):
        if self._stroke_backup is not None and z not in self._stroke_backup:
            self._stroke_backup[z] = self.masks[:, z].copy()

    def _stroke_start(self, panel):
        self._stroke_backup = {}
        self.painting = panel

    def _stroke_end(self):
        if self._stroke_backup:
            self.undo_stack.append(self._stroke_backup)
        self._stroke_backup = None
        self.painting = None
        self._set_status()

    def _paint_xy(self, x, y):
        hit = disk_indices(x, y, self.brush_r, self.W, self.H)
        if hit is None:
            return
        rows, cols = hit
        self._backup_slice(self.z)
        self.masks[:, self.z, rows, cols] = self.mode == "add"
        self.dirty = True
        self._overlay_update()

    def _paint_xz(self, x, z):
        """Paint a disk in (X, Z) at the current Y row -- edits multiple slices."""
        hit = disk_indices(x, z, self.brush_r, self.W, self.nZ)
        if hit is None:
            return
        rows, cols = hit
        for zz in np.unique(rows):
            self._backup_slice(int(zz))
        self.masks[:, rows, self.y_row, cols] = self.mode == "add"
        self.dirty = True
        self._overlay_update()

    # ── mouse ────────────────────────────────────────────────────────────────
    def _on_mouse_moved(self, scene_pos):
        panel = self._panel_at(scene_pos)
        if panel is None:
            self._hide_brushes()
            return

        pos = panel.vb.mapSceneToView(scene_pos)
        for p in self.panels:
            p.brush.setVisible(p is panel)
        panel.move_brush(pos.x(), pos.y())

        # Focus-follows-mouse: entering a panel makes it the arrow-key target.
        if self.painting is None and panel is not self.active_panel:
            self.active_panel = panel
            self._highlight_panels()

    def _hide_brushes(self):
        for p in self.panels:
            p.brush.setVisible(False)

    def _navigate(self, panel, x, y):
        """Right-click moves the crosshair, linking the two views."""
        self.x_col = int(round(np.clip(x, 0, self.W - 1)))
        if panel is self.panel_xy:
            self.s_y.setValue(int(round(np.clip(y, 0, self.H - 1))))
        else:
            self.s_z.setValue(int(round(np.clip(y, 0, self.nZ - 1))))
        # setValue only refreshes when the value changed; the X column may
        # have moved regardless.
        self._structural_update()

    def _on_z_slider(self, val):
        self.z = int(val)
        self._structural_update()

    def _on_y_slider(self, val):
        self.y_row = int(val)
        self._structural_update()

    # ── keyboard actions ─────────────────────────────────────────────────────
    def _set_mode(self, mode):
        self.mode = mode
        self._set_status()

    def _set_brush_r(self, r):
        self.brush_r = int(np.clip(r, 1, 60))
        for p in self.panels:
            p.set_brush(radius=self.brush_r)
        self._set_status()

    def _toggle_mask(self):
        self.mask_visible = not self.mask_visible
        for p in self.panels:
            p.im_over.setVisible(self.mask_visible)

    def _reset_zoom(self):
        for p in self.panels:
            p.reset_zoom()

    def _step_active(self, step):
        if self.active_panel is self.panel_xz:
            self.s_y.setValue(int(np.clip(self.y_row + step, 0, self.H - 1)))
        else:
            self.s_z.setValue(int(np.clip(self.z + step, 0, self.nZ - 1)))

    def _copy_prev_frame(self):
        """Replace the current Z slice's mask with the previous slice's.

        Useful when adjacent slices are nearly identical: copy then touch up.
        The overwrite is captured on the undo stack like a paint stroke.
        """
        if self.z == 0:
            return
        backup = {self.z: self.masks[:, self.z].copy()}
        self.masks[:, self.z] = self.masks[:, self.z - 1]
        self.undo_stack.append(backup)
        self.dirty = True
        self._structural_update()

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

    def closeEvent(self, event):
        if not self.dirty:
            event.accept()
            return
        SB = QtWidgets.QMessageBox.StandardButton
        resp = QtWidgets.QMessageBox.question(
            self,
            "Unsaved changes",
            "Save before quitting?",
            SB.Save | SB.Discard | SB.Cancel,
            SB.Save,
        )
        if resp == SB.Save:
            self._save()
            event.accept()
        elif resp == SB.Discard:
            event.accept()
        else:
            event.ignore()


def main():
    name, mask_path, save_path, dicom_path = pick_series()

    app = QtWidgets.QApplication(sys.argv)
    editor = OrthoMaskEditor(name, mask_path, save_path, dicom_path)
    editor.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
