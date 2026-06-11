"""
review_masks.py
---------------
Read-only ortho-slice mask viewer (PyQtGraph).

The viewing half of mask_editor.py: the same linked XY and XZ panels,
crosshair navigation, sliders and Fourier-upscaled display images, with the
editing stripped out. Masks can be inspected but never modified, so there
is no brush, no undo and no saving.

Masks can come from masks_out/ (SAM) or edited_masks/ (edited), in
either the classic one-file-per-series format (*_mask.npy) or the echo
format (bf_masks_echo*.npy, one file per echo; set DICOM_SERIES and
SLICE_OFFSET below to line the masks up with their DICOM volume).

Controls
  Click/drag (L or R)  move the crosshair (navigate): in XY it picks the XZ
                       row; in XZ it picks the XY slice
  Middle-drag          pan the panel
  Z slider             step through anatomical slices (XY)
  Y slider             step the XZ cross-section row
  Up/Down              step the active (hovered) panel
  Scroll wheel         zoom the panel under the cursor
  R                    reset zoom on both panels
  V                    toggle mask overlay
  Q                    quit

Usage:
    python review_masks.py [--upscale N]   # N=2 default, 1 disables
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from mask_editor import (
    Panel,
    fourier_upscale,
    load_dicom_images,
    read_voxel_spacing,
)

pg.setConfigOptions(imageAxisOrder="row-major")


PROJECT_ROOT = Path(__file__).parent
DICOM_ROOT = PROJECT_ROOT / "DICOM_Files"

# ── Echo-format settings (only used when the mask dir contains bf_masks_echo*.npy)
# Set DICOM_SERIES to the relative path under DICOM_ROOT, e.g.:
#   "20240709/GRE2D_FATWATER_WAYLON_0012"
# Set SLICE_OFFSET to the first DICOM slice the masks correspond to, e.g. 13.
DICOM_SERIES = "20240709/GRE2D_FATWATER_WAYLON_0012"
SLICE_OFFSET = 0


# ── Series selection ──────────────────────────────────────────────────────────
def pick_mask_dir():
    print("Which masks would you like to review?")
    print("  [1] SAM masks in masks_out/")
    print("  [2] Edited masks in edited_masks/")
    choice = input("Choice [1/2]: ").strip()
    if choice == "2":
        return PROJECT_ROOT / "edited_masks", "edited"
    return PROJECT_ROOT / "masks_out", "SAM"


def load_series():
    """Prompt for a mask source and series.

    Returns (name, src_label, masks, imgs, xz_aspect) with masks and imgs both
    shaped (n_echo, Z, Y, X) and imgs normalized to [0, 1].
    """
    mask_dir, src_label = pick_mask_dir()
    if not mask_dir.exists():
        print(f"Folder not found: {mask_dir}")
        sys.exit(1)

    classic_files = sorted(mask_dir.glob("*_mask.npy"))
    echo_files = sorted(mask_dir.glob("bf_masks_echo*.npy"))

    if classic_files:
        # ── Classic format: one file per series ──────────────────────────────
        available = [p.name.replace("_mask.npy", "") for p in classic_files]
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

        date, subseries = name.split("_", 1)
        dicom_path = DICOM_ROOT / date / subseries
        if not dicom_path.exists():
            print(f"DICOM folder not found: {dicom_path}")
            sys.exit(1)

        print(f"\nLoading {name} ...")
        masks = np.load(mask_dir / f"{name}_mask.npy")  # (E, Z, Y, X)
        imgs = load_dicom_images(str(dicom_path))  # (E, Z, Y, X)

    elif echo_files:
        # ── Echo format: one file per echo, stack into (E, Z, Y, X) ──────────
        masks = np.stack([np.load(p) for p in echo_files], axis=0)
        name = mask_dir.name
        n_slices = masks.shape[1]

        if DICOM_SERIES:
            dicom_path = DICOM_ROOT / DICOM_SERIES
            if not dicom_path.exists():
                print(f"DICOM folder not found: {dicom_path}")
                sys.exit(1)
            print(f"\nLoading {name} ...")
            all_imgs = load_dicom_images(str(dicom_path))  # (E, Z, Y, X)
            imgs = all_imgs[:, SLICE_OFFSET : SLICE_OFFSET + n_slices]
            if imgs.shape[1] < n_slices:
                print(
                    f"Warning: DICOM only has {all_imgs.shape[1]} slices; "
                    f"SLICE_OFFSET={SLICE_OFFSET} covers {imgs.shape[1]} of "
                    f"{n_slices} mask slices -- padding the rest with black."
                )
                pad = np.zeros(
                    (imgs.shape[0], n_slices - imgs.shape[1], *imgs.shape[2:]),
                    dtype=imgs.dtype,
                )
                imgs = np.concatenate([imgs, pad], axis=1)
        else:
            print(
                f"\nLoading {name} (no DICOM -- set DICOM_SERIES to show originals) ..."
            )
            dicom_path = None
            imgs = np.zeros(masks.shape, dtype=np.float32)

    else:
        print(
            f"No masks found in {mask_dir}\n"
            f"  Expected '*_mask.npy' (classic) or 'bf_masks_echo*.npy' (echo format)."
        )
        sys.exit(1)

    # Physical aspect for the XZ panel (Z is vertical, X horizontal).
    if dicom_path is not None:
        sz, _sy, sx = read_voxel_spacing(dicom_path)
        xz_aspect = sz / sx
    else:
        xz_aspect = 1.0

    return name, src_label, masks, imgs, xz_aspect


# ── ViewBox with navigate-only mouse bindings ─────────────────────────────────
class NavViewBox(pg.ViewBox):
    """ViewBox where left or right click/drag moves the crosshair.

    With no painting in the viewer, both buttons navigate.  The viewer assigns
    `on_navigate(x, y)` after construction.  Wheel zoom is inherited;
    middle-drag falls through to the default pan behavior.
    """

    NAV_BUTTONS = (
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.MouseButton.RightButton,
    )

    def __init__(self):
        super().__init__(invertY=True, enableMenu=False)
        self.on_navigate = None

    def _view_pos(self, ev):
        p = self.mapSceneToView(ev.scenePos())
        return p.x(), p.y()

    def mouseClickEvent(self, ev):
        if ev.button() in self.NAV_BUTTONS:
            ev.accept()
            self.on_navigate(*self._view_pos(ev))
        else:
            super().mouseClickEvent(ev)

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() in self.NAV_BUTTONS:
            ev.accept()
            self.on_navigate(*self._view_pos(ev))
        else:
            super().mouseDragEvent(ev, axis=axis)


# ── Viewer ────────────────────────────────────────────────────────────────────
class OrthoMaskViewer(QtWidgets.QMainWindow):
    def __init__(self, name, src_label, masks, imgs, xz_aspect, upscale=2):
        super().__init__()
        self.name = name
        self.src_label = src_label
        self.masks = masks
        self.imgs = imgs
        self.xz_aspect = xz_aspect

        self.upscale = max(1, int(upscale))
        if self.upscale > 1:
            print(f"Fourier-upscaling display images x{self.upscale} in-plane ...")
            self.imgs = fourier_upscale(self.imgs, self.upscale)

        self.echo = 0
        self.n_echo, self.nZ, self.H, self.W = self.masks.shape

        # Crosshair state
        self.z = 0  # XY slice (start at the first slice)
        self.y_row = self.H // 2  # XZ cross-section row
        self.x_col = self.W // 2

        self.mask_visible = True

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
        self.gw.addLabel("XY  ·  slice", row=0, col=0)
        self.gw.addLabel("XZ  ·  cross-section", row=0, col=1)

        # Same base-image-to-mask-grid mapping as the editor (upscaled pixel j
        # sits at original coordinate j/f).  Panel is reused from mask_editor;
        # its paint callback is never wired to the ViewBoxes and the brush
        # cursor is never shown, so the masks cannot be modified.
        f = self.upscale
        vb_xy, vb_xz = NavViewBox(), NavViewBox()
        self.gw.addItem(vb_xy, row=1, col=0)
        self.gw.addItem(vb_xz, row=1, col=1)

        self.panel_xy = Panel(
            vb_xy,
            get_base=lambda: self.imgs[self.echo, self.z],
            get_mask=lambda: self.masks[self.echo, self.z],
            paint=None,
            n_rows=self.H,
            n_cols=self.W,
            base_rect=QtCore.QRectF(-0.5 / f, -0.5 / f, self.W, self.H),
        )
        # XZ panel: rows = Z (native resolution), cols = X (upscaled).
        self.panel_xz = Panel(
            vb_xz,
            get_base=lambda: self.imgs[self.echo, :, self.y_row * self.upscale, :],
            get_mask=lambda: self.masks[self.echo, :, self.y_row, :],
            paint=None,
            n_rows=self.nZ,
            n_cols=self.W,
            base_rect=QtCore.QRectF(-0.5 / f, -0.5, self.W, self.nZ),
            aspect=self.xz_aspect,
        )
        self.panels = (self.panel_xy, self.panel_xz)
        self.active_panel = self.panel_xy  # arrow keys act on this panel

        for panel in self.panels:
            panel.vb.on_navigate = lambda x, y, p=panel: self._navigate(p, x, y)

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
        self.status.setText(
            "[VIEW ONLY]  click=move crosshair  Up/Dn=step active panel  "
            "V=toggle mask  R=reset zoom  Q=quit"
        )

        # Focus-follows-mouse across both panels.
        self.gw.scene().sigMouseMoved.connect(self._on_mouse_moved)

        self._highlight_panels()

        # Keyboard ------------------------------------------------------------
        for keys, cb in [
            ("V", self._toggle_mask),
            ("R", self._reset_zoom),
            ("Q", self.close),
            ("Up", lambda: self._step_active(+1)),
            ("Down", lambda: self._step_active(-1)),
        ]:
            QtGui.QShortcut(QtGui.QKeySequence(keys), self).activated.connect(cb)

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
        self.setWindowTitle(
            f"{self.name}  [{self.src_label}]    Z={self.z}/{self.nZ - 1}   "
            f"Y={self.y_row}/{self.H - 1}"
        )

    def _highlight_panels(self):
        for p in self.panels:
            p.set_active(p is self.active_panel)

    # ── mouse ────────────────────────────────────────────────────────────────
    def _on_mouse_moved(self, scene_pos):
        # Focus-follows-mouse: entering a panel makes it the arrow-key target.
        panel = self._panel_at(scene_pos)
        if panel is not None and panel is not self.active_panel:
            self.active_panel = panel
            self._highlight_panels()

    def _navigate(self, panel, x, y):
        """Clicking moves the crosshair, linking the two views."""
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


def main():
    parser = argparse.ArgumentParser(description="Read-only ortho-slice mask viewer")
    parser.add_argument(
        "--upscale",
        type=int,
        default=2,
        metavar="N",
        help="in-plane Fourier upscale factor for display images "
        "(X/Y only, never Z); 1 disables (default: 2)",
    )
    args = parser.parse_args()
    name, src_label, masks, imgs, xz_aspect = load_series()

    app = QtWidgets.QApplication(sys.argv)
    viewer = OrthoMaskViewer(
        name, src_label, masks, imgs, xz_aspect, upscale=args.upscale
    )
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
