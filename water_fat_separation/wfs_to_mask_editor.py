"""
wfs_to_mask_editor.py
---------------------
Turn the water/fat separation .mat into series that mask_editor.py can open.

The .mat (see info.txt) holds:
  * Water           complex (Y, X, Z) = (376, 376, 50)  -- water-signal image
  * Fat             complex (Y, X, Z) = (376, 376, 50)  -- fat-signal image
  * TrimmedIndices  uint16  (3, 2)                       -- 1-based, inclusive
                    [start, end] the recon kept along (rows, cols, slices) of
                    the full 384x384x50 upscaled grid.

mask_editor.py shows a "..._WATER" / "..._FAT" series whenever its DICOM folder
exists under DICOM_Files_upscaled/<date>/<subseries>_{WATER,FAT}/, and routes all
edits to the shared base mask masks_out/<base>_mask.npy. So this script only needs
to write the DICOM series -- no per-variant mask file.

For each requested image (water and/or fat) this script:
  * takes the complex magnitude,
  * pads it back onto the full 384x384x50 grid at the TrimmedIndices offsets
    (so it lines up 1:1 with the existing 0012 mask and DICOMs), and
  * writes a new 7-echo DICOM series by CLONING the source 0012 DICOM headers
    -- one clone per (echo, slice) -- so geometry, spacing, EchoNumbers and
    SliceLocation are inherited exactly; only the pixel data is replaced (the
    same water/fat slice is written to all 7 echoes, since the anatomy is echo-
    independent and the mask is broadcast across echoes anyway).

The base series must already be segmented (its mask in masks_out/), since the
water/fat variants edit that shared mask. After running, just start
mask_editor.py and pick the new "..._WATER" / "..._FAT" series.

Assumptions (flip the flags below if a series looks wrong):
  * mat axis order is (Y rows, X cols, Z slices) matching the DICOM grid.
  * mat slice index k corresponds to DICOM InstanceNumber k+1 (same order the
    pipeline used). This project has hit slice-reversal before; if water/fat is
    upside-down in Z relative to the mask, set REVERSE_SLICES = True.

Usage:
    python water_fat_separation/wfs_to_mask_editor.py
    # then pick one of the .mat files in this directory when prompted; its date
    # prefix selects the matching base series under DICOM_Files_upscaled/.
"""

from pathlib import Path

import numpy as np
import pydicom
from pydicom.uid import generate_uid

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WFS_DIR = Path(__file__).resolve().parent  # the .mat files live next to this script
UPSCALED_DICOM_ROOT = PROJECT_ROOT / "DICOM_Files_upscaled"
MASK_DIR = PROJECT_ROOT / "masks_out"

# The .mat to convert and the source series whose DICOM headers + mask we clone
# (the 384x384 upscaled grid) are chosen at runtime: pick_mat_file() lists the
# .mat files in WFS_DIR, and resolve_source_series() maps the chosen file's date
# prefix to its base series folder under UPSCALED_DICOM_ROOT.

# Which separated images to emit, and the suffix each gets.
IMAGES = {"Water": "WATER", "Fat": "FAT"}

# Orientation safety valves (see module docstring).
REVERSE_SLICES = False
TRANSPOSE_INPLANE = False


# ── Helpers ───────────────────────────────────────────────────────────────────
def source_files_by_echo(src_dir):
    """Map echo number (1-based) -> list of (InstanceNumber, Path), slice-sorted.

    mask_editor's loader bins by EchoNumbers and walks files in sorted order, so
    slice k of each echo is the k-th instance.  We mirror that here to keep the
    new series' slice order identical to the mask's.
    """
    by_echo = {}
    for p in sorted(q for q in src_dir.iterdir() if q.is_file()):
        ds = pydicom.dcmread(str(p), force=True, stop_before_pixels=True)
        en = int(ds.EchoNumbers)
        inst = int(getattr(ds, "InstanceNumber", 0))
        by_echo.setdefault(en, []).append((inst, p))
    for en in by_echo:
        by_echo[en].sort(key=lambda t: t[0])
    return by_echo


def pad_to_full_grid(vol, trimmed_indices, full_shape):
    """Place a trimmed (Y, X, Z) volume back into a zero `full_shape` grid.

    `trimmed_indices` is the .mat's (3, 2) uint16 of 1-based inclusive
    [start, end] for each axis; we convert to 0-based half-open slices.
    """
    out = np.zeros(full_shape, dtype=vol.dtype)
    slices = []
    for axis, (start, end) in enumerate(trimmed_indices):
        lo = int(start) - 1
        hi = int(end)  # inclusive end -> half-open
        if hi - lo != vol.shape[axis]:
            raise ValueError(
                f"TrimmedIndices axis {axis} spans {hi - lo} px but the data is "
                f"{vol.shape[axis]} px; the .mat and grid disagree."
            )
        if hi > full_shape[axis]:
            raise ValueError(
                f"TrimmedIndices axis {axis} ends at {hi} but the full grid is "
                f"only {full_shape[axis]} px."
            )
        slices.append(slice(lo, hi))
    out[tuple(slices)] = vol
    return out


def write_series(name, vol_yxz, by_echo, out_dir):
    """Write a 7-echo DICOM series from a (Y, X, Z) magnitude volume.

    Pixels are scaled to fill the source's BitsStored range so the image has
    good contrast; mask_editor re-normalizes the whole volume anyway.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clear any stale files from a previous run so the loader's count is exact.
    for old in out_dir.iterdir():
        if old.is_file():
            old.unlink()

    series_uid = generate_uid()
    vmax = float(vol_yxz.max())

    n_written = 0
    for en, instances in sorted(by_echo.items()):
        for z, (_inst, src_path) in enumerate(instances):
            ds = pydicom.dcmread(str(src_path), force=True)
            maxval = (1 << int(ds.BitsStored)) - 1
            scale = (maxval / vmax) if vmax > 0 else 1.0

            pix = vol_yxz[:, :, z]
            pix = np.clip(np.round(pix * scale), 0, maxval).astype(np.uint16)
            if pix.shape != (int(ds.Rows), int(ds.Columns)):
                raise ValueError(
                    f"Slice {z} is {pix.shape} but DICOM grid is "
                    f"{(int(ds.Rows), int(ds.Columns))}."
                )

            ds.PixelData = pix.tobytes()
            # Mark it as a distinct series so it doesn't collide with the source.
            ds.SeriesInstanceUID = series_uid
            ds.SOPInstanceUID = generate_uid()
            ds.SeriesDescription = name
            # EchoNumbers / InstanceNumber / geometry are inherited unchanged.

            fname = f"echo{en}_slice{z:03d}.dcm"
            ds.save_as(str(out_dir / fname))
            n_written += 1
    return n_written


# ── Selection ───────────────────────────────────────────────────────────────--
def pick_mat_file():
    """List the .mat files next to this script and prompt for one."""
    mats = sorted(WFS_DIR.glob("*.mat"))
    if not mats:
        raise FileNotFoundError(f"No .mat files found in {WFS_DIR}")

    print("Available water/fat .mat files:")
    for i, p in enumerate(mats):
        print(f"  [{i:2d}] {p.name}")

    choice = input("\nEnter index or filename: ").strip()
    if choice.isdigit():
        idx = int(choice)
        if not (0 <= idx < len(mats)):
            raise IndexError(f"Index {idx} out of range (0-{len(mats) - 1})")
        return mats[idx]
    for p in mats:
        if choice in (p.name, p.stem):
            return p
    raise ValueError(f"Not found: {choice!r}")


def resolve_source_series(mat_path):
    """Return (date, subseries) for the base series a .mat corresponds to.

    The .mat filenames are inconsistent (some omit the '0012' token), so rather
    than parse the subseries out of the name we take the date prefix and find
    the base '*_0012' series folder for that date under UPSCALED_DICOM_ROOT
    (the generated '..._WATER' / '..._FAT' variants end in their suffix, so they
    are excluded). If a date has more than one base series, prompt.
    """
    date = mat_path.name.split("_", 1)[0]
    date_dir = UPSCALED_DICOM_ROOT / date
    if not date_dir.exists():
        raise FileNotFoundError(
            f"No upscaled DICOMs for date {date} ({date_dir}).\n"
            f"Run 'python fourier_upscale.py' first."
        )

    candidates = [
        d for d in sorted(date_dir.iterdir()) if d.is_dir() and d.name.endswith("_0012")
    ]
    if not candidates:
        raise FileNotFoundError(f"No base '*_0012' series folder under {date_dir}.")
    if len(candidates) > 1:
        print(f"\nMultiple base series for {date}:")
        for i, d in enumerate(candidates):
            print(f"  [{i:2d}] {d.name}")
        idx = int(input("Enter index: ").strip())
        candidates = [candidates[idx]]
    return date, candidates[0].name


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import scipy.io as sio

    mat_path = pick_mat_file()
    src_date, src_subseries = resolve_source_series(mat_path)
    src_name = f"{src_date}_{src_subseries}"
    print(f"\nUsing {mat_path.name}  ->  source series {src_name}\n")

    src_dir = UPSCALED_DICOM_ROOT / src_date / src_subseries
    if not src_dir.exists():
        raise FileNotFoundError(
            f"Source upscaled DICOMs not found: {src_dir}\n"
            f"Run 'python fourier_upscale.py' first."
        )

    by_echo = source_files_by_echo(src_dir)
    sample = pydicom.dcmread(str(next(iter(by_echo.values()))[0][1]), force=True)
    full_rows, full_cols = int(sample.Rows), int(sample.Columns)
    n_slices = len(next(iter(by_echo.values())))
    full_shape = (full_rows, full_cols, n_slices)  # (Y, X, Z)
    print(f"Source grid: {full_shape} (Y, X, Z), echoes {sorted(by_echo)}")

    mat = sio.loadmat(str(mat_path))
    trimmed = np.asarray(mat["TrimmedIndices"])

    # No per-variant mask is written: mask_editor lists the ..._WATER / ..._FAT
    # series from their DICOM folders and edits the shared base mask. Warn if
    # that base mask is missing, since the variants have nothing to edit without
    # it (re-run biceps_pipeline.py for this series).
    if not (MASK_DIR / f"{src_name}_mask.npy").exists():
        print(
            f"  ! Warning: base mask {src_name}_mask.npy not found in {MASK_DIR}.\n"
            f"            mask_editor's water/fat variants share it, so segment "
            f"the base series first."
        )

    for var, suffix in IMAGES.items():
        if var not in mat:
            raise KeyError(f"'{var}' not found in {mat_path.name}")

        mag = np.abs(np.asarray(mat[var])).astype(np.float64)  # (Y, X, Z)
        if TRANSPOSE_INPLANE:
            mag = np.swapaxes(mag, 0, 1)
        if REVERSE_SLICES:
            mag = mag[:, :, ::-1]

        full = pad_to_full_grid(mag, trimmed, full_shape)

        new_name = f"{src_name}_{suffix}"
        out_dir = UPSCALED_DICOM_ROOT / src_date / f"{src_subseries}_{suffix}"
        n = write_series(new_name, full, by_echo, out_dir)
        print(f"{var}: wrote {n} DICOMs -> {out_dir}")

    print("\nDone. Launch 'python mask_editor.py' and pick a '..._WATER' / "
          "'..._FAT' series.")


if __name__ == "__main__":
    main()
