"""
fourier_upscale.py
------------------
One-time preprocessing for mask_editor.py.

mask_editor used to Fourier-upscale the display images on every open, which is
slow and left the masks at half the image resolution. This script does that
work once and writes it to disk:

  * DICOM images  ->  Fourier (k-space zero-fill) upscaled 2x in-plane and
                      written to DICOM_Files_upscaled/<date>/<subseries>/.
                      The originals in DICOM_Files/ are left untouched.
  * Boolean masks ->  pixel-doubled 2x in-plane (nearest neighbor) and
                      rewritten in place in masks_out/ and edited_masks/, so
                      each mask lands on the same fine grid as its image.

The Z (slice) axis is never upscaled -- the 3 mm slice spacing carries no
extra information to interpolate. Only the series that mask_editor lists
(i.e. those with a mask in masks_out/ or edited_masks/) are processed.

The script is idempotent: a mask whose in-plane size already matches the
upscaled image is skipped, and existing mirror DICOMs are skipped unless
--force is given, so re-running never double-upscales.

Usage:
    python fourier_upscale.py [--factor N] [--force]
"""

import argparse
from pathlib import Path

import numpy as np
import pydicom

PROJECT_ROOT = Path(__file__).parent
DICOM_ROOT = PROJECT_ROOT / "DICOM_Files"
UPSCALED_DICOM_ROOT = PROJECT_ROOT / "DICOM_Files_upscaled"
MASK_DIRS = (PROJECT_ROOT / "masks_out", PROJECT_ROOT / "edited_masks")


def fourier_upscale_2d(img, factor):
    """Zero-fill (sinc) interpolate a single 2D image up by `factor`.

    Inverse-FFT to k-space, zero-pad the centered spectrum symmetrically, then
    FFT back. numpy's 1/N convention on ifftn means the original sample values
    are reproduced exactly at their positions; the magnitude is returned (the
    images are MR magnitude data) and the caller clips it to the valid range.
    """
    H, W = img.shape
    k = np.fft.fftshift(np.fft.ifftn(img.astype(np.float64)))
    py, px = H * (factor - 1), W * (factor - 1)
    pad = ((py // 2, py - py // 2), (px // 2, px - px // 2))
    k = np.pad(k, pad)
    up = np.fft.fftn(np.fft.ifftshift(k))
    return np.abs(up)


def upscale_dicom_file(src, dst, factor):
    """Upscale one DICOM's pixel data and write it to `dst`, halving the
    in-plane PixelSpacing so the image stays physically the same size."""
    ds = pydicom.dcmread(str(src), force=True)
    up = fourier_upscale_2d(ds.pixel_array, factor)
    maxval = (1 << int(ds.BitsStored)) - 1
    up = np.clip(np.round(up), 0, maxval).astype(np.uint16)

    ds.PixelData = up.tobytes()
    ds.Rows, ds.Columns = up.shape
    if "PixelSpacing" in ds:
        py, px = (float(v) for v in ds.PixelSpacing)
        ds.PixelSpacing = [py / factor, px / factor]

    dst.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(dst))


def upscale_series(name, factor, force):
    """Mirror one DICOM series (date/subseries) into UPSCALED_DICOM_ROOT.

    Returns the original in-plane size (rows) so callers can size-check masks,
    or None if the source series is missing.
    """
    date, subseries = name.split("_", 1)
    src_dir = DICOM_ROOT / date / subseries
    if not src_dir.exists():
        print(f"  ! DICOM folder not found, skipping: {src_dir}")
        return None
    dst_dir = UPSCALED_DICOM_ROOT / date / subseries

    files = sorted(p for p in src_dir.iterdir() if p.is_file())
    orig_rows = int(pydicom.dcmread(str(files[0]), force=True).Rows)

    written = skipped = 0
    for src in files:
        dst = dst_dir / src.name
        if dst.exists() and not force:
            skipped += 1
            continue
        upscale_dicom_file(src, dst, factor)
        written += 1
    print(f"  DICOM: {written} written, {skipped} already present -> {dst_dir}")
    return orig_rows


def upscale_mask_file(path, orig_rows, factor):
    """Pixel-double a boolean mask in-plane and rewrite it in place.

    Skips masks already at the upscaled size so re-running is safe.
    """
    arr = np.load(path)
    h = arr.shape[-2]
    if h == orig_rows * factor:
        print(f"  mask already upscaled, skipping: {path.name}")
        return
    if h != orig_rows:
        print(
            f"  ! mask {path.name} is {h}px in-plane but its image is "
            f"{orig_rows}px; skipping (size mismatch)"
        )
        return
    up = np.repeat(np.repeat(arr, factor, axis=-1), factor, axis=-2)
    np.save(path, up)
    print(f"  mask: {arr.shape} -> {up.shape}  {path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="One-time 2x Fourier upscale of mask_editor's images + masks"
    )
    parser.add_argument(
        "--factor", type=int, default=2, metavar="N",
        help="in-plane upscale factor (X/Y only, never Z; default: 2)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="re-write mirror DICOMs even if they already exist",
    )
    args = parser.parse_args()
    factor = args.factor
    if factor < 2:
        parser.error("--factor must be >= 2")

    # The series mask_editor uses are exactly those with a mask on disk.
    names = sorted(
        {p.name.replace("_mask.npy", "")
         for d in MASK_DIRS if d.exists()
         for p in d.glob("*_mask.npy")}
    )
    if not names:
        print("No *_mask.npy files found in masks_out/ or edited_masks/.")
        return

    print(f"Upscaling {len(names)} series by {factor}x in-plane ...\n")
    for i, name in enumerate(names, 1):
        print(f"[{i}/{len(names)}] {name}")
        orig_rows = upscale_series(name, factor, args.force)
        if orig_rows is None:
            continue
        for d in MASK_DIRS:
            mp = d / f"{name}_mask.npy"
            if mp.exists():
                upscale_mask_file(mp, orig_rows, factor)
    print("\nDone.")


if __name__ == "__main__":
    main()
