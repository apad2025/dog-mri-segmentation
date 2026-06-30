"""
mat_to_dcm.py
-------------
Convert corrected masks in mat_out/ into a DICOM segmentation that ITK-SNAP can
open as an overlay on top of the original upscaled series.

Usage:
    python mat_to_dcm.py
"""

from pathlib import Path

import numpy as np
import pydicom
import scipy.io
import SimpleITK as sitk
from pydicom.uid import generate_uid

PROJECT_ROOT = Path(__file__).resolve().parent

# Output is always written here (alongside the script); not prompted for.
OUT_ROOT = PROJECT_ROOT / "converted_to_dcm"

# Value written for voxels inside the mask (ITK-SNAP treats any nonzero as a
# label); background stays 0.
LABEL_VALUE = 1

# Orientation safety valve: flip the mask in Z (ascending slice order) if a
# future mask reads upside-down relative to the base DICOMs.
REVERSE_SLICES = False


def resolve_source_dir(mask_name, dicom_root):
    """Map a mask base name to its base series folder under `dicom_root`."""
    date, subseries = mask_name.split("_", 1)
    src_dir = dicom_root / date / subseries
    return date, subseries, src_dir


def single_echo_geometry(src_dir):
    """Return (echo_files_ascending, sitk_image) for ONE echo of the folder."""
    all_files = list(sitk.ImageSeriesReader().GetGDCMSeriesFileNames(str(src_dir)))

    normal = None
    rows = []
    for f in all_files:
        ds = pydicom.dcmread(f, force=True, stop_before_pixels=True)
        if normal is None:
            iop = np.asarray(ds.ImageOrientationPatient, float)
            normal = np.cross(iop[:3], iop[3:])
        proj = float(np.dot(np.asarray(ds.ImagePositionPatient, float), normal))
        rows.append((str(ds.EchoNumbers), proj, f))

    first_echo = min(e for e, _, _ in rows)
    echo_files = [f for _, f in sorted((p, f) for e, p, f in rows if e == first_echo)]

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(echo_files)
    image = reader.Execute()
    return echo_files, image


def build_mask_multiframe(label_stack, first_file, sitk_image, description):
    """Build one legacy multi-frame DICOM carrying the single-echo geometry."""
    template = pydicom.dcmread(first_file, force=True)
    ds = template.copy()
    ds.file_meta = template.file_meta

    n, rows, cols = label_stack.shape
    origin = sitk_image.GetOrigin()
    spacing = sitk_image.GetSpacing()

    # Brand as its own series so it never collides with the source.
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = generate_uid()
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.SeriesDescription = description

    ds.NumberOfFrames = n
    ds.Rows = rows
    ds.Columns = cols
    # Match the single-echo geometry exactly (origin == first frame position).
    ds.ImagePositionPatient = [float(c) for c in origin]
    ds.ImageOrientationPatient = list(template.ImageOrientationPatient)
    ds.PixelSpacing = [float(spacing[1]), float(spacing[0])]  # row, col
    ds.SpacingBetweenSlices = float(spacing[2])
    ds.SliceThickness = float(spacing[2])

    # 16-bit unsigned, no rescale, so labels read back as raw values.
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.RescaleSlope = 1
    ds.RescaleIntercept = 0
    ds.PixelData = label_stack.astype(np.uint16).tobytes()
    return ds


def prompt_dir(label):
    """Ask for a directory, re-prompting until a non-empty path is given."""
    while True:
        raw = input(f"{label}: ").strip().strip('"')
        if raw:
            return Path(raw).expanduser()
        print("  Entering a path is required.")


def main():
    mat_dir = prompt_dir("MAT_DIR (input .mat masks)")
    dicom_root = prompt_dir("DICOM_ROOT (base upscaled images)")
    out_root = OUT_ROOT

    mask_files = sorted(mat_dir.glob("*_mask.mat"))
    if not mask_files:
        raise FileNotFoundError(f"No *_mask.mat files found in {mat_dir}")

    # ── Selection prompt (same index pattern as convert_to_mat.py) ────────────
    print("Available masks:")
    for i, p in enumerate(mask_files):
        print(f"  [{i:2d}] {p.name.replace('_mask.mat', '')}")
    print("  [ a] All")

    choice = input("\nEnter index or 'a' for all: ").strip().lower()
    if choice == "a":
        selected = mask_files
    elif choice.isdigit():
        idx = int(choice)
        if not (0 <= idx < len(mask_files)):
            raise IndexError(f"Index {idx} out of range (0-{len(mask_files) - 1})")
        selected = [mask_files[idx]]
    else:
        raise ValueError(f"Invalid input: '{choice}'")

    # ── Convert ───────────────────────────────────────────────────────────────
    out_root.mkdir(parents=True, exist_ok=True)
    for mask_path in selected:
        mask_name = mask_path.name.replace("_mask.mat", "")
        mask_3d = np.asarray(scipy.io.loadmat(mask_path)["mask"]).astype(bool)
        if REVERSE_SLICES:
            mask_3d = mask_3d[::-1]

        date, subseries, src_dir = resolve_source_dir(mask_name, dicom_root)
        if not src_dir.exists():
            print(
                f"  ! Skipping {mask_name}: base DICOMs not found at {src_dir}.\n"
                f"    Run 'python fourier_upscale.py' first."
            )
            continue

        echo_files, sitk_image = single_echo_geometry(src_dir)
        n_slices = sitk_image.GetSize()[2]
        if n_slices != mask_3d.shape[0]:
            print(
                f"  ! Skipping {mask_name}: mask has {mask_3d.shape[0]} slices "
                f"but echo has {n_slices} slice positions."
            )
            continue

        # One frame per anatomical slice, ascending order, matching the echo.
        label_stack = (mask_3d > 0).astype(np.uint16) * LABEL_VALUE

        case_dir = out_root / date
        case_dir.mkdir(parents=True, exist_ok=True)
        mask_out = case_dir / f"{subseries}_MASK.dcm"
        build_mask_multiframe(
            label_stack, echo_files[0], sitk_image, "BICEPS_MASK"
        ).save_as(str(mask_out))
        print(
            f"{mask_name}: wrote {mask_out.name} "
            f"({label_stack.shape[0]} frames / slices) -> {case_dir}"
        )

    print(f"\nDone. Output under {out_root}")


if __name__ == "__main__":
    main()
