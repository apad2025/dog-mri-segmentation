"""One-time fix for the reversed-slice mask 20240925 Waylon.

The SAM2 notebook segmented this one series with `reverse_slices=True` but never
flipped the resulting volume back, so its mask is stored in reversed slice order
(axis 1) relative to how the DICOMs are read from disk.  Every other series is
fine.  This script reverses the slice axis to realign the mask with the images.

A backup (`*_mask.prereversal.npy`) is written before any change.  If that backup
already exists the file is treated as already-fixed and left alone, so re-running
is safe.
"""

from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
SERIES = "20240925_GRE2D_FATWATER_WAYLON_0012"
TARGETS = [
    PROJECT_ROOT / "edited_masks" / f"{SERIES}_mask.npy",
    PROJECT_ROOT / "masks_out" / f"{SERIES}_mask.npy",
]
SLICE_AXIS = 1  # shape is (n_echo, n_slices, H, W)


def fix_one(path: Path) -> None:
    if not path.exists():
        print(f"  skip (not found): {path}")
        return

    backup = path.with_name(path.stem + ".prereversal.npy")
    if backup.exists():
        print(f"  skip (already fixed, backup exists): {path.name}")
        return

    mask = np.load(path)
    before = [s for s in range(mask.shape[SLICE_AXIS]) if mask[0, s].any()]

    np.save(backup, mask)  # backup the original, unflipped mask
    fixed = np.flip(mask, axis=SLICE_AXIS).copy()
    np.save(path, fixed)

    after = [s for s in range(fixed.shape[SLICE_AXIS]) if fixed[0, s].any()]

    def rng(nz):
        return f"{nz[0]}-{nz[-1]}" if nz else "none"

    print(f"  fixed {path.name}: nonzero slices {rng(before)} -> {rng(after)}")
    print(f"        backup -> {backup.name}")


def main() -> None:
    print(f"Fixing reversed slice order for {SERIES}")
    for path in TARGETS:
        fix_one(path)
    print("Done.")


if __name__ == "__main__":
    main()
