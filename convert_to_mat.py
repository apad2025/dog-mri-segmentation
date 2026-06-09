"""
convert_to_mat.py
-----------------
Converts 4D mask .npy files (7, 50, 192, 192) to 3D MATLAB .mat files (50, 192, 192)
by dropping redundant echo dimension (all echoes identical).
"""

from pathlib import Path
import numpy as np
import scipy.io

PROJECT_ROOT = Path(__file__).parent
MASK_DIR = PROJECT_ROOT / "edited_masks"
MAT_DIR = PROJECT_ROOT / "mat_out"

MAT_DIR.mkdir(exist_ok=True)

mask_files = sorted(MASK_DIR.glob("*_mask.npy"))
if not mask_files:
    raise FileNotFoundError(f"No *_mask.npy files found in {MASK_DIR}")

# ── Selection prompt ──────────────────────────────────────────────────────────
print("Available masks:")
for i, p in enumerate(mask_files):
    print(f"  [{i:2d}] {p.name.replace('_mask.npy', '')}")
print("  [ a] All")

choice = input("\nEnter index or 'a' for all: ").strip().lower()

if choice == "a":
    selected = mask_files
elif choice.isdigit():
    idx = int(choice)
    if not (0 <= idx < len(mask_files)):
        raise IndexError(f"Index {idx} out of range (0–{len(mask_files) - 1})")
    selected = [mask_files[idx]]
else:
    raise ValueError(f"Invalid input: '{choice}'")

# ── Convert ───────────────────────────────────────────────────────────────────
for mask_path in selected:
    mask_4d = np.load(mask_path)  # (7, 50, 192, 192) bool
    mask_3d = mask_4d[0].astype(bool)  # (50, 192, 192) logical for MATLAB

    out_path = MAT_DIR / mask_path.name.replace("_mask.npy", "_mask.mat")
    scipy.io.savemat(out_path, {"mask": mask_3d})
    print(f"Saved {out_path.name}")

print(f"\nDone. {len(selected)} file(s) written to {MAT_DIR}")
