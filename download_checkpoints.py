"""Download SAM2.1 model checkpoints into the project root."""

import hashlib
import sys
import urllib.request
from pathlib import Path

CHECKPOINTS = [
    {
        "filename": "sam2.1_hiera_large.pt",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "md5": "2b30654b6112c42a115563c638d238d9",
    },
]

PROJECT_ROOT = Path(__file__).parent


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        mb_done = downloaded / 1e6
        mb_total = total_size / 1e6
        bar = "#" * int(pct // 2)
        print(f"\r  [{bar:<50}] {pct:5.1f}%  {mb_done:.0f}/{mb_total:.0f} MB", end="", flush=True)
    else:
        print(f"\r  {downloaded / 1e6:.0f} MB downloaded", end="", flush=True)


def _md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while data := f.read(chunk):
            h.update(data)
    return h.hexdigest()


def download(ckpt):
    dest = PROJECT_ROOT / ckpt["filename"]

    if dest.exists():
        print(f"{ckpt['filename']} already exists, verifying checksum...")
        if _md5(dest) == ckpt["md5"]:
            print("  OK — checksum matches, skipping download.")
            return
        print("  Checksum mismatch — re-downloading.")
        dest.unlink()

    print(f"Downloading {ckpt['filename']} ({ckpt['url']})")
    urllib.request.urlretrieve(ckpt["url"], dest, reporthook=_progress)
    print()

    actual = _md5(dest)
    if actual != ckpt["md5"]:
        dest.unlink()
        print(f"ERROR: checksum mismatch after download.\n  expected {ckpt['md5']}\n  got      {actual}", file=sys.stderr)
        sys.exit(1)

    print(f"  Saved to {dest}")


if __name__ == "__main__":
    for ckpt in CHECKPOINTS:
        download(ckpt)
    print("Done.")
