"""Download and extract the Oxford-IIIT Pet and FS2K datasets.

Usage:
    python scripts/download_data.py --archive-dir <dir> --data-dir <dir> [--only pets|fs2k]

Archives are kept in --archive-dir (e.g. Google Drive, so they persist across
Colab sessions) and extracted into --data-dir (e.g. fast local /content disk).
Already-downloaded archives and already-extracted folders are skipped.
"""
import argparse
import tarfile
import urllib.request
import zipfile
from pathlib import Path

PETS_FILES = {
    "images.tar.gz": "https://thor.robots.ox.ac.uk/pets/images.tar.gz",
    "annotations.tar.gz": "https://thor.robots.ox.ac.uk/pets/annotations.tar.gz",
}
FS2K_GDRIVE_ID = "1saIMhQ3dc5_ftkfGmBPbCluRn_zy7QQp"  # from github.com/DengPingFan/FS2K


def _progress(block_num, block_size, total_size):
    if total_size > 0:
        pct = min(100, block_num * block_size * 100 // total_size)
        print(f"\r  {pct:3d}%", end="", flush=True)


def download(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.name} already downloaded")
        return
    print(f"[download] {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp, _progress)
    tmp.rename(dest)
    print()


def extract(archive, out_dir, marker):
    if (out_dir / marker).exists():
        print(f"[skip] {marker} already extracted")
        return
    print(f"[extract] {archive.name} -> {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out_dir)
    else:
        with tarfile.open(archive) as tf:
            tf.extractall(out_dir)


def get_pets(archive_dir, data_dir):
    pets_dir = data_dir / "oxford_pets"
    for name, url in PETS_FILES.items():
        download(url, archive_dir / name)
    extract(archive_dir / "images.tar.gz", pets_dir, "images")
    extract(archive_dir / "annotations.tar.gz", pets_dir, "annotations")
    # Official split files: annotations/trainval.txt and annotations/test.txt
    n_trainval = sum(1 for _ in open(pets_dir / "annotations" / "trainval.txt"))
    n_test = sum(1 for _ in open(pets_dir / "annotations" / "test.txt"))
    print(f"[ok] Oxford-IIIT Pet: {n_trainval} trainval, {n_test} test")


def get_fs2k(archive_dir, data_dir):
    archive = archive_dir / "FS2K.zip"
    if not (archive.exists() and archive.stat().st_size > 0):
        import gdown  # pip install gdown

        print("[download] FS2K from Google Drive")
        gdown.download(id=FS2K_GDRIVE_ID, output=str(archive), quiet=False)
    else:
        print("[skip] FS2K.zip already downloaded")
    extract(archive, data_dir, "FS2K")
    print(f"[ok] FS2K extracted to {data_dir / 'FS2K'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-dir", type=Path, default=Path("data/archives"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--only", choices=["pets", "fs2k"])
    args = parser.parse_args()
    args.archive_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    if args.only in (None, "pets"):
        get_pets(args.archive_dir, args.data_dir)
    if args.only in (None, "fs2k"):
        get_fs2k(args.archive_dir, args.data_dir)


if __name__ == "__main__":
    main()
