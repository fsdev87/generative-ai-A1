"""FS2K face-to-sketch data (Task 4): official pairs, stratified validation split,
128x128 paired cache and the paired training dataset.

Layout: photo/photo{k}/image{n}.* pairs with sketch/sketch{k}/sketch{n}.*;
anno_train.json / anno_test.json define the official split and the style (0, 1, 2).
Training data and application inputs share one preprocessing path (`preprocess_photo`):
EXIF-upright, transparency flattened onto white, centre square crop, bicubic resize to
128x128. Sketches keep one channel (every FS2K sketch has R = G = B).
The data investigation behind these choices is in docs/fs2k_notes.md.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import Dataset

from ..common.paths import MANIFEST_DIR, get_dir

IMAGE_SIZE = 128
SPLIT_SEED = 42
VAL_FRACTION = 0.15
NUM_STYLES = 3
STYLE_NAMES = ("Style 1", "Style 2", "Style 3")  # annotation style 0/1/2 as shown to users
SKETCH_CHANNELS = 1
SOURCES = ("photo1", "photo2", "photo3")
SPLITS = ("train", "val", "test")
SPLIT_PATH = MANIFEST_DIR / "fs2k_split.json"
MAX_ZOOM = 286 / 256  # pix2pix jitter: 256 -> 286 then crop, i.e. up to 143 px for 128
_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


# --------------------------------------------------------------------------- #
# Official pairs and split
# --------------------------------------------------------------------------- #
def _index_folder(folder, prefix):
    """{number: path} for files named prefix<number>.<ext>. Extensions are matched
    case-insensitively: photo3/image0449 is a .JPG, which matters on Linux/Colab."""
    index = {}
    for path in Path(folder).iterdir():
        if path.suffix.lower() in _IMAGE_EXTS and path.stem.startswith(prefix):
            number = path.stem[len(prefix):]
            if number in index:
                raise ValueError(f"two files for {path.stem} in {folder}")
            index[number] = path
    return index


def _upright_size(path):
    """(width, height) after the EXIF orientation is applied, read from the header only."""
    with Image.open(path) as im:
        w, h = im.size
        return (h, w) if im.getexif().get(0x0112) in (5, 6, 7, 8) else (w, h)


def official_pairs(fs2k_dir, check_sizes=True):
    """Pairs of the official lists: {"train"|"test": [dict(id, photo, sketch, style)]}, sorted by id.

    The id is the annotation's image_name, e.g. "photo1/image0110", whose sketch is
    sketch/sketch1/sketch0110.*. Raises if an annotated file is missing, an image file is
    not annotated, an id is listed twice, a style is not 0/1/2 or (check_sizes) a photo
    and its sketch differ in size, so a successful call means a complete one-to-one pairing.
    """
    fs2k_dir = Path(fs2k_dir)
    photos, sketches = {}, {}
    for source in SOURCES:
        for number, path in _index_folder(fs2k_dir / "photo" / source, "image").items():
            photos[f"{source}/image{number}"] = path
        for number, path in _index_folder(fs2k_dir / "sketch" / f"sketch{source[-1]}", "sketch").items():
            sketches[f"{source}/image{number}"] = path
    pairs, seen = {}, set()
    for split in ("train", "test"):
        with open(fs2k_dir / f"anno_{split}.json") as f:
            annotations = json.load(f)
        items = []
        for entry in annotations:
            pid, style = entry["image_name"], int(entry["style"])
            if pid in seen:
                raise ValueError(f"{pid} is annotated twice")
            if pid not in photos or pid not in sketches:
                raise FileNotFoundError(f"photo or sketch missing for {pid}")
            if not 0 <= style < NUM_STYLES:
                raise ValueError(f"{pid}: unknown style {style}")
            seen.add(pid)
            items.append({"id": pid, "photo": photos[pid], "sketch": sketches[pid], "style": style})
        pairs[split] = sorted(items, key=lambda p: p["id"])
    unannotated = sorted((photos.keys() | sketches.keys()) - seen)
    if unannotated:
        raise ValueError(f"{len(unannotated)} image files without annotation, e.g. {unannotated[:3]}")
    if check_sizes:
        for p in pairs["train"] + pairs["test"]:
            if _upright_size(p["photo"]) != _upright_size(p["sketch"]):
                raise ValueError(f"{p['id']}: photo and sketch sizes differ")
    return pairs


def make_split(pairs, seed=SPLIT_SEED, val_fraction=VAL_FRACTION):
    """Style-stratified train/val split of the official training pairs; official test kept as is.

    Validation gets round(val_fraction * N) ids, allocated to the styles by largest remainder
    (every style gets its exact share rounded down or up) and drawn per style from a
    permutation of its sorted ids by a generator seeded with `seed`.
    """
    ids = np.array([p["id"] for p in pairs["train"]])
    styles = np.array([p["style"] for p in pairs["train"]])
    exact = val_fraction * np.bincount(styles, minlength=NUM_STYLES)
    n_val = np.floor(exact).astype(int)
    extra = int(round(val_fraction * len(ids))) - int(n_val.sum())
    n_val[np.argsort(n_val - exact, kind="stable")[:extra]] += 1
    rng = np.random.default_rng(seed)
    val = set()
    for s in range(NUM_STYLES):
        members = ids[styles == s]
        val.update(members[rng.permutation(len(members))[:n_val[s]]].tolist())
    style = {p["id"]: p["style"] for p in pairs["train"] + pairs["test"]}
    split = {
        "dataset": "FS2K",
        "seed": seed,
        "val_fraction": val_fraction,
        "stratify": "style",
        "train": sorted(set(ids.tolist()) - val),
        "val": sorted(val),
        "test": [p["id"] for p in pairs["test"]],
        "style": style,
    }
    split["counts"] = split_counts(split)
    return split


def split_counts(split):
    """{split: [number of pairs of style 0, 1, 2]}."""
    return {
        name: np.bincount([split["style"][i] for i in split[name]], minlength=NUM_STYLES).tolist()
        for name in SPLITS
    }


def check_split(split, pairs):
    """Raise unless `split` partitions the official lists and agrees with the annotated styles."""
    train, val = set(split["train"]), set(split["val"])
    if train & val or train | val != {p["id"] for p in pairs["train"]}:
        raise ValueError("train/val do not partition the official FS2K training list")
    if split["test"] != [p["id"] for p in pairs["test"]]:
        raise ValueError("test ids differ from the official FS2K test list")
    if split["style"] != {p["id"]: p["style"] for p in pairs["train"] + pairs["test"]}:
        raise ValueError("style labels differ from the FS2K annotations")


def save_split(split, path=SPLIT_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(split, f, separators=(",", ":"))


def load_split(path=SPLIT_PATH):
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Preprocessing (also used by the application backend) and cache
# --------------------------------------------------------------------------- #
def _flatten(image, mode):
    """Upright copy in `mode` ("RGB" or "L"); transparent pixels become white."""
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info:
        white = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(white, image.convert("RGBA"))
    return image.convert(mode)


def square_resize(image, size=IMAGE_SIZE):
    """Centre square crop (side = shorter side), then bicubic resize to size x size.
    Pillow's bicubic filter widens with the reduction factor, so downscaling is antialiased."""
    w, h = image.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    return image.crop((left, top, left + side, top + side)).resize((size, size), Image.BICUBIC)


def photo_to_uint8(image, size=IMAGE_SIZE):
    """PIL photo -> uint8 (size, size, 3)."""
    return np.asarray(square_resize(_flatten(image, "RGB"), size))


def sketch_to_uint8(image, size=IMAGE_SIZE):
    """PIL sketch -> uint8 (size, size)."""
    return np.asarray(square_resize(_flatten(image, "L"), size))


def preprocess_photo(image, size=IMAGE_SIZE):
    """Generator input for any face photo (dataset file, upload or webcam frame):
    float32 (3, size, size) in [-1, 1]. The training cache is built with the same steps."""
    return photo_to_uint8(image, size).transpose(2, 0, 1).astype(np.float32) / 127.5 - 1.0


def to_uint8(x):
    """Array in [-1, 1] (e.g. a generated sketch) -> uint8 pixel values, same shape."""
    return np.clip(np.rint((np.asarray(x, dtype=np.float32) + 1.0) * 127.5), 0, 255).astype(np.uint8)


def cache_path(split, cache_dir=None, size=IMAGE_SIZE):
    return Path(cache_dir or get_dir("CACHE_DIR")) / f"fs2k_{split}_{size}.npz"


def build_cache(pairs, out_path, size=IMAGE_SIZE):
    """Preprocess pairs once and save photos uint8 (N, size, size, 3), sketches uint8
    (N, size, size), styles int64 (N,) and ids (N,) to a compressed .npz file."""
    photos = np.empty((len(pairs), size, size, 3), dtype=np.uint8)
    sketches = np.empty((len(pairs), size, size), dtype=np.uint8)
    for i, p in enumerate(pairs):
        with Image.open(p["photo"]) as photo, Image.open(p["sketch"]) as sketch:
            photos[i] = photo_to_uint8(photo, size)
            sketches[i] = sketch_to_uint8(sketch, size)
    data = {
        "photos": photos,
        "sketches": sketches,
        "styles": np.array([p["style"] for p in pairs], dtype=np.int64),
        "ids": np.array([p["id"] for p in pairs]),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **data)
    return data


def load_cache(split, cache_dir=None):
    with np.load(cache_path(split, cache_dir)) as f:
        return {key: f[key] for key in ("photos", "sketches", "styles", "ids")}


def load_fs2k(smoke=False, cache_dir=None):
    """{"train"|"val"|"test": dict(photos, sketches, styles, ids)} from the cache;
    smoke=True returns a tiny synthetic dataset with the same layout instead."""
    if smoke:
        return make_smoke_data()
    missing = [s for s in SPLITS if not cache_path(s, cache_dir).exists()]
    if missing:
        raise FileNotFoundError(f"FS2K cache missing for {missing}: run python scripts/prepare_fs2k.py")
    return {s: load_cache(s, cache_dir) for s in SPLITS}


# --------------------------------------------------------------------------- #
# Synthetic data for smoke tests
# --------------------------------------------------------------------------- #
def _smoke_pair(style, rng, size):
    """A coloured 'face' photo and its pixel-aligned sketch drawn in a style-dependent way:
    0 thin grey lines, 1 thick black lines, 2 medium lines over diagonal hatching."""
    cx, cy = (int(v) for v in rng.integers(size * 3 // 8, size * 5 // 8 + 1, 2))
    rx, ry = int(rng.integers(size // 6, size // 4)), int(rng.integers(size // 5, size // 3))
    face = [cx - rx, cy - ry, cx + rx, cy + ry]
    eyes = [[cx + dx - 5, cy - ry // 3 - 4, cx + dx + 5, cy - ry // 3 + 4] for dx in (-rx // 2, rx // 2)]
    mouth = [cx - rx // 2, cy + ry // 2, cx + rx // 2, cy + ry // 2]

    photo = Image.new("RGB", (size, size), tuple(int(v) for v in rng.integers(0, 256, 3)))
    draw = ImageDraw.Draw(photo)
    draw.ellipse(face, fill=tuple(int(v) for v in rng.integers(120, 256, 3)))
    for eye in eyes:
        draw.ellipse(eye, fill=(30, 30, 30))
    draw.line(mouth, fill=(150, 40, 40), width=3)

    ink, width = [(110, 1), (0, 3), (60, 2)][style]
    sketch = Image.new("L", (size, size), 255)
    if style == 2:
        hatch = Image.new("L", (size, size), 255)
        for t in range(-size, size, 6):
            ImageDraw.Draw(hatch).line([(t, size), (t + size, 0)], fill=ink + 80, width=1)
        mask = Image.new("L", (size, size), 0)
        ImageDraw.Draw(mask).ellipse(face, fill=255)
        sketch = Image.composite(hatch, sketch, mask)
    draw = ImageDraw.Draw(sketch)
    draw.ellipse(face, outline=ink, width=width)
    for eye in eyes:
        draw.ellipse(eye, outline=ink, width=width)
    draw.line(mouth, fill=ink, width=width)
    return np.asarray(photo), np.asarray(sketch)


def make_smoke_data(n_per_style=(8, 4, 4), seed=0, size=IMAGE_SIZE):
    """Tiny synthetic stand-in for load_fs2k(): n_per_style pairs of every style in train/val/test."""
    rng = np.random.default_rng(seed)
    data = {}
    for split, n in zip(SPLITS, n_per_style):
        pairs = [(s, j, *_smoke_pair(s, rng, size)) for s in range(NUM_STYLES) for j in range(n)]
        data[split] = {
            "photos": np.stack([p[2] for p in pairs]),
            "sketches": np.stack([p[3] for p in pairs]),
            "styles": np.array([p[0] for p in pairs], dtype=np.int64),
            "ids": np.array([f"smoke/{split}_s{p[0]}_{p[1]:03d}" for p in pairs]),
        }
    return data


# --------------------------------------------------------------------------- #
# Paired dataset
# --------------------------------------------------------------------------- #
def paired_augment(x, hflip=True, max_zoom=MAX_ZOOM):
    """Random horizontal flip and pix2pix-style jitter on a stacked (C, H, W) photo+sketch tensor.

    Jitter upscales by a factor drawn from [1, max_zoom] (bilinear, so values stay in range)
    and crops a random H x W window; the unaugmented framing used at test time is included.
    Because photo and sketch are channels of one tensor, they always get the same transform.
    """
    if hflip and torch.rand(()) < 0.5:
        x = x.flip(-1)
    h, w = x.shape[-2:]
    zoom = 1.0 + (max_zoom - 1.0) * float(torch.rand(()))
    zh, zw = round(h * zoom), round(w * zoom)
    if (zh, zw) != (h, w):
        x = F.interpolate(x[None], size=(zh, zw), mode="bilinear", align_corners=False)[0]
        top = int(torch.randint(0, zh - h + 1, ()))
        left = int(torch.randint(0, zw - w + 1, ()))
        x = x[:, top:top + h, left:left + w]
    return x


class FS2KPairs(Dataset):
    """Paired photo/sketch dataset over cached (or smoke) arrays.

    Returns dict(photo float32 (3, H, W), sketch float32 (1, H, W), both in [-1, 1],
    style int index 0/1/2, id str). augment=True applies `paired_augment`; randomness
    comes from torch, which DataLoader seeds differently in every worker.
    """

    def __init__(self, data, augment=False, hflip=True, max_zoom=MAX_ZOOM):
        self.photos, self.sketches = data["photos"], data["sketches"]
        self.styles, self.ids = data["styles"], data["ids"]
        self.augment, self.hflip, self.max_zoom = augment, hflip, max_zoom

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        photo = torch.from_numpy(self.photos[i]).permute(2, 0, 1)
        sketch = torch.from_numpy(self.sketches[i])[None]
        x = torch.cat([photo, sketch]).float() / 127.5 - 1.0
        if self.augment:
            x = paired_augment(x, self.hflip, self.max_zoom)
        return {"photo": x[:3], "sketch": x[3:], "style": int(self.styles[i]), "id": str(self.ids[i])}
