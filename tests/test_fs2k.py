"""FS2K data pipeline (Task 4). Run: python -m pytest tests/test_fs2k.py -q

Most tests use the synthetic smoke data; tests marked `needs_fs2k` / `needs_cache` are
skipped when data/FS2K or the 128x128 cache is not available.
"""
import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.common.paths import get_dir
from src.data.fs2k import (
    IMAGE_SIZE, NUM_STYLES, SPLIT_PATH, FS2KPairs, cache_path, check_split, load_cache, load_split,
    make_smoke_data, make_split, official_pairs, paired_augment, photo_to_uint8, preprocess_photo,
    split_counts, to_uint8,
)

FS2K_DIR = get_dir("DATA_DIR") / "FS2K"
needs_fs2k = pytest.mark.skipif(not (FS2K_DIR / "anno_train.json").exists(), reason="data/FS2K not available")
needs_cache = pytest.mark.skipif(not cache_path("val").exists(), reason="FS2K cache not built")
SMOKE = make_smoke_data()


# --------------------------------------------------------------------------- #
# Smoke data and dataset
# --------------------------------------------------------------------------- #
def test_smoke_data_layout():
    for split, data in SMOKE.items():
        n = len(data["ids"])
        assert data["photos"].shape == (n, IMAGE_SIZE, IMAGE_SIZE, 3) and data["photos"].dtype == np.uint8
        assert data["sketches"].shape == (n, IMAGE_SIZE, IMAGE_SIZE) and data["sketches"].dtype == np.uint8
        assert data["styles"].dtype == np.int64
        assert np.bincount(data["styles"], minlength=NUM_STYLES).min() > 0
        assert len(set(data["ids"].tolist())) == n


def test_dataset_values_without_augmentation():
    data = SMOKE["val"]
    ds = FS2KPairs(data)
    item = ds[5]
    assert item["photo"].shape == (3, IMAGE_SIZE, IMAGE_SIZE) and item["photo"].dtype == torch.float32
    assert item["sketch"].shape == (1, IMAGE_SIZE, IMAGE_SIZE) and item["sketch"].dtype == torch.float32
    expected = torch.from_numpy(data["photos"][5]).permute(2, 0, 1).float() / 127.5 - 1
    assert torch.equal(item["photo"], expected)
    assert np.array_equal(to_uint8(item["sketch"][0].numpy()), data["sketches"][5])
    assert item["style"] == data["styles"][5] and item["id"] == data["ids"][5]

    batch = next(iter(DataLoader(FS2KPairs(SMOKE["train"], augment=True), batch_size=8, shuffle=True)))
    assert batch["style"].dtype == torch.int64 and len(batch["id"]) == 8
    for key in ("photo", "sketch"):
        assert batch[key].min() >= -1 and batch[key].max() <= 1


def test_augmentation_is_identical_for_photo_and_sketch():
    # photo channels == sketch: any transform not shared exactly would break the equality
    data = {**SMOKE["train"], "photos": np.repeat(SMOKE["train"]["sketches"][..., None], 3, axis=-1)}
    ds = FS2KPairs(data, augment=True)
    plain = FS2KPairs(data)
    torch.manual_seed(0)
    changed = 0
    for i in range(len(ds)):
        for _ in range(3):
            item = ds[i]
            for c in range(3):
                assert torch.equal(item["photo"][c], item["sketch"][0])
            assert item["sketch"].min() >= -1 and item["sketch"].max() <= 1
            changed += not torch.equal(item["sketch"], plain[i]["sketch"])
    assert changed > 0.5 * 3 * len(ds)  # augmentation really happens


def test_flip_only_gives_original_or_mirror():
    x = torch.rand(4, IMAGE_SIZE, IMAGE_SIZE) * 2 - 1
    seen = set()
    torch.manual_seed(1)
    for _ in range(20):
        y = paired_augment(x, hflip=True, max_zoom=1.0)
        if torch.equal(y, x):
            seen.add("same")
        else:
            assert torch.equal(y, x.flip(-1))
            seen.add("flip")
    assert seen == {"same", "flip"}


def test_jitter_keeps_shape_and_moves_content():
    x = torch.zeros(4, IMAGE_SIZE, IMAGE_SIZE)
    x[:, 60:68, 60:68] = 1.0
    torch.manual_seed(2)
    outs = [paired_augment(x, hflip=False) for _ in range(30)]
    assert all(o.shape == x.shape for o in outs)
    assert any(not torch.allclose(o, x) for o in outs)


# --------------------------------------------------------------------------- #
# Split logic (synthetic ids)
# --------------------------------------------------------------------------- #
def _fake_pairs(counts=(357, 350, 351), n_test=10):
    train = [{"id": f"photo1/image{s}{i:04d}", "style": s} for s, n in enumerate(counts) for i in range(n)]
    test = [{"id": f"photo3/image{i:04d}", "style": i % NUM_STYLES} for i in range(n_test)]
    return {"train": sorted(train, key=lambda p: p["id"]), "test": test}


def test_split_is_stratified_disjoint_and_deterministic():
    pairs = _fake_pairs()
    split = make_split(pairs)
    assert split["counts"]["val"] == [54, 52, 53]  # 15% of 357/350/351, largest remainder, total 159
    assert len(split["val"]) == round(0.15 * 1058)
    assert not set(split["train"]) & set(split["val"])
    assert set(split["train"]) | set(split["val"]) == {p["id"] for p in pairs["train"]}
    assert split["test"] == [p["id"] for p in pairs["test"]]
    check_split(split, pairs)
    assert make_split(pairs) == split
    assert make_split(pairs, seed=7)["val"] != split["val"]
    for counts in [(10, 10, 10), (7, 1, 0), (100, 3, 41)]:
        s = make_split(_fake_pairs(counts))
        exact = 0.15 * np.array(counts)
        assert np.all(np.abs(np.array(s["counts"]["val"]) - exact) < 1)
        assert len(s["val"]) == round(0.15 * sum(counts))


def test_check_split_rejects_leaks():
    pairs = _fake_pairs()
    split = make_split(pairs)
    leaked = {**split, "val": split["val"] + split["train"][:1]}
    with pytest.raises(ValueError):
        check_split(leaked, pairs)


# --------------------------------------------------------------------------- #
# Preprocessing shared with the application
# --------------------------------------------------------------------------- #
def test_preprocess_photo_crops_the_centre_square():
    w, h = 300, 200  # landscape: the centre square is columns 50..250
    arr = np.zeros((h, w, 3), np.uint8)
    arr[:, 50:250] = (200, 100, 50)
    out = preprocess_photo(Image.fromarray(arr))
    assert out.shape == (3, IMAGE_SIZE, IMAGE_SIZE) and out.dtype == np.float32
    assert np.allclose(out.transpose(1, 2, 0), np.array([200, 100, 50]) / 127.5 - 1, atol=1e-6)
    tall = preprocess_photo(Image.fromarray(np.ascontiguousarray(arr.transpose(1, 0, 2))))
    assert np.allclose(tall, out, atol=1e-6)


def test_preprocess_photo_handles_alpha_grayscale_and_exif(tmp_path):
    rgba = np.zeros((64, 64, 4), np.uint8)
    rgba[..., 3] = 0  # fully transparent -> white
    assert np.all(preprocess_photo(Image.fromarray(rgba, "RGBA")) == 1.0)
    gray = preprocess_photo(Image.fromarray(np.full((50, 80), 64, np.uint8), "L"))
    assert np.allclose(gray, 64 / 127.5 - 1)

    arr = np.zeros((40, 60, 3), np.uint8)
    arr[:, :30] = 255  # left half white
    img = Image.fromarray(arr)
    exif = img.getexif()
    exif[0x0112] = 6  # stored rotated: viewers rotate 90 degrees clockwise
    upright = Image.fromarray(np.ascontiguousarray(np.rot90(arr, k=-1)))
    path = tmp_path / "rotated.jpg"
    img.save(path, exif=exif, quality=100)
    with Image.open(path) as tagged:
        got = preprocess_photo(tagged)
    assert np.abs(got - preprocess_photo(upright)).mean() < 0.05


def test_preprocess_matches_cache_path_and_to_uint8_inverts():
    img = Image.fromarray(SMOKE["test"]["photos"][0])
    x = preprocess_photo(img)
    assert np.array_equal(to_uint8(x).transpose(1, 2, 0), photo_to_uint8(img))
    assert x.min() >= -1 and x.max() <= 1


# --------------------------------------------------------------------------- #
# Real data (skipped when unavailable)
# --------------------------------------------------------------------------- #
def test_committed_split_manifest():
    if not SPLIT_PATH.exists():
        pytest.skip("manifests/fs2k_split.json not generated yet")
    split = load_split()
    assert (len(split["train"]), len(split["val"]), len(split["test"])) == (899, 159, 1046)
    assert not set(split["train"]) & set(split["val"])
    assert not (set(split["train"]) | set(split["val"])) & set(split["test"])
    assert split_counts(split) == {"train": [303, 298, 298], "val": [54, 52, 53], "test": [619, 381, 46]}
    assert split["counts"] == split_counts(split) and split["seed"] == 42


@needs_fs2k
def test_official_pairs_and_split_reproduce():
    pairs = official_pairs(FS2K_DIR)  # raises on missing/unannotated/size-mismatched pairs
    assert (len(pairs["train"]), len(pairs["test"])) == (1058, 1046)
    for p in pairs["train"][:20] + pairs["test"][:20]:
        source, stem = p["id"].split("/")
        assert p["photo"].parent.name == source and p["photo"].stem == stem
        assert p["sketch"].parent.name == "sketch" + source[-1] and p["sketch"].stem == stem.replace("image", "sketch")
    if SPLIT_PATH.exists():
        committed = load_split()
        check_split(committed, pairs)
        assert make_split(pairs) == committed


@needs_fs2k
@needs_cache
def test_cache_matches_split_and_preprocessing():
    split = load_split()
    for name in ("val", "test"):
        data = load_cache(name)
        assert data["ids"].tolist() == split[name]
        assert data["styles"].tolist() == [split["style"][i] for i in split[name]]
        assert data["photos"].shape == (len(split[name]), IMAGE_SIZE, IMAGE_SIZE, 3)
        assert data["sketches"].shape == (len(split[name]), IMAGE_SIZE, IMAGE_SIZE)
    # the application path (preprocess_photo on the original file) reproduces the cache exactly
    pairs = {p["id"]: p for p in official_pairs(FS2K_DIR, check_sizes=False)["train"]}
    val = load_cache("val")
    for k in (0, len(val["ids"]) - 1):
        with Image.open(pairs[str(val["ids"][k])]["photo"]) as im:
            assert np.array_equal(to_uint8(preprocess_photo(im)).transpose(1, 2, 0), val["photos"][k])
