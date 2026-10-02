"""Corruption definitions shared by training, evaluation and the application backend.

Pure NumPy (no torch) so the FastAPI backend can reuse exactly the same code.
Images are float32 arrays in [0, 1] with shape (H, W, 3).

A corruption is described by a JSON-serialisable *spec* dict. Sampling a spec
uses randomness; applying a spec is fully deterministic. Manifests store specs,
so validation/test corruptions can be reproduced exactly.
"""
import numpy as np

# Class order is shared by the classifier (Task 2) and the MoE gate (Task 3):
# index 0 = clean/identity branch, 1 = salt-and-pepper, 2 = blur, 3 = occlusion.
CLASSES = ("clean", "salt", "blur", "occlusion")
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
LEVELS = ("low", "medium", "high")

# Training ranges (assignment spec)
SALT_P_RANGE = (0.02, 0.15)
BLUR_KERNELS = (3, 5, 7)
BLUR_SIGMA_RANGE = (0.5, 2.5)
OCC_N_RANGE = (1, 3)
OCC_COVER_RANGE = (0.10, 0.35)

# Fixed test severities (assignment spec)
TEST_LEVELS = {
    "salt": {"low": {"p": 0.03}, "medium": {"p": 0.08}, "high": {"p": 0.15}},
    "blur": {"low": {"k": 3, "sigma": 0.7}, "medium": {"k": 5, "sigma": 1.5}, "high": {"k": 7, "sigma": 2.5}},
    "occlusion": {"low": {"n": 1, "cover": 0.10}, "medium": {"n": 2, "cover": 0.20}, "high": {"n": 3, "cover": 0.35}},
}

# Bin edges (midpoints between the test levels) used to assign a severity level
# to randomly sampled training/validation corruptions for per-severity reporting.
# Salt is binned on p, occlusion on the covered fraction and blur on blur_strength():
# the blur test levels (3, 0.7), (5, 1.5), (7, 2.5) have strengths 0.647, 1.195, 1.765.
_LEVEL_EDGES = {"salt": (0.055, 0.115), "blur": (0.921, 1.480), "occlusion": (0.15, 0.275)}


def blur_strength(k, sigma):
    """Standard deviation (pixels) of the k-tap Gaussian kernel that is actually applied.

    The kernel is truncated to k taps, so a large sigma flattens it towards a box
    filter and sigma alone mis-ranks blur strength: (3, 2.5) gives 0.81, which is
    milder than (5, 1.5) at 1.20.
    """
    g = gaussian_kernel1d(k, sigma).astype(np.float64)
    x = np.arange(k) - (k - 1) / 2.0
    return float(np.sqrt((g * x**2).sum()))


def severity_level(spec):
    """Map a spec to 'low' / 'medium' / 'high' (None for clean)."""
    t = spec["type"]
    if t == "clean":
        return None
    if t == "blur":
        value = blur_strength(spec["k"], spec["sigma"])
    else:
        value = spec["p"] if t == "salt" else spec["cover"]
    lo, hi = _LEVEL_EDGES[t]
    return LEVELS[0] if value <= lo else LEVELS[1] if value <= hi else LEVELS[2]


# --------------------------------------------------------------------------- #
# Occlusion rectangle generation
# --------------------------------------------------------------------------- #
def _union_cover(rects, size):
    mask = np.zeros((size, size), dtype=bool)
    for y, x, h, w in rects:
        mask[y:y + h, x:x + w] = True
    return float(mask.mean())


def make_occlusion_rects(n, cover, rng, size=128, tol=0.01, max_tries=500):
    """Place `n` non-overlapping black rectangles whose union covers ~`cover` of the image.

    Returns (rects, achieved_cover) where rects is a list of [y, x, h, w].
    The target area is split among the rectangles (Dirichlet shares, each at least
    half of an equal share) and each rectangle gets a random aspect ratio in [1/2, 2].
    """
    total = cover * size * size
    best = None
    for _ in range(max_tries):
        shares = rng.dirichlet(np.full(n, 2.0)) if n > 1 else np.ones(1)
        if n > 1 and shares.min() < 0.5 / n:
            continue
        rects, placed = [], True
        for share in shares:
            area = share * total
            aspect = np.exp(rng.uniform(np.log(0.5), np.log(2.0)))  # h / w
            h = int(np.clip(round(np.sqrt(area * aspect)), 1, size))
            w = int(np.clip(round(area / h), 1, size))
            for _ in range(50):
                y = int(rng.integers(0, size - h + 1))
                x = int(rng.integers(0, size - w + 1))
                overlaps = any(
                    y < ry + rh and ry < y + h and x < rx + rw and rx < x + w for ry, rx, rh, rw in rects
                )
                if not overlaps:
                    rects.append([y, x, h, w])
                    break
            else:
                placed = False
                break
        if not placed:
            continue
        achieved = _union_cover(rects, size)
        if abs(achieved - cover) <= tol:
            return rects, achieved
        if best is None or abs(achieved - cover) < abs(best[1] - cover):
            best = (rects, achieved)
    if best is None:
        raise RuntimeError(f"could not place {n} rectangles covering {cover:.2f}")
    return best


# --------------------------------------------------------------------------- #
# Spec sampling
# --------------------------------------------------------------------------- #
def sample_spec(ctype, rng, size=128):
    """Sample a corruption spec from the *training* ranges."""
    if ctype == "clean":
        spec = {"type": "clean"}
    elif ctype == "salt":
        spec = {"type": "salt", "p": float(rng.uniform(*SALT_P_RANGE)), "seed": int(rng.integers(2**31))}
    elif ctype == "blur":
        spec = {"type": "blur", "k": int(rng.choice(BLUR_KERNELS)), "sigma": float(rng.uniform(*BLUR_SIGMA_RANGE))}
    elif ctype == "occlusion":
        n = int(rng.integers(OCC_N_RANGE[0], OCC_N_RANGE[1] + 1))
        target = float(rng.uniform(*OCC_COVER_RANGE))
        # n=1 at 35% is a single 76x76 box; all combinations are feasible at 128x128.
        # Rounding to whole pixels can miss the target slightly, so resample until
        # the achieved union stays inside the required range.
        while True:
            rects, cover = make_occlusion_rects(n, target, rng, size)
            if OCC_COVER_RANGE[0] <= cover <= OCC_COVER_RANGE[1]:
                break
        spec = {"type": "occlusion", "n": n, "cover": cover, "rects": rects}
    else:
        raise ValueError(f"unknown corruption type: {ctype}")
    spec["level"] = severity_level(spec)
    return spec


def level_spec(ctype, level, rng, size=128):
    """Build a spec with one of the fixed *test* severity levels."""
    if ctype == "clean":
        return {"type": "clean", "level": None}
    params = TEST_LEVELS[ctype][level]
    if ctype == "salt":
        return {"type": "salt", "p": params["p"], "seed": int(rng.integers(2**31)), "level": level}
    if ctype == "blur":
        return {"type": "blur", "k": params["k"], "sigma": params["sigma"], "level": level}
    rects, cover = make_occlusion_rects(params["n"], params["cover"], rng, size)
    return {"type": "occlusion", "n": params["n"], "cover": cover, "rects": rects, "level": level}


# --------------------------------------------------------------------------- #
# Applying specs
# --------------------------------------------------------------------------- #
def salt_and_pepper(img, p, seed):
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    hit = rng.random((h, w)) < p
    white = rng.random((h, w)) < 0.5
    out = img.copy()
    out[hit & white] = 1.0
    out[hit & ~white] = 0.0
    return out


def gaussian_kernel1d(k, sigma):
    x = np.arange(k, dtype=np.float64) - (k - 1) / 2.0
    g = np.exp(-(x**2) / (2.0 * sigma**2))
    return (g / g.sum()).astype(np.float32)


def gaussian_blur(img, k, sigma):
    """Separable Gaussian blur with reflect padding (matches torchvision's gaussian_blur)."""
    g = gaussian_kernel1d(k, sigma)
    pad = k // 2
    h, w = img.shape[:2]
    padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    tmp = sum(g[i] * padded[i:i + h] for i in range(k))
    return sum(g[j] * tmp[:, j:j + w] for j in range(k)).astype(np.float32)


def occlude(img, rects):
    out = img.copy()
    for y, x, h, w in rects:
        out[y:y + h, x:x + w] = 0.0
    return out


def apply_spec(img, spec):
    """Apply a corruption spec to a float32 (H, W, 3) image in [0, 1]."""
    t = spec["type"]
    if t == "clean":
        return img.copy()
    if t == "salt":
        return salt_and_pepper(img, spec["p"], spec["seed"])
    if t == "blur":
        return gaussian_blur(img, spec["k"], spec["sigma"])
    if t == "occlusion":
        return occlude(img, spec["rects"])
    raise ValueError(f"unknown corruption type: {t}")
