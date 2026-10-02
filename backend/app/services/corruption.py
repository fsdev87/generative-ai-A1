"""Corruptions applied by the API, built on src/data/corruptions.py (the exact code used in training).

A request names a corruption type plus either a fixed test level (via level_spec) or custom
parameters inside the training ranges, and an optional seed. The seed initialises the random
generator that draws the random parts of the spec (the salt-and-pepper mask seed, the occlusion
rectangles) in the same order as level_spec, so the same request with the same seed always
gives the same image. Without a seed a random one is drawn; it is always returned.
"""
import secrets
import sys
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

try:
    from src.data import corruptions
except ModuleNotFoundError:  # repository checkout with only backend/ on sys.path
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from src.data import corruptions

from ..errors import ApiError
from .images import quantize

CLASSES = corruptions.CLASSES
LEVELS = corruptions.LEVELS
TEST_LEVELS = corruptions.TEST_LEVELS
SEED_LIMIT = 2**31

# Custom parameters of each type: integer parameters list their allowed values, float
# parameters give an inclusive (low, high) training range.
CUSTOM_PARAMS = {
    "salt": {"p": corruptions.SALT_P_RANGE},
    "blur": {"k": corruptions.BLUR_KERNELS, "sigma": corruptions.BLUR_SIGMA_RANGE},
    "occlusion": {
        "n": tuple(range(corruptions.OCC_N_RANGE[0], corruptions.OCC_N_RANGE[1] + 1)),
        "cover": corruptions.OCC_COVER_RANGE,
    },
}
INTEGER_PARAMS = ("k", "n")


@dataclass(frozen=True)
class CorruptionParams:
    """Optional request fields describing a corruption (everything except its type)."""

    level: str | None = None
    p: float | None = None
    k: int | None = None
    sigma: float | None = None
    n: int | None = None
    cover: float | None = None
    seed: int | None = None

    def custom(self) -> dict:
        """The custom parameters that were given."""
        names = ("p", "k", "sigma", "n", "cover")
        return {name: getattr(self, name) for name in names if getattr(self, name) is not None}

    def is_empty(self) -> bool:
        return all(getattr(self, f.name) is None for f in fields(self))


@dataclass(frozen=True)
class AppliedCorruption:
    spec: dict  # exactly what corruptions.apply_spec receives (the manifest format)
    seed: int  # request seed: sending it again with the same fields reproduces the spec
    custom: bool  # custom parameters (True) or a fixed test level (False)

    def describe(self) -> dict:
        """Fields of the CorruptionSettings response model."""
        spec = self.spec
        return {
            "type": spec["type"],
            "level": spec["level"],
            "custom": self.custom,
            "seed": self.seed,
            "params": {name: spec[name] for name in CUSTOM_PARAMS.get(spec["type"], {})},
            "rects": spec.get("rects"),
            "spec": spec,
        }


def build_spec(ctype: str, params: CorruptionParams) -> AppliedCorruption:
    """Validate the request fields and turn them into a corruption spec."""
    custom = params.custom()
    if ctype == "clean" and (custom or params.level):
        raise ApiError(400, "corruption=clean takes no level or parameters.")
    if custom and params.level:
        raise ApiError(400, "Give either a level or custom parameters, not both.")
    if ctype != "clean" and not custom and not params.level:
        names = " and ".join(CUSTOM_PARAMS[ctype])
        raise ApiError(400, f"corruption={ctype} needs a level (low, medium or high) or custom parameters ({names}).")
    if params.seed is not None and not 0 <= params.seed < SEED_LIMIT:
        raise ApiError(422, f"seed must be between 0 and {SEED_LIMIT - 1}.")
    seed = secrets.randbelow(SEED_LIMIT) if params.seed is None else params.seed
    rng = np.random.default_rng(seed)
    if custom:
        spec = _custom_spec(ctype, custom, rng)
    else:
        spec = corruptions.level_spec(ctype, params.level, rng)
    return AppliedCorruption(spec, seed, bool(custom))


def _custom_spec(ctype: str, values: dict, rng: np.random.Generator) -> dict:
    """Like corruptions.level_spec (same structure and random draws) with the given parameters."""
    allowed = CUSTOM_PARAMS[ctype]
    unknown = [name for name in values if name not in allowed]
    if unknown:
        raise ApiError(400, f"{', '.join(unknown)} is not a parameter of corruption={ctype} "
                            f"(it takes {' and '.join(allowed)}).")
    missing = [name for name in allowed if name not in values]
    if missing:
        raise ApiError(400, f"Custom corruption={ctype} needs {' and '.join(allowed)}; missing {', '.join(missing)}.")
    for name, value in values.items():
        _check_value(name, value, allowed[name])

    if ctype == "salt":
        spec = {"type": "salt", "p": float(values["p"]), "seed": int(rng.integers(2**31))}
    elif ctype == "blur":
        spec = {"type": "blur", "k": int(values["k"]), "sigma": float(values["sigma"])}
    else:
        n = int(values["n"])
        rects, cover = corruptions.make_occlusion_rects(n, float(values["cover"]), rng)
        spec = {"type": "occlusion", "n": n, "cover": cover, "rects": rects}
    spec["level"] = corruptions.severity_level(spec)
    return spec


def _check_value(name, value, allowed):
    if name in INTEGER_PARAMS:
        if value not in allowed:
            raise ApiError(422, f"{name}={value} is not allowed; use one of {', '.join(map(str, allowed))}.")
    elif not allowed[0] <= value <= allowed[1]:  # also rejects NaN
        raise ApiError(422, f"{name}={value} is outside the training range [{allowed[0]}, {allowed[1]}].")


def corrupt(clean: np.ndarray, spec: dict) -> np.ndarray:
    """Apply a spec with the training code, then round to 8 bits.

    The rounding (at most half a grey level, only blur produces in-between values) makes the
    model input identical to the PNG that is returned, so re-uploading that PNG reproduces
    exactly the same model input.
    """
    return quantize(corruptions.apply_spec(clean, spec))
