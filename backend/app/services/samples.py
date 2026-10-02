"""Bundled clean sample images: app/samples/pets/ (Tasks 1-3) and app/samples/faces/ (Task 4)."""
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from ..errors import ApiError

log = logging.getLogger("uvicorn.error")

CATEGORIES = ("pets", "faces")
MEDIA_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
               ".webp": "image/webp", ".bmp": "image/bmp"}


@dataclass(frozen=True)
class Sample:
    id: str
    category: str
    path: Path

    @property
    def media_type(self) -> str:
        return MEDIA_TYPES[self.path.suffix.lower()]


class SampleCatalog:
    """Scans the sample folders once at startup. A sample's id is '<folder>-<file stem>'."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.samples: dict[str, Sample] = {}
        for category in CATEGORIES:
            folder = self.root / category
            for path in sorted(folder.iterdir()) if folder.is_dir() else []:
                if not path.is_file() or path.suffix.lower() not in MEDIA_TYPES:
                    continue
                sample_id = f"{category}-{_slug(path.stem)}"
                if sample_id in self.samples:
                    log.warning("skipping sample %s: id %s is already used", path, sample_id)
                    continue
                self.samples[sample_id] = Sample(sample_id, category, path)

    def select(self, category: str | None = None) -> list[Sample]:
        return [s for s in self.samples.values() if category is None or s.category == category]

    def get(self, sample_id: str) -> Sample:
        if sample_id not in self.samples:
            raise ApiError(404, f"Unknown sample '{sample_id}'; GET /api/samples lists the available ids.")
        return self.samples[sample_id]

    def counts(self) -> dict[str, int]:
        return {category: len(self.select(category)) for category in CATEGORIES}


def _slug(text: str) -> str:
    """URL-safe id part: lower case letters, digits, '_' and '-'."""
    return re.sub(r"[^a-z0-9_]+", "-", text.lower()).strip("-") or "image"
