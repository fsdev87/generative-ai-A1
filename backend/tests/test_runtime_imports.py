"""The backend runtime must not need torch: the Docker image installs only backend/requirements.txt
and copies src/data/corruptions.py alone."""
import subprocess
import sys

from conftest import BACKEND_DIR, REPO_ROOT

CHECK = """
import sys
sys.path[:0] = [{repo!r}, {backend!r}]
import app.main
heavy = sorted(m for m in ("torch", "torchvision", "src.data.pets") if m in sys.modules)
assert not heavy, heavy
"""


def test_backend_imports_without_torch():
    code = CHECK.format(repo=str(REPO_ROOT), backend=str(BACKEND_DIR))
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
