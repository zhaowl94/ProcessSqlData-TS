"""Test package bootstrap for the local src/ layout."""

from contextlib import contextmanager
from pathlib import Path
import shutil
import sys
from typing import Iterator
import uuid


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


@contextmanager
def temporary_directory() -> Iterator[str]:
    """Create disposable test data only inside the ignored repository runtime."""

    temporary_root = REPOSITORY_ROOT / ".runtime" / "test-temp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    path = temporary_root / f"case-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield str(path)
    finally:
        shutil.rmtree(path)
