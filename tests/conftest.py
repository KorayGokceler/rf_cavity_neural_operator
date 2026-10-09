"""Shared fixtures and path setup for the test suite."""
import importlib.util
import sys
from pathlib import Path

import pytest

# Repo root sys.path'e eklenmeli — testler `from src.*` import'larını kullanıyor
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _gmsh_available() -> bool:
    """gmsh wheel imports fail without libGLU/libXrender etc., so try the import."""
    if importlib.util.find_spec("gmsh") is None:
        return False
    try:
        import gmsh  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests marked `@pytest.mark.gmsh` when gmsh cannot be imported."""
    gmsh_items = [item for item in items if item.get_closest_marker("gmsh")]
    if not gmsh_items or _gmsh_available():
        return
    skip = pytest.mark.skip(reason="gmsh (or its system libraries) not available")
    for item in gmsh_items:
        item.add_marker(skip)
