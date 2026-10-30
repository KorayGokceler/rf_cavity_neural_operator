"""Shared fixtures and path setup for the test suite."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

# Repo root sys.path'e eklenmeli — testler `from src.*` import'larını kullanıyor
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# torch before NGSolve: with ngsolve loaded first, the field model's torch forward segfaults later in the
# process (their bundled OpenMP / BLAS runtimes clash); test modules that importorskip("ngsolve") at
# collection would otherwise decide the order.  train.py, FieldDataset and the web API import torch first.
try:
    import torch  # noqa: F401
except ImportError:
    pass


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


@pytest.fixture
def run_generator():
    """Run src/data_gen/dataset_generator_3d.py as a subprocess (the real CLI).  Its worker pool forks, and a fork
    of a process that already ran NGSolve threads hangs in the child — the test process usually has (the
    production CLI never does), so the CLI tests must not call gen.main() in-process."""
    def run(*argv):
        r = subprocess.run([sys.executable, str(REPO_ROOT / "src" / "data_gen" / "dataset_generator_3d.py"),
                            *map(str, argv)], cwd=REPO_ROOT, capture_output=True, text=True, timeout=900)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        return r
    return run
