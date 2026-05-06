"""Shared fixtures and path setup for the test suite."""
import sys
from pathlib import Path

# Repo root sys.path'e eklenmeli — testler `from src.*` import'larını kullanıyor
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
