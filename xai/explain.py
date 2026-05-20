"""
Shim so you can run: python3 xai/explain.py [--patient-id N ...]
Implementation lives in the repo root explain.py.
"""
import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("fedxai_explain", _ROOT / "explain.py")
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

if __name__ == "__main__":
    _mod.main()
