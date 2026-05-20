"""Entry for `streamlit run dashboard/app.py` — loads dashboard code from repo root `app.py`."""
import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("fedxai_streamlit_app", _ROOT / "app.py")
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
