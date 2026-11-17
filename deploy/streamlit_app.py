"""Streamlit Community Cloud entrypoint.

Cloud installs deploy/requirements.txt (it is found before the root uv.lock because it
sits next to this file) but does not install the project itself, so put src/ on the
import path and run the real dashboard script on every rerun.
"""

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

runpy.run_path(str(ROOT / "src" / "pharmasignal" / "dashboard" / "app.py"), run_name="__main__")
