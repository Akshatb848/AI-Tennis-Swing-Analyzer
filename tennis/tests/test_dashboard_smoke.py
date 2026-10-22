"""Smoke tests for the Streamlit dashboard (tennis/dashboard/tennis_dashboard.py)."""

import os
import subprocess
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD = REPO_ROOT / "tennis" / "dashboard" / "tennis_dashboard.py"


def test_dashboard_renders_without_exceptions():
    at = AppTest.from_file(str(DASHBOARD), default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    # Without live sessions the dashboard falls back to labelled sample data.
    assert at.title or at.markdown


def test_dashboard_imports_when_run_as_a_script():
    """`streamlit run tennis/dashboard/tennis_dashboard.py` puts only the script's
    folder on sys.path; the dashboard must still be able to import `tennis`."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    code = (
        "import runpy, sys; "
        f"sys.path[0] = {str(DASHBOARD.parent)!r}; "
        f"runpy.run_path({str(DASHBOARD)!r}, run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(DASHBOARD.parent),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert "No module named 'tennis'" not in result.stderr
    assert result.returncode == 0, result.stderr[-2000:]
