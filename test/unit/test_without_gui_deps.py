"""The carm CLI must work when the optional GUI dependencies are not installed.

Simulates a minimal install (``pip install carm-roofline`` without the ``[gui]``
extra) by running the CLI in a fresh subprocess with ``plotly``, ``dash``, and
``dash_bootstrap_components`` blocked in ``sys.modules``.  Any module that
imports them raises ImportError, exactly as when the distributions are missing.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

# Optional GUI extras; the CLI must be usable without them.
_GUI_MODULES = ("dash", "dash_bootstrap_components", "plotly")

_REPO_ROOT = Path(__file__).resolve().parents[1]

_SCRIPT = """\
import sys
for _module in {modules!r}:
    sys.modules[_module] = None
{calls}
"""


def _run_cli(argv: list[str], extra_calls: str = "") -> subprocess.CompletedProcess[str]:
    """Run the carm CLI in a fresh interpreter with the GUI modules blocked."""
    script = _SCRIPT.format(
        modules=_GUI_MODULES,
        calls=extra_calls + f"from carm_roofline.carm import main\nraise SystemExit(main({argv!r}))\n",
    )
    env = {**os.environ, "PYTHONPATH": str(_REPO_ROOT)}
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        env=env,
    )


@pytest.mark.parametrize("argv", [["--help"], ["benchmark", "--help"], ["profile", "--help"]])
def test_cli_help_without_gui_deps(argv: list[str]) -> None:
    result = _run_cli(argv)
    assert result.returncode == 0, result.stderr
    assert "halted" not in result.stderr  # poisoned-import failure signature
    assert result.stdout


def test_benchmark_emit_config_without_gui_deps(tmp_path: Path) -> None:
    output_path = tmp_path / "topology.toml"
    result = _run_cli(["benchmark", "--emit-config", str(output_path)])
    assert result.returncode == 0, result.stderr
    assert output_path.is_file()


def test_profile_command_without_gui_deps() -> None:
    # No command -> profile_main raises UserError; the point is the failure is
    # the expected one, not a poisoned plotly import.
    result = _run_cli(["profile"])
    assert result.returncode != 0
    assert "halted" not in result.stderr


def test_gui_command_reports_missing_deps() -> None:
    result = _run_cli(["gui"])
    assert result.returncode == 1
    assert "Failed to import GUI dependencies" in result.stderr


def test_gui_config_importable_without_gui_deps() -> None:
    result = _run_cli([], extra_calls="from carm_roofline.gui.config import GUIConfig, GUISettings\n")
    assert result.returncode == 0, result.stderr
