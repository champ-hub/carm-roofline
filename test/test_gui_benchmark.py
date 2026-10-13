from __future__ import annotations

from typing import Any

import pytest

from carm_roofline.gui.benchmark_runner import build_benchmark_command
from carm_roofline.gui.components import build_benchmark_panel
from carm_roofline.gui.ids import BenchmarkPanelID

pytestmark = pytest.mark.unit


def _find_component(node: Any, component_id: str) -> Any | None:
    if getattr(node, "id", None) == component_id:
        return node
    children = getattr(node, "children", None)
    if isinstance(children, list):
        for child in children:
            found = _find_component(child, component_id)
            if found is not None:
                return found
    return None


def test_benchmark_panel_has_compact_duration_label_and_help() -> None:
    panel = build_benchmark_panel("/tmp/carm-results")

    labels = [node for node in panel.children[2].children if getattr(node, "className", None) == "benchmark-field"]
    duration_field = next(field for field in labels if field.children[0].children == "Run time (s)")
    duration_label = duration_field.children[0]

    assert "title" not in duration_label.to_plotly_json()["props"]
    assert duration_field.children[2].target == duration_label.id
    assert duration_field.children[2].placement == "top"
    assert _find_component(panel, BenchmarkPanelID.DATA_TYPES).value == ["f32"]
    assert _find_component(panel, BenchmarkPanelID.START) is not None
    assert _find_component(panel, BenchmarkPanelID.PROGRESS) is not None
    advanced = panel.children[3]
    assert advanced.to_plotly_json()["props"].get("open", False) is False


def test_benchmark_navigation_button_is_present_and_active_in_both_modes() -> None:
    from carm_roofline.gui.components import build_navbar
    from carm_roofline.gui.config import GUIMode
    from carm_roofline.gui.data import ActivePanel
    from carm_roofline.gui.ids import NavbarID

    for mode in (GUIMode.CARM, GUIMode.PARAVER):
        navbar = build_navbar(ActivePanel.BENCHMARK, mode)
        button = _find_component(navbar, NavbarID.BTN_BENCHMARK)

        assert button is not None
        assert button.children == "Benchmark"
        assert "navbar-btn--active" in button.className


def test_benchmark_command_keeps_simulator_as_one_argument_and_resolves_auto_isa() -> None:
    command = build_benchmark_command(
        {
            "test": "arithmetic",
            "isa": ["auto"],
            "data_types": ["f32"],
            "threads": "1, 4",
            "test_time": 2,
            "simulator": "qemu-x86_64 {binary}",
            "output_dir": "/tmp/benchmark results",
            "output_formats": ["jsonl"],
        }
    )

    assert "--isa" not in command
    assert command[command.index("--threads") + 1 : command.index("--threads") + 3] == ["1", "4"]
    assert command[command.index("--sim-cmd") + 1] == "qemu-x86_64 {binary}"
    assert command[command.index("--output-dir") + 1] == "/tmp/benchmark results"
    assert command[-2:] == ["--progress-format", "jsonl"]


def test_benchmark_job_tracks_measurement_progress_and_errors() -> None:
    from carm_roofline.gui.benchmark_runner import BenchmarkJob

    job = BenchmarkJob()
    job.update_event('CARM_PROGRESS {"event":"measurement","completed":3,"total":5}')
    snapshot = job.snapshot()

    assert snapshot["status"] == "Running: 3 / 5 measurements"
    assert snapshot["completed"] == 3
    assert snapshot["total"] == 5

    job.update_event('CARM_PROGRESS {"event":"error","message":"compile failed"}')
    assert job.snapshot()["status"] == "Failed: compile failed"
