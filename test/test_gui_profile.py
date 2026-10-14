from __future__ import annotations

from typing import Any

import pytest

from carm_roofline.gui.components import build_profile_panel
from carm_roofline.gui.ids import ProfilePanelID
from carm_roofline.gui.profile_runner import build_profile_command, command_preview

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
    elif children is not None:
        return _find_component(children, component_id)
    return None


def _profile_values(**changes: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "backend": "perf",
        "command": "./bin/my app --input 'input file.dat'",
        "aggregation": "region_merged",
        "machine_name": "test host",
        "app_name": "test app",
        "output_dir": "/tmp/profile results",
        "metrics": ["cache-residency", "cache-line-utilization"],
        "isa": ["x86_avx2", "x86_scalar"],
        "data_type": "f64",
        "merge_runs": True,
        "keep_artifacts": True,
        "no_papi_cache": True,
        "perf_interval": 100.0,
    }
    values.update(changes)
    return values


def test_profile_command_preserves_argv_and_backend_options() -> None:
    command = build_profile_command(_profile_values())

    assert command[command.index("--backend") + 1] == "perf"
    assert command[command.index("--aggregation") + 1] == "region_merged"
    assert command[command.index("--perf-interval") + 1] == "100"
    assert "--no-papi-cache" not in command
    assert command[command.index("--machine-name") + 1] == "test host"
    assert command[command.index("--output-dir") + 1] == "/tmp/profile results"
    assert command[command.index("--") + 1 :] == ["./bin/my", "app", "--input", "input file.dat"]
    assert command_preview(command).find("'input file.dat'") >= 0


def test_profile_command_emits_papi_only_options() -> None:
    command = build_profile_command(_profile_values(backend="papi", perf_interval=100))

    assert "--no-papi-cache" in command
    assert "--perf-interval" not in command


@pytest.mark.parametrize("interval", [0, -1, 1.5, "not-a-number"])
def test_profile_command_rejects_invalid_perf_intervals(interval: object) -> None:
    with pytest.raises(ValueError, match="positive whole number"):
        build_profile_command(_profile_values(perf_interval=interval))


def test_profile_command_requires_application_command() -> None:
    with pytest.raises(ValueError, match="enter an application command"):
        build_profile_command(_profile_values(command="  "))



def test_profile_page_output_directory_is_blank_with_default_placeholder() -> None:
    panel = build_profile_panel("/tmp/carm-results")
    output_dir = _find_component(panel, ProfilePanelID.OUTPUT_DIR)

    assert output_dir.value == ""
    assert output_dir.placeholder == "/tmp/carm-results"
