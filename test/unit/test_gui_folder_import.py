"""Focused validation and merge tests for GUI folder imports."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from carm_roofline.gui.callbacks.folder_import import _load_payload, _merge_imports
from carm_roofline.gui.callbacks.profile import _refresh_profile_apps
from carm_roofline.roofline_assembly import ApplicationRecord

pytestmark = pytest.mark.unit


def _benchmarks(machine: str) -> str:
    common = {"machine": machine, "isa": "test-isa", "num_threads": 1, "data_type": "f32"}
    records = [
        {"type": "arithmetic", "name": f"{machine}-compute", "performance_gops": 10.0, **common},
        {
            "type": "memory",
            "name": f"{machine}-l1",
            "cache_level": "L1",
            "load_store_ratio": "2:1",
            "bandwidth_gbps": 5.0,
            **common,
        },
    ]
    return "\n".join(json.dumps(record) for record in records) + "\n"


def _application(name: str) -> str:
    return json.dumps(
        {
            "format_version": "2.0",
            "metadata": {"name": name, "date": "2026-01-01", "command": name},
            "aggregation": "global",
            "points": [{"label": "main", "runtime_s": 1.0, "arithmetic_intensity": 2.0}],
        }
    ) + "\n"


def _payload(*files: tuple[str, Any]) -> dict[str, Any]:
    return {"files": [{"path": path, "content": content} for path, content in files]}


def _state() -> SimpleNamespace:
    return SimpleNamespace(
        imported_records=[],
        imported_apps_by_id={},
        records=[],
        apps_by_id={},
        app_options=[],
        options=None,
    )


def _import(state: SimpleNamespace, payload: dict[str, Any]) -> int:
    benchmarks, apps, count = _load_payload(payload)
    _merge_imports(state, benchmarks, apps)
    return count


def test_imports_single_machine_results_without_machine_metadata() -> None:
    payload = _payload(
        ("MareNostrum_5/benchmarks.jsonl", _benchmarks("record-machine")),
        ("MareNostrum_5/applications.jsonl", _application("solver")),
        ("MareNostrum_5/notes.txt", "ignored"),
    )

    benchmarks, apps, machine_count = _load_payload(payload)

    assert machine_count == 1
    assert {record["machine"] for record in benchmarks} == {"record-machine"}
    assert {app.machine for app in apps} == {"MareNostrum_5"}


def test_imports_results_root_and_keeps_application_only_machine_identity() -> None:
    payload = _payload(
        ("carm/compute-node/benchmarks.jsonl", _benchmarks("compute-node")),
        ("carm/application-node/applications.jsonl", _application("solver")),
        ("carm/application-node/machine.json", "ignored"),
        ("carm/application-node/image.png", "ignored"),
    )
    state = _state()

    assert _import(state, payload) == 2
    assert {record["machine"] for record in state.records} == {"compute-node"}
    assert {app.machine for app in state.apps_by_id.values()} == {"application-node"}
    assert state.options["machine"] == ["compute-node"]


def test_repeated_import_deduplicates_benchmarks_and_applications() -> None:
    payload = _payload(
        ("node/benchmarks.jsonl", _benchmarks("node")),
        ("node/applications.jsonl", _application("solver")),
    )
    state = _state()

    _import(state, payload)
    first_counts = (len(state.records), len(state.apps_by_id), len(state.app_options))
    _import(state, payload)

    assert (len(state.records), len(state.apps_by_id), len(state.app_options)) == first_counts


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (_payload(("node/benchmarks.jsonl", '{"machine":\n')), "malformed JSONL"),
        (_payload(("node/benchmarks.jsonl", "[]\n")), "not an object"),
        (_payload(("node/../benchmarks.jsonl", _benchmarks("node"))), "unsafe"),
        (_payload(("node\\benchmarks.jsonl", _benchmarks("node"))), "unsafe"),
        (_payload(("node/benchmarks.jsonl", _benchmarks("node")), ("node/benchmarks.jsonl", _benchmarks("node"))), "duplicate"),
        (_payload(("node/benchmarks.jsonl", 5)), "text"),
        ({"files": []}, "no benchmark or application"),
    ],
)
def test_invalid_drop_does_not_change_state(payload: dict[str, Any], message: str) -> None:
    state = _state()
    before = (list(state.records), dict(state.apps_by_id), list(state.app_options), state.options)

    with pytest.raises(ValueError, match=message):
        _import(state, payload)

    assert (state.records, state.apps_by_id, state.app_options, state.options) == before


def test_machine_folder_wins_when_direct_and_child_results_both_exist() -> None:
    payload = _payload(
        ("root/benchmarks.jsonl", _benchmarks("root-machine")),
        ("root/child/benchmarks.jsonl", _benchmarks("child-machine")),
    )

    benchmarks, _apps, machine_count = _load_payload(payload)

    assert machine_count == 1
    assert {record["machine"] for record in benchmarks} == {"root-machine"}


def test_profile_refresh_keeps_imported_apps_with_disk_apps(tmp_path: Path) -> None:
    results = tmp_path / "results"
    machine_dir = results / "disk-machine"
    machine_dir.mkdir(parents=True)
    (machine_dir / "applications.jsonl").write_text(_application("disk-app"), encoding="utf-8")
    imported = ApplicationRecord(
        id="imported-app",
        label="imported app",
        aggregation="global",
        metadata={"name": "imported app"},
        points=[],
        machine="imported-machine",
    )
    state = SimpleNamespace(
        config=SimpleNamespace(results_dir=results),
        apps_by_id={"stale": imported},
        app_options=[],
        imported_apps_by_id={imported.id: imported},
    )

    _refresh_profile_apps(state, {"finished": True, "return_code": 0, "output_dir": str(results)})

    assert "stale" not in state.apps_by_id
    assert imported.id in state.apps_by_id
    assert any(app.label.startswith("disk-app") for app in state.apps_by_id.values())
    assert {option["value"] for option in state.app_options} == set(state.apps_by_id)
