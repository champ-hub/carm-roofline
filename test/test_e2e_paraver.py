"""End-to-end Paraver tests; Any is limited to Dash's dynamically shaped JSON protocol."""
from __future__ import annotations

import json
import math
import shutil
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from carm_roofline.gui.config import GUIConfig
from carm_roofline.gui.factory import create_app
from carm_roofline.gui.ids import ExportPanelID, ParaverID, PlotAreaID, StoreID
from carm_roofline.gui.providers import ParaverProvider
from carm_roofline.gui.data import RoofConfig, RoofStore
from carm_roofline.paraver import shim
from carm_roofline.paraver.counters import INTEL_COUNTERS
from carm_roofline.paraver.pipeline import write_counter_configs

from test.paraver_fixture import prepare_fixture

pytestmark = pytest.mark.e2e


@pytest.fixture
def paraver_fixture() -> Path:
    return prepare_fixture()


def _write_roof_records(results: Path) -> None:
    machine_dir = results / "fixture-machine"
    machine_dir.mkdir(parents=True)
    records: list[dict[str, Any]] = [
        {
            "type": "memory",
            "machine": "fixture-machine",
            "isa": "x86",
            "data_type": "f32",
            "num_threads": 1,
            "cache_level": level,
            "memory_level_name": level,
            "bandwidth_gbps": bandwidth,
            "load_store_ratio": "2:1",
            "timestamp": "2000-01-01T00:00:00",
        }
        for level, bandwidth in (("L1", 100.0), ("L2", 50.0), ("L3", 25.0), ("DRAM", 12.5))
    ]
    records.append(
        {
            "type": "arithmetic",
            "machine": "fixture-machine",
            "isa": "x86",
            "data_type": "f32",
            "num_threads": 1,
            "operation": "fma",
            "performance_gops": 100.0,
            "timestamp": "2000-01-01T00:00:00",
        }
    )
    (machine_dir / "benchmarks.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def _replay_real_paramedir_outputs(counter_dir: Path) -> Callable[[str | Path, str | Path, str], None]:
    """Render real Paramedir configs, then replay CSVs extracted from the real fixture trace."""

    def replay(trace_path: str | Path, output_dir: str | Path, time_unit: str) -> None:
        assert Path(trace_path).is_file()
        output = Path(output_dir)
        configs = write_counter_configs(output, time_unit)
        assert len(configs) == len(INTEL_COUNTERS)
        counter_csvs = tuple(counter_dir.glob("*.csv"))
        assert {path.stem for path in counter_csvs} == {
            "fp-avx2-dp",
            "fp-avx512-dp",
            "fp-scalar-dp",
            "fp-sse-dp",
            "mem-loads",
            "mem-stores",
        }
        for source in counter_csvs:
            shutil.copyfile(source, output / source.name)

    return replay


def _layout_components(node: Any) -> list[dict[str, Any]]:
    if isinstance(node, dict):
        result = [node] if "props" in node and "type" in node else []
        for value in node.values():
            result.extend(_layout_components(value))
        return result
    if isinstance(node, list):
        return [component for child in node for component in _layout_components(child)]
    return []


def _component_values(layout: dict[str, Any]) -> dict[tuple[str, str], Any]:
    values: dict[tuple[str, str], Any] = {}
    for component in _layout_components(layout):
        props = component["props"]
        component_id = props.get("id")
        if component_id is None:
            continue
        key = json.dumps(component_id, separators=(",", ":"), sort_keys=True) if isinstance(component_id, dict) else component_id
        for prop, value in props.items():
            values[(key, prop)] = value
    return values


def _dash_callback(
    client: Any,
    app: Any,
    layout: dict[str, Any],
    output_id: str,
    overrides: dict[tuple[str, str], Any],
    trigger_id: str,
) -> Any:
    values = _component_values(layout)
    values.update(overrides)
    candidates = [
        (key, metadata)
        for key, metadata in app.callback_map.items()
        if any(
            output.component_id == output_id
            for output in (metadata["output"] if isinstance(metadata["output"], list) else [metadata["output"]])
        )
        and any(item["id"] == trigger_id for item in metadata["inputs"])
    ]
    assert len(candidates) == 1, f"Expected one callback for {output_id} triggered by {trigger_id}, got {len(candidates)}"
    key, metadata = candidates[0]
    inputs = []
    for item in metadata["inputs"]:
        component_id = item["id"]
        pattern_id = component_id
        if isinstance(component_id, str) and component_id.startswith("{"):
            pattern_id = json.loads(component_id)
        if isinstance(pattern_id, dict):
            value = [
                component["props"][item["property"]]
                for component in _layout_components(layout)
                if isinstance(component["props"].get("id"), dict)
                and component["props"]["id"].get("type") == pattern_id.get("type")
            ]
        else:
            value = values.get((component_id, item["property"]))
        inputs.append({**item, "value": value})
    states = [{**item, "value": values.get((item["id"], item["property"]))} for item in metadata["state"]]
    outputs = metadata["output"]
    if isinstance(outputs, list):
        output_payload: Any = [
            {"id": output.component_id, "property": output.component_property} for output in outputs
        ]
        output_property = outputs[0].component_property
    else:
        output_payload = {"id": outputs.component_id, "property": outputs.component_property}
        output_property = outputs.component_property
    trigger_property = next(item["property"] for item in metadata["inputs"] if item["id"] == trigger_id)
    response = client.post(
        "/_dash-update-component",
        json={
            "output": key,
            "outputs": output_payload,
            "inputs": inputs,
            "state": states,
            "changedPropIds": [f"{trigger_id}.{trigger_property}"],
        },
    )
    assert response.status_code == 200, response.get_data(as_text=True)
    return response.get_json()["response"][output_id][output_property]


def test_paraver_trace_to_gui_and_all_exports(
    paraver_fixture: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exercise a complete anonymized Paraver trace through CLI, GUI routes and exports."""
    trace = tmp_path / "trace.prv"
    window = tmp_path / "window.csv"
    legend = tmp_path / "window.legend.csv"
    counter_outputs = paraver_fixture / "counters"
    shutil.copyfile(paraver_fixture / "trace.prv", trace)
    shutil.copyfile(paraver_fixture / "window.csv", window)
    shutil.copyfile(paraver_fixture / "window.legend.csv", legend)
    assert trace.is_file() and window.is_file() and legend.is_file()
    assert counter_outputs.is_dir()
    monkeypatch.setattr("carm_roofline.gui.providers.shutil.which", lambda _name: "/usr/bin/paramedir")
    monkeypatch.setattr(
        "carm_roofline.gui.providers.run_paramedir",
        _replay_real_paramedir_outputs(counter_outputs),
    )
    results = tmp_path / "benchmarks"
    _write_roof_records(results)
    apps: list[Any] = []

    def launch_gui(config: GUIConfig) -> int:
        config.results_dir = results
        assert config.paraver_trace == trace
        assert config.paraver_window_csv == window
        assert config.paraver_use_semantic_window
        apps.append(create_app(config))
        return 0

    monkeypatch.setattr("carm_roofline.gui.run_app", launch_gui)
    assert shim.main(["--color_csv", "--mask_csv", "--csv", str(window), str(trace)]) == 0
    app = apps[0]
    client = app.server.test_client()
    assert client.get("/").status_code == 200
    assert client.get("/_dash-dependencies").status_code == 200
    layout_response = client.get("/_dash-layout")
    assert layout_response.status_code == 200
    layout = layout_response.get_json()
    assert layout is not None

    provider = ParaverProvider(trace, window)
    data = provider.load()
    assert data.time_unit == "Microseconds"
    assert data.window_mode.value == "code"
    assert data.legend is not None
    assert provider.window_extent == pytest.approx((0.0, 0.3070358))
    assert len(data.trace) == 1624
    assert data.trace["time_s"].max() == pytest.approx(0.30703356)
    assert set(data.trace["legend_label"].dropna().unique()) == {
        "End",
        *(f"Region {index}" for index in range(1, 11)),
    }
    first_active = data.trace.loc[data.trace["flops"] > 0.0].iloc[0]
    assert first_active["thread_id"] == "1.4.1"
    assert first_active["time_s"] == pytest.approx(0.0)
    assert first_active["duration_s"] == pytest.approx(0.12253634)
    assert first_active["state_code"] == pytest.approx(0.0)
    assert first_active["legend_label"] == "End"
    assert first_active["flops"] == pytest.approx(329568768.0)
    assert first_active["bytes"] == pytest.approx(2803569328.0)
    assert first_active["ai"] == pytest.approx(0.11755327921036522)
    assert first_active["perf"] == pytest.approx(2689559423.7595153)
    assert first_active["load_share"] == pytest.approx(0.7360031383536381)
    expected_extent = list(provider.window_extent)

    store = _component_values(layout)[(StoreID.ROOF_STORE.value, "data")]
    assert store["roofs"] == []
    store["roofs"] = RoofStore(roof_template=RoofConfig(machine="fixture-machine")).to_dict()["roofs"]
    assert store["paraver"]["time_window"] == pytest.approx(expected_extent)
    time_window = (0.0, float(data.trace["time_s"].iloc[len(data.trace) // 2 - 1]))
    store = _dash_callback(
        client,
        app,
        layout,
        StoreID.ROOF_STORE.value,
        {(ParaverID.SLIDER_TIME_WINDOW.value, "value"): list(time_window), (StoreID.ROOF_STORE.value, "data"): store},
        ParaverID.SLIDER_TIME_WINDOW.value,
    )
    store = _dash_callback(
        client,
        app,
        layout,
        StoreID.ROOF_STORE.value,
        {
            (ExportPanelID.SLIDER_AI_THRESHOLD.value, "value"): -1.0,
            (StoreID.ROOF_STORE.value, "data"): store,
        },
        ExportPanelID.SLIDER_AI_THRESHOLD.value,
    )
    store = _dash_callback(
        client,
        app,
        layout,
        StoreID.ROOF_STORE.value,
        {
            (ExportPanelID.SLIDER_DURATION_THRESHOLD.value, "value"): -4.0,
            (StoreID.ROOF_STORE.value, "data"): store,
        },
        ExportPanelID.SLIDER_DURATION_THRESHOLD.value,
    )
    assert store["paraver"]["time_window"] == pytest.approx(time_window)
    assert store["paraver"]["ai_threshold"] == pytest.approx(0.1)
    assert store["paraver"]["duration_threshold"] == pytest.approx(1e-4)
    window_mask = (data.trace["time_s"] >= time_window[0]) & (data.trace["time_s"] <= time_window[1])
    ai_mask = data.trace["ai"] >= 0.1
    duration_mask = data.trace["duration_s"] >= 1e-4
    filtered_trace = data.trace.loc[window_mask & ai_mask & duration_mask]
    assert len(filtered_trace) == 8
    figure = _dash_callback(
        client,
        app,
        layout,
        PlotAreaID.ROOFLINE_PLOT.value,
        {(StoreID.ROOF_STORE.value, "data"): store},
        StoreID.ROOF_STORE.value,
    )
    point_series = [
        series
        for series in figure["data"]
        if series.get("mode") == "markers" and series.get("name") == "End"
    ]
    plotted_ai = [float(value) for series in point_series for value in series["x"]]
    assert len(plotted_ai) == len(filtered_trace)
    assert sorted(plotted_ai) == pytest.approx(sorted(filtered_trace["ai"].tolist()))
    assert len(plotted_ai) < len(data.trace), "filters affect the plot, not the full trace"

    colors: dict[str, list[Any]] = {}
    for color_mode in ("paraver", "age", "thread", "ldst", "isa"):
        store = _dash_callback(
            client,
            app,
            layout,
            StoreID.ROOF_STORE.value,
            {
                (ExportPanelID.RADIO_COLOR_MODE.value, "value"): color_mode,
                (StoreID.ROOF_STORE.value, "data"): store,
            },
            ExportPanelID.RADIO_COLOR_MODE.value,
        )
        figure = _dash_callback(
            client,
            app,
            layout,
            PlotAreaID.ROOFLINE_PLOT.value,
            {(StoreID.ROOF_STORE.value, "data"): store},
            StoreID.ROOF_STORE.value,
        )
        point_name = "End" if color_mode == "paraver" else data.label
        point_series = [
            series
            for series in figure["data"]
            if series.get("mode") == "markers" and series.get("name") == point_name
        ]
        assert sum(len(series.get("x", [])) for series in point_series) == len(filtered_trace)
        colors[color_mode] = [
            color
            for series in point_series
            for color in (
                series["marker"]["color"]
                if isinstance(series["marker"]["color"], list)
                else [series["marker"]["color"]]
            )
        ]
    assert set(colors["paraver"]) == set(filtered_trace["legend_color"])
    assert len(set(colors["age"])) == len(filtered_trace)
    assert colors["age"][0] != colors["age"][-1]
    assert len(set(colors["thread"])) == filtered_trace["thread_id"].nunique()
    assert len(set(colors["ldst"])) > 1
    assert filtered_trace["isa_scalar_pct"].eq(100.0).all()
    assert len(set(colors["isa"])) == 1

    exports = (
        (ExportPanelID.BTN_EXPORT_PERFORMANCE, "carm_gflops.csv"),
        (ExportPanelID.BTN_EXPORT_AI, "carm_ai.csv"),
        (ExportPanelID.BTN_EXPORT_LDST_PERCENT, "carm_ldst_percent.csv"),
        (ExportPanelID.BTN_EXPORT_ROOF_LABELS, "carm_roofs.csv"),
        (ExportPanelID.BTN_EXPORT_REGION, "carm_roofline_region.csv"),
        (ExportPanelID.BTN_EXPORT_PROXIMITY, "carm_rel_l1.csv"),
    )
    for button, filename in exports:
        status_id = f"export-status-{button.value.removeprefix('btn-export-')}"
        _dash_callback(
            client,
            app,
            layout,
            status_id,
            {(button.value, "n_clicks"): 1, (StoreID.ROOF_STORE.value, "data"): store},
            button.value,
        )
        assert (tmp_path / filename).is_file()

    expected_rows_by_file: dict[str, Counter[tuple[str, str, str, str]]] = {
        "carm_gflops.csv": Counter(),
        "carm_ai.csv": Counter(),
        "carm_ldst_percent.csv": Counter(),
    }
    for row in data.trace.itertuples(index=False):
        key = (
            str(row.thread_id),
            f"{float(row.time_s) * 1e6:.{data.precision.time}f}",
            f"{float(row.duration_s) * 1e6:.{data.precision.duration}f}",
        )
        load_share = float(row.load_share)
        load_percent = (
            max(round(load_share * 100.0, 1), 0.01)
            if math.isfinite(load_share) and row.flops != 0
            else 0.0
        )
        expected_values = (
            float(row.perf) / 1e9,
            float(row.ai),
            load_percent,
        )
        for filename, value in zip(expected_rows_by_file, expected_values):
            expected_rows_by_file[filename][(*key, f"{value:.{data.precision.value}f}")] += 1

    for filename, expected_rows in expected_rows_by_file.items():
        lines = (tmp_path / filename).read_text(encoding="utf-8").splitlines()
        metadata = lines[0].split(":")
        assert metadata[0].startswith("#") and len(metadata[0]) == 15
        assert metadata[1:5] == ["CSV", "RUNAPP", "trace.prv", "Microseconds"]
        rows = [tuple(line.split("\t")) for line in lines[1:]]
        assert len(rows) == len(data.trace), "every output must retain the full trace despite plot filters"
        assert Counter(rows) == expected_rows, f"{filename} must preserve each full-trace key/value association"

    def exported_value(filename: str) -> float:
        rows = [
            line.split("\t")
            for line in (tmp_path / filename).read_text(encoding="utf-8").splitlines()[1:]
        ]
        match = [
            row
            for row in rows
            if row[0] == str(first_active["thread_id"])
            and row[1] == f"{float(first_active['time_s']) * 1e6:.{data.precision.time}f}"
            and row[2] == f"{float(first_active['duration_s']) * 1e6:.{data.precision.duration}f}"
        ]
        assert len(match) == 1
        return float(match[0][3])

    assert exported_value("carm_roofs.csv") == 3.0
    assert exported_value("carm_roofline_region.csv") == 1.0
    performance_gflops = float(first_active["perf"]) / 1e9
    arithmetic_intensity = float(first_active["ai"])
    for level, bandwidth_gbps in (("l1", 100.0), ("l2", 50.0), ("l3", 25.0), ("dram", 12.5)):
        roof_gflops = min(bandwidth_gbps * arithmetic_intensity, 100.0)
        expected_proximity = min(performance_gflops / roof_gflops, 1.0)
        assert exported_value(f"carm_rel_{level}.csv") == pytest.approx(expected_proximity, abs=0.01)
    assert (tmp_path / "carm_roofs.legend.csv").is_file()
    assert (tmp_path / "carm_roofline_region.legend.csv").is_file()
    assert '"DRAM"' in (tmp_path / "carm_roofs.legend.csv").read_text(encoding="utf-8")
    assert '"Memory Bound"' in (tmp_path / "carm_roofline_region.legend.csv").read_text(encoding="utf-8")
    for level in ("l1", "l2", "l3", "dram"):
        assert (tmp_path / f"carm_rel_{level}.csv").is_file()
    semantic_store = _dash_callback(
        client,
        app,
        layout,
        StoreID.ROOF_STORE.value,
        {
            (ExportPanelID.SWITCH_SEMANTIC_WINDOW.value, "value"): True,
            (StoreID.ROOF_STORE.value, "data"): store,
        },
        ExportPanelID.SWITCH_SEMANTIC_WINDOW.value,
    )
    assert semantic_store["paraver"]["use_semantic_window"] is True
    default_store = _dash_callback(
        client,
        app,
        layout,
        StoreID.ROOF_STORE.value,
        {
            (ExportPanelID.SWITCH_SEMANTIC_WINDOW.value, "value"): False,
            (StoreID.ROOF_STORE.value, "data"): semantic_store,
        },
        ExportPanelID.SWITCH_SEMANTIC_WINDOW.value,
    )
    assert default_store["paraver"]["use_semantic_window"] is False
    assert capsys.readouterr().out.count("PARAVER_PROTOCOL open_csv") == 9
