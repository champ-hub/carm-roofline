"""End-to-end dashboard workflows through Dash's HTTP interface."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import dash
import pytest

from carm_roofline.gui import benchmark_runner, profile_runner
from carm_roofline.gui.callbacks.helpers import RoofControlPatch, apply_roof_controls
from carm_roofline.gui.config import GUIConfig
from carm_roofline.gui.components import build_carm_view_panel
from carm_roofline.gui.data import RoofConfig, RoofStore
from carm_roofline.gui.factory import create_app
from carm_roofline.gui.providers import BenchmarkAppsProvider
from carm_roofline.gui.ids import (
    BenchmarkPanelID,
    CarmViewPanelID,
    FolderImportID,
    PlotAreaID,
    ProfilePanelID,
    RoofCardID,
    SettingsPanelID,
    SidebarID,
    StoreID,
    WelcomePanelID,
)

pytestmark = pytest.mark.e2e


@pytest.fixture
def results_dir(tmp_path: Path) -> Path:
    """Create benchmark roofs and a profiled application using the on-disk user formats."""
    root = tmp_path / "results"
    root.mkdir()
    machine_dir = root / "fixture-host"
    machine_dir.mkdir()
    benchmarks = [
        {
            "type": "arithmetic",
            "name": "roof-fma",
            "isa": "x86",
            "machine": "fixture-host",
            "data_type": "f32",
            "num_threads": 2,
            "operation": "fma",
            "performance_gops": 100.0,
            "timestamp": "2026-01-01T00:00:00",
            "actual_frequency_hz": 3000000000,
        },
        {
            "type": "memory",
            "name": "roof-l1",
            "isa": "x86",
            "machine": "fixture-host",
            "data_type": "f32",
            "num_threads": 2,
            "load_store_ratio": "2:1",
            "cache_level": "L1",
            "bandwidth_gbps": 200.0,
            "timestamp": "2026-01-01T00:00:00",
            "actual_frequency_hz": 3000000000,
        },
        {
            "type": "memory",
            "name": "roof-dram",
            "isa": "x86",
            "machine": "fixture-host",
            "data_type": "f32",
            "num_threads": 2,
            "load_store_ratio": "2:1",
            "cache_level": "DRAM",
            "bandwidth_gbps": 20.0,
            "timestamp": "2026-01-01T00:00:00",
            "actual_frequency_hz": 3000000000,
        },
        {
            "type": "arithmetic",
            "name": "roof-fma-single-thread",
            "isa": "x86",
            "machine": "fixture-host",
            "data_type": "f32",
            "num_threads": 1,
            "operation": "fma",
            "performance_gops": 160.0,
            "timestamp": "2026-01-01T00:00:00",
            "actual_frequency_hz": 3000000000,
        },
        {
            "type": "memory",
            "name": "roof-l1-single-thread",
            "isa": "x86",
            "machine": "fixture-host",
            "data_type": "f32",
            "num_threads": 1,
            "load_store_ratio": "2:1",
            "cache_level": "L1",
            "bandwidth_gbps": 100.0,
            "timestamp": "2026-01-01T00:00:00",
            "actual_frequency_hz": 3000000000,
        },
        {
            "type": "memory",
            "name": "roof-dram-single-thread",
            "isa": "x86",
            "machine": "fixture-host",
            "data_type": "f32",
            "num_threads": 1,
            "load_store_ratio": "2:1",
            "cache_level": "DRAM",
            "bandwidth_gbps": 10.0,
            "timestamp": "2026-01-01T00:00:00",
            "actual_frequency_hz": 3000000000,
        },
    ]
    (machine_dir / "benchmarks.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in benchmarks), encoding="utf-8"
    )
    application = {
        "format_version": "2.0",
        "metadata": {"name": "solver", "date": "2026-01-02", "command": "solver --small"},
        "aggregation": "region_merged",
        "points": [
            {
                "label": "assemble",
                "total_flops": 4_000_000_000,
                "total_bytes": 1_000_000_000,
                "runtime_s": 2.0,
                "num_ranks": 1,
                "num_threads": 2,
                "num_regions": 1,
                "arithmetic_intensity": 4.0,
                "flops_per_second": 2_000_000_000,
                "bandwidth": 500_000_000,
                "optional_fractions": {"cache-residency": {"l1": 0.7, "dram": 0.3}},
            },
            {
                "label": "second-region",
                "total_flops": 2_000_000_000,
                "total_bytes": 2_000_000_000,
                "runtime_s": 1.0,
                "num_ranks": 1,
                "num_threads": 2,
                "num_regions": 1,
                "arithmetic_intensity": 1.0,
                "flops_per_second": 1_000_000_000,
                "bandwidth": 1_000_000_000,
            },
        ],
    }
    (machine_dir / "applications.jsonl").write_text(json.dumps(application) + "\n", encoding="utf-8")
    return root


def _config(results_dir: Path) -> GUIConfig:
    return GUIConfig(
        argparse.Namespace(
            verbose=0,
            results_dir=results_dir,
            gui_host="127.0.0.1",
            gui_port=8050,
            gui_debug=False,
            paraver_trace=None,
            paraver_window_csv=None,
            paraver_use_semantic_window=False,
            paraver_protocol=False,
        )
    )


def _walk_components(node: Any) -> list[dict[str, Any]]:
    """Return serialized Dash component nodes from a layout response."""
    if isinstance(node, list):
        return [component for item in node for component in _walk_components(item)]
    if not isinstance(node, dict):
        return []
    components = [node] if isinstance(node.get("props"), dict) else []
    props = node.get("props")
    if isinstance(props, dict):
        components.extend(_walk_components(props.get("children")))
    return components


def _component(layout: dict[str, Any], component_id: str | dict[str, Any]) -> dict[str, Any]:
    for node in _walk_components(layout):
        if node.get("props", {}).get("id") == component_id:
            return node["props"]
    raise AssertionError(f"component not found in rendered layout: {component_id}")


def _layout_with_roof_controls(layout: dict[str, Any], store: RoofStore) -> dict[str, Any]:
    """Render actual roof cards into a copy of the server layout for callback dispatch."""
    from plotly.utils import PlotlyJSONEncoder

    rendered_layout = copy.deepcopy(layout)
    controls = json.loads(json.dumps(build_carm_view_panel(store), cls=PlotlyJSONEncoder))
    _component(rendered_layout, str(SidebarID.SIDEBAR_CONTENT))["children"] = controls
    return rendered_layout


def _output_parts(output: Any) -> list[tuple[str, str]]:
    outputs = output if isinstance(output, list) else [output]
    return [(str(item.component_id), str(item.component_property)) for item in outputs]


def _callback_key(
    app: dash.Dash, component_id: str, property_name: str, input_id: str | None = None
) -> str:
    for key, metadata in app.callback_map.items():
        if (component_id, property_name) not in _output_parts(metadata["output"]):
            continue
        if input_id is None or any(str(item["id"]) == input_id for item in metadata["inputs"]):
            return key
    raise AssertionError(f"callback output not registered: {component_id}.{property_name}")


def _layout_value(layout: dict[str, Any], callback_id: str, property_name: str) -> Any:
    """Find a component property's current value in the server-rendered layout."""
    if callback_id.startswith("{"):
        pattern = json.loads(callback_id)
        try:
            return _component(layout, pattern).get(property_name)
        except AssertionError:
            return None
    try:
        return _component(layout, callback_id).get(property_name)
    except AssertionError:
        return None


def _dispatch(
    app: dash.Dash,
    client: Any,
    layout: dict[str, Any],
    component_id: str,
    property_name: str,
    *,
    overrides: dict[str, Any] | None = None,
    changed_id: str,
    input_id: str | None = None,
    expected_status: int = 200,
) -> dict[str, Any]:
    """Dispatch one real Dash callback request through Flask's HTTP test client."""
    key = _callback_key(app, component_id, property_name, input_id)
    metadata = app.callback_map[key]
    values = overrides or {}

    def value_for(item_id: str, property_name: str) -> Any:
        if item_id in values:
            return values[item_id]
        if item_id.startswith("{"):
            serialized_id = json.loads(item_id)
            for override_id, value in values.items():
                if override_id.startswith("{") and json.loads(override_id) == serialized_id:
                    return value
        return _layout_value(layout, item_id, property_name)

    def make_items(specifications: list[dict[str, Any]]) -> list[Any]:
        items: list[Any] = []
        for specification in specifications:
            item_id = str(specification["id"])
            prop = str(specification["property"])
            if item_id.startswith("{"):
                pattern_id = json.loads(item_id)
                if pattern_id.get("roof_id") == ["MATCH"]:
                    changed_pattern = json.loads(changed_id.rsplit(".", 1)[0])
                    items.append({
                        "id": changed_pattern,
                        "property": prop,
                        "value": value_for(
                            json.dumps(changed_pattern, separators=(",", ":"), sort_keys=True), prop
                        ),
                    })
                else:
                    matches = []
                    for component in _walk_components(layout):
                        concrete = component.get("props", {}).get("id")
                        if not isinstance(concrete, dict) or any(
                            key not in concrete
                            or (
                                isinstance(pattern_id[key], list)
                                and pattern_id[key] not in (["ALL"], ["MATCH"])
                                and concrete.get(key) not in pattern_id[key]
                            )
                            or (not isinstance(pattern_id[key], list) and pattern_id[key] != concrete.get(key))
                            for key in pattern_id
                        ):
                            continue
                        concrete_id = json.dumps(concrete, separators=(",", ":"), sort_keys=True)
                        matches.append({"id": concrete, "property": prop, "value": value_for(concrete_id, prop)})
                    matches.reverse()
                    items.append(matches)
                continue
            items.append({"id": item_id, "property": prop, "value": value_for(item_id, prop)})
        return items

    output_parts = _output_parts(metadata["output"])
    outputs: Any = [
        {"id": item_id, "property": item_property} for item_id, item_property in output_parts
    ]
    if len(outputs) == 1:
        outputs = outputs[0]
    response = client.post(
        "/_dash-update-component",
        json={
            "output": key,
            "outputs": outputs,
            "inputs": make_items(metadata["inputs"]),
            "state": make_items(metadata["state"]),
            "changedPropIds": [changed_id],
        },
    )
    assert response.status_code == expected_status, response.get_data(as_text=True)
    if expected_status == 204:
        return {}
    return response.get_json()["response"]


def _figure_from_response(response: dict[str, Any]) -> dict[str, Any]:
    return response[str(PlotAreaID.ROOFLINE_PLOT)]["figure"]


def _marker_traces(figure: dict[str, Any]) -> list[dict[str, Any]]:
    return [trace for trace in figure["data"] if "markers" in trace.get("mode", "")]


def test_welcome_panel_dismissal_and_config_persistence(
    results_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings_path = tmp_path / "gui-settings.json"
    monkeypatch.setattr("carm_roofline.gui.factory.gui_settings_path", lambda: settings_path)
    monkeypatch.setattr("carm_roofline.gui.callbacks.settings.gui_settings_path", lambda: settings_path)
    monkeypatch.setattr("carm_roofline.gui.callbacks.welcome.gui_settings_path", lambda: settings_path)

    app = create_app(_config(results_dir))
    client = app.server.test_client()
    layout = client.get("/_dash-layout").get_json()
    assert _component(layout, str(StoreID.ROOF_STORE))["data"]["roofs"] == []
    welcome = _component(layout, str(WelcomePanelID.CONTAINER))
    assert welcome["style"] == {"display": "block"}

    dismissed = _dispatch(
        app,
        client,
        layout,
        str(WelcomePanelID.CONTAINER),
        "style",
        overrides={str(WelcomePanelID.GOT_IT): 1},
        changed_id=f"{WelcomePanelID.GOT_IT}.n_clicks",
    )
    assert dismissed[str(WelcomePanelID.CONTAINER)]["style"] == {"display": "none"}
    assert not settings_path.exists()

    app = create_app(_config(results_dir))
    client = app.server.test_client()
    layout = client.get("/_dash-layout").get_json()
    stale_roof_store = _component(layout, str(StoreID.ROOF_STORE))["data"]
    assert _component(layout, str(WelcomePanelID.CONTAINER))["style"] == {"display": "block"}
    hidden = _dispatch(
        app,
        client,
        layout,
        str(WelcomePanelID.CONTAINER),
        "style",
        overrides={str(WelcomePanelID.DO_NOT_SHOW_AGAIN): ["hide"]},
        changed_id=f"{WelcomePanelID.DO_NOT_SHOW_AGAIN}.value",
    )
    assert hidden[str(WelcomePanelID.CONTAINER)]["style"] == {"display": "none"}
    assert json.loads(settings_path.read_text(encoding="utf-8"))["show_welcome_panel"] is False

    _dispatch(
        app,
        client,
        layout,
        str(StoreID.ROOF_STORE),
        "data",
        overrides={
            str(SettingsPanelID.SWITCH_NORMALIZE): True,
            str(StoreID.ROOF_STORE): stale_roof_store,
        },
        changed_id=f"{SettingsPanelID.SWITCH_NORMALIZE}.value",
        input_id=str(SettingsPanelID.SWITCH_NORMALIZE),
    )
    saved = json.loads(settings_path.read_text(encoding="utf-8"))
    assert saved["show_welcome_panel"] is False
    assert saved["normalize_by_threads"] is True

    restarted = create_app(_config(results_dir))
    restarted_layout = restarted.server.test_client().get("/_dash-layout").get_json()
    assert _component(restarted_layout, str(WelcomePanelID.CONTAINER))["style"] == {"display": "none"}


def test_carm_results_to_roofline_and_settings_via_dash_http(
    results_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Load real JSONL results, select a profiled point, change settings, and add a comparison roof."""
    settings_path = tmp_path / "gui-settings.json"
    monkeypatch.setattr("carm_roofline.gui.factory.gui_settings_path", lambda: settings_path)
    monkeypatch.setattr("carm_roofline.gui.callbacks.settings.gui_settings_path", lambda: settings_path)
    app = create_app(_config(results_dir))
    client = app.server.test_client()

    assert client.get("/").status_code == 200
    layout_response = client.get("/_dash-layout")
    dependencies_response = client.get("/_dash-dependencies")
    assert layout_response.status_code == dependencies_response.status_code == 200
    layout = layout_response.get_json()
    store = _component(layout, str(StoreID.ROOF_STORE))["data"]
    assert not store["roofs"]
    app_id = next(iter(BenchmarkAppsProvider(results_dir).load()))
    store["roofs"] = RoofStore(roof_template=RoofConfig(machine="fixture-host", app_ids=[app_id])).to_dict()["roofs"]
    roof_id = store["roofs"][0]["id"]
    base_layout = layout
    layout = _layout_with_roof_controls(base_layout, RoofStore.from_dict(store))
    roof_input_overrides = {
        json.dumps(
            {"roof_id": roof_id, "type": RoofCardID.DROPDOWN_APPS}, sort_keys=True, separators=(",", ":")
        ): [app_id]
    }
    # Dispatch the actual figure callback with the application selected in its roof card.
    plot_response = _dispatch(
        app,
        client,
        layout,
        str(PlotAreaID.ROOFLINE_PLOT),
        "figure",
        overrides=roof_input_overrides,
        changed_id=f"{StoreID.ROOF_STORE}.data",
    )
    figure = _figure_from_response(plot_response)
    points = _marker_traces(figure)
    assert any("solver" in trace.get("name", "") for trace in points), figure["data"]
    app_trace = next(trace for trace in points if "solver" in trace.get("name", ""))
    assert app_trace["x"] == [4.0, 1.0]
    assert app_trace["y"] == [2.0, 1.0]
    assert any(trace.get("mode") == "lines" for trace in figure["data"])

    # Click the actual rendered profile point, then clear it through the background-click control.
    click_payload = {"points": [{"customdata": app_trace["customdata"][0]}]}
    selection_response = _dispatch(
        app,
        client,
        layout,
        str(StoreID.SELECTED_POINT),
        "data",
        overrides={
            str(PlotAreaID.ROOFLINE_PLOT): click_payload,
            str(PlotAreaID.BG_CLICK_CLEAR): "0",
            str(StoreID.SELECTED_POINT): None,
        },
        changed_id=f"{PlotAreaID.ROOFLINE_PLOT}.clickData",
    )
    selected = selection_response[str(StoreID.SELECTED_POINT)]["data"]
    assert selected["app_id"] == app_id
    assert selected["point_index"] == 0
    assert selected["roof_id"] == roof_id
    assert selected["fractions"] == {"l1": 0.7, "dram": 0.3}
    selected_plot = _dispatch(
        app,
        client,
        layout,
        str(PlotAreaID.ROOFLINE_PLOT),
        "figure",
        overrides={
            **roof_input_overrides,
            str(StoreID.SELECTED_POINT): selected,
        },
        changed_id=f"{StoreID.SELECTED_POINT}.data",
    )
    selected_figure = _figure_from_response(selected_plot)
    assert selected_figure["layout"]["meta"]["carm_selected_point"] == {
        "roof_id": roof_id,
        "app_id": app_id,
        "point_index": 0,
    }
    selected_guides = [
        trace
        for trace in selected_figure["data"]
        if isinstance(trace.get("meta"), dict)
        and trace["meta"].get("carm_role") == "point-guide"
        and trace["meta"].get("guide") == "selected"
    ]
    selected_line = next(trace for trace in selected_guides if trace["meta"]["kind"] == "line")
    selected_markers = next(trace for trace in selected_guides if trace["meta"]["kind"] == "intersections")
    assert selected_line["x"] == [4.0, 4.0]
    assert selected_line["y"][0] == pytest.approx(10 ** selected_figure["layout"]["yaxis"]["range"][0])
    assert selected_markers["mode"] == "markers"
    assert all("x @" in text for text in selected_markers["hovertext"])
    assert selected_markers["hovertemplate"] == "%{hovertext}<extra></extra>"
    selected_lines = [
        trace
        for trace in selected_figure["data"]
        if trace.get("mode") == "lines" and trace.get("legendgroup") == roof_id
    ]
    line_opacities = [trace["opacity"] for trace in selected_lines if "opacity" in trace]
    line_widths = [trace["line"]["width"] for trace in selected_lines]
    assert any(opacity == pytest.approx(0.79) for opacity in line_opacities)
    assert any(opacity == pytest.approx(0.51) for opacity in line_opacities)
    assert any(width == pytest.approx(5.0625) for width in line_widths)
    assert any(width == pytest.approx(2.8125) for width in line_widths)

    # A point without residency remains selected without changing roof emphasis.
    click_payload = {"points": [{"customdata": app_trace["customdata"][1]}]}
    no_residency_response = _dispatch(
        app,
        client,
        layout,
        str(StoreID.SELECTED_POINT),
        "data",
        overrides={
            str(PlotAreaID.ROOFLINE_PLOT): click_payload,
            str(PlotAreaID.BG_CLICK_CLEAR): "0",
            str(StoreID.SELECTED_POINT): selected,
        },
        changed_id=f"{PlotAreaID.ROOFLINE_PLOT}.clickData",
    )
    no_residency = no_residency_response[str(StoreID.SELECTED_POINT)]["data"]
    assert no_residency == {"roof_id": roof_id, "fractions": {}, "app_id": app_id, "point_index": 1}
    no_residency_figure = _figure_from_response(
        _dispatch(
            app,
            client,
            layout,
            str(PlotAreaID.ROOFLINE_PLOT),
            "figure",
            overrides={**roof_input_overrides, str(StoreID.SELECTED_POINT): no_residency},
            changed_id=f"{StoreID.SELECTED_POINT}.data",
        )
    )
    assert no_residency_figure["layout"]["meta"]["carm_selected_point"] == {
        "roof_id": roof_id,
        "app_id": app_id,
        "point_index": 1,
    }
    no_residency_lines = [
        trace
        for trace in no_residency_figure["data"]
        if trace.get("mode") == "lines" and trace.get("legendgroup") == roof_id
    ]
    assert all("opacity" not in trace for trace in no_residency_lines)

    clear_response = _dispatch(
        app,
        client,
        layout,
        str(StoreID.SELECTED_POINT),
        "data",
        overrides={
            str(PlotAreaID.ROOFLINE_PLOT): click_payload,
            str(PlotAreaID.BG_CLICK_CLEAR): "1",
            str(StoreID.SELECTED_POINT): no_residency,
        },
        changed_id=f"{PlotAreaID.BG_CLICK_CLEAR}.value",
    )
    assert clear_response[str(StoreID.SELECTED_POINT)]["data"] is None
    cleared_plot = _dispatch(
        app,
        client,
        layout,
        str(PlotAreaID.ROOFLINE_PLOT),
        "figure",
        overrides={
            **roof_input_overrides,
            str(StoreID.SELECTED_POINT): None,
        },
        changed_id=f"{StoreID.SELECTED_POINT}.data",
    )
    assert _figure_from_response(cleared_plot)["layout"]["meta"]["carm_selected_point"] is None
    assert _figure_from_response(cleared_plot) == figure

    # Normalize by the two-thread roof; the application point moves from 2 to 1 GFLOP/s.
    settings_response = _dispatch(
        app,
        client,
        layout,
        str(StoreID.ROOF_STORE),
        "data",
        overrides={
            str(SettingsPanelID.SWITCH_NORMALIZE): True,
            str(StoreID.ROOF_STORE): store,
        },
        changed_id=f"{SettingsPanelID.SWITCH_NORMALIZE}.value",
        input_id=str(SettingsPanelID.SWITCH_NORMALIZE),
    )
    normalized_store = settings_response[str(StoreID.ROOF_STORE)]["data"]
    normalized_plot = _dispatch(
        app,
        client,
        layout,
        str(PlotAreaID.ROOFLINE_PLOT),
        "figure",
        overrides={
            str(StoreID.ROOF_STORE): normalized_store,
            json.dumps(
                {"roof_id": roof_id, "type": RoofCardID.DROPDOWN_APPS}, sort_keys=True, separators=(",", ":")
            ): [app_id],
        },
        changed_id=f"{StoreID.ROOF_STORE}.data",
    )
    normalized_point = next(
        trace
        for trace in _marker_traces(_figure_from_response(normalized_plot))
        if "solver" in trace.get("name", "")
    )
    assert normalized_point["y"] == [1.0, 0.5]
    assert json.loads(settings_path.read_text(encoding="utf-8"))["normalize_by_threads"] is True
    assert any(trace.get("fill") == "toself" for trace in _figure_from_response(normalized_plot)["data"])

    # Appearance settings persist and alter the Plotly artifact seen by the dashboard.
    appearance_response = _dispatch(
        app,
        client,
        layout,
        str(StoreID.ROOF_STORE),
        "data",
        overrides={
            str(SettingsPanelID.DROPDOWN_ROOF_FILL_MODE): "none",
            str(StoreID.ROOF_STORE): normalized_store,
        },
        changed_id=f"{SettingsPanelID.DROPDOWN_ROOF_FILL_MODE}.value",
        input_id=str(SettingsPanelID.DROPDOWN_ROOF_FILL_MODE),
    )
    appearance_store = appearance_response[str(StoreID.ROOF_STORE)]["data"]
    appearance_plot = _dispatch(
        app,
        client,
        layout,
        str(PlotAreaID.ROOFLINE_PLOT),
        "figure",
        overrides={
            str(StoreID.ROOF_STORE): appearance_store,
            json.dumps(
                {"roof_id": roof_id, "type": RoofCardID.DROPDOWN_APPS}, sort_keys=True, separators=(",", ":")
            ): [app_id],
        },
        changed_id=f"{StoreID.ROOF_STORE}.data",
    )
    assert all(trace.get("fill") != "toself" for trace in _figure_from_response(appearance_plot)["data"])
    assert json.loads(settings_path.read_text(encoding="utf-8"))["roof_fill_mode"] == "none"

    # Adding a roof card and configuring a second thread count creates a visible comparison roof.
    add_roof_response = _dispatch(
        app,
        client,
        layout,
        str(StoreID.ROOF_STORE),
        "data",
        overrides={str(StoreID.ROOF_STORE): appearance_store, "btn-add-roof": 1},
        changed_id="btn-add-roof.n_clicks",
        input_id="btn-add-roof",
    )
    added_store = add_roof_response[str(StoreID.ROOF_STORE)]["data"]
    assert len(added_store["roofs"]) == 2
    comparison_layout = _layout_with_roof_controls(base_layout, RoofStore.from_dict(added_store))
    comparison_overrides: dict[str, Any] = {str(StoreID.ROOF_STORE): added_store}
    for roof, threads, app_ids in zip(added_store["roofs"], ("2", "1"), ([app_id], [])):
        for control_type, value in (
            (RoofCardID.DROPDOWN_MACHINE, "fixture-host"),
            (RoofCardID.DROPDOWN_ISA, "x86"),
            (RoofCardID.DROPDOWN_THREADS, threads),
            (RoofCardID.DROPDOWN_COMPUTE, ["fma"]),
            (RoofCardID.DROPDOWN_DATA_TYPE, "f32"),
            (RoofCardID.DROPDOWN_LS_RATIO, "2:1"),
            (RoofCardID.DROPDOWN_FREQUENCY, "3000000000"),
            (RoofCardID.DROPDOWN_APPS, app_ids),
        ):
            comparison_overrides[
                json.dumps(
                    {"roof_id": roof["id"], "type": control_type}, separators=(",", ":"), sort_keys=True
                )
            ] = value
    comparison_plot = _dispatch(
        app,
        client,
        comparison_layout,
        str(PlotAreaID.ROOFLINE_PLOT),
        "figure",
        overrides=comparison_overrides,
        changed_id=f"{StoreID.ROOF_STORE}.data",
    )
    comparison_figure = _figure_from_response(comparison_plot)
    app_points = [trace for trace in _marker_traces(comparison_figure) if "solver" in trace.get("name", "")]
    assert len(app_points) == 1
    compute_ceilings = [
        trace["y"]
        for trace in comparison_figure["data"]
        if trace.get("mode") == "lines"
        and len(trace.get("y", [])) == 2
        and trace["y"][0] == trace["y"][1]
    ]
    assert [50.0, 50.0] in compute_ceilings
    assert [160.0, 160.0] in compute_ceilings


def test_benchmark_action_tracks_real_job_and_refreshes_results(
    results_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A dashboard launch emits live progress, rejects a concurrent run, and refreshes roofs on success."""
    output_dir = tmp_path / "benchmark-output"
    machine_output_dir = output_dir / "fixture-host"
    machine_output_dir.mkdir(parents=True)
    (machine_output_dir / "benchmarks.jsonl").write_text(
        json.dumps(
            {
                "type": "arithmetic",
                "name": "new-roof-fma",
                "isa": "x86",
                "machine": "fixture-host",
                "data_type": "f32",
                "num_threads": 1,
                "operation": "fma",
                "performance_gops": 250.0,
                "timestamp": "2026-01-03T00:00:00",
                "actual_frequency_hz": 3000000000,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    settings_path = tmp_path / "gui-settings.json"
    monkeypatch.setattr("carm_roofline.gui.factory.gui_settings_path", lambda: settings_path)
    app = create_app(_config(results_dir))
    client = app.server.test_client()
    layout = client.get("/_dash-layout").get_json()

    release_file = tmp_path / "release-worker"
    worker_script = "\n".join(
        (
            "import json, pathlib, sys, time",
            "print('CARM_PROGRESS ' + json.dumps({'event': 'stage', 'stage': 'running'}), file=sys.stderr, flush=True)",
            "print('CARM_PROGRESS ' + json.dumps({'event': 'measurement', 'completed': 1, 'total': 1}), file=sys.stderr, flush=True)",
            "print('worker output', flush=True)",
            f"while not pathlib.Path({json.dumps(str(release_file))}).exists(): time.sleep(0.01)",
        )
    )
    real_popen = subprocess.Popen
    launches: list[bool] = []

    def launch_fake_external(command: Any, stdout: Any = None, stderr: Any = None) -> subprocess.Popen[bytes]:
        del command
        launches.append(True)
        return real_popen([sys.executable, "-c", worker_script], stdout=stdout, stderr=stderr)

    monkeypatch.setitem(benchmark_runner._job_state, "current", None)
    monkeypatch.setattr(benchmark_runner.subprocess, "Popen", launch_fake_external)

    store = _component(layout, str(StoreID.ROOF_STORE))["data"]
    store["roofs"] = RoofStore(roof_template=RoofConfig(machine="fixture-host")).to_dict()["roofs"]
    baseline = _figure_from_response(
        _dispatch(
            app,
            client,
            layout,
            str(PlotAreaID.ROOFLINE_PLOT),
            "figure",
            changed_id=f"{StoreID.ROOF_STORE}.data",
        )
    )

    start_response = _dispatch(
        app,
        client,
        layout,
        str(BenchmarkPanelID.JOB),
        "data",
        overrides={
            "benchmark-output-dir": str(output_dir),
            "benchmark-start": 1,
        },
        changed_id=f"{BenchmarkPanelID.START}.n_clicks",
        input_id=str(BenchmarkPanelID.START),
    )
    job = start_response[str(BenchmarkPanelID.JOB)]["data"]
    assert job["finished"] is False
    assert start_response[str(BenchmarkPanelID.START)]["disabled"] is True
    assert start_response[str(BenchmarkPanelID.INTERVAL)]["disabled"] is False

    try:
        concurrent_response = _dispatch(
            app,
            client,
            layout,
            str(BenchmarkPanelID.JOB),
            "data",
            overrides={
                "benchmark-output-dir": str(output_dir),
                "benchmark-start": 2,
                str(BenchmarkPanelID.JOB): job,
            },
            changed_id=f"{BenchmarkPanelID.START}.n_clicks",
            input_id=str(BenchmarkPanelID.START),
        )
        concurrent_job = concurrent_response[str(BenchmarkPanelID.JOB)]["data"]
        assert concurrent_job["finished"] is False
        assert concurrent_job["output_dir"] == job["output_dir"]
        assert concurrent_response[str(BenchmarkPanelID.START)]["disabled"] is True
        assert len(launches) == 1

        release_file.touch()
        finished = False
        for tick in range(100):
            progress_response = _dispatch(
                app,
                client,
                layout,
                str(BenchmarkPanelID.JOB),
                "data",
                overrides={
                    str(BenchmarkPanelID.JOB): job,
                    str(BenchmarkPanelID.INTERVAL): tick + 1,
                },
                changed_id=f"{BenchmarkPanelID.INTERVAL}.n_intervals",
                input_id=str(BenchmarkPanelID.INTERVAL),
            )
            if str(BenchmarkPanelID.JOB) in progress_response:
                job = progress_response[str(BenchmarkPanelID.JOB)]["data"]
            if job["finished"]:
                finished = True
                break
            time.sleep(0.02)

        assert finished
        assert job["return_code"] == 0
        assert job["completed"] == 1
        assert "worker output" in job["log"]
        assert job["stage"] == "running"
        assert job["output_dir"] == str(output_dir)
        assert progress_response[str(BenchmarkPanelID.PROGRESS)]["value"] == "1"

        refreshed_figure = _figure_from_response(
            _dispatch(
                app,
                client,
                layout,
                str(PlotAreaID.ROOFLINE_PLOT),
                "figure",
                overrides={
                    str(StoreID.ROOF_STORE): _component(layout, str(StoreID.ROOF_STORE))["data"],
                    str(BenchmarkPanelID.JOB): job,
                },
                changed_id=f"{BenchmarkPanelID.JOB}.data",
            )
        )
        assert refreshed_figure["layout"]["yaxis"]["range"][1] > baseline["layout"]["yaxis"]["range"][1]
    finally:
        release_file.touch()
        for _ in range(100):
            active_job = benchmark_runner.get_current_job()
            if active_job is None or active_job.finished:
                break
            time.sleep(0.01)
        with benchmark_runner._job_lock:
            benchmark_runner._job_state["current"] = None


def test_profile_completion_refreshes_application_selector_before_sidebar_render(
    results_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings_path = tmp_path / "gui-settings.json"
    monkeypatch.setattr("carm_roofline.gui.factory.gui_settings_path", lambda: settings_path)
    app = create_app(_config(results_dir))
    client = app.server.test_client()
    layout = client.get("/_dash-layout").get_json()

    record = {
        "format_version": "3.0",
        "aggregation": "global",
        "metadata": {"name": "fresh profile", "date": "2026-01-03", "command": "./profiled"},
        "points": [{"label": "main", "arithmetic_intensity": 2.0}],
    }
    applications_file = results_dir / "fixture-host" / "applications.jsonl"
    worker_script = (
        "import pathlib; "
        f"pathlib.Path({json.dumps(str(applications_file))}).open('a', encoding='utf-8').write("
        f"{json.dumps(json.dumps(record) + chr(10))})"
    )
    real_popen = subprocess.Popen

    def launch_profile_worker(command: Any, stdout: Any = None, stderr: Any = None) -> subprocess.Popen[bytes]:
        del command
        return real_popen([sys.executable, "-c", worker_script], stdout=stdout, stderr=stderr)

    with profile_runner._job_lock:
        profile_runner._job_state["current"] = None
    monkeypatch.setattr(profile_runner.subprocess, "Popen", launch_profile_worker)

    try:
        response = _dispatch(
            app,
            client,
            layout,
            str(ProfilePanelID.JOB),
            "data",
            overrides={
                str(ProfilePanelID.START): 1,
                str(ProfilePanelID.COMMAND): "/usr/bin/true",
                str(ProfilePanelID.OUTPUT_DIR): "",
            },
            changed_id=f"{ProfilePanelID.START}.n_clicks",
            input_id=str(ProfilePanelID.START),
        )
        job = response[str(ProfilePanelID.JOB)]["data"]
        for tick in range(100):
            if job["finished"]:
                break
            progress_response = _dispatch(
                app,
                client,
                layout,
                str(ProfilePanelID.JOB),
                "data",
                overrides={
                    str(ProfilePanelID.JOB): job,
                    str(ProfilePanelID.INTERVAL): tick + 1,
                },
                changed_id=f"{ProfilePanelID.INTERVAL}.n_intervals",
                input_id=str(ProfilePanelID.INTERVAL),
            )
            if str(ProfilePanelID.JOB) in progress_response:
                job = progress_response[str(ProfilePanelID.JOB)]["data"]
            time.sleep(0.02)

        assert job["finished"] is True
        assert job["return_code"] == 0

        sidebar = _dispatch(
            app,
            client,
            layout,
            str(SidebarID.SIDEBAR_CONTENT),
            "children",
            overrides={
                str(ProfilePanelID.JOB): job,
                str(StoreID.ROOF_STORE): RoofStore(roof_template=RoofConfig(machine="fixture-host")).to_dict(),
            },
            changed_id=f"{ProfilePanelID.JOB}.data",
        )[str(SidebarID.SIDEBAR_CONTENT)]["children"]
        app_options = [
            option
            for component in _walk_components(sidebar)
            if isinstance((component_id := component.get("props", {}).get("id")), dict)
            and component_id.get("type") == RoofCardID.DROPDOWN_APPS
            for option in component.get("props", {}).get("options", [])
        ]
        assert any(option["label"] == "fresh profile — 2026-01-03 (global)" for option in app_options)
    finally:
        with profile_runner._job_lock:
            profile_runner._job_state["current"] = None


def test_apply_roof_controls_preserves_absent_values_and_applies_explicit_clears() -> None:
    store = RoofStore()
    store.roofs = [
        RoofConfig(roof_id="roof-a", machine="machine-a", isa="x86", compute_insts=["fma"], app_ids=["app-a"]),
        RoofConfig(roof_id="roof-b", machine="machine-b", isa="arm", compute_insts=["add"], app_ids=["app-b"]),
    ]
    patches: dict[str, RoofControlPatch] = {
        "roof-b": {"machine": None, "compute_insts": [], "app_ids": None},
        "roof-a": {"compute_insts": None},
    }

    apply_roof_controls(store, patches)

    assert store.roofs[0].machine == "machine-a"
    assert store.roofs[0].isa == "x86"
    assert store.roofs[0].compute_insts == ["fma"]
    assert store.roofs[0].app_ids == ["app-a"]
    assert store.roofs[1].machine is None
    assert store.roofs[1].isa == "arm"
    assert store.roofs[1].compute_insts == []
    assert store.roofs[1].app_ids == []


def test_remove_roof_targets_clicked_card(results_dir: Path) -> None:
    app = create_app(_config(results_dir))
    client = app.server.test_client()
    layout = client.get("/_dash-layout").get_json()
    store = RoofStore()
    store.roofs = [
        RoofConfig(
            roof_id="roof-a", machine="fixture-host", num_threads=2, collapsed=True, advanced_collapsed=False
        ),
        RoofConfig(roof_id="roof-b", machine="fixture-host", num_threads=1),
        RoofConfig(roof_id="roof-c", machine="fixture-host", num_threads=2),
    ]
    layout = _layout_with_roof_controls(layout, store)
    overrides: dict[str, Any] = {str(StoreID.ROOF_STORE): store.to_dict()}
    for roof_id, compute_insts in (("roof-a", ["add"]), ("roof-b", ["mul"]), ("roof-c", ["fma"])):
        control_id = json.dumps(
            {"type": RoofCardID.DROPDOWN_COMPUTE, "roof_id": roof_id}, separators=(",", ":"), sort_keys=True
        )
        overrides[control_id] = compute_insts
    remove_pattern_id = json.dumps(
        {"type": RoofCardID.BTN_REMOVE_ROOF, "roof_id": ["MATCH"]}, separators=(",", ":"), sort_keys=True
    )
    changed_id = json.dumps(
        {"type": RoofCardID.BTN_REMOVE_ROOF, "roof_id": "roof-b"}, separators=(",", ":"), sort_keys=True
    )
    overrides[changed_id] = 1
    response = _dispatch(
        app, client, layout, str(StoreID.ROOF_STORE), "data",
        overrides=overrides,
        changed_id=f"{changed_id}.n_clicks", input_id=remove_pattern_id,
    )
    remaining = response[str(StoreID.ROOF_STORE)]["data"]["roofs"]
    assert [roof["id"] for roof in remaining] == ["roof-a", "roof-c"]
    assert remaining[0]["compute_insts"] == ["add"]
    assert remaining[1]["compute_insts"] == ["fma"]
    assert remaining[0]["collapsed"] is True
    assert remaining[0]["advanced_collapsed"] is False
    no_click = _dispatch(
        app,
        client,
        layout,
        str(StoreID.ROOF_STORE),
        "data",
        overrides={str(StoreID.ROOF_STORE): response[str(StoreID.ROOF_STORE)]["data"]},
        changed_id=json.dumps(
            {"type": RoofCardID.BTN_REMOVE_ROOF, "roof_id": "roof-c"}, separators=(",", ":"), sort_keys=True
        ) + ".n_clicks",
        input_id=remove_pattern_id,
        expected_status=204,
    )
    assert no_click == {}
    solo_store = RoofStore.from_dict(response[str(StoreID.ROOF_STORE)]["data"])
    solo_store.roofs = [solo_store.roofs[0]]
    solo_id = json.dumps(
        {"type": RoofCardID.BTN_REMOVE_ROOF, "roof_id": "roof-a"}, separators=(",", ":"), sort_keys=True
    )
    final = _dispatch(
        app, client, layout, str(StoreID.ROOF_STORE), "data",
        overrides={str(StoreID.ROOF_STORE): solo_store.to_dict(), solo_id: 1},
        changed_id=f"{solo_id}.n_clicks", input_id=remove_pattern_id,
    )
    assert final[str(StoreID.ROOF_STORE)]["data"]["roofs"] == []


def test_folder_import_refreshes_machine_app_dropdowns_and_plot(results_dir: Path, tmp_path: Path) -> None:
    app = create_app(_config(results_dir))
    client = app.server.test_client()
    layout = client.get("/_dash-layout").get_json()

    def benchmark_files(machine: str) -> str:
        shared = {"machine": machine, "isa": "x86", "num_threads": 2, "data_type": "f32"}
        records = (
            {
                "type": "arithmetic",
                "name": f"{machine}-compute",
                "operation": "fma",
                "performance_gops": 100.0,
                **shared,
            },
            {
                "type": "memory",
                "name": f"{machine}-l1",
                "cache_level": "L1",
                "load_store_ratio": "2:1",
                "bandwidth_gbps": 50.0,
                **shared,
            },
            {
                "type": "memory",
                "name": f"{machine}-dram",
                "cache_level": "DRAM",
                "load_store_ratio": "2:1",
                "bandwidth_gbps": 10.0,
                **shared,
            },
        )
        return "\n".join(json.dumps(record) for record in records) + "\n"

    def application(name: str) -> str:
        return json.dumps(
            {
                "format_version": "2.0",
                "metadata": {"name": name, "date": "2026-10-29", "command": name},
                "aggregation": "global",
                "points": [
                    {
                        "label": "main",
                        "runtime_s": 1.0,
                        "arithmetic_intensity": 3.0,
                        "flops_per_second": 2.0,
                        "bandwidth": 1.0,
                    }
                ],
            }
        ) + "\n"

    payload = {
        "files": [
            {"path": "import-root/node-a/benchmarks.jsonl", "content": benchmark_files("node-a")},
            {"path": "import-root/node-a/applications.jsonl", "content": application("app-a")},
            {"path": "import-root/node-b/benchmarks.jsonl", "content": benchmark_files("node-b")},
            {"path": "import-root/node-b/applications.jsonl", "content": application("app-b")},
        ]
    }
    imported = _dispatch(
        app,
        client,
        layout,
        str(FolderImportID.STATUS),
        "children",
        overrides={str(StoreID.FOLDER_IMPORT_PAYLOAD): payload},
        changed_id=f"{StoreID.FOLDER_IMPORT_PAYLOAD}.data",
    )
    assert "Imported 2 machine folder(s)" in imported[str(FolderImportID.STATUS)]["children"]
    completion = imported[str(StoreID.FOLDER_IMPORT_COMPLETE)]["data"]

    app_ids: dict[str, str] = {}
    for machine, other in (("node-a", "node-b"), ("node-b", "node-a")):
        roof_store = RoofStore(roof_template=RoofConfig(machine=machine))
        sidebar = _dispatch(
            app,
            client,
            layout,
            str(SidebarID.SIDEBAR_CONTENT),
            "children",
            overrides={
                str(StoreID.ROOF_STORE): roof_store.to_dict(),
                str(StoreID.FOLDER_IMPORT_COMPLETE): completion,
            },
            changed_id=f"{StoreID.FOLDER_IMPORT_COMPLETE}.data",
        )[str(SidebarID.SIDEBAR_CONTENT)]["children"]
        options = [
            option
            for component in _walk_components(sidebar)
            if isinstance((component_id := component.get("props", {}).get("id")), dict)
            and component_id.get("type") == RoofCardID.DROPDOWN_APPS
            for option in component.get("props", {}).get("options", [])
        ]
        own = [option for option in options if f"app-{machine[-1]}" in option["label"]]
        assert len(own) == 1, options
        assert not any(f"app-{other[-1]}" in option["label"] for option in options), options
        app_ids[machine] = own[0]["value"]

    selected_store = RoofStore(
        roof_template=RoofConfig(machine="node-a", app_ids=[app_ids["node-a"]])
    )
    plot_layout = _layout_with_roof_controls(layout, selected_store)
    figure_response = _dispatch(
        app,
        client,
        plot_layout,
        str(PlotAreaID.ROOFLINE_PLOT),
        "figure",
        overrides={
            str(StoreID.FOLDER_IMPORT_COMPLETE): completion,
            str(StoreID.ROOF_STORE): selected_store.to_dict(),
        },
        changed_id=f"{StoreID.FOLDER_IMPORT_COMPLETE}.data",
    )
    markers = _marker_traces(_figure_from_response(figure_response))
    assert any("app-a" in trace.get("name", "") for trace in markers), markers

    repeated = _dispatch(
        app,
        client,
        layout,
        str(FolderImportID.STATUS),
        "children",
        overrides={str(StoreID.FOLDER_IMPORT_PAYLOAD): payload},
        changed_id=f"{StoreID.FOLDER_IMPORT_PAYLOAD}.data",
    )
    assert "6 benchmark record(s)" in repeated[str(FolderImportID.STATUS)]["children"]
    repeated_sidebar = _dispatch(
        app,
        client,
        layout,
        str(SidebarID.SIDEBAR_CONTENT),
        "children",
        overrides={
            str(StoreID.ROOF_STORE): RoofStore(roof_template=RoofConfig(machine="node-a")).to_dict(),
            str(StoreID.FOLDER_IMPORT_COMPLETE): repeated[str(StoreID.FOLDER_IMPORT_COMPLETE)]["data"],
        },
        changed_id=f"{StoreID.FOLDER_IMPORT_COMPLETE}.data",
    )[str(SidebarID.SIDEBAR_CONTENT)]["children"]
    repeated_app_options = [
        option
        for component in _walk_components(repeated_sidebar)
        if isinstance((component_id := component.get("props", {}).get("id")), dict)
        and component_id.get("type") == RoofCardID.DROPDOWN_APPS
        for option in component.get("props", {}).get("options", [])
    ]
    assert len(repeated_app_options) == 1

    refresh_root = tmp_path / "benchmark-refresh"
    refresh_machine = refresh_root / "refresh-node"
    refresh_machine.mkdir(parents=True)
    (refresh_machine / "benchmarks.jsonl").write_text(benchmark_files("refresh-node"), encoding="utf-8")
    refreshed = _dispatch(
        app,
        client,
        plot_layout,
        str(PlotAreaID.ROOFLINE_PLOT),
        "figure",
        overrides={
            str(StoreID.ROOF_STORE): selected_store.to_dict(),
            str(StoreID.FOLDER_IMPORT_COMPLETE): completion,
            str(BenchmarkPanelID.JOB): {
                "finished": True,
                "return_code": 0,
                "dry_run": False,
                "output_dir": str(refresh_root),
            },
        },
        changed_id=f"{BenchmarkPanelID.JOB}.data",
    )
    refreshed_markers = _marker_traces(_figure_from_response(refreshed))
    assert any("app-a" in trace.get("name", "") for trace in refreshed_markers), refreshed_markers
