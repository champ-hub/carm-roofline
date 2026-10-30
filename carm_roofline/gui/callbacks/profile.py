from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import dash
from dash import Input, Output, State, callback_context, no_update

from carm_roofline.gui.callbacks.state import CallbackState
from carm_roofline.gui.ids import ProfilePanelID
from carm_roofline.gui.profile_runner import (
    build_profile_command,
    command_preview,
    get_current_job,
    start_profile,
)
from carm_roofline.gui.providers import BenchmarkAppsProvider


def _refresh_profile_apps(state: CallbackState, snapshot: dict[str, Any]) -> None:
    """Update shared application options before completion triggers plot/sidebar callbacks."""
    if not snapshot.get("finished") or snapshot.get("return_code") != 0:
        return
    output_root = Path(str(snapshot.get("output_dir") or state.config.results_dir))
    loaded_apps = BenchmarkAppsProvider(output_root).load()
    if output_root.resolve() == state.config.results_dir.resolve():
        state.apps_by_id.clear()
    state.apps_by_id.update(loaded_apps)
    state.apps_by_id.update(state.imported_apps_by_id)
    state.app_options[:] = [{"label": item.label, "value": item.id} for item in state.apps_by_id.values()]


def register_profile_callbacks(app: dash.Dash, state: CallbackState) -> None:
    """Register profile backend controls and launch/progress handling."""

    @app.callback(
        Output(ProfilePanelID.PAPI_OPTIONS, "style"),
        Output(ProfilePanelID.PERF_OPTIONS, "style"),
        Input(ProfilePanelID.BACKEND, "value"),
    )
    def _show_backend_options(backend: str | None) -> tuple[dict[str, str], dict[str, str]]:
        if backend == "perf":
            return {"display": "none"}, {"display": "block"}
        return {"display": "block"}, {"display": "none"}

    config = state.config

    @app.callback(
        Output(ProfilePanelID.JOB, "data"),
        Output(ProfilePanelID.STATUS, "children"),
        Output(ProfilePanelID.PROGRESS, "value"),
        Output(ProfilePanelID.PROGRESS, "max"),
        Output(ProfilePanelID.LOG, "children"),
        Output(ProfilePanelID.START, "disabled"),
        Output(ProfilePanelID.COMMAND_PREVIEW, "children"),
        Output(ProfilePanelID.INTERVAL, "disabled"),
        Output(ProfilePanelID.ELAPSED, "children"),
        Input(ProfilePanelID.INTERVAL, "n_intervals"),
        Input(ProfilePanelID.START, "n_clicks"),
        State(ProfilePanelID.JOB, "data"),
        State(ProfilePanelID.BACKEND, "value"),
        State(ProfilePanelID.COMMAND, "value"),
        State(ProfilePanelID.AGGREGATION, "value"),
        State(ProfilePanelID.MACHINE_NAME, "value"),
        State(ProfilePanelID.APP_NAME, "value"),
        State(ProfilePanelID.OUTPUT_DIR, "value"),
        State(ProfilePanelID.METRICS, "value"),
        State(ProfilePanelID.ISA, "value"),
        State(ProfilePanelID.DATA_TYPE, "value"),
        State(ProfilePanelID.MERGE_RUNS, "value"),
        State(ProfilePanelID.KEEP_ARTIFACTS, "value"),
        State(ProfilePanelID.NO_PAPI_CACHE, "value"),
        State(ProfilePanelID.PERF_INTERVAL, "value"),
        prevent_initial_call=True,
    )
    def _run_profile(
        _n_intervals: int,
        _n_clicks: int,
        old_data: dict[str, Any] | None,
        backend: str | None,
        command_text: str | None,
        aggregation: str | None,
        machine_name: str | None,
        app_name: str | None,
        output_dir: str | None,
        metrics: list[str] | None,
        isa: list[str] | None,
        data_type: str | None,
        merge_runs: list[str] | None,
        keep_artifacts: list[str] | None,
        no_papi_cache: list[str] | None,
        perf_interval: int | None,
    ) -> tuple[Any, str, str, str, str, bool, Any, bool, str]:
        trigger = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""
        if ProfilePanelID.START in trigger:
            output_path = output_dir or str(config.results_dir)
            try:
                command = build_profile_command(
                    {
                        "backend": backend,
                        "command": command_text,
                        "aggregation": aggregation,
                        "machine_name": machine_name,
                        "app_name": app_name,
                        "output_dir": output_path,
                        "metrics": metrics,
                        "isa": isa,
                        "data_type": data_type,
                        "merge_runs": bool(merge_runs),
                        "keep_artifacts": bool(keep_artifacts),
                        "no_papi_cache": bool(no_papi_cache),
                        "perf_interval": perf_interval,
                    }
                )
                started, job = start_profile(command, output_path)
                snapshot = job.snapshot()
                if not started and not snapshot["finished"]:
                    snapshot["status"] = "A profile is already running in this GUI."
                _refresh_profile_apps(state, snapshot)
                return (
                    snapshot,
                    str(snapshot["status"]),
                    "0",
                    "1",
                    str(snapshot["log"]),
                    not bool(snapshot["finished"]),
                    command_preview(command),
                    bool(snapshot["finished"]),
                    f"Elapsed: {int(float(snapshot['elapsed_seconds']))} s",
                )
            except (TypeError, ValueError, OSError) as exc:
                return (
                    old_data or no_update,
                    f"Invalid profile settings: {exc}",
                    "0",
                    "1",
                    str(exc),
                    False,
                    "Command was not started.",
                    True,
                    "Elapsed: 0 s",
                )

        current_job = get_current_job()
        if current_job is None:
            return no_update, "Ready", "0", "1", "", False, no_update, True, "Elapsed: 0 s"
        snapshot = current_job.snapshot()
        _refresh_profile_apps(state, snapshot)
        serialized = json.dumps(snapshot, sort_keys=True)
        old_serialized = json.dumps(old_data, sort_keys=True) if old_data is not None else None
        job_output: Any = snapshot if serialized != old_serialized else no_update
        return (
            job_output,
            str(snapshot["status"]),
            "0",
            "1",
            str(snapshot["log"]),
            not bool(snapshot["finished"]),
            no_update,
            bool(snapshot["finished"]),
            f"Elapsed: {int(float(snapshot['elapsed_seconds']))} s",
        )
