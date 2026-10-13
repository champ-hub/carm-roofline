from __future__ import annotations

import json
from typing import Any

import dash
from dash import Input, Output, State, callback_context, no_update

from carm_roofline.gui.benchmark_runner import (
    build_benchmark_command,
    command_preview,
    get_current_job,
    start_benchmark,
)
from carm_roofline.gui.callbacks.state import CallbackState
from carm_roofline.gui.ids import BenchmarkPanelID


def register_benchmark_callbacks(app: dash.Dash, state: CallbackState) -> None:
    """Register benchmark launch and progress controls."""
    config = state.config

    @app.callback(
        Output(BenchmarkPanelID.JOB, "data"),
        Output(BenchmarkPanelID.STATUS, "children"),
        Output(BenchmarkPanelID.PROGRESS, "value"),
        Output(BenchmarkPanelID.PROGRESS, "max"),
        Output(BenchmarkPanelID.LOG, "children"),
        Output(BenchmarkPanelID.START, "disabled"),
        Output(BenchmarkPanelID.COMMAND_PREVIEW, "children"),
        Output(BenchmarkPanelID.INTERVAL, "disabled"),
        Output(BenchmarkPanelID.ELAPSED, "children"),
        Input(BenchmarkPanelID.INTERVAL, "n_intervals"),
        Input(BenchmarkPanelID.START, "n_clicks"),
        State(BenchmarkPanelID.JOB, "data"),
        State(BenchmarkPanelID.TEST, "value"),
        State(BenchmarkPanelID.ISA, "value"),
        State(BenchmarkPanelID.DATA_TYPES, "value"),
        State(BenchmarkPanelID.THREADS, "value"),
        State(BenchmarkPanelID.TEST_TIME, "value"),
        State(BenchmarkPanelID.NAME, "value"),
        State(BenchmarkPanelID.MEMORY_TARGET, "value"),
        State(BenchmarkPanelID.NUM_OPS, "value"),
        State(BenchmarkPanelID.INSTRUCTIONS, "value"),
        State(BenchmarkPanelID.LD_ST_RATIO, "value"),
        State(BenchmarkPanelID.ARITH_MEM_RATIO, "value"),
        State(BenchmarkPanelID.AI_MIN, "value"),
        State(BenchmarkPanelID.AI_MAX, "value"),
        State(BenchmarkPanelID.AI_POINTS, "value"),
        State(BenchmarkPanelID.MEMORY_SIZES, "value"),
        State(BenchmarkPanelID.INTERLEAVED, "value"),
        State(BenchmarkPanelID.TOPOLOGY, "value"),
        State(BenchmarkPanelID.VECTOR_LENGTH, "value"),
        State(BenchmarkPanelID.VECTOR_LMUL, "value"),
        State(BenchmarkPanelID.FREQUENCY, "value"),
        State(BenchmarkPanelID.SET_FREQUENCY, "value"),
        State(BenchmarkPanelID.COMPILER, "value"),
        State(BenchmarkPanelID.SIMULATOR, "value"),
        State(BenchmarkPanelID.OUTPUT_DIR, "value"),
        State(BenchmarkPanelID.OUTPUT_FORMATS, "value"),
        State(BenchmarkPanelID.DRY_RUN, "value"),
        State(BenchmarkPanelID.KEEP_ARTIFACTS, "value"),
        prevent_initial_call=True,
    )
    def _run_benchmark(
        _n_intervals: int,
        _n_clicks: int,
        old_data: dict[str, Any] | None,
        test: str,
        isa: list[str] | None,
        data_types: list[str] | None,
        threads: str,
        test_time: float | None,
        name: str | None,
        memory_target: list[str] | None,
        num_ops: int | None,
        instructions: list[str] | None,
        ld_st_ratio: str,
        arith_mem_ratio: str,
        ai_min: float | None,
        ai_max: float | None,
        ai_points: int | None,
        memory_sizes: str | None,
        interleaved: list[str] | None,
        topology: str | None,
        vector_length: int | None,
        vector_lmul: int | None,
        frequency: str | None,
        set_frequency: list[str] | None,
        compiler: str | None,
        simulator: str | None,
        output_dir: str | None,
        output_formats: list[str] | None,
        dry_run: list[str] | None,
        keep_artifacts: list[str] | None,
    ) -> tuple[Any, str, str, str, str, bool, Any, bool, str]:
        trigger = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""
        if BenchmarkPanelID.START in trigger:
            try:
                command = build_benchmark_command(
                    {
                        "test": test,
                        "isa": isa,
                        "data_types": data_types,
                        "threads": threads,
                        "test_time": test_time or 25.0,
                        "name": name,
                        "memory_target": memory_target,
                        "num_ops": num_ops,
                        "instructions": instructions,
                        "ld_st_ratio": ld_st_ratio,
                        "arith_mem_ratio": arith_mem_ratio,
                        "ai_min": ai_min,
                        "ai_max": ai_max,
                        "ai_points": ai_points,
                        "memory_sizes": memory_sizes,
                        "interleaved": bool(interleaved),
                        "topology": topology,
                        "vector_length": vector_length,
                        "vector_lmul": vector_lmul,
                        "frequency": frequency,
                        "set_frequency": bool(set_frequency),
                        "compiler": compiler,
                        "simulator": simulator,
                        "output_dir": output_dir or str(config.results_dir),
                        "output_formats": output_formats,
                        "dry_run": bool(dry_run),
                        "keep_artifacts": bool(keep_artifacts),
                    }
                )
                started, job = start_benchmark(
                    command,
                    output_dir or str(config.results_dir),
                    bool(dry_run),
                )
                snapshot = job.snapshot()
                if not started and not snapshot["finished"]:
                    snapshot["status"] = "A benchmark is already running in this GUI."
                return (
                    snapshot,
                    str(snapshot["status"]),
                    str(snapshot["completed"]),
                    str(max(1, int(snapshot["total"]))),
                    str(snapshot["log"]),
                    not bool(snapshot["finished"]),
                    command_preview(command),
                    bool(snapshot["finished"]),
                    f"Elapsed: {int(float(snapshot['elapsed_seconds']))} s",
                )
            except (TypeError, ValueError, OSError) as exc:
                return (
                    old_data or no_update,
                    f"Invalid benchmark settings: {exc}",
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
        serialized = json.dumps(snapshot, sort_keys=True)
        old_serialized = json.dumps(old_data, sort_keys=True) if old_data is not None else None
        job_output: Any = snapshot if serialized != old_serialized else no_update
        return (
            job_output,
            str(snapshot["status"]),
            str(snapshot["completed"]),
            str(max(1, int(snapshot["total"]))),
            str(snapshot["log"]),
            not bool(snapshot["finished"]),
            no_update,
            bool(snapshot["finished"]),
            f"Elapsed: {int(float(snapshot['elapsed_seconds']))} s",
        )
