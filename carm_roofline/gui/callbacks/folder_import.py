from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path
from typing import Any

import dash
from dash import Input, Output, no_update

from carm_roofline.gui.callbacks.state import CallbackState
from carm_roofline.gui.data import DropdownOption
from carm_roofline.gui.ids import FolderImportID, StoreID
from carm_roofline.roofline_assembly import (
    ApplicationRecord,
    BenchmarkRecord,
    discover_filter_options,
    load_applications,
    load_benchmarks,
)

_RECOGNIZED = {"benchmarks.jsonl", "applications.jsonl"}


def _decode_payload(payload: Any) -> list[tuple[str, str]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("files"), list):
        raise ValueError("the folder payload is invalid")
    paths: set[str] = set()
    files: list[tuple[str, str]] = []
    for item in payload["files"]:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("a file path is invalid")
        path = item["path"]
        components = path.split("/")
        if any(not part or part in {".", ".."} or "\\" in part for part in components):
            raise ValueError("a file path contains an unsafe component")
        if path in paths:
            raise ValueError("the folder contains duplicate file paths")
        paths.add(path)
        content = item.get("content")
        if not isinstance(content, str):
            raise ValueError("a file does not contain text")
        if components[-1] not in _RECOGNIZED:
            continue
        if len(components) not in {2, 3}:
            continue
        for line in content.splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} contains malformed JSONL: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path} contains a JSON value that is not an object")
        files.append((path, content))
    if not files:
        raise ValueError("the folder contains no benchmark or application JSONL files")
    roots = {path.split("/")[0] for path, _ in files}
    if len(roots) != 1:
        raise ValueError("the drop must contain one results root")
    return files


def _load_payload(payload: Any) -> tuple[list[BenchmarkRecord], list[ApplicationRecord], int]:
    files = _decode_payload(payload)
    root_name = files[0][0].split("/")[0]
    direct = {path.split("/")[-1]: content for path, content in files if len(path.split("/")) == 2}
    machines: dict[str, dict[str, str]] = {}
    if direct:
        machines[root_name] = direct
    else:
        for path, content in files:
            parts = path.split("/")
            if len(parts) == 3:
                machines.setdefault(parts[1], {})[parts[2]] = content
    if not machines:
        raise ValueError("the folder contains no machine results")

    benchmarks: list[BenchmarkRecord] = []
    applications: list[ApplicationRecord] = []
    with tempfile.TemporaryDirectory(prefix="carm-import-") as temp_dir:
        temp_root = Path(temp_dir)
        for machine_index, (machine, machine_files) in enumerate(machines.items()):
            paths: dict[str, Path] = {}
            for file_index, (name, content) in enumerate(machine_files.items()):
                generated = temp_root / f"{machine_index}-{file_index}.jsonl"
                generated.write_text(content, encoding="utf-8")
                paths[name] = generated
            if "benchmarks.jsonl" in paths:
                benchmarks.extend(load_benchmarks(paths["benchmarks.jsonl"]))
            if "applications.jsonl" in paths:
                applications.extend(load_applications(paths["applications.jsonl"], machine=machine))
    if not benchmarks and not applications:
        raise ValueError("the folder contains no valid benchmark or application records")
    return benchmarks, applications, len(machines)


def _merge_imports(state: CallbackState, benchmarks: list[BenchmarkRecord], apps: list[ApplicationRecord]) -> None:
    imported_by_json = {json.dumps(record, sort_keys=True): record for record in state.imported_records}
    imported_by_json.update({json.dumps(record, sort_keys=True): record for record in benchmarks})
    imported_records = list(imported_by_json.values())
    imported_apps = dict(state.imported_apps_by_id)
    imported_apps.update({app.id: app for app in apps})

    combined_by_json = {json.dumps(record, sort_keys=True): record for record in state.records}
    combined_by_json.update(imported_by_json)
    records = list(combined_by_json.values())
    apps_by_id = dict(state.apps_by_id)
    apps_by_id.update(imported_apps)
    options = discover_filter_options(records) if records else None
    app_options: list[DropdownOption] = [{"label": app.label, "value": app.id} for app in apps_by_id.values()]

    state.imported_records[:] = imported_records
    state.imported_apps_by_id.clear()
    state.imported_apps_by_id.update(imported_apps)
    state.records[:] = records
    state.apps_by_id.clear()
    state.apps_by_id.update(apps_by_id)
    state.app_options[:] = app_options
    state.options = options


def register_folder_import_callbacks(app: dash.Dash, state: CallbackState) -> None:
    @app.callback(
        Output(FolderImportID.STATUS, "children"),
        Output(StoreID.FOLDER_IMPORT_COMPLETE, "data"),
        Input(StoreID.FOLDER_IMPORT_PAYLOAD, "data"),
        prevent_initial_call=True,
    )
    def _import_folder(payload: Any) -> tuple[Any, Any]:
        if payload is None:
            return no_update, no_update
        try:
            benchmarks, applications, machine_count = _load_payload(payload)
            _merge_imports(state, benchmarks, applications)
        except Exception as exc:
            return f"Import error: {exc}", no_update
        summary = (
            f"Imported {machine_count} machine folder(s): "
            f"{len(benchmarks)} benchmark record(s), {len(applications)} application record(s)."
        )
        return summary, {"token": uuid.uuid4().hex}
