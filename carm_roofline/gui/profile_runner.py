from __future__ import annotations

import io
import math
import selectors
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, cast

from carm_roofline.results_paths import default_results_root


@dataclass
class ProfileJob:
    """State for one profile subprocess and its captured output."""

    status: str = "Ready"
    output_dir: str = ""
    log: str = ""
    finished: bool = False
    return_code: int | None = None
    started_at: float = field(default_factory=time.monotonic)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def append_output(self, text: str) -> None:
        with self._lock:
            self.log = (self.log + text)[-20000:]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": self.status,
                "output_dir": self.output_dir,
                "log": self.log,
                "finished": self.finished,
                "return_code": self.return_code,
                "elapsed_seconds": max(0.0, time.monotonic() - self.started_at),
            }


_job_lock = threading.Lock()
_job_state: dict[str, ProfileJob | None] = {"current": None}


def build_profile_command(values: dict[str, Any]) -> list[str]:
    """Build a safe argv for ``carm profile`` from the configurator values."""
    command_text = str(values.get("command") or "").strip()
    application_argv = shlex.split(command_text)
    if not application_argv:
        raise ValueError("enter an application command to profile")

    args = [sys.executable, "-m", "carm_roofline.carm", "profile"]
    args.extend(["--backend", str(values.get("backend") or "papi")])
    args.extend(["--aggregation", str(values.get("aggregation") or "global")])
    for key, option in (("machine_name", "--machine-name"), ("app_name", "--app-name")):
        if values.get(key):
            args.extend([option, str(values[key])])

    args.extend(["--output-dir", str(values.get("output_dir") or default_results_root())])
    metrics = values.get("metrics") or []
    if metrics:
        args.append("--metrics")
        args.extend(str(metric) for metric in metrics)
    isas = values.get("isa") or []
    if isas:
        args.append("--isa")
        args.extend(str(isa) for isa in isas)
    if values.get("data_type"):
        args.extend(["--data-type", str(values["data_type"])])
    if values.get("merge_runs"):
        args.append("--merge-runs")
    if values.get("keep_artifacts"):
        args.append("--keep-artifacts")
    if values.get("backend") == "papi" and values.get("no_papi_cache"):
        args.append("--no-papi-cache")
    if values.get("backend") == "perf" and values.get("perf_interval") is not None:
        raw_interval = values["perf_interval"]
        try:
            interval = float(raw_interval)
        except (TypeError, ValueError) as exc:
            raise ValueError("perf interval must be a positive whole number of milliseconds") from exc
        if not math.isfinite(interval) or interval <= 0 or not interval.is_integer():
            raise ValueError("perf interval must be a positive whole number of milliseconds")
        args.extend(["--perf-interval", str(int(interval))])
    args.append("--")
    args.extend(application_argv)
    return args


def command_preview(command: list[str]) -> str:
    """Render argv in a shell-safe, human-readable form."""
    return shlex.join(command)


def start_profile(command: list[str], output_dir: str) -> tuple[bool, ProfileJob]:
    """Start one server-wide profile subprocess, or return the active job."""
    with _job_lock:
        current_job = _job_state["current"]
        if current_job is not None and not current_job.finished:
            return False, current_job
        job = ProfileJob(status="Starting profile…", output_dir=output_dir)
        _job_state["current"] = job
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as exc:
            job.status = f"Failed to start profile: {exc}"
            job.finished = True
            job.return_code = 127
            return True, job
        threading.Thread(target=_monitor_process, args=(process, job), daemon=True).start()
        return True, job


def get_current_job() -> ProfileJob | None:
    with _job_lock:
        return _job_state["current"]


def _monitor_process(process: subprocess.Popen[bytes], job: ProfileJob) -> None:
    selector = selectors.DefaultSelector()
    if process.stdout is not None:
        selector.register(process.stdout, selectors.EVENT_READ, process.stdout)
    if process.stderr is not None:
        selector.register(process.stderr, selectors.EVENT_READ, process.stderr)
    with job._lock:
        job.status = "Running profile"
    while selector.get_map():
        for key, _ in selector.select(timeout=0.2):
            stream = cast(io.BufferedReader, key.data)
            chunk = stream.read1(4096)
            if not chunk:
                selector.unregister(key.fileobj)
                continue
            text = chunk.decode(errors="replace")
            job.append_output(text)
    return_code = process.wait()
    with job._lock:
        job.return_code = return_code
        job.finished = True
        job.status = "Profile complete" if return_code == 0 else f"Profile failed with exit code {return_code}"
    selector.close()
