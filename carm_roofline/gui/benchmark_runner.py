from __future__ import annotations

import json
import os
import selectors
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any

_PROGRESS_PREFIX = "CARM_PROGRESS "
_LOG_LIMIT = 20_000


def _tokens(value: str | None) -> list[str]:
    return [part for part in (value or "").replace(",", " ").split() if part]


def build_benchmark_command(values: dict[str, Any]) -> list[str]:
    """Build a safe argument vector for the selected benchmark controls."""
    import sys

    args = [sys.executable, "-m", "carm_roofline.carm", "benchmark"]
    args.extend(["--test", str(values["test"]), "--test-time", str(values["test_time"])])
    isas = values.get("isa") or ["auto"]
    if "auto" not in isas:
        args.extend(["--isa", *[str(isa) for isa in isas]])
    args.extend(["--data-type", *[str(value) for value in values.get("data_types") or ["f32"]]])
    args.extend(["--threads", *_tokens(str(values.get("threads", "1")))])

    if values.get("name"):
        args.extend(["--name", str(values["name"])])
    memory_targets = values.get("memory_target") or ["all"]
    args.extend(["--mem-target", *[str(value) for value in memory_targets]])
    args.extend(["--num-ops", str(values.get("num_ops") or 32768)])
    instructions = values.get("instructions") or ["add", "fma"]
    args.extend(["--instruction", *[str(value) for value in instructions]])
    args.extend(["--ld-st-ratio", *_tokens(str(values.get("ld_st_ratio", "2:1")))])
    args.extend(["--arith-mem-ratio", str(values.get("arith_mem_ratio", "2:3"))])
    ai_min = values.get("ai_min") if values.get("ai_min") is not None else 2**-4
    ai_max = values.get("ai_max") if values.get("ai_max") is not None else 2**5
    args.extend(["--ai-range", str(ai_min), str(ai_max), "--ai-points", str(values.get("ai_points") or 10)])
    memory_sizes_value = values.get("memory_sizes")
    memory_sizes = _tokens(memory_sizes_value if isinstance(memory_sizes_value, str) else "")
    if memory_sizes:
        args.extend(["--mem-test-sizes", *memory_sizes])
    if values.get("interleaved"):
        args.append("--interleaved")
    optional_args = (
        ("topology", "--topology-config"),
        ("vector_length", "--vector-length"),
        ("vector_lmul", "--vector-lmul"),
        ("frequency", "--frequency"),
        ("compiler", "--compiler"),
        ("simulator", "--sim-cmd"),
    )
    for key, option in optional_args:
        value = values.get(key)
        if value is not None and str(value).strip():
            args.extend([option, str(value).strip()])
    if values.get("set_frequency"):
        args.append("--set-frequency")
    args.extend(["--output-dir", str(values.get("output_dir", ""))])
    args.extend(["--output-fmt", *[str(value) for value in values.get("output_formats") or ["jsonl"]]])
    if values.get("dry_run"):
        args.append("--dry-run")
    if values.get("keep_artifacts"):
        args.append("--keep-artifacts")
    args.extend(["--progress-format", "jsonl"])
    return args


def command_preview(command: list[str]) -> str:
    """Return a safely quoted display form of a command argument vector."""
    return shlex.join(command)


@dataclass
class BenchmarkJob:
    status: str = "Ready"
    stage: str = ""
    completed: int = 0
    total: int = 1
    return_code: int | None = None
    log: str = ""
    finished: bool = False
    output_dir: str = ""
    dry_run: bool = False
    _started_at: float = field(default_factory=time.monotonic, repr=False)
    _finished_at: float | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def append_log(self, text: str) -> None:
        with self._lock:
            self.log = (self.log + text)[-_LOG_LIMIT:]

    def update_event(self, line: str) -> None:
        try:
            event = json.loads(line[len(_PROGRESS_PREFIX) :])
        except (json.JSONDecodeError, TypeError):
            self.append_log(line)
            return
        if not isinstance(event, dict):
            self.append_log(line)
            return
        with self._lock:
            name = event.get("event")
            if name == "stage":
                self.stage = str(event.get("stage", ""))
                self.status = f"{self.stage.title()}…"
            elif name == "measurement":
                self.stage = "running"
                self.completed = int(event.get("completed", 0))
                self.total = max(1, int(event.get("total", 1)))
                self.status = f"Running: {self.completed} / {self.total} measurements"
            elif name == "complete":
                self.status = "Writing results…"
            elif name == "error":
                self.status = f"Failed: {event.get('message', 'benchmark error')}"

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": self.status,
                "stage": self.stage,
                "completed": self.completed,
                "total": self.total,
                "return_code": self.return_code,
                "log": self.log,
                "finished": self.finished,
                "output_dir": self.output_dir,
                "dry_run": self.dry_run,
                "elapsed_seconds": (self._finished_at or time.monotonic()) - self._started_at,
            }


_job_lock = threading.Lock()
_job_state: dict[str, BenchmarkJob | None] = {"current": None}


def start_benchmark(command: list[str], output_dir: str, dry_run: bool) -> tuple[bool, BenchmarkJob]:
    """Start one server-wide benchmark subprocess, or return the active job."""
    with _job_lock:
        current_job = _job_state["current"]
        if current_job is not None and not current_job.finished:
            return False, current_job
        job = BenchmarkJob(status="Starting benchmark…", output_dir=output_dir, dry_run=dry_run)
        _job_state["current"] = job
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as exc:
            job.status = f"Failed to start benchmark: {exc}"
            job.finished = True
            job._finished_at = time.monotonic()
            job.return_code = 127
            return False, job
        threading.Thread(target=_monitor_process, args=(process, job), daemon=True).start()
        return True, job


def get_current_job() -> BenchmarkJob | None:
    with _job_lock:
        return _job_state["current"]


def _monitor_process(process: subprocess.Popen[bytes], job: BenchmarkJob) -> None:
    selector = selectors.DefaultSelector()
    if process.stdout is not None:
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    if process.stderr is not None:
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    buffers: dict[str, bytearray] = {"stdout": bytearray(), "stderr": bytearray()}
    while selector.get_map():
        for key, _ in selector.select(timeout=0.25):
            chunk = os.read(key.fd, 4096)
            if not chunk:
                selector.unregister(key.fileobj)
                continue
            stream = str(key.data)
            buffer = buffers[stream]
            buffer.extend(chunk)
            while b"\n" in buffer:
                raw_line, _, remainder = buffer.partition(b"\n")
                buffers[stream] = bytearray(remainder)
                buffer = buffers[stream]
                line = raw_line.decode("utf-8", errors="replace")
                if stream == "stderr" and line.startswith(_PROGRESS_PREFIX):
                    job.update_event(line)
                elif line:
                    job.append_log(line + "\n")
    for stream, buffer in buffers.items():
        if buffer:
            line = buffer.decode("utf-8", errors="replace")
            if stream == "stderr" and line.startswith(_PROGRESS_PREFIX):
                job.update_event(line)
            else:
                job.append_log(line)
    return_code = process.wait()
    with job._lock:
        job.return_code = return_code
        job.finished = True
        job._finished_at = time.monotonic()
        if return_code == 0:
            if job.dry_run:
                job.status = "Code generated"
            elif not job.status.startswith("Failed"):
                job.status = "Benchmark complete"
        elif not job.status.startswith("Failed"):
            job.status = f"Benchmark failed with exit code {return_code}"
