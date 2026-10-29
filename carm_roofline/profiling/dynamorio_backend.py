"""DynamoRIO software instruction-count profiler with native region timing."""

from __future__ import annotations

import os
import shutil
import subprocess
from contextlib import suppress
from pathlib import Path

from carm_roofline.core import UserError
from carm_roofline.output_utils import detail, info

from .backends import ProfilerBackend, RunResult, RunSpec
from .dynamorio_client import get_or_build_client
from .dynamorio_loader import load_dynamorio_results
from .dynamorio_metrics import AVAILABLE_EVENTS, resolve_dynamorio_metrics
from .shared import MetricDefinition, MetricResolutionConfig, MetricType

_TIMING_ENV = "CARM_REGION_TIMING_OUTPUT"


class DynamoRIOBackend(ProfilerBackend):
    """Collect software counters under DynamoRIO and native timing in a second pass."""

    def __init__(
        self,
        resolution_config: MetricResolutionConfig,
        *,
        dynamorio_home: Path | None = None,
        client_path: Path | None = None,
    ) -> None:
        self._resolution_config = resolution_config or MetricResolutionConfig()
        self._dynamorio_home = dynamorio_home
        self._dynamorio_client_from_cli = client_path is not None
        env_client = os.environ.get("CARM_DYNAMORIO_CLIENT")
        self._client_path = (
            client_path if client_path is not None else (Path(env_client) if env_client is not None else None)
        )
        self._drrun: Path | None = None
        self._resolved_metrics: dict[MetricType, MetricDefinition] = {}

    @property
    def resolved_metrics(self) -> dict[MetricType, MetricDefinition]:
        return dict(self._resolved_metrics)

    @property
    def available_events(self) -> frozenset[str]:
        return AVAILABLE_EVENTS

    @property
    def run_method_name(self) -> str:
        return "DynamoRIO instruction-derived FLOP/byte estimates + native region timing"

    def can_collect(self, events: frozenset[str]) -> bool:
        """DynamoRIO uses fixed software counters and has no hardware counter budget."""
        return events <= self.available_events

    def check_prerequisites(self) -> bool:
        """Locate DynamoRIO and the built client; resolve fixed software metrics."""
        drrun: Path | None = None
        configured_home = self._dynamorio_home
        env_home = os.environ.get("DYNAMORIO_HOME")
        if configured_home is None and env_home is not None:
            configured_home = Path(env_home)
        if configured_home is not None:
            candidate = configured_home / "bin64" / "drrun"
            if candidate.is_file():
                drrun = candidate
            else:
                raise UserError(
                    f"DynamoRIO drrun not found in explicitly selected installation {configured_home}: {candidate}"
                )
        else:
            found = shutil.which("drrun")
            if found:
                drrun = Path(found)
        if drrun is None:
            raise UserError(
                "DynamoRIO drrun not found. Install DynamoRIO 11+ and set DYNAMORIO_HOME or pass --dynamorio-home."
            )
        if self._client_path is not None:
            if not self._client_path.is_file():
                option = "--dynamorio-client" if self._dynamorio_client_from_cli else "CARM_DYNAMORIO_CLIENT"
                raise UserError(
                    f"DynamoRIO client override from {option} does not exist or is not a file: {self._client_path}"
                )
        else:
            self._client_path = get_or_build_client(drrun)
        self._drrun = drrun
        self._resolved_metrics = resolve_dynamorio_metrics()
        detail(f"DynamoRIO runtime: {drrun}")
        detail(f"DynamoRIO client: {self._client_path}")
        return False

    def profile(self, run_spec: RunSpec, command: list[str], cwd: Path) -> RunResult:
        if self._drrun is None:
            raise RuntimeError("DynamoRIOBackend.check_prerequisites() must be called before profile()")
        if not command:
            raise UserError("No command specified for DynamoRIO profiling")

        run_spec.output_dir.mkdir(parents=True, exist_ok=True)
        count_path = run_spec.output_dir / "dynamorio_counts.jsonl"
        timing_path = run_spec.output_dir / "native_timing.jsonl"

        def clear_outputs() -> None:
            for output_path in (count_path, timing_path):
                with suppress(FileNotFoundError):
                    output_path.unlink()

        clear_outputs()
        dr_command = [
            str(self._drrun),
            "-follow_children",
            "-c",
            str(self._client_path),
            "-o",
            str(count_path),
            "--",
            *command,
        ]
        info(f"Running DynamoRIO instruction-count pass: {' '.join(dr_command)}")
        count_env = os.environ.copy()
        count_env.pop(_TIMING_ENV, None)
        try:
            count_run = subprocess.run(dr_command, cwd=cwd, env=count_env, check=False)
        except FileNotFoundError as exc:
            clear_outputs()
            raise UserError(f"Command not found: {dr_command[0]}") from exc
        except OSError as exc:
            clear_outputs()
            raise UserError(f"Failed to start DynamoRIO: {exc}") from exc
        if count_run.returncode:
            clear_outputs()
            raise UserError(
                f"DynamoRIO instruction-count pass failed with exit code {count_run.returncode}; "
                "discarded count output to avoid partial results. Native timing was not run."
            )

        native_env = os.environ.copy()
        native_env[_TIMING_ENV] = str(timing_path)
        info(f"Running uninstrumented native timing pass: {' '.join(command)}")
        try:
            timing_run = subprocess.run(command, cwd=cwd, env=native_env, check=False)
        except FileNotFoundError as exc:
            clear_outputs()
            raise UserError(f"Command not found: {command[0]}") from exc
        except OSError as exc:
            clear_outputs()
            raise UserError(f"Failed to start native timing pass: {exc}") from exc

        if timing_run.returncode:
            clear_outputs()
            raise UserError(
                f"Native timing pass failed with exit code {timing_run.returncode}; "
                "discarded count and timing output to avoid partial results."
            )
        if not count_path.is_file():
            raise UserError(
                f"DynamoRIO instruction-count pass produced no output ({count_path}; "
                f"exit code {count_run.returncode}). "
                "Check the DynamoRIO client and application marker symbols."
            )
        if not timing_path.is_file():
            raise UserError(
                f"Native timing pass produced no output ({timing_path}; exit code {timing_run.returncode}). "
                "Check that the application links carm_region_timing.c and that its region markers run."
            )
        ranks = load_dynamorio_results(count_path, timing_path)
        return RunResult(exit_code=count_run.returncode or timing_run.returncode, ranks=ranks)
