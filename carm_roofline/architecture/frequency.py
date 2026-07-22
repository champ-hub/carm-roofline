"""Utilities for setting CPU frequency via sysfs.

These helpers are Python equivalents to the legacy C frequency setter and
are useful when you do not want to rebuild or run the autoconfig helper.
"""

from __future__ import annotations

from collections.abc import Generator, Iterable
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING

from carm_roofline.core import UserError

if TYPE_CHECKING:
    from carm_roofline.context import CARMContext

SYS_CPU = Path("/sys/devices/system/cpu")


def _iter_scaling_max_freq_files() -> Iterable[Path]:
    for cpu_dir in SYS_CPU.glob("cpu[0-9]*"):
        candidate = cpu_dir / "cpufreq" / "scaling_max_freq"
        if candidate.exists():
            yield candidate


def set_cpu_frequency(new_max_freq_khz: int) -> None:
    """Set max frequency for all CPUs (requires root on most systems)."""
    if new_max_freq_khz <= 0:
        raise ValueError("new_max_freq_khz must be positive")

    paths = list(_iter_scaling_max_freq_files())
    if not paths:
        raise UserError(
            "No cpufreq control files found at /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq. "
            "Frequency setting is not supported on this system (no cpufreq driver)."
        )

    errors = []
    for path in paths:
        try:
            with path.open("w") as f:
                f.write(str(new_max_freq_khz))
        except OSError as exc:
            errors.append((path, exc))

    if errors:
        err = errors[0][1]
        count = len(errors)
        raise OSError(f"could not write to {count} CPU cpufreq file(s): {err.strerror}")


def read_cpu_frequencies() -> dict[str, str]:
    """Read current max frequencies (for verification/debug)."""
    freqs: dict[str, str] = {}
    for path in _iter_scaling_max_freq_files():
        try:
            freqs[path.parent.parent.name] = path.read_text().strip()
        except OSError:
            freqs[path.parent.parent.name] = "<unreadable>"
    return freqs


def read_single_cpu_frequency_hz() -> int | None:
    """Read scaling_max_freq from the first available CPU, return value in Hz.

    Returns ``None`` when no cpufreq sysfs entry is readable (containers, VMs, systems without cpufreq driver).
    """
    for path in _iter_scaling_max_freq_files():
        try:
            return int(path.read_text().strip()) * 1000  # sysfs is kHz
        except (OSError, ValueError):
            continue
    return None


def _save_cpu_frequencies() -> dict[str, int]:
    """Read current scaling_max_freq from all CPUs for later restoration."""
    freqs: dict[str, int] = {}
    for path in _iter_scaling_max_freq_files():
        with suppress(OSError, ValueError):
            freqs[path.parent.parent.name] = int(path.read_text().strip())
    return freqs


def _restore_cpu_frequencies(original: dict[str, int]) -> None:
    """Restore saved CPU frequencies. Best-effort — warns on failure, never raises."""
    from carm_roofline.output_utils import warn

    errors: list[tuple[str, OSError]] = []
    for cpu, freq_khz in original.items():
        path = SYS_CPU / cpu / "cpufreq" / "scaling_max_freq"
        try:
            with path.open("w") as f:
                f.write(str(freq_khz))
        except OSError as exc:
            errors.append((cpu, exc))

    if errors:
        for cpu, err in errors:
            warn(f"Failed to restore frequency for {cpu}: {err}")
        warn(
            f"Failed to restore frequency on {len(errors)} CPU(s) — they may remain at the benchmark frequency."
            "You can try to restore them manually: "
            "echo <freq> | sudo tee /sys/.../cpufreq/scaling_max_freq"
        )


def _poll_frequencies(
    target: dict[str, int],
    *,
    tolerance: float = 0.0,
    retries: int = 5,
    delay: float = 0.05,
) -> list[str]:
    """Poll read_cpu_frequencies until all CPUs match target. Returns mismatch descriptions."""
    import time

    for attempt in range(retries):
        current = read_cpu_frequencies()
        mismatches: list[str] = []
        for cpu, target_khz in target.items():
            actual_str = current.get(cpu)
            try:
                if actual_str is not None:
                    actual_khz = int(actual_str)
                    if tolerance > 0:
                        deviation = abs(actual_khz - target_khz) / target_khz
                        if deviation > tolerance:
                            mismatches.append(f"{cpu}: target {target_khz} kHz, got {actual_khz} kHz")
                    elif actual_khz != target_khz:
                        mismatches.append(f"{cpu}: target {target_khz} kHz, got {actual_khz} kHz")
            except ValueError:
                mismatches.append(f"{cpu}: unreadable ({actual_str})")
        if not mismatches:
            return []
        if attempt < retries - 1:
            time.sleep(delay)
    return mismatches


@contextmanager
def maybe_set_cpu_frequency(context: CARMContext) -> Generator[None, None, None]:
    """Context manager: save original CPU freq, set new, restore on exit.

    Skips (no-op) when set_frequency is False, in dry-run mode, or when running under a simulator. Raises UserError if
    the sysfs write fails (e.g. not root) or no cpufreq driver is available. Restores saved frequencies on normal exit,
    exceptions, and KeyboardInterrupt (SIGINT).
    """
    from carm_roofline.output_utils import detail, warn

    # Guard: skip when not needed
    if not context.architecture.set_frequency:
        yield
        return

    if context.run_config.dry_run:
        detail("Dry run: skipping CPU frequency set.")
        yield
        return

    if context.exec_interface.sim_cmd:
        detail("Simulation mode: skipping CPU frequency set.")
        yield
        return

    # Save original
    original = _save_cpu_frequencies()
    detail(f"Saved original CPU frequencies for {len(original)} CPU(s)")

    # Set new frequency
    target_freq = context.architecture.get_frequency_for_isa(context.architecture.isa[0].name)
    freq_khz = int(target_freq.as_kilohertz())
    detail(f"Setting CPU frequency to {target_freq} ({freq_khz} kHz)...")
    try:
        set_cpu_frequency(freq_khz)
    except OSError as e:
        raise UserError(
            f"Failed to set CPU frequency: {e}\nTry running with sudo/root privileges, or omit --set-frequency."
        ) from e

    # Verify set
    mismatches = _poll_frequencies(
        dict.fromkeys(original, freq_khz),
        tolerance=0.01,
    )
    if mismatches:
        warn("CPU frequency verification issues:\n" + "\n".join(mismatches))
    else:
        detail("CPU frequency set and verified on all CPUs.")

    # Store verified actual frequency so JSONL serializers tag records with it.
    context.architecture.actual_frequency_hz = freq_khz * 1000

    # Yield to caller (benchmark runs here)
    try:
        yield
    finally:
        # Restore original
        if original:
            detail("Restoring original CPU frequencies...")
            _restore_cpu_frequencies(original)

            # Verify restore
            restore_issues = _poll_frequencies(original)
            if restore_issues:
                warn("CPU frequency restore verification issues:\n" + "\n".join(restore_issues))
            else:
                detail("Original CPU frequencies restored and verified on all CPUs.")
