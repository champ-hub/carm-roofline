"""Hardware/ISA detection helper wrapping the C probe (probe_arch.c).

Produces a structured dict parsed from the JSON emitted by the probe.
Fields include: arch, vendor, isa list, caches_kib, vector_bytes.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from carm_roofline.core import Frequency, UserError
from carm_roofline.output_utils import debug, detail, warn

if TYPE_CHECKING:
    from carm_roofline.isa import BaseISA

from .memory import MemoryTopology


@dataclass
class DetectedArchitecture:
    """Type-safe container for auto-detected architecture parameters.

    Holds only finalized, normalized typed fields. No raw or intermediate values.

    Attributes:
        isa: List of detected ISA names (e.g., ["x86", "x86_avx2"])
        memory_topology: System memory topology parsed from sysfs (Linux native) or None if unavailable
        vector_length: Vector register length in bytes (ARM SVE, RISC-V RVV)
        frequency: Detected processor frequency as Frequency wrapper
        frequency_nominal: Optional nominal/base frequency as Frequency wrapper
        isa_frequencies: Per-ISA frequency mapping as dict of Frequency wrappers
        arch: Architecture string (e.g., "x86_64", "aarch64")
        vendor: CPU vendor string (e.g., "GenuineIntel", "AuthenticAMD")
        model_name: CPU model name (e.g., "AMD Ryzen 7 7735HS with Radeon Graphics")
    """

    # ISA detection
    isa: list[str] | None = None
    vendor: str | None = None
    arch: str | None = None
    model_name: str | None = None

    # Memory topology (sysfs-based, native Linux only; None for cross-execution)
    memory_topology: MemoryTopology | None = None

    # Vector configuration
    vector_length: int | None = None

    # Frequency detection (already normalized to Frequency objects)
    frequency: Frequency | None = None
    frequency_nominal: Frequency | None = None
    isa_frequencies: dict[str, Frequency] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for logging/debugging, excluding None values.

        Returns:
            Dict containing only non-None normalized fields.
        """
        result = {}
        for field_name in [
            "isa",
            "memory_topology",
            "vector_length",
            "frequency",
            "frequency_nominal",
            "isa_frequencies",
            "arch",
            "vendor",
            "model_name",
        ]:
            value = getattr(self, field_name, None)
            if value is not None:
                result[field_name] = value
        return result


class DetectionBuilder:
    """Accumulates DetectedArchitecture fields from multiple probes with conflict detection.

    Enforces error-on-conflict: if a field is already set with a different value,
    raises ValueError with source tracking.
    """

    def __init__(self) -> None:
        """Initialize empty field accumulator."""
        self._fields: dict[str, Any] = {}
        self._sources: dict[str, str] = {}  # Track which source set each field

    def merge_fields(self, source: str, fields: dict[str, Any]) -> None:
        """Merge fields from a probe/detector with conflict checking.

        Args:
            source: Name of the source (e.g., "detect_features", "detect_cache")
            fields: Dict of field_name -> value to merge

        Raises:
            ValueError: If a field conflicts with an existing value from a different source
        """
        for field_name, value in fields.items():
            if field_name in self._fields:
                existing = self._fields[field_name]
                existing_source = self._sources[field_name]
                # Check for actual conflict (different non-None values)
                if existing != value:
                    raise ValueError(
                        f"Conflict in field '{field_name}': "
                        f"source '{existing_source}' set {existing!r}, "
                        f"but source '{source}' set {value!r}"
                    )
            else:
                self._fields[field_name] = value
                self._sources[field_name] = source

    def build(self) -> DetectedArchitecture:
        """Construct final DetectedArchitecture from accumulated fields.

        Returns:
            DetectedArchitecture instance with all merged fields set
        """
        return DetectedArchitecture(**self._fields)


ROOT = Path(__file__).resolve().parent
GENERIC_PROBE_SRC = ROOT / "tests" / "probe_arch.c"
PROBE_BUILD_ROOT = Path(tempfile.gettempdir()) / "carm-arch-probes"


@dataclass(frozen=True)
class ProbeSpec:
    """Declarative configuration for an architecture detection probe."""

    name: str
    extra_compile_args: tuple[str, ...] = ()
    required_features: frozenset[str] = frozenset()
    needs_frequency_support: bool = False
    accepts_thread_count: bool = False


GENERIC_PROBE_SPECS = (
    ProbeSpec(name="features"),
    ProbeSpec(name="cache"),
    ProbeSpec(name="vlen"),
    ProbeSpec(name="frequency", needs_frequency_support=True, accepts_thread_count=True),
)
ARM_PROBE_SPECS = (
    GENERIC_PROBE_SPECS[0],
    GENERIC_PROBE_SPECS[1],
    ProbeSpec(
        name="vlen",
        extra_compile_args=("-march=armv8-a+sve",),
        required_features=frozenset({"arm_sve"}),
    ),
    GENERIC_PROBE_SPECS[3],
)


@dataclass
class TestContext:
    """Context for finding and running architecture detection tests."""

    family: str
    isa: str | None = None

    def probe_specs(self) -> tuple[ProbeSpec, ...]:
        """Return the probe declarations for this architecture family."""
        if self.family == "arm":
            return ARM_PROBE_SPECS
        return GENERIC_PROBE_SPECS

    def probe_spec(self, name: str) -> ProbeSpec:
        """Return the declared probe specification with the given name."""
        for spec in self.probe_specs():
            if spec.name == name:
                return spec
        raise ValueError(f"Probe '{name}' is not declared for architecture family '{self.family}'")

    def find_test(self, test_name: str) -> Path | None:
        """Find test with ISA > family > generic hierarchy.

        Looks for:
        1. tests/{family}/{isa}/{test_name}.c       (ISA-specific)
        2. tests/{family}/{test_name}.c             (family-generic)
        3. tests/{family}/{family}_{test_name}.c    (family-generic with prefix)
        4. tests/{test_name}.c                      (universal)

        Returns:
            First matching test path, or None if no test found
        """
        # ISA-specific test
        if self.isa:
            path = ROOT / "tests" / self.family / self.isa / f"{test_name}.c"
            if path.exists():
                return path

        # Family-specific test (exact name)
        path = ROOT / "tests" / self.family / f"{test_name}.c"
        if path.exists():
            return path

        # Family-specific test (with prefix)
        path = ROOT / "tests" / self.family / f"{self.family}_{test_name}.c"
        if path.exists():
            return path

        # Universal test
        path = ROOT / "tests" / f"{test_name}.c"
        if path.exists():
            return path

        return None


def _run_detection_test(ctx: TestContext, spec: ProbeSpec, threads: int = 1) -> dict[str, Any]:
    """Find, compile, and run a detection probe."""
    probe_path = ctx.find_test(spec.name)
    if not probe_path:
        return {}
    return run_test(probe_path, ctx, threads=threads, spec=spec)


def detect_features(ctx: TestContext) -> dict[str, Any]:
    """Run features detection test and extract ISA, arch, vendor.

    Returns:
        Dict with keys: isa (list[str]), arch (str), vendor (str).
        Returns empty dict if test not found.
    """
    raw = _run_detection_test(ctx, ctx.probe_spec("features"))
    if not raw:
        return {}

    result: dict[str, Any] = {}
    if "isa" in raw:
        result["isa"] = raw["isa"]
    if "arch" in raw:
        result["arch"] = raw["arch"]
    if "vendor" in raw:
        result["vendor"] = raw["vendor"]
    return result


def detect_cache(ctx: TestContext) -> dict[str, Any]:
    """Run or retrieve cache detection and return MemoryTopology.

    For native execution: attempts sysfs MemoryTopology first, raises if unavailable.
    For non-native (cross-compiled): returns empty dict; user must provide --topology-config.

    Returns:
        Dict with key: memory_topology (MemoryTopology), or empty dict for non-native.

    Raises:
        ValueError: If native execution and sysfs parsing fails.
    """
    from . import get_execution_interface

    exec_iface = get_execution_interface()
    is_native = exec_iface.sim_cmd is None

    if is_native:
        debug("Attempting to detect cache hierarchy from Linux /sys (native benchmark)")
        try:
            topology = MemoryTopology()
            if topology and topology.cpus:
                return {"memory_topology": topology}
            else:
                raise ValueError("MemoryTopology parsed but contains no CPU data")
        except Exception as e:
            raise UserError(
                f"Failed to parse native system memory topology from /sys: {e}. "
                "For non-native execution or when sysfs is unavailable, "
                "please provide cache hierarchy explicitly with --topology-config."
            ) from e
    else:
        # Non-native execution: user must provide cache hierarchy explicitly
        # Comment for future: could synthesize a minimal best-effort topology
        # from C cache probe caches_kib, but would require assumptions about
        # thread sharing (currently, just defer to --topology-config requirement)
        debug("Non-native execution detected; skipping cache detection (requires --topology-config)")
        return {}


def detect_vlen(ctx: TestContext) -> dict[str, Any]:
    """Run vector length detection test and extract vector_length.

    Returns:
        Dict with key: vector_length (int), or empty dict if test not found.
    """
    raw = _run_detection_test(ctx, ctx.probe_spec("vlen"))
    if not raw:
        return {}

    result: dict[str, Any] = {}
    if "vector_length" in raw:
        result["vector_length"] = raw["vector_length"]
    return result


def detect_frequency(ctx: TestContext, threads: int = 1) -> dict[str, Any]:
    """Run frequency detection test and normalize to Frequency objects.

    Converts raw hz values from probe to typed Frequency wrappers, using normalized keys.

    Returns:
        Dict with keys: frequency (Frequency), frequency_nominal (Frequency, optional).
        Returns empty dict if test not found.
    """
    raw = _run_detection_test(ctx, ctx.probe_spec("frequency"), threads=threads)
    if not raw:
        return {}

    result: dict[str, Any] = {}
    # Normalize frequency_hz (raw float) -> frequency (Frequency object)
    if "frequency_hz" in raw:
        result["frequency"] = Frequency(raw["frequency_hz"])
    # Normalize frequency_nominal_hz (raw float) -> frequency_nominal (Frequency object)
    if "frequency_nominal_hz" in raw:
        result["frequency_nominal"] = Frequency(raw["frequency_nominal_hz"])
    return result


def _ensure_test_built(src: Path, ctx: TestContext, spec: ProbeSpec) -> str:
    """Compile an architecture test into an executable if needed."""
    from . import get_execution_interface

    exec_iface = get_execution_interface()
    compile_args = ["-O2", *spec.extra_compile_args]
    if spec.needs_frequency_support:
        include_dir = ROOT / "tests" / ctx.family
        if ctx.isa:
            isa_include_dir = ROOT / "tests" / ctx.family / ctx.isa
            if (isa_include_dir / "frequency.h").exists():
                include_dir = isa_include_dir
        compile_args.insert(0, f"-I{include_dir}")
        compile_args.append("-pthread")

    cache_fields = [
        str(src.resolve()),
        ctx.family,
        ctx.isa or "generic",
        exec_iface.compiler,
        *compile_args,
    ]
    cache_key = "\0".join(cache_fields)
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:16]
    test_bin = PROBE_BUILD_ROOT / ctx.family / (ctx.isa or "generic") / f"{src.stem}-{digest}"
    test_bin.parent.mkdir(parents=True, exist_ok=True)
    src_mtime = src.stat().st_mtime
    if test_bin.exists() and test_bin.stat().st_mtime >= src_mtime:
        debug(f"Using cached architecture probe binary {test_bin}.")
        return str(test_bin)

    try:
        debug(f"Compiling architecture probe {src} to {test_bin}...")
        exec_iface.compile(str(src), str(test_bin), *compile_args, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to compile architecture probe {src}: {e}") from e

    return str(test_bin)


def run_test(src: Path, ctx: TestContext, threads: int = 1, spec: ProbeSpec | None = None) -> dict[str, Any]:
    """Run an architecture probe test and return its parsed JSON output."""
    if spec is None:
        spec = ctx.probe_spec(src.stem)
    bin_path = _ensure_test_built(src, ctx, spec)
    from . import get_execution_interface

    exec_iface = get_execution_interface()
    test_args = ["--threads", str(threads)] if spec.accepts_thread_count else []
    proc = exec_iface.run(bin_path, *test_args, check=True, capture_output=True, text=True)
    debug(f"Architecture probe output: {proc.stdout.strip()}")

    decoded = json.loads(proc.stdout)
    if not isinstance(decoded, dict):
        raise ValueError(f"Expected test JSON output to be a dict, got {type(decoded)}: {decoded}")

    return decoded


def run_generic_tests(ctx: TestContext, threads: int = 1) -> DetectedArchitecture:
    """Run the declared detection probes for an architecture."""
    builder = DetectionBuilder()
    specs = ctx.probe_specs()
    test_names_found = [spec.name for spec in specs if ctx.find_test(spec.name)]
    detail(f"Running the available auto-detection tests for {ctx.family}: {', '.join(test_names_found)}")

    features_fields = detect_features(ctx)
    if features_fields:
        builder.merge_fields("detect_features", features_fields)
    feature_set = frozenset(features_fields.get("isa", []))

    for spec in specs[1:]:
        if not spec.required_features.issubset(feature_set):
            continue
        if spec.name == "cache":
            fields = detect_cache(ctx)
        elif spec.name == "vlen":
            fields = detect_vlen(ctx)
        elif spec.name == "frequency":
            fields = detect_frequency(ctx, threads=threads)
        else:
            raise ValueError(f"Unsupported probe '{spec.name}'")
        if fields:
            builder.merge_fields(f"detect_{spec.name}", fields)

    detected = builder.build()

    from . import get_execution_interface

    if get_execution_interface().sim_cmd is None:
        from .identity import read_cpuinfo

        cpu_info = read_cpuinfo()
        if cpu_info.model_name:
            detected.model_name = cpu_info.model_name

    return detected


def native_detect(threads: int = 1) -> DetectedArchitecture:
    """Autodetects the platform's ISA information, including available features or microarchitecture details.

    Args:
        threads: Number of threads to use for frequency detection
    """
    machine = platform.machine().lower()
    if machine == "":
        raise UserError(
            "Unable to detect machine architecture (platform.machine() returned empty string). Please specify the "
            "architecture manually (--isa <arch>)."
        )

    debug(f"Detected machine architecture: {machine}")

    from carm_roofline.architecture import arm, riscv, x86

    platform_specific_fn = {
        "x86_64": x86.detect,
        "aarch64": arm.detect,
        "riscv": riscv.detect,
    }

    return platform_specific_fn[machine](threads=threads)


def detect_for_isa(isa: type[BaseISA], threads: int = 1) -> DetectedArchitecture:
    """Detect features for a specific ISA, using cross-compilation/simulation if needed.

    This works regardless of host platform by:
    1. Mapping ISA name to architecture family (x86/arm/riscv)
    2. Running that architecture's detector
    3. The detector uses ExecutionInterface for compilation and execution

    Args:
        isa: ISA class to detect for
        threads: Number of threads to use for frequency detection
    """
    if not isa.family:
        warn(f"Cannot autodetect features for ISA '{isa.name}'. Unknown ISA family.")
        return DetectedArchitecture()
    from . import arm, riscv, x86

    detectors = {
        "x86": x86.detect,
        "riscv": riscv.detect,
        "arm": arm.detect,
    }

    return detectors[isa.family](threads=threads)
