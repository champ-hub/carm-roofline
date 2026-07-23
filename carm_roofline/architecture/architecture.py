from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

from carm_roofline.arguments import InsertsArguments, positive_int
from carm_roofline.core import Frequency, UserError
from carm_roofline.isa import INCOMPATIBLE_ISAS, BaseISA
from carm_roofline.output_utils import configure_verbosity, debug, detail, format_if_container, warn

if TYPE_CHECKING:
    from carm_roofline.isa import BaseISA

from .config import load_memory_topology_from_toml
from .detect import DetectedArchitecture, detect_for_isa, native_detect
from .memory import MemoryTopologyLike


def check_isa_compatibility(selected_isas: list[type[BaseISA]]) -> None:
    """Check if the selected ISAs are compatible with each other. Raises ValueError if not."""
    # Check that all ISAs are from the same family
    families = set()
    for isa in selected_isas:
        family = isa.family
        if family:
            families.add(family)

    if len(families) > 1:
        raise UserError(f"Incompatible ISA families selected: {', '.join(families)}")

    # Check for special incompatibilities within the same family
    isa_set = set(selected_isas)
    for incompatible_pair in INCOMPATIBLE_ISAS:
        if incompatible_pair.issubset(isa_set):
            pair_names = ", ".join(isa.name for isa in incompatible_pair)
            raise UserError(f"Incompatible ISAs selected: {pair_names}")


def positive_po2_int(arg: str) -> int:
    """Parse a positive power-of-two integer."""
    value = positive_int(arg)
    if (value & (value - 1)) != 0:
        raise argparse.ArgumentTypeError(f"value is not a power of two: {arg!r}")
    return value


def get_common_field_names() -> set[str]:
    """Get field names common to both DetectedArchitecture and Architecture.

    Returns:
        Set of field names that appear in both classes, useful for comparison.
    """
    # DetectedArchitecture fields (normalized, not raw *_kib or *_hz)
    detected_fields = {
        "isa",
        "caches",
        "vector_length",
        "frequency",
        "frequency_nominal",
        "isa_frequencies",
        "arch",
        "vendor",
    }

    # Architecture fields (some differ: isa is list[type[BaseISA]] vs list[str])
    # Fields that are semantically comparable:
    arch_fields = {"vector_length", "arch"}

    # Note: caches and frequency have different structures in Architecture
    # (SimpleMemoryTopology vs list[Bytes], ISAFrequencies vs dict[str, Frequency])
    # but represent the same underlying data

    return detected_fields & arch_fields


class ISAFrequencies:
    """Per-ISA frequency storage using type-safe Frequency wrappers.

    Stores frequency values per ISA name, ensuring type consistency.
    """

    def __init__(self, isa_frequencies: dict[str, Frequency]) -> None:
        """Initialize ISA frequencies from a dict mapping ISA names to Frequency wrappers.

        Args:
            isa_frequencies: Dict mapping ISA names to Frequency.
        """
        self._isa_frequencies: dict[str, Frequency] = {}
        for isa_name, freq_value in isa_frequencies.items():
            self._isa_frequencies[isa_name] = freq_value

    @staticmethod
    def from_base_frequency(base_frequency: Frequency | Any, isas: list[type[BaseISA]]) -> ISAFrequencies:
        """Create using a single base frequency for all ISAs.

        Args:
            base_frequency: Frequency wrapper or pint Quantity to use for all ISAs
            isas: List of ISA classes to create entries for
        """
        if not isinstance(base_frequency, Frequency):
            base_frequency = Frequency(base_frequency)
        return ISAFrequencies({isa.name: base_frequency for isa in isas})

    @staticmethod
    def from_detected(detected: DetectedArchitecture, isa_list: list[type[BaseISA]]) -> ISAFrequencies:
        """Create ISAFrequencies from auto-detected architecture data.

        Initializes all ISAs to the base frequency, then overrides with any per-ISA
        specific frequencies that were detected.

        Args:
            detected: DetectedArchitecture object from probing
            isa_list: List of ISA classes for this architecture

        Returns:
            ISAFrequencies with appropriate frequencies from detected data

        Raises:
            ValueError: If no frequency information is available
        """
        base_freq = detected.frequency
        if not base_freq:
            raise ValueError("Frequency could not be auto-detected, specify it using --frequency <freq>")

        # Start with all ISAs at base frequency
        isa_freqs_dict = {isa.name: base_freq for isa in isa_list}

        # Override with any specific ISA frequencies that were detected
        if detected.isa_frequencies:
            isa_freqs_dict.update(detected.isa_frequencies)

        return ISAFrequencies(isa_freqs_dict)

    def __getitem__(self, isa_name: str) -> Frequency:
        """Get frequency for a specific ISA.

        Args:
            isa_name: Name of the ISA

        Returns:
            Frequency wrapper for that ISA

        Raises:
            KeyError: If ISA not found
        """
        return self._isa_frequencies[isa_name]

    def __repr__(self) -> str:
        return f"ISAFrequencies({self._isa_frequencies})"


def _override_warn(name: str, arg_val: Any, detected_val: Any) -> None:
    warn(f"Overriding auto-detected {name} {detected_val} with user-specified {arg_val}")


class Architecture(InsertsArguments):
    """Auto-detected and user-configured hardware architecture.

    Attributes:
        isa: List of supported ISA classes for this architecture
        memory_topology: Memory topology (either detected or CLI-provided fallback)
        frequency: Per-ISA frequency mapping using Frequency wrappers
        nominal_frequency: Optional nominal (base) frequency for the architecture
        vector_length: Vector register length for SIMD ISAs
        vector_lmul: Vector LMUL (RISC-V specific)
        set_frequency: Whether to set processor frequency
        arch: Architecture string (e.g., "x86_64", "armv8")
        vendor: CPU vendor string (e.g., "GenuineIntel", "AuthenticAMD")
        model_name: CPU model name (e.g., "AMD Ryzen 7 7735HS with Radeon Graphics")
    """

    memory_topology: MemoryTopologyLike
    frequency: ISAFrequencies
    nominal_frequency: Frequency | None
    vector_length: int | None
    vector_lmul: int | None
    set_frequency: bool | None
    arch: str | None
    vendor: str | None
    model_name: str | None
    actual_frequency_hz: int | None

    isa: list[type[BaseISA]]

    def _replace_and_warn(self, args: argparse.Namespace, detected: DetectedArchitecture) -> None:
        """Apply user args over detected values, warning on overrides."""

        def pick(name: str, arg_val: Any, detected_val: Any) -> Any:
            if arg_val is None:
                return detected_val
            if detected_val is not None and arg_val != detected_val:
                dv = format_if_container(detected_val)
                av = format_if_container(arg_val)
                _override_warn(name, av, dv)
            return arg_val

        self.vector_length: int | None = pick("vector length", args.vector_length, detected.vector_length)
        # vector_lmul is CLI-only (RISC-V specific, never auto-detected)
        self.vector_lmul: int | None = args.vector_lmul

        # Get memory topology - prefer TOML config, then detected, else error
        if args.topology_config is not None:
            # TOML config override: load memory hierarchy from file
            config_path = Path(args.topology_config)
            debug(f"Loading memory topology from TOML config: {config_path}")
            self.memory_topology = load_memory_topology_from_toml(config_path)
        elif detected.memory_topology:
            debug("Using auto-detected memory topology from sysfs")
            self.memory_topology = detected.memory_topology
        else:
            raise ValueError(
                "Memory level sizes could not be auto-detected. "
                "Specify them using a TOML config file with --topology-config <path>"
            )

        # Handle frequency: user arg (Frequency wrapper) overrides all, otherwise use detected
        if args.frequency is not None:
            # args.frequency is already a Frequency wrapper from frequency_type()
            self.frequency = ISAFrequencies.from_base_frequency(args.frequency, self.isa)
            if not args.set_frequency and (detected.isa_frequencies is not None or detected.frequency is not None):
                d_freq = detected.frequency if detected.isa_frequencies is None else detected.isa_frequencies
                _override_warn("frequency", args.frequency, d_freq)
        else:
            self.frequency = ISAFrequencies.from_detected(detected, self.isa)

        self.nominal_frequency: Frequency | None = detected.frequency_nominal

        # set_frequency is CLI-only (never auto-detected)
        self.set_frequency: bool | None = args.set_frequency
        self.arch = detected.arch
        self.vendor = detected.vendor
        self.model_name = detected.model_name
        from carm_roofline.architecture.frequency import read_single_cpu_frequency_hz

        self.actual_frequency_hz = read_single_cpu_frequency_hz()

    def __init__(self, args: argparse.Namespace):
        super().__init__()

        # Respect CLI-provided verbosity if present on the namespace.
        configure_verbosity(getattr(args, "verbose", None))

        # Extract threads for frequency detection
        num_threads = max(args.threads)

        if args.isa is None:
            detected = native_detect(threads=num_threads)
            isa_strs = detected.isa or []

            self.isa: list[type[BaseISA]] = [BaseISA.from_name(isa_str) for isa_str in isa_strs]
        else:
            self.isa = [BaseISA.from_name(isa_name) for isa_name in args.isa]
            # Verify that the selected ISAs are compatible
            check_isa_compatibility(self.isa)
            # Detect additional ISA info (VLEN, etc.) for the first ISA
            # (they all belong to the same group necessarily)
            detected = detect_for_isa(self.isa[0], threads=num_threads)

        # TODO: warn when specifying avx512 AND other x86 isa AND a
        # specific frequency (since avx512 freq is typically lower)

        # Format every dict value, then format the dict itself
        detected_str = format_if_container(
            {k: format_if_container(v) for k, v in detected.__dict__.items() if v is not None}
        )
        detail(f"Auto-detected architecture parameters:\t{detected_str}")
        self._replace_and_warn(args, detected)

    def get_frequency_for_isa(self, isa_name: str) -> Frequency:
        """Get the frequency wrapper for a specific ISA.

        Args:
            isa_name: Name of the ISA

        Returns:
            Frequency wrapper for that ISA
        """
        return self.frequency[isa_name]

    @staticmethod
    def insert_arguments(parser: argparse.ArgumentParser) -> None:
        from carm_roofline.isa import BaseISA

        parser.add_argument(
            "-i",
            "--isa",
            nargs="+",
            choices=BaseISA.names(),
            help="Set of ISAs to test. (Default: automatically detects all available ISAs)",
        )
        parser.add_argument(
            "-l",
            "--vector-length",
            type=positive_po2_int,
            help="Vector register length in bytes. (Default: automatically detected)",
        )
        parser.add_argument(
            "--vector-lmul",
            type=int,
            choices=(1, 2, 4, 8),
            help="Vector register group modifier (LMUL). RISC-V specific.",
        )
        parser.add_argument(
            "--topology-config",
            dest="topology_config",
            type=str,
            metavar="PATH",
            help="Path to TOML configuration file defining cache hierarchy. "
            "See --emit-config to generate a template. "
            "(Default: Auto-detect from sysfs)",
        )
        parser.add_argument(
            "--emit-config",
            type=str,
            metavar="PATH",
            nargs="?",
            const="topology.toml",
            help="Generate a template TOML configuration file at the specified path and exit. "
            "Edit the template and pass it back with --topology-config.",
        )
        parser.add_argument(
            "--frequency",
            type=Frequency.from_argparse,
            help="Processor frequency, e.g. '2GHz', '3200mhz' (Default: automatically detected if possible)",
        )
        parser.add_argument(
            "--set-frequency",
            action="store_true",
            help=(
                "Set the processor frequency to the value specified by --frequency "
                "(or the detected frequency if --frequency is omitted). "
                "Requires root/sudo for sysfs access. "
                "Supported on Linux systems with cpufreq drivers (x86, ARM, RISC-V)."
            ),
        )
