"""Counter registry for the Paraver pipeline.

Each shipped Paraver counter config differed only in its window name, time unit,
and the ``evt_type`` filter line, so they collapse into one template
(:data:`TEMPLATE_FILENAME`) rendered per counter. This module owns the per-counter
substitution values — the Paraver event ids and the FLOP/bytes weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

TEMPLATE_FILENAME = "counter_template.cfg"


@dataclass(frozen=True)
class CounterSpec:
    """Static description of one hardware counter.

    Attributes:
        name: Counter name — CSV stem, rendered config filename, and ``evt_type_label``.
        evt_type: Paraver event type id for the hardware counter (from the .pcf / cfgs).
        flops_multiplier: Retired instructions to FLOPs (0 for memory counters).
        bytes_per_inst: Bytes per retired FP instruction for the weighted
            ``bytes_modifier`` (0.0 for memory counters).
        is_memory: Counts toward ``mem_ops`` (loads/stores).
    """

    name: str
    evt_type: int
    flops_multiplier: int
    bytes_per_inst: float
    is_memory: bool


INTEL_COUNTERS: tuple[CounterSpec, ...] = (
    CounterSpec("fp-scalar-dp", 44548973, 1, 8.0, False),
    CounterSpec("fp-scalar-sp", 42001053, 1, 4.0, False),
    CounterSpec("fp-sse-dp", 44561246, 2, 16.0, False),
    CounterSpec("fp-sse-sp", 42001056, 4, 16.0, False),
    CounterSpec("fp-avx2-dp", 44995982, 4, 32.0, False),
    CounterSpec("fp-avx2-sp", 42001055, 8, 32.0, False),
    CounterSpec("fp-avx512-dp", 44021956, 8, 64.0, False),
    CounterSpec("fp-avx512-sp", 42001054, 16, 64.0, False),
    CounterSpec("mem-loads", 44723342, 0, 0.0, True),
    CounterSpec("mem-stores", 44604811, 0, 0.0, True),
)


def configs_dir() -> Path:
    """Directory holding the shipped Paraver config template."""
    return Path(__file__).parent / "configs"


def counter_config_template() -> Path:
    """Path of the single counter config template (rendered per counter)."""
    return configs_dir() / TEMPLATE_FILENAME


# Derived accessors, aligned to the FP counter subset of INTEL_COUNTERS.
fp_names: tuple[str, ...] = tuple(spec.name for spec in INTEL_COUNTERS if not spec.is_memory)
memory_names: tuple[str, ...] = tuple(spec.name for spec in INTEL_COUNTERS if spec.is_memory)
flops_weights: tuple[int, ...] = tuple(spec.flops_multiplier for spec in INTEL_COUNTERS if not spec.is_memory)
bytes_weights: tuple[float, ...] = tuple(spec.bytes_per_inst for spec in INTEL_COUNTERS if not spec.is_memory)


# ISA grouping of the FP counters, derived from INTEL_COUNTERS (each FP counter
# name is 'fp-<isa>-<precision>'; order = first appearance in INTEL_COUNTERS:
# scalar, sse, avx2, avx512). Feeds the per-ISA operation percentages.
def _fp_isa_of(name: str) -> str:
    """ISA group of one FP counter name ('fp-avx2-dp' → 'avx2')."""
    return name.split("-", 2)[1]


fp_isa: tuple[str, ...] = tuple(_fp_isa_of(spec.name) for spec in INTEL_COUNTERS if not spec.is_memory)
isa_names: tuple[str, ...] = tuple(dict.fromkeys(fp_isa))
