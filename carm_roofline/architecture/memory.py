from __future__ import annotations

import math
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol, runtime_checkable

from carm_roofline.core import Bytes, UserError
from carm_roofline.output_utils import debug, format_if_container

_CPU_ROOT = Path("/sys/devices/system/cpu")


class CacheType(Enum):
    """Cache type classification."""

    DATA = "Data"
    INSTRUCTION = "Instruction"
    UNIFIED = "Unified"


@dataclass
class CacheDescriptor:
    """Represents one cache index entry from sysfs."""

    level: int  # 1, 2, 3, …
    cache_type: CacheType
    size: Bytes
    ways: int
    sets: int
    line_size: Bytes
    shared_cpus: frozenset[int]  # logical CPUs sharing this instance

    @property
    def is_data_cache(self) -> bool:
        return self.cache_type in (CacheType.DATA, CacheType.UNIFIED)

    def __repr__(self) -> str:
        return f"L{self.level}{self.cache_type.value[0]}({self.size}, shared={sorted(self.shared_cpus)})"


@dataclass
class LogicalCPU:
    """A single hardware thread (logical CPU)."""

    cpu_id: int
    package_id: int
    core_id: int  # physical core within package
    thread_siblings: frozenset[int]  # SMT siblings (same physical core)
    core_siblings: frozenset[int]  # all CPUs in same package
    caches: list[CacheDescriptor] = field(default_factory=list)

    @property
    def global_core_key(self) -> tuple[int, int]:
        """(package_id, core_id): uniquely identifies a physical core."""
        return (self.package_id, self.core_id)

    def data_caches(self, level: int | None = None) -> list[CacheDescriptor]:
        return [c for c in self.caches if c.is_data_cache and (level is None or c.level == level)]


@dataclass(frozen=True)
class MemoryLevelInfo:
    """Represents a single level in the memory hierarchy."""

    size: Bytes
    name: str
    num_sharing_threads: int
    instances: int


@dataclass
class CacheAwareThreadAffinity:
    """Represents a thread affinity plan based on cache topology.

    cache_bytes_per_level maps each numeric hierarchy level (cache levels and DRAM) to the total bytes
    accessible to the selected threads at that level.  This covers all levels in the hierarchy,
    so callers can detect unexpected caching: if size_per_thread > cache_bytes_per_level[L-1]
    the working set will spill into level L.
    """

    cache_level: int
    "Cache level this affinity targets (1, 2, 3, ...)"
    cpu_ids: list[int]
    "CPU IDs to bind threads to, optimized for cache coverage"
    cache_bytes_per_level: dict[int, Bytes]
    "Total cache bytes accessible to the thread group at each cache level"

    @property
    def total_cache_bytes(self) -> Bytes:
        """Total cache bytes at the target cache level."""
        return self.cache_bytes_per_level[self.cache_level]

    @property
    def num_threads(self) -> int:
        return len(self.cpu_ids)


@runtime_checkable
class MemoryTopologyLike(Protocol):
    """Common interface for memory topology implementations.

    Both MemoryTopology (detected from sysfs) and SimpleMemoryTopology (CLI fallback)
    implement this protocol, enabling unified handling throughout the codebase.
    """

    def __iter__(self) -> Iterator[MemoryLevelInfo]:
        """Iterate over MemoryLevelInfo objects (all levels in hierarchy order)."""
        ...

    def available_cache_levels(self) -> list[int]:
        """Get list of numeric hierarchy levels present (e.g., [1, 2, 3, 4])."""
        ...

    def plan_thread_affinity(
        self,
        n_threads: int,
        cache_level: int,
        prefer_no_smt: bool = True,
    ) -> CacheAwareThreadAffinity:
        """Plan optimal CPU affinity for n_threads targeting a specific cache level.

        Args:
            n_threads: number of threads to place
            cache_level: target cache level (1, 2, 3, …)
            prefer_no_smt: if True, prefer placing threads on separate physical cores

        Returns:
            CacheAwareThreadAffinity with cpu_ids and cache_bytes_per_level for all
            cache levels, so callers can detect if the working set fits in the target
            level or will spill into a higher one.
        """
        ...


class SimpleMemoryTopology:
    """Simple memory hierarchy from CLI args - fallback when detailed topology unavailable.

    Represents an ordered memory hierarchy (L1, L2, L3, DRAM, etc.) with thread sharing info,
    but without detailed CPU topology. Used when cache hierarchy is manually specified via
    command-line arguments rather than auto-detected from sysfs.
    """

    def __init__(
        self,
        level_sizes: list[Bytes],
        instances_per_level: list[int],
        total_cpus: int | None = None,
        smt_degree: int = 1,
        cpu_offset: int = 0,
    ) -> None:
        """Initialize memory hierarchy from sizes and cache instance counts.

        The hierarchy levels are numbered 1..N in the order supplied.  The last
        entry is treated as DRAM; all preceding entries are named L1, L2, …

        Args:
            level_sizes: Cache size per level instance, in hierarchy order (closest first).
            instances_per_level: Number of cache instances for each level.
            total_cpus: Total logical CPUs available (optional, for better synthetic affinity).
            smt_degree: Logical CPUs per physical core (>=1).
            cpu_offset: Base CPU index for generated synthetic CPU IDs.
        """

        if len(level_sizes) != len(instances_per_level):
            raise UserError(
                "Level sizes and instances_per_level must have same length: "
                f"{len(level_sizes)} vs {len(instances_per_level)}"
            )
        if not level_sizes:
            raise UserError("At least one cache level must be provided")

        if any(inst <= 0 for inst in instances_per_level):
            raise UserError("All values in instances_per_level must be positive integers")

        if total_cpus is not None and total_cpus <= 0:
            raise UserError(f"total_cpus must be positive when provided, got {total_cpus}")

        if cpu_offset < 0:
            raise UserError(f"cpu_offset must be >= 0, got {cpu_offset}")

        if total_cpus is not None:
            for idx, instances in enumerate(instances_per_level, 1):
                if instances > total_cpus:
                    raise UserError(
                        f"Cache level {idx}: instances ({instances}) cannot exceed total_cpus ({total_cpus})"
                    )

        self._total_cpus = total_cpus
        self._smt_degree = max(1, smt_degree)
        self._cpu_offset = cpu_offset

        # Names are assigned once here: the last entry is DRAM; the rest are L1, L2, …
        n = len(level_sizes)
        self._levels: list[MemoryLevelInfo] = []
        for i, (size, instances) in enumerate(zip(level_sizes, instances_per_level)):
            name = "DRAM" if i == n - 1 else f"L{i + 1}"
            num_sharing_threads = max(1, math.ceil(total_cpus / instances)) if total_cpus is not None else 1
            self._levels.append(
                MemoryLevelInfo(
                    size=size,
                    name=name,
                    num_sharing_threads=num_sharing_threads,
                    instances=instances,
                )
            )

    def __iter__(self) -> Iterator[MemoryLevelInfo]:
        """Iterate over MemoryLevelInfo objects in ascending level-number order."""
        return iter(self._levels)

    def available_cache_levels(self) -> list[int]:
        """Get list of hierarchy level numbers in ascending order (e.g., [1, 2, 3, 4])."""
        return list(range(1, len(self._levels) + 1))

    def plan_thread_affinity(
        self,
        n_threads: int,
        cache_level: int,
        prefer_no_smt: bool = True,
    ) -> CacheAwareThreadAffinity:
        """Plan thread affinity for simple topology without detailed CPU information.

        Since SimpleMemoryTopology lacks actual CPU topology data, this returns a
        synthetic affinity plan that approximates no-SMT placement when possible.

        Args:
            n_threads: number of threads to place
            cache_level: target cache level (1, 2, 3, …)
            prefer_no_smt: ignored (no topology to optimize SMT placement)

        Returns:
            CacheAwareThreadAffinity with synthetic CPU assignment and
            cache_bytes_per_level populated for all numeric cache levels.

        Raises:
            ValueError: if cache_level is not present in the hierarchy
        """
        available = self.available_cache_levels()
        if cache_level not in available:
            raise ValueError(f"Cache level {cache_level} not found. Available cache levels: {available}")

        # Synthetic CPU assignment: prefer one thread per core first if SMT is known.
        n_selected = min(n_threads, self._total_cpus) if self._total_cpus is not None else n_threads

        if prefer_no_smt and self._smt_degree > 1 and self._total_cpus is not None:
            n_cores = math.ceil(self._total_cpus / self._smt_degree)
            primary = [self._cpu_offset + core * self._smt_degree for core in range(n_cores)]
            secondary: list[int] = []
            for sibling in range(1, self._smt_degree):
                secondary.extend(self._cpu_offset + core * self._smt_degree + sibling for core in range(n_cores))
            candidate_ids = [cpu_id for cpu_id in primary + secondary if cpu_id < self._cpu_offset + self._total_cpus]
            cpu_ids = candidate_ids[:n_selected]
        else:
            cpu_ids = list(range(self._cpu_offset, self._cpu_offset + n_selected))

        # Compute bytes accessible at each level for the selected thread group.
        # Covered instances = min(threads selected, instances at that level).
        cache_bytes_per_level: dict[int, Bytes] = {}
        for level_num, lvl in enumerate(self._levels, 1):
            covered_instances = min(len(cpu_ids), lvl.instances)
            cache_bytes_per_level[level_num] = lvl.size * covered_instances

        debug(f"Planned simple thread affinity for {n_threads} threads targeting L{cache_level}:")
        debug(f"CPU IDs: {cpu_ids}")
        debug(f"Cache bytes per level: {cache_bytes_per_level}")

        return CacheAwareThreadAffinity(
            cache_level=cache_level,
            cpu_ids=cpu_ids,
            cache_bytes_per_level=cache_bytes_per_level,
        )

    def __repr__(self) -> str:
        parts = [f"{lvl.name}={lvl.size}x{lvl.instances}" for lvl in self._levels]
        return f"SimpleMemoryTopology({', '.join(parts)})"


def _read(path: Path) -> str:
    """Read a sysfs file, return stripped string or '' on error."""
    return path.read_text().strip()


def _parse_cpu_list(s: str) -> list[int]:
    """'0-3,6,8-10' -> [0,1,2,3,6,8,9,10]"""
    cpus: list[int] = []
    for part in s.split(","):
        stripped_part = part.strip()
        if not stripped_part:
            continue
        m = re.fullmatch(r"(\d+)-(\d+)", stripped_part)
        if m:
            cpus.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            cpus.append(int(stripped_part))
    return sorted(set(cpus))


# Can't use Bytes here as the suffix is different (e.g., "32K" for 32 KiB), so we parse manually.
def _parse_size_kb(s: str) -> int:
    """'32K' -> 32768, '4096K' -> 4194304, '8M' -> 8388608, '512' -> 512"""
    s = s.strip().upper()
    m = re.fullmatch(r"(\d+)\s*([KMG]?)", s)
    if not m:
        return 0
    val = int(m.group(1))
    suffix = m.group(2)
    return val * {"K": 1024, "M": 1024**2, "G": 1024**3}.get(suffix, 1)


class MemoryTopology:
    "Parses /sys/devices/system/cpu/ and exposes the full topology."

    def __init__(self, sysfs_root: Path = _CPU_ROOT):
        self._root = sysfs_root
        self.cpus: dict[int, LogicalCPU] = {}
        "cpu_id -> LogicalCPU"
        self.packages: dict[int, list[int]] = {}
        "package_id -> list of cpu_ids in that package"
        self.cache_instances: dict[int, list[frozenset[int]]] = {}
        "hierarchy_level -> list of instances (each instance is a set of cpu_ids sharing that level)"
        self.level_sizes: dict[int, Bytes] = {}
        "hierarchy_level -> size per instance for that level"
        self._levels: list[MemoryLevelInfo] = []  # pre-built by _parse(); names are stable after construction
        self._parse()

    @staticmethod
    def _detect_dram_size() -> Bytes:
        """Detect total DRAM bytes on Linux with robust fallbacks."""
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            try:
                for line in meminfo.read_text().splitlines():
                    if line.startswith("MemTotal:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            return Bytes(int(parts[1]) * 1024)
            except (OSError, ValueError):
                pass

        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            phys_pages = int(os.sysconf("SC_PHYS_PAGES"))
            if page_size > 0 and phys_pages > 0:
                return Bytes(page_size * phys_pages)
        except (OSError, ValueError):
            pass

        return Bytes(0)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _online_cpu_ids(self) -> list[int]:
        online_path = self._root / "online"
        if online_path.exists():
            return _parse_cpu_list(_read(online_path))
        # Fallback: scan directories
        ids = []
        for p in sorted(self._root.glob("cpu[0-9]*")):
            m = re.fullmatch(r"cpu(\d+)", p.name)
            if m:
                ids.append(int(m.group(1)))
        return sorted(ids)

    def _parse_cpu_topology(self, cpu_id: int) -> LogicalCPU | None:
        topo_dir = self._root / f"cpu{cpu_id}" / "topology"
        if not topo_dir.exists():
            # cpu0 sometimes lacks topology on single-core systems
            package_id = 0
            core_id = 0
            thread_siblings: frozenset[int] = frozenset([cpu_id])
            core_siblings: frozenset[int] = frozenset([cpu_id])
        else:
            package_id = int(_read(topo_dir / "physical_package_id") or "0")
            core_id = int(_read(topo_dir / "core_id") or "0")
            thread_siblings = frozenset(_parse_cpu_list(_read(topo_dir / "thread_siblings_list") or str(cpu_id)))
            core_siblings = frozenset(_parse_cpu_list(_read(topo_dir / "core_siblings_list") or str(cpu_id)))
        return LogicalCPU(
            cpu_id=cpu_id,
            package_id=package_id,
            core_id=core_id,
            thread_siblings=thread_siblings,
            core_siblings=core_siblings,
        )

    def _parse_cpu_caches(self, cpu_id: int) -> list[CacheDescriptor]:
        cache_dir = self._root / f"cpu{cpu_id}" / "cache"
        if not cache_dir.exists():
            return []
        descriptors: list[CacheDescriptor] = []
        for idx_dir in sorted(cache_dir.glob("index*")):
            level = int(_read(idx_dir / "level") or "0")
            cache_type = _read(idx_dir / "type") or "Unknown"
            size_str = _read(idx_dir / "size") or "0K"
            ways = int(_read(idx_dir / "ways_of_associativity") or "0")
            sets = int(_read(idx_dir / "number_of_sets") or "0")
            line_size = int(_read(idx_dir / "coherency_line_size") or "64")
            shared = frozenset(_parse_cpu_list(_read(idx_dir / "shared_cpu_list") or str(cpu_id)))
            descriptors.append(
                CacheDescriptor(
                    level=level,
                    cache_type=CacheType(cache_type),
                    size=Bytes(_parse_size_kb(size_str)),
                    ways=ways,
                    sets=sets,
                    line_size=Bytes(line_size),
                    shared_cpus=shared,
                )
            )
        return descriptors

    def _parse(self) -> None:
        online_ids = self._online_cpu_ids()
        for cpu_id in online_ids:
            logical_cpu = self._parse_cpu_topology(cpu_id)
            if logical_cpu is None:
                continue
            logical_cpu.caches = self._parse_cpu_caches(cpu_id)
            self.cpus[cpu_id] = logical_cpu
            self.packages.setdefault(logical_cpu.package_id, []).append(cpu_id)

        # Build cache instance registry (data caches only)
        seen_sets: dict[int, set[frozenset[int]]] = {}  # level -> set of frozensets
        level_sizes: dict[int, Bytes] = {}
        for cpu in self.cpus.values():
            for c in cpu.data_caches():
                seen_sets.setdefault(c.level, set())
                seen_sets[c.level].add(c.shared_cpus)
                level_sizes.setdefault(c.level, c.size)
        self.cache_instances = {lvl: sorted(instances, key=min) for lvl, instances in seen_sets.items()}

        if self.cpus:
            dram_level = (max(self.cache_instances.keys()) + 1) if self.cache_instances else 1
            self.cache_instances[dram_level] = [frozenset(self.cpus.keys())]
            level_sizes[dram_level] = self._detect_dram_size()

        self.level_sizes = level_sizes

        # Pre-build MemoryLevelInfo list with names assigned once.
        # dram_level is the numerically largest level (inserted as max+1 above).
        if self.cpus:
            all_levels = sorted(self.cache_instances.keys())
            dram_lvl = all_levels[-1]
            for level in all_levels:
                instances = self.cache_instances[level]
                if not instances:
                    continue
                first_instance_cpus = instances[0]
                num_sharing_threads = len(first_instance_cpus)
                level_size = self.level_sizes.get(level, Bytes(0))
                name = "DRAM" if level == dram_lvl else f"L{level}"
                self._levels.append(
                    MemoryLevelInfo(
                        size=level_size,
                        name=name,
                        num_sharing_threads=num_sharing_threads,
                        instances=len(instances),
                    )
                )

    def available_cache_levels(self) -> list[int]:
        return sorted(self.cache_instances.keys())

    def num_physical_cores(self) -> int:
        return len({cpu.global_core_key for cpu in self.cpus.values()})

    def num_packages(self) -> int:
        return len(self.packages)

    def plan_thread_affinity(
        self,
        n_threads: int,
        cache_level: int,
        prefer_no_smt: bool = True,
    ) -> CacheAwareThreadAffinity:
        """
        Compute the optimal CPU affinity list for *n_threads* threads such that the number of distinct L*cache_level*
        cache instancesthey collectively use is maximised (i.e. threads are spread across cache domains as widely as
        possible).

        Args:
            n_threads: number of threads to place
            cache_level: target cache level (1, 2, 3, …)
            prefer_no_smt: if True, prefer placing threads on separate physical cores before using SMT siblings
        Returns:
            CacheAwareThreadAffinity object describing the thread affinity plan.
        Raises:
            ValueError: if cache_level is not present in the topology
        """
        if cache_level not in self.cache_instances:
            available = self.available_cache_levels()
            raise ValueError(f"Cache level {cache_level} not found. Available data-cache levels: {available}")

        instances = self.cache_instances[cache_level]
        n_instances = len(instances)

        # For each cache instance build an ordered list of candidate CPUs.
        # Order: physical cores first (one thread per core), then SMT threads.
        def _ordered_candidates(cpu_set: frozenset[int]) -> list[int]:
            seen_cores: set[tuple[int, int]] = set()
            primary: list[int] = []  # first HW-thread of each core
            secondary: list[int] = []  # SMT siblings
            for cpu_id in sorted(cpu_set):
                cpu = self.cpus.get(cpu_id)
                if cpu is None:
                    continue
                core_key = cpu.global_core_key
                if core_key not in seen_cores:
                    seen_cores.add(core_key)
                    primary.append(cpu_id)
                else:
                    secondary.append(cpu_id)
            return primary + secondary if prefer_no_smt else sorted(cpu_set)

        candidates_per_instance = [_ordered_candidates(inst) for inst in instances]

        # Round-robin across cache instances to maximise instance coverage.
        # Each round picks one CPU from each instance (in order).
        selected: list[int] = []
        round_idx = 0
        while len(selected) < n_threads:
            added_this_round = 0
            for inst_candidates in candidates_per_instance:
                if len(selected) >= n_threads:
                    break
                if round_idx < len(inst_candidates):
                    selected.append(inst_candidates[round_idx])
                    added_this_round += 1
            if added_this_round == 0:
                # All candidates exhausted, more threads than logical CPUs
                break
            round_idx += 1

        selected = sorted(selected)  # sort final list for easier reading

        # Compute cache_bytes_per_level for all hierarchy levels using selected CPUs.
        # For each level, count the distinct instances touched by selected threads.
        cache_bytes_per_level: dict[int, Bytes] = {}
        hit_instances_at_target = 0
        for lvl in self.available_cache_levels():
            seen_sets_lvl: set[frozenset[int]] = set()
            lvl_bytes = Bytes(0)
            for cpu_id in selected:
                for instance in self.cache_instances[lvl]:
                    if cpu_id in instance and instance not in seen_sets_lvl:
                        seen_sets_lvl.add(instance)
                        lvl_bytes = lvl_bytes + self.level_sizes[lvl]
            cache_bytes_per_level[lvl] = lvl_bytes
            if lvl == cache_level:
                hit_instances_at_target = len(seen_sets_lvl)

        debug(f"Calculated the following thread affinity for {n_threads} threads targeting L{cache_level} cache:")
        debug(f"Selected CPUs: {selected}")
        debug(f"Hit instances: {hit_instances_at_target} out of {n_instances}")
        debug(f"Cache bytes per level: {format_if_container(cache_bytes_per_level)}")

        return CacheAwareThreadAffinity(
            cache_level=cache_level, cpu_ids=selected, cache_bytes_per_level=cache_bytes_per_level
        )

    def __iter__(self) -> Iterator[MemoryLevelInfo]:
        """Iterate over MemoryLevelInfo objects for each hierarchy level in ascending order."""
        return iter(self._levels)

    def __repr__(self) -> str:
        return (
            f"MemoryTopology(packages={self.num_packages()}, "
            f"cores={self.num_physical_cores()}, "
            f"logical_cpus={len(self.cpus)})"
        )

    def __str__(self) -> str:
        return (
            f"MemoryTopology with {self.num_packages()} package(s), "
            f"{self.num_physical_cores()} physical core(s), "
            f"{len(self.cpus)} logical CPU(s): "
            + ", ".join(f"{lvl.name}=({lvl.size}, {lvl.instances} instances)" for lvl in self._levels)
        )
