"""Profile data model: hierarchical representation of profiled application results.

The model mirrors the structure produced by PAPI HL output files:

    Run (metadata + list of Ranks)
    ├── Rank 0 (rank_id, event_definitions, list of Threads)
    │   ├── Thread 0 (thread_id, list of Regions)
    │   │   ├── Region "daxpy" (name, cycles, time_nsec, raw counters)
    │   │   └── ...
    │   ├── Thread 1
    │   └── ...
    ├── Rank 1
    └── ...

Raw PAPI event counters live on each region.  Metric computation (flops, bytes)
happens at aggregation time using the appropriate ``MetricDefinition``, keeping
the model generic and backend-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RegionMetrics:
    """Raw performance counters for one annotated code region.

    Attributes:
        name: Region name (set by the application via ``PAPI_hl_region_begin``).
        parent_region_id: ID of the enclosing region (``"-1"`` for top-level).
        cycles: Hardware cycle count for this region.
        time_nsec: Wall-clock time in nanoseconds for this region.
        counters: Raw PAPI event counter values keyed by event name (e.g. ``PAPI_FP_OPS``).
    """

    name: str
    parent_region_id: str
    cycles: int
    time_nsec: int
    counters: dict[str, int]


@dataclass
class ThreadMetrics:
    """PAPI HL output for a single thread within an MPI rank.

    Attributes:
        thread_id: Thread index within the rank.
        regions: List of annotated code regions measured on this thread.
    """

    thread_id: int
    regions: list[RegionMetrics]


@dataclass
class RankMetrics:
    """PAPI HL output for a single MPI rank.

    Attributes:
        rank_id: MPI rank index.
        event_definitions: Metadata about collected events (from JSON ``event_definitions`` key).
        threads: List of thread measurements within this rank.
    """

    rank_id: int
    event_definitions: dict[str, Any] = field(default_factory=dict)
    threads: list[ThreadMetrics] = field(default_factory=list)


@dataclass
class RunMetadata:
    """Metadata describing the profiled run."""

    name: str = ""
    date: str = ""
    method: str = "PAPI_HL"
    isa: str = ""
    precision: str = ""
    threads_per_rank: int = 1
    command: str = ""
    notes: str = ""


@dataclass
class RunResults:
    """Complete profiling results for one execution.

    Contains the full rank/thread/region hierarchy plus metadata.
    """

    metadata: RunMetadata = field(default_factory=RunMetadata)
    ranks: list[RankMetrics] = field(default_factory=list)

    @property
    def num_ranks(self) -> int:
        return len(self.ranks)

    @property
    def total_threads(self) -> int:
        return sum(len(r.threads) for r in self.ranks)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "metadata": {
                "name": self.metadata.name,
                "date": self.metadata.date,
                "method": self.metadata.method,
                "isa": self.metadata.isa,
                "precision": self.metadata.precision,
                "threads_per_rank": self.metadata.threads_per_rank,
                "command": self.metadata.command,
                "notes": self.metadata.notes,
                "num_ranks": self.num_ranks,
                "total_threads": self.total_threads,
            },
            "ranks": [
                {
                    "rank_id": r.rank_id,
                    "num_threads": len(r.threads),
                    "threads": [
                        {
                            "thread_id": t.thread_id,
                            "regions": [
                                {
                                    "name": reg.name,
                                    "parent_region_id": reg.parent_region_id,
                                    "cycles": reg.cycles,
                                    "time_nsec": reg.time_nsec,
                                    "counters": dict(reg.counters),
                                }
                                for reg in t.regions
                            ],
                        }
                        for t in r.threads
                    ],
                }
                for r in self.ranks
            ],
        }
