from __future__ import annotations

from errno import EIO
from types import SimpleNamespace
import sys

import pytest

import carm_roofline.architecture as architecture
import carm_roofline.architecture.detect as detection
import carm_roofline.architecture.identity as identity
import carm_roofline.architecture.memory as memory
from carm_roofline.architecture.darwin import sysctl_int, sysctl_string
from carm_roofline.architecture.detect import TestContext as ProbeContext
from carm_roofline.architecture.memory import DarwinMemoryTopology, MemoryTopology
from carm_roofline.core import Bytes

pytestmark = pytest.mark.unit


def _apple_sysctls() -> dict[str, int]:
    return {
        "hw.nperflevels": 2,
        "hw.ncpu": 8,
        "hw.memsize": 32 * 1024**3,
        "hw.perflevel0.physicalcpu": 4,
        "hw.perflevel0.logicalcpu": 4,
        "hw.perflevel0.l1dcachesize": 128 * 1024,
        "hw.perflevel0.l2cachesize": 12 * 1024**2,
        "hw.perflevel0.cpusperl2": 4,
        "hw.perflevel0.l3cachesize": 0,
        "hw.perflevel1.physicalcpu": 4,
        "hw.perflevel1.logicalcpu": 4,
        "hw.perflevel1.l1dcachesize": 64 * 1024,
        "hw.perflevel1.l2cachesize": 4 * 1024**2,
        "hw.perflevel1.cpusperl2": 4,
        "hw.perflevel1.l3cachesize": 0,
    }


def _patch_sysctls(monkeypatch: pytest.MonkeyPatch, values: dict[str, int]) -> None:
    def read(name: str) -> int | None:
        return values.get(name)

    monkeypatch.setattr(memory, "sysctl_int_or_none", read)


def _patch_native_interface(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(architecture, "get_execution_interface", lambda: SimpleNamespace(sim_cmd=None))



def test_darwin_arm_feature_detection_uses_sysctl_not_arm_system_registers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {
        "hw.optional.neon": 1,
        "hw.optional.arm.FEAT_SVE": 0,
    }
    monkeypatch.setattr(detection.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(detection.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(detection, "sysctl_int_or_none", lambda name: values.get(name))
    _patch_native_interface(monkeypatch)

    assert detection.detect_features(ProbeContext(family="arm")) == {
        "isa": ["arm", "arm_neon"],
        "arch": "arm64",
    }


def test_darwin_feature_detection_propagates_unexpected_sysctl_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(detection.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(detection.platform, "machine", lambda: "arm64")
    _patch_native_interface(monkeypatch)

    def fail(name: str) -> int | None:
        raise OSError(EIO, "sysctl failed")

    monkeypatch.setattr(detection, "sysctl_int_or_none", fail)

    with pytest.raises(OSError, match="sysctl failed"):
        detection.detect_features(ProbeContext(family="arm"))


def test_native_apple_invalid_topology_warns_and_allows_explicit_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(detection, "warn", warnings.append)
    monkeypatch.setattr(architecture, "get_execution_interface", lambda: SimpleNamespace(sim_cmd=None))
    monkeypatch.setattr(detection.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(detection.platform, "machine", lambda: "arm64")

    def unavailable() -> MemoryTopology:
        raise ValueError("required Apple sysctl is unavailable")

    monkeypatch.setattr(detection, "native_memory_topology", unavailable)

    assert detection.detect_cache(ProbeContext(family="arm")) == {}
    assert any("--topology-config <path>" in warning for warning in warnings)


def test_native_apple_unexpected_topology_sysctl_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(architecture, "get_execution_interface", lambda: SimpleNamespace(sim_cmd=None))
    monkeypatch.setattr(detection.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(detection.platform, "machine", lambda: "arm64")

    def fail() -> MemoryTopology:
        raise OSError(EIO, "sysctl I/O failure")

    monkeypatch.setattr(detection, "native_memory_topology", fail)

    with pytest.raises(OSError, match="sysctl I/O failure"):
        detection.detect_cache(ProbeContext(family="arm"))


def test_darwin_topology_uses_cache_sharing_and_conservative_heterogeneous_sizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_sysctls(monkeypatch, _apple_sysctls())

    topology = DarwinMemoryTopology()

    assert topology.cpu_ids_are_synthetic
    assert topology.available_cache_levels() == [1, 2, 3]
    assert topology.level_sizes[1] == Bytes(64 * 1024)
    assert topology.level_sizes[2] == Bytes(4 * 1024**2)
    assert topology.level_sizes[3] == Bytes(32 * 1024**3)
    assert topology.cache_instances[1] == [frozenset({cpu}) for cpu in range(8)]
    assert topology.cache_instances[2] == [frozenset(range(4)), frozenset(range(4, 8))]

    affinity = topology.plan_thread_affinity(n_threads=2, cache_level=2)
    assert affinity.cpu_ids == [0, 4]
    assert affinity.cache_bytes_per_level[1] == Bytes(64 * 1024)
    assert affinity.cache_bytes_per_level[2] == Bytes(4 * 1024**2)
    assert [level.size for level in topology] == [
        Bytes(64 * 1024),
        Bytes(4 * 1024**2),
        Bytes(32 * 1024**3),
    ]


def test_darwin_topology_models_l3_sharing_per_performance_level(monkeypatch: pytest.MonkeyPatch) -> None:
    values = _apple_sysctls()
    values.update(
        {
            "hw.perflevel0.l3cachesize": 16 * 1024**2,
            "hw.perflevel0.cpusperl3": 4,
            "hw.perflevel1.l3cachesize": 4 * 1024**2,
            "hw.perflevel1.cpusperl3": 4,
        }
    )
    _patch_sysctls(monkeypatch, values)

    topology = DarwinMemoryTopology()
    affinity = topology.plan_thread_affinity(n_threads=2, cache_level=3)

    assert topology.cache_instances[3] == [frozenset(range(4)), frozenset(range(4, 8))]
    assert topology.level_sizes[3] == Bytes(4 * 1024**2)
    assert affinity.cache_bytes_per_level[3] == Bytes(4 * 1024**2)


def test_darwin_topology_rejects_unknown_l2_sharing(monkeypatch: pytest.MonkeyPatch) -> None:
    values = _apple_sysctls()
    values["hw.perflevel0.physicalcpu"] = 3
    values["hw.perflevel0.logicalcpu"] = 3
    values["hw.perflevel0.cpusperl2"] = 2
    values["hw.ncpu"] = 7
    _patch_sysctls(monkeypatch, values)

    with pytest.raises(ValueError, match="cannot derive complete cache-sharing groups"):
        DarwinMemoryTopology()


def test_darwin_topology_rejects_inconsistent_cpu_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    values = _apple_sysctls()
    values["hw.ncpu"] = 10
    _patch_sysctls(monkeypatch, values)

    with pytest.raises(ValueError, match="hw.ncpu"):
        DarwinMemoryTopology()


def test_native_topology_uses_apple_sysctls_only_on_apple_arm(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_sysctls(monkeypatch, _apple_sysctls())
    monkeypatch.setattr(memory.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(memory.platform, "machine", lambda: "arm64")

    assert isinstance(memory.native_memory_topology(), DarwinMemoryTopology)

    monkeypatch.setattr(memory.platform, "system", lambda: "Linux")
    monkeypatch.setattr(memory.platform, "machine", lambda: "aarch64")
    assert type(memory.native_memory_topology()) is MemoryTopology


def test_darwin_identity_uses_sysctl_cpu_brand(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(identity.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(identity, "sysctl_string", lambda name: "Apple M3 Pro")

    info = identity.read_cpuinfo()

    assert info.model_name == "Apple M3 Pro"
    assert info.vendor == "Apple"


def test_sysctl_value_decoding(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "hw.test.count": (23).to_bytes(4, byteorder=sys.byteorder),
        "hw.test.name": b"Apple M3\0",
    }
    monkeypatch.setattr("carm_roofline.architecture.darwin._sysctl_bytes", lambda name: values[name])

    assert sysctl_int("hw.test.count") == 23
    assert sysctl_string("hw.test.name") == "Apple M3"
