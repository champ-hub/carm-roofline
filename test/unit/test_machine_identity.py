"""Unit tests for machine identity, signature hashing, and run-name generation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from architecture.identity import (
    CpuInfo,
    MachineSignature,
    MemoryLevelSignature,
    _get_physical_ram_bytes,
    _levels_from_topology,
    _short_model_name,
    detect_machine_signature,
    generate_run_name,
    signature_from_architecture,
    write_machine_json,
)
from architecture.memory import MemoryLevelInfo
from units import Bytes

pytestmark = pytest.mark.unit


def _make_signature(
    model_name: str = "AMD Ryzen 7 7735HS with Radeon Graphics",
    dram_bytes: int = 17_179_869_184,
    arch: str = "x86_64",
    vendor: str = "AuthenticAMD",
) -> MachineSignature:
    """Build a signature with a fixed cache hierarchy and a configurable DRAM size."""
    memory_levels = (
        MemoryLevelSignature(name="L1", size_bytes=32768, instances=8, num_sharing_threads=1),
        MemoryLevelSignature(name="L2", size_bytes=1048576, instances=8, num_sharing_threads=1),
        MemoryLevelSignature(name="L3", size_bytes=33554432, instances=1, num_sharing_threads=16),
        MemoryLevelSignature(name="DRAM", size_bytes=dram_bytes, instances=1, num_sharing_threads=16),
    )
    return MachineSignature(
        model_name=model_name,
        arch=arch,
        vendor=vendor,
        memory_levels=memory_levels,
    )


# ---------------------------------------------------------------------------
# _short_model_name heuristic
# ---------------------------------------------------------------------------


def test_short_model_name_amd() -> None:
    assert _short_model_name("AMD Ryzen 7 7735HS with Radeon Graphics") == "Ryzen-7-7735HS"


def test_short_model_name_intel() -> None:
    assert _short_model_name("Intel(R) Core(TM) i7-14700K") == "Core-i7-14700K"


def test_short_model_name_empty() -> None:
    assert _short_model_name("") == "unknown"


def test_short_model_name_truncation() -> None:
    # A model string whose processed form exceeds 30 chars; must be truncated.
    result = _short_model_name("AMD EPYC 9654X 96-Core Processor Extended Edition")
    assert len(result) == 30
    assert result == "EPYC-9654X-96-Core-Processor-E"


# ---------------------------------------------------------------------------
# config_hash
# ---------------------------------------------------------------------------


def test_config_hash_deterministic() -> None:
    sig = _make_signature()
    assert sig.config_hash == sig.config_hash == _make_signature().config_hash


def test_config_hash_differs_on_dram() -> None:
    base = _make_signature(dram_bytes=17_179_869_184)
    other = _make_signature(dram_bytes=34_359_738_368)
    assert base.config_hash != other.config_hash


def test_config_hash_excludes_model_name() -> None:
    base = _make_signature(model_name="AMD Ryzen 7 7735HS with Radeon Graphics")
    other = _make_signature(model_name="AMD Ryzen 9 7950X")
    # Same topology/config -> identical hash regardless of the model name.
    assert base.config_hash == other.config_hash


# ---------------------------------------------------------------------------
# generate_run_name
# ---------------------------------------------------------------------------


def test_generate_run_name_format() -> None:
    name = generate_run_name(_make_signature())
    assert pytest.importorskip("re").match(r"^[a-zA-Z0-9-]+_[0-9a-f]{8}$", name) is not None
    assert name.startswith("Ryzen-7-7735HS_")


# ---------------------------------------------------------------------------
# detect_machine_signature
# ---------------------------------------------------------------------------


class _FakeTopology:
    """A minimal iterable topology stand-in for detect_machine_signature."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def __iter__(self) -> object:
        # Empty hierarchy keeps the test independent of real sysfs layout.
        return iter(())


def test_detect_machine_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    import architecture.identity as identity

    monkeypatch.setattr(
        identity,
        "read_cpuinfo",
        lambda: CpuInfo(model_name="AMD Ryzen 7 7735HS with Radeon Graphics", vendor="AuthenticAMD"),
    )
    monkeypatch.setattr(identity, "MemoryTopology", _FakeTopology)

    sig = detect_machine_signature()
    assert sig.model_name == "AMD Ryzen 7 7735HS with Radeon Graphics"
    assert sig.vendor == "AuthenticAMD"
    assert sig.memory_levels == ()
    assert isinstance(sig.arch, str) and sig.arch


# ---------------------------------------------------------------------------
# signature_from_architecture
# ---------------------------------------------------------------------------


class _FakeTopoForArch:
    def __iter__(self) -> object:
        return iter(())


def test_signature_from_architecture() -> None:
    arch = SimpleNamespace(
        model_name="AMD Ryzen 7 7735HS with Radeon Graphics",
        arch="x86_64",
        vendor="AuthenticAMD",
        memory_topology=_FakeTopoForArch(),
    )
    sig = signature_from_architecture(arch)  # type: ignore[arg-type]
    assert sig.model_name == "AMD Ryzen 7 7735HS with Radeon Graphics"
    assert sig.arch == "x86_64"
    assert sig.vendor == "AuthenticAMD"
    assert sig.memory_levels == ()
    assert generate_run_name(sig).startswith("Ryzen-7-7735HS_")


def test_signature_from_architecture_falls_back_model_name() -> None:
    # When arch.model_name is None, the signature falls back to arch then "unknown".
    arch_with_arch = SimpleNamespace(
        model_name=None, arch="aarch64", vendor="", memory_topology=None
    )
    sig = signature_from_architecture(arch_with_arch)  # type: ignore[arg-type]
    assert sig.model_name == "aarch64"



# ---------------------------------------------------------------------------
# _levels_from_topology DRAM stabilization
# ---------------------------------------------------------------------------


class _FakeTopo:
    """Mock topology yielding MemoryLevelInfo objects."""

    def __init__(self, levels: list[MemoryLevelInfo]) -> None:
        self._levels = levels

    def __iter__(self):
        return iter(self._levels)


def _make_dram_info(size_bytes: int, sharing: int = 16) -> MemoryLevelInfo:
    return MemoryLevelInfo(name="DRAM", size=Bytes(size_bytes), instances=1, num_sharing_threads=sharing)


def _make_cache_info(name: str, size: int, instances: int, sharing: int) -> MemoryLevelInfo:
    return MemoryLevelInfo(name=name, size=Bytes(size), instances=instances, num_sharing_threads=sharing)


def test_levels_from_topology_rounds_dram_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """DRAM size is rounded to the nearest GiB when zoneinfo is unavailable."""
    import architecture.identity as identity

    monkeypatch.setattr(identity, "_get_physical_ram_bytes", lambda: None)
    topo = _FakeTopo([_make_dram_info(15_966_246_912)])  # 14.87 GiB
    result = _levels_from_topology(topo)
    assert len(result) == 1
    assert result[0].size_bytes == 15 * 1024**3  # 15 GiB = 16_106_127_360


def test_levels_from_topology_zeroes_dram_sharing() -> None:
    """DRAM num_sharing_threads is 0 in the signature."""
    topo = _FakeTopo([_make_dram_info(17_179_869_184, sharing=16)])
    result = _levels_from_topology(topo)
    assert result[0].num_sharing_threads == 0


def test_levels_from_topology_preserves_cache_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cache levels retain their exact fields."""
    import architecture.identity as identity

    monkeypatch.setattr(identity, "_get_physical_ram_bytes", lambda: None)
    topo = _FakeTopo([
        _make_cache_info("L1", 32768, 8, 1),
        _make_cache_info("L2", 1048576, 8, 1),
        _make_cache_info("L3", 33554432, 1, 16),
        _make_dram_info(17_179_869_184),
    ])
    result = _levels_from_topology(topo)
    assert len(result) == 4
    # L1
    assert result[0].name == "L1"
    assert result[0].size_bytes == 32768
    assert result[0].instances == 8
    assert result[0].num_sharing_threads == 1
    # L2
    assert result[1].name == "L2"
    assert result[1].size_bytes == 1048576
    assert result[1].instances == 8
    assert result[1].num_sharing_threads == 1
    # L3
    assert result[2].name == "L3"
    assert result[2].size_bytes == 33554432
    assert result[2].instances == 1
    assert result[2].num_sharing_threads == 16
    # DRAM — rounded + zeroed
    assert result[3].name == "DRAM"
    assert result[3].size_bytes == 16 * 1024**3  # rounds to 16 GiB
    assert result[3].num_sharing_threads == 0


def test_dram_size_rounding_produces_stable_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two signatures with DRAM sizes 10 MiB apart produce same hash (same GiB bucket)."""
    import architecture.identity as identity

    monkeypatch.setattr(identity, "_get_physical_ram_bytes", lambda: None)
    sig1 = MachineSignature(
        model_name="test",
        arch="x86_64",
        vendor="AMD",
        memory_levels=_levels_from_topology(_FakeTopo([_make_dram_info(16_106_127_360)])),  # 15 GiB exact
    )
    sig2 = MachineSignature(
        model_name="test",
        arch="x86_64",
        vendor="AMD",
        memory_levels=_levels_from_topology(_FakeTopo([_make_dram_info(16_216_666_112)])),  # ~15.1 GiB
    )
    assert sig1.config_hash == sig2.config_hash


def test_machine_signature_to_dict() -> None:
    """to_dict() contains all expected keys with correct values."""
    sig = _make_signature(dram_bytes=17_179_869_184)  # 16 GiB
    d = sig.to_dict()
    assert isinstance(d, dict)
    assert "model_name" in d
    assert "arch" in d
    assert "vendor" in d
    assert "config_hash" in d
    assert "hash_input" in d
    assert "memory_levels" in d
    assert isinstance(d["memory_levels"], list)
    assert len(d["memory_levels"]) == 4
    assert d["config_hash"] == sig.config_hash
    assert d["hash_input"] == sig._canonical_hash_string()
    assert d["hash_input"].startswith("arch=x86_64;vendor=AuthenticAMD;levels=")
    for lvl in d["memory_levels"]:
        assert "name" in lvl
        assert "size_bytes" in lvl
        assert "instances" in lvl
        assert "num_sharing_threads" in lvl


def test_write_machine_json_creates_file(tmp_path: object) -> None:
    """write_machine_json creates machine.json with valid JSON."""
    from pathlib import Path

    sig = _make_signature()
    directory = Path(tmp_path)  # type: ignore[arg-type]
    write_machine_json(sig, directory)
    json_path = directory / "machine.json"
    assert json_path.exists()
    import json as _json

    data = _json.loads(json_path.read_text())
    assert data["config_hash"] == sig.config_hash
    assert data["hash_input"] == sig._canonical_hash_string()


def test_write_machine_json_does_not_overwrite(tmp_path: object) -> None:
    """write_machine_json does not overwrite an existing machine.json."""
    from pathlib import Path

    sig = _make_signature()
    directory = Path(tmp_path)  # type: ignore[arg-type]
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "machine.json"
    json_path.write_text('{"sentinel": true}')
    write_machine_json(sig, directory)
    import json as _json

    data = _json.loads(json_path.read_text())
    assert data == {"sentinel": True}
