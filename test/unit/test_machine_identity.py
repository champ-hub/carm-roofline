"""Unit tests for machine identity, signature hashing, and run-name generation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from architecture.identity import (
    CpuInfo,
    MachineSignature,
    MemoryLevelSignature,
    _short_model_name,
    detect_machine_signature,
    generate_run_name,
    signature_from_architecture,
)

pytestmark = pytest.mark.unit


def _make_signature(
    model_name: str = "AMD Ryzen 7 7735HS with Radeon Graphics",
    dram_bytes: int = 17_179_869_184,
    arch: str = "x86_64",
    vendor: str = "AuthenticAMD",
    vector_length: int | None = None,
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
        vector_length=vector_length,
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
    assert sig.vector_length is None
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
        vector_length=32,
    )
    sig = signature_from_architecture(arch)  # type: ignore[arg-type]
    assert sig.model_name == "AMD Ryzen 7 7735HS with Radeon Graphics"
    assert sig.arch == "x86_64"
    assert sig.vendor == "AuthenticAMD"
    assert sig.memory_levels == ()
    assert sig.vector_length == 32
    assert generate_run_name(sig).startswith("Ryzen-7-7735HS_")


def test_signature_from_architecture_falls_back_model_name() -> None:
    # When arch.model_name is None, the signature falls back to arch then "unknown".
    arch_with_arch = SimpleNamespace(
        model_name=None, arch="aarch64", vendor="", memory_topology=None, vector_length=None
    )
    sig = signature_from_architecture(arch_with_arch)  # type: ignore[arg-type]
    assert sig.model_name == "aarch64"
