"""Comprehensive test suite for the register refactoring."""
from __future__ import annotations

import pytest

from benchmark.generation.code_gen import DataType
from benchmark.generation.code_gen.register import (
    CyclicRegisterSet,
    HelperRegisterSet,
    RegisterCollection,
    TypedRegisterSets,
)


# Basic functionality tests
@pytest.mark.unit
def test_register_collection_immutable():
    """RegisterCollection should be immutable with indexed access."""
    immutable = RegisterCollection("r{}", [0, (5, 7), 15])
    assert immutable[0] == "r0"
    assert immutable[1] == "r5"
    assert immutable[2] == "r6"
    assert immutable[3] == "r7"
    assert immutable[4] == "r15"
    assert len(immutable.indices) == 5
    assert not hasattr(immutable, "get"), "Immutable collection should not have get()"


@pytest.mark.unit
def test_cyclic_register_set_cycles():
    """CyclicRegisterSet should cycle through registers."""
    cyclic = CyclicRegisterSet("v{}", [(0, 2)])
    r1, r2, r3, r4 = cyclic.get(), cyclic.get(), cyclic.get(), cyclic.get()
    assert r1 == "v0"
    assert r2 == "v1"
    assert r3 == "v2"
    assert r4 == "v0"  # Cycles back


@pytest.mark.unit
def test_helper_register_set_immutable():
    """HelperRegisterSet should provide named accessors without cycling."""
    helpers = HelperRegisterSet("r{}", [0, 1, 2, 3, 4])
    assert helpers.outer_iterator == "r0"
    assert helpers.inner_iterator == "r1"
    assert helpers.pointer == "r2"
    assert helpers.pointer_increment == "r3"
    assert helpers.write_pointer == "r4"
    assert isinstance(helpers, RegisterCollection)
    assert not isinstance(helpers, CyclicRegisterSet)
    assert not hasattr(helpers, "get")


@pytest.mark.unit
def test_typed_register_sets():
    """TypedRegisterSets should map DataType to CyclicRegisterSet."""
    typed = TypedRegisterSets(
        {
            DataType.f32: CyclicRegisterSet("s{}", [(0, 7)]),
            DataType.f64: CyclicRegisterSet("d{}", [(0, 7)]),
        }
    )
    assert typed[DataType.f32].get() == "s0"
    assert typed[DataType.f64].get() == "d0"


@pytest.mark.unit
def test_typed_register_sets_type_validation():
    """TypedRegisterSets should enforce CyclicRegisterSet type."""
    with pytest.raises(TypeError, match="CyclicRegisterSet"):
        TypedRegisterSets(
            {
                DataType.f32: RegisterCollection("s{}", [(0, 7)])  # Wrong type!
            }
        )


# ISA compatibility tests
@pytest.mark.unit
def test_bench_registers_cycle(x86avx_isa):
    """ISA bench registers should cycle through different values."""
    regs = x86avx_isa.bench_registers[DataType.f32]
    reg_values = {regs.get() for _ in range(3)}
    assert len(reg_values) == 3


@pytest.mark.unit
def test_helper_registers_are_fixed(x86avx_isa):
    """ISA helper registers should be fixed (immutable, no cycling)."""
    helpers = x86avx_isa.helper_registers
    helper_values = (
        helpers.outer_iterator,
        helpers.inner_iterator,
        helpers.pointer,
        helpers.pointer_increment,
    )
    assert all(isinstance(value, str) for value in helper_values)
    with pytest.raises(AttributeError):
        helpers.get()


@pytest.mark.unit
@pytest.mark.parametrize(
    "isa_fixture",
    ["x86avx_isa", "x86sse_isa", "arm_neon_isa", "riscv_scalar_isa"],
)
def test_isa_register_structure(isa_fixture, request):
    """All ISAs should have properly structured helper and bench registers."""
    isa = request.getfixturevalue(isa_fixture)

    # Check helper registers are immutable
    assert isinstance(isa.helper_registers, HelperRegisterSet)
    assert not hasattr(isa.helper_registers, "get")

    # Check benchmark registers are cyclic
    bench_regs = isa.bench_registers[DataType.f32]
    assert isinstance(bench_regs, CyclicRegisterSet)

    # Test cycling behavior
    r1 = bench_regs.get()
    r2 = bench_regs.get()
    assert r1 != r2, f"{isa.__class__.__name__}: registers should be different"


# Edge cases and error handling
@pytest.mark.unit
def test_helper_register_set_validates_count():
    """HelperRegisterSet should require at least 5 registers."""
    with pytest.raises(ValueError, match="5 are required"):
        HelperRegisterSet("r{}", [0, 1, 2, 3])  # Only 4, need 5


@pytest.mark.unit
def test_empty_cyclic_set_raises():
    """Empty CyclicRegisterSet should raise on get()."""
    empty = CyclicRegisterSet("r{}", [])
    with pytest.raises((ZeroDivisionError, IndexError)):
        empty.get()


@pytest.mark.unit
def test_single_register_cycles_to_self():
    """Single register in CyclicRegisterSet should cycle to itself."""
    single = CyclicRegisterSet("r{}", [42])
    assert single.get() == "r42"
    assert single.get() == "r42"  # Same register
