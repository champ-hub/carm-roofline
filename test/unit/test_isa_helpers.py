from __future__ import annotations

from typing import Any

import pytest

from carm_roofline.benchmark.benchmarking import LoadStoreRatio
from carm_roofline.core import DataType
from carm_roofline.benchmark.generation.code_gen import instruction as inst
from carm_roofline.benchmark.generation.code_gen.register import RegisterCollection
from carm_roofline.isa.base import BaseISA, InlineASM
from carm_roofline.benchmark.generation.parameters import BenchParamError, MemoryBenchmarkParams
from carm_roofline.core import Bytes


class DummyISA(BaseISA):
    name = "dummy"
    family = "dummy"
    max_branch_insts = 64
    max_mem_offset_bytes = 32
    max_immediate = 16
    instruction_limit = 64
    unroll_loop = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)


def test_split_loop_no_inner() -> None:
    isa = DummyISA()

    config = isa._split_loop(num_ops=10, max_loop_size=32)

    assert config.instance_inner_loop is False
    assert config.inner_repeats == 0
    assert config.outer_repeats == 10
    assert config.num_iterations == 0


def test_split_loop_with_inner() -> None:
    isa = DummyISA()

    config = isa._split_loop(num_ops=70, max_loop_size=32)

    assert config.instance_inner_loop is True
    assert config.inner_repeats == 32
    assert config.outer_repeats == 6
    assert config.num_iterations == 2


def test_validate_memory_size() -> None:
    isa = DummyISA()

    params = MemoryBenchmarkParams(
        data_type=DataType.f32,
        thread_affinity=[0],
        load_store_ratio=LoadStoreRatio(1, 1),
        size_per_thread=Bytes(64),
        memory_level_name="L1",
    )

    info = isa._validate_memory_size(
        params,
        bytes_per_inst=4,
        num_loads_per_repeat=1,
        num_stores_per_repeat=1,
    )

    assert info.repeats == 8
    assert info.bytes_per_repeat == 8
    assert info.actual_working_set_size == 64


def test_validate_memory_size_too_small() -> None:
    isa = DummyISA()

    params = MemoryBenchmarkParams(
        data_type=DataType.f32,
        thread_affinity=[0],
        load_store_ratio=LoadStoreRatio(1, 1),
        size_per_thread=Bytes(7),
        memory_level_name="L1",
    )

    with pytest.raises(BenchParamError, match="too small"):
        isa._validate_memory_size(
            params,
            bytes_per_inst=4,
            num_loads_per_repeat=1,
            num_stores_per_repeat=1,
        )


def test_calculate_loop_configuration_ptr_offset() -> None:
    isa = DummyISA()

    params = MemoryBenchmarkParams(
        data_type=DataType.f32,
        thread_affinity=[0],
        load_store_ratio=LoadStoreRatio(1, 0),
        size_per_thread=Bytes(256),
        memory_level_name="L1",
    )

    load_format = inst.Memory("ld {reg}, {off}({ptr})")

    config = isa._calculate_loop_configuration(
        params=params,
        repeats=100,
        insts_per_repeat=1,
        branch_distance_limit=16,
        bytes_per_inst=4,
        load_format=load_format,
    )

    assert config.block_size_offsets == 8
    assert config.bytes_per_block == 32
    assert config.mem_insts_per_loop == 8
    assert config.max_loop_size == 8
    assert config.instance_inner_loop is True
    assert config.inner_repeats == 8
    assert config.outer_repeats == 4


def test_calculate_loop_configuration_ptr_only_requires_unroll() -> None:
    isa = DummyISA()

    params = MemoryBenchmarkParams(
        data_type=DataType.f32,
        thread_affinity=[0],
        load_store_ratio=LoadStoreRatio(1, 0),
        size_per_thread=Bytes(256),
        memory_level_name="L1",
    )

    load_format = inst.Memory("ld {reg}, ({ptr})")

    with pytest.raises(BenchParamError, match="loop unrolling"):
        isa._calculate_loop_configuration(
            params=params,
            repeats=10,
            insts_per_repeat=1,
            branch_distance_limit=16,
            bytes_per_inst=4,
            load_format=load_format,
        )


def test_calculate_loop_configuration_ptr_only_with_unroll() -> None:
    isa = DummyISA()
    isa.unroll_loop = True

    params = MemoryBenchmarkParams(
        data_type=DataType.f32,
        thread_affinity=[0],
        load_store_ratio=LoadStoreRatio(1, 0),
        size_per_thread=Bytes(256),
        memory_level_name="L1",
    )

    load_format = inst.Memory("ld {reg}, ({ptr})")

    config = isa._calculate_loop_configuration(
        params=params,
        repeats=20,
        insts_per_repeat=1,
        branch_distance_limit=10,
        bytes_per_inst=4,
        load_format=load_format,
    )

    assert config.block_size_offsets == 1
    assert config.bytes_per_block == 4
    assert config.mem_insts_per_loop == 5
    assert config.max_loop_size == 5


def test_format_iasm_input_default() -> None:
    isa = DummyISA()

    var = InlineASM.Input(c_name="num_reps", asm_name="num_reps")

    assert isa.format_iasm_input(var) == "%[num_reps]"


def test_offsets_use_bytes_per_inst() -> None:
    isa = DummyISA()

    assert isa.bytes_per_inst(DataType.f32) == 4
    assert isa.max_unique_offsets(DataType.f32) == 8
    assert isa.offset_increment(DataType.f32) == 4


def test_inline_asm_formatting() -> None:
    asm = InlineASM(
        asm=["add r0, r1", "sub r2, r3"],
        inputs=[InlineASM.Input(c_name="data_ptr", asm_name="data_ptr")],
        clobbers=RegisterCollection("r{}", [0, 1]),
    )

    formatted = asm.format()

    assert "__asm__ __volatile__" in formatted
    assert '"add r0, r1\\n\\t"' in formatted
    assert '"sub r2, r3\\n\\t"' in formatted
    assert ': [data_ptr] "m" (data_ptr)' in formatted
    assert ': "r0", "r1"' in formatted
    assert asm.as_function_body() == formatted
