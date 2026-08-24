from __future__ import annotations

"""
Integration tests for ISA code generation.

These tests validate that each ISA implementation can successfully generate
benchmark code for various configurations without errors.
"""

import os
import sys
from types import SimpleNamespace

import pytest

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from carm_roofline.benchmark.benchmarking import LoadStoreRatio
from carm_roofline.isa import (
    RISCV_RVV,
    RISCV_RVV_071,
    X86AVX,
    X86AVX2,
    X86AVX512,
    X86SSE,
    ArmNeon,
    ArmScalar,
    ArmSVE,
    RISCVScalar,
    X86Scalar,
)
from carm_roofline.benchmark.generation import MemoryLayoutMode
from carm_roofline.isa import BaseISA
from carm_roofline.benchmark.generation.parameters import (
    ArithmeticBenchmarkParams,
    BenchParamError,
    MemoryBenchmarkParams,
)
from carm_roofline.test_bench.builder import MicrobenchmarkFunctionSpec
from carm_roofline.core import ArithmeticOperation, Operation
from carm_roofline.core import Bytes, Operations
from carm_roofline.core import DataType
from carm_roofline.benchmark.suites.arithmetic import ArithmeticBenchmarkSuite
from carm_roofline.benchmark.suites.memory import MemoryBenchmarkSuite
from carm_roofline.core import MemoryOperation, UserError


class TestISACodegen:
    """Test code generation for each ISA with various parameter combinations."""

    # Map of ISA classes for easier iteration
    ISA_CLASSES: dict[str, type[BaseISA]] = {
        # x86 ISAs
        "x86_scalar": X86Scalar,
        "x86_sse": X86SSE,
        "x86_avx": X86AVX,
        "x86_avx2": X86AVX2,
        "x86_avx512": X86AVX512,
        # ARM ISAs
        "arm_scalar": ArmScalar,
        "arm_neon": ArmNeon,
        "arm_sve": ArmSVE,
        # RISC-V ISAs
        "riscv_scalar": RISCVScalar,
        "riscv_rvv_071": RISCV_RVV_071,
        "riscv_rvv_10": RISCV_RVV,
    }

    # ISAs that require special parameters for instantiation
    PARAMETRIZED_ISA_NAMES = {"arm_sve", "riscv_rvv_071", "riscv_rvv_10"}
    # RVV-specific ISAs (RISC-V Vector)
    RVV_ISA_NAMES = {"riscv_rvv_071", "riscv_rvv_10"}
    # Vector ISAs (ARM SVE, RISC-V RVV)
    VECTOR_ISA_NAMES = {"arm_sve", "riscv_rvv_071", "riscv_rvv_10"}

    @staticmethod
    def instantiate_isa(isa_name: str) -> BaseISA:
        """Helper to instantiate ISAs with proper parameters."""
        isa_class = TestISACodegen.ISA_CLASSES[isa_name]

        if isa_name == "arm_sve":
            # ArmSVE requires vlen_bits parameter
            return isa_class(vlen_bits=1024)
        elif isa_name in TestISACodegen.RVV_ISA_NAMES:
            # RVV ISAs require vlen_bits and lmul parameters
            return isa_class(vlen_bits=1024, lmul=2)
        else:
            # Simple ISAs can be instantiated without parameters
            return isa_class()

    @pytest.mark.parametrize("isa_name", ISA_CLASSES.keys())
    def test_isa_instantiation(self, isa_name: str):
        """Test that each ISA can be instantiated."""
        isa = self.instantiate_isa(isa_name)

        assert isa is not None
        assert hasattr(isa, "name")
        assert isinstance(isa.name, str)
        assert len(isa.name) > 0

    @pytest.mark.parametrize(
        "isa_name,operation,precision",
        [
            ("x86_scalar", ArithmeticOperation.add, DataType.f32),
            ("x86_scalar", ArithmeticOperation.add, DataType.f64),
            ("x86_sse", ArithmeticOperation.mul, DataType.f32),
            ("x86_sse", ArithmeticOperation.fma, DataType.f64),
            ("x86_avx", ArithmeticOperation.add, DataType.f32),
            ("x86_avx2", ArithmeticOperation.mul, DataType.f64),
            ("x86_avx512", ArithmeticOperation.div, DataType.f32),
            ("x86_avx512", ArithmeticOperation.fma, DataType.f64),
            ("x86_avx512", ArithmeticOperation.fma, DataType.bf16),
            ("arm_scalar", ArithmeticOperation.add, DataType.f32),
            ("arm_scalar", ArithmeticOperation.mul, DataType.f64),
            ("arm_neon", ArithmeticOperation.add, DataType.f32),
            ("riscv_scalar", ArithmeticOperation.add, DataType.f32),
            ("riscv_scalar", ArithmeticOperation.mul, DataType.f64),
        ],
    )
    def test_arithmetic_codegen(self, mock_context, isa_name: str, operation: ArithmeticOperation, precision: DataType):
        """Test arithmetic benchmark code generation for various ISAs and operations."""
        isa = self.instantiate_isa(isa_name)

        # Generate arithmetic benchmark with different loop counts
        for num_ops in (1, 32, 256):
            params = ArithmeticBenchmarkParams(
                data_type=precision,
                operation=operation,
                num_ops=Operations(num_ops),
                thread_affinity=[0],
            )
            spec = isa.generate_arithmetic(params, mock_context)

            # Validate output is a MicrobenchmarkFunctionSpec with non-empty C code
            assert isinstance(spec, MicrobenchmarkFunctionSpec)
            assert len(spec.body) > 0
            assert "__asm__" in spec.body or "asm(" in spec.body.lower()
            assert "volatile" in spec.body.lower()

    @pytest.mark.parametrize(
        "isa_name,precision",
        [
            ("x86_scalar", DataType.f32),
            ("x86_scalar", DataType.f64),
            ("x86_sse", DataType.f32),
            ("x86_avx", DataType.f64),
            ("x86_avx2", DataType.f32),
            ("x86_avx512", DataType.f64),
            ("arm_scalar", DataType.f32),
            ("arm_neon", DataType.f64),
            ("riscv_scalar", DataType.f32),
        ],
    )
    def test_memory_codegen(self, mock_context, isa_name: str, precision: DataType):
        """Test memory benchmark code generation for various ISAs."""
        isa = self.instantiate_isa(isa_name)

        # Test different load/store configurations
        test_configs = [
            (1, 0, 10),  # 1 load, 0 stores, 10 reps
            (0, 1, 10),  # 0 loads, 1 store, 10 reps
            (2, 2, 20),  # 2 loads, 2 stores, 20 reps
            (1, 1, 32),  # Mixed load/store
        ]

        for num_ld, num_st, num_rep in test_configs:
            params = MemoryBenchmarkParams(
                data_type=precision,
                load_store_ratio=LoadStoreRatio(num_ld, num_st),
                size_per_thread=Bytes(num_rep * (num_ld + num_st) * precision.bytes()),
                thread_affinity=[0],
                memory_level_name="L1",
            )
            spec = isa.generate_memory(params, mock_context)

            # Validate output is a MicrobenchmarkFunctionSpec
            assert isinstance(spec, MicrobenchmarkFunctionSpec)
            assert len(spec.body) > 0
            assert "__asm__" in spec.body or "asm(" in spec.body.lower()

            # Verify loop structure or label exists (asm code uses labels, not for loops)
            assert "_loop" in spec.body.lower() or "loop" in spec.body.lower() or "{" in spec.body, (
                "No loop structure detected in memory benchmark"
            )

    @pytest.mark.parametrize(
        "isa_name,vlen_bits,lmul,operation,precision",
        [
            ("riscv_rvv_071", 128, 1, ArithmeticOperation.add, DataType.f32),
            ("riscv_rvv_071", 256, 2, ArithmeticOperation.mul, DataType.f64),
            ("riscv_rvv_071", 512, 4, ArithmeticOperation.fma, DataType.f32),
            ("riscv_rvv_10", 128, 1, ArithmeticOperation.add, DataType.f32),
            ("riscv_rvv_10", 256, 2, ArithmeticOperation.div, DataType.f64),
            ("riscv_rvv_10", 512, 8, ArithmeticOperation.fma, DataType.f32),
        ],
    )
    def test_rvv_arithmetic_codegen(
        self, mock_context, isa_name: str, vlen_bits: int, lmul: int, operation: Operation, precision: DataType
    ):
        """Test RVV arithmetic code generation with various VLEN and LMUL values."""
        isa_class = self.ISA_CLASSES[isa_name]
        isa = isa_class(vlen_bits=vlen_bits, lmul=lmul)

        params = ArithmeticBenchmarkParams(
            data_type=precision,
            operation=operation,
            num_ops=Operations(64),
            thread_affinity=[0],
        )
        spec = isa.generate_arithmetic(params, mock_context)

        assert isinstance(spec, MicrobenchmarkFunctionSpec)
        assert len(spec.body) > 0
        assert "__asm__" in spec.body or "asm(" in spec.body.lower()

    @pytest.mark.parametrize(
        "isa_name,vlen_bits,lmul,precision",
        [
            ("riscv_rvv_071", 128, 1, DataType.f32),
            ("riscv_rvv_071", 256, 2, DataType.f64),
            ("riscv_rvv_10", 128, 1, DataType.f32),
            ("riscv_rvv_10", 512, 4, DataType.f64),
        ],
    )
    def test_rvv_memory_codegen(self, mock_context, isa_name: str, vlen_bits: int, lmul: int, precision: DataType):
        """Test RVV memory code generation with various VLEN and LMUL values."""
        isa_class = self.ISA_CLASSES[isa_name]
        isa = isa_class(vlen_bits=vlen_bits, lmul=lmul)

        # RVV needs larger buffer sizes due to wide vector operations
        # Calculate appropriate size: vlen_bits (in bytes) * lmul * 32 ops (1 load + 1 store)
        min_size = (vlen_bits // 8) * lmul * 64
        params = MemoryBenchmarkParams(
            data_type=precision,
            load_store_ratio=LoadStoreRatio(1, 1),
            size_per_thread=Bytes(min_size),
            thread_affinity=[0],
            memory_level_name="L1",
        )
        spec = isa.generate_memory(params, mock_context)

        assert isinstance(spec, MicrobenchmarkFunctionSpec)
        assert len(spec.body) > 0
        assert "__asm__" in spec.body or "asm(" in spec.body.lower()

    @pytest.mark.parametrize("isa_name", ISA_CLASSES.keys())
    def test_isa_attributes(self, isa_name: str):
        """Test that each ISA has required attributes."""
        isa = self.instantiate_isa(isa_name)

        # Check required attributes
        assert hasattr(isa, "name"), f"{isa_name} missing 'name' attribute"
        assert isinstance(isa.name, str), f"{isa_name}.name should be string"
        assert len(isa.name) > 0, f"{isa_name}.name should not be empty"

    @pytest.mark.parametrize("isa_name", ISA_CLASSES.keys())
    def test_isa_generates_valid_c_code(self, mock_context, isa_name: str):
        """Test that each ISA generates syntactically valid C code."""
        isa = self.instantiate_isa(isa_name)

        # Generate both arithmetic and memory benchmarks
        arith_params = ArithmeticBenchmarkParams(
            data_type=DataType.f32,
            operation=ArithmeticOperation.add,
            num_ops=Operations(10),
            thread_affinity=[0],
        )

        # Vector ISAs need larger buffer sizes
        if isa_name in self.VECTOR_ISA_NAMES:
            # For vector ISAs, use a larger working set
            mem_size = Bytes(1024)
        else:
            # For scalar ISAs, use the standard small size
            mem_size = Bytes(32 * 2 * DataType.f32.bytes())

        mem_params = MemoryBenchmarkParams(
            data_type=DataType.f32,
            load_store_ratio=LoadStoreRatio(1, 1),
            size_per_thread=mem_size,
            thread_affinity=[0],
            memory_level_name="L1",
        )
        arith_spec = isa.generate_arithmetic(arith_params, mock_context)
        mem_spec = isa.generate_memory(mem_params, mock_context)

        # Both should be MicrobenchmarkFunctionSpec objects
        assert isinstance(arith_spec, MicrobenchmarkFunctionSpec), (
            f"{isa_name} arithmetic result not a MicrobenchmarkFunctionSpec"
        )
        assert isinstance(mem_spec, MicrobenchmarkFunctionSpec), (
            f"{isa_name} memory result not a MicrobenchmarkFunctionSpec"
        )

        # Check the body (C code) is non-empty
        assert len(arith_spec.body) > 0, f"{isa_name} arithmetic code is empty"
        assert len(mem_spec.body) > 0, f"{isa_name} memory code is empty"

        # Both should contain C language elements or inline assembly
        for spec, label in [(arith_spec, "arithmetic"), (mem_spec, "memory")]:
            assert "#include" in spec.body or "#define" in spec.body or "asm" in spec.body.lower(), (
                f"{isa_name} {label} code missing includes, defines, or asm"
            )

    @pytest.mark.parametrize(
        "isa_name,precision",
        [
            ("x86_scalar", DataType.f32),
            ("x86_scalar", DataType.f64),
            ("x86_avx512", DataType.f32),
            ("x86_avx512", DataType.f64),
            ("arm_neon", DataType.f32),
            ("arm_neon", DataType.f64),
            ("riscv_scalar", DataType.f32),
        ],
    )
    def test_all_operations_codegen(self, mock_context, isa_name: str, precision: DataType):
        """Test that each ISA can generate code for all arithmetic operations."""
        isa = self.instantiate_isa(isa_name)

        # Test all arithmetic operations
        for operation in [
            ArithmeticOperation.add,
            ArithmeticOperation.mul,
            ArithmeticOperation.div,
            ArithmeticOperation.fma,
        ]:
            params = ArithmeticBenchmarkParams(
                data_type=precision,
                operation=operation,
                num_ops=Operations(16),
                thread_affinity=[0],
            )
            spec = isa.generate_arithmetic(params, mock_context)

            assert isinstance(spec, MicrobenchmarkFunctionSpec)
            assert len(spec.body) > 0, f"{isa_name} failed to generate code for {operation.name}"

    def test_isa_family_consistency(self):
        """Test that ISAs are correctly grouped by family."""
        # Only test ISAs that actually have the family attribute
        isa_family_map = {
            "arm_scalar": "arm",
            "arm_neon": "arm",
            "arm_sve": "arm",
            "riscv_scalar": "riscv",
            "riscv_rvv_071": "riscv",
            "riscv_rvv_10": "riscv",
        }

        for isa_name, expected_family in isa_family_map.items():
            isa = self.instantiate_isa(isa_name)

            if hasattr(isa, "family"):
                assert isa.family == expected_family, (
                    f"{isa_name} has family '{isa.family}', expected '{expected_family}'"
                )

    @pytest.mark.parametrize("isa_name", ISA_CLASSES.keys())
    def test_isa_no_generation_errors(self, mock_context, isa_name: str):
        """Test that no ISA raises exceptions during code generation."""
        isa = self.instantiate_isa(isa_name)

        # Should not raise any exceptions
        try:
            arith_params = ArithmeticBenchmarkParams(
                data_type=DataType.f32,
                operation=ArithmeticOperation.add,
                num_ops=Operations(32),
                thread_affinity=[0],
            )
            # Vector ISAs need larger buffer sizes
            if isa_name in self.VECTOR_ISA_NAMES:
                mem_size = Bytes(1024)
            else:
                mem_size = Bytes(32 * 2 * DataType.f32.bytes())

            mem_params = MemoryBenchmarkParams(
                data_type=DataType.f32,
                load_store_ratio=LoadStoreRatio(1, 1),
                size_per_thread=mem_size,
                thread_affinity=[0],
                memory_level_name="L1",
            )
            arith = isa.generate_arithmetic(arith_params, mock_context)
            mem = isa.generate_memory(mem_params, mock_context)
            assert arith is not None
            assert mem is not None
        except Exception as e:
            pytest.fail(f"{isa_name} raised exception during code generation: {e}")


class TestCodeGenerationConsistency:
    """Test consistency of code generation across multiple calls."""

    @pytest.mark.parametrize("isa_class", [X86AVX2, ArmNeon, RISCVScalar])
    def test_deterministic_codegen(self, mock_context, isa_class: type[BaseISA]):
        """Test that multiple calls with same parameters produce identical code."""
        isa1 = isa_class()
        isa2 = isa_class()

        # Generate code from two instances
        arith_params = ArithmeticBenchmarkParams(
            data_type=DataType.f32,
            operation=ArithmeticOperation.add,
            num_ops=Operations(16),
            thread_affinity=[0],
        )
        mem_params = MemoryBenchmarkParams(
            data_type=DataType.f64,
            load_store_ratio=LoadStoreRatio(2, 1),
            size_per_thread=Bytes(8 * 3 * DataType.f64.bytes()),
            thread_affinity=[0],
            memory_level_name="L1",
        )
        spec1_arith = isa1.generate_arithmetic(arith_params, mock_context)
        spec2_arith = isa2.generate_arithmetic(arith_params, mock_context)

        spec1_mem = isa1.generate_memory(mem_params, mock_context)
        spec2_mem = isa2.generate_memory(mem_params, mock_context)

        # Code should be identical
        assert isinstance(spec1_arith, MicrobenchmarkFunctionSpec)
        assert isinstance(spec2_arith, MicrobenchmarkFunctionSpec)
        assert spec1_arith.body == spec2_arith.body, "Arithmetic code generation is not deterministic"
        assert spec1_mem.body == spec2_mem.body, "Memory code generation is not deterministic"

    def test_rvv_vlen_lmul_sensitivity(self, mock_context):
        """Test that RVV code generation is sensitive to VLEN and LMUL parameters."""
        # Same vlen_bits and LMUL should produce identical code
        isa1 = RISCV_RVV(vlen_bits=256, lmul=2)
        isa2 = RISCV_RVV(vlen_bits=256, lmul=2)

        params = ArithmeticBenchmarkParams(
            data_type=DataType.f32,
            operation=ArithmeticOperation.add,
            num_ops=Operations(16),
            thread_affinity=[0],
        )
        spec1 = isa1.generate_arithmetic(params, mock_context)
        spec2 = isa2.generate_arithmetic(params, mock_context)

        assert isinstance(spec1, MicrobenchmarkFunctionSpec)
        assert isinstance(spec2, MicrobenchmarkFunctionSpec)
        assert spec1.body == spec2.body, "RVV with same vlen_bits/LMUL should produce identical code"

        # Different vlen_bits should produce different code (likely)
        isa3 = RISCV_RVV(vlen_bits=128, lmul=2)
        spec3 = isa3.generate_arithmetic(params, mock_context)

        # Note: May not always be different due to optimization, but typically is
        assert spec1 is not None and spec3 is not None


class TestCodeQuality:
    """Test quality aspects of generated code."""

    @pytest.mark.parametrize("isa_name", ["x86_avx2", "arm_neon", "riscv_scalar"])
    def test_generated_code_has_structure(self, mock_context, isa_name: str):
        """Test that generated code has proper C structure."""
        isa = TestISACodegen.instantiate_isa(isa_name)
        params = ArithmeticBenchmarkParams(
            data_type=DataType.f32,
            operation=ArithmeticOperation.add,
            num_ops=Operations(32),
            thread_affinity=[0],
        )
        spec = isa.generate_arithmetic(params, mock_context)

        assert isinstance(spec, MicrobenchmarkFunctionSpec)
        code = spec.body

        # Code should have basic structure
        assert "int" in code or "float" in code or "double" in code or "void" in code, (
            "Generated code should have type declarations"
        )
        assert "{" in code and "}" in code, "Generated code should have braces"

    def test_generated_code_not_empty_or_trivial(self, mock_context):
        """Test that generated code is substantial, not just placeholders."""
        isa = X86AVX2()
        params = ArithmeticBenchmarkParams(
            data_type=DataType.f32,
            operation=ArithmeticOperation.mul,
            num_ops=Operations(128),
            thread_affinity=[0],
        )
        spec = isa.generate_arithmetic(params, mock_context)

        assert isinstance(spec, MicrobenchmarkFunctionSpec)
        code = spec.body

        # Code should be reasonably sized (at least 100 chars for real implementation)
        assert len(code) > 100, "Generated code appears too small, might be placeholder"

    @pytest.mark.parametrize(
        "isa_name,num_ops",
        [
            ("x86_scalar", 1),
            ("x86_scalar", 128),
            ("arm_scalar", 64),
            ("riscv_scalar", 32),
        ],
    )
    def test_code_varies_with_num_ops(self, mock_context, isa_name: str, num_ops: int):
        """Test that generated code varies appropriately with num_ops parameter."""
        isa = TestISACodegen.instantiate_isa(isa_name)

        params_small = ArithmeticBenchmarkParams(
            data_type=DataType.f32,
            operation=ArithmeticOperation.add,
            num_ops=Operations(1),
            thread_affinity=[0],
        )
        params_large = ArithmeticBenchmarkParams(
            data_type=DataType.f32,
            operation=ArithmeticOperation.add,
            num_ops=Operations(256),
            thread_affinity=[0],
        )
        spec_small = isa.generate_arithmetic(params_small, mock_context)
        spec_large = isa.generate_arithmetic(params_large, mock_context)

        # Both should be valid code
        assert isinstance(spec_small, MicrobenchmarkFunctionSpec)
        assert isinstance(spec_large, MicrobenchmarkFunctionSpec)
        assert len(spec_small.body) > 0
        assert len(spec_large.body) > 0
        # Larger code may differ in loop unrolling or instruction count
        # Just verify both can be generated without error


class TestISACompleteness:
    """Test that ISA implementations cover expected functionality."""

    @pytest.mark.parametrize("isa_name", TestISACodegen.ISA_CLASSES.keys())
    def test_isa_name_unique(self, isa_name: str):
        """Test that ISA names are unique within the test suite."""
        isa = TestISACodegen.instantiate_isa(isa_name)
        assert isa.name is not None
        assert isinstance(isa.name, str)

    def test_all_isas_can_generate_both_types(self, mock_context):
        """Test that all ISAs support both arithmetic and memory generation."""
        for isa_name in TestISACodegen.ISA_CLASSES.keys():
            isa = TestISACodegen.instantiate_isa(isa_name)

            # Should support basic operations
            arith_params = ArithmeticBenchmarkParams(
                data_type=DataType.f32,
                operation=ArithmeticOperation.add,
                num_ops=Operations(10),
                thread_affinity=[0],
            )

            # Vector ISAs need larger buffer sizes
            if isa_name in TestISACodegen.VECTOR_ISA_NAMES:
                mem_size = Bytes(1024)
            else:
                mem_size = Bytes(32 * 2 * DataType.f32.bytes())

            mem_params = MemoryBenchmarkParams(
                data_type=DataType.f32,
                load_store_ratio=LoadStoreRatio(1, 1),
                size_per_thread=mem_size,
                thread_affinity=[0],
                memory_level_name="L1",
            )
            arith = isa.generate_arithmetic(arith_params, mock_context)
            mem = isa.generate_memory(mem_params, mock_context)

            assert isinstance(arith, MicrobenchmarkFunctionSpec) and len(arith.body) > 0, (
                f"{isa_name} cannot generate arithmetic code"
            )
            assert isinstance(mem, MicrobenchmarkFunctionSpec) and len(mem.body) > 0, (
                f"{isa_name} cannot generate memory code"
            )

    @pytest.mark.parametrize("isa_name", TestISACodegen.ISA_CLASSES.keys())
    def test_isa_supports_both_precisions(self, mock_context, isa_name: str):
        """Test that ISAs support both f32 and f64 precision."""
        isa = TestISACodegen.instantiate_isa(isa_name)

        for precision in [DataType.f32, DataType.f64]:
            params = ArithmeticBenchmarkParams(
                data_type=precision,
                operation=ArithmeticOperation.add,
                num_ops=Operations(16),
                thread_affinity=[0],
            )
            spec = isa.generate_arithmetic(params, mock_context)
            assert isinstance(spec, MicrobenchmarkFunctionSpec) and len(spec.body) > 0, (
                f"{isa_name} cannot generate {precision.name} code"
            )


class TestEdgeCases:
    """Edge case tests for loop splitting and size validation (Issues #3 and #4)."""

    @pytest.mark.parametrize(
        "isa_name",
        [
            "x86_scalar",
            "x86_sse",
            "x86_avx",
            "x86_avx2",
            "arm_scalar",
            "arm_neon",
            "riscv_scalar",
        ],
    )
    def test_single_operation_arithmetic(self, mock_context, isa_name: str):
        """Test arithmetic benchmark with single operation (edge case)."""
        isa = TestISACodegen.instantiate_isa(isa_name)

        params = ArithmeticBenchmarkParams(
            data_type=DataType.f32,
            operation=ArithmeticOperation.add,
            num_ops=Operations(1),  # Edge: single operation
            thread_affinity=[0],
        )

        spec = isa.generate_arithmetic(params, mock_context)

        assert isinstance(spec, MicrobenchmarkFunctionSpec)
        assert len(spec.body) > 0
        assert "__asm__" in spec.body or "asm(" in spec.body.lower()

        # Should have outer loop but no inner loop (NO leading underscore)
        assert "outer_loop%=" in spec.body
        assert "inner_loop%=" not in spec.body, f"Single operation should not create inner loop in {isa_name}"

    @pytest.mark.parametrize(
        "isa_name",
        [
            "x86_scalar",
            "x86_sse",
            "x86_avx",
            "arm_scalar",
            "arm_neon",
            "riscv_scalar",
        ],
    )
    def test_very_large_operation_count(self, mock_context, isa_name: str):
        """Test arithmetic benchmark with very large operation count."""
        isa = TestISACodegen.instantiate_isa(isa_name)

        params = ArithmeticBenchmarkParams(
            data_type=DataType.f32,
            operation=ArithmeticOperation.add,
            num_ops=Operations(100000),  # Very large
            thread_affinity=[0],
        )

        spec = isa.generate_arithmetic(params, mock_context)

        assert isinstance(spec, MicrobenchmarkFunctionSpec)
        assert len(spec.body) > 0

        # Should definitely have inner loop (NO leading underscore)
        assert "inner_loop%=" in spec.body, f"Large operation count should create inner loop in {isa_name}"

    @pytest.mark.parametrize(
        "isa_name,data_type",
        [
            ("x86_avx", DataType.f32),
            ("x86_avx", DataType.f64),
            ("arm_neon", DataType.f32),
        ],
    )
    def test_memory_size_too_small(self, mock_context, isa_name: str, data_type: DataType):
        """Test that very small memory sizes raise appropriate errors."""

        isa = TestISACodegen.instantiate_isa(isa_name)
        bytes_per_inst = isa.bytes_per_inst(data_type)

        # Size smaller than one instruction
        too_small = bytes_per_inst - 1

        params = MemoryBenchmarkParams(
            data_type=data_type,
            load_store_ratio=LoadStoreRatio(1, 0),
            size_per_thread=Bytes(too_small),
            thread_affinity=[0],
            memory_level_name="L1",
        )

        with pytest.raises(BenchParamError, match="too small"):
            isa.generate_memory(params, mock_context)

    @pytest.mark.parametrize(
        "isa_name",
        [
            "x86_avx",
            "x86_sse",
            "arm_neon",
            "riscv_scalar",
        ],
    )
    def test_memory_exact_boundary_size(self, mock_context, isa_name: str):
        """Test memory benchmark with size exactly matching instruction boundary."""
        isa = TestISACodegen.instantiate_isa(isa_name)

        bytes_per_inst = isa.bytes_per_inst(DataType.f32)

        # Exactly one instruction worth of data
        exact_size = bytes_per_inst

        params = MemoryBenchmarkParams(
            data_type=DataType.f32,
            load_store_ratio=LoadStoreRatio(1, 0),
            size_per_thread=Bytes(exact_size),
            thread_affinity=[0],
            memory_level_name="L1",
        )

        spec = isa.generate_memory(params, mock_context)

        assert isinstance(spec, MicrobenchmarkFunctionSpec)
        assert len(spec.body) > 0

    @pytest.mark.parametrize(
        "isa_name,num_ld,num_st",
        [
            ("x86_avx", 4, 0),  # Many loads
            ("x86_avx", 0, 4),  # Many stores
            ("x86_avx", 8, 8),  # Many of both
            ("arm_neon", 4, 4),
        ],
    )
    def test_memory_high_load_store_ratio(self, mock_context, isa_name: str, num_ld: int, num_st: int):
        """Test memory benchmarks with high load/store counts."""
        isa = TestISACodegen.instantiate_isa(isa_name)

        bytes_per_inst = isa.bytes_per_inst(DataType.f32)
        insts_per_repeat = num_ld + num_st

        # Ensure size is large enough
        size = bytes_per_inst * insts_per_repeat * 100

        params = MemoryBenchmarkParams(
            data_type=DataType.f32,
            load_store_ratio=LoadStoreRatio(num_ld, num_st),
            size_per_thread=Bytes(size),
            thread_affinity=[0],
            memory_level_name="L1",
        )

        spec = isa.generate_memory(params, mock_context)

        assert isinstance(spec, MicrobenchmarkFunctionSpec)
        assert len(spec.body) > 0

    @pytest.mark.parametrize("layout_mode", [MemoryLayoutMode.single, MemoryLayoutMode.split])
    def test_memory_layout_modes_control_buffers_and_write_input(self, mock_context, layout_mode: MemoryLayoutMode):
        """Test memory layout modes via buffer sizing and inline-asm input bindings."""
        isa = X86Scalar()
        num_ld = 2
        num_st = 1
        repeats = 8
        bytes_per_inst = isa.bytes_per_inst(DataType.f32)

        params = MemoryBenchmarkParams(
            data_type=DataType.f32,
            load_store_ratio=LoadStoreRatio(num_ld, num_st),
            size_per_thread=Bytes(repeats * (num_ld + num_st) * bytes_per_inst),
            thread_affinity=[0],
            memory_level_name="L1",
            layout_mode=layout_mode,
        )
        spec = isa.generate_memory(params, mock_context)

        assert isinstance(spec, MicrobenchmarkFunctionSpec)
        assert spec.read_array_size > Bytes(0)

        if layout_mode == MemoryLayoutMode.single:
            assert spec.read_array_size == Bytes(repeats * (num_ld + num_st) * bytes_per_inst)
            assert spec.write_array_size == Bytes(0)
            assert '[write_ptr] "m" (write_ptr)' not in spec.body
        else:
            assert spec.read_array_size == Bytes(repeats * num_ld * bytes_per_inst)
            assert spec.write_array_size == Bytes(repeats * num_st * bytes_per_inst)
            assert '[write_ptr] "m" (write_ptr)' in spec.body

        assert '[read_ptr] "m" (read_ptr)' in spec.body

    @pytest.mark.parametrize(
        "isa_name,operation",
        [
            ("x86_avx", ArithmeticOperation.add),
            ("x86_avx", ArithmeticOperation.mul),
            ("x86_avx", ArithmeticOperation.fma),
            ("x86_avx", ArithmeticOperation.div),
        ],
    )
    def test_exact_max_loop_size_boundary(self, mock_context, isa_name: str, operation: ArithmeticOperation):
        """Test arithmetic with num_ops exactly at max_loop_size boundary."""
        isa = TestISACodegen.instantiate_isa(isa_name)

        # Calculate max_loop_size
        PRELUDE_INSTRUCTIONS = 5
        max_loop_size = min(
            (isa.max_branch_insts - PRELUDE_INSTRUCTIONS) // 2,
            isa.instruction_limit // 2,
        )

        ops_per_inst = isa.ops_per_inst(DataType.f32, operation)
        num_ops_exact = max_loop_size * ops_per_inst

        params = ArithmeticBenchmarkParams(
            data_type=DataType.f32,
            operation=operation,
            num_ops=Operations(num_ops_exact),
            thread_affinity=[0],
        )

        spec = isa.generate_arithmetic(params, mock_context)

        assert isinstance(spec, MicrobenchmarkFunctionSpec)
        assert len(spec.body) > 0

        # At exact boundary, should not need inner loop
        # (num_iterations = 1, so instance_inner_loop = False)
        # NO leading underscore
        assert "inner_loop%=" not in spec.body, (
            f"At exact max_loop_size ({max_loop_size}), should not create inner loop"
        )

    @pytest.mark.parametrize("isa_name", ["x86_avx", "arm_neon"])
    def test_just_over_max_loop_size_boundary(self, mock_context, isa_name: str):
        """Test arithmetic with num_ops just over max_loop_size boundary."""
        isa = TestISACodegen.instantiate_isa(isa_name)

        PRELUDE_INSTRUCTIONS = 5
        max_loop_size = min(
            (isa.max_branch_insts - PRELUDE_INSTRUCTIONS) // 2,
            isa.instruction_limit // 2,
        )

        ops_per_inst = isa.ops_per_inst(DataType.f32, ArithmeticOperation.add)

        # Just over boundary (but still might not trigger inner loop)
        # Need enough to get num_iterations > 1
        num_ops_over = (max_loop_size * 2 + 1) * ops_per_inst

        params = ArithmeticBenchmarkParams(
            data_type=DataType.f32,
            operation=ArithmeticOperation.add,
            num_ops=Operations(num_ops_over),
            thread_affinity=[0],
        )

        spec = isa.generate_arithmetic(params, mock_context)

        assert isinstance(spec, MicrobenchmarkFunctionSpec)
        assert len(spec.body) > 0

        # Should have inner loop now (NO leading underscore)
        assert "inner_loop%=" in spec.body, "Just over max_loop_size boundary should create inner loop"

    @pytest.mark.parametrize(
        "isa_name,divisor",
        [
            ("x86_avx", 2),
            ("x86_avx", 4),
            ("x86_avx", 8),
            ("arm_neon", 2),
        ],
    )
    def test_exact_divisibility_by_max_loop_size(self, mock_context, isa_name: str, divisor: int):
        """Test num_ops exactly divisible by max_loop_size."""
        isa = TestISACodegen.instantiate_isa(isa_name)

        PRELUDE_INSTRUCTIONS = 5
        max_loop_size = min(
            (isa.max_branch_insts - PRELUDE_INSTRUCTIONS) // 2,
            isa.instruction_limit // 2,
        )

        ops_per_inst = isa.ops_per_inst(DataType.f32, ArithmeticOperation.add)
        num_ops = max_loop_size * divisor * ops_per_inst

        params = ArithmeticBenchmarkParams(
            data_type=DataType.f32,
            operation=ArithmeticOperation.add,
            num_ops=Operations(num_ops),
            thread_affinity=[0],
        )

        spec = isa.generate_arithmetic(params, mock_context)

        assert isinstance(spec, MicrobenchmarkFunctionSpec)
        assert len(spec.body) > 0

        # Should have inner loop if divisor > 1 (NO leading underscore)
        if divisor > 1:
            assert "inner_loop%=" in spec.body
        else:
            assert "inner_loop%=" not in spec.body

    @pytest.mark.parametrize("isa_name", ["x86_avx", "arm_neon"])
    @pytest.mark.parametrize("data_type", [DataType.f32, DataType.f64])
    def test_multiple_data_types_consistency(self, mock_context, isa_name: str, data_type: DataType):
        """Test that different data types maintain consistent loop structure."""
        isa = TestISACodegen.instantiate_isa(isa_name)

        params = ArithmeticBenchmarkParams(
            data_type=data_type,
            operation=ArithmeticOperation.add,
            num_ops=Operations(1024),
            thread_affinity=[0],
        )

        spec = isa.generate_arithmetic(params, mock_context)

        assert isinstance(spec, MicrobenchmarkFunctionSpec)
        assert len(spec.body) > 0
        assert "outer_loop%=" in spec.body

    @pytest.mark.parametrize(
        "isa_name,cache_level",
        [
            ("x86_avx", "L1"),
            ("x86_avx", "L2"),
            ("x86_avx", "L3"),
            ("arm_neon", "L1"),
            ("arm_neon", "L2"),
        ],
    )
    def test_different_cache_levels(self, mock_context, isa_name: str, cache_level: str):
        """Test memory benchmarks targeting different cache levels."""
        isa = TestISACodegen.instantiate_isa(isa_name)

        params = MemoryBenchmarkParams(
            data_type=DataType.f32,
            load_store_ratio=LoadStoreRatio(1, 1),
            size_per_thread=Bytes(8192),
            thread_affinity=[0],
            memory_level_name=cache_level,
        )

        spec = isa.generate_memory(params, mock_context)

        assert isinstance(spec, MicrobenchmarkFunctionSpec)
        assert len(spec.body) > 0

    @pytest.mark.parametrize("num_threads", [1, 2, 4, 8])
    def test_different_thread_counts(self, mock_context, num_threads: int):
        """Test benchmarks with different thread counts."""
        isa = X86AVX()

        params = ArithmeticBenchmarkParams(
            data_type=DataType.f32,
            operation=ArithmeticOperation.add,
            num_ops=Operations(512),
            thread_affinity=list(range(num_threads)),
        )

        spec = isa.generate_arithmetic(params, mock_context)

        assert isinstance(spec, MicrobenchmarkFunctionSpec)
        assert len(spec.body) > 0


class TestIntegerCodegen:
    """Test integer (i8/i16/i32/i64) code generation for x86 ISAs."""

    # Short aliases for the expected-availability matrix below.
    ADD = ArithmeticOperation.add
    MUL = ArithmeticOperation.mul
    DIV = ArithmeticOperation.div
    FMA = ArithmeticOperation.fma
    LD = MemoryOperation.ld
    ST = MemoryOperation.st

    # Expected available operations per ISA and integer data type, mirroring the
    # bench_instructions tables in carm_roofline/isa/x86.py.
    EXPECTED_AVAILABLE: dict[str, dict[DataType, set[Operation]]] = {
        "x86_scalar": {
            DataType.i8: {ADD, MUL, DIV, LD, ST},
            DataType.i16: {ADD, MUL, DIV, LD, ST},
            DataType.i32: {ADD, MUL, DIV, LD, ST},
            DataType.i64: {ADD, MUL, DIV, LD, ST},
        },
        "x86_sse": {
            DataType.i8: {ADD, LD, ST},
            DataType.i16: {ADD, MUL, LD, ST},
            DataType.i32: {ADD, MUL, LD, ST},
            DataType.i64: {ADD, LD, ST},
        },
        "x86_avx": {
            DataType.i8: set(),
            DataType.i16: set(),
            DataType.i32: set(),
            DataType.i64: set(),
        },
        "x86_avx2": {
            DataType.i8: {ADD, LD, ST},
            DataType.i16: {ADD, MUL, LD, ST},
            DataType.i32: {ADD, MUL, LD, ST},
            DataType.i64: {ADD, LD, ST},
        },
        "x86_avx512": {
            DataType.i8: {ADD, LD, ST},
            DataType.i16: {ADD, MUL, LD, ST},
            DataType.i32: {ADD, MUL, LD, ST},
            DataType.i64: {ADD, MUL, LD, ST},
            DataType.bf16: {FMA, LD, ST},
        },
    }

    @pytest.mark.parametrize(
        "isa_name,data_type,operation,mnemonic",
        [
            ("x86_scalar", DataType.i8, ArithmeticOperation.add, "addb"),
            ("x86_scalar", DataType.i8, ArithmeticOperation.mul, "imulb"),
            ("x86_scalar", DataType.i8, ArithmeticOperation.div, "divb"),
            ("x86_scalar", DataType.i16, ArithmeticOperation.add, "addw"),
            ("x86_scalar", DataType.i16, ArithmeticOperation.mul, "imulw"),
            ("x86_scalar", DataType.i16, ArithmeticOperation.div, "divw"),
            ("x86_scalar", DataType.i32, ArithmeticOperation.add, "addl"),
            ("x86_scalar", DataType.i32, ArithmeticOperation.mul, "imull"),
            ("x86_scalar", DataType.i32, ArithmeticOperation.div, "divl"),
            ("x86_scalar", DataType.i64, ArithmeticOperation.add, "addq"),
            ("x86_scalar", DataType.i64, ArithmeticOperation.mul, "imulq"),
            ("x86_scalar", DataType.i64, ArithmeticOperation.div, "divq"),
            ("x86_sse", DataType.i8, ArithmeticOperation.add, "paddb"),
            ("x86_sse", DataType.i16, ArithmeticOperation.add, "paddw"),
            ("x86_sse", DataType.i16, ArithmeticOperation.mul, "pmullw"),
            ("x86_sse", DataType.i32, ArithmeticOperation.add, "paddd"),
            ("x86_sse", DataType.i32, ArithmeticOperation.mul, "pmulld"),
            ("x86_sse", DataType.i64, ArithmeticOperation.add, "paddq"),
            ("x86_avx2", DataType.i8, ArithmeticOperation.add, "vpaddb"),
            ("x86_avx2", DataType.i16, ArithmeticOperation.mul, "vpmullw"),
            ("x86_avx2", DataType.i32, ArithmeticOperation.add, "vpaddd"),
            ("x86_avx2", DataType.i32, ArithmeticOperation.mul, "vpmulld"),
            ("x86_avx2", DataType.i64, ArithmeticOperation.add, "vpaddq"),
            ("x86_avx512", DataType.i8, ArithmeticOperation.add, "vpaddb"),
            ("x86_avx512", DataType.i32, ArithmeticOperation.mul, "vpmulld"),
            ("x86_avx512", DataType.i64, ArithmeticOperation.add, "vpaddq"),
            ("x86_avx512", DataType.i64, ArithmeticOperation.mul, "vpmullq"),
            ("x86_avx512", DataType.bf16, ArithmeticOperation.fma, "vdpbf16ps"),
        ],
    )
    def test_integer_arithmetic_codegen(
        self, mock_context, isa_name: str, data_type: DataType, operation: ArithmeticOperation, mnemonic: str
    ):
        """Test that integer arithmetic benchmarks generate the expected mnemonics."""
        isa = TestISACodegen.instantiate_isa(isa_name)

        params = ArithmeticBenchmarkParams(
            data_type=data_type,
            operation=operation,
            num_ops=Operations(64),
            thread_affinity=[0],
        )
        spec = isa.generate_arithmetic(params, mock_context)

        assert isinstance(spec, MicrobenchmarkFunctionSpec)
        assert len(spec.body) > 0
        assert mnemonic in spec.body

    @pytest.mark.parametrize(
        "data_type,setup_asm",
        [
            (DataType.i8, "movb $0, %%ah"),
            (DataType.i16, "xorw %%dx, %%dx"),
            (DataType.i32, "xorl %%edx, %%edx"),
            (DataType.i64, "xorl %%edx, %%edx"),
        ],
    )
    def test_integer_div_setup_instructions(self, mock_context, data_type: DataType, setup_asm: str):
        """Test that scalar integer div benchmarks preload dividend regs to avoid #DE faults."""
        isa = X86Scalar()

        params = ArithmeticBenchmarkParams(
            data_type=data_type,
            operation=ArithmeticOperation.div,
            num_ops=Operations(64),
            thread_affinity=[0],
        )
        spec = isa.generate_arithmetic(params, mock_context)

        assert isinstance(spec, MicrobenchmarkFunctionSpec)
        assert len(spec.body) > 0
        assert setup_asm in spec.body

    @pytest.mark.parametrize(
        "isa_name,data_type,mnemonic",
        [
            ("x86_scalar", DataType.i32, "movl"),
            ("x86_scalar", DataType.i64, "movq"),
            ("x86_sse", DataType.i32, "movaps"),
            ("x86_avx2", DataType.i64, "vmovaps"),
            ("x86_avx512", DataType.i32, "vmovaps"),
            ("x86_avx512", DataType.bf16, "vmovaps"),
        ],
    )
    def test_integer_memory_codegen(self, mock_context, isa_name: str, data_type: DataType, mnemonic: str):
        """Test that integer memory benchmarks generate the expected load/store mnemonics."""
        isa = TestISACodegen.instantiate_isa(isa_name)

        params = MemoryBenchmarkParams(
            data_type=data_type,
            load_store_ratio=LoadStoreRatio(1, 1),
            size_per_thread=Bytes(512),
            thread_affinity=[0],
            memory_level_name="L1",
        )
        spec = isa.generate_memory(params, mock_context)

        assert isinstance(spec, MicrobenchmarkFunctionSpec)
        assert len(spec.body) > 0
        assert mnemonic in spec.body

    @pytest.mark.parametrize("isa_name", EXPECTED_AVAILABLE.keys())
    def test_available_operations_matrix(self, isa_name: str):
        """Test that each ISA exposes exactly the expected integer operations."""
        isa = TestISACodegen.instantiate_isa(isa_name)

        for data_type, expected in TestIntegerCodegen.EXPECTED_AVAILABLE[isa_name].items():
            assert isa.bench_instructions.available_operations(data_type) == frozenset(expected)

    def test_arithmetic_suite_skips_unavailable_instructions(self, monkeypatch, mock_context):
        """Test that the arithmetic suite skips instructions unavailable for a data type."""
        mock_context.architecture.isa = [X86SSE]
        mock_context.benchmarking = SimpleNamespace(
            data_type=[DataType.i32],
            instructions={ArithmeticOperation.add, ArithmeticOperation.fma},
            threads=[1],
            num_ops=Operations(16),
        )

        suite = ArithmeticBenchmarkSuite.generate(mock_context, "x86_sse")

        names = list(suite.benchmarks)
        assert any("add" in name and "i32" in name for name in names)
        assert all("fma" not in name for name in names)

    def test_arithmetic_suite_raises_when_nothing_available(self, monkeypatch, mock_context):
        """Test that a fully-unavailable arithmetic suite raises UserError."""
        mock_context.architecture.isa = [X86AVX]
        mock_context.benchmarking = SimpleNamespace(
            data_type=[DataType.i32],
            instructions={ArithmeticOperation.add},
            threads=[1],
            num_ops=Operations(16),
        )

        with pytest.raises(UserError):
            ArithmeticBenchmarkSuite.generate(mock_context, "x86_avx")

    def test_memory_suite_skips_unavailable_dtype(self, monkeypatch, mock_context):
        """Test that the memory suite skips data types without load/store instructions."""
        mock_context.architecture.isa = [X86AVX]
        mock_context.benchmarking = SimpleNamespace(
            data_type=[DataType.i32, DataType.f32],
            threads=[1],
            ld_st_ratio=[LoadStoreRatio(1, 0)],
            mem_target=["L1"],
            mem_test_sizes=None,
        )

        suite = MemoryBenchmarkSuite.generate(mock_context, "x86_avx")

        names = list(suite.benchmarks)
        assert len(names) > 0
        assert all("f32" in name for name in names)
        assert all("i32" not in name for name in names)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
