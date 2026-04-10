from __future__ import annotations
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum, auto

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
from benchmark.generation import arm, riscv, x86
from benchmark.generation.code_gen import *
from benchmark.generation.isa import BaseISA


@dataclass
class ParsedBench:
    # NOTE: attribute order is important for ParsedBench.parse() to work!
    defines: str
    asm: str
    outputs: str
    inputs: str
    clobbers: str

    def parse(benchmark: str) -> "ParsedBench":
        """
        Parses a benchmark `str` into a `ParsedAsm`.

        The order of the class attributes must stay for the parsing to work.
        """
        pattern = r"(^.+)static.+__volatile__ \((.+):(.+):(.+):(.+).\);"

        groups = re.compile(pattern, flags=re.DOTALL).match(benchmark).groups()

        return ParsedBench(*groups)


@dataclass
class TestResult:
    """The result of a single test"""

    __test__ = False  # to prevent pytest from collecting this class
    success: bool
    info: str
    asm: ParsedBench | None


@dataclass
class Test:
    """A single test, a set of parameters and ISA"""

    __test__ = False  # to prevent pytest from collecting this class
    isa: str
    args: "dict[str]"

    class Type(Enum):
        MEM = auto()
        FLOPS = auto()
        MIXED = auto()

        def is_a(self, cmp: "Test.Type") -> bool:
            return self.value == cmp.value

    _TYPE_MAP = {"MEM": Type.MEM, "FLOPS": Type.FLOPS, "MIXED": Type.MIXED}

    def get_type(self) -> "Test.Type":
        return self._TYPE_MAP[self.args["-test"]]

    def get_legacy_args(self) -> "list[str]":
        binary = f"legacy_bench_gen/bench_{self.isa}"
        return [binary, *[str(item) for k, v in self.args.items() for item in (k, v)]]

    def __str__(self):
        return " ".join(self.get_legacy_args())

    def run_legacy_benchgen(self) -> "TestResult":
        "runs the legacy benchmark generator and returns the results"

        proc = subprocess.run(self.get_legacy_args(), capture_output=True, text=True, timeout=10)

        with open("Test/test_params.h") as f:
            ubench_code = f.read()

        parsed_code = ParsedBench.parse(ubench_code)
        return TestResult(proc.returncode == 0, proc.stdout + proc.stderr, parsed_code)

    def run_new_benchgen(self) -> "TestResult":
        "runs the new benchmark generator and returns the results"
        isa_to_class: dict[str, type] = {
            "riscvscalar": riscv.RISCVScalar,
            "rvv0.7": riscv.RISCV_RVV_071,
            "rvv1.0": riscv.RISCV_RVV,
            "scalar": x86.X86Scalar,
            "sse": x86.X86SSE,
            "avx": x86.X86AVX,
            "avx2": x86.X86AVX2,
            "avx512": x86.X86AVX512,
            "armscalar": arm.ArmScalar,
            "neon": arm.ArmNeon,
        }
        precision_to_datatype = {"sp": DataType.f32, "dp": DataType.f64}

        try:
            isa_class = isa_to_class[self.isa]
        except KeyError:
            raise NotImplementedError(f"ISA {self.isa}")

        test_type = self.args["-test"]
        data_type = precision_to_datatype[self.args["-precision"]]

        # TODO: Add condition to pass vlen, lmul args if RVV
        if self.isa in ("rvv0.7", "rvv1.0"):
            isa_instance: riscv.RISCV_RVV_071 = isa_class(self.args["-Vlen"], self.args["-LMUL"])
        else:
            isa_instance: BaseISA = isa_class()

        if test_type == "MEM":
            bench_code = isa_instance.generate_memory(
                data_type, self.args["-num_LD"], self.args["-num_ST"], self.args["-num_rep"]
            )
        elif test_type == "FLOPS":
            operation = Operation[self.args["-op"]]
            bench_code = isa_instance.generate_arithmetic(data_type, operation, self.args["-fp"])
        else:  # mixed
            raise NotImplementedError("Mixed test")

        with open("Test/test_new.h", "w") as f:
            f.write(bench_code)

        return TestResult(True, "", ParsedBench.parse(bench_code))
