from __future__ import annotations
import itertools
import os
import sys
from collections.abc import Generator

from .asm_comparison import compare_asm
from .classes import *
from .utils import *

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
from benchmark.generation.code_gen import *
from benchmark.generation.isa import BenchParamError

NUM_OP_TEST_VALUES = (0, 1, 32, 1024)
PRECISIONS = ("dp", "sp")

FLOP_COMBINATION_SET = {
    "-precision": PRECISIONS,
    "-test": ("FLOPS",),
    "-op": ("add", "mul", "div", "fma", "mad"),
    "-fp": NUM_OP_TEST_VALUES,
}

MEM_COMBINATION_SET = {
    "-precision": PRECISIONS,
    "-test": ("MEM",),
    "-num_LD": (0, 1, 2),
    "-num_ST": (0, 1, 2),
    "-num_rep": NUM_OP_TEST_VALUES,
}

BASIC_COMBINATION_SETS = (FLOP_COMBINATION_SET, MEM_COMBINATION_SET)

# All ISAs that don't require additional parameters
ISAS: "dict[str, tuple[dict[str, tuple]]]" = dict.fromkeys(
    ("riscvscalar", "scalar", "sse", "avx", "avx2", "avx512", "neon", "armscalar", "sve"), BASIC_COMBINATION_SETS
)

VLEN_TEST_VALUES = (1, 512, 1024)
LMUL_TEST_VALUES = (1, 2, 8)

RVV_ADDITIONAL_SET = {"-Vlen": VLEN_TEST_VALUES, "-LMUL": LMUL_TEST_VALUES}

RVV_COMBINATION_SETS = ({**FLOP_COMBINATION_SET, **RVV_ADDITIONAL_SET}, {**MEM_COMBINATION_SET, **RVV_ADDITIONAL_SET})

ISAS.update(dict.fromkeys(("rvv0.7", "rvv1.0"), RVV_COMBINATION_SETS))


def generate_tests() -> "Generator[Test, None, None]":
    for isa, comb_sets in ISAS.items():
        for param_set in comb_sets:
            try:
                param_set.values()
            except:
                print(comb_sets)
            for param_vals in itertools.product(*param_set.values()):
                # Pair argument names and parameter values and flatten them
                # arguments = [item for pair in zip(param_set.keys(), param_vals) for item in pair]
                arguments = {k: v for k, v in zip(param_set.keys(), param_vals)}
                yield Test(isa, arguments)


def compare_tests(test: Test, legacy_res: TestResult, new_res: TestResult) -> "tuple[bool, str]":
    "returns whether the test results match, call with legacy as self"

    # No problem if both tests fail
    if not legacy_res.success and not new_res.success:
        return True, "both tests failed"
    # If one fails and the other does not
    elif legacy_res.success != new_res.success:
        is_mem = test.get_type() == Test.Type.MEM
        if is_mem and test.args["-num_rep"] == 0 and legacy_res.success and not new_res.success:
            # Ignore the divergence in return values if num_rep is zero
            # The old generator does not fail, but given it is a redundant benchmark maybe it should
            return True, f"return value diverged, but num_rep = 0 exception ({get_file_line()})"

        return (
            False,
            f"Return code mismatch\n"
            f"    legacy: success = {legacy_res.success}, stdout = {legacy_res.info}"
            f"    new:    success = {new_res.success}, info = {new_res.info}",
        )
    # If both tests succeed, compare in more detail
    else:
        return compare_asm(test, legacy_res.asm, new_res.asm)


if __name__ == "__main__":
    # Change to base directory
    os.chdir(os.path.dirname(__file__) + "/../")

    # RESULTS_DIR = "refactor_tests/results"
    # os.makedirs(RESULTS_DIR, exist_ok=True)

    unimplemented = set()

    for test in generate_tests():
        binary = f"legacy_bench_gen/bench_{test.isa}"
        if not os.path.isfile(binary):
            print(f"[WARN] ISA binary not found: {binary}")
            continue

        try:
            new_results = test.run_new_benchgen()
        # Skip test if some aspect is yet to be implemented
        except NotImplementedError as e:
            unimplemented.add(str(e))
            continue
        # Test failed for another reason
        except BenchParamError as e:
            new_results = TestResult(False, str(e), None)

        legacy_results = test.run_legacy_benchgen()

        print(f"Test '{test}':")

        result, info = compare_tests(test, legacy_results, new_results)
        if result:
            print(f"    [OKAY]: {info}")
            continue
        else:
            print(f"    [ERROR]: {info}")
            input()

    print(f"The following tests were skipped due to missing implementations:\n    {unimplemented}")
