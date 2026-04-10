from __future__ import annotations
import math
import re
from dataclasses import dataclass

from refactor_tests.classes import *

from .utils import *


def comparison_str(legacy: str, new: str) -> str:
    legacy = "\n".join([f"    {l}" for l in legacy.splitlines()])
    new = "\n".join([f"    {l}" for l in new.splitlines()])
    return f"legacy:\n{legacy}\nnew:\n{new}"


@dataclass
class ParsedLoops:
    sequence: "dict[str, int]"
    "Maps instructions to how many there are per rep"
    iterations: int
    inner_insts: int
    outer_insts: int
    ptr_increment: int
    register_format: str

    def total_insts(self) -> int:
        return self.iterations * self.inner_insts + self.outer_insts

    def __inst_seq_str(self):
        return ", ".join([f"{k} x{v}" for k, v in self.sequence.items()])

    def deep_comparison(self, legacy: "ParsedLoops", test: Test) -> "tuple[bool, str]":
        mad_exception = test.args.get("-op") == "mad"
        arm_neon_exception = test.isa == "neon"

        if not mad_exception and self.total_insts() != legacy.total_insts():
            return (
                False,
                f"assembly does not match, different number of instructions:\n"
                f"is {self.total_insts()}, should be {legacy.total_insts()}\n",
            )

        if self.ptr_increment != legacy.ptr_increment:
            return (
                False,
                f"assembly does not match, different pointer increments:\n"
                f"is {self.ptr_increment}, should be {legacy.ptr_increment}\n",
            )

        if mad_exception:
            if any(k not in legacy.sequence.keys() for k in self.sequence.keys()):
                return (
                    False,
                    f"assembly does not match, 'mad' exception new instruction is not in legacy set:\n"
                    f"new: {self.__inst_seq_str()}, legacy: {legacy.__inst_seq_str()}\n",
                )
        else:
            if self.sequence != legacy.sequence:
                return (
                    False,
                    f"assembly does not match, different instruction sequence:\n"
                    f"is {self.__inst_seq_str()}, should be {legacy.__inst_seq_str()}\n",
                )

        if not arm_neon_exception and self.register_format.lower() != legacy.register_format.lower():
            return (
                False,
                f"assembly does not match, different registers in use:\n"
                f"is {self.register_format}, should be {legacy.register_format}\n",
            )

        # The mad operation generates interleaved muls and adds in the legacy generator, but only for x86.
        #  Unsure if this is meant to stay as it is.
        additional_info = (
            f"Instruction count and sequence tests were skipped due to 'mad' exception ({get_file_line()})"
            if mad_exception
            else f"Register format test was skipped due to ARM NEON exception ({get_file_line()})"
            if arm_neon_exception
            else ""
        )

        if (
            self.iterations == legacy.iterations
            and self.inner_insts == legacy.inner_insts
            and self.outer_insts == legacy.outer_insts
        ):
            return True, f"assembly matches completely. {additional_info}"
        else:
            return (
                True,
                f"assembly matches, loop structure is different (is {self}, should be {legacy})"
                f" but same instruction count. {additional_info}",
            )

    def __str__(self):
        return f"{self.iterations} * {self.inner_insts} + {self.outer_insts}"

    def __eq__(self, value: "ParsedLoops"):
        return (
            self.iterations == value.iterations
            and self.inner_insts == value.inner_insts
            and self.outer_insts == value.outer_insts
        )


class ISAPatterns:
    def __init__(self):
        # Pattern type, but initialize with str
        self.outer_setup: str
        self.outer_load_ptr: str
        self.outer_iter: str
        self.inner_load_iters: str
        "Captures the number of iterations"
        self.inner_inc_iter: str
        self.inner_inc_ptr: str
        self.branch: str
        self.arithmetic: str
        self.memory: str
        self.memory_has_offset: bool = True
        "Captures the instruction name in group 1"
        self.opcode: str
        "Captures the opcode"
        self.register: str
        "Captures everything in the register but the number (i.g. (%%xmm)0 )"
        self.address: str = r" (\d+)\(%*\w+\)"
        "Caputures the pointer offset"

    def opcode_pattern(self, instructions: str) -> "dict[str, int]":
        "Returns {opcode: count, ...}, where the count is minimized (greatest common factor)"
        pattern = re.compile(self.opcode, re.MULTILINE)
        instructions: list[str] = pattern.findall(instructions)

        counts: dict[str, int] = {}
        for i in instructions:
            counts[i] = counts.get(i, 0) + 1

        gcd = math.gcd(*counts.values())

        for k in counts.keys():
            counts[k] //= gcd

        return counts

    def parse_loops(self, asm: str, test_type: Test.Type) -> ParsedLoops:
        def get_ptr_increment(match: "re.Match[str]", insts: str) -> int:
            # try finding an explicit pointer increment at the start
            try:
                inc = match.group("ptr_inc")
                return int(inc)
            except (IndexError, TypeError):
                pass
            # look for immediate offsets instead
            try:
                offsets = re.findall(self.address, insts)
                if len(offsets) == 0:
                    raise ValueError(
                        f"Could not parse memory offset. Pattern: '{self.address}', string: {insts[0:100]}"
                    )

                inc = int(offsets[1]) - int(offsets[0])
                return inc
            except IndexError:
                return 0

        def get_register(insts: str) -> str:
            reg = re.search(self.register, insts)
            if not reg:
                raise ValueError(f"Could not parse register. Pattern: '{self.register}', string: {insts[0:100]}")

            return reg.group(0)

        LOOP_LABEL = r"\w+%=:"
        NL = r"\s"

        instance_pointer = test_type.is_a(Test.Type.MEM) or test_type.is_a(Test.Type.MIXED)
        increment_pointer = instance_pointer and self.memory_has_offset

        instruction_pattern = self.memory if test_type.is_a(Test.Type.MEM) else self.arithmetic

        # pattern for a loop with an inner loop
        with_inner_pattern = re.compile(
            self.outer_setup
            + NL
            + LOOP_LABEL
            + NL
            + (self.outer_load_ptr + NL if instance_pointer else "")
            + self.inner_load_iters
            + NL
            + LOOP_LABEL
            + NL
            + instruction_pattern.replace("<name>", "<inner>")
            + (self.inner_inc_ptr + NL if increment_pointer else "")
            + self.inner_inc_iter
            + NL
            + self.branch
            + NL
            + instruction_pattern.replace("<name>", "<outer>")
            + "?"  # there may be no instructions in the outer loop
            + self.outer_iter
            + NL
            + self.branch
        )

        # code has an inner and outer loop
        if with_inner_match := with_inner_pattern.match(asm):
            iterations = int(with_inner_match.group("iters"))
            inner = with_inner_match.group("inner")
            outer = with_inner_match.group("outer") or ""
            n_inner = len(inner.splitlines())
            n_outer = len(outer.splitlines())
            if test_type.is_a(Test.Type.MEM):
                ptr_inc = get_ptr_increment(with_inner_match, inner)
            else:
                ptr_inc = 0

            reg = get_register(inner)

            return ParsedLoops(self.opcode_pattern(inner), iterations, n_inner, n_outer, ptr_inc, reg)

        # pattern for a loop with an outer loop only, no inner loop
        outer_only_pattern = re.compile(
            self.outer_setup
            + NL
            + LOOP_LABEL
            + NL
            + (self.outer_load_ptr + NL if instance_pointer else "")
            + instruction_pattern.replace("<name>", "<outer>")
            + self.outer_iter
            + NL
            + self.branch
        )
        # code only has an outer loop, no inner loop
        if outer_only_match := outer_only_pattern.match(asm):
            outer = outer_only_match.group("outer")
            n_outer = len(outer.splitlines())
            if test_type.is_a(Test.Type.MEM):
                ptr_inc = get_ptr_increment(outer_only_match, outer)
            else:
                ptr_inc = 0
            reg = get_register(outer)

            return ParsedLoops(self.opcode_pattern(outer), 0, 0, n_outer, ptr_inc, reg)

        print(asm)
        print(with_inner_pattern.pattern)
        print(outer_only_pattern.pattern)
        raise ValueError("Could not parse loops, no match")


class RISCVScalarPatterns(ISAPatterns):
    def __init__(self):
        super().__init__()
        self.outer_setup = r"l[dw] \w+, %(?:0|\(\w+\))"
        self.outer_load_ptr = r"l[dw] \w+, %(?:1|\(\w+\))"
        self.outer_iter = r"addi \w+, \w+, -1"
        self.inner_load_iters = r"li \w+, (?P<iters>\d+)"
        self.inner_inc_iter = r"addi \w+, \w+, -\d+"
        self.inner_inc_ptr = r"addi \w+, \w+, \d+"
        self.branch = r"bgtz \w+, \w+%="
        self.arithmetic = r"(?P<name>(?:[\w\.]+ [\w\d\(\), ]+\s)+?)"
        self.memory = self.arithmetic
        self.opcode = r"^([\w\.]+)\s"
        self.register = r"([fx])\d+"


class RVV0_7Patterns(RISCVScalarPatterns):
    def __init__(self):
        super().__init__()
        # pointer increment is optional (mem only)
        li_ptr_inc = r"(?:li \w+, (?P<ptr_inc>\d+)\s)?"
        li_vlen = r"li \w+, \d+\s"
        vsetvli = r"vsetvli \w+, \w+, e\d+, m\d\s"
        self.outer_setup = li_ptr_inc + li_vlen + vsetvli + r"l[dw] \w+, %(?:0|\(\w+\))"
        self.arithmetic = r"(?P<name>(?:[\w\.]+ [\w\d\(\), ]+\s)+?)"
        self.memory = r"(?P<name>(?:v[ls]e.v [\w\d\(\), ]+\sadd \w+, \w+, \w+\s)+?)"
        self.memory_has_offset = False
        self.register = r"(v)\d+"


class RVV1_0Patterns(RVV0_7Patterns):
    def __init__(self):
        super().__init__()
        self.memory = r"(?P<name>(?:v[ls]e(?:32|64).v [\w\d\(\), ]+\sadd \w+, \w+, \w+\s)+?)"


class X86Patterns(ISAPatterns):
    def __init__(self):
        super().__init__()
        # optional field is the memory instructions that get added for the div test in legacy x86
        self.outer_setup = r"mov[ql]? (?:%0|\(%\(\w+\)\)), %%\w+(?:\smovq %1, %%rax\sv?mov\w+ \(%%rax\), %%[xyz]mm0)?"
        self.outer_load_ptr = r"movq (?:%1|\(%\(\w+\)\)), %%\w+"
        self.outer_iter = r"subq? \$1, %%\w+"
        self.inner_load_iters = r"mov[ql]? \$(?P<iters>\d+), %%\w+"
        self.inner_inc_iter = r"sub[ql]? \$1, %%\w+"
        self.inner_inc_ptr = r"add[q]? \$\d+, %%\w+"
        self.branch = r"jnz \w+%="
        self.arithmetic = r"(?P<name>(?:\w+ [%\w, ]+\s)+?)"
        self.memory = r"(?P<name>(?:v?mov\w+ (?:\d+\(%%\w+\), %%\w+|%%\w+, \d+\(%%\w+\))\s)+?)"
        self.opcode = r"^([\w\.]+)\s"
        self.register = r"(%%[xyz]mm)\d+"


class ArmScalarPatterns(ISAPatterns):
    def __init__(self):
        super().__init__()
        self.outer_setup = r"(?:ldr \w+, %\[\w+\]|mov w0, %w0)"
        self.outer_load_ptr = r"(?:ldr \w+, %\[\w+\]|mov x3, %1)"
        self.outer_iter = r"sub \w+, \w+, 1"
        self.inner_load_iters = r"mov \w+, (?P<iters>\d+)"
        self.inner_inc_iter = r"sub \w+, \w+, 1"
        self.inner_inc_ptr = r"add \w+, \w+, #?\d+"
        self.branch = r"cbnz \w+, \w+%="
        self.arithmetic = r"(?P<name>(?:\w+ [\w#\d\[\], ]+\s)+?)"
        self.memory = self.arithmetic
        self.opcode = r"^([\w\.]+)\s"
        self.register = r"([dsxw])\d+"
        self.address = r" \[\w+, #(\d+)\]"


class ArmNeonPatterns(ArmScalarPatterns):
    def __init__(self):
        super().__init__()
        self.arithmetic = r"(?P<name>(?:\w+ [\w\d., ]+\s)+?)"
        self.memory = r"(?P<name>(?:(?:ld|st)r \w\d+, \[\w\d+, #\d+\]\s)+?)"
        self.register = r"([vV])\d+\.\d\w|\w\d+"


ISA_PATTERNS_MAP: "dict[str, ISAPatterns]" = {
    "riscvscalar": RISCVScalarPatterns(),
    "rvv0.7": RVV0_7Patterns(),
    "rvv1.0": RVV1_0Patterns(),
    "scalar": X86Patterns(),
    "sse": X86Patterns(),
    "avx": X86Patterns(),
    "avx2": X86Patterns(),
    "avx512": X86Patterns(),
    "armscalar": ArmScalarPatterns(),
    "neon": ArmNeonPatterns(),
    "sve": ArmNeonPatterns(),
}


def compare_asm(test: Test, legacy: ParsedBench, new: ParsedBench) -> "tuple[bool, str]":
    def defines_to_dict(bench: ParsedBench):
        pattern = re.compile(r"^#define (\w+) (\w+)", re.MULTILINE)

        return {m.group(1): m.group(2) for m in pattern.finditer(bench.defines)}

    legacy_defines = defines_to_dict(legacy)
    new_defines = defines_to_dict(new)

    def compare_defines(blacklist: list[str] = []) -> "tuple[bool, str]":
        for k, lv in legacy_defines.items():
            if k in blacklist:
                continue

            if nv := new_defines.get(k, None):
                if nv != lv:
                    return False, f"#define directives do not match, constant '{k}' is '{nv}', should be '{lv}'"
                else:
                    return True, ""
            else:
                return False, f"#define directives do not match, constant '{k}' is missing from new benchmark"

    # div exception, legacy generator introduces (seemingly redundant) mem instructions and defines when op = div
    if test.args.get("-op", None) == "div" and test.get_type().name == Test.Type.FLOPS.name:
        blacklist = ["MEM", "DIV", "NUM_LD", "NUM_ST", "OPS", "NUM_REP", "PRECISION", "ALIGN"]
        okay, info = compare_defines(blacklist=blacklist)
        if okay:
            # When the legacy generator makes a FLOP test with the div operation, it generates one memory
            # operation to load xmm0. However, this register is the destination register, so it does not seem necessary
            print(f"   [WARN] #define directives differ partialy, but -op div exception {get_file_line()}")
        else:
            return okay, info
    # normal comparison of the defines
    else:
        okay, info = compare_defines()
        if not okay:
            return okay, info

    # for whatever reason sse addpd inserts ; instead of \n\t\t
    #  captures any assembly, discarding the quotes, newlines and ;
    asm_stripping_pattern = re.compile(r'"(.+)(?:\\n(?:\\t)+|;)"')
    wspace_or_empty = re.compile(r"^\s*$")

    # Strip the inline assembly of whitespace, quotations, \n, etc.
    def iasm_strip(s: str) -> str:
        out = []
        # discard whitespace or empty lines
        out = [line for line in s.splitlines() if not wspace_or_empty.match(line)]
        out = [asm_stripping_pattern.search(line).group(1) for line in out]
        return "\n".join(out)

    legacy_asm = iasm_strip(legacy.asm)
    new_asm = iasm_strip(new.asm)

    isa_pattern = ISA_PATTERNS_MAP[test.isa]
    test_type = test.get_type()
    legacy_loop = isa_pattern.parse_loops(legacy_asm, test_type)
    new_loop = isa_pattern.parse_loops(new_asm, test_type)

    okay, info = new_loop.deep_comparison(legacy_loop, test)

    if not okay:
        return okay, info

    # # Compare the asm outputs, inputs, clobbers TODO
    # for attr in ("outputs", "inputs", "clobbers"):
    #     legacy_attr_val = getattr(legacy, attr).strip()
    #     new_attr_val    = getattr(new, attr).strip()

    #     if legacy_attr_val != new_attr_val:
    #         return (
    #             False,
    #             f"{attr} do not match:\n"
    #             + comparison_str(repr(legacy_attr_val), repr(new_attr_val))
    #         )

    return True, info
