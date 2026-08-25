from __future__ import annotations

from enum import Enum, auto
from typing import Union, overload

from carm_roofline.core import ArithmeticOperation, DataType, MemoryOperation, Operation

from .register import CyclicRegisterSet

# Type alias for objects that can be converted to string
StrLike = Union[str, int, float]


def escape_for_inline_asm(s: str) -> str:
    """Escape '%' characters for inline assembly (gcc's __asm__ __volatile__).

    In inline assembly strings, '%' is special (used for operand substitution like %0, %1).
    To output a literal '%' register prefix, it must be escaped as '%%'.

    This function converts single '%' to '%%' for register names, but preserves
    GCC's special inline assembly syntax:
    - %=  (unique label suffix)
    - %(var) or %[var] (named input/output operands)
    - %0, %1, ... (positional operands)

    Args:
        s: String containing register names with '%' prefix

    Returns:
        String with '%' escaped to '%%' for inline assembly context, preserving special syntax

    Example:
        >>> escape_for_inline_asm("%rax")
        '%%rax'
        >>> escape_for_inline_asm("addq %rsi, %rax")
        'addq %%rsi, %%rax'
        >>> escape_for_inline_asm("loop%=:")
        'loop%=:'
        >>> escape_for_inline_asm("movq %(ptr), %rax")
        'movq %(ptr), %%rax'
    """
    import re

    # Match % followed by register name (letters/numbers, not special chars)
    # Negative lookahead to avoid matching:
    # - %= (unique label)
    # - %( (named operand start)
    # - %[ (named operand start, ARM style)
    # - %\d (positional operand like %0, %1)
    return re.sub(r"%(?![=\(\[\d])", r"%%", s)


class _Instruction:
    """Base class for instruction format patterns used in benchmark code generation.

    Subclasses define specific instruction types (load, branch, arithmetic, etc.) and
    validate that required fields are present in the pattern string.
    """

    def __init__(self, pattern: str):
        missing_fields = []
        for field in self.required_fields():
            if field not in pattern:
                missing_fields.append(field)

        num_missing = len(missing_fields)
        if num_missing > 0:
            plural = num_missing > 1
            raise ValueError(
                f"Pattern '{pattern}' is not a valid {type(self).__name__}. Missing field{'s' if plural else ''}: "
                f"{'[ ' if plural else ''}{', '.join(missing_fields)}{' ]' if plural else ''}"
            )

        self.pattern = pattern

    def required_fields(self) -> tuple[str, ...]:
        """Return a tuple of required field names (e.g., '{dst}', '{imm}')"""
        raise NotImplementedError(
            "Subclasses must implement required_fields() method to specify required pattern fields."
        )


class LoadImm(_Instruction):
    """Load immediate value into destination register.

    Each element returned by :meth:`fmt` contains one assembly line.

    Required fields: {dst} (destination register), {imm} (immediate value)
    """

    def required_fields(self) -> tuple[str, ...]:
        return ("{dst}", "{imm}")

    def fmt(self, dst: str, imm: StrLike) -> list[str]:
        formatted = self.pattern.format(dst=dst, imm=imm)
        return [escape_for_inline_asm(formatted)]


class LoadWord(_Instruction):
    """Load word (4 bytes) from memory.

    Required fields: {dst} (destination register), {ptr} (memory pointer)
    """

    def required_fields(self) -> tuple[str, ...]:
        return ("{dst}", "{ptr}")

    def fmt(self, dst: str, ptr: str) -> str:
        formatted = self.pattern.format(dst=dst, ptr=ptr)
        return escape_for_inline_asm(formatted)


class BranchNotZero(_Instruction):
    """Conditional branch when register is not zero or greater than zero.

    Required fields: {tgt} (jump target label)
    Optional fields: {src} (source register to test)

    Some ISAs (e.g., x86) don't require {src} and test based on flags instead.
    """

    def required_fields(self) -> tuple[str, ...]:
        return ("{tgt}",)

    def fmt(self, src: str, tgt: str) -> str:
        formatted = self.pattern.format(src=src, tgt=tgt) if "{src}" in self.pattern else self.pattern.format(tgt=tgt)

        return escape_for_inline_asm(formatted)


class AddImm(_Instruction):
    """Add immediate value to a register (typically for pointer increment).

    Required fields: {src} (source register), {imm} (immediate value)
    Optional fields: {dst} (destination register)

    Some ISAs (e.g., x86) write result back to {src} and don't need {dst}.
    """

    def required_fields(self) -> tuple[str, ...]:
        return ("{src}", "{imm}")

    def fmt(self, dst: str, src: str, imm: StrLike) -> str:
        formatted = self.pattern.format_map({"dst": dst, "src": src, "imm": imm})
        return escape_for_inline_asm(formatted)


class SubImm(AddImm):
    """Subtract immediate value from a register (typically for counter decrement).

    Required fields: {src} (source register), {imm} (immediate value)
    Optional fields: {dst} (destination register)

    Some ISAs (e.g., x86) write result back to {src} and don't need {dst}.
    Alternatively, ISAs supporting negative immediates can reuse AddImm with negated {imm}.
    """


class Add(_Instruction):
    """Add two registers together (typically for pointer increment).

    Required fields: {dst} (destination register)
    Optional fields: {src1} (first source), {src2} (second source / increment register)

    3-operand ISAs (ARM, RISC-V) use all three: {dst}, {src1}, {src2}
    2-operand ISAs (x86 AT&T) use {src2} and {dst}: "addq {src2}, {dst}" means dst += src2
    """

    def required_fields(self) -> tuple[str, ...]:
        return ("{dst}",)

    def fmt(self, dst: str, src1: str, src2: str) -> str:
        formatted = self.pattern.format(dst=dst, src1=src1, src2=src2)
        return escape_for_inline_asm(formatted)


class ControlInstructions:
    """Collection of instruction formats for benchmark loop control.

    Includes pointer increment, conditional branches, and immediate loads needed
    for outer loop orchestration.
    """

    load_imm: LoadImm
    load_word: LoadWord
    branch_nz: BranchNotZero
    add_imm: AddImm
    sub_imm: SubImm
    add: Add


class Arithmetic(_Instruction):
    """Arithmetic/compute instruction (FMA, ADD, MUL, DIV, etc.) for benchmarking.

    These are the actual operations measured in FLOPs, not control instructions.
    Pattern uses {} as a placeholder that gets replaced with a register.
    """

    def __init__(self, pattern: str, register_format: str | None = None):
        super().__init__(pattern)
        self.register_format = register_format

    def required_fields(self) -> tuple[str, ...]:
        return ()

    def fmt(self, bench_regs: CyclicRegisterSet) -> str:
        bench_reg = bench_regs.get(self.register_format)
        formatted = self.pattern.replace("{}", bench_reg)
        return escape_for_inline_asm(formatted)


class Memory(_Instruction):
    """Memory load/store instruction for benchmarking.

    Supports two addressing modes: pointer with offset or pointer only.
    """

    def required_fields(self) -> tuple[str, ...]:
        return ("{ptr}", "{reg}")

    class AddressingMode(Enum):
        """Addressing mode for memory access: with or without offset."""

        ptr_offset = auto()
        ptr_only = auto()

    def __init__(self, pattern: str):
        """Initialize memory instruction with addressing mode detection.

        Required fields: {ptr} (memory pointer), {reg} (register operand)
        Optional fields: {off} (byte offset)

        Addressing mode is auto-detected:
        - If {off} present: ptr_offset mode (e.g., "movq {off}({ptr}), {reg}")
        - Otherwise: ptr_only mode (e.g., "ldr {reg}, [{ptr}]")
        """
        super().__init__(pattern)

        if "{off}" in pattern:
            self.type = Memory.AddressingMode.ptr_offset
        else:
            self.type = Memory.AddressingMode.ptr_only

    def fmt(self, bench_regs: CyclicRegisterSet, ptr_reg: str, offset: int | None = None) -> str:
        """Format memory instruction with register and pointer.

        Args:
            bench_regs: Register set to allocate from
            ptr_reg: Pointer register name
            offset: Byte offset (required only for ptr_offset addressing mode)

        Returns:
            Formatted instruction string ready for assembly

        Raises:
            TypeError: If offset is None but instruction requires it (ptr_offset mode)
        """
        bench_reg = bench_regs.get()

        if self.type == Memory.AddressingMode.ptr_offset:
            if offset is None:
                raise TypeError(
                    f"Missing argument: 'offset'. "
                    f"Memory instruction uses {Memory.AddressingMode.ptr_offset.name} (has '{{off}}' field), "
                    f"thus requiring an offset."
                )
            formatted = self.pattern.format(reg=bench_reg, ptr=ptr_reg, off=offset)
        else:
            formatted = self.pattern.format(reg=bench_reg, ptr=ptr_reg)

        return escape_for_inline_asm(formatted)


class TypedInstructions:
    """Collection of instruction formats organized by data type and operation.

    Enables efficient lookup of the right instruction variant (e.g., fadd.s vs fadd.d,
    or flw vs fld for different precisions).
    """

    def __init__(self, formats: dict[DataType, dict[Operation, str | _Instruction]]):
        """Initialize typed instruction formats.

        The available types and operations are defined in the `DataType` and `Operation` enums.
        Instruction format strings are automatically converted to appropriate _Instruction subclasses.

        Args:
            formats: Nested dict with structure `{data_type: {operation: format_string_or_object}}`

        Example:
            ```
            inst_fmts = TypedInstructions({
                DataType.f32: {
                    ArithmeticOperation.add: "fadd.s {}, {}, {}",
                    ArithmeticOperation.mul: "fmul.s {}, {}, {}",
                    MemoryOperation.ld:  "flw {dst}, {off}({ptr})",
                    MemoryOperation.st:  "fsw {dst}, {off}({ptr})",
                },
                DataType.f64: {
                    ArithmeticOperation.add: "fadd.d {}, {}, {}",
                    MemoryOperation.ld:  "fld {dst}, {off}({ptr})",
                }
            })
            ```
        """

        typed_instructions: dict[DataType, dict[Operation, _Instruction]] = {}

        for data_type, inst_formats in formats.items():
            DataType.check_validity(data_type)
            operation_to_format = self._process_data_type_instructions(data_type, inst_formats)
            typed_instructions[data_type] = operation_to_format

        self.formats = typed_instructions

    def _process_data_type_instructions(
        self, data_type: DataType, inst_formats: dict[Operation, str | _Instruction]
    ) -> dict[Operation, _Instruction]:
        "Process and validate all instructions for a single data type."
        operation_to_format: dict[Operation, _Instruction] = {}
        mem_mode_tracker = MemoryModeTracker()

        for op, format_spec in inst_formats.items():
            if not isinstance(op, (MemoryOperation, ArithmeticOperation)):
                raise TypeError(
                    f"Object of type '{type(op).__name__}' is not a valid operation. "
                    f"Must be a variant of ArithmeticOperation or MemoryOperation"
                )
            instruction = self._convert_to_instruction(op, format_spec)
            operation_to_format[op] = instruction

            # Register memory instructions for tracking
            if isinstance(op, MemoryOperation):
                mem_mode_tracker.register_memory_instruction(op, instruction)

        return operation_to_format

    def _convert_to_instruction(self, op: Operation, format_spec: str | _Instruction) -> _Instruction:
        "Convert a format specification to an _Instruction object."
        if isinstance(format_spec, str):
            return self._create_instruction_from_string(op, format_spec)
        elif isinstance(format_spec, _Instruction):
            return format_spec
        else:
            raise TypeError(
                f"Type '{type(format_spec).__name__}' is not a valid instruction format. "
                f"Must be a '{str.__name__}' or '{_Instruction.__name__}'"
            )

    def _create_instruction_from_string(self, op: Operation, format_string: str) -> _Instruction:
        "Create an appropriate _Instruction subclass from a format string."
        if isinstance(op, ArithmeticOperation):
            return Arithmetic(format_string)
        elif isinstance(op, MemoryOperation):
            return Memory(format_string)
        else:
            # Should be unreachable as long as there is no issue with the Operation enum
            raise ValueError(f"Unknown operation category for operation: {op}")

    @overload
    def get(self, data_type: DataType, operation: ArithmeticOperation) -> Arithmetic: ...

    @overload
    def get(self, data_type: DataType, operation: MemoryOperation) -> Memory: ...

    def get(self, data_type: DataType, operation: Operation) -> Arithmetic | Memory:
        """Type-safe access to instruction format for a given data type and operation.

        Narrows instruction type based on operation category.

        Args:
            data_type: The data type (f32, f64, etc.)
            operation: The operation (add, mul, ld, st, etc.)

        Returns:
            Arithmetic instruction for arithmetic operations, Memory instruction for memory operations
        """
        instruction = self.formats[data_type][operation]
        assert isinstance(instruction, (Arithmetic, Memory))
        return instruction

    def available_operations(self, data_type: DataType) -> frozenset[Operation]:
        """Operations with an instruction format for this data type (empty set if unsupported)."""
        return frozenset(self.formats.get(data_type, {}))

    def __getitem__(self, arg: DataType) -> dict[Operation, _Instruction]:
        return self.formats[arg]


class MemoryModeTracker:
    """Tracks and validates memory instruction addressing modes for consistency.

    Ensures that all memory instructions within a data type use the same addressing mode
    (either all ptr_offset or all ptr_only, not mixed).
    """

    def __init__(self) -> None:
        """Initialize the memory mode tracker."""
        self._seen_modes: dict[Memory.AddressingMode, bool] = {
            Memory.AddressingMode.ptr_offset: False,
            Memory.AddressingMode.ptr_only: False,
        }

    def register_memory_instruction(self, op: Operation, instruction: _Instruction) -> None:
        "Register a memory instruction and validate addressing mode consistency."
        if not isinstance(instruction, Memory):
            raise TypeError(
                f"Instruction format for memory operation '{op}' must be of type "
                f"'{Memory.__name__}'. Got type '{type(instruction).__name__}' instead."
            )

        self._seen_modes[instruction.type] = True

        if self._has_conflicting_modes():
            raise ValueError(
                "Cannot mix memory instructions with and without pointer offset within a data type. "
                "All memory instructions must be consistent in addressing mode."
            )

    def _has_conflicting_modes(self) -> bool:
        return sum(1 for seen in self._seen_modes.values() if seen) > 1
