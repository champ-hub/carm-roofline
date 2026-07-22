from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from carm_roofline.benchmark.generation.code_gen import ControlInstructions, TypedInstructions, instruction as inst
from carm_roofline.benchmark.generation.code_gen.register import CyclicRegisterSet, HelperRegisterSet, TypedRegisterSets
from carm_roofline.benchmark.generation.parameters import (
    ArithmeticBenchmarkParams,
    BenchParamError,
    MemoryBenchmarkParams,
    MemoryLayoutMode,
)
from carm_roofline.core import ArithmeticOperation, Bytes, DataType, MemoryOperation
from carm_roofline.output_utils import debug

if TYPE_CHECKING:
    from carm_roofline.architecture import Architecture
    from carm_roofline.context import CARMContext
    from carm_roofline.test_bench.builder import MicrobenchmarkFunctionSpec


@dataclass
class LoopSplitConfig:
    """Configuration for splitting operations into inner/outer loops."""

    instance_inner_loop: bool
    inner_repeats: int
    outer_repeats: int
    num_iterations: int


@dataclass
class SizeInfo:
    """Derived sizes for memory benchmark generation."""

    repeats: int
    bytes_per_repeat: int
    actual_working_set_size: int
    read_array_size: int
    "Total bytes needed for the read (load) buffer: repeats * num_ld * bytes_per_inst"
    write_array_size: int
    "Total bytes needed for the write (store) buffer: repeats * num_st * bytes_per_inst"


@dataclass
class LoopConfig:
    """Loop configuration for memory benchmark generation."""

    block_size_offsets: int
    bytes_per_block: int
    mem_insts_per_loop: int
    max_loop_size: int
    num_iterations: int
    instance_inner_loop: bool
    inner_repeats: int
    outer_repeats: int
    loop_instruction_limit: int


@dataclass
class InlineASM:
    @dataclass
    class Input:
        c_name: str
        asm_name: str

    asm: list[str]
    inputs: list[Input]
    clobbers: Iterable[str]

    def format(self) -> str:
        indent = " " * 4

        # Assembly with quotes, \n\t
        processed_asm = "".join(f'{indent * 2}"{s}\\n\\t"\n' for s in self.asm)
        inputs = ", ".join(f'[{v.asm_name}] "m" ({v.c_name})' for v in self.inputs)
        clobbers = ", ".join(f'"{r}"' for r in self.clobbers)

        return (
            f"{indent}__asm__ __volatile__ (\n"
            f"{processed_asm}"
            # Outputs
            f"{indent * 2}:\n"
            # Input variables
            f"{indent * 2}: {inputs}\n"
            # Clobbers
            f"{indent * 2}: {clobbers}\n"
            f"{indent});\n"
        )

    def as_function_body(self) -> str:
        """Format inline assembly as the body of a static function."""
        return self.format()


def _add_if(condition: bool, *asm: str) -> list[str]:
    return list(asm) if condition else []


class BaseISA:
    INNER_LOOP_LABEL = "inner_loop%="
    OUTER_LOOP_LABEL = "outer_loop%="
    name: str
    "Identifier name, should contain only alphanumeric characters, e.g., 'riscv_rvv', 'x86_avx2'"
    family: str
    "ISA family name, e.g., 'riscv', 'x86'. ISAs within the same family should be generally compatible."
    max_branch_insts: int
    "Maximum branch distance in number of instructions"
    max_mem_offset_bytes: int
    "Maximum memory offset of memory instructions, in bytes"
    # TODO: deprecate this?
    max_immediate: int
    "Maximum value for an immediate value, for the purposes of pointer incrementation"
    unroll_loop: bool = False
    """Whether to unroll loops in generated benchmarks, instancing multiple pointer increments per loop iteration.
    Only affects the memory benchmark."""
    # Fits in a Gracemont uop cache (512 entries)
    instruction_limit: int = 512
    "Maximum number of instructions in a benchmark (inner + outer loops combined)"

    bench_instructions: TypedInstructions
    control_instructions: ControlInstructions
    "Formats for control instructions (branches, pointer arithmetic, etc.)"
    helper_registers: HelperRegisterSet
    "Registers for iterators, pointers, etc. (immutable, shared across instances)"

    @classmethod
    def from_architecture(cls, arch: Architecture) -> BaseISA:
        """Create an ISA instance with architecture-specific configuration.

        Default implementation for scalar ISAs that need no special initialization.
        Override in subclasses that require architecture-specific parameters
        (e.g., vector ISAs that need vector length).

        Args:
            arch: Architecture instance with detected/specified hardware parameters.

        Returns:
            Initialized ISA instance configured for the target architecture.
        """
        return cls()

    def __init__(self, **kwargs: Any):
        """
        ISA configuration and metadata for benchmark generation. This class encapsulates parameters and register/
        instruction set selections used to generate microbenchmarks. It holds limits and formats that guide how
        instruction streams and loops are constructed.

        ### Benchmark Register Management (IMPORTANT):
        `bench_registers` must be initialized in `__init__` of each concrete ISA class (not as a class attribute)
        because:
        - RegisterSets contain mutable state: `CyclicRegisterSet` maintains a `running_index` counter that cycles
            through registers
        - Each instance needs its own independent register counter to ensure deterministic code generation
        - Sharing mutable state as a class attribute would cause different instances to interfere with each other
        - Always use `CyclicRegisterSet` for benchmark registers to enable proper register cycling

        Example (DO NOT make this a class attribute):
        ```python
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.bench_registers = TypedRegisterSets({
                DataType.f32: CyclicRegisterSet("xmm{}", [(0, 16)]),
                DataType.f64: CyclicRegisterSet("xmm{}", [(0, 16)]),
            })
        ```

        ### Keyword arguments (optional):
        - `instruction_limit` (int): Maximum total number of instructions
            allowed in a generated benchmark (sum of inner and outer loop bodies).
            Defaults to 2048.
        """

        self.bench_registers: TypedRegisterSets
        "Benchmark registers with mutable cycling state (must be instance-level, not class-level)"

    def setup_assembly(self, data_type: DataType) -> list[str]:
        "Returns the assembly instructions that are run prior to the benchmark, e.g. RVV's vsetvli instruction"
        return []

    def format_iasm_input(self, var: InlineASM.Input) -> str:
        "Formats an inline asm input variable, may vary between ISAs"
        return f"%[{var.asm_name}]"

    def __generate_generic_benchmark(self, asm: InlineASM, func_name: str = "test_function") -> str:
        """Generate a benchmark function with standardized signature.

        All benchmarks use the same signature: void fn(void* read_ptr, void* write_ptr, uint64_t num_reps)
        Arithmetic benchmarks ignore both pointer parameters; memory benchmarks use them.
        """
        func_header = (
            f"static inline __attribute__((always_inline))\nvoid {func_name}"
            f"(void* read_ptr, void* write_ptr, uint64_t num_reps)"
        )
        return f"{func_header} {{\n{asm.format()}}}"

    def _split_loop(self, num_ops: int, max_loop_size: int) -> LoopSplitConfig:
        """Calculate how to split operations into inner/outer loop.

        Args:
            num_ops: Total number of operations to perform.
            max_loop_size: Maximum operations per loop iteration.

        Returns:
            Configuration specifying how to split into inner/outer loops.
        """

        num_iterations = num_ops // max_loop_size
        instance_inner_loop = num_iterations > 1

        if instance_inner_loop:
            inner_repeats = max_loop_size
            outer_repeats = num_ops % max_loop_size
            debug(f"{num_iterations} * {inner_repeats} + {outer_repeats} repeats")
        else:
            inner_repeats = 0
            outer_repeats = num_ops
            debug(f"{outer_repeats} repeats, no inner loop")

        return LoopSplitConfig(
            instance_inner_loop=instance_inner_loop,
            inner_repeats=inner_repeats,
            outer_repeats=outer_repeats,
            num_iterations=num_iterations,
        )

    def _validate_memory_size(
        self,
        params: MemoryBenchmarkParams,
        bytes_per_inst: int,
        insts_per_repeat: int,
    ) -> SizeInfo:
        """Validate memory benchmark size and derive working set size.

        Args:
            params: Memory benchmark parameters.
            bytes_per_inst: Bytes transferred per memory instruction.
            insts_per_repeat: Number of memory instructions per repeat.

        Returns:
            SizeInfo with repeats, bytes per repeat, and working set size.
        """

        actual_bytes_per_repeat = insts_per_repeat * bytes_per_inst
        repeats = params.size_per_thread.value // actual_bytes_per_repeat
        debug(
            f"_validate_memory_size: size_per_thread={params.size_per_thread.value}"
            f" bytes_per_inst={bytes_per_inst} insts_per_repeat={insts_per_repeat}"
            f" actual_bytes_per_repeat={actual_bytes_per_repeat} repeats={repeats}"
        )
        if repeats < 1:
            raise BenchParamError(
                f"Target size {params.size_per_thread.value} too small for ISA instruction size "
                f"({bytes_per_inst}B x {insts_per_repeat} ops = {actual_bytes_per_repeat}B per repeat)"
            )

        actual_working_set_size = repeats * actual_bytes_per_repeat
        if params.layout_mode == MemoryLayoutMode.single:
            # Single layout: one buffer covers both loads and stores sequentially.
            read_array_size = actual_working_set_size
            write_array_size = 0
        else:
            # Split layout: independent read and write buffers.
            read_array_size = repeats * params.num_ld * bytes_per_inst
            write_array_size = repeats * params.num_st * bytes_per_inst

        truncation = params.size_per_thread.value - actual_working_set_size
        if truncation != 0:
            debug(
                f"size_per_thread={params.size_per_thread.value} truncated to "
                f"actual_working_set={actual_working_set_size} "
                f"(delta={truncation} bytes, "
                f"factor={params.size_per_thread.value / actual_working_set_size:.6f})"
            )

        return SizeInfo(
            repeats=repeats,
            bytes_per_repeat=actual_bytes_per_repeat,
            actual_working_set_size=actual_working_set_size,
            read_array_size=read_array_size,
            write_array_size=write_array_size,
        )

    def _calculate_loop_configuration(
        self,
        params: MemoryBenchmarkParams,
        repeats: int,
        insts_per_repeat: int,
        branch_distance_limit: int,
        bytes_per_inst: int,
        load_format: inst.Memory,
    ) -> LoopConfig:
        """Calculate loop configuration for memory benchmarks.

        Args:
            params: Memory benchmark parameters.
            repeats: Number of repeats to execute per benchmark.
            insts_per_repeat: Number of memory instructions per repeat.
            branch_distance_limit: Max instructions reachable by a branch (from max_branch_insts).
                Distinct from self.instruction_limit, which caps for uOp cache fit.
            bytes_per_inst: Bytes per memory instruction for the ISA.
            load_format: Load instruction format for addressing mode checks.

        Returns:
            LoopConfig with instruction and loop sizing details.
        """

        insts_are_ptr_offset = load_format.type == inst.Memory.AddressingMode.ptr_offset
        if not insts_are_ptr_offset and not self.unroll_loop:
            raise BenchParamError(
                "Memory instructions with no pointer offset require loop unrolling. Enable loop"
                " unrolling by setting the ISA's 'unroll_loop' attribute to True."
            )

        block_size_offsets = self.max_unique_offsets(params.data_type) if insts_are_ptr_offset else 1
        bytes_per_block = block_size_offsets * bytes_per_inst
        if self.unroll_loop:
            # Each block needs block_size_offsets memory instructions plus insts_per_repeat
            # pointer-advance instructions (one per pointer per block boundary).
            insts_per_block = insts_per_repeat * (block_size_offsets + 1)
            blocks_per_loop = branch_distance_limit // insts_per_block
            mem_insts_per_loop = blocks_per_loop * block_size_offsets * insts_per_repeat
        else:
            # Non-unroll: one block per loop iteration (one pointer bump per block).
            # Cap to block_size_offsets so the body never needs a second pointer bump.
            mem_insts_per_loop = min(block_size_offsets, branch_distance_limit)

        mem_insts_per_loop_uncapped = mem_insts_per_loop
        # Also cap to fit the inner loop body within the uOp cache (self.instruction_limit).
        mem_insts_per_loop = min(mem_insts_per_loop, self.instruction_limit // 2)

        max_loop_size = mem_insts_per_loop // insts_per_repeat
        if max_loop_size == 0:
            raise BenchParamError(
                f"Memory instruction configuration results in a loop size of 0 "
                f"(mem_insts_per_loop={mem_insts_per_loop}, insts_per_repeat={insts_per_repeat}). "
                f"Increase max_mem_offset_bytes or reduce the number of loads/stores."
            )
        loop_split = self._split_loop(repeats, max_loop_size)
        debug(
            f"_calculate_loop_configuration: mem_level={params.memory_level_name}"
            f" dtype={params.data_type.name}"
            f" block_size_offsets={block_size_offsets} bytes_per_block={bytes_per_block}"
            f" branch_distance_limit={branch_distance_limit}"
            f" mem_insts_per_loop={mem_insts_per_loop_uncapped}->{mem_insts_per_loop}"
            f" max_loop_size={max_loop_size}"
            f" instance_inner_loop={loop_split.instance_inner_loop}"
            f" num_iterations={loop_split.num_iterations}"
            f" inner_repeats={loop_split.inner_repeats}"
            f" outer_repeats={loop_split.outer_repeats}"
        )

        return LoopConfig(
            block_size_offsets=block_size_offsets,
            bytes_per_block=bytes_per_block,
            mem_insts_per_loop=mem_insts_per_loop,
            max_loop_size=max_loop_size,
            num_iterations=loop_split.num_iterations,
            instance_inner_loop=loop_split.instance_inner_loop,
            inner_repeats=loop_split.inner_repeats,
            outer_repeats=loop_split.outer_repeats,
            loop_instruction_limit=branch_distance_limit,
        )

    def _generate_memory_instruction_stream(
        self,
        params: MemoryBenchmarkParams,
        bench_registers: CyclicRegisterSet,
        load_format: inst.Memory,
        store_format: inst.Memory,
        block_size_offsets: int,
        repeats: int,
    ) -> list[str]:
        """Generate the memory instruction stream for a given repeat count.

        In single layout mode, loads and stores share one pointer and one array.
        In split layout mode, loads and stores use independent pointers/arrays.
        """

        insts = []
        offset_increment = self.offset_increment(params.data_type)
        max_offset = block_size_offsets * offset_increment
        hregs = self.helper_registers

        if params.layout_mode == MemoryLayoutMode.single:
            # Single-array layout: loads and stores are interleaved over one
            # sequential pointer (hregs.pointer), advancing through a shared buffer.
            running_offset = 0
            for _ in range(repeats):
                for fmt, num_insts in ((load_format, params.num_ld), (store_format, params.num_st)):
                    for _ in range(num_insts):
                        inst_line = fmt.fmt(bench_registers, hregs.pointer, offset=running_offset)
                        insts.append(inst_line)
                        running_offset += offset_increment
                        if running_offset >= max_offset:
                            running_offset = 0
                            insts.append(
                                self.control_instructions.add.fmt(hregs.pointer, hregs.pointer, hregs.pointer_increment)
                            )

            remaining_offset = self.bytes_per_inst(params.data_type) * running_offset // offset_increment
            if remaining_offset > 0:
                insts.append(self.control_instructions.add_imm.fmt(hregs.pointer, hregs.pointer, remaining_offset))
        else:
            # Split-array layout: loads advance through read buffer (hregs.pointer),
            # stores advance through write buffer (hregs.write_pointer) independently.
            running_read_offset = 0
            running_write_offset = 0

            for _ in range(repeats):
                # Loads: advance through read buffer via hregs.pointer
                for _ in range(params.num_ld):
                    inst_line = load_format.fmt(bench_registers, hregs.pointer, offset=running_read_offset)
                    insts.append(inst_line)
                    running_read_offset += offset_increment
                    if running_read_offset >= max_offset:
                        running_read_offset = 0
                        insts.append(
                            self.control_instructions.add.fmt(hregs.pointer, hregs.pointer, hregs.pointer_increment)
                        )

                # Stores: advance through write buffer via hregs.write_pointer
                for _ in range(params.num_st):
                    inst_line = store_format.fmt(bench_registers, hregs.write_pointer, offset=running_write_offset)
                    insts.append(inst_line)
                    running_write_offset += offset_increment
                    if running_write_offset >= max_offset:
                        running_write_offset = 0
                        insts.append(
                            self.control_instructions.add.fmt(
                                hregs.write_pointer, hregs.write_pointer, hregs.pointer_increment
                            )
                        )

            # Advance each pointer past any remaining partial block
            remaining_read_offset = self.bytes_per_inst(params.data_type) * running_read_offset // offset_increment
            if remaining_read_offset > 0:
                insts.append(self.control_instructions.add_imm.fmt(hregs.pointer, hregs.pointer, remaining_read_offset))

            remaining_write_offset = self.bytes_per_inst(params.data_type) * running_write_offset // offset_increment
            if remaining_write_offset > 0:
                insts.append(
                    self.control_instructions.add_imm.fmt(
                        hregs.write_pointer, hregs.write_pointer, remaining_write_offset
                    )
                )

        return insts

    def ops_per_inst(self, data_type: DataType, op: ArithmeticOperation) -> int:
        "Returns the number of operations per instruction for the given operation"
        # data_type is unused in the base implementation, but may be used by subclasses (e.g. RVV)
        return op.ops()

    def bytes_per_inst(self, data_type: DataType) -> int:
        "Returns the number of bytes per memory instruction"
        return data_type.bytes()

    def max_unique_offsets(self, data_type: DataType) -> int:
        """Returns the maximum number of unique memory offsets for the given data type.
        For example, if the data type is 4 bytes and the maximum memory offset is 64 bytes,
        then the maximum number of unique offsets is 16 (0, 4, 8, ..., 60). This can be
        overridden by ISAs with specific memory instruction limitations, such as ARM SVE
        and how its offsets don't depend on the data type size."""
        return self.max_mem_offset_bytes // self.bytes_per_inst(data_type)

    def offset_increment(self, data_type: DataType) -> int:
        "Returns the offset increment per memory instruction"
        return self.bytes_per_inst(data_type)

    def generate_arithmetic(
        self, params: ArithmeticBenchmarkParams, context: CARMContext
    ) -> MicrobenchmarkFunctionSpec:
        """Generates an arithmetic benchmark from a parameter object.

        Args:
            params: ArithmeticBenchmarkParams instance with benchmark configuration.
            context: CARMContext with architecture and configuration details.

        Returns:
            MicrobenchmarkFunctionSpec with generated code and metadata.

        Example:
            params = ArithmeticBenchmarkParams(
                data_type=DataType.f32,
                operation=ArithmeticOperation.fma,
                num_ops=1000,
                num_threads=1
            )
            spec = isa.generate_arithmetic(params)
        """

        from carm_roofline.test_bench.builder import MicrobenchmarkFunctionSpec

        # Generate function name from ISA, operation, and data type
        func_name = f"{self.name}_arith_{params.operation.name}_{params.data_type.name}_t{params.num_threads}"

        bench_registers = self.bench_registers[params.data_type]
        inst_format = self.bench_instructions.get(params.data_type, params.operation)

        var_num_reps = InlineASM.Input(c_name="num_reps", asm_name="num_reps")

        PRELUDE_INSTRUCTIONS = 5
        max_loop_size = min(
            # Half the branch limit (one half for the loop, other half for out-of-loop insts)
            (self.max_branch_insts - PRELUDE_INSTRUCTIONS) // 2,
            self.instruction_limit // 2,
        )
        num_instructions = params.num_ops.value // self.ops_per_inst(params.data_type, params.operation)
        loop_split = self._split_loop(num_instructions, max_loop_size)
        debug(
            f"generate_arithmetic: func={func_name}"
            f" inner_loop={loop_split.instance_inner_loop}"
            f" inner_repeats={loop_split.inner_repeats}"
            f" outer_repeats={loop_split.outer_repeats}"
            f" num_iterations={loop_split.num_iterations}"
        )

        def generate_insts(num: int) -> list[str]:
            return [inst_format.fmt(bench_registers) for _ in range(num)]

        asm = [
            *self.setup_assembly(params.data_type),
            # Load the outer loop iteration count
            self.control_instructions.load_word.fmt(
                self.helper_registers.outer_iterator, self.format_iasm_input(var_num_reps)
            ),
            self.OUTER_LOOP_LABEL + ":",
            # If an inner loop is to be instanced, add inner instructions
            *_add_if(
                loop_split.instance_inner_loop,
                # Load the number of iters, add loop label
                self.control_instructions.load_imm.fmt(self.helper_registers.inner_iterator, loop_split.num_iterations),
                self.INNER_LOOP_LABEL + ":",
                *generate_insts(loop_split.inner_repeats),
                # Loop control (iterator and pointer operations, branch)
                # TODO: Moving the iter decrement above the instructions should improve performance
                self.control_instructions.sub_imm.fmt(
                    self.helper_registers.inner_iterator, self.helper_registers.inner_iterator, 1
                ),
                self.control_instructions.branch_nz.fmt(self.helper_registers.inner_iterator, self.INNER_LOOP_LABEL),
            ),
            # Outer instructions
            *generate_insts(loop_split.outer_repeats),
            # Outer loop control (decrement iterator, branch)
            # TODO: Also move above the instructions?
            self.control_instructions.sub_imm.fmt(
                self.helper_registers.outer_iterator, self.helper_registers.outer_iterator, 1
            ),
            self.control_instructions.branch_nz.fmt(self.helper_registers.outer_iterator, self.OUTER_LOOP_LABEL),
        ]

        # Clobbers: bench registers + helper registers used by the outer/inner loop
        clobbers = [*bench_registers, self.helper_registers.outer_iterator, self.helper_registers.inner_iterator]
        iasm = InlineASM(asm, [var_num_reps], clobbers)
        bench_code = self.__generate_generic_benchmark(iasm, func_name=func_name)

        # Extract ISA-specific frequency from context
        frequency = context.architecture.get_frequency_for_isa(self.name)

        return MicrobenchmarkFunctionSpec(
            function_name=func_name,
            body=bench_code,
            read_array_size=Bytes(0),
            write_array_size=Bytes(0),
            frequency=frequency,
            thread_affinity=params.thread_affinity,
            nominal_frequency=context.architecture.nominal_frequency,
        )

    def generate_memory(self, params: MemoryBenchmarkParams, context: CARMContext) -> MicrobenchmarkFunctionSpec:
        """Generates a memory benchmark from a parameter object.

        Args:
            params: MemoryBenchmarkParams instance with benchmark configuration.
            context: CARMContext with architecture and configuration details.

        Returns:
            MicrobenchmarkFunctionSpec with generated code and metadata.

        Example:
            params = MemoryBenchmarkParams(
                data_type=DataType.f64,
                num_ld=4,
                num_st=2,
                repeats=256
            )
            spec = isa.generate_memory(params, context)
        """

        from carm_roofline.test_bench.builder import MicrobenchmarkFunctionSpec

        # Calculate actual bytes per instruction for this ISA (accounts for vector width)
        bytes_per_inst = self.bytes_per_inst(params.data_type)
        insts_per_repeat = params.num_ld + params.num_st
        size_info = self._validate_memory_size(params, bytes_per_inst, insts_per_repeat)

        # Generate function name with actual working set size
        func_name = (
            f"{self.name}_mem_{params.num_ld}ld_{params.num_st}st_"
            f"{params.memory_level_name.lower()}_{size_info.actual_working_set_size}_{params.data_type.name}"
            f"_t{params.num_threads}"
        )
        # Half the branch limit (one half for the loop, other half for out-of-loop insts)
        PRELUDE_INSTRUCTIONS = 7
        # branch_distance_limit: max instructions that can fit in one loop body without
        # violating the ISA's branch encoding range (max_branch_insts).
        # Distinct from self.instruction_limit, which caps for uOp cache fit.
        branch_distance_limit = (self.max_branch_insts - PRELUDE_INSTRUCTIONS) // 2
        # Can we fit all the instructions in one loop iteration?
        if insts_per_repeat > branch_distance_limit:
            raise BenchParamError(
                f"Maximum branch distance 'max_branch_insts' cannot support so many instructions"
                f" ({params.num_ld} loads, {params.num_st} stores). Reduce the number of loads and/or stores."
            )

        ## TODO: Redo notes
        # Instruction generation should be dynamic, such that any ld/st ratio is supported while
        # keeping pointer additions to a minimum. i.e. a pointer addition can be inserted in the
        # middle of a ld/st sequence if needed.
        # The loop size is then only limited by the branch distance, as the base pointer can be
        # updated as needed.
        # Coining "instruction block" as a sequence of instructions between pointer updates.
        # The block size is limited by the maximum memory offset.
        # RVV then becomes easy to represent, as the block size is 1.

        bench_registers = self.bench_registers[params.data_type]
        load_format = self.bench_instructions.get(params.data_type, MemoryOperation.ld)
        store_format = self.bench_instructions.get(params.data_type, MemoryOperation.st)

        loop_config = self._calculate_loop_configuration(
            params=params,
            repeats=size_info.repeats,
            insts_per_repeat=insts_per_repeat,
            branch_distance_limit=branch_distance_limit,
            bytes_per_inst=bytes_per_inst,
            load_format=load_format,
        )
        debug(
            f"generate_memory: func={func_name}"
            f" inner_loop={loop_config.instance_inner_loop}"
            f" inner_repeats={loop_config.inner_repeats}"
            f" outer_repeats={loop_config.outer_repeats}"
            f" num_iterations={loop_config.num_iterations}"
        )

        var_num_reps = InlineASM.Input(c_name="num_reps", asm_name="num_reps")
        var_read_ptr = InlineASM.Input(c_name="read_ptr", asm_name="read_ptr")
        var_write_ptr = InlineASM.Input(c_name="write_ptr", asm_name="write_ptr")

        asm = [
            self.control_instructions.load_imm.fmt(
                self.helper_registers.pointer_increment, loop_config.bytes_per_block
            ),
            *self.setup_assembly(params.data_type),
            # Load the outer loop iteration count
            self.control_instructions.load_word.fmt(
                self.helper_registers.outer_iterator, self.format_iasm_input(var_num_reps)
            ),
            self.OUTER_LOOP_LABEL + ":",
            # Reload pointer(s) from C variable(s) at start of each outer iteration
            self.control_instructions.load_word.fmt(
                self.helper_registers.pointer, self.format_iasm_input(var_read_ptr)
            ),
            *_add_if(
                params.layout_mode == MemoryLayoutMode.split and params.num_st > 0,
                self.control_instructions.load_word.fmt(
                    self.helper_registers.write_pointer, self.format_iasm_input(var_write_ptr)
                ),
            ),
            # If an inner loop is to be instanced, add inner instructions
            *_add_if(
                loop_config.instance_inner_loop,
                # Load the number of iters, add loop label
                self.control_instructions.load_imm.fmt(
                    self.helper_registers.inner_iterator, loop_config.num_iterations
                ),
                self.INNER_LOOP_LABEL + ":",
                *self._generate_memory_instruction_stream(
                    params,
                    bench_registers,
                    load_format,
                    store_format,
                    loop_config.block_size_offsets,
                    loop_config.inner_repeats,
                ),
                # Loop control (iterator operation, branch)
                # TODO: Moving the iter decrement above the instructions should improve performance
                self.control_instructions.sub_imm.fmt(
                    self.helper_registers.inner_iterator, self.helper_registers.inner_iterator, 1
                ),
                self.control_instructions.branch_nz.fmt(self.helper_registers.inner_iterator, self.INNER_LOOP_LABEL),
            ),
            # Outer instructions
            *self._generate_memory_instruction_stream(
                params,
                bench_registers,
                load_format,
                store_format,
                loop_config.block_size_offsets,
                loop_config.outer_repeats,
            ),
            # Outer loop control (decrement iterator, branch)
            # TODO: Also move above the instructions?
            self.control_instructions.sub_imm.fmt(
                self.helper_registers.outer_iterator, self.helper_registers.outer_iterator, 1
            ),
            self.control_instructions.branch_nz.fmt(self.helper_registers.outer_iterator, self.OUTER_LOOP_LABEL),
        ]

        # Build inputs list and clobbers depending on array mode
        asm_inputs = [var_num_reps, var_read_ptr]
        hregs = self.helper_registers
        if params.layout_mode == MemoryLayoutMode.single:
            # Single-array: write_ptr argument is not used in the asm; no write_pointer clobber.
            helper_clobbers = [
                hregs.outer_iterator,
                hregs.inner_iterator,
                hregs.pointer,
                hregs.pointer_increment,
            ]
        else:
            # Split-array: include write_ptr if there are stores.
            if params.num_st > 0:
                asm_inputs.append(var_write_ptr)
            helper_clobbers = [
                hregs.outer_iterator,
                hregs.inner_iterator,
                hregs.pointer,
                hregs.pointer_increment,
                hregs.write_pointer,
            ]
        clobbers = [*bench_registers, *helper_clobbers]
        iasm = InlineASM(asm, asm_inputs, clobbers)
        bench_code = self.__generate_generic_benchmark(iasm, func_name=func_name)

        # Extract ISA-specific frequency from context
        frequency = context.architecture.get_frequency_for_isa(self.name)

        return MicrobenchmarkFunctionSpec(
            function_name=func_name,
            body=bench_code,
            read_array_size=Bytes(size_info.read_array_size),
            write_array_size=Bytes(size_info.write_array_size),
            frequency=frequency,
            thread_affinity=params.thread_affinity,
            nominal_frequency=context.architecture.nominal_frequency,
        )
