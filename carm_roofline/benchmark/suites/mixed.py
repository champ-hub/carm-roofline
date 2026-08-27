"""Mixed benchmark suite for arithmetic intensity sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import TYPE_CHECKING

from carm_roofline.benchmark.generation import MixedBenchmarkParams
from carm_roofline.benchmark.generation.parameters import BenchParamError
from carm_roofline.core import ArithmeticIntensity, Operations, UserError
from carm_roofline.output_utils import warn

from ..benchmark import MixedBenchmark
from .base import ISABenchmarkSuite
from .memory import iter_memory_benchmark_params

if TYPE_CHECKING:
    from carm_roofline.context import CARMContext


@dataclass
class MixedBenchmarkSuite(ISABenchmarkSuite):
    """Suite for mixed arithmetic and memory benchmarks."""

    @classmethod
    def generate(cls, context: CARMContext, isa_name: str) -> MixedBenchmarkSuite:
        """Generate mixed microbenchmarks for one ISA."""
        architecture = context.architecture
        isa_class = next((candidate for candidate in architecture.isa if candidate.name == isa_name), None)
        if isa_class is None:
            raise ValueError(f"Unknown ISA: {isa_name}")
        isa = isa_class.from_architecture(architecture)
        suite = cls(isa_name=isa.name)
        ai_min, ai_max = context.benchmarking.ai_range
        point_count = context.benchmarking.ai_points
        requested_ais = (
            [ai_min]
            if point_count == 1
            else [
                ArithmeticIntensity(
                    exp(log(float(ai_min)) + index / (point_count - 1) * (log(float(ai_max)) - log(float(ai_min))))
                )
                for index in range(point_count)
            ]
        )
        for memory_params in iter_memory_benchmark_params(context, isa):
            for operation in sorted(context.benchmarking.instructions, key=lambda item: item.name):
                if operation not in isa.bench_instructions.available_operations(memory_params.data_type):
                    warn(
                        f"Skipping mixed operation '{operation.name}' for data type "
                        f"'{memory_params.data_type.name}' on ISA '{isa.name}': instruction is unavailable"
                    )
                    continue
                for point_index, requested_ai in enumerate(requested_ais):
                    try:
                        arithmetic_count, pattern_repeats, achieved_ai = isa._select_mixed_instruction_counts(
                            memory_params.data_type, operation, memory_params.load_store_ratio, requested_ai
                        )
                    except BenchParamError as error:
                        warn(
                            f"Skipping mixed operation '{operation.name}' at requested AI {requested_ai} "
                            f"for ISA '{isa.name}': {error}"
                        )
                        continue
                    params = MixedBenchmarkParams(
                        data_type=memory_params.data_type,
                        thread_affinity=memory_params.thread_affinity,
                        load_store_ratio=memory_params.load_store_ratio,
                        size_per_thread=memory_params.size_per_thread,
                        memory_level_name=memory_params.memory_level_name,
                        operation=operation,
                        point_index=point_index,
                        requested_arithmetic_intensity=requested_ai,
                        num_arithmetic_instructions=arithmetic_count,
                        memory_pattern_repeats=pattern_repeats,
                        achieved_arithmetic_intensity=achieved_ai,
                        layout_mode=memory_params.layout_mode,
                    )
                    spec = isa.generate_mixed(params, context)
                    memory_events = (params.num_ld + params.num_st) * params.memory_pattern_repeats
                    blocks_per_thread = (spec.read_array_size + spec.write_array_size) / (
                        memory_events * isa.bytes_per_inst(params.data_type)
                    )
                    operations_per_thread = Operations(
                        int(blocks_per_thread)
                        * params.num_arithmetic_instructions
                        * isa.ops_per_inst(params.data_type, operation)
                    )
                    benchmark = MixedBenchmark(
                        params=params,
                        spec=spec,
                        operations_per_thread=operations_per_thread,
                        working_set_bytes=(spec.read_array_size + spec.write_array_size) * params.num_threads,
                        cache_level=params.memory_level_name,
                    )
                    suite.add_benchmark(benchmark.name, benchmark)
        if not suite.benchmarks:
            raise UserError(f"No mixed benchmarks can be generated for ISA '{isa_name}'")
        return suite
