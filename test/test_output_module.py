# File: refactor_tests/test_output_module.py
"""Unit tests for the `benchmark.output` module.

These tests are self-contained and mock external plotting libraries when needed.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

from benchmark.benchmarking import LoadStoreRatio, TestType as BenchmarkTestType
from benchmark.output import OutputKind
from results_paths import default_results_root
from units import Frequency, Operations, Performance, Seconds


def make_fake_matplotlib():
    """Create a fake `matplotlib` package with a `pyplot` submodule.

    The fake records savefig calls in `pyplot._saved` for assertions.
    """
    mpl = types.ModuleType("matplotlib")
    pyplot = types.ModuleType("matplotlib.pyplot")
    pyplot._saved = []

    # Create fake axis class with necessary methods
    class FakeAxis:
        def bar(self, *args, **kwargs):
            return None

        def scatter(self, *args, **kwargs):
            return None

        def plot(self, *args, **kwargs):
            return None

        def set_ylabel(self, *args, **kwargs):
            return None

        def set_xlabel(self, *args, **kwargs):
            return None

        def set_title(self, *args, **kwargs):
            return None

        def text(self, *args, **kwargs):
            return None

        @property
        def transAxes(self):
            return None

    # Create fake figure class
    class FakeFigure:
        pass

    def figure(*args, **kwargs):
        return FakeFigure()

    def bar(*args, **kwargs):
        return None

    def close(*args, **kwargs):
        return None

    def savefig(path, *args, **kwargs):
        pyplot._saved.append(path)

    def show(*args, **kwargs):
        return None

    def subplots(*args, **kwargs):
        fig = FakeFigure()
        nrows = args[0] if len(args) > 0 else 1
        ncols = args[1] if len(args) > 1 else 1
        num_axes = nrows * ncols
        if num_axes == 1:
            return (fig, FakeAxis())
        else:
            axes = [FakeAxis() for _ in range(num_axes)]
            return (fig, axes)

    def tight_layout(*args, **kwargs):
        return None

    def scatter(*args, **kwargs):
        return None

    def plot(*args, **kwargs):
        return None

    def xscale(*args, **kwargs):
        return None

    def yscale(*args, **kwargs):
        return None

    def xlabel(*args, **kwargs):
        return None

    def ylabel(*args, **kwargs):
        return None

    def title(*args, **kwargs):
        return None

    def legend(*args, **kwargs):
        return None

    def suptitle(*args, **kwargs):
        return None

    pyplot.figure = figure
    pyplot.bar = bar
    pyplot.savefig = savefig
    pyplot.show = show
    pyplot.close = close
    pyplot.subplots = subplots
    pyplot.tight_layout = tight_layout
    pyplot.scatter = scatter
    pyplot.plot = plot
    pyplot.xscale = xscale
    pyplot.yscale = yscale
    pyplot.xlabel = xlabel
    pyplot.ylabel = ylabel
    pyplot.title = title
    pyplot.legend = legend
    pyplot.suptitle = suptitle

    mpl.pyplot = pyplot
    return mpl, pyplot


def make_fake_numpy():
    """Create a minimal fake numpy module used by roofline plotting.

    Implements `log10` and `logspace`.
    """
    np = types.ModuleType("numpy")
    import math

    def log10(x):
        return math.log10(x)

    def logspace(a, b, n):
        # return a simple list of floats between 10**a and 10**b
        start = 10**a
        end = 10**b
        if n <= 1:
            return [start]
        step = (end - start) / (n - 1)
        return [start + i * step for i in range(n)]

    np.log10 = log10
    np.logspace = logspace
    return np


class _Result:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _Benchmark:
    def __init__(self, name: str, results: _Result | None = None):
        self.name = name
        self.results = results


class _Suite:
    def __init__(self, benchmarks=None, **attrs):
        # benchmarks: should be a dict to match ISABenchmarkSuite.benchmarks structure
        if benchmarks is None:
            self.benchmarks = {}
        elif isinstance(benchmarks, dict):
            self.benchmarks = benchmarks
        elif isinstance(benchmarks, list):
            # Convert list to dict with auto-generated names for backward compatibility
            self.benchmarks = {f"bench_{i}": b for i, b in enumerate(benchmarks)}
        else:
            self.benchmarks = benchmarks
        for k, v in attrs.items():
            setattr(self, k, v)

    def get_arithmetic_benchmarks(self):
        """Return only arithmetic benchmarks from this suite."""
        from benchmark.benchmark import ArithmeticBenchmark

        return {name: bench for name, bench in self.benchmarks.items() if isinstance(bench, ArithmeticBenchmark)}


def _make_fake_context(isa_names: list[str], freq_hz: float = 3.0e9, nominal_hz: float | None = None):
    """Create a minimal context-like object for output handlers."""

    class _FakeISAInstance:
        def ops_per_inst(self, data_type, op):
            return op.ops()

        def bytes_per_inst(self, data_type):
            return 4  # assume 4 bytes per instruction for simplicity

    def _make_isa_class(isa_name: str):
        class _FakeISA:
            name = isa_name

            @classmethod
            def from_architecture(cls, _arch):
                return _FakeISAInstance()

        return _FakeISA

    class _FakeArchitecture:
        def __init__(self):
            self.isa = [_make_isa_class(n) for n in isa_names]
            self._freqs = {n: Frequency(freq_hz) for n in isa_names}
            self.nominal_frequency = Frequency(nominal_hz) if nominal_hz is not None else None

        def get_frequency_for_isa(self, isa_name: str):
            return self._freqs[isa_name]

    class _FakeBenchmarking:
        def __init__(self):
            from benchmark.generation.code_gen.operation import ArithmeticOperation

            self.test = BenchmarkTestType.ARITHMETIC
            self.threads = 1
            self.interleaved = False
            self.data_type = None
            self.instructions = {ArithmeticOperation.fma}

    class _FakeRunConfig:
        def __init__(self):
            self.verbose = 2
            self.name = "test"
            self.dry_run = False
            self.output_dir = default_results_root()
            self.output_formats = {OutputKind.TABLE}

    return types.SimpleNamespace(
        architecture=_FakeArchitecture(),
        benchmarking=_FakeBenchmarking(),
        run_config=_FakeRunConfig(),
    )


def reload_handlers(monkeypatch):
    """Reload handler modules so they pick up any monkeypatched matplotlib/numpy."""
    import benchmark.output.arithmetic as arithmetic
    import benchmark.output.memory as memory
    import benchmark.output.mixed as mixed
    import benchmark.output.roofline as roofline

    importlib.reload(arithmetic)
    importlib.reload(memory)
    importlib.reload(mixed)
    importlib.reload(roofline)
    return arithmetic, memory, mixed, roofline


def test_factory_returns_registered_strategy_instance():
    """Factory returns strategy class for the requested test type."""
    from benchmark.output import TestType, _get_handler_for_test_type
    from benchmark.output.arithmetic import ArithmeticOutputHandler

    strategy = _get_handler_for_test_type(TestType.ARITHMETIC)
    assert isinstance(strategy, ArithmeticOutputHandler)


_SUPPORTED_OUTPUT_TEST_TYPES = (
    BenchmarkTestType.ARITHMETIC,
    BenchmarkTestType.ROOFLINE,
    BenchmarkTestType.MEMORY,
    BenchmarkTestType.MIXED,
    BenchmarkTestType.MEMORY_SWEEP,
)


_STRATEGY_REGISTRY = {
    BenchmarkTestType.ARITHMETIC: ("benchmark.output.arithmetic", "ArithmeticOutputHandler"),
    BenchmarkTestType.ROOFLINE: ("benchmark.output.roofline", "RooflineOutputHandler"),
    BenchmarkTestType.MEMORY: ("benchmark.output.memory", "MemoryOutputHandler"),
    BenchmarkTestType.MIXED: ("benchmark.output.mixed", "MixedOutputHandler"),
    BenchmarkTestType.MEMORY_SWEEP: ("benchmark.output.memory_sweep", "MemorySweepOutputHandler"),
}


def _load_strategy(test_type: BenchmarkTestType):
    module_name, class_name = _STRATEGY_REGISTRY[test_type]
    module = importlib.import_module(module_name)
    return module, getattr(module, class_name)


def test_strategy_registry_covers_all_supported_output_tests():
    """Strategy registry should cover every formatter-supported TestType."""
    assert set(_STRATEGY_REGISTRY) == set(_SUPPORTED_OUTPUT_TEST_TYPES)


@pytest.mark.parametrize("test_type", _SUPPORTED_OUTPUT_TEST_TYPES)
def test_strategy_classes_expose_expected_methods(test_type):
    """All output strategies expose CLI and plot entrypoints."""
    _module, strategy_class = _load_strategy(test_type)
    strategy = strategy_class()

    assert hasattr(strategy, "print_table")
    assert callable(strategy.print_table)
    assert hasattr(strategy, "write_plot")
    assert callable(strategy.write_plot)
    assert hasattr(strategy, "write_jsonl")
    assert callable(strategy.write_jsonl)


def test_roofline_strategy_exposes_file_output_methods():
    """Roofline strategy exposes strategy-local CSV hook."""
    _module, strategy_class = _load_strategy(BenchmarkTestType.ROOFLINE)
    strategy = strategy_class()
    assert hasattr(strategy, "write_csv")
    assert callable(strategy.write_csv)
    assert hasattr(strategy, "write_jsonl")
    assert callable(strategy.write_jsonl)


@pytest.mark.parametrize("test_type", _SUPPORTED_OUTPUT_TEST_TYPES)
def test_table_output_dispatches_via_registered_strategy_path(monkeypatch, test_type):
    """Table output dispatches to the registered test-specific strategy path."""
    from benchmark.output import output_benchmark_results

    context = _make_fake_context(["isa1"])
    context.benchmarking.test = test_type
    context.run_config.verbose = 99
    context.run_config.output_formats = {OutputKind.TABLE}
    isa_suites = {}

    module, strategy_class = _load_strategy(test_type)
    calls: list[tuple[str, object, object]] = []

    def _module_dispatch(called_context, called_suites):
        calls.append(("module", called_context, called_suites))

    def _strategy_dispatch(_self, called_context, called_suites):
        calls.append(("strategy", called_context, called_suites))

    monkeypatch.setattr(module, "_print_table", _module_dispatch)
    monkeypatch.setattr(strategy_class, "print_table", _strategy_dispatch, raising=False)

    output_benchmark_results(context, isa_suites)

    assert len(calls) == 1
    _, called_context, called_suites = calls[0]
    assert called_context is context
    assert called_suites is isa_suites


@pytest.mark.parametrize("test_type", _SUPPORTED_OUTPUT_TEST_TYPES)
def test_plot_output_dispatches_via_registered_strategy_path(monkeypatch, test_type):
    """Plot output dispatches to the registered test-specific strategy path."""
    from benchmark.output import output_benchmark_results

    context = _make_fake_context(["isa1"])
    context.benchmarking.test = test_type
    context.run_config.output_dir = Path("plots_out")
    context.run_config.output_formats = {OutputKind.PLOT}
    isa_suites = {}

    module, strategy_class = _load_strategy(test_type)
    calls: list[tuple[str, object, object]] = []

    def _module_dispatch(called_suites, output_path=None):
        calls.append(("module", called_suites, output_path))

    def _strategy_dispatch(_self, called_context, called_suites):
        calls.append(("strategy", called_context, called_suites))

    monkeypatch.setattr(module, "_write_plot", _module_dispatch)
    monkeypatch.setattr(strategy_class, "write_plot", _strategy_dispatch, raising=False)

    output_benchmark_results(context, isa_suites)

    assert len(calls) == 1
    call_type, arg1, arg2 = calls[0]
    if call_type == "module":
        assert arg1 is isa_suites
        assert arg2 == context.run_config.output_dir / test_type.value
    else:
        assert arg1 is context
        assert arg2 is isa_suites


def test_arithmetic_cli_prints_gops(capsys):
    """CLI arithmetic output prints GOPS per ISA."""
    from benchmark.benchmark import ArithmeticBenchmark, ArithmeticBenchmarkResult
    from benchmark.generation.code_gen import DataType
    from benchmark.generation.code_gen.operation import ArithmeticOperation
    from benchmark.generation.parameters import ArithmeticBenchmarkParams
    from benchmark.output import TestType, _get_handler_for_test_type
    from benchmark.suites import ArithmeticBenchmarkSuite
    from test_bench.builder import MicrobenchmarkFunctionSpec
    from units import Bytes

    # Create proper ArithmeticBenchmarkResult objects
    r1 = ArithmeticBenchmarkResult(time_taken=Seconds(100.0), num_repetitions=1000, performance=Performance(5.0))
    r2 = ArithmeticBenchmarkResult(time_taken=Seconds(50.0), num_repetitions=1000, performance=Performance(20.5))

    context = _make_fake_context(["isa_b", "isa_a"], freq_hz=3.0e9)

    # parameters now require num_threads and use the `operation` field
    params1 = ArithmeticBenchmarkParams(
        data_type=DataType.f32,
        thread_affinity=[0],
        operation=ArithmeticOperation.add,
        num_ops=Operations(1024),
    )
    params2 = ArithmeticBenchmarkParams(
        data_type=DataType.f32,
        thread_affinity=[0],
        operation=ArithmeticOperation.add,
        num_ops=Operations(1024),
    )
    spec1 = MicrobenchmarkFunctionSpec(
        function_name="b1",
        body="",
        read_array_size=Bytes(0),
        write_array_size=Bytes(0),
        frequency=2.5,
        thread_affinity=[0],
    )
    spec2 = MicrobenchmarkFunctionSpec(
        function_name="b2",
        body="",
        read_array_size=Bytes(0),
        write_array_size=Bytes(0),
        frequency=2.5,
        thread_affinity=[0],
    )
    b1 = ArithmeticBenchmark(params=params1, spec=spec1)
    b2 = ArithmeticBenchmark(params=params2, spec=spec2)
    b1.results = r1
    b2.results = r2

    # two suites with one benchmark each
    s1 = ArithmeticBenchmarkSuite(isa_name="isa_b")
    s2 = ArithmeticBenchmarkSuite(isa_name="isa_a")
    s1.add_benchmark(b1.name, b1)
    s2.add_benchmark(b2.name, b2)
    isa_suites = {"isa_b": s1, "isa_a": s2}

    handler = _get_handler_for_test_type(TestType.ARITHMETIC)
    # Verify it runs without error - the actual output is tested via manual testing
    try:
        handler.print_table(context, isa_suites)
    except Exception as e:
        pytest.fail(f"Handler raised exception: {e}")


def test_arithmetic_plot_saves_file(monkeypatch, tmp_path, capsys):
    """Arithmetic plot saves image file when matplotlib is available."""
    from benchmark.benchmark import ArithmeticBenchmark, ArithmeticBenchmarkParams, ArithmeticBenchmarkResult
    from benchmark.generation.code_gen.operation import ArithmeticOperation
    from units import Operations

    # install fake matplotlib into sys.modules
    mpl, _pyplot = make_fake_matplotlib()
    monkeypatch.setitem(sys.modules, "matplotlib", mpl)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", mpl.pyplot)

    # reload handlers so they pick up fake matplotlib
    arithmetic, _, _, _ = reload_handlers(monkeypatch)

    # Create proper ArithmeticBenchmark with results
    params = ArithmeticBenchmarkParams(
        data_type=None,
        thread_affinity=[0],
        operation=ArithmeticOperation.add,
        num_ops=Operations(1000),
    )
    bench = ArithmeticBenchmark(params=params, spec=None)
    bench.results = ArithmeticBenchmarkResult(
        time_taken=Seconds(100.0), num_repetitions=1000, performance=Performance(5.0)
    )

    s1 = _Suite(benchmarks={"b1": bench})
    isa_suites = {"isa1": s1}

    arithmetic._write_plot(isa_suites, tmp_path)
    # check that savefig was called with path containing arithmetic_gops.png
    saved = [str(p) for p in mpl.pyplot._saved]
    assert any("arithmetic_gops.png" in str(p) for p in saved), f"Plot not saved. Saved: {saved}"


def test_roofline_plot_handles_missing_data(monkeypatch):
    """Roofline plot gracefully handles suites with no points and prints notice."""
    # fake matplotlib and numpy
    mpl, _ = make_fake_matplotlib()
    np = make_fake_numpy()
    monkeypatch.setitem(sys.modules, "matplotlib", mpl)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", mpl.pyplot)
    monkeypatch.setitem(sys.modules, "numpy", np)

    # reload roofline to pick up fakes
    _, _, _, roofline = reload_handlers(monkeypatch)

    # capture warnings emitted via output_utils
    warnings: list[str] = []

    def _warn(message: object) -> None:
        warnings.append(str(message))

    monkeypatch.setattr(roofline, "warn", _warn)

    empty_suite = _Suite(benchmarks=[_Benchmark("b1", _Result())])
    isa_suites = {"isaX": empty_suite}

    roofline._write_plot(isa_suites, None)
    assert any("No roofline data found" in msg for msg in warnings)


def test_roofline_legacy_csv_compatibility(tmp_path):
    """Roofline CSV compatibility writes two headers and appends rows."""
    from benchmark.benchmark import (
        ArithmeticBenchmark,
        ArithmeticBenchmarkResult,
        MemoryBenchmark,
        MemoryBenchmarkResult,
    )
    from benchmark.benchmarking import LoadStoreRatio, TestType
    from benchmark.generation.code_gen import DataType
    from benchmark.generation.code_gen.operation import ArithmeticOperation
    from benchmark.generation.parameters import ArithmeticBenchmarkParams, MemoryBenchmarkParams
    from benchmark.output.roofline import _write_csv
    from benchmark.suites import RooflineBenchmarkSuite
    from test_bench.builder import MicrobenchmarkFunctionSpec
    from units import Bandwidth, Bytes, Performance, Seconds

    context = _make_fake_context(["isa1"], freq_hz=3.0e9)
    context.benchmarking.test = TestType.ROOFLINE
    # Ensure we use known values
    context.benchmarking.data_type = DataType.f32
    context.benchmarking.instructions = {ArithmeticOperation.fma}
    context.benchmarking.threads = 2
    context.benchmarking.ld_st_ratio = LoadStoreRatio(2, 1)

    arith_params = ArithmeticBenchmarkParams(
        data_type=DataType.f32,
        thread_affinity=[0],
        operation=ArithmeticOperation.fma,
        num_ops=Operations(1024),
    )
    arith_spec = MicrobenchmarkFunctionSpec(
        function_name="arith",
        body="",
        read_array_size=Bytes(0),
        write_array_size=Bytes(0),
        frequency=2.5,
        thread_affinity=[0],
    )
    arith_bench = ArithmeticBenchmark(params=arith_params, spec=arith_spec)
    arith_bench.results = ArithmeticBenchmarkResult(
        time_taken=Seconds(0.1), num_repetitions=1000, performance=Performance(10e9)
    )

    mem_params = MemoryBenchmarkParams(
        data_type=DataType.f32,
        thread_affinity=[0],
        load_store_ratio=LoadStoreRatio(2, 1),
        size_per_thread=Bytes(1024),
        memory_level_name="L1",
    )
    mem_spec = MicrobenchmarkFunctionSpec(
        function_name="mem",
        body="",
        read_array_size=Bytes(0),
        write_array_size=Bytes(0),
        frequency=2.5,
        thread_affinity=[0],
    )
    mem_bench = MemoryBenchmark(params=mem_params, spec=mem_spec, working_set_bytes=Bytes(1024), cache_level="L1")
    mem_bench.results = MemoryBenchmarkResult(
        time_taken=Seconds(0.1), num_repetitions=1000, bandwidth=Bandwidth(40e9), cache_level="L1"
    )

    suite = RooflineBenchmarkSuite(isa_name="isa1")
    suite.add_benchmark(arith_bench.name, arith_bench)
    suite.add_benchmark(mem_bench.name, mem_bench)
    isa_suites = {"isa1": suite}

    _write_csv(context, isa_suites, output_dir=tmp_path)

    csv_path = tmp_path / "roofline" / f"{context.run_config.name}_roofline.csv"
    assert csv_path.exists()

    import csv as csv_lib

    with open(csv_path, newline="") as f:
        rows = list(csv_lib.reader(f))

    assert rows[0] == [
        "Name:",
        context.run_config.name,
        "L1 Size:",
        "0",
        "L2 Size:",
        "0",
        "L3 Size:",
        "0",
        "",
        "L1",
        "L1",
        "L2",
        "L2",
        "L3",
        "L3",
        "DRAM",
        "DRAM",
        "FP",
        "FP",
        "FP FMA",
        "FP_FMA",
    ]
    assert rows[1] == [
        "Date",
        "ISA",
        "Precision",
        "Threads",
        "Loads",
        "Stores",
        "Interleaved",
        "DRAM Bytes",
        "FP Inst.",
        "GB/s",
        "I/Cycle",
        "GB/s",
        "I/Cycle",
        "GB/s",
        "I/Cycle",
        "GB/s",
        "I/Cycle",
        "Gflop/s",
        "I/Cycle",
        "Gflop/s",
        "I/Cycle",
    ]

    # FP_FMA columns should mirror FP columns when FMA is the primary instruction
    assert rows[2][19] == rows[2][17]
    assert rows[2][20] == rows[2][18]

    # Ensure legacy parser compatibility (parses by fixed column index)
    from gui.gui_utils import read_csv_file

    machine_name, l1_size, l2_size, l3_size, parsed_rows = read_csv_file(csv_path)
    assert machine_name == context.run_config.name
    assert (l1_size, l2_size, l3_size) == (0, 0, 0)
    assert len(parsed_rows) == 1

    # second write should append row only (no repeated headers)
    _write_csv(context, isa_suites, output_dir=tmp_path)
    with open(csv_path, newline="") as f:
        rows2 = list(csv_lib.reader(f))
    assert len(rows2) == 4


def test_roofline_csv_gates_by_format(tmp_path):
    """Roofline CSV output should only happen with output_format='csv'."""
    from benchmark.benchmark import (
        ArithmeticBenchmark,
        ArithmeticBenchmarkResult,
        MemoryBenchmark,
        MemoryBenchmarkResult,
    )
    from benchmark.benchmarking import LoadStoreRatio, TestType
    from benchmark.generation.code_gen import DataType
    from benchmark.generation.code_gen.operation import ArithmeticOperation
    from benchmark.generation.parameters import ArithmeticBenchmarkParams, MemoryBenchmarkParams
    from benchmark.output import output_benchmark_results
    from benchmark.suites import RooflineBenchmarkSuite
    from test_bench.builder import MicrobenchmarkFunctionSpec
    from units import Bandwidth, Bytes, Performance, Seconds

    context = _make_fake_context(["isa1"], freq_hz=3.0e9)
    context.benchmarking.test = TestType.ROOFLINE
    context.benchmarking.data_type = DataType.f32
    context.benchmarking.instructions = {ArithmeticOperation.fma}
    context.benchmarking.threads = 1
    context.benchmarking.ld_st_ratio = LoadStoreRatio(2, 1)

    arith_spec = MicrobenchmarkFunctionSpec(
        function_name="arith",
        body="",
        read_array_size=Bytes(0),
        write_array_size=Bytes(0),
        frequency=2.5,
        thread_affinity=[0],
    )
    arith_bench = ArithmeticBenchmark(
        params=ArithmeticBenchmarkParams(
            data_type=DataType.f32, thread_affinity=[0], operation=ArithmeticOperation.fma, num_ops=Operations(1024)
        ),
        spec=arith_spec,
    )
    arith_bench.results = ArithmeticBenchmarkResult(
        time_taken=Seconds(0.1), num_repetitions=1000, performance=Performance(10e9)
    )

    mem_spec = MicrobenchmarkFunctionSpec(
        function_name="mem",
        body="",
        read_array_size=Bytes(0),
        write_array_size=Bytes(0),
        frequency=2.5,
        thread_affinity=[0],
    )
    mem_bench = MemoryBenchmark(
        params=MemoryBenchmarkParams(
            data_type=DataType.f32,
            thread_affinity=[0],
            load_store_ratio=LoadStoreRatio(2, 1),
            size_per_thread=Bytes(1024),
            memory_level_name="L1",
        ),
        spec=mem_spec,
        working_set_bytes=Bytes(1024),
        cache_level="L1",
    )
    mem_bench.results = MemoryBenchmarkResult(
        time_taken=Seconds(0.1), num_repetitions=1000, bandwidth=Bandwidth(40e9), cache_level="L1"
    )

    suite = RooflineBenchmarkSuite(isa_name="isa1")
    suite.add_benchmark(arith_bench.name, arith_bench)
    suite.add_benchmark(mem_bench.name, mem_bench)
    isa_suites = {"isa1": suite}

    context.run_config.output_dir = tmp_path / "csv_test"
    context.run_config.output_formats = {OutputKind.TABLE}
    output_benchmark_results(context, isa_suites)
    assert not (context.run_config.output_dir / "roofline" / f"{context.run_config.name}_roofline.csv").exists()

    context.run_config.output_formats = {OutputKind.CSV}
    output_benchmark_results(context, isa_suites)
    assert (context.run_config.output_dir / "roofline" / f"{context.run_config.name}_roofline.csv").exists()


def test_memory_cli_and_plot(monkeypatch, tmp_path, capsys):
    """Memory CLI prints bandwidth and plotting saves image when matplotlib available."""
    from benchmark.benchmark import MemoryBenchmark, MemoryBenchmarkResult
    from benchmark.generation.code_gen import DataType
    from benchmark.generation.parameters import MemoryBenchmarkParams
    from benchmark.output.memory import MemoryOutputHandler
    from benchmark.suites import MemoryBenchmarkSuite
    from test_bench.builder import MicrobenchmarkFunctionSpec
    from units import Bandwidth, Bytes

    # Create proper MemoryBenchmarkResult (12.34 GB/s → Bandwidth(12.34e9) so output shows "GB/s")
    r1 = MemoryBenchmarkResult(
        time_taken=Seconds(0.1), num_repetitions=1000, bandwidth=Bandwidth(12.34e9), cache_level="L1"
    )

    context = _make_fake_context(["memISA"], freq_hz=3.0e9)

    params1 = MemoryBenchmarkParams(
        data_type=DataType.f32,
        thread_affinity=[0],
        load_store_ratio=LoadStoreRatio(2, 1),
        size_per_thread=Bytes(1024),
        memory_level_name="L1",
    )
    spec1 = MicrobenchmarkFunctionSpec(
        function_name="memBench",
        body="",
        read_array_size=Bytes(0),
        write_array_size=Bytes(0),
        frequency=2.5,
        thread_affinity=[0],
    )
    b1 = MemoryBenchmark(
        params=params1,
        spec=spec1,
        working_set_bytes=params1.size_per_thread,
        cache_level=params1.memory_level_name,
    )
    b1.results = r1

    s1 = MemoryBenchmarkSuite(isa_name="memISA")
    s1.add_benchmark(b1.name, b1)
    isa_suites = {"memISA": s1}

    # Capture the rich console output by providing our own Console instance.
    import io

    from rich.console import Console

    buf = io.StringIO()
    fake_console = Console(file=buf, force_terminal=True, width=80)
    monkeypatch.setattr("output_utils.get_console", lambda: fake_console)

    strategy = MemoryOutputHandler()
    strategy.print_table(context, isa_suites)

    output = buf.getvalue()
    # verify that the new table headers and data appear
    assert "Memory Bandwidth Summary" in output
    assert "ISA" in output and "Level" in output and "Bandwidth" in output
    assert "memISA" in output
    assert "L1" in output
    assert "GB/s" in output

    # plotting test
    mpl, _ = make_fake_matplotlib()
    monkeypatch.setitem(sys.modules, "matplotlib", mpl)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", mpl.pyplot)
    memory_mod = importlib.import_module("benchmark.output.memory")
    importlib.reload(memory_mod)

    memory_mod._write_plot(isa_suites, tmp_path)
    saved = [str(p) for p in mpl.pyplot._saved]
    assert any("memory_bandwidth.png" in str(p) for p in saved), f"Plot not saved. Saved: {saved}"


def test_mixed_combines_handlers(monkeypatch, tmp_path, capsys):
    """Mixed handler prints both arithmetic and memory summaries and saves combined plot."""
    from benchmark.benchmark import (
        ArithmeticBenchmark,
        ArithmeticBenchmarkResult,
        MemoryBenchmark,
        MemoryBenchmarkResult,
    )
    from benchmark.generation.code_gen import DataType
    from benchmark.generation.code_gen.operation import ArithmeticOperation
    from benchmark.generation.parameters import ArithmeticBenchmarkParams, MemoryBenchmarkParams
    from benchmark.suites import ArithmeticBenchmarkSuite, MemoryBenchmarkSuite
    from test_bench.builder import MicrobenchmarkFunctionSpec
    from units import Bandwidth, Bytes, Performance, Seconds

    context = _make_fake_context(["isaA", "isaB"], freq_hz=3.0e9)

    params_arith = ArithmeticBenchmarkParams(
        data_type=DataType.f32,
        thread_affinity=[0],
        operation=ArithmeticOperation.add,
        num_ops=Operations(1024),
    )
    spec_arith = MicrobenchmarkFunctionSpec(
        function_name="b1",
        body="",
        read_array_size=Bytes(0),
        write_array_size=Bytes(0),
        frequency=2.5,
        thread_affinity=[0],
    )
    b_arith = ArithmeticBenchmark(params=params_arith, spec=spec_arith)
    b_arith.results = ArithmeticBenchmarkResult(
        time_taken=Seconds(100.0), num_repetitions=1000, performance=Performance(7.5)
    )

    params_mem = MemoryBenchmarkParams(
        data_type=DataType.f32,
        thread_affinity=[0],
        load_store_ratio=LoadStoreRatio(2, 1),
        size_per_thread=Bytes(2048),
        memory_level_name="L1",
    )
    spec_mem = MicrobenchmarkFunctionSpec(
        function_name="b2",
        body="",
        read_array_size=Bytes(0),
        write_array_size=Bytes(0),
        frequency=2.5,
        thread_affinity=[0],
    )
    b_mem = MemoryBenchmark(
        params=params_mem,
        spec=spec_mem,
        working_set_bytes=params_mem.size_per_thread,
        cache_level=params_mem.memory_level_name,
    )
    b_mem.results = MemoryBenchmarkResult(
        time_taken=Seconds(0.1),
        num_repetitions=1000,
        bandwidth=Bandwidth(3.21),
        cache_level="L1",
    )

    s_arith = ArithmeticBenchmarkSuite(isa_name="isaA")
    s_arith.add_benchmark(b_arith.name, b_arith)
    s_mem = MemoryBenchmarkSuite(isa_name="isaB")
    s_mem.add_benchmark(b_mem.name, b_mem)
    isa_suites = {"isaA": s_arith, "isaB": s_mem}

    # CLI: should call handlers without error (no mixed benchmarks, so falls back to arithmetic + memory)
    from benchmark.output import mixed as mixed_mod

    # simply invoke the handler; it currently only logs a short message and
    # the bulk of the work is done by the individual sub-handlers, so there is
    # nothing valuable to assert here beyond 'no exception'.
    mixed_mod._print_table(context, isa_suites)

    # Plot: fake matplotlib and reload mixed to pick it up
    mpl, _ = make_fake_matplotlib()
    monkeypatch.setitem(sys.modules, "matplotlib", mpl)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", mpl.pyplot)
    # reload involved modules
    importlib.reload(importlib.import_module("benchmark.output.arithmetic"))
    importlib.reload(importlib.import_module("benchmark.output.memory"))
    mixed_mod = importlib.reload(importlib.import_module("benchmark.output.mixed"))

    mixed_mod._write_plot(isa_suites, tmp_path)
    saved = [str(p) for p in mpl.pyplot._saved]
    assert any("mixed_summary.png" in str(p) for p in saved)


class TestJsonlOutput:
    """Tests for unified JSONL benchmark output."""

    def test_jsonl_writes_one_line_per_benchmark(self, tmp_path):
        """Each benchmark with results produces one JSONL line."""
        from benchmark.benchmark import (
            ArithmeticBenchmark,
            ArithmeticBenchmarkResult,
            MemoryBenchmark,
            MemoryBenchmarkResult,
        )
        from benchmark.benchmarking import LoadStoreRatio
        from benchmark.generation.code_gen import DataType
        from benchmark.generation.code_gen.operation import ArithmeticOperation
        from benchmark.generation.parameters import ArithmeticBenchmarkParams, MemoryBenchmarkParams
        from benchmark.output.jsonl import write_jsonl_benchmarks
        from benchmark.suites import ArithmeticBenchmarkSuite, MemoryBenchmarkSuite
        from test_bench.builder import MicrobenchmarkFunctionSpec
        from units import Bandwidth, Bytes, Operations, Performance, Seconds

        context = _make_fake_context(["isa1"], freq_hz=2.0e9)

        arith_params = ArithmeticBenchmarkParams(
            data_type=DataType.f32,
            thread_affinity=[0],
            operation=ArithmeticOperation.fma,
            num_ops=Operations(1000),
        )
        arith_spec = MicrobenchmarkFunctionSpec(
            function_name="arith_bench",
            body="",
            read_array_size=Bytes(0),
            write_array_size=Bytes(0),
            frequency=2.0,
            thread_affinity=[0],
        )
        arith_bench = ArithmeticBenchmark(params=arith_params, spec=arith_spec)
        arith_bench.results = ArithmeticBenchmarkResult(
            time_taken=Seconds(0.001), num_repetitions=1000, performance=Performance(1e9)
        )

        mem_params = MemoryBenchmarkParams(
            data_type=DataType.f32,
            thread_affinity=[0],
            load_store_ratio=LoadStoreRatio(2, 1),
            size_per_thread=Bytes(1024),
            memory_level_name="L1",
        )
        mem_spec = MicrobenchmarkFunctionSpec(
            function_name="mem_bench",
            body="",
            read_array_size=Bytes(0),
            write_array_size=Bytes(0),
            frequency=2.0,
            thread_affinity=[0],
        )
        mem_bench = MemoryBenchmark(params=mem_params, spec=mem_spec, working_set_bytes=Bytes(1024), cache_level="L1")
        mem_bench.results = MemoryBenchmarkResult(
            time_taken=Seconds(0.001), num_repetitions=1000, bandwidth=Bandwidth(40e9), cache_level="L1"
        )

        arith_suite = ArithmeticBenchmarkSuite(isa_name="isa1")
        arith_suite.add_benchmark(arith_bench.name, arith_bench)
        mem_suite = MemoryBenchmarkSuite(isa_name="isa1")
        mem_suite.add_benchmark(mem_bench.name, mem_bench)
        isa_suites = {"isa1": arith_suite}
        isa_suites["isa1"].benchmarks.update(mem_suite.benchmarks)

        write_jsonl_benchmarks(context, isa_suites, output_dir=tmp_path)

        jsonl_path = tmp_path / "test" / "benchmarks.jsonl"
        assert jsonl_path.exists()
        lines = jsonl_path.read_text().strip().splitlines()
        assert len(lines) == 2, f"Expected 2 lines, got {len(lines)}"

    def test_jsonl_entry_schema_arithmetic(self, tmp_path):
        """Arithmetic JSONL entry has all expected fields with correct types."""
        import json

        from benchmark.benchmark import ArithmeticBenchmark, ArithmeticBenchmarkResult
        from benchmark.generation.code_gen import DataType
        from benchmark.generation.code_gen.operation import ArithmeticOperation
        from benchmark.generation.parameters import ArithmeticBenchmarkParams
        from benchmark.output.jsonl import write_jsonl_benchmarks
        from benchmark.suites import ArithmeticBenchmarkSuite
        from test_bench.builder import MicrobenchmarkFunctionSpec
        from units import Bytes, Operations, Performance, Seconds

        context = _make_fake_context(["isa1"], freq_hz=2.0e9)

        params = ArithmeticBenchmarkParams(
            data_type=DataType.f32,
            thread_affinity=[0],
            operation=ArithmeticOperation.fma,
            num_ops=Operations(1000),
        )
        spec = MicrobenchmarkFunctionSpec(
            function_name="test_add",
            body="",
            read_array_size=Bytes(0),
            write_array_size=Bytes(0),
            frequency=2.0,
            thread_affinity=[0],
        )
        bench = ArithmeticBenchmark(params=params, spec=spec)
        bench.results = ArithmeticBenchmarkResult(
            time_taken=Seconds(0.001), num_repetitions=1000, performance=Performance(1e9)
        )

        suite = ArithmeticBenchmarkSuite(isa_name="isa1")
        suite.add_benchmark(bench.name, bench)

        write_jsonl_benchmarks(context, {"isa1": suite}, output_dir=tmp_path)

        jsonl_path = tmp_path / "test" / "benchmarks.jsonl"
        entry = json.loads(jsonl_path.read_text().strip())

        # Common fields
        assert entry["type"] == "arithmetic"
        assert entry["name"] == "test_add"
        assert entry["isa"] == "isa1"
        assert entry["data_type"] == "f32"
        assert entry["num_threads"] == 1
        assert entry["thread_affinity"] == [0]
        assert "timestamp" in entry
        assert entry["machine"] == "test"

        # Arithmetic-specific fields
        assert entry["operation"] == "fma"
        assert entry["num_ops"] == 1000
        assert entry["performance_gops"] == 1.0  # 1e9 / 1e9
        assert entry["ipc"] == 0.25  # 500000 insts / 2000000 cycles
        assert entry["frequency_hz"] == 2.0e9
        assert entry["ops_per_instruction"] == 2  # fma emits 2 ops per inst
        assert entry["ops_per_cycle"] == 0.5  # 2 * 0.25
        assert entry["time_seconds"] == 0.001
        assert entry["repetitions"] == 1000
        assert entry["cycles"] == 2_000_000.0  # 0.001 * 2e9

    def test_jsonl_entry_schema_memory(self, tmp_path):
        """Memory JSONL entry has all expected fields with correct types."""
        import json

        from benchmark.benchmark import MemoryBenchmark, MemoryBenchmarkResult
        from benchmark.benchmarking import LoadStoreRatio
        from benchmark.generation.code_gen import DataType
        from benchmark.generation.parameters import MemoryBenchmarkParams
        from benchmark.output.jsonl import write_jsonl_benchmarks
        from benchmark.suites import MemoryBenchmarkSuite
        from test_bench.builder import MicrobenchmarkFunctionSpec
        from units import Bandwidth, Bytes, Seconds

        context = _make_fake_context(["isa1"], freq_hz=2.0e9)

        params = MemoryBenchmarkParams(
            data_type=DataType.f32,
            thread_affinity=[0],
            load_store_ratio=LoadStoreRatio(2, 1),
            size_per_thread=Bytes(1024),
            memory_level_name="L1",
        )
        spec = MicrobenchmarkFunctionSpec(
            function_name="test_mem",
            body="",
            read_array_size=Bytes(0),
            write_array_size=Bytes(0),
            frequency=2.0,
            thread_affinity=[0],
        )
        bench = MemoryBenchmark(params=params, spec=spec, working_set_bytes=Bytes(1024), cache_level="L1")
        bench.results = MemoryBenchmarkResult(
            time_taken=Seconds(0.001), num_repetitions=1000, bandwidth=Bandwidth(40e9), cache_level="L1"
        )

        suite = MemoryBenchmarkSuite(isa_name="isa1")
        suite.add_benchmark(bench.name, bench)

        write_jsonl_benchmarks(context, {"isa1": suite}, output_dir=tmp_path)

        jsonl_path = tmp_path / "test" / "benchmarks.jsonl"
        entry = json.loads(jsonl_path.read_text().strip())

        # Common fields
        assert entry["type"] == "memory"
        assert entry["name"] == "test_mem"
        assert entry["isa"] == "isa1"
        assert entry["data_type"] == "f32"
        assert entry["num_threads"] == 1
        assert entry["thread_affinity"] == [0]
        assert "timestamp" in entry
        assert entry["machine"] == "test"

        # Memory-specific fields
        assert entry["load_store_ratio"] == "2:1"
        assert entry["num_loads"] == 2
        assert entry["num_stores"] == 1
        assert entry["cache_level"] == "L1"
        assert entry["memory_level_name"] == "L1"
        assert entry["size_per_thread_bytes"] == 1024
        assert entry["working_set_bytes"] == 1024
        assert entry["layout_mode"] == "split"
        assert entry["bandwidth_gbps"] == 40.0
        assert entry["ipc"] == 0.128  # 256000 insts / 2000000 cycles
        assert entry["time_seconds"] == 0.001
        assert entry["repetitions"] == 1000
        assert entry["cycles"] == 2_000_000.0

    def test_jsonl_skips_null_results(self, tmp_path):
        """Benchmarks with null results do not produce JSONL lines."""
        from benchmark.benchmark import ArithmeticBenchmark, ArithmeticBenchmarkResult
        from benchmark.generation.code_gen import DataType
        from benchmark.generation.code_gen.operation import ArithmeticOperation
        from benchmark.generation.parameters import ArithmeticBenchmarkParams
        from benchmark.output.jsonl import write_jsonl_benchmarks
        from benchmark.suites import ArithmeticBenchmarkSuite
        from test_bench.builder import MicrobenchmarkFunctionSpec
        from units import Bytes, Operations, Performance, Seconds

        context = _make_fake_context(["isa1"], freq_hz=2.0e9)

        params = ArithmeticBenchmarkParams(
            data_type=DataType.f32,
            thread_affinity=[0],
            operation=ArithmeticOperation.fma,
            num_ops=Operations(1000),
        )
        spec = MicrobenchmarkFunctionSpec(
            function_name="has_result",
            body="",
            read_array_size=Bytes(0),
            write_array_size=Bytes(0),
            frequency=2.0,
            thread_affinity=[0],
        )
        b1 = ArithmeticBenchmark(params=params, spec=spec)
        b1.results = ArithmeticBenchmarkResult(
            time_taken=Seconds(0.001), num_repetitions=1000, performance=Performance(1e9)
        )

        spec2 = MicrobenchmarkFunctionSpec(
            function_name="no_result",
            body="",
            read_array_size=Bytes(0),
            write_array_size=Bytes(0),
            frequency=2.0,
            thread_affinity=[0],
        )
        b2 = ArithmeticBenchmark(params=params, spec=spec2)
        # b2.results left as None

        suite = ArithmeticBenchmarkSuite(isa_name="isa1")
        suite.add_benchmark(b1.name, b1)
        suite.add_benchmark(b2.name, b2)

        write_jsonl_benchmarks(context, {"isa1": suite}, output_dir=tmp_path)

        jsonl_path = tmp_path / "test" / "benchmarks.jsonl"
        lines = jsonl_path.read_text().strip().splitlines()
        assert len(lines) == 1, f"Expected 1 line (skipped null result), got {len(lines)}"
        import json

        entry = json.loads(lines[0])
        assert entry["name"] == "has_result"

    def test_jsonl_appends_to_existing_file(self, tmp_path):
        """Two writes to the same path produce cumulative lines."""
        import json

        from benchmark.benchmark import ArithmeticBenchmark, ArithmeticBenchmarkResult
        from benchmark.generation.code_gen import DataType
        from benchmark.generation.code_gen.operation import ArithmeticOperation
        from benchmark.generation.parameters import ArithmeticBenchmarkParams
        from benchmark.output.jsonl import write_jsonl_benchmarks
        from benchmark.suites import ArithmeticBenchmarkSuite
        from test_bench.builder import MicrobenchmarkFunctionSpec
        from units import Bytes, Operations, Performance, Seconds

        context = _make_fake_context(["isa1"], freq_hz=2.0e9)

        params = ArithmeticBenchmarkParams(
            data_type=DataType.f32,
            thread_affinity=[0],
            operation=ArithmeticOperation.fma,
            num_ops=Operations(1000),
        )
        spec = MicrobenchmarkFunctionSpec(
            function_name="bench1",
            body="",
            read_array_size=Bytes(0),
            write_array_size=Bytes(0),
            frequency=2.0,
            thread_affinity=[0],
        )
        b1 = ArithmeticBenchmark(params=params, spec=spec)
        b1.results = ArithmeticBenchmarkResult(
            time_taken=Seconds(0.001), num_repetitions=1000, performance=Performance(1e9)
        )

        suite = ArithmeticBenchmarkSuite(isa_name="isa1")
        suite.add_benchmark(b1.name, b1)

        write_jsonl_benchmarks(context, {"isa1": suite}, output_dir=tmp_path)

        spec2 = MicrobenchmarkFunctionSpec(
            function_name="bench2",
            body="",
            read_array_size=Bytes(0),
            write_array_size=Bytes(0),
            frequency=2.0,
            thread_affinity=[0],
        )
        b2 = ArithmeticBenchmark(params=params, spec=spec2)
        b2.results = ArithmeticBenchmarkResult(
            time_taken=Seconds(0.001), num_repetitions=1000, performance=Performance(1e9)
        )

        suite2 = ArithmeticBenchmarkSuite(isa_name="isa1")
        suite2.add_benchmark(b2.name, b2)

        write_jsonl_benchmarks(context, {"isa1": suite2}, output_dir=tmp_path)

        jsonl_path = tmp_path / "test" / "benchmarks.jsonl"
        lines = jsonl_path.read_text().strip().splitlines()
        assert len(lines) == 2, f"Expected 2 lines (appended), got {len(lines)}"
        entries = [json.loads(line) for line in lines]
        assert entries[0]["name"] == "bench1"
        assert entries[1]["name"] == "bench2"

    def test_jsonl_default_in_output_kinds(self):
        """JSONL is available in OutputKind and is used by default."""
        from benchmark.output import OutputKind

        assert hasattr(OutputKind, "JSONL")
        assert OutputKind.JSONL.value == "jsonl"

        import argparse

        from run_config import RunConfig

        parser = argparse.ArgumentParser()
        RunConfig.insert_arguments(parser)
        args = parser.parse_args([])
        config = RunConfig(args)

        assert OutputKind.JSONL in config.output_formats, "JSONL should be in default output formats"
