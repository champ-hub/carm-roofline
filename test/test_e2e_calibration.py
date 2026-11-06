from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.native]


@pytest.mark.skipif(shutil.which("gcc") is None, reason="requires gcc to compile the native harness")
def test_native_harness_scales_down_slow_initial_calibration_and_keeps_normal_case(tmp_path: Path) -> None:
    """Exercise calibration through the real C harness with controlled pass costs."""
    harness_dir = Path(__file__).parents[1] / "carm_roofline" / "test_bench"
    header = tmp_path / "microbenchmarks.h"
    header.write_text(
        '#include "test_bench.h"\n'
        "#include <stdatomic.h>\n"
        "static _Atomic unsigned int next_thread_role;\n"
        "static __thread int thread_role = -1;\n"
        "static void delay_for_reps(uint64_t reps, uint64_t ns_per_rep) {\n"
        "    struct timespec delay = {0, (long)(reps * ns_per_rep)};\n"
        "    while (nanosleep(&delay, &delay) != 0) {}\n"
        "}\n"
        "static void slow_initial_pass(void *read_ptr, void *write_ptr, uint64_t reps) {\n"
        "    (void)read_ptr; (void)write_ptr; delay_for_reps(reps, 250000000ULL);\n"
        "}\n"
        "static void normal_pass(void *read_ptr, void *write_ptr, uint64_t reps) {\n"
        "    (void)read_ptr; (void)write_ptr; delay_for_reps(reps, 1000000ULL);\n"
        "}\n"
        "static void heterogeneous_pass(void *read_ptr, void *write_ptr, uint64_t reps) {\n"
        "    (void)read_ptr; (void)write_ptr;\n"
        "    if (thread_role < 0) thread_role = atomic_fetch_add(&next_thread_role, 1) % 2;\n"
        "    delay_for_reps(reps, thread_role == 0 ? 50000000ULL : 1000000ULL);\n"
        "}\n"
        "#define UBENCH_NAME slow_initial_pass\n"
        '#include "wrapper.inl"\n'
        "#undef UBENCH_NAME\n"
        "#define UBENCH_NAME normal_pass\n"
        '#include "wrapper.inl"\n'
        "#undef UBENCH_NAME\n"
        "#define UBENCH_NAME heterogeneous_pass\n"
        '#include "wrapper.inl"\n'
        "#undef UBENCH_NAME\n"
        "static const benchmark_metadata_t metadata_slow = {\n"
        '    .name = "slow_initial", .frequency_ghz = 1.0f, .thread_affinity = NULL,\n'
        "    .num_threads = 2, .read_array_size_bytes = 0, .write_array_size_bytes = 0\n"
        "};\n"
        "static const benchmark_metadata_t metadata_normal = {\n"
        '    .name = "normal", .frequency_ghz = 1.0f, .thread_affinity = NULL,\n'
        "    .num_threads = 2, .read_array_size_bytes = 0, .write_array_size_bytes = 0\n"
        "};\n"
        "static const benchmark_metadata_t metadata_heterogeneous = {\n"
        '    .name = "heterogeneous", .frequency_ghz = 1.0f, .thread_affinity = NULL,\n'
        "    .num_threads = 2, .read_array_size_bytes = 0, .write_array_size_bytes = 0\n"
        "};\n"
        "#define MICROBENCHMARK_LIST \\\n"
        "    X(wrapper_slow_initial_pass, &metadata_slow) \\\n"
        "    X(wrapper_normal_pass, &metadata_normal) \\\n"
        "    X(wrapper_heterogeneous_pass, &metadata_heterogeneous)\n"
    )
    binary = tmp_path / "calibration_harness"
    subprocess.run(
        [
            "gcc",
            "-DVERBOSITY=0",
            "-DNUM_RUNS=1",
            "-pthread",
            f"-I{tmp_path}",
            f"-I{harness_dir}",
            str(harness_dir / "test_bench.c"),
            "-lm",
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=10)
    records = {
        name: (float(milliseconds), int(repetitions))
        for name, milliseconds, repetitions in re.findall(
            r"^(slow_initial|normal|heterogeneous), ([0-9.]+), (\d+)$", result.stdout, re.MULTILINE
        )
    }

    assert set(records) == {"slow_initial", "normal", "heterogeneous"}
    _, slow_reps = records["slow_initial"]
    _, normal_reps = records["normal"]
    _, heterogeneous_reps = records["heterogeneous"]
    # One 250ms initial pass already exceeds calibration time; elapsed-time scaling
    # must select one repetition rather than multiplying it against the 60ms threshold.
    assert slow_reps == 1
    # A normal 1ms pass begins below threshold and still calibrates upward.
    # The injected 1GHz metadata need not match this host's TSC.
    assert 2 <= normal_reps <= 120
    # With one 50ms/pass thread and one 1ms/pass thread, the slower thread
    # controls the common count. Selecting the fast thread's count overshoots.
    assert 1 <= heterogeneous_reps <= 2
