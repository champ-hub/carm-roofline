from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

import pytest
from carm_roofline.profiling.dynamorio_loader import load_dynamorio_results

pytestmark = [pytest.mark.integration, pytest.mark.native]


_FMA_SMOKE_SOURCE = r'''#include "carm_region.h"
volatile unsigned int carm_test_slots[4];
#define FMA3(stem, form) \
    __asm__ volatile("v" stem form "ps %%ymm1, %%ymm2, %%ymm3" ::: "ymm1", "ymm2", "ymm3")
#define READ(slot) \
    __asm__ volatile("movl carm_test_slots+" #slot "(%%rip), %%eax" ::: "eax", "memory")
#define WRITE(slot) \
    __asm__ volatile("movl $1, carm_test_slots+" #slot "(%%rip)" ::: "memory")
int main(void)
{
    carm_region_set_process_identity(0, 1);
    carm_region_set_thread_identity(0, 1);
    carm_region_begin("outer");
    READ(0);
    FMA3("fmadd", "132"); FMA3("fmadd", "213"); FMA3("fmadd", "231");
    FMA3("fmsub", "132"); FMA3("fmsub", "213"); FMA3("fmsub", "231");
    FMA3("fnmadd", "132"); FMA3("fnmadd", "213"); FMA3("fnmadd", "231");
    FMA3("fnmsub", "132"); FMA3("fnmsub", "213"); FMA3("fnmsub", "231");
    FMA3("fmaddsub", "132"); FMA3("fmaddsub", "213"); FMA3("fmaddsub", "231");
    FMA3("fmsubadd", "132"); FMA3("fmsubadd", "213"); FMA3("fmsubadd", "231");
    __asm__ volatile("vmaxps %%ymm1, %%ymm2, %%ymm3" ::: "ymm1", "ymm2", "ymm3");
    carm_region_begin("inner");
    WRITE(1);
    FMA3("fmadd", "132");
    carm_region_end("inner");
    READ(2);
    carm_region_end("outer");
    return 0;
}
'''


def test_dynamorio_region_accounting_and_fma_opcode_classification(tmp_path: Path) -> None:
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        pytest.skip("requires x86-64 and AVX/FMA")
    cpuinfo = Path("/proc/cpuinfo")
    cpu_flags = f" {cpuinfo.read_text(encoding='utf-8')} " if cpuinfo.is_file() else ""
    if any(f" {flag} " not in cpu_flags for flag in ("avx", "fma")):
        pytest.skip("requires an AVX/FMA-capable host")

    dynamorio_home = Path(os.environ.get("DYNAMORIO_HOME", ""))
    cmake = shutil.which("cmake")
    compiler = shutil.which("cc")
    cxx_compiler = shutil.which("c++")
    drrun = dynamorio_home / "bin64" / "drrun"
    dynamorio_config = dynamorio_home / "cmake" / "DynamoRIOConfig.cmake"
    if (
        not drrun.is_file()
        or not dynamorio_config.is_file()
        or cmake is None
        or compiler is None
        or cxx_compiler is None
    ):
        pytest.skip("requires DYNAMORIO_HOME, CMake, and C/C++ compilers")

    repo_root = Path(__file__).parents[1]
    client_source = repo_root / "carm_roofline" / "native" / "dynamorio"
    build_dir = tmp_path / "client-build"
    client_fixture = Path(__file__).parent / "fixtures" / "dynamorio_fma4_decoder_client.c"
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\n"
        "project(carm_dynamorio_fma4_test C CXX)\n"
        "find_package(DynamoRIO REQUIRED)\n"
        f'add_library(carm_dynamorio_fma4_test SHARED "{client_fixture}")\n'
        "configure_DynamoRIO_client(carm_dynamorio_fma4_test)\n"
        "use_DynamoRIO_extension(carm_dynamorio_fma4_test drmgr)\n"
        "use_DynamoRIO_extension(carm_dynamorio_fma4_test drwrap)\n",
        encoding="utf-8",
    )
    subprocess.run(
        [cmake, "-S", str(tmp_path), "-B", str(build_dir), f"-DDynamoRIO_DIR={dynamorio_home / 'cmake'}"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([cmake, "--build", str(build_dir)], check=True, capture_output=True, text=True)

    source = tmp_path / "fma-smoke.c"
    source.write_text(_FMA_SMOKE_SOURCE, encoding="utf-8")
    app = tmp_path / "fma-smoke"
    subprocess.run(
        [
            compiler,
            "-O2",
            "-mavx",
            "-mfma",
            "-rdynamic",
            f"-I{client_source}",
            str(source),
            str(client_source / "carm_region_timing.c"),
            "-o",
            str(app),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    output = tmp_path / "counts.jsonl"
    client = build_dir / "libcarm_dynamorio_fma4_test.so"
    subprocess.run([str(drrun), "-c", str(client), "-o", str(output), "--", str(app)], check=True)
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert {record["name"] for record in records} == {"outer", "inner"}
    counts = {record["name"]: record for record in records}
    assert counts["outer"]["instruction_counts"] == {"fma_256_f32": 18}
    assert counts["outer"]["read_bytes"] == 8
    assert counts["outer"]["write_bytes"] == 0
    assert counts["inner"]["instruction_counts"] == {"fma_256_f32": 1}
    assert counts["inner"]["read_bytes"] == 0
    assert counts["inner"]["write_bytes"] == 4
    timing_output = tmp_path / "timings.jsonl"
    timing_env = os.environ.copy()
    timing_env["CARM_REGION_TIMING_OUTPUT"] = str(timing_output)
    subprocess.run([str(app)], env=timing_env, check=True)
    ranks = load_dynamorio_results(output, timing_output)
    loaded_regions = {region.name: region for region in ranks[0].threads[0].regions}
    assert set(loaded_regions) == {"outer", "inner"}
    assert loaded_regions["outer"].counters["dr_fma_256_f32"] == 18
    assert loaded_regions["outer"].counters["dr_read_bytes"] == 8
    assert loaded_regions["outer"].counters["dr_write_bytes"] == 0
    assert loaded_regions["inner"].counters["dr_fma_256_f32"] == 1
    assert loaded_regions["inner"].counters["dr_read_bytes"] == 0
    assert loaded_regions["inner"].counters["dr_write_bytes"] == 4
    assert all(region.time_nsec >= 0 for region in loaded_regions.values())
