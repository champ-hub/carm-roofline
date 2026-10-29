from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from carm_roofline.core import UserError
from carm_roofline.profiling import dynamorio_client
from carm_roofline.profiling.backends import RunSpec
from carm_roofline.profiling.dynamorio_backend import DynamoRIOBackend
from carm_roofline.profiling.shared import MetricResolutionConfig

pytestmark = [pytest.mark.integration, pytest.mark.native]


def _drrun() -> Path:
    home = os.environ.get("DYNAMORIO_HOME")
    candidate = Path(home) / "bin64" / "drrun" if home else None
    if candidate is not None and candidate.is_file():
        return candidate
    found = shutil.which("drrun")
    if found:
        return Path(found)
    pytest.skip("requires a DynamoRIO 11+ installation")




def test_sdk_replacement_rebuilds_cached_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installation = tmp_path / "dynamorio"
    drrun = installation / "bin64" / "drrun"
    drrun.parent.mkdir(parents=True)
    drrun.write_text(
        "#!/bin/sh\nprintf '%s\\n' 'DynamoRIO version 11.3.0'\n",
        encoding="utf-8",
    )
    drrun.chmod(0o755)
    sdk_input = installation / "cmake" / "DynamoRIOConfig.cmake"
    sdk_input.parent.mkdir(parents=True)
    sdk_input.write_bytes(b"original SDK")
    for relative in (
        "include/dr_api.h",
        "ext/include/drmgr.h",
        "lib64/release/libdynamorio.so",
        "ext/lib64/release/libdrmgr.so",
        "ext/lib64/release/libdrwrap.so",
    ):
        path = installation / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"SDK component")

    package = tmp_path / "carm_roofline"
    source = package / "native" / "dynamorio"
    source.mkdir(parents=True)
    (source / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.16)", encoding="utf-8")
    (source / "carm_dynamorio_client.c").write_text("/* client */", encoding="utf-8")
    fake_module = package / "profiling" / "dynamorio_client.py"
    fake_module.parent.mkdir()
    monkeypatch.setattr(dynamorio_client, "__file__", str(fake_module))
    monkeypatch.setattr(dynamorio_client, "user_cache_dir_for_carm", lambda: tmp_path / "cache")

    build_count = 0

    def fake_build(_source: Path, _drrun: Path, staging: Path) -> Path:
        nonlocal build_count
        build_count += 1
        artifact = staging / "built-client.so"
        artifact.write_bytes(f"build {build_count}".encode())
        return artifact

    monkeypatch.setattr(dynamorio_client, "_build_client", fake_build)
    original_client = dynamorio_client.get_or_build_client(drrun)
    assert original_client.is_file()
    assert dynamorio_client.get_or_build_client(drrun) == original_client
    assert build_count == 1

    sdk_input.write_bytes(b"replacement SDK, same reported version")
    replacement_client = dynamorio_client.get_or_build_client(drrun)
    assert replacement_client.is_file()
    assert replacement_client != original_client
    assert build_count == 2


def _client_source() -> Path:
    return Path(dynamorio_client.__file__).resolve().parents[1] / "native" / "dynamorio"


def test_cache_reuses_build_and_invalidates_changed_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    drrun = _drrun()
    if any(shutil.which(name) is None for name in ("cmake", "cc", "c++")):
        pytest.skip("requires CMake and C/C++ compilers")

    source = _client_source()
    copied_source = tmp_path / "carm_roofline" / "native" / "dynamorio"
    shutil.copytree(source, copied_source)
    fake_module = tmp_path / "carm_roofline" / "profiling" / "dynamorio_client.py"
    fake_module.parent.mkdir(parents=True)
    monkeypatch.setattr(dynamorio_client, "__file__", str(fake_module))
    monkeypatch.setattr(dynamorio_client, "user_cache_dir_for_carm", lambda: tmp_path / "cache")

    first = dynamorio_client.get_or_build_client(drrun)
    original_mtime = first.stat().st_mtime_ns
    assert first.is_file()
    assert dynamorio_client.get_or_build_client(drrun) == first
    assert first.stat().st_mtime_ns == original_mtime

    with monkeypatch.context() as without_build_tools:
        without_build_tools.setenv("PATH", str(tmp_path / "path-without-build-tools"))
        without_build_tools.setattr(dynamorio_client.shutil, "which", lambda _name: None)
        assert dynamorio_client.get_or_build_client(drrun) == first

    app_source = tmp_path / "profile.c"
    app_source.write_text(
        '#include "carm_region.h"\n'
        "volatile double value = 2.0;\n"
        "int main(void) {\n"
        "  carm_region_set_process_identity(0, 1);\n"
        "  carm_region_set_thread_identity(0, 1);\n"
        '  carm_region_begin("kernel");\n'
        "  value = value * 3.0 + 1.0;\n"
        '  carm_region_end("kernel");\n'
        "  return 0;\n"
        "}\n",
        encoding="utf-8",
    )
    app = tmp_path / "profile-app"
    compiler = shutil.which("cc")
    assert compiler is not None
    subprocess.run(
        [
            compiler,
            "-O0",
            "-rdynamic",
            f"-I{copied_source}",
            str(app_source),
            str(copied_source / "carm_region_timing.c"),
            "-o",
            str(app),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    backend = DynamoRIOBackend(MetricResolutionConfig(), dynamorio_home=drrun.parents[1])
    backend.check_prerequisites()
    result = backend.profile(RunSpec(tmp_path / "profile-output"), [str(app)], tmp_path)
    assert {region.name for region in result.ranks[0].threads[0].regions} == {"kernel"}
    first.unlink()
    recovered = dynamorio_client.get_or_build_client(drrun)
    assert recovered == first
    assert recovered.is_file()

    cmake_lists = copied_source / "CMakeLists.txt"
    cmake_lists.write_bytes(cmake_lists.read_bytes() + b"\n")
    rebuilt = dynamorio_client.get_or_build_client(drrun)
    assert rebuilt != first
    assert rebuilt.is_file()
    assert dynamorio_client.get_or_build_client(drrun) == rebuilt


def test_concurrent_cache_misses_publish_one_usable_client(tmp_path: Path) -> None:
    drrun = _drrun()
    if any(shutil.which(name) is None for name in ("cmake", "cc", "c++")):
        pytest.skip("requires CMake and C/C++ compilers")

    cache_home = tmp_path / "cache-home"
    home = tmp_path / "home"
    home.mkdir()
    script = (
        "from pathlib import Path; "
        "from carm_roofline.profiling.dynamorio_client import get_or_build_client; "
        f"print(get_or_build_client(Path({str(drrun)!r})))"
    )
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["XDG_CACHE_HOME"] = str(cache_home)
    python_path = str(Path(__file__).parents[2])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (python_path, env.get("PYTHONPATH"))))
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script],
            cwd=Path(__file__).parents[2],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    results = [process.communicate(timeout=180) for process in processes]
    assert [process.returncode for process in processes] == [0, 0], [stderr for _, stderr in results]
    client_paths = [Path(stdout.strip()) for stdout, _ in results]
    assert client_paths[0] == client_paths[1]
    assert client_paths[0].is_file()


def test_missing_cmake_reports_an_actionable_build_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    drrun = _drrun()
    monkeypatch.setattr(dynamorio_client, "user_cache_dir_for_carm", lambda: tmp_path / "cache")
    monkeypatch.setattr(dynamorio_client.shutil, "which", lambda _name: None)

    with pytest.raises(UserError, match="CMake was not found on PATH"):
        dynamorio_client.get_or_build_client(drrun)
