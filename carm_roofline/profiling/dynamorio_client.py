from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

from carm_roofline.core import UserError
from carm_roofline.results_paths import user_cache_dir_for_carm

_CLIENT_NAME = "libcarm_dynamorio_client.so"
_BUILD_DIR_NAME = "dynamorio-client"


_BUILD_INPUTS = ("CMakeLists.txt", "carm_dynamorio_client.c")


_BUILD_ENVIRONMENT = (
    "CC",
    "CXX",
    "CFLAGS",
    "CXXFLAGS",
    "CPPFLAGS",
    "LDFLAGS",
    "CMAKE_GENERATOR",
    "CMAKE_GENERATOR_PLATFORM",
    "CMAKE_GENERATOR_TOOLSET",
    "CMAKE_MODULE_PATH",
    "CMAKE_PREFIX_PATH",
    "CMAKE_TOOLCHAIN_FILE",
)


def _runtime_fingerprint(drrun: Path) -> str:
    """Describe the selected DynamoRIO executable and SDK files for cache identity."""
    installation = drrun.resolve().parent.parent
    try:
        version = subprocess.run(
            [str(drrun), "-version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise UserError(f"Unable to query DynamoRIO version using {drrun}: {exc}") from exc
    if version.returncode != 0:
        raise UserError(
            f"Unable to identify DynamoRIO installation using {drrun} (exit code {version.returncode}): "
            f"{version.stderr.strip() or version.stdout.strip()}"
        )

    digest = hashlib.sha256()
    digest.update(str(installation).encode())
    digest.update(version.stdout.strip().encode())
    digest.update(version.stderr.strip().encode())
    sdk_paths = (
        drrun.resolve(),
        installation / "cmake",
        installation / "include",
        installation / "ext" / "include",
        installation / "lib64" / "release" / "libdynamorio.so",
        installation / "ext" / "lib64" / "release" / "libdrmgr.so",
        installation / "ext" / "lib64" / "release" / "libdrwrap.so",
    )
    for root in sdk_paths:
        paths = sorted(root.rglob("*")) if root.is_dir() else [root]
        if not root.exists():
            raise UserError(f"DynamoRIO SDK/runtime input is missing: {root}")
        for path in paths:
            if path.is_file():
                try:
                    contents = path.read_bytes()
                except OSError as exc:
                    raise UserError(f"Unable to read DynamoRIO SDK/runtime input {path}: {exc}") from exc
                digest.update(path.relative_to(installation).as_posix().encode())
                digest.update(contents)
    return digest.hexdigest()


def _client_identity(source: Path, drrun: Path) -> str:
    digest = hashlib.sha256()
    digest.update(platform.platform().encode())
    digest.update(platform.machine().encode())
    digest.update(_runtime_fingerprint(drrun).encode())
    environment = {key: os.environ.get(key) for key in _BUILD_ENVIRONMENT}
    toolchain_file = environment.get("CMAKE_TOOLCHAIN_FILE")
    if toolchain_file:
        toolchain_path = Path(toolchain_file).expanduser()
        try:
            digest.update(toolchain_path.resolve().read_bytes())
        except OSError as exc:
            raise UserError(f"Unable to read configured CMake toolchain file {toolchain_path}: {exc}") from exc
    digest.update(json.dumps(environment, sort_keys=True).encode())
    digest.update(b"cmake:-DCMAKE_BUILD_TYPE=Release")
    for relative_path in _BUILD_INPUTS:
        path = source / relative_path
        try:
            contents = path.read_bytes()
        except OSError as exc:
            raise UserError(f"Bundled DynamoRIO client build input is unavailable: {path}: {exc}") from exc
        digest.update(relative_path.encode())
        digest.update(contents)
    return digest.hexdigest()


def _run_build(command: list[str], *, cwd: Path, step: str) -> None:
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise UserError(
            f"Cannot build the bundled DynamoRIO client: {step} requires {command[0]}. Install it and retry."
        ) from exc
    except OSError as exc:
        raise UserError(f"Cannot build the bundled DynamoRIO client: failed to start {step}: {exc}") from exc
    if result.returncode:
        output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
        raise UserError(
            f"Cannot build the bundled DynamoRIO client: {step} failed (exit code {result.returncode}).\n{output}"
        )


def _build_client(source: Path, drrun: Path, staging: Path) -> Path:
    cmake = shutil.which("cmake")
    if cmake is None:
        raise UserError(
            "Cannot build the bundled DynamoRIO client: CMake was not found on PATH. Install CMake and retry."
        )
    build = staging / "build"
    installation = drrun.resolve().parent.parent
    cmake_dir = installation / "cmake"
    if not cmake_dir.is_dir():
        raise UserError(
            f"DynamoRIO SDK was not found at {cmake_dir} (selected drrun: {drrun}). "
            "Install the DynamoRIO SDK or select a complete installation with --dynamorio-home."
        )
    _run_build(
        [cmake, "-S", str(source), "-B", str(build), "-DDynamoRIO_DIR=" + str(cmake_dir), "-DCMAKE_BUILD_TYPE=Release"],
        cwd=staging,
        step="CMake configure",
    )
    _run_build(
        [cmake, "--build", str(build), "--config", "Release", "--parallel"], cwd=staging, step="client compile/link"
    )
    candidates = tuple(build.rglob(_CLIENT_NAME))
    if len(candidates) != 1 or not candidates[0].is_file():
        raise UserError(
            f"DynamoRIO client build completed but {_CLIENT_NAME} was not produced exactly once in {build}."
        )
    return candidates[0]


def get_or_build_client(drrun: Path) -> Path:
    """Return the bundled DynamoRIO client, building and caching it atomically if needed."""
    source = Path(__file__).resolve().parents[1] / "native" / "dynamorio"
    if not (source / "CMakeLists.txt").is_file():
        raise UserError(f"Bundled DynamoRIO client build inputs are missing: {source}")
    cache_root = user_cache_dir_for_carm() / _BUILD_DIR_NAME
    try:
        identity = _client_identity(source, drrun)
        cache_root.mkdir(parents=True, exist_ok=True)
        final_dir = cache_root / identity
        client = final_dir / _CLIENT_NAME
        if client.is_file():
            return client
        lock_path = cache_root / f"{identity}.lock"
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if client.is_file():
                return client
            staging = Path(tempfile.mkdtemp(prefix=f".{identity}.", dir=cache_root))
            try:
                built = _build_client(source, drrun, staging)
                published = staging / "published"
                published.mkdir()
                shutil.copy2(built, published / _CLIENT_NAME)
                if final_dir.exists() or final_dir.is_symlink():
                    if final_dir.is_dir() and not final_dir.is_symlink():
                        shutil.rmtree(final_dir)
                    else:
                        final_dir.unlink()
                os.replace(published, final_dir)
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        return client
    except UserError:
        raise
    except OSError as exc:
        raise UserError(f"Unable to use DynamoRIO client cache at {cache_root}: {exc}") from exc
