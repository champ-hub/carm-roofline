from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
from argparse import ArgumentTypeError
from typing import Any

from carm_roofline.arguments import InsertsArguments
from carm_roofline.output_utils import debug, warn


def _existing_executable(arg: str, option_name: str) -> str:
    """Validate that the given executable can be resolved in PATH."""
    if shutil.which(arg) is None:
        raise ArgumentTypeError(f"{option_name} executable {arg!r} not found in PATH")
    return arg


def _sim_cmd_type(arg: str) -> str:
    """Validate simulator command by checking the first executable token."""
    try:
        split_cmd = shlex.split(arg)
    except ValueError as exc:
        raise ArgumentTypeError(f"Invalid --sim-cmd value: {arg!r}") from exc

    if not split_cmd:
        raise ArgumentTypeError("--sim-cmd must not be empty")

    _existing_executable(split_cmd[0], "--sim-cmd")
    return arg


def _compiler_type(arg: str) -> str:
    """Validate compiler executable for --compiler."""
    return _existing_executable(arg, "--compiler")


class ExecutionInterface(InsertsArguments):
    """Manages command execution for commands, allowing optional simulator/emulator prefix,
    and optional cross-compiler for building tests.

    Supports command templates with {binary} placeholder:
        --sim-cmd "sde -mix -- {binary}"      # Intel SDE
        --sim-cmd "qemu-x86_64 {binary}"      # QEMU
        --compiler "riscv64-linux-gnu-gcc"    # Cross-compiler for RISC-V
        (none)                                # Native execution and build
    """

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.sim_cmd: str | None = getattr(args, "sim_cmd", None)
        self.compiler: str = getattr(args, "compiler", None) or "gcc"

    @staticmethod
    def insert_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--sim-cmd",
            type=_sim_cmd_type,
            default=None,
            help="Simulator command template with {binary} placeholder (e.g., 'sde -mix -- {binary}'). "
            "Default: None (run natively)",
        )
        parser.add_argument(
            "--compiler",
            type=_compiler_type,
            default="gcc",
            help="C compiler to use for building tests (e.g., 'riscv64-linux-gnu-gcc'). Default: gcc",
        )

    def run(self, binary_path: str, *args: str, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        """Execute a binary, optionally via simulator, using subprocess.run.

        Internally builds the command string (with optional simulator prefix) and
        executes it via subprocess.run. Forwards all kwargs to subprocess.run.

        Args:
            binary_path: Path to the benchmark binary
            *args: Additional command-line arguments to pass to the binary
            **kwargs: Additional arguments forwarded to subprocess.run
                     (e.g., capture_output=True, timeout=60, check=True, cwd='...', env=...)

        Returns:
            subprocess.CompletedProcess instance with returncode, stdout, stderr, etc.

        Examples:
            >>> exec_iface = ExecutionInterface(argparse.Namespace(sim_cmd=None))
            >>> result = exec_iface.run('./bench', capture_output=True, text=True)
            >>> print(result.returncode)
            0

            >>> exec_sde = ExecutionInterface(argparse.Namespace(sim_cmd="sde -mix -- {binary}"))
            >>> result = exec_sde.run('./bench', timeout=120)

            >>> result = exec_iface.run('./bench', '--freq', '2400000000', capture_output=True, text=True)
        """
        # Build complete command with optional simulator prefix and arguments
        cmd: str | list[str] = self._get_command(binary_path, args)
        debug(f"Executing command: {cmd}")

        # Handle shell parameter: if not explicitly set, use False and split command
        shell = kwargs.pop("shell", False)
        if not shell and isinstance(cmd, str):
            cmd = shlex.split(cmd)

        check = kwargs.pop("check", True)

        return subprocess.run(cmd, shell=shell, check=check, **kwargs)

    def popen(self, binary_path: str, *args: str, **kwargs: Any) -> subprocess.Popen[Any]:
        """Launch a binary, optionally via simulator, using subprocess.Popen.

        The command construction and shell handling match :meth:`run`.
        All keyword arguments are forwarded to :class:`subprocess.Popen`.
        """
        cmd: str | list[str] = self._get_command(binary_path, args)
        debug(f"Launching command: {cmd}")

        shell = kwargs.pop("shell", False)
        if not shell and isinstance(cmd, str):
            cmd = shlex.split(cmd)

        return subprocess.Popen(cmd, shell=shell, **kwargs)

    def _get_command(self, binary_path: str, args: tuple[str, ...] = ()) -> str:
        """Internal: build the complete command string.

        Returns a string suitable for passing to subprocess.run
        (will be split unless shell=True).

        Args:
            binary_path: Path to the binary
            args: Additional command-line arguments
        """
        if self.sim_cmd is None:
            # Native execution: binary + args
            if args:
                return f"{binary_path} {' '.join(args)}"
            return binary_path
        # Simulated execution: simulator + binary + args
        cmd_with_binary = self.sim_cmd.format(binary=binary_path)
        if args:
            return f"{cmd_with_binary} {' '.join(args)}"
        return cmd_with_binary

    def is_native(self) -> bool:
        """Check if using native execution (no simulator)."""
        return self.sim_cmd is None

    def compile(self, source: str, output: str, *extra_flags: str, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        """Compile a source file using the configured compiler.

        Args:
            source: Path to source file
            output: Path to output binary
            *extra_flags: Additional compiler flags (e.g., '-O2', '-march=rv64gcv')
            **kwargs: Additional arguments forwarded to subprocess.run

        Returns:
            subprocess.CompletedProcess instance

        Examples:
            >>> exec_iface = ExecutionInterface(argparse.Namespace(sim_cmd=None, compiler=None))
            >>> result = exec_iface.compile('test.c', 'test', '-O2', '-Wall')

            >>> riscv_exec = ExecutionInterface(argparse.Namespace(sim_cmd=None, compiler='riscv64-linux-gnu-gcc'))
            >>> result = riscv_exec.compile('probe.c', 'probe', '-march=rv64gcv', '-mabi=lp64d')
        """
        cmd = [self.compiler, "-o", output, *extra_flags, source]
        check_result = kwargs.pop("check", True)
        debug(f"Compiling with command: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, **kwargs, check=check_result)
            if result.stderr:
                warn(f"Compilation produced warnings or errors:\n{result.stderr}")
            debug(f"Compilation completed with return code: {result.returncode}")
            return result
        except Exception as e:
            raise RuntimeError(f"Compilation failed: {e}") from e
