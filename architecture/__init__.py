from __future__ import annotations

import contextvars

from exec_interface import ExecutionInterface

from .architecture import Architecture
from .identity import MachineSignature, detect_machine_signature, generate_run_name, signature_from_architecture
from .memory import SimpleMemoryTopology

# Context variable to hold the execution interface, for compatibility with simulation and native execution
_exec_context: contextvars.ContextVar[ExecutionInterface | None] = contextvars.ContextVar("execution", default=None)


def set_execution_interface(exec_interface: ExecutionInterface) -> None:
    _exec_context.set(exec_interface)


def get_execution_interface() -> ExecutionInterface:
    if exec_iface := _exec_context.get():
        return exec_iface
    raise RuntimeError("Execution interface needs to be set before Architecture is instanced for detection tests")


__all__ = [
    "Architecture",
    "MachineSignature",
    "SimpleMemoryTopology",
    "detect_machine_signature",
    "generate_run_name",
    "get_execution_interface",
    "set_execution_interface",
    "signature_from_architecture",
]
