from __future__ import annotations

from architecture.detect import DetectedArchitecture, TestContext, run_generic_tests
from isa import ArmNeon, ArmScalar, ArmSVE


def detect(threads: int = 1) -> DetectedArchitecture:
    """Detect ARM ISA features and vector length (if applicable)."""
    ctx = TestContext(family="arm")
    detected = run_generic_tests(ctx, threads=threads)

    # Build ISA list with scalar as base
    isa_list = [ArmScalar.name]
    if detected.isa:
        if "arm_neon" in detected.isa:
            isa_list.append(ArmNeon.name)
        if "arm_sve" in detected.isa:
            isa_list.append(ArmSVE.name)

    detected.isa = isa_list

    return detected
