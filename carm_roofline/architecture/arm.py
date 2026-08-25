from __future__ import annotations

from carm_roofline.architecture.detect import DetectedArchitecture, TestContext, detect_vlen, run_generic_tests
from carm_roofline.isa import ArmNeon, ArmScalar, ArmSVE


def detect(threads: int = 1) -> DetectedArchitecture:
    """Detect ARM ISA features and vector length (if applicable)."""
    ctx = TestContext(family="arm")
    detected = run_generic_tests(ctx, threads=threads, include_vlen=False)

    # Build ISA list with scalar as base
    isa_list = [ArmScalar.name]
    if detected.isa:
        if "arm_neon" in detected.isa:
            isa_list.append(ArmNeon.name)
        if "arm_sve" in detected.isa:
            isa_list.append(ArmSVE.name)

    if ArmSVE.name in isa_list:
        detected.vector_length = detect_vlen(ctx).get("vector_length")

    detected.isa = isa_list

    return detected
