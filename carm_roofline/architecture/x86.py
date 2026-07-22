from __future__ import annotations

from carm_roofline.architecture.detect import ROOT, DetectedArchitecture, TestContext, run_generic_tests, run_test
from carm_roofline.core import Frequency


def detect(threads: int = 1) -> DetectedArchitecture:
    """Detect x86 ISA features and cache sizes."""
    ctx = TestContext(family="x86")
    detected = run_generic_tests(ctx, threads=threads)

    # Detect frequency per ISA (AVX512 may have different frequency than base)
    if detected.isa:
        isa_frequencies: dict[str, Frequency] = {}
        for isa_name in detected.isa:
            # Check if ISA-specific frequency header exists
            freq_header = ROOT / "tests" / "x86" / isa_name / "frequency.h"
            if freq_header.exists():
                # Run generic frequency.c with ISA-specific header
                freq_test = ROOT / "tests" / "frequency.c"
                isa_ctx = TestContext(family="x86", isa=isa_name)
                freq_result = run_test(freq_test, isa_ctx)
                # Convert raw Hz value to Frequency object
                if "frequency_hz" in freq_result:
                    isa_frequencies[isa_name] = Frequency(freq_result["frequency_hz"])

        # Store per-ISA frequencies if any were detected
        if isa_frequencies:
            detected.isa_frequencies = isa_frequencies

    return detected
