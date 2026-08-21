# ISA Module Documentation

This module defines the ISA identity hierarchy for CARM. Each ISA subclass declares the instruction templates, register sets, and control instructions that the benchmark generator uses to emit inline-assembly microbenchmarks.
## Module Overview

- **[base.py](base.py)** - `BaseISA` registry, `from_name()`, `all()`
- **[x86.py](x86.py)** - x86 ISAs: `X86Scalar`, `X86SSE`, `X86AVX`, `X86AVX2`, `X86AVX512`
- **[arm.py](arm.py)** - ARM ISAs: `ArmScalar`, `ArmNeon`, `ArmSVE`
- **[riscv.py](riscv.py)** - RISC-V ISAs: `RISCVScalar`, `RISCV_RVV`, `RISCV_RVV_071`
- **[__init__.py](__init__.py)** - Public re-exports and `INCOMPATIBLE_ISAS`

## X86Scalar Bench Register Exclusions

The `X86Scalar` bench register sets exclude, for all operations at once: the inner iterator `rdi` (and `dil`), the outer iterator `r12` (`r12b`), the memory pointer regs `r8`/`r9`/`r10` (and their sub-width forms), the stack pointer `rsp`/`spl`, and BOTH div dividend register halves (`rax`/`eax`/`ax`/`al` and `rdx`/`edx`/`dx`, plus `ah`): div writes the quotient to the low half and the remainder to the high half, so a dividend register used as a divisor could be zero (previous quotient) -> `#DE` (`SIGFPE`). With the dividend regs excluded, the divisors stay at their preloaded nonzero constant, and the dividend is always `high*2^w + low` with `high < divisor` (remainder invariant) and `low < 2^w`, so the unsigned quotient always fits -> provably trap-free.

The `r8b`-`r15b` / `r8w`-`r15w` / `r8d`-`r15d` sub-width names are NOT accepted as inline-asm clobber names by GCC, so they cannot be bench registers either.

For `i8` the dividend is AX (`ah`:`al`); the high-byte regs `bh`/`ch`/`dh` are excluded too: memory ops pair a bench register with an `r8`/`r9`/`r10` pointer, and a REX prefix (required for the pointer) cannot encode `ah`/`bh`/`ch`/`dh` in the same instruction ("can't encode register ... requiring REX prefix").

**When modifying this module:** Keep the rationale for register exclusions and instruction selection here, and refer to it from the code with a short comment.
