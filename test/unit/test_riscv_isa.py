from __future__ import annotations

from types import SimpleNamespace

import pytest

from carm_roofline.core import ArithmeticOperation, DataType
from carm_roofline.isa.riscv import RISCV_RVV, RISCV_RVV_071

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("isa_class", [RISCV_RVV_071, RISCV_RVV])
def test_rvv_uses_detected_vector_length_in_bytes(isa_class: type[RISCV_RVV_071]) -> None:
    isa = isa_class.from_architecture(SimpleNamespace(vector_length=16, vector_lmul=None))

    assert isa.vlen_bits == 128
    assert isa.bytes_per_inst(DataType.f32) == 16
    assert isa.ops_per_inst(DataType.f32, ArithmeticOperation.add) == 4
