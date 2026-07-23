from __future__ import annotations

import pytest

from carm_roofline.core import DataType
from carm_roofline.gpu import ComputeCapability, GPULaunchConfig, GPUVendor, TensorPrecision

pytestmark = pytest.mark.unit


class TestGPUVendor:
    def test_members(self):
        assert GPUVendor.NVIDIA.value == "nvidia"
        assert GPUVendor.AMD.value == "amd"

    def test_enum_distinct(self):
        assert GPUVendor.NVIDIA != GPUVendor.AMD


class TestGPULaunchConfig:
    def test_defaults(self):
        cfg = GPULaunchConfig(blocks=1024)
        assert cfg.blocks == 1024
        assert cfg.threads_per_block == 1024
        assert cfg.sm_targets is None

    def test_explicit(self):
        cfg = GPULaunchConfig(blocks=512, threads_per_block=256, sm_targets=80)
        assert cfg.num_threads == 512 * 256

    def test_frozen(self):
        cfg = GPULaunchConfig(blocks=1024)
        with pytest.raises(AttributeError):
            cfg.blocks = 2048


class TestComputeCapability:
    def test_as_int_nvidia(self):
        cc = ComputeCapability(major=8, minor=9, vendor=GPUVendor.NVIDIA)
        assert cc.as_int == 89

    def test_from_string_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            ComputeCapability.from_string("8.9", GPUVendor.NVIDIA)


class TestTensorPrecision:
    FP16_32 = TensorPrecision(
        name="fp16_32",
        precision_triple=(DataType.f16, DataType.f16, DataType.f32),
        tile_mnk=(16, 8, 16),
        flops_per_mma=2 * 16 * 8 * 16,
    )

    def test_flops_formula(self):
        assert self.FP16_32.flops_per_mma == 4096  # 2 * 16 * 8 * 16

    def test_frozen(self):
        with pytest.raises(AttributeError):
            self.FP16_32.name = "changed"


class TestTensorCoreOperation:
    def test_mma_repr(self):
        from carm_roofline.core import TensorCoreOperation

        assert repr(TensorCoreOperation.mma) == "mma"
        assert TensorCoreOperation.mma.name == "mma"


class TestDataTypeGPU:
    @pytest.mark.parametrize(
        "dt,expected_bytes",
        [
            (DataType.f16, 2),
            (DataType.bf16, 2),
            (DataType.tf32, 4),
            (DataType.f8, 1),
            (DataType.i4, 1),
            (DataType.i1, 1),
        ],
    )
    def test_bytes(self, dt, expected_bytes):
        assert dt.bytes() == expected_bytes

    @pytest.mark.parametrize(
        "dt,expected_bits",
        [
            (DataType.f16, 16),
            (DataType.bf16, 16),
            (DataType.tf32, 32),
            (DataType.f8, 8),
            (DataType.i4, 4),
            (DataType.i1, 1),
        ],
    )
    def test_bits(self, dt, expected_bits):
        assert dt.bits() == expected_bits

    def test_bits_from_bytes_consistency(self):
        """For byte-aligned types, bytes*8 == bits. For sub-byte types, ceil(bits/8) == bytes."""
        assert DataType.f16.bits() == DataType.f16.bytes() * 8
        assert DataType.i4.bytes() == 1  # storage, not arithmetic width
        assert DataType.i4.bits() == 4  # arithmetic width

    def test_to_c_type(self):
        assert DataType.f16.to_c_type() == "f16"
        assert DataType.bf16.to_c_type() == "bf16"
