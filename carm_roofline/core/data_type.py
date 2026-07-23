from __future__ import annotations

from enum import Enum


class DataType(Enum):
    i8 = "int8_t"
    i16 = "int16_t"
    i32 = "int32_t"
    i64 = "int64_t"
    f32 = "float"
    f64 = "double"
    f16 = "f16"
    bf16 = "bf16"
    tf32 = "tf32"
    f8 = "f8"
    i4 = "i4"
    i1 = "i1"

    @staticmethod
    def check_validity(obj: object) -> None:
        """Checks if an object is a valid `DataType`, raises an error if not"""
        if not isinstance(obj, DataType):
            raise TypeError(
                f"Object of type '{type(obj).__name__}' is not a valid data type. "
                f"Must be a variant of {DataType.__name__}: {DataType._member_names_}"
            )

    def to_c_type(self) -> str:
        return self.value

    def bits(self) -> int:
        "Returns the size of one element of this data type in bits"
        bit_map = {
            DataType.i8: 8,
            DataType.i16: 16,
            DataType.i32: 32,
            DataType.i64: 64,
            DataType.f32: 32,
            DataType.f64: 64,
            DataType.f16: 16,
            DataType.bf16: 16,
            DataType.tf32: 32,
            DataType.f8: 8,
            DataType.i4: 4,
            DataType.i1: 1,
        }
        if self not in bit_map:
            raise ValueError(f"The bit width for data type {self} is unknown")
        return bit_map[self]

    def bytes(self) -> int:
        "Returns the size of one element of this data type in bytes (ceil(bits/8))"
        return (self.bits() + 7) // 8
