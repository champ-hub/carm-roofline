from __future__ import annotations

from enum import Enum


class DataType(Enum):
    i8 = "int8_t"
    i16 = "int16_t"
    i32 = "int32_t"
    i64 = "int64_t"
    f32 = "float"
    f64 = "double"
    bf16 = "uint16_t"

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

    def bytes(self) -> int:
        "Returns the size of one element of this data type in bytes"
        size_map = {
            DataType.i8: 1,
            DataType.i16: 2,
            DataType.i32: 4,
            DataType.i64: 8,
            DataType.f32: 4,
            DataType.f64: 8,
            DataType.bf16: 2,
        }
        if self not in size_map:
            raise ValueError(f"The size for data type {self} is unknown")
        return size_map[self]

    def bits(self) -> int:
        "Returns the size of one element of this data type in bits"
        return self.bytes() * 8
