from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

import carm_roofline.architecture.detect as detection

pytestmark = pytest.mark.unit


def test_native_detect_dispatches_riscv64_to_riscv_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    from carm_roofline.architecture import riscv

    expected = detection.DetectedArchitecture(isa=["riscv_scalar"])
    detector = Mock(return_value=expected)
    monkeypatch.setattr(detection.platform, "machine", lambda: "riscv64")
    monkeypatch.setattr(riscv, "detect", detector)

    detected = detection.native_detect(threads=2)

    assert detected is expected
    detector.assert_called_once_with(threads=2)


@pytest.mark.parametrize(
    ("features", "expected_vlen_calls"),
    [(frozenset(), 0), (frozenset({"riscv_vector"}), 1)],
)
def test_riscv_vlen_probe_runs_only_when_rvv_compiles(
    monkeypatch: pytest.MonkeyPatch,
    features: frozenset[str],
    expected_vlen_calls: int,
) -> None:
    import carm_roofline.architecture as architecture

    monkeypatch.setattr(detection, "detect_features", lambda _ctx: {})
    monkeypatch.setattr(detection, "detect_cache", lambda _ctx: {})
    monkeypatch.setattr(detection, "detect_frequency", lambda _ctx, threads=1: {})
    detect_vlen = Mock(return_value={})
    monkeypatch.setattr(detection, "detect_vlen", detect_vlen)
    monkeypatch.setattr(architecture, "get_execution_interface", lambda: type("Exec", (), {"sim_cmd": "test"})())

    detection.run_generic_tests(detection.TestContext(family="riscv", features=features))

    assert detect_vlen.call_count == expected_vlen_calls


def test_riscv_detector_uses_semantic_target_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    import carm_roofline.architecture as architecture
    from carm_roofline.architecture import riscv

    class Target:
        sim_cmd = "qemu-riscv64 {binary}"

        @staticmethod
        def is_native() -> bool:
            return False

    monkeypatch.setattr(architecture, "get_execution_interface", lambda: Target())
    monkeypatch.setattr(riscv, "_linux_hwprobe_has_v", lambda: pytest.fail("host hwprobe used"))
    attempts: list[str] = []

    def run_probe(_target: object, _source: object, define: str) -> bool:
        attempts.append(define)
        return define == "RISCV_RVV_0_7_1"

    monkeypatch.setattr(riscv, "_can_run_rvv", run_probe)
    expected = riscv.DetectedArchitecture()
    monkeypatch.setattr(riscv, "run_generic_tests", lambda _ctx, threads=1: expected)

    detected = riscv.detect()

    assert attempts == ["RISCV_RVV", "RISCV_RVV_0_7_1"]
    assert detected is expected
    assert detected.isa == ["riscv", "riscv_rvv_0_7_1"]


def test_riscv_probe_requires_expected_semantic_result() -> None:
    from types import SimpleNamespace

    from carm_roofline.architecture.riscv import _can_run_rvv

    class Target:
        def __init__(self, stdout: str, returncode: int = 0) -> None:
            self.stdout = stdout
            self.returncode = returncode
            self.ran = False

        @staticmethod
        def compile(*_args: object, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(returncode=0)

        def run(self, *_args: object, **_kwargs: object) -> SimpleNamespace:
            self.ran = True
            return SimpleNamespace(returncode=self.returncode, stdout=self.stdout)

    target = Target('{"rvv_version":"RISCV_RVV_0_7_1"}')
    assert _can_run_rvv(target, Path("probe.c"), "RISCV_RVV_0_7_1")
    assert target.ran

    wrong_version = Target('{"rvv_version":"RISCV_RVV"}')
    assert not _can_run_rvv(wrong_version, Path("probe.c"), "RISCV_RVV_0_7_1")


def test_riscv_hwprobe_reports_only_kernel_v_bit(monkeypatch: pytest.MonkeyPatch) -> None:
    from carm_roofline.architecture import riscv

    monkeypatch.setattr(riscv.platform, "system", lambda: "Linux")
    monkeypatch.setattr(riscv.platform, "machine", lambda: "riscv64")

    class Syscall:
        restype: object = None

        def __call__(self, _number: int, pair: object, *_args: object) -> int:
            pair._obj.value = 1 << 2
            return 0

    libc = type("LibC", (), {"syscall": Syscall()})()
    monkeypatch.setattr(riscv.ctypes, "CDLL", lambda *_args, **_kwargs: libc)
    assert riscv._linux_hwprobe_has_v() is True
