from __future__ import annotations

import ctypes
from dataclasses import FrozenInstanceError
from unittest.mock import Mock, patch

import pytest

from carm_roofline.core import Bytes, UserError
from carm_roofline.gpu import GPUMemoryLevel, GPUMemoryTopology, GPUVendor, discover_gpu_memory_topology
from carm_roofline.gpu.compute_capability import ComputeCapability
from carm_roofline.gpu.memory import _discover_amd_from_kfd

pytestmark = pytest.mark.unit

# Known-good NVIDIA attribute values: SM=24, L2=32 MiB,
# shared opt-in=101376 B (99 KiB), VRAM=8177975296 B.
_ATTRS = {16: 24, 38: 33554432, 97: 101376}


class FakeCUDALibrary:
    """Scriptable stand-in for the CUDA driver API library."""

    def __init__(self, attrs: dict[int, int] | None = None, init_rc: int = 0):
        self.attrs = attrs or {}
        self.init_rc = init_rc
        self.attr_calls: list[int] = []

    def cuInit(self, _flags: int) -> int:
        return self.init_rc

    def cuDeviceGet(self, handle, device: int) -> int:
        handle.contents.value = device
        return 0

    def cuDeviceGetAttribute(self, value, attr: int, _handle) -> int:
        self.attr_calls.append(attr)
        value.contents.value = self.attrs.get(attr, 0)
        return 0


class _FakeByRef:
    """Stand-in for a byref() pointer: wraps the object as ``.contents``."""

    def __init__(self, obj):
        self.contents = obj


class FakeCtypes:
    """Minimal ctypes stand-in exposing only what gpu.memory uses."""

    def __init__(self, library: FakeCUDALibrary):
        self._library = library

    def CDLL(self, _name: str) -> FakeCUDALibrary:
        return self._library

    @staticmethod
    def c_int(value: int = 0) -> ctypes.c_int:
        return ctypes.c_int(value)

    @staticmethod
    def byref(obj) -> _FakeByRef:
        return _FakeByRef(obj)


_NVIDIA_SMI_OK = Mock(stdout="7799.125 MiB\n", stderr="", returncode=0)


def _write_kfd_tree(root, nodes, caches=None, mem_banks=None):
    """Build a fake KFD topology under ``root``.

    nodes: dict[int, dict[str, int]] node-id -> properties
    caches: dict[int, list[tuple[int, int]]] node-id -> [(level, size_kb), ...]
    mem_banks: dict[int, list[tuple[int, int]]] node-id -> [(heap_type, size_bytes), ...]
    """
    caches = caches or {}
    mem_banks = mem_banks or {}
    for node_id, props in nodes.items():
        node_dir = root / str(node_id)
        node_dir.mkdir(parents=True)
        node_dir.joinpath("properties").write_text("".join(f"{key} {value}\n" for key, value in props.items()))
        for i, (level, size) in enumerate(caches.get(node_id, [])):
            cache_dir = node_dir / "caches" / str(i)
            cache_dir.mkdir(parents=True)
            cache_dir.joinpath("properties").write_text(f"level {level}\nsize {size}\n")
        for i, (heap_type, size) in enumerate(mem_banks.get(node_id, [])):
            bank_dir = node_dir / "mem_banks" / str(i)
            bank_dir.mkdir(parents=True)
            bank_dir.joinpath("properties").write_text(f"heap_type {heap_type}\nsize_in_bytes {size}\n")
    return root


class TestGPUMemoryLevel:
    """Construction of individual memory levels."""

    def test_fields_and_defaults(self):
        level = GPUMemoryLevel(name="L2", size=Bytes(33554432), sm_count=24)
        assert level.name == "L2"
        assert level.size == Bytes(33554432)
        assert level.size.value == 33554432
        assert level.sm_count == 24
        assert level.bandwidth is None

    def test_frozen(self):
        level = GPUMemoryLevel(name="L2", size=Bytes(1), sm_count=1)
        with pytest.raises(FrozenInstanceError):
            level.size = Bytes(2)  # type: ignore[misc]


class TestGPUMemoryTopology:
    """Property helpers and sm_count delegation."""

    @pytest.fixture
    def topology(self) -> GPUMemoryTopology:
        return GPUMemoryTopology(
            vendor=GPUVendor.NVIDIA,
            levels=(
                GPUMemoryLevel(name="Shared/L1", size=Bytes(101376), sm_count=24),
                GPUMemoryLevel(name="L2", size=Bytes(33554432), sm_count=24),
                GPUMemoryLevel(name="Global", size=Bytes(8177975296), sm_count=24),
            ),
        )

    def test_property_helpers(self, topology: GPUMemoryTopology):
        assert topology.shared_l1 is topology.levels[0]
        assert topology.l2 is topology.levels[1]
        assert topology.global_ is topology.levels[2]

    def test_sm_count_delegates_to_first_level(self, topology: GPUMemoryTopology):
        assert topology.sm_count == 24

    def test_empty_topology(self):
        topo = GPUMemoryTopology(vendor=GPUVendor.AMD, levels=())
        assert topo.sm_count == 0
        assert topo.shared_l1 is None
        assert topo.l2 is None
        assert topo.global_ is None


class TestDiscoverNVIDIA:
    """NVIDIA discovery via the mocked CUDA driver API."""

    @patch("carm_roofline.gpu.memory.shutil.which", return_value="/usr/bin/nvidia-smi")
    @patch("carm_roofline.gpu.memory.subprocess.run", return_value=_NVIDIA_SMI_OK)
    def test_nvidia_levels(self, mock_run, _mock_which):
        library = FakeCUDALibrary(attrs=_ATTRS)
        with patch("carm_roofline.gpu.memory.ctypes", FakeCtypes(library)):
            topo = discover_gpu_memory_topology(GPUVendor.NVIDIA)

        assert topo.vendor == GPUVendor.NVIDIA
        assert topo.sm_count == 24
        shared, l2, global_ = topo.levels
        assert (shared.name, shared.size) == ("Shared/L1", Bytes(101376))
        assert (l2.name, l2.size) == ("L2", Bytes(33554432))
        assert (global_.name, global_.size) == ("Global", Bytes(8177975296))
        assert all(level.sm_count == 24 for level in topo.levels)
        assert all(level.bandwidth is None for level in topo.levels)
        mock_run.assert_called_once_with(
            ["nvidia-smi", "--query-gpu=memory.total", "-i", "0", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )

    @patch("carm_roofline.gpu.memory.shutil.which", return_value="/usr/bin/nvidia-smi")
    @patch("carm_roofline.gpu.memory.subprocess.run", return_value=_NVIDIA_SMI_OK)
    def test_nvidia_optin_shared_mem_fallback(self, mock_run, _mock_which):
        """Pre-CC 7.0: attr 97 reports 0, falls back to attr 8."""
        library = FakeCUDALibrary(attrs={16: 24, 38: 33554432, 97: 0, 8: 49152})
        with patch("carm_roofline.gpu.memory.ctypes", FakeCtypes(library)):
            topo = discover_gpu_memory_topology(GPUVendor.NVIDIA)

        assert topo.shared_l1.size == Bytes(49152)
        # attr 97 queried first, then attr 8
        assert library.attr_calls.index(8) > library.attr_calls.index(97)

    @patch("carm_roofline.gpu.memory.shutil.which", return_value="/usr/bin/nvidia-smi")
    @patch("carm_roofline.gpu.memory.subprocess.run", return_value=_NVIDIA_SMI_OK)
    def test_cuinit_failure_raises(self, mock_run, _mock_which):
        library = FakeCUDALibrary(attrs=_ATTRS, init_rc=1)
        with patch("carm_roofline.gpu.memory.ctypes", FakeCtypes(library)), pytest.raises(UserError, match=r"cuInit"):
            discover_gpu_memory_topology(GPUVendor.NVIDIA)

    def test_libcuda_missing_raises(self):
        with (
            patch("carm_roofline.gpu.memory.ctypes.CDLL", side_effect=OSError("no libcuda")),
            pytest.raises(UserError, match=r"libcuda\.so not found"),
        ):
            discover_gpu_memory_topology(GPUVendor.NVIDIA)


class TestDiscoverAMD:
    """AMD discovery via mocked amd-smi/rocminfo output."""

    @pytest.fixture(autouse=True)
    def _kfd_unavailable(self, tmp_path):
        """KFD sysfs absent: these tests exercise the amd-smi/rocminfo fallback chain."""
        with patch("carm_roofline.gpu.memory._KFD_TOPOLOGY_ROOT", tmp_path / "no-kfd"):
            yield

    _AMD_SMI_JSON = (
        '{"gpu": 0, "asic": {"market_name": "AMD Instinct MI300X", "target_graphics_version": "N/A"},'
        ' "vram": {"type": "HBM3", "vendor": "N/A", "size": {"value": 196592, "unit": "MB"}}}'
    )
    _ROCMINFO = """*******
Agent 1
*******
  Name:                    gfx942
  Marketing Name:          AMD Instinct MI300X
  Device Type:             GPU
  Compute Unit:            220
  Pool Info:
    Pool 1
      Size:                    201310208(0xc000000) KB

*******
Agent 2
*******
  Name:                    CPU
  Device Type:             CPU
  Compute Unit:            16
  Pool Info:
    Pool 1
      Size:                    15591952(0xedea10) KB
*******
"""
    # 196592 MiB == 206141652992 bytes
    _VRAM_BYTES = 206141652992

    _AMD_SMI_NO_VRAM_JSON = (
        '{"gpu": 0, "asic": {"market_name": "AMD Instinct MI300X", "target_graphics_version": "N/A"}}'
    )
    _AMD_SMI_VRAM_JSON = '{"gpu": 0, "vram": {"size": {"value": 196592, "unit": "MB"}}}'

    @patch("carm_roofline.gpu.memory.shutil.which", return_value="/usr/bin/amd-smi")
    @patch("carm_roofline.gpu.memory.subprocess.run")
    def test_amd_vram_from_v_flag(self, mock_run, _mock_which):
        """Older amd-smi: -a lacks VRAM, the -v section supplies it."""

        def fake_run(args, **kwargs):
            if args == ["amd-smi", "static", "-g0", "-a", "--json"]:
                return Mock(stdout=self._AMD_SMI_NO_VRAM_JSON, stderr="", returncode=0)
            if args == ["amd-smi", "static", "-g0", "-v", "--json"]:
                return Mock(stdout=self._AMD_SMI_VRAM_JSON, stderr="", returncode=0)
            if args[0] == "rocminfo":
                return Mock(stdout=self._ROCMINFO, stderr="", returncode=0)
            raise AssertionError(f"unexpected command: {args}")

        mock_run.side_effect = fake_run
        topo = discover_gpu_memory_topology(GPUVendor.AMD)

        assert topo.global_.size == Bytes(self._VRAM_BYTES)

    @patch("carm_roofline.gpu.memory.shutil.which", return_value="/usr/bin/amd-smi")
    @patch("carm_roofline.gpu.memory.subprocess.run")
    def test_amd_levels(self, mock_run, _mock_which):
        def fake_run(args, **kwargs):
            if args[0] == "amd-smi":
                return Mock(stdout=self._AMD_SMI_JSON, stderr="", returncode=0)
            if args[0] == "rocminfo":
                return Mock(stdout=self._ROCMINFO, stderr="", returncode=0)
            raise AssertionError(f"unexpected command: {args}")

        mock_run.side_effect = fake_run
        topo = discover_gpu_memory_topology(GPUVendor.AMD)

        assert topo.vendor == GPUVendor.AMD
        assert topo.sm_count == 220  # Compute Unit count of the GPU agent, not the CPU
        shared, l2, global_ = topo.levels
        assert (shared.name, shared.size) == ("Shared/L1", Bytes(64 * 1024))
        assert (l2.name, l2.size) == ("L2", Bytes(16 * 1024 * 1024))  # gfx942
        assert (global_.name, global_.size) == ("Global", Bytes(self._VRAM_BYTES))

    @patch("carm_roofline.gpu.memory.shutil.which", return_value="/usr/bin/amd-smi")
    @patch("carm_roofline.gpu.memory.subprocess.run")
    def test_amd_gfx_arch_from_compute_capability(self, mock_run, _mock_which):
        """Pre-detected CC supplies gfx arch; amd-smi/rocminfo fill sizes."""
        cc = ComputeCapability.from_string("gfx90a", GPUVendor.AMD)

        def fake_run(args, **kwargs):
            if args[0] == "amd-smi":
                return Mock(stdout=self._AMD_SMI_JSON, stderr="", returncode=0)
            if args[0] == "rocminfo":
                return Mock(stdout=self._ROCMINFO, stderr="", returncode=0)
            raise AssertionError(f"unexpected command: {args}")

        mock_run.side_effect = fake_run
        topo = discover_gpu_memory_topology(GPUVendor.AMD, compute_capability=cc)

        assert topo.l2.size == Bytes(8 * 1024 * 1024)  # gfx90a

    @patch("carm_roofline.gpu.memory.shutil.which", return_value=None)
    @patch("carm_roofline.gpu.memory.subprocess.run")
    def test_amd_rocminfo_only(self, mock_run, _mock_which):
        """amd-smi absent: derive everything from rocminfo + lookup tables."""
        mock_run.return_value = Mock(stdout=self._ROCMINFO, stderr="", returncode=0)
        topo = discover_gpu_memory_topology(GPUVendor.AMD)

        assert topo.sm_count == 220
        assert topo.l2.size == Bytes(16 * 1024 * 1024)
        assert topo.global_.size == Bytes(self._VRAM_BYTES)  # Pool Info size

    @patch("carm_roofline.gpu.memory.shutil.which", return_value="/usr/bin/amd-smi")
    @patch("carm_roofline.gpu.memory.subprocess.run")
    def test_amd_unknown_gfx_arch_defaults_l2(self, mock_run, _mock_which):
        """Unrecognized gfx architecture falls back to 4 MiB L2 with a warning."""
        cc = ComputeCapability.from_string("gfx9999", GPUVendor.AMD)

        def fake_run(args, **kwargs):
            if args[0] == "amd-smi":
                return Mock(stdout=self._AMD_SMI_JSON, stderr="", returncode=0)
            if args[0] == "rocminfo":
                return Mock(stdout=self._ROCMINFO, stderr="", returncode=0)
            raise AssertionError(f"unexpected command: {args}")

        mock_run.side_effect = fake_run
        topo = discover_gpu_memory_topology(GPUVendor.AMD, compute_capability=cc)

        assert topo.l2.size == Bytes(4 * 1024 * 1024)

    @patch("carm_roofline.gpu.memory.shutil.which", return_value=None)
    @patch("carm_roofline.gpu.memory.subprocess.run", side_effect=FileNotFoundError("rocminfo missing"))
    def test_amd_no_tools_raises(self, _mock_run, _mock_which):
        with pytest.raises(UserError, match=r"gfx architecture"):
            discover_gpu_memory_topology(GPUVendor.AMD)


class TestDiscoverAMDKFD:
    """AMD discovery via a fake KFD sysfs topology (no subprocess calls)."""

    # Fixtures for the all-or-nothing fallback test (copied from TestDiscoverAMD).
    _AMD_SMI_JSON = (
        '{"gpu": 0, "asic": {"market_name": "AMD Instinct MI300X", "target_graphics_version": "N/A"},'
        ' "vram": {"type": "HBM3", "vendor": "N/A", "size": {"value": 196592, "unit": "MB"}}}'
    )
    _ROCMINFO = """*******
Agent 1
*******
  Name:                    gfx942
  Marketing Name:          AMD Instinct MI300X
  Device Type:             GPU
  Compute Unit:            220
  Pool Info:
    Pool 1
      Size:                    201310208(0xc000000) KB

*******
Agent 2
*******
  Name:                    CPU
  Device Type:             CPU
  Compute Unit:            16
  Pool Info:
    Pool 1
      Size:                    15591952(0xedea10) KB
*******
"""

    @patch("carm_roofline.gpu.memory.subprocess.run", side_effect=AssertionError("KFD path must not shell out"))
    @patch("carm_roofline.gpu.memory.shutil.which", return_value=None)
    def test_kfd_happy_path(self, _mock_which, _mock_run, tmp_path):
        """KFD sysfs is primary and tool-free: no subprocess is reached."""
        root = _write_kfd_tree(
            tmp_path,
            nodes={
                0: {"simd_count": 0},
                1: {
                    "simd_count": 24,
                    "simd_per_cu": 2,
                    "lds_size_in_kb": 64,
                    "location_id": 13568,
                    "drm_render_minor": 128,
                },
            },
            caches={1: [(1, 16), (2, 2048)]},
            mem_banks={1: [(1, 7983079424)]},
        )
        with patch("carm_roofline.gpu.memory._KFD_TOPOLOGY_ROOT", root):
            topo = discover_gpu_memory_topology(GPUVendor.AMD)

        assert topo.vendor == GPUVendor.AMD
        assert topo.sm_count == 12  # 24 SIMDs / 2 per CU
        shared, l2, global_ = topo.levels
        assert (shared.name, shared.size) == ("Shared/L1", Bytes(64 * 1024))
        assert (l2.name, l2.size) == ("L2", Bytes(2 * 1024 * 1024))
        assert (global_.name, global_.size) == ("Global", Bytes(7983079424))
        assert all(level.bandwidth is None for level in topo.levels)

    def test_kfd_device_index_uses_pci_order(self, tmp_path):
        """location_id (BDF-packed) ordering maps device index to PCI-bus order."""
        root = _write_kfd_tree(
            tmp_path,
            nodes={
                1: {"simd_count": 6, "simd_per_cu": 2, "lds_size_in_kb": 64, "location_id": 0x100},  # BDF 01:00.0
                2: {"simd_count": 24, "simd_per_cu": 2, "lds_size_in_kb": 64, "location_id": 13568},  # BDF 35:00.0
            },
            caches={1: [(2, 1024)], 2: [(2, 1024)]},
            mem_banks={1: [(1, 1 << 30)], 2: [(1, 1 << 30)]},
        )
        with patch("carm_roofline.gpu.memory._KFD_TOPOLOGY_ROOT", root):
            topo = discover_gpu_memory_topology(GPUVendor.AMD, device=1)

        assert topo.sm_count == 12  # device 1 is the node at BDF 0x3500

    def test_kfd_sums_l2_and_device_heaps(self, tmp_path):
        """Level-2 cache entries sum; only heap_types 1 and 2 count as VRAM."""
        root = _write_kfd_tree(
            tmp_path,
            nodes={1: {"simd_count": 24, "simd_per_cu": 2, "lds_size_in_kb": 64, "location_id": 13568}},
            caches={1: [(2, 1024), (1, 16), (2, 1024)]},
            mem_banks={1: [(0, 999999999), (1, 1 << 30), (2, 1 << 29), (3, 999)]},
        )
        with patch("carm_roofline.gpu.memory._KFD_TOPOLOGY_ROOT", root):
            topo = discover_gpu_memory_topology(GPUVendor.AMD)

        assert topo.l2.size == Bytes(2 * 1024 * 1024)  # level-1 ignored, level-2 summed
        assert topo.global_.size == Bytes((1 << 30) + (1 << 29))  # 1.5 GiB: heaps 1+2 only

    def test_kfd_cpu_node_skipped(self, tmp_path):
        """A CPU-only topology yields no KFD result."""
        root = _write_kfd_tree(tmp_path, nodes={0: {"simd_count": 0}})
        with patch("carm_roofline.gpu.memory._KFD_TOPOLOGY_ROOT", root):
            assert _discover_amd_from_kfd(0) is None

    @patch("carm_roofline.gpu.memory.shutil.which", return_value="/usr/bin/amd-smi")
    @patch("carm_roofline.gpu.memory.subprocess.run")
    def test_kfd_missing_field_falls_back(self, mock_run, _mock_which, tmp_path):
        """All-or-nothing: a GPU node without mem_banks drops to the CLI chain."""
        root = _write_kfd_tree(
            tmp_path,
            nodes={1: {"simd_count": 24, "simd_per_cu": 2, "lds_size_in_kb": 64, "location_id": 13568}},
            caches={1: [(2, 2048)]},
        )

        def fake_run(args, **kwargs):
            if args[0] == "amd-smi":
                return Mock(stdout=self._AMD_SMI_JSON, stderr="", returncode=0)
            if args[0] == "rocminfo":
                return Mock(stdout=self._ROCMINFO, stderr="", returncode=0)
            raise AssertionError(f"unexpected command: {args}")

        mock_run.side_effect = fake_run
        with patch("carm_roofline.gpu.memory._KFD_TOPOLOGY_ROOT", root):
            topo = discover_gpu_memory_topology(GPUVendor.AMD)

        assert topo.sm_count == 220  # from rocminfo, proving the fallback chain ran
        mock_run.assert_called()

    def test_kfd_absent_root_falls_back(self, tmp_path):
        """No KFD root: discovery reports nothing rather than raising."""
        with patch("carm_roofline.gpu.memory._KFD_TOPOLOGY_ROOT", tmp_path / "no-kfd"):
            assert _discover_amd_from_kfd(0) is None


class TestBytesRoundTrip:
    """Sizes round-trip through the Bytes type."""

    def test_bytes_round_trip(self):
        size = Bytes(8589934592)  # 8 GiB
        assert size == Bytes.from_string(str(size))
        assert Bytes.from_string("8 GiB") == size
        assert Bytes.from_string("8192 MiB") == size
        assert Bytes.from_kibibytes(8 * 1024 * 1024) == size
        assert size.value == 8589934592
