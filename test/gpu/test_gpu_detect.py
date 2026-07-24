from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from carm_roofline.core import UserError
from carm_roofline.gpu import ComputeCapability, GPUVendor
from carm_roofline.gpu.detect import (
    _enumerate_gpus,
    detect_compute_capability,
    detect_gpu,
    detect_gpu_vendor,
    lock_gpu_frequencies,
    read_gpu_frequencies,
    reset_gpu_clocks,
)

pytestmark = pytest.mark.unit


class TestEnumerateGPUs:
    """Tests for _enumerate_gpus via temp-directory file structure."""

    def _make_gpu_dir(self, pci_dir: Path, pci_addr: str, class_code: str, vendor_id: str) -> None:
        """Create a fake PCI device directory with class and vendor files."""
        dev_dir = pci_dir / pci_addr
        dev_dir.mkdir()
        (dev_dir / "class").write_text(class_code + "\n")
        (dev_dir / "vendor").write_text(vendor_id + "\n")

    def test_single_nvidia(self):
        """Single NVIDIA GPU detected."""
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir_str:
            pci_dir = Path(tmpdir_str)
            self._make_gpu_dir(pci_dir, "0000:01:00.0", "0x030000", "0x10de")
            with patch("carm_roofline.gpu.detect._PCI_DEVICES", pci_dir):
                result = _enumerate_gpus()
                assert result == [(GPUVendor.NVIDIA, 0)]

    def test_single_amd(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir_str:
            pci_dir = Path(tmpdir_str)
            self._make_gpu_dir(pci_dir, "0000:01:00.0", "0x030000", "0x1002")
            with patch("carm_roofline.gpu.detect._PCI_DEVICES", pci_dir):
                result = _enumerate_gpus()
                assert result == [(GPUVendor.AMD, 0)]

    def test_mixed_vendors(self):
        """NVIDIA then AMD in PCI bus order."""
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir_str:
            pci_dir = Path(tmpdir_str)
            self._make_gpu_dir(pci_dir, "0000:01:00.0", "0x030000", "0x10de")
            self._make_gpu_dir(pci_dir, "0000:02:00.0", "0x030000", "0x1002")
            with patch("carm_roofline.gpu.detect._PCI_DEVICES", pci_dir):
                result = _enumerate_gpus()
                assert result == [(GPUVendor.NVIDIA, 0), (GPUVendor.AMD, 0)]

    def test_mixed_vendors_reverse_pci_order(self):
        """AMD before NVIDIA in PCI order."""
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir_str:
            pci_dir = Path(tmpdir_str)
            self._make_gpu_dir(pci_dir, "0000:01:00.0", "0x030000", "0x1002")
            self._make_gpu_dir(pci_dir, "0000:02:00.0", "0x030000", "0x10de")
            with patch("carm_roofline.gpu.detect._PCI_DEVICES", pci_dir):
                result = _enumerate_gpus()
                assert result == [(GPUVendor.AMD, 0), (GPUVendor.NVIDIA, 0)]

    def test_multiple_nvidia(self):
        """Two NVIDIA GPUs get vendor-relative indices 0, 1."""
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir_str:
            pci_dir = Path(tmpdir_str)
            self._make_gpu_dir(pci_dir, "0000:01:00.0", "0x030000", "0x10de")
            self._make_gpu_dir(pci_dir, "0000:02:00.0", "0x030000", "0x10de")
            with patch("carm_roofline.gpu.detect._PCI_DEVICES", pci_dir):
                result = _enumerate_gpus()
                assert result == [(GPUVendor.NVIDIA, 0), (GPUVendor.NVIDIA, 1)]

    def test_two_nvidia_one_amd(self):
        """Complex mixed setup: [N0, N1, A0]."""
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir_str:
            pci_dir = Path(tmpdir_str)
            self._make_gpu_dir(pci_dir, "0000:01:00.0", "0x030000", "0x10de")
            self._make_gpu_dir(pci_dir, "0000:02:00.0", "0x030000", "0x10de")
            self._make_gpu_dir(pci_dir, "0000:03:00.0", "0x030000", "0x1002")
            with patch("carm_roofline.gpu.detect._PCI_DEVICES", pci_dir):
                result = _enumerate_gpus()
                assert result == [
                    (GPUVendor.NVIDIA, 0),
                    (GPUVendor.NVIDIA, 1),
                    (GPUVendor.AMD, 0),
                ]

    def test_non_gpu_device_filtered(self):
        """Non-GPU PCI devices (ethernet, etc.) are filtered out."""
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir_str:
            pci_dir = Path(tmpdir_str)
            self._make_gpu_dir(pci_dir, "0000:01:00.0", "0x020000", "0x8086")
            self._make_gpu_dir(pci_dir, "0000:02:00.0", "0x030000", "0x10de")
            with patch("carm_roofline.gpu.detect._PCI_DEVICES", pci_dir):
                result = _enumerate_gpus()
                assert result == [(GPUVendor.NVIDIA, 0)]

    def test_unsupported_vendor_filtered(self):
        """Intel GPU (vendor 0x8086) is not NVIDIA or AMD."""
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir_str:
            pci_dir = Path(tmpdir_str)
            self._make_gpu_dir(pci_dir, "0000:01:00.0", "0x030000", "0x8086")
            with patch("carm_roofline.gpu.detect._PCI_DEVICES", pci_dir):
                result = _enumerate_gpus()
                assert result == []

    def test_3d_controller_class(self):
        """3D controllers (0x0302) are detected as GPUs."""
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir_str:
            pci_dir = Path(tmpdir_str)
            self._make_gpu_dir(pci_dir, "0000:01:00.0", "0x030200", "0x10de")
            with patch("carm_roofline.gpu.detect._PCI_DEVICES", pci_dir):
                result = _enumerate_gpus()
                assert result == [(GPUVendor.NVIDIA, 0)]

    def test_display_controller_class(self):
        """Display controllers (0x0380) are detected as GPUs."""
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir_str:
            pci_dir = Path(tmpdir_str)
            self._make_gpu_dir(pci_dir, "0000:01:00.0", "0x038000", "0x1002")
            with patch("carm_roofline.gpu.detect._PCI_DEVICES", pci_dir):
                result = _enumerate_gpus()
                assert result == [(GPUVendor.AMD, 0)]

    def test_permission_error(self):
        """PermissionError reading sysfs files is handled gracefully."""
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir_str:
            pci_dir = Path(tmpdir_str)
            self._make_gpu_dir(pci_dir, "0000:01:00.0", "0x030000", "0x10de")
            # Remove read permission on vendor file
            vendor_file = pci_dir / "0000:01:00.0" / "vendor"
            vendor_file.unlink()
            with patch("carm_roofline.gpu.detect._PCI_DEVICES", pci_dir):
                result = _enumerate_gpus()
                assert result == []

    def test_sysfs_unavailable(self):
        """When /sys/bus/pci/devices/ does not exist, return empty list."""
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir_str:
            pci_dir = Path(tmpdir_str) / "nonexistent"
            with patch("carm_roofline.gpu.detect._PCI_DEVICES", pci_dir):
                result = _enumerate_gpus()
                assert result == []


class TestDetectGPU:
    """Tests for detect_gpu with mocked sysfs, subprocess, and shutil."""

    # --- PCI enumeration path (sysfs available) ---

    @patch("carm_roofline.gpu.detect.shutil.which")
    @patch("carm_roofline.gpu.detect.subprocess.run")
    @patch("carm_roofline.gpu.detect._enumerate_gpus")
    def test_detect_gpu_pci_nvidia(self, mock_enumerate, mock_run, mock_which):
        """detect_gpu uses sysfs to route to nvidia-smi."""
        mock_enumerate.return_value = [(GPUVendor.NVIDIA, 0)]
        mock_which.return_value = "/usr/bin/nvidia-smi"
        mock_run.return_value = Mock(
            stdout="8.9, NVIDIA GeForce RTX 4090\n",
            stderr="",
            returncode=0,
        )

        vendor, cc, _model = detect_gpu(device=0)
        assert vendor == GPUVendor.NVIDIA
        assert cc.as_int == 89
        # Verify nvidia-smi called with vendor-relative index 0
        mock_run.assert_called_once_with(
            ["nvidia-smi", "--query-gpu=compute_cap,gpu_name", "-i", "0", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )

    @patch("carm_roofline.gpu.detect._get_amd_gfx_arch")
    @patch("carm_roofline.gpu.detect.shutil.which")
    @patch("carm_roofline.gpu.detect.subprocess.run")
    @patch("carm_roofline.gpu.detect._enumerate_gpus")
    def test_detect_gpu_pci_amd(self, mock_enumerate, mock_run, mock_which, mock_gfx):
        """detect_gpu routes to amd-smi when sysfs says AMD."""
        mock_enumerate.return_value = [(GPUVendor.AMD, 0)]
        mock_which.return_value = "/usr/bin/amd-smi"
        mock_gfx.return_value = "gfx942"
        mock_run.return_value = Mock(
            stdout='{"gpu": 0, "asic": {"market_name": "AMD Radeon RX 7900 XTX","target_graphics_version": "N/A"}}\n',
            stderr="",
            returncode=0,
        )

        vendor, cc, _model = detect_gpu(device=0)
        assert vendor == GPUVendor.AMD
        assert cc.gfx_arch == "gfx942"

    @patch("carm_roofline.gpu.detect._get_amd_gfx_arch")
    @patch("carm_roofline.gpu.detect.shutil.which")
    @patch("carm_roofline.gpu.detect.subprocess.run")
    @patch("carm_roofline.gpu.detect._enumerate_gpus")
    def test_detect_gpu_pci_mixed_vendor(self, mock_enumerate, mock_run, mock_which, mock_gfx):
        """On mixed system, device=1 routes to AMD (second GPU in PCI order)."""
        mock_enumerate.return_value = [
            (GPUVendor.NVIDIA, 0),
            (GPUVendor.AMD, 0),
        ]
        mock_which.return_value = "/usr/bin/amd-smi"
        mock_gfx.return_value = "gfx942"
        mock_run.return_value = Mock(
            stdout='{"gpu": 0, "asic": {"market_name": "AMD Radeon RX 7900 XTX","target_graphics_version": "N/A"}}\n',
            stderr="",
            returncode=0,
        )

        vendor, _cc, _model = detect_gpu(device=1)
        assert vendor == GPUVendor.AMD
        # Verify amd-smi called with AMD-relative index 0
        mock_run.assert_called_once_with(
            ["amd-smi", "static", "-g0", "-a", "--json"],
            capture_output=True,
            text=True,
            check=True,
        )

    @patch("carm_roofline.gpu.detect.shutil.which")
    @patch("carm_roofline.gpu.detect.subprocess.run")
    @patch("carm_roofline.gpu.detect._enumerate_gpus")
    def test_detect_gpu_pci_second_nvidia(self, mock_enumerate, mock_run, mock_which):
        """Device=2 routes to second NVIDIA on [N0, A0, N1] system."""
        mock_enumerate.return_value = [
            (GPUVendor.NVIDIA, 0),
            (GPUVendor.AMD, 0),
            (GPUVendor.NVIDIA, 1),
        ]
        mock_which.return_value = "/usr/bin/nvidia-smi"
        mock_run.return_value = Mock(
            stdout="8.9, NVIDIA GeForce RTX 4090\n",
            stderr="",
            returncode=0,
        )

        vendor, _cc, _model = detect_gpu(device=2)
        assert vendor == GPUVendor.NVIDIA
        # Verify nvidia-smi called with vendor-relative index 1
        mock_run.assert_called_once_with(
            ["nvidia-smi", "--query-gpu=compute_cap,gpu_name", "-i", "1", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )

    @patch("carm_roofline.gpu.detect.shutil.which")
    @patch("carm_roofline.gpu.detect._enumerate_gpus")
    def test_detect_gpu_pci_out_of_range(self, mock_enumerate, mock_which):
        """Out-of-range index with sysfs raises FileNotFoundError."""
        mock_enumerate.return_value = [(GPUVendor.NVIDIA, 0)]
        mock_which.return_value = "/usr/bin/nvidia-smi"

        with pytest.raises(FileNotFoundError, match="GPU device 1 not found"):
            detect_gpu(device=1)

    @patch("carm_roofline.gpu.detect.shutil.which")
    @patch("carm_roofline.gpu.detect._enumerate_gpus")
    def test_detect_gpu_pci_smi_unavailable(self, mock_enumerate, mock_which):
        """PCI says NVIDIA but no nvidia-smi — precise error message."""
        mock_enumerate.return_value = [(GPUVendor.NVIDIA, 0)]
        mock_which.return_value = None

        with pytest.raises(FileNotFoundError, match="SMI tool is unavailable"):
            detect_gpu(device=0)

    # --- Fallback path (no sysfs / empty enumeration) ---

    @patch("carm_roofline.gpu.detect.shutil.which")
    @patch("carm_roofline.gpu.detect.subprocess.run")
    @patch("carm_roofline.gpu.detect._enumerate_gpus")
    def test_detect_gpu_fallback_nvidia(self, mock_enumerate, mock_run, mock_which):
        """Fallback: no sysfs, tries NVIDIA first, succeeds."""
        mock_enumerate.return_value = []
        mock_which.return_value = "/usr/bin/nvidia-smi"
        mock_run.return_value = Mock(
            stdout="8.9, NVIDIA GeForce RTX 4090\n",
            stderr="",
            returncode=0,
        )

        vendor, _cc, _model = detect_gpu(device=0)
        assert vendor == GPUVendor.NVIDIA

    @patch("carm_roofline.gpu.detect._get_amd_gfx_arch")
    @patch("carm_roofline.gpu.detect.shutil.which")
    @patch("carm_roofline.gpu.detect.subprocess.run")
    @patch("carm_roofline.gpu.detect._enumerate_gpus")
    def test_detect_gpu_fallback_amd(self, mock_enumerate, mock_run, mock_which, mock_gfx):
        """Fallback: no sysfs, no nvidia-smi, tries AMD."""
        mock_enumerate.return_value = []

        def which_side_effect(name):
            return "/usr/bin/amd-smi" if name == "amd-smi" else None

        mock_which.side_effect = which_side_effect
        mock_gfx.return_value = "gfx942"
        mock_run.return_value = Mock(
            stdout='{"gpu": 0, "asic": {"market_name": "AMD Radeon RX 7900 XTX"}}\n',
            stderr="",
            returncode=0,
        )

        vendor, _cc, _model = detect_gpu(device=0)
        assert vendor == GPUVendor.AMD

    @patch("carm_roofline.gpu.detect._get_amd_gfx_arch")
    @patch("carm_roofline.gpu.detect.shutil.which")
    @patch("carm_roofline.gpu.detect.subprocess.run")
    @patch("carm_roofline.gpu.detect._enumerate_gpus")
    def test_detect_gpu_fallback_nvidia_fails_amd(self, mock_enumerate, mock_run, mock_which, mock_gfx):
        """Fallback: nvidia-smi exists but fails, falls through to AMD."""
        mock_enumerate.return_value = []
        mock_which.return_value = "/usr/bin/nvidia-smi"
        mock_gfx.return_value = "gfx942"
        from subprocess import CalledProcessError

        amd_mock = Mock(
            stdout='{"gpu": 0, "asic": {"market_name": "AMD Radeon RX 7900 XTX"}}\n',
            stderr="",
            returncode=0,
        )
        mock_run.side_effect = [
            CalledProcessError(2, "nvidia-smi", stderr="No devices found"),
            amd_mock,
        ]

        vendor, _cc, _model = detect_gpu(device=0)
        assert vendor == GPUVendor.AMD

    @patch("carm_roofline.gpu.detect.shutil.which")
    @patch("carm_roofline.gpu.detect._enumerate_gpus")
    def test_detect_gpu_fallback_no_tools(self, mock_enumerate, mock_which):
        """Fallback: no SMI tools at all."""
        mock_enumerate.return_value = []
        mock_which.return_value = None

        with pytest.raises(FileNotFoundError):
            detect_gpu(device=0)

    # --- detect_gpu_vendor ---

    @patch("carm_roofline.gpu.detect._enumerate_gpus")
    def test_detect_gpu_vendor_from_sysfs_nvidia(self, mock_enumerate):
        mock_enumerate.return_value = [(GPUVendor.NVIDIA, 0)]
        assert detect_gpu_vendor() == GPUVendor.NVIDIA

    @patch("carm_roofline.gpu.detect._enumerate_gpus")
    def test_detect_gpu_vendor_from_sysfs_amd(self, mock_enumerate):
        mock_enumerate.return_value = [(GPUVendor.AMD, 0)]
        assert detect_gpu_vendor() == GPUVendor.AMD

    @patch("carm_roofline.gpu.detect.shutil.which")
    @patch("carm_roofline.gpu.detect._enumerate_gpus")
    def test_detect_gpu_vendor_fallback_nvidia(self, mock_enumerate, mock_which):
        """Fallback: no sysfs, nvidia-smi on PATH."""
        mock_enumerate.return_value = []
        mock_which.side_effect = lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None
        assert detect_gpu_vendor() == GPUVendor.NVIDIA

    @patch("carm_roofline.gpu.detect.shutil.which")
    @patch("carm_roofline.gpu.detect._enumerate_gpus")
    def test_detect_gpu_vendor_fallback_amd(self, mock_enumerate, mock_which):
        """Fallback: no sysfs, only amd-smi on PATH."""
        mock_enumerate.return_value = []
        mock_which.side_effect = lambda name: "/usr/bin/amd-smi" if name == "amd-smi" else None
        assert detect_gpu_vendor() == GPUVendor.AMD

    @patch("carm_roofline.gpu.detect.shutil.which")
    @patch("carm_roofline.gpu.detect._enumerate_gpus")
    def test_detect_gpu_vendor_fallback_none(self, mock_enumerate, mock_which):
        """Fallback: no sysfs, no SMI tools."""
        mock_enumerate.return_value = []
        mock_which.return_value = None
        assert detect_gpu_vendor() is None

    # --- detect_compute_capability ---

    @patch("carm_roofline.gpu.detect.detect_gpu")
    def test_detect_compute_capability(self, mock_detect):
        cc = ComputeCapability(major=8, minor=9, vendor=GPUVendor.NVIDIA)
        mock_detect.return_value = (GPUVendor.NVIDIA, cc, "RTX 4090")
        result = detect_compute_capability()
        assert result is not None
        assert result.as_int == 89

    @patch("carm_roofline.gpu.detect.detect_gpu")
    def test_detect_compute_capability_failure(self, mock_detect):
        mock_detect.side_effect = FileNotFoundError("no tools")
        assert detect_compute_capability() is None


class TestReadGPUFrequencies:
    """Tests for read_gpu_frequencies."""

    @patch("carm_roofline.gpu.detect.shutil.which")
    @patch("carm_roofline.gpu.detect.subprocess.run")
    def test_read_gpu_frequencies(self, mock_run, mock_which):
        mock_which.return_value = "/usr/bin/nvidia-smi"
        mock_run.return_value = Mock(
            stdout="2100, 5000\n",
            stderr="",
            returncode=0,
        )

        freqs = read_gpu_frequencies()
        assert freqs == {"sm": 2100, "mem": 5000}

        mock_run.assert_called_once_with(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem", "-i", "0", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )

    @patch("carm_roofline.gpu.detect.shutil.which")
    @patch("carm_roofline.gpu.detect.subprocess.run")
    def test_read_gpu_frequencies_with_mhz_suffix(self, mock_run, mock_which):
        """Real nvidia-smi output includes ' MHz' suffix."""
        mock_which.return_value = "/usr/bin/nvidia-smi"
        mock_run.return_value = Mock(
            stdout="855 MHz, 810 MHz\n",
            stderr="",
            returncode=0,
        )

        freqs = read_gpu_frequencies()
        assert freqs == {"sm": 855, "mem": 810}

    @patch("carm_roofline.gpu.detect.shutil.which")
    def test_read_gpu_frequencies_no_tool(self, mock_which):
        mock_which.return_value = None
        with pytest.raises(UserError):
            read_gpu_frequencies()

    @patch("carm_roofline.gpu.detect.shutil.which")
    @patch("carm_roofline.gpu.detect.subprocess.run")
    def test_read_gpu_frequencies_empty(self, mock_run, mock_which):
        mock_which.return_value = "/usr/bin/nvidia-smi"
        mock_run.return_value = Mock(stdout="\n", stderr="", returncode=0)
        with pytest.raises(UserError, match="empty frequency data"):
            read_gpu_frequencies()


class TestLockGPUFrequencies:
    """Tests for lock_gpu_frequencies and reset_gpu_clocks."""

    @patch("carm_roofline.gpu.detect.shutil.which")
    @patch("carm_roofline.gpu.detect.subprocess.run")
    def test_lock_sm_clock(self, mock_run, mock_which):
        mock_which.return_value = "/usr/bin/nvidia-smi"
        mock_run.return_value = Mock(stdout="", stderr="", returncode=0)

        lock_gpu_frequencies(sm_clock=2100)
        mock_run.assert_called_once_with(
            ["nvidia-smi", "-i", "0", "-lgc", "2100"],
            capture_output=True,
            text=True,
            check=True,
        )

    @patch("carm_roofline.gpu.detect.shutil.which")
    @patch("carm_roofline.gpu.detect.subprocess.run")
    def test_lock_mem_clock(self, mock_run, mock_which):
        mock_which.return_value = "/usr/bin/nvidia-smi"
        mock_run.return_value = Mock(stdout="", stderr="", returncode=0)

        lock_gpu_frequencies(mem_clock=5000)
        mock_run.assert_called_once_with(
            ["nvidia-smi", "-i", "0", "-lmc", "5000"],
            capture_output=True,
            text=True,
            check=True,
        )

    @patch("carm_roofline.gpu.detect.shutil.which")
    @patch("carm_roofline.gpu.detect.subprocess.run")
    def test_lock_both_clocks(self, mock_run, mock_which):
        mock_which.return_value = "/usr/bin/nvidia-smi"
        mock_run.return_value = Mock(stdout="", stderr="", returncode=0)

        lock_gpu_frequencies(sm_clock=2100, mem_clock=5000)
        assert mock_run.call_count == 2

    @patch("carm_roofline.gpu.detect.shutil.which")
    def test_lock_no_tool(self, mock_which):
        mock_which.return_value = None
        with pytest.raises(UserError):
            lock_gpu_frequencies(sm_clock=2100)

    @patch("carm_roofline.gpu.detect.shutil.which")
    @patch("carm_roofline.gpu.detect.subprocess.run")
    def test_lock_failure(self, mock_run, mock_which):
        mock_which.return_value = "/usr/bin/nvidia-smi"
        from subprocess import CalledProcessError

        mock_run.side_effect = CalledProcessError(2, "nvidia-smi", stderr="lock failed")
        with pytest.raises(UserError, match="lock failed"):
            lock_gpu_frequencies(sm_clock=2100)

    @patch("carm_roofline.gpu.detect.shutil.which")
    @patch("carm_roofline.gpu.detect.subprocess.run")
    def test_reset_gpu_clocks(self, mock_run, mock_which):
        mock_which.return_value = "/usr/bin/nvidia-smi"
        mock_run.return_value = Mock(stdout="", stderr="", returncode=0)

        reset_gpu_clocks()
        assert mock_run.call_count == 2  # -rgc + -rmc

    @patch("carm_roofline.gpu.detect.shutil.which")
    @patch("carm_roofline.gpu.detect.subprocess.run")
    def test_reset_gpu_clocks_no_tool(self, mock_run, mock_which):
        mock_which.return_value = None
        reset_gpu_clocks()
        mock_run.assert_not_called()

    @patch("carm_roofline.gpu.detect.shutil.which")
    @patch("carm_roofline.gpu.detect.subprocess.run")
    def test_reset_gpu_clocks_partial_failure(self, mock_run, mock_which):
        """reset_gpu_clocks warns on failure, does not raise."""
        mock_which.return_value = "/usr/bin/nvidia-smi"
        from subprocess import CalledProcessError

        mock_run.side_effect = CalledProcessError(2, "nvidia-smi")
        # Should not raise
        reset_gpu_clocks()
