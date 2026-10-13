from __future__ import annotations

from dataclasses import dataclass, field

from carm_roofline.gui.config import GUIConfig, GUIMode
from carm_roofline.gui.data import DropdownOption
from carm_roofline.gui.providers import ParaverData
from carm_roofline.output_utils import debug
from carm_roofline.roofline_assembly import ApplicationRecord, BenchmarkRecord, FilterOptions


@dataclass
class CallbackState:
    """Data shared by GUI callback groups and updated after benchmark runs."""

    config: GUIConfig
    records: list[BenchmarkRecord] = field(default_factory=list)
    options: FilterOptions | None = None
    apps_by_id: dict[str, ApplicationRecord] = field(default_factory=dict)
    app_options: list[DropdownOption] = field(default_factory=list)
    paraver_data: ParaverData | None = None
    mode: GUIMode = GUIMode.CARM
    callback_count: int = 0

    def trace(self, message: str) -> None:
        """Emit sequence-numbered callback debug output when enabled."""
        self.callback_count += 1
        if self.config.gui_debug:
            debug(f"[CB#{self.callback_count}] {message}")
