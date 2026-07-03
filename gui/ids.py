"""Dash component ID constants for the CARM Roofline GUI.

IDs are grouped into ``str, Enum`` classes by UI element.  Using a
``str, Enum`` means each member compares equal to its bare string value
(``NavbarID.BTN_CARM_VIEW == "btn-carm-view"``) while carrying a single
importable name for autocomplete and mypy enforcement.
"""

from __future__ import annotations

from enum import Enum


class ID(str, Enum):
    """Base class for Dash ID constants.

    ``__str__`` returns the member's value so Dash serialization and
    string formatting produce the bare string ID.
    """

    def __str__(self) -> str:
        return str(self.value)


class StoreID(ID):
    """IDs for ``dcc.Store`` components and their callback I/O."""

    ROOF_STORE = "roof-store"
    ACTIVE_PANEL = "active-panel"


class NavbarID(ID):
    """IDs for the top navigation bar."""

    BTN_CARM_VIEW = "btn-carm-view"
    BTN_SETTINGS = "btn-settings"


class SidebarID(ID):
    """IDs for the sidebar container."""

    SIDEBAR_CONTENT = "sidebar-content"


class CarmViewPanelID(ID):
    """IDs for the CARM View panel (add-roof button)."""

    BTN_ADD_ROOF = "btn-add-roof"


class RoofCardID(ID):
    """IDs for roof configuration cards (pattern-matching type strings)."""

    BTN_REMOVE_ROOF = "btn-remove-roof"
    DROPDOWN_MACHINE = "dropdown-machine"
    DROPDOWN_ISA = "dropdown-isa"
    DROPDOWN_THREADS = "dropdown-threads"
    DROPDOWN_DATA_TYPE = "dropdown-data-type"
    DROPDOWN_COMPUTE = "dropdown-compute"
    DROPDOWN_LS_RATIO = "dropdown-ls-ratio"
    DROPDOWN_APPS = "dropdown-apps"
    BTN_COLLAPSE_ROOF = "btn-collapse-roof"
    SWITCH_APPS = "switch-apps"


class SettingsPanelID(ID):
    """IDs for the settings panel (select/deselect all buttons)."""

    BTN_SELECT_ALL = "btn-select-all"
    BTN_DESELECT_ALL = "btn-deselect-all"


class SelectionRowID(ID):
    """IDs for roof visibility selection rows in the settings panel."""

    CHECKBOX_ROOF_VIS = "checkbox-roof-vis"


class PlotAreaID(ID):
    """IDs for the roofline plot graph."""

    ROOFLINE_PLOT = "roofline-plot"
