from __future__ import annotations

import json
from typing import Any, cast

from dash import callback_context

from carm_roofline.gui.data import DropdownOption, RoofConfig
from carm_roofline.gui.ids import PlotAreaID
from carm_roofline.roofline_assembly import ApplicationRecord


def parse_trigger_id() -> dict[str, Any]:
    """Parse the pattern-matching ID dict from the triggered input's prop_id."""
    context = callback_context
    if not context.triggered:
        return {}
    prop_id = context.triggered[0].get("prop_id", "")
    id_string, _prop = prop_id.rsplit(".", 1)
    try:
        return cast(dict[str, Any], json.loads(id_string))
    except (json.JSONDecodeError, ValueError):
        return {}


def clicked_point_residency(click_data: dict[str, Any] | None) -> tuple[str, dict[str, float]] | None:
    """Return a roof ID and residency fractions when clickData identifies an application point."""
    if not click_data:
        return None
    points = click_data.get("points") or []
    if not points:
        return None
    customdata = points[0].get("customdata")
    if not isinstance(customdata, list) or len(customdata) != 3 or not customdata[1]:
        return None
    roof_id, fractions = customdata[2], customdata[1]
    if not isinstance(roof_id, str) or not isinstance(fractions, dict):
        return None
    return roof_id, {key: float(value) for key, value in fractions.items()}


def selection_payload(
    click_data: dict[str, Any] | None,
    triggered_prop_id: str,
    current_selection: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return selected roof data, preserving selection on callback echoes and clearing on background clicks."""
    if triggered_prop_id.startswith(f"{PlotAreaID.BG_CLICK_CLEAR}."):
        return None
    if click_data is None:
        return current_selection
    clicked = clicked_point_residency(click_data)
    if clicked is None:
        return None
    return {"roof_id": clicked[0], "fractions": clicked[1]}


def get_trigger_index() -> int:
    """Extract the ``index`` field from the triggered input's pattern-matching ID."""
    trigger_id = parse_trigger_id()
    try:
        return int(trigger_id.get("index", -1))
    except (ValueError, TypeError):
        return -1


def filter_app_options_for_roofs(
    roofs: list[RoofConfig],
    app_dropdown_options: list[DropdownOption] | None,
    app_by_id: dict[str, ApplicationRecord] | None,
) -> list[list[DropdownOption]] | None:
    """Filter app dropdown options per roof by machine."""
    if not app_dropdown_options or not app_by_id:
        return None
    return [
        [option for option in app_dropdown_options if app_by_id[option["value"]].machine == roof.machine]
        for roof in roofs
    ]
