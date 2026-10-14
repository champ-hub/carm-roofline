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


def selection_payload(
    click_data: dict[str, Any] | None,
    triggered_prop_id: str,
    current_selection: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return selected application identity and residency, or clear on background clicks."""
    if triggered_prop_id.startswith(f"{PlotAreaID.BG_CLICK_CLEAR}."):
        return None
    if click_data is None:
        return current_selection
    points = click_data.get("points")
    if not isinstance(points, list) or not points or not isinstance(points[0], dict):
        return None
    customdata = points[0].get("customdata")
    if not isinstance(customdata, list) or len(customdata) != 5:
        return None
    fractions, roof_id, app_id, point_index = customdata[1:]
    if (
        not isinstance(fractions, dict)
        or not isinstance(roof_id, str)
        or not roof_id
        or not isinstance(app_id, str)
        or not app_id
        or not isinstance(point_index, int)
        or isinstance(point_index, bool)
        or point_index < 0
    ):
        return None
    try:
        parsed_fractions = {key: float(value) for key, value in fractions.items()}
    except (TypeError, ValueError):
        return None
    return {"roof_id": roof_id, "fractions": parsed_fractions, "app_id": app_id, "point_index": point_index}


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
