from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypedDict, cast

from carm_roofline.gui.data import DropdownOption, RoofConfig, RoofStore, format_roof_label
from carm_roofline.gui.ids import PlotAreaID, RoofCardID
from carm_roofline.roofline_assembly import ApplicationRecord


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


class RoofControlPatch(TypedDict, total=False):
    """Sparse values received from one roof card's controls."""

    machine: str | None
    isa: str | None
    data_type: str | None
    load_store_ratio: str | None
    num_threads: int | None
    actual_frequency_hz: int | None
    compute_insts: list[str] | None
    app_ids: list[str] | None


def decode_roof_control_patches(input_groups: Sequence[object]) -> dict[str, RoofControlPatch]:
    """Decode concrete Dash control components into sparse patches by stable roof ID."""
    patches: dict[str, RoofControlPatch] = {}
    for raw_group in input_groups:
        group = raw_group if isinstance(raw_group, list) else [raw_group]
        for raw_component in group:
            if not isinstance(raw_component, dict):
                continue
            component = cast(dict[str, object], raw_component)
            component_id = component.get("id")
            if not isinstance(component_id, dict) or component.get("property") != "value":
                continue
            component_id = cast(dict[str, object], component_id)
            roof_id = component_id.get("roof_id")
            if not isinstance(roof_id, str):
                continue
            component_type = component_id.get("type")
            value = component.get("value")
            patch = patches.setdefault(roof_id, {})
            if component_type == RoofCardID.DROPDOWN_MACHINE:
                patch["machine"] = cast("str | None", value)
            elif component_type == RoofCardID.DROPDOWN_ISA:
                patch["isa"] = cast("str | None", value)
            elif component_type == RoofCardID.DROPDOWN_DATA_TYPE:
                patch["data_type"] = cast("str | None", value)
            elif component_type == RoofCardID.DROPDOWN_LS_RATIO:
                patch["load_store_ratio"] = cast("str | None", value)
            elif component_type == RoofCardID.DROPDOWN_THREADS:
                patch["num_threads"] = None if value is None else int(cast("str | int", value))
            elif component_type == RoofCardID.DROPDOWN_FREQUENCY:
                patch["actual_frequency_hz"] = None if value is None else int(cast("str | int", value))
            elif component_type == RoofCardID.DROPDOWN_COMPUTE:
                patch["compute_insts"] = cast("list[str] | None", value)
            elif component_type == RoofCardID.DROPDOWN_APPS:
                patch["app_ids"] = cast("list[str] | None", value)
    return patches


def apply_roof_controls(store: RoofStore, patches: Mapping[str, RoofControlPatch]) -> None:
    """Merge submitted controls into matching roofs and refresh their labels."""
    for roof in store.roofs:
        values = patches.get(roof.id)
        if values is None:
            continue
        if "machine" in values:
            previous_machine = roof.machine
            roof.machine = values["machine"]
            if roof.machine != previous_machine:
                roof.app_ids = []
        if "isa" in values:
            roof.isa = values["isa"]
        if "num_threads" in values:
            roof.num_threads = values["num_threads"]
        if "compute_insts" in values and values["compute_insts"] is not None:
            roof.compute_insts = values["compute_insts"]
        if "data_type" in values:
            roof.data_type = values["data_type"]
        if "load_store_ratio" in values:
            roof.load_store_ratio = values["load_store_ratio"]
        if "actual_frequency_hz" in values:
            roof.actual_frequency_hz = values["actual_frequency_hz"]
        if "app_ids" in values:
            roof.app_ids = list(values["app_ids"] or [])
    for roof in store.roofs:
        roof.label = format_roof_label(roof.machine, roof.isa, roof.num_threads, roof.data_type)


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
