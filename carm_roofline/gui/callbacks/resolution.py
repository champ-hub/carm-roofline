from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from carm_roofline.gui.callbacks.helpers import RoofControlPatch, apply_roof_controls
from carm_roofline.gui.callbacks.state import CallbackState
from carm_roofline.gui.data import ActivePanel, RoofConfig, RoofStore
from carm_roofline.output_utils import debug
from carm_roofline.roofline_assembly import FilterOptions, RooflineFilter, discover_filter_options


def _first_or_none(values: list[str]) -> str | None:
    return values[0] if values else None


def _first_int_or_none(values: list[int]) -> int | None:
    return values[0] if values else None


_FILTER_FIELDS: tuple[tuple[str, str, bool], ...] = (
    ("machine", "machine", False),
    ("isa", "isa", False),
    ("num_threads", "num_threads", True),
    ("data_type", "data_type", False),
    ("load_store_ratio", "load_store_ratio", False),
    ("actual_frequency_hz", "actual_frequency_hz", True),
)


def normalize_selected_roof_controls(state: CallbackState, store: RoofStore) -> None:
    """Clear selected filters no longer available after applying current controls."""
    if not state.records:
        return
    for roof in store.roofs:
        base_filter = RooflineFilter(
            machine=roof.machine,
            isa=roof.isa,
            num_threads=roof.num_threads,
            data_type=roof.data_type,
            actual_frequency_hz=roof.actual_frequency_hz,
            load_store_ratio=roof.load_store_ratio,
        )
        available = cast(dict[str, list[Any]], discover_filter_options(state.records, base_filter))
        for field_name, filter_key, _is_int in _FILTER_FIELDS:
            value = getattr(roof, field_name)
            if value is not None and value not in available[filter_key]:
                debug(f"_resolve_roof_data[{roof.id}]: {field_name} '{value}' not in options -> None")
                setattr(roof, field_name, None)


def resolve_roofs(
    state: CallbackState,
    store_data: dict[str, Any] | None,
    roof_values: Mapping[str, RoofControlPatch],
    active_panel: str | None = None,
) -> tuple[RoofStore, list[RoofConfig], list[FilterOptions | None]]:
    """Merge current controls into the store and resolve incomplete roof fields."""
    store = RoofStore.from_dict(store_data or {})
    if active_panel is not None:
        store.active_panel = ActivePanel(active_panel) if active_panel else ActivePanel.CARM_VIEW
    apply_roof_controls(store, roof_values)
    normalize_selected_roof_controls(state, store)

    debug(f"_resolve_roof_data: {len(store.roofs)} roof(s), panel={store.active_panel}")
    resolved_roofs: list[RoofConfig] = []
    per_roof_options: list[FilterOptions | None] = []
    records = state.records
    for roof in store.roofs:
        if not records:
            per_roof_options.append(None)
            resolved_roofs.append(roof)
            continue

        user_locks = RooflineFilter(**{name: getattr(roof, name) for name, _, _ in _FILTER_FIELDS})
        accumulated: dict[str, Any] = {name: getattr(roof, name) for name, _, _ in _FILTER_FIELDS}
        resolved_values: dict[str, Any] = {
            "roof_id": roof.id,
            "compute_insts": roof.compute_insts,
            "app_ids": roof.app_ids,
        }
        for field_name, filter_key, is_int in _FILTER_FIELDS:
            value = getattr(roof, field_name)
            if value is not None:
                resolved_values[field_name] = value
            else:
                available_for_field = cast(
                    dict[str, list[Any]], discover_filter_options(records, RooflineFilter(**accumulated))
                )[filter_key]
                resolved_values[field_name] = (
                    _first_int_or_none(available_for_field) if is_int else _first_or_none(available_for_field)
                )
            accumulated[field_name] = resolved_values[field_name]

        resolved_roofs.append(
            RoofConfig(
                roof_id=roof.id,
                label=roof.label,
                machine=resolved_values["machine"],
                isa=resolved_values["isa"],
                num_threads=resolved_values["num_threads"],
                data_type=resolved_values["data_type"],
                compute_insts=roof.compute_insts,
                load_store_ratio=resolved_values["load_store_ratio"],
                actual_frequency_hz=resolved_values["actual_frequency_hz"],
                app_ids=roof.app_ids,
            )
        )
        per_roof_options.append(discover_filter_options(records, user_locks))
    return store, resolved_roofs, per_roof_options
