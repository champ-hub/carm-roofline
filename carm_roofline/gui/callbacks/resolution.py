from __future__ import annotations

from typing import Any, cast

from carm_roofline.gui.callbacks.state import CallbackState
from carm_roofline.gui.data import ActivePanel, RoofConfig, RoofStore, format_roof_label
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


def resolve_roofs(
    state: CallbackState,
    store_data: dict[str, Any] | None,
    machine_values: list[str | None],
    isa_values: list[str | None],
    thread_values: list[str | None],
    compute_values: list[list[str] | None],
    data_type_values: list[str | None],
    load_store_values: list[str | None],
    frequency_values: list[str | None],
    app_id_values: list[list[str] | None],
    active_panel: str | None = None,
) -> tuple[RoofStore, list[RoofConfig], list[FilterOptions | None]]:
    """Merge current controls into the store and resolve incomplete roof fields."""
    store = RoofStore.from_dict(store_data or {})
    if active_panel is not None:
        store.active_panel = ActivePanel(active_panel) if active_panel else ActivePanel.CARM_VIEW
    for index, roof in enumerate(store.roofs):
        if index < len(machine_values):
            previous_machine = roof.machine
            roof.machine = machine_values[index]
            if roof.machine != previous_machine:
                roof.app_ids = []
        if index < len(isa_values):
            roof.isa = isa_values[index]
        if index < len(thread_values):
            value = thread_values[index]
            roof.num_threads = int(value) if value is not None else None
        if index < len(compute_values):
            compute_value = compute_values[index]
            if compute_value is not None:
                roof.compute_insts = compute_value
        if index < len(data_type_values):
            roof.data_type = data_type_values[index]
        if index < len(load_store_values):
            roof.load_store_ratio = load_store_values[index]
        if index < len(frequency_values):
            value = frequency_values[index]
            roof.actual_frequency_hz = int(value) if value is not None else None
        if index < len(app_id_values):
            roof.app_ids = list(app_id_values[index] or [])

    for roof in store.roofs:
        roof.label = format_roof_label(roof.machine, roof.isa, roof.num_threads, roof.data_type)

    debug(f"_resolve_roof_data: {len(store.roofs)} roof(s), panel={store.active_panel}")
    resolved_roofs: list[RoofConfig] = []
    per_roof_options: list[FilterOptions | None] = []
    records = state.records
    for roof in store.roofs:
        if not records:
            per_roof_options.append(None)
            resolved_roofs.append(roof)
            continue

        base_filter = RooflineFilter(
            machine=roof.machine,
            isa=roof.isa,
            num_threads=roof.num_threads,
            data_type=roof.data_type,
            actual_frequency_hz=roof.actual_frequency_hz,
            load_store_ratio=roof.load_store_ratio,
        )
        available = cast(dict[str, list[Any]], discover_filter_options(records, base_filter))
        for field_name, filter_key, _is_int in _FILTER_FIELDS:
            value = getattr(roof, field_name)
            if value is not None and value not in available[filter_key]:
                debug(f"_resolve_roof_data[{roof.id}]: {field_name} '{value}' not in options -> None")
                setattr(roof, field_name, None)

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
