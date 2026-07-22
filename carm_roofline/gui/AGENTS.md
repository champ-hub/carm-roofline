# GUI Module Conventions

## Module Structure

```
gui/
├── __init__.py        # run_app() entry point (lazy-imports create_app)
├── ids.py             # ID(str, Enum) hierarchy — all Dash component IDs
├── data.py            # Data models: RoofConfig, RoofStore, build_roofline_figure
├── components.py      # Pure-function component builders (no callbacks)
├── factory.py         # Dash app factory + callback registration
├── config.py          # GUIConfig dataclass (InsertsArguments)
└── assets/            # Static assets (CSS, images)
```

## Dash ID Conventions

All Dash ``id`` values — whether used in component definitions or callback
``Input``/``Output``/``State`` decorators — are ``str, Enum`` members in
``gui/ids.py``.  Never use bare string literals as IDs.

### Base class

```python
class ID(str, Enum):
    def __str__(self) -> str:
        return str(self.value)
```

``ID`` inherits from both ``str`` and ``Enum`` so each member is a string
instance.  This means:
- ``json.dumps(RoofCardID.DROPDOWN_MACHINE)`` → ``"dropdown-machine"``
  (the JSON encoder uses the underlying ``str`` value)
- ``NavbarID.BTN_SETTINGS in "btn-settings.n_clicks"`` → ``True``
  (string containment works directly)
- ``Output(NavbarID.BTN_CARM_VIEW, "className")`` — Dash receives a string
- ``NavbarID.BTN_CARM_VIEW == "btn-carm-view"`` → ``True``

``__str__`` returns ``str(self.value)`` for clean display; the JSON encoder
bypasses ``__str__`` and reads the raw ``str`` payload directly.

### Grouping by UI element

IDs are split into classes matching the component that owns them.
When adding a new component, create a new ``ID`` subclass for its IDs.

```python
class NavbarID(ID):
    """Top navigation bar."""
    BTN_CARM_VIEW = "btn-carm-view"
    BTN_SETTINGS = "btn-settings"

class RoofCardID(ID):
    """Roof configuration cards (pattern-matching type strings)."""
    DROPDOWN_MACHINE = "dropdown-machine"
    DROPDOWN_ISA = "dropdown-isa"
    DROPDOWN_THREADS = "dropdown-threads"
    ...
```

### Pattern-matching IDs

Pattern-matching IDs (``{"type": ..., "index": ...}``) use the same enum
members as the ``"type"`` value.  ``_make_id()`` in ``components.py``
builds the dict:

```python
def _make_id(type_: str, **parts: int) -> dict[str, str | int]:
    return {"type": type_, **parts}

# Usage:
_make_id(RoofCardID.DROPDOWN_MACHINE, index=roof_idx)
```

Because the enum member is a ``str`` subclass, ``json.dumps`` serialises it
to the bare string.  The resulting JSON dicts match the patterns in callback
decorators.

### Import style

Import only the classes you need — the class name documents the UI element:

```python
# Good — compact, self-documenting
from gui.ids import NavbarID, RoofCardID, StoreID

# Bad — sprawling flat import
from gui.ids import BTN_CARM_VIEW, BTN_SETTINGS, ROOF_STORE, ...
```

## Component Builder Pattern

Component builders live in ``components.py`` as pure functions.  They take
a ``RoofStore`` (and optional filter ``options``) and return Dash component
trees.  No callbacks — those all go in ``factory.py``.

```python
def build_navbar(active_panel: ActivePanel) -> html.Div:
    return html.Div(
        className="navbar",
        children=[
            html.Button(
                "CARM View",
                id=NavbarID.BTN_CARM_VIEW,
                ...
            ),
        ],
    )
```

Application options flow as ``app_options: list[AppOption]`` from
``load_all_applications`` in ``factory.py`` through ``build_roof_card``;
``roof.app_ids`` (selected run content hashes) and ``roof.apps_enabled`` are
the persisted state in the ``dcc.Store`` roof data.

## Callback Registration Pattern

All callbacks are registered in ``_register_callbacks()`` in ``factory.py``.
Each callback uses ID enum members in its decorators:

```python
@app.callback(
    Output(StoreID.ACTIVE_PANEL, "data"),
    Input(NavbarID.BTN_CARM_VIEW, "n_clicks"),
    Input(NavbarID.BTN_SETTINGS, "n_clicks"),
)
def _toggle_panel(...) -> ActivePanel:
    ...
```

### Trigger detection

When a callback needs to know which input triggered it:

```python
trigger = ctx.triggered[0]["prop_id"]
if NavbarID.BTN_SETTINGS in trigger:
    return ActivePanel.SETTINGS
```

String containment works because enum members are ``str`` instances.

## Serialization Round-Trip Tests

Any ``to_dict`` → ``from_dict`` path MUST have a unit test verifying every
field survives the round trip.  This pattern catches silent data loss when
``from_dict`` forgets to read a field that ``to_dict`` writes.

```python
def test_roofstore_round_trip_preserves_all_fields() -> None:
    roof = RoofConfig(compute_insts=["mul", "div"], ...)
    store = RoofStore(roof_template=roof)
    restored = RoofStore.from_dict(store.to_dict())
    assert restored.roofs[0].compute_insts == ["mul", "div"]
```
