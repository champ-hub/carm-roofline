## Type safety requirements

Write Python as if it were a statically typed language. The code must be designed to pass `mypy` and `ruff` without suppressions.

- Add complete type annotations to all public functions, methods, and non-trivial variables.
- Prefer precise types over `Any`. Do not introduce `Any` unless unavoidable; if used, explain why.
- Use `TypedDict`, `dataclass`, `Protocol`, `Literal`, `Enum`, and generics to model data instead of loose dictionaries and strings.
- Avoid dynamic patterns that defeat static analysis (untyped dicts, magic attribute access, runtime monkey patching).
- Do not use `# type: ignore` unless there is no reasonable typed solution; document the reason inline.
- Prefer explicit return types and narrow interfaces.

## Avoid stringly-typed code

Do not duplicate important string literals across the codebase.

- Use enums/constants for identifiers, command names, event names, API fields, Dash component IDs, etc.
- If a value has a fixed set of valid options, represent it as an `Enum` or `Literal`, not `str`.
- Centralize shared identifiers so renaming causes a type-checker failure instead of a runtime bug.

## Error handling

- Make invalid states unrepresentable where practical.
- Prefer typed exceptions and explicit error handling over returning sentinel values like `None`, `-1`, or empty strings.
- Use `Optional` (`| None` in practice) only when absence is a valid state.

## Tooling expectations

Before considering a task complete:

- Run `mypy` on changed code.
- Run `ruff check --fix` on changed code.
- Fix all new type errors and lint errors.
- Do not weaken type-checking configuration to make code pass.

The goal is for refactoring mistakes to fail during development rather than at runtime.

## Prefer these patterns

- Use `dataclass(frozen=True)` for immutable value objects.
- Prefer small typed functions over large functions with implicit state.

Avoid:
```python
dict[str, Any]
tuple
list
object
str
```
when a more precise type can describe the data.

## General

In documentation, refer to this software as "the CARM Tool".
