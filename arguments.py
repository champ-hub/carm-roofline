from __future__ import annotations

import argparse
from argparse import ArgumentTypeError
from collections.abc import Iterable, Iterator
from enum import Enum
from typing import Any

from rich.console import Console, ConsoleOptions, RenderResult
from rich.text import Text
from rich_argparse import RichHelpFormatter


def positive_int(arg: str | int) -> int:
    if (val := int(arg)) <= 0:
        raise ArgumentTypeError(f"{arg} is not a positive integer")

    return val


def positive_float(arg: str | float) -> float:
    if (val := float(arg)) <= 0:
        raise ArgumentTypeError(f"{arg} is not a positive float")

    return val


class TopLevelHelpFormatter(RichHelpFormatter):
    "Custom help formatter for top-level parser to adjust formatting of subparser help"

    class _Section(RichHelpFormatter._Section):
        def _render_actions(self, console: Console, options: ConsoleOptions) -> RenderResult:
            if not self.rich_actions:
                return
            options = options.update(no_wrap=True, overflow="ignore")
            help_pos = min(self.formatter._action_max_length + 4, self.formatter._max_help_position)
            help_width = max(self.formatter._width - help_pos, 11)
            indent = Text(" " * help_pos)

            for action_header, action_help in self.rich_actions:
                if not action_help:
                    yield from console.render(action_header, options)
                    continue

                action_help_lines = self.formatter._rich_split_lines(action_help, help_width)
                if len(action_header) > help_pos - 2:
                    yield from console.render(action_header, options)
                    action_header = indent  # noqa: PLW2901

                action_header.set_length(help_pos)
                action_help_lines[0].rstrip()
                yield from console.render(action_header + action_help_lines[0], options)

                for line in action_help_lines[1:]:
                    line.rstrip()
                    yield from console.render(indent + line, options)

            yield ""

    def _rich_format_action(self, action: argparse.Action) -> Iterator[tuple[Text, Text | None]]:
        if isinstance(action, argparse._SubParsersAction):
            for subaction in self._iter_indented_subactions(action):
                yield from self._rich_format_action(subaction)
            return
        yield from super()._rich_format_action(action)


class InsertsArguments:
    @staticmethod
    def insert_arguments(parser: argparse.ArgumentParser) -> None:
        raise NotImplementedError

    @staticmethod
    def subclasses() -> set[type[InsertsArguments]]:
        """Get all subclasses of InsertsArguments."""
        subclasses = set()
        classes_to_check = [InsertsArguments]
        while classes_to_check:
            parent = classes_to_check.pop()
            for children in parent.__subclasses__():
                if children not in subclasses:
                    subclasses.add(children)
                    classes_to_check.append(children)

        return subclasses


# Source - https://stackoverflow.com/a/78910354
# Posted by Hai Vu, modified by community. See post 'Timeline' for change history
# Retrieved 2026-02-10, License - CC BY-SA 4.0


def enum_action(enum_class: type[Enum]) -> type[argparse.Action]:
    class EnumAction(argparse.Action):
        def __init__(self, *args: Any, **kwargs: Any):
            # set our table as choices if not already set
            if "choices" in kwargs:
                choices = kwargs["choices"]
                assert isinstance(choices, dict), "choices must be a dict[str, Enum]"
                table: dict[str, Enum] = choices
            else:
                table = {member.name.casefold(): member for member in enum_class}
                kwargs["choices"] = table

            super().__init__(*args, **kwargs)
            self.table = table

        def __call__(
            self,
            _parser: argparse.ArgumentParser,
            namespace: argparse.Namespace,
            values: str | Iterable[Any] | None,
            _option_string: str | None = None,
        ) -> None:
            if values is None:
                raise TypeError("Expected string or iterable values, got None")

            if isinstance(values, str):
                setattr(namespace, self.dest, self.table[values])
                return

            # if it isn't a str, it should be an iterable of str
            mapped_values: list[Enum] = []
            for value in values:
                if not isinstance(value, str):
                    raise TypeError(f"Expected string enum value, got {type(value)}")
                mapped_values.append(self.table[value])

            setattr(namespace, self.dest, mapped_values)

    return EnumAction


def inheritors(cls: type) -> set[type]:
    subclasses = set()
    work = [cls]
    while work:
        parent = work.pop()
        for child in parent.__subclasses__():
            if child not in subclasses:
                subclasses.add(child)
                work.append(child)
    return subclasses


def check_args_validity(args: argparse.Namespace) -> None:
    pass
