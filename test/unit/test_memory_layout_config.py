from __future__ import annotations

import pytest

from carm_roofline.architecture.config import emit_template_toml, load_memory_topology_from_toml
from carm_roofline.architecture.memory import SimpleMemoryTopology
from carm_roofline.carm import _create_parser
from carm_roofline.core import AutoMemoryLayout, Bytes, MemoryLayoutMode, UserError

pytestmark = pytest.mark.unit


def test_topology_config_loads_per_entry_layouts_and_defaults_to_auto(tmp_path):
    config = tmp_path / "topology.toml"
    config.write_text(
        '[[cache_levels]]\ninstances = 4\nsize = "32KiB"\nlayout = "single"\n'
        '[[cache_levels]]\ninstances = 1\nsize = "8GiB"\n',
        encoding="utf-8",
    )

    topology = load_memory_topology_from_toml(config)

    assert topology.memory_layouts == (MemoryLayoutMode.single, AutoMemoryLayout.auto)


def test_topology_config_rejects_invalid_layout(tmp_path):
    config = tmp_path / "topology.toml"
    config.write_text(
        '[[cache_levels]]\ninstances = 1\nsize = "32KiB"\nlayout = "together"\n',
        encoding="utf-8",
    )

    with pytest.raises(UserError, match="Cache level layout"):
        load_memory_topology_from_toml(config)


def test_emitted_template_has_per_level_auto_layout(tmp_path):
    config = tmp_path / "template.toml"
    emit_template_toml(config)

    topology = load_memory_topology_from_toml(config)

    assert topology.memory_layouts == (AutoMemoryLayout.auto,) * 4


def test_cli_memory_layout_parser_accepts_space_separated_modes():
    args = _create_parser().parse_args(["benchmark", "--mem-layouts", "single", "auto"])
    assert args.mem_layouts == [MemoryLayoutMode.single, AutoMemoryLayout.auto]


def test_cli_memory_layout_parser_rejects_unknown_mode():
    with pytest.raises(SystemExit):
        _create_parser().parse_args(["benchmark", "--mem-layouts", "interleaved"])


def test_topology_rejects_empty_layout_list_when_explicit():
    with pytest.raises(UserError, match="same length"):
        SimpleMemoryTopology([Bytes(32)], [1], memory_layouts=())
