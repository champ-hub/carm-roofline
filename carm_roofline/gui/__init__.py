from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from carm_roofline.gui.config import GUIConfig


def run_app(config: GUIConfig) -> int:
    """Create and run the Dash application.

    Args:
        config: GUI configuration with host, port, debug, and results_dir settings.

    Returns:
        Exit code (0 on clean shutdown).
    """
    from .factory import create_app

    app = create_app(config)
    app.run(host=config.gui_host, port=config.gui_port, debug=config.gui_debug)
    return 0
