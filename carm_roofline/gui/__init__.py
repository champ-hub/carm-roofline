from __future__ import annotations

from typing import TYPE_CHECKING

from carm_roofline.output_utils import info

if TYPE_CHECKING:
    from carm_roofline.gui.config import GUIConfig


def run_app(config: GUIConfig) -> int:
    """Create and run the Dash application.

    Arms the parent-death watchdog and reserves a free TCP port BEFORE the
    expensive app setup (paraver trace loading), so a stale GUI instance can
    neither orphan this process nor block its launch.

    Args:
        config: GUI configuration with host, port, debug, and results_dir settings.

    Returns:
        Exit code (0 on clean shutdown).
    """
    from .factory import create_app
    from .launch import reserve_free_port, set_parent_death_signal

    set_parent_death_signal()
    port, reserved = reserve_free_port(config.gui_host, config.gui_port)
    try:
        app = create_app(config)
    finally:
        # Release the reservation only once the server is about to bind.
        reserved.close()

    display_host = "localhost" if config.gui_host in ("0.0.0.0", "::") else config.gui_host
    info(f"Starting CARM GUI on http://{display_host}:{port}/")
    app.run(host=config.gui_host, port=port, debug=config.gui_debug)
    return 0
