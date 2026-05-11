"""
Monitoring Module

Provides telemetry, monitoring, and dashboard capabilities for
shadow mode testing, acclimation phase, and closed-loop deployment.

Components:
- ShadowModeTelemetry: Performance monitoring for passive pipeline testing
- AcclimationMonitor: Colony response monitoring during acclimation
- DeploymentDashboard: Real-time metrics for closed-loop interaction

Example Usage:
    >>> from monitoring import (
    ...     ShadowModeTelemetry,
    ...     AcclimationMonitor,
    ...     DeploymentDashboard,
    ... )
    >>>
    >>> # Shadow mode telemetry
    >>> telemetry = ShadowModeTelemetry()
    >>> telemetry.record_frame(...)
    >>> report = telemetry.generate_report()
    >>>
    >>> # Acclimation monitoring
    >>> monitor = AcclimationMonitor("rousettus_aegyptiacus")
    >>> with monitor.start_response_window() as window:
    ...     window.add_bat_response(...)

Author: Zoo Vox Research Team
License: CC BY-ND 4.0 International
"""

from .acclimation_monitor import (
    AcclimationAlert,
    AcclimationMonitor,
    ColonyResponse,
    DeploymentDashboard,
    PlaybackEvent,
    ResponseWindow,
    BAT_ACCLIMATION_MONITOR,
)
from .shadow_mode_telemetry import (
    FrameTiming,
    ShadowModeReport,
    ShadowModeRunner,
    ShadowModeTelemetry,
)

__all__ = [
    # Shadow Mode
    "FrameTiming",
    "ShadowModeReport",
    "ShadowModeTelemetry",
    "ShadowModeRunner",
    # Acclimation
    "PlaybackEvent",
    "ColonyResponse",
    "AcclimationAlert",
    "AcclimationMonitor",
    "ResponseWindow",
    "DeploymentDashboard",
    # Presets
    "BAT_ACCLIMATION_MONITOR",
]

__version__ = "1.0.0"
