"""
System Components for Field-Ready Operations

This module contains system-level components for autonomous field operations,
including checkpoint/recovery, self-healing, and systemd integration.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

# State persistence and recovery
from .state_persistor import StatePersistor
from .self_heal import HealthStatus, SelfHeal

__all__ = [
    "StatePersistor",
    "SelfHeal",
    "HealthStatus",
]
