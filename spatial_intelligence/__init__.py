#!/usr/bin/env python3
"""
Spatial Intelligence Module (Level 2.5)

Fuses acoustic data with spatial tracking for closed-loop interaction.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

from spatial_intelligence.deeplabcut_ingestor import (
    CameraSource,
    create_test_camera_config,
    DeepLabCutIngestor,
    DLCCameraConfig,
    PoseKeypoints,
)
from spatial_intelligence.spatial_ingestor import (
    SimulatedIngestor,
    SpatialFrame,
    SpatialIngestor,
    SpatialObservation,
    TrackingSource,
)
from spatial_intelligence.topology_engine import (
    AgentState,
    ColonyTopology,
    LineOfSightResult,
    ProximityResult,
    TopologyEngine,
)

__all__ = [
    # Spatial Ingestion
    "SpatialObservation",
    "SpatialFrame",
    "SpatialIngestor",
    "SimulatedIngestor",
    "TrackingSource",
    # DeepLabCut Integration
    "DeepLabCutIngestor",
    "DLCCameraConfig",
    "PoseKeypoints",
    "CameraSource",
    "create_test_camera_config",
    # Topology Engine
    "TopologyEngine",
    "AgentState",
    "ProximityResult",
    "LineOfSightResult",
    "ColonyTopology",
]
