"""
Spatial Awareness Module (Level 2.5)

Provides spatial topology management and receiver inference for
targeted vocalization and colony-level awareness.

Components:
- TopologyEngine: Manages 3D spatial graph of bat colony
- ReceiverInferenceEngine: Predicts intended receiver of vocalizations
- EmitterSelection: Selects optimal speaker for targeted playback

Example Usage:
    >>> from spatial import (
    ...     TopologyEngine,
    ...     ReceiverInferenceEngine,
    ...     EmitterSelection,
    ... )
    >>>
    >>> # Create topology
    >>> topology = TopologyEngine()
    >>> topology.update_node(bat_id=1, position=[0, 0, 1])
    >>>
    >>> # Infer receiver
    >>> engine = ReceiverInferenceEngine()
    >>> prediction = engine.infer_receiver(1, topology)
    >>> print(f"Address type: {prediction.address_type}")

Author: Zoo Vox Research Team
License: CC BY-ND 4.0 International
"""

from .receiver_inference import (
    AddressType,
    InteractionLogger,
    ReceiverInferenceEngine,
    ReceiverPrediction,
    DEFAULT_RECEIVER_INFERENCE,
)
from .topology_engine import (
    BatNode,
    Edge,
    EmitterSelection,
    TopologyEngine,
)

__all__ = [
    # Topology
    "BatNode",
    "Edge",
    "TopologyEngine",
    "EmitterSelection",
    # Receiver Inference
    "AddressType",
    "ReceiverInferenceEngine",
    "ReceiverPrediction",
    "InteractionLogger",
    "DEFAULT_RECEIVER_INFERENCE",
]

__version__ = "1.0.0"
