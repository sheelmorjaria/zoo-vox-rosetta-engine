#!/usr/bin/env python3
"""
Spatial Topology Engine for Level 2.5 Awareness

Manages 3D spatial relationships between bats in the colony.
Enables receiver inference and emitter selection for targeted
vocalization.

Author: Zoo Vox Research Team
License: CC BY-ND 4.0 International
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BatNode:
    """
    Represents a single bat in 3D space.

    Attributes:
        bat_id: Unique identifier for the bat
        position: (x, y, z) coordinates in world space (meters)
        velocity: (vx, vy, vz) velocity vector (m/s)
        last_update_ms: Timestamp of last position update
        pose_confidence: Confidence of pose estimation (0-1)
        is_vocalizing: Whether bat is currently vocalizing
        vocalization_start_ms: Timestamp of vocalization start
    """
    bat_id: int
    position: np.ndarray  # (3,) array
    velocity: np.ndarray  # (3,) array
    last_update_ms: float
    pose_confidence: float = 1.0
    is_vocalizing: bool = False
    vocalization_start_ms: Optional[float] = None
    arousal_level: float = 0.0  # From VAE affect vector

    def __post_init__(self):
        """Ensure arrays are numpy arrays."""
        self.position = np.asarray(self.position, dtype=np.float32)
        self.velocity = np.asarray(self.velocity, dtype=np.float32)


@dataclass
class Edge:
    """
    Represents relationship between two bats.

    Attributes:
        from_bat: Source bat ID
        to_bat: Target bat ID
        distance: Euclidean distance (meters)
        line_of_sight: Whether clear line of sight exists
        social_tie_strength: Historical interaction strength (0-1)
        last_interaction_ms: Timestamp of last interaction
    """
    from_bat: int
    to_bat: int
    distance: float
    line_of_sight: bool
    social_tie_strength: float = 0.5
    last_interaction_ms: Optional[float] = None


class TopologyEngine:
    """
    Manages spatial topology of the bat colony.

    Maintains a graph of bat nodes and edges, tracking positions,
    velocities, and social relationships. Enables spatial inference
    for receiver prediction and emitter selection.
    """

    def __init__(
        self,
        max_age_seconds: float = 1.0,
        distance_threshold: float = 5.0,
    ):
        """
        Initialize topology engine.

        Args:
            max_age_seconds: Maximum age of position data before staleness
            distance_threshold: Maximum distance for edge creation (meters)
        """
        self.nodes: Dict[int, BatNode] = {}
        self.edges: Dict[Tuple[int, int], Edge] = {}
        self.max_age_ms = max_age_seconds * 1000
        self.distance_threshold = distance_threshold

        # Obstacles for line-of-sight calculation
        self.obstacles: List[np.ndarray] = []  # List of obstacle centers

        logger.info("TopologyEngine initialized")

    def update_node(
        self,
        bat_id: int,
        position: np.ndarray,
        velocity: Optional[np.ndarray] = None,
        confidence: float = 1.0,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Update or create a bat node.

        Args:
            bat_id: Unique bat identifier
            position: (x, y, z) position in meters
            velocity: Optional velocity vector
            confidence: Pose confidence (0-1)
            timestamp: Optional timestamp (defaults to current time)
        """
        import time

        if timestamp is None:
            timestamp = time.time() * 1000

        if bat_id in self.nodes:
            # Update existing node
            node = self.nodes[bat_id]
            old_position = node.position.copy()

            # Update position
            node.position = np.asarray(position, dtype=np.float32)

            # Compute velocity if not provided
            if velocity is None and node.last_update_ms > 0:
                dt = (timestamp - node.last_update_ms) / 1000.0
                if dt > 0:
                    velocity = (node.position - old_position) / dt

            if velocity is not None:
                node.velocity = np.asarray(velocity, dtype=np.float32)

            node.last_update_ms = timestamp
            node.pose_confidence = confidence
        else:
            # Create new node
            if velocity is None:
                velocity = np.zeros(3)

            self.nodes[bat_id] = BatNode(
                bat_id=bat_id,
                position=np.asarray(position, dtype=np.float32),
                velocity=np.asarray(velocity, dtype=np.float32),
                last_update_ms=timestamp,
                pose_confidence=confidence,
            )

            logger.debug(f"Created node for bat {bat_id}")

        # Recompute edges for this node
        self._compute_edges_for_node(bat_id)

    def _compute_edges_for_node(self, bat_id: int) -> None:
        """Compute edges between bat and all nearby bats."""
        if bat_id not in self.nodes:
            return

        node = self.nodes[bat_id]

        for other_id, other_node in self.nodes.items():
            if other_id == bat_id:
                continue

            # Compute distance
            distance = np.linalg.norm(node.position - other_node.position)

            # Skip if too far
            if distance > self.distance_threshold:
                # Remove existing edge
                key = (bat_id, other_id)
                if key in self.edges:
                    del self.edges[key]
                continue

            # Check line of sight
            line_of_sight = self._check_line_of_sight(
                node.position,
                other_node.position
            )

            # Get or create edge
            key = (bat_id, other_id)
            if key not in self.edges:
                self.edges[key] = Edge(
                    from_bat=bat_id,
                    to_bat=other_id,
                    distance=distance,
                    line_of_sight=line_of_sight,
                    social_tie_strength=0.5,  # Default
                )
            else:
                # Update existing edge
                edge = self.edges[key]
                edge.distance = distance
                edge.line_of_sight = line_of_sight

    def _check_line_of_sight(
        self,
        from_pos: np.ndarray,
        to_pos: np.ndarray,
    ) -> bool:
        """
        Check if two positions have clear line of sight.

        Simple implementation: checks if any obstacle is close
        to the line segment between positions.

        Args:
            from_pos: Starting position
            to_pos: Ending position

        Returns:
            True if clear line of sight
        """
        if not self.obstacles:
            return True

        # Vector from start to end
        direction = to_pos - from_pos
        distance = np.linalg.norm(direction)
        if distance < 0.01:
            return True

        direction = direction / distance

        # Check each obstacle
        for obstacle in self.obstacles:
            # Compute closest point on line segment to obstacle
            to_obstacle = obstacle - from_pos
            projection = np.dot(to_obstacle, direction)

            # Clamp to segment
            projection = max(0, min(distance, projection))

            closest_point = from_pos + projection * direction
            dist_to_obstacle = np.linalg.norm(obstacle - closest_point)

            # If obstacle is too close to line, block line of sight
            if dist_to_obstacle < 0.3:  # 30cm threshold
                return False

        return True

    def get_edge(self, from_id: int, to_id: int) -> Optional[Edge]:
        """Get edge between two bats (bidirectional)."""
        # Try both directions
        if (from_id, to_id) in self.edges:
            return self.edges[(from_id, to_id)]
        if (to_id, from_id) in self.edges:
            return self.edges[(to_id, from_id)]
        return None

    def find_nearest_neighbors(
        self,
        bat_id: int,
        k: int = 5,
    ) -> List[Tuple[int, float]]:
        """
        Find k nearest neighbors to a bat.

        Args:
            bat_id: Bat to find neighbors for
            k: Number of neighbors to return

        Returns:
            List of (neighbor_id, distance) tuples
        """
        if bat_id not in self.nodes:
            return []

        node = self.nodes[bat_id]
        distances = []

        for other_id, other_node in self.nodes.items():
            if other_id == bat_id:
                continue

            dist = np.linalg.norm(node.position - other_node.position)
            distances.append((other_id, dist))

        # Sort by distance and return top k
        distances.sort(key=lambda x: x[1])
        return distances[:k]

    def get_nearby_bats(
        self,
        position: np.ndarray,
        radius: float = 2.0,
    ) -> List[int]:
        """
        Find all bats within radius of a position.

        Args:
            position: Query position (x, y, z)
            radius: Search radius in meters

        Returns:
            List of bat IDs within radius
        """
        position = np.asarray(position)
        nearby = []

        for bat_id, node in self.nodes.items():
            dist = np.linalg.norm(node.position - position)
            if dist <= radius:
                nearby.append(bat_id)

        return nearby

    def prune_stale_nodes(self, current_time: Optional[float] = None) -> int:
        """
        Remove nodes with stale position data.

        Args:
            current_time: Current timestamp (ms)

        Returns:
            Number of nodes pruned
        """
        import time

        if current_time is None:
            current_time = time.time() * 1000

        to_remove = []
        for bat_id, node in self.nodes.items():
            age = current_time - node.last_update_ms
            if age > self.max_age_ms:
                to_remove.append(bat_id)

        for bat_id in to_remove:
            del self.nodes[bat_id]
            # Remove associated edges
            self.edges = {
                k: v for k, v in self.edges.items()
                if bat_id not in k
            }

        if to_remove:
            logger.debug(f"Pruned {len(to_remove)} stale nodes")

        return len(to_remove)

    def update_social_tie(
        self,
        from_id: int,
        to_id: int,
        interaction_strength: float,
    ) -> None:
        """
        Update social tie strength between two bats.

        Args:
            from_id: First bat ID
            to_id: Second bat ID
            interaction_strength: New interaction strength (0-1)
        """
        edge = self.get_edge(from_id, to_id)
        if edge is not None:
            # Exponential moving average of tie strength
            alpha = 0.3
            edge.social_tie_strength = (
                alpha * interaction_strength +
                (1 - alpha) * edge.social_tie_strength
            )
            logger.debug(
                f"Updated social tie {from_id}->{to_id}: "
                f"{edge.social_tie_strength:.2f}"
            )

    def mark_vocalization(
        self,
        bat_id: int,
        is_vocalizing: bool,
        arousal: float = 0.0,
    ) -> None:
        """
        Mark a bat as vocalizing or not.

        Args:
            bat_id: Bat ID
            is_vocalizing: Whether bat is currently vocalizing
            arousal: Arousal level (0-1) from VAE
        """
        if bat_id in self.nodes:
            node = self.nodes[bat_id]
            node.is_vocalizing = is_vocalizing
            node.arousal_level = arousal

            import time

            if is_vocalizing and node.vocalization_start_ms is None:
                node.vocalization_start_ms = time.time() * 1000
            elif not is_vocalizing:
                node.vocalization_start_ms = None

    def get_colony_center(self) -> np.ndarray:
        """Compute the centroid of all bat positions."""
        if not self.nodes:
            return np.zeros(3)

        positions = np.stack([n.position for n in self.nodes.values()])
        return np.mean(positions, axis=0)

    def get_cluster_centers(
        self,
        n_clusters: int = 3,
    ) -> np.ndarray:
        """
        Find cluster centers using k-means.

        Args:
            n_clusters: Number of clusters to find

        Returns:
            Array of cluster centers (n_clusters, 3)
        """
        if len(self.nodes) < n_clusters:
            return np.zeros((n_clusters, 3))

        positions = np.stack([n.position for n in self.nodes.values()])

        # Simple k-means initialization
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(positions)

        return kmeans.cluster_centers_

    def get_statistics(self) -> Dict:
        """Get topology statistics."""
        if not self.nodes:
            return {
                "num_nodes": 0,
                "num_edges": 0,
                "mean_distance": 0,
                "max_distance": 0,
            }

        distances = [e.distance for e in self.edges.values()]

        return {
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "mean_distance": np.mean(distances) if distances else 0,
            "max_distance": np.max(distances) if distances else 0,
            "vocalizing_count": sum(
                1 for n in self.nodes.values() if n.is_vocalizing
            ),
        }


class EmitterSelection:
    """
    Selects optimal speaker emitter for targeted vocalization.

    Uses spatial information to choose the best emitter for
    reaching a specific bat location.
    """

    def __init__(
        self,
        emitter_positions: List[np.ndarray],
        max_range: float = 10.0,
    ):
        """
        Initialize emitter selection.

        Args:
            emitter_positions: List of (x, y, z) emitter positions
            max_range: Maximum effective range of emitters (meters)
        """
        self.emitter_positions = [
            np.asarray(pos, dtype=np.float32)
            for pos in emitter_positions
        ]
        self.max_range = max_range

    def select_emitter(
        self,
        target_position: np.ndarray,
        preferred_emitter: Optional[int] = None,
    ) -> Tuple[int, float]:
        """
        Select best emitter for target position.

        Args:
            target_position: Target (x, y, z) position
            preferred_emitter: Preferred emitter ID (if in range)

        Returns:
            (emitter_id, distance) tuple
        """
        target = np.asarray(target_position)

        # Check preferred emitter first
        if preferred_emitter is not None:
            if 0 <= preferred_emitter < len(self.emitter_positions):
                dist = np.linalg.norm(
                    self.emitter_positions[preferred_emitter] - target
                )
                if dist <= self.max_range:
                    return preferred_emitter, dist

        # Find nearest emitter
        distances = [
            np.linalg.norm(pos - target)
            for pos in self.emitter_positions
        ]

        nearest_id = int(np.argmin(distances))
        nearest_dist = distances[nearest_id]

        if nearest_dist <= self.max_range:
            return nearest_id, nearest_dist

        # All out of range, return nearest anyway
        return nearest_id, nearest_dist

    def select_emitters_for_broadcast(
        self,
        target_positions: List[np.ndarray],
    ) -> Dict[int, List[int]]:
        """
        Select emitters for multiple targets (broadcast mode).

        Args:
            target_positions: List of target positions

        Returns:
            Dictionary mapping emitter_id to list of target indices
        """
        emitter_targets = {i: [] for i in range(len(self.emitter_positions))}

        for idx, pos in enumerate(target_positions):
            emitter_id, _ = self.select_emitter(pos)
            emitter_targets[emitter_id].append(idx)

        return emitter_targets


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Topology Engine Demo")
    print("=" * 50)

    # Create topology engine
    topology = TopologyEngine()

    # Add some bat positions
    positions = [
        (1, 0, 2, np.array([0.0, 0.0, 1.0])),
        (2, 0, 2, np.array([1.0, 0.0, 1.0])),
        (3, 0, 2, np.array([2.0, 0.0, 1.0])),
        (4, 0, 2, np.array([0.5, 1.0, 1.5])),
    ]

    for bat_id, timestamp, _, pos in positions:
        topology.update_node(bat_id, pos, timestamp=timestamp)

    print(f"Nodes: {topology.get_statistics()['num_nodes']}")
    print(f"Edges: {topology.get_statistics()['num_edges']}")

    # Find nearest neighbors
    neighbors = topology.find_nearest_neighbors(1, k=3)
    print(f"Nearest to bat 1: {neighbors}")

    # Emitter selection
    emitter_positions = [
        np.array([0.0, 0.0, 3.0]),  # Emitter 0
        np.array([2.0, 0.0, 3.0]),  # Emitter 1
        np.array([1.0, 2.0, 3.0]),  # Emitter 2
    ]
    selector = EmitterSelection(emitter_positions)

    emitter_id, dist = selector.select_emitter(np.array([1.0, 0.0, 1.0]))
    print(f"Selected emitter {emitter_id} at distance {dist:.2f}m")
