#!/usr/bin/env python3
"""
Topology Engine (Level 2.5)

Maintains the real-time state of the animal colony's spatial configuration.
Calculates proximity maps, line-of-sight, and social network metrics.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from spatial_intelligence.spatial_ingestor import SpatialObservation, SpatialFrame

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """Extended state for an agent in the topology."""
    observation: SpatialObservation
    nearby_agents: Dict[str, float] = field(default_factory=dict)  # agent_id -> distance
    visible_agents: List[str] = field(default_factory=list)  # Agents in field of view
    last_updated_ns: int = 0


@dataclass
class ProximityResult:
    """Result of a proximity query."""
    agent_id: str
    nearby_agents: List[Tuple[str, float]]  # (agent_id, distance) sorted by distance
    nearest_agent: Optional[str]
    nearest_distance: float


@dataclass
class LineOfSightResult:
    """Result of a line-of-sight check."""
    emitter_id: str
    target_id: str
    has_los: bool
    angle_rad: float
    distance: float
    in_field_of_view: bool


class TopologyEngine:
    """
    Maintains the real-time spatial topology of the colony.

    Features:
    - Proximity maps (which agents are near each other)
    - Line-of-sight calculations (field of view checks)
    - Spatial queries (find agents within radius, etc.)
    - Colony state management
    """

    def __init__(
        self,
        max_agents: int = 100,
        proximity_radius: float = 5.0,  # meters
        field_of_view_deg: float = 120.0,  # degrees
    ):
        self.max_agents = max_agents
        self.proximity_radius = proximity_radius
        self.field_of_view_rad = np.deg2rad(field_of_view_deg)

        # Agent storage
        self.agent_states: Dict[str, AgentState] = {}

        # Timestamp of last topology update
        self.last_topology_update_ns: int = 0

        logger.info(
            f"TopologyEngine initialized: max_agents={max_agents}, "
            f"proximity_radius={proximity_radius}m, fov={field_of_view_deg}°"
        )

    def update_topology(self, frame: SpatialFrame) -> int:
        """
        Update topology from a new spatial frame.

        Args:
            frame: SpatialFrame with current observations

        Returns:
            Number of agents updated
        """
        updated_count = 0

        for obs in frame.observations:
            # Create or update agent state
            if obs.agent_id not in self.agent_states:
                self.agent_states[obs.agent_id] = AgentState(observation=obs)
            else:
                self.agent_states[obs.agent_id].observation = obs

            self.agent_states[obs.agent_id].last_updated_ns = obs.timestamp_ns
            updated_count += 1

        # Recalculate proximity relationships
        self._update_proximity_maps(frame.timestamp_ns)

        # Recalculate line-of-sight
        self._update_line_of_sight(frame.timestamp_ns)

        self.last_topology_update_ns = frame.timestamp_ns

        logger.debug(f"Topology updated: {updated_count} agents")

        return updated_count

    def _update_proximity_maps(self, timestamp_ns: int):
        """Update proximity maps for all agents."""
        # Get current positions
        positions = {}
        for agent_id, state in self.agent_states.items():
            positions[agent_id] = state.observation.to_array()

        if not positions:
            return

        # Calculate distance matrix
        agent_ids = list(positions.keys())
        pos_matrix = np.array([positions[aid] for aid in agent_ids])
        distances = cdist(pos_matrix, pos_matrix)

        # Update each agent's nearby list
        for i, agent_id in enumerate(agent_ids):
            nearby = {}
            visible = []

            for j, other_id in enumerate(agent_ids):
                if i == j:
                    continue

                dist = distances[i, j]

                # Add to nearby if within radius
                if dist <= self.proximity_radius:
                    nearby[other_id] = dist

                # Check for line-of-sight
                other_state = self.agent_states.get(other_id)
                if other_state:
                    los_result = self.check_line_of_sight(agent_id, other_id)
                    if los_result.in_field_of_view:
                        visible.append(other_id)

            self.agent_states[agent_id].nearby_agents = nearby
            self.agent_states[agent_id].visible_agents = visible

    def _update_line_of_sight(self, timestamp_ns: int):
        """Update line-of-sight relationships."""
        # This is handled in _update_proximity_maps via check_line_of_sight
        pass

    def get_proximity_map(
        self,
        agent_id: str,
        max_radius: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Get agents within a specific radius of the given agent.

        Args:
            agent_id: Agent to query around
            max_radius: Maximum radius (defaults to self.proximity_radius)

        Returns:
            Dictionary of {agent_id: distance} for nearby agents
        """
        if agent_id not in self.agent_states:
            return {}

        radius = max_radius or self.proximity_radius
        state = self.agent_states[agent_id]

        # Filter by radius if different from default
        if radius == self.proximity_radius:
            return state.nearby_agents.copy()
        else:
            return {
                aid: dist
                for aid, dist in state.nearby_agents.items()
                if dist <= radius
            }

    def get_proximity_result(self, agent_id: str) -> Optional[ProximityResult]:
        """
        Get detailed proximity result for an agent.

        Returns:
            ProximityResult with sorted nearby agents and nearest agent info
        """
        if agent_id not in self.agent_states:
            return None

        nearby = self.get_proximity_map(agent_id)

        # Sort by distance
        sorted_nearby = sorted(nearby.items(), key=lambda x: x[1])

        if sorted_nearby:
            nearest_agent, nearest_distance = sorted_nearby[0]
        else:
            nearest_agent = None
            nearest_distance = float('inf')

        return ProximityResult(
            agent_id=agent_id,
            nearby_agents=sorted_nearby,
            nearest_agent=nearest_agent,
            nearest_distance=nearest_distance,
        )

    def check_line_of_sight(self, emitter_id: str, target_id: str) -> LineOfSightResult:
        """
        Calculate if target is within emitter's field of view.

        Args:
            emitter_id: Agent doing the "looking"
            target_id: Agent being looked at

        Returns:
            LineOfSightResult with detailed information
        """
        if emitter_id not in self.agent_states or target_id not in self.agent_states:
            return LineOfSightResult(
                emitter_id=emitter_id,
                target_id=target_id,
                has_los=False,
                angle_rad=np.pi,
                distance=float('inf'),
                in_field_of_view=False,
            )

        emitter = self.agent_states[emitter_id].observation
        target = self.agent_states[target_id].observation

        # Calculate distance
        distance = emitter.distance_to(target)

        # Calculate angle from emitter's heading to target
        angle = emitter.angle_to(target)

        # Check if within field of view
        in_fov = angle <= (self.field_of_view_rad / 2.0)

        # Has line-of-sight if in field of view and within proximity radius
        has_los = in_fov and (distance <= self.proximity_radius)

        return LineOfSightResult(
            emitter_id=emitter_id,
            target_id=target_id,
            has_los=has_los,
            angle_rad=angle,
            distance=distance,
            in_field_of_view=in_fov,
        )

    def get_nearby_agents(
        self,
        agent_id: str,
        max_radius: Optional[float] = None,
        require_los: bool = False,
    ) -> List[str]:
        """
        Get list of nearby agents, optionally filtered by line-of-sight.

        Args:
            agent_id: Agent to query
            max_radius: Maximum radius (default: self.proximity_radius)
            require_los: Only return agents with line-of-sight

        Returns:
            List of agent IDs
        """
        if agent_id not in self.agent_states:
            return []

        nearby = self.get_proximity_map(agent_id, max_radius)

        if require_los:
            visible = set(self.agent_states[agent_id].visible_agents)
            return [aid for aid in nearby.keys() if aid in visible]
        else:
            return list(nearby.keys())

    def get_agent_position(self, agent_id: str) -> Optional[np.ndarray]:
        """Get current position of an agent."""
        if agent_id not in self.agent_states:
            return None
        return self.agent_states[agent_id].observation.to_array()

    def get_agent_state(self, agent_id: str) -> Optional[AgentState]:
        """Get full state for an agent."""
        return self.agent_states.get(agent_id)

    def get_all_agent_ids(self) -> List[str]:
        """Get all known agent IDs."""
        return list(self.agent_states.keys())

    def get_colony_center(self) -> np.ndarray:
        """
        Calculate the geometric center of the colony.

        Returns:
            Position array [x, y, z]
        """
        positions = []
        for state in self.agent_states.values():
            positions.append(state.observation.to_array())

        if not positions:
            return np.zeros(3)

        return np.mean(positions, axis=0)

    def get_colony_spread(self) -> float:
        """
        Calculate the spread (standard deviation of distances from center).

        Returns:
            Spread in meters
        """
        if len(self.agent_states) < 2:
            return 0.0

        center = self.get_colony_center()
        positions = np.array([s.observation.to_array() for s in self.agent_states.values()])

        distances = np.linalg.norm(positions - center, axis=1)
        return float(np.std(distances))

    def remove_stale_agents(self, current_timestamp_ns: int, max_age_ms: float = 1000.0) -> int:
        """
        Remove agents that haven't been updated recently.

        Args:
            current_timestamp_ns: Current time
            max_age_ms: Maximum age in milliseconds

        Returns:
            Number of agents removed
        """
        cutoff_time = current_timestamp_ns - int(max_age_ms * 1_000_000)
        to_remove = []

        for agent_id, state in self.agent_states.items():
            if state.last_updated_ns < cutoff_time:
                to_remove.append(agent_id)

        for agent_id in to_remove:
            del self.agent_states[agent_id]

        if to_remove:
            logger.info(f"Removed {len(to_remove)} stale agents from topology")

        return len(to_remove)

    def get_topology_summary(self) -> Dict:
        """Get summary statistics of current topology."""
        if not self.agent_states:
            return {
                "num_agents": 0,
                "colony_center": [0.0, 0.0, 0.0],
                "colony_spread": 0.0,
                "avg_nearby_count": 0.0,
            }

        nearby_counts = [len(s.nearby_agents) for s in self.agent_states.values()]

        return {
            "num_agents": len(self.agent_states),
            "colony_center": self.get_colony_center().tolist(),
            "colony_spread": self.get_colony_spread(),
            "avg_nearby_count": np.mean(nearby_counts) if nearby_counts else 0.0,
            "last_update_ns": self.last_topology_update_ns,
        }


class ColonyTopology:
    """
    High-level interface for colony-wide spatial queries.

    Provides convenience methods for common queries about the
    spatial configuration of the entire colony.
    """

    def __init__(self, topology_engine: TopologyEngine):
        self.topology = topology_engine

    def find_clusters(
        self,
        cluster_distance: float = 2.0,
        min_cluster_size: int = 2,
    ) -> List[List[str]]:
        """
        Find spatial clusters of agents.

        Uses simple distance-based clustering.

        Args:
            cluster_distance: Maximum distance to be in same cluster
            min_cluster_size: Minimum agents to form a cluster

        Returns:
            List of clusters, where each cluster is a list of agent IDs
        """
        agent_ids = self.topology.get_all_agent_ids()

        if len(agent_ids) < min_cluster_size:
            return []

        # Build adjacency list based on distance
        adjacency = {aid: [] for aid in agent_ids}

        for i, aid1 in enumerate(agent_ids):
            for aid2 in agent_ids[i+1:]:
                pos1 = self.topology.get_agent_position(aid1)
                pos2 = self.topology.get_agent_position(aid2)

                if pos1 is not None and pos2 is not None:
                    dist = np.linalg.norm(pos1 - pos2)
                    if dist <= cluster_distance:
                        adjacency[aid1].append(aid2)
                        adjacency[aid2].append(aid1)

        # Find connected components (clusters)
        visited = set()
        clusters = []

        for aid in agent_ids:
            if aid not in visited:
                cluster = []
                stack = [aid]

                while stack:
                    current = stack.pop()
                    if current in visited:
                        continue

                    visited.add(current)
                    cluster.append(current)

                    for neighbor in adjacency[current]:
                        if neighbor not in visited:
                            stack.append(neighbor)

                if len(cluster) >= min_cluster_size:
                    clusters.append(cluster)

        return clusters

    def find_isolated_agents(self, max_distance: float = 3.0) -> List[str]:
        """
        Find agents that are far from any other agent.

        Args:
            max_distance: Maximum distance to still be considered "social"

        Returns:
            List of isolated agent IDs
        """
        isolated = []

        for agent_id in self.topology.get_all_agent_ids():
            nearby = self.topology.get_nearby_agents(agent_id, max_radius=max_distance)
            if not nearby:
                isolated.append(agent_id)

        return isolated

    def get_social_graph(self) -> Dict[str, Dict[str, float]]:
        """
        Build a social graph based on spatial proximity.

        Edge weights are inverse distances (closer = stronger connection).

        Returns:
            Nested dict: {agent_id: {other_id: weight}}
        """
        graph = {}

        for agent_id in self.topology.get_all_agent_ids():
            nearby = self.topology.get_proximity_map(agent_id)
            # Convert distance to weight (inverse distance)
            weights = {
                other_id: 1.0 / (dist + 0.1)  # +0.1 to avoid division by zero
                for other_id, dist in nearby.items()
            }
            graph[agent_id] = weights

        return graph


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from spatial_intelligence.spatial_ingestor import SimulatedIngestor

    # Create simulated ingestor and topology engine
    ingestor = SimulatedIngestor(num_agents=10, area_size=10.0)
    topology = TopologyEngine(max_agents=10, proximity_radius=5.0)

    # Generate a few frames and update topology
    for i in range(3):
        timestamp_ns = i * 33_000_000
        frame = ingestor.generate_frame(timestamp_ns)
        topology.update_topology(frame)

        print(f"\n=== Frame {i} ===")
        print(f"Topology: {topology.get_topology_summary()}")

        # Show proximity for first agent
        if frame.observations:
            agent_id = frame.observations[0].agent_id
            prox = topology.get_proximity_result(agent_id)
            if prox and prox.nearby_agents:
                print(f"\n{agent_id} nearby agents:")
                for other_id, dist in prox.nearby_agents[:3]:
                    los = topology.check_line_of_sight(agent_id, other_id)
                    los_str = "✓" if los.in_field_of_view else "✗"
                    print(f"  {other_id}: {dist:.2f}m (LoS: {los_str})")

    # Find clusters
    colony = ColonyTopology(topology)
    clusters = colony.find_clusters(cluster_distance=3.0)
    print(f"\nFound {len(clusters)} spatial clusters")
    for i, cluster in enumerate(clusters):
        print(f"  Cluster {i}: {cluster}")
