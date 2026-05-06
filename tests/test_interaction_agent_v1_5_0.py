#!/usr/bin/env python3
"""
TDD Tests for InteractionAgent v1.5.0 - Ethological Validation Protocol

This test suite validates the field deployment validation features including:
1. Response Appropriateness Score (RAS) metric
2. Session metrics tracking
3. Interaction history management
4. Ethological mode configuration

Red Phase: Failing tests that define the requirements for:
1. InteractionEvent dataclass for tracking events
2. SessionMetrics dataclass for session tracking
3. RAS calculation from interaction sequences
4. Real-time RAS tracking in agent

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
"""

import pytest
import numpy as np
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from realtime.interaction_agent import (
    InteractionAgent,
    InteractionAgentConfig,
    InteractionEvent,
    SessionMetrics,
    calculate_ras,
)


# =============================================================================
# FIXTURES: Test Data
# =============================================================================

@pytest.fixture
def valid_bigrams():
    """The 50 valid bigrams from LRN-6 analysis (simplified subset)."""
    return {
        (8, 12), (8, 15), (8, 18), (8, 25),  # Cluster 8 openers
        (12, 8), (12, 20), (12, 25),
        (15, 8), (15, 12), (15, 22),
        (18, 8), (18, 15), (18, 30),
        (20, 8),
        (22, 8),
        (25, 8), (25, 12),
    }


@pytest.fixture
def cluster_context_map():
    """Minimal cluster context map."""
    return {
        8: "contact",
        12: "contact",
        15: "contact",
        18: "contact",
        20: "territorial",
        22: "territorial",
        25: "alarm",
        30: "alarm",
    }


# =============================================================================
# TEST SUITE 1: InteractionEvent Dataclass
# =============================================================================

class TestInteractionEvent:
    """Test InteractionEvent dataclass structure."""

    def test_interaction_event_creation(self):
        """InteractionEvent should store all event properties."""
        event = InteractionEvent(
            timestamp=1715011200.123,
            source="animal",
            cluster_id=8,
            emitter_id=1,
            response_to=12,
            time_since_previous=0.5,
        )

        assert event.timestamp == 1715011200.123
        assert event.source == "animal"
        assert event.cluster_id == 8
        assert event.emitter_id == 1
        assert event.response_to == 12
        assert event.time_since_previous == 0.5

    def test_system_event_has_no_emitter(self):
        """System events should have emitter_id=None."""
        event = InteractionEvent(
            timestamp=1715011200.450,
            source="system",
            cluster_id=12,
            emitter_id=None,
            response_to=8,
        )

        assert event.source == "system"
        assert event.emitter_id is None


# =============================================================================
# TEST SUITE 2: SessionMetrics Dataclass
# =============================================================================

class TestSessionMetrics:
    """Test SessionMetrics dataclass structure."""

    def test_session_metrics_creation(self):
        """SessionMetrics should store all session properties."""
        metrics = SessionMetrics(
            session_id="test_session_001",
            duration_seconds=120.0,
            condition="full_system",
            total_animal_vocalizations=10,
            total_system_responses=5,
            positive_responses=4,
            negative_responses=1,
            ras_score=0.8,
        )

        assert metrics.session_id == "test_session_001"
        assert metrics.duration_seconds == 120.0
        assert metrics.condition == "full_system"
        assert metrics.total_animal_vocalizations == 10
        assert metrics.total_system_responses == 5
        assert metrics.positive_responses == 4
        assert metrics.negative_responses == 1
        assert metrics.ras_score == 0.8

    def test_session_metrics_to_dict(self):
        """to_dict() should serialize for JSON export."""
        metrics = SessionMetrics(
            session_id="test_session",
            duration_seconds=60.0,
            condition="baseline",
        )

        d = metrics.to_dict()

        assert d["session_id"] == "test_session"
        assert d["duration_seconds"] == 60.0
        assert d["condition"] == "baseline"
        assert d["total_animal_vocalizations"] == 0


# =============================================================================
# TEST SUITE 3: RAS Calculation
# =============================================================================

class TestRASCalculation:
    """Test Response Appropriateness Score calculation."""

    def test_perfect_ras_score(self):
        """Perfect RAS = 1.0 when all system responses get valid follow-ups."""
        events = [
            InteractionEvent(timestamp=0.0, source="animal", cluster_id=8),
            InteractionEvent(timestamp=1.0, source="system", cluster_id=12, response_to=8),
            InteractionEvent(timestamp=2.0, source="animal", cluster_id=8, response_to=12),
            InteractionEvent(timestamp=3.0, source="system", cluster_id=12, response_to=8),
            InteractionEvent(timestamp=4.0, source="animal", cluster_id=8, response_to=12),
        ]

        valid_bigrams = {(8, 12), (12, 8)}
        ras = calculate_ras(events, valid_bigrams)

        assert ras == 1.0

    def test_zero_ras_score(self):
        """RAS = 0.0 when no system responses get follow-ups."""
        events = [
            InteractionEvent(timestamp=0.0, source="animal", cluster_id=8),
            InteractionEvent(timestamp=1.0, source="system", cluster_id=12, response_to=8),
            InteractionEvent(timestamp=3.0, source="system", cluster_id=15, response_to=12),
        ]

        valid_bigrams = {(8, 12), (12, 8), (12, 15)}
        ras = calculate_ras(events, valid_bigrams)

        assert ras == 0.0

    def test_partial_ras_score(self):
        """Partial RAS when some responses get follow-ups."""
        events = [
            InteractionEvent(timestamp=0.0, source="animal", cluster_id=8),
            InteractionEvent(timestamp=1.0, source="system", cluster_id=12, response_to=8),
            InteractionEvent(timestamp=2.0, source="animal", cluster_id=8, response_to=12),
            InteractionEvent(timestamp=3.0, source="system", cluster_id=15, response_to=8),
            # No follow-up to 15
        ]

        valid_bigrams = {(8, 12), (8, 15), (12, 8)}
        ras = calculate_ras(events, valid_bigrams)

        assert ras == 0.5

    def test_invalid_bigram_counts_as_negative(self):
        """Invalid bigrams should not count as positive responses."""
        events = [
            InteractionEvent(timestamp=0.0, source="animal", cluster_id=8),
            InteractionEvent(timestamp=1.0, source="system", cluster_id=12, response_to=8),
            InteractionEvent(timestamp=2.0, source="animal", cluster_id=99, response_to=12),  # Invalid
        ]

        valid_bigrams = {(8, 12), (12, 8)}  # (12, 99) is not valid
        ras = calculate_ras(events, valid_bigrams)

        assert ras == 0.0

    def test_ras_with_no_valid_bigrams(self):
        """When valid_bigrams is None, all animal follow-ups count as positive."""
        events = [
            InteractionEvent(timestamp=0.0, source="animal", cluster_id=8),
            InteractionEvent(timestamp=1.0, source="system", cluster_id=12, response_to=8),
            InteractionEvent(timestamp=2.0, source="animal", cluster_id=99, response_to=12),
        ]

        ras = calculate_ras(events, valid_bigrams=None)

        assert ras == 1.0  # No syntax checking, all follow-ups count

    def test_ras_with_empty_sequence(self):
        """Empty sequence should return RAS = 0.0."""
        ras = calculate_ras([], valid_bigrams={(8, 12)})
        assert ras == 0.0


# =============================================================================
# TEST SUITE 4: Agent Ethological Mode
# =============================================================================

class TestAgentEthologicalMode:
    """Test agent behavior in ethological validation mode."""

    def test_agent_initializes_session_metrics(self, valid_bigrams, cluster_context_map):
        """Agent should initialize SessionMetrics when ethological mode enabled."""
        config = InteractionAgentConfig(
            cluster_context_map=cluster_context_map,
            valid_bigrams=valid_bigrams,
            enable_ethological_mode=True,
            experimental_condition="full_system",
            session_id="test_session_001",
        )

        agent = InteractionAgent(config=config)
        agent._last_response_time = 0

        from realtime.feature_subscriber import FeatureEvent

        # Start agent to initialize session
        agent.start()

        assert agent._session_metrics is not None
        assert agent._session_metrics.session_id == "test_session_001"
        assert agent._session_metrics.condition == "full_system"
        assert agent._session_metrics.start_time is not None

        agent.stop()

    def test_agent_generates_session_id_if_not_provided(self, valid_bigrams, cluster_context_map):
        """Agent should auto-generate session_id if not provided."""
        config = InteractionAgentConfig(
            cluster_context_map=cluster_context_map,
            valid_bigrams=valid_bigrams,
            enable_ethological_mode=True,
            session_id=None,
        )

        agent = InteractionAgent(config=config)

        # Session ID should be generated during start
        assert agent.config.session_id is not None
        assert agent.config.session_id.startswith("session_")

    def test_agent_tracks_animal_events(self, valid_bigrams, cluster_context_map):
        """Agent should track animal events in interaction history."""
        config = InteractionAgentConfig(
            cluster_context_map=cluster_context_map,
            valid_bigrams=valid_bigrams,
            enable_ethological_mode=True,
            confidence_threshold=0.5,
        )

        agent = InteractionAgent(config=config)
        agent._last_response_time = 0

        from realtime.feature_subscriber import FeatureEvent

        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=8,
            features_112d=np.zeros(112, dtype=np.float32),
            timestamp=0.0,
            sequence=1,
            emitter_id=1,
            confidence=0.9,
        )

        agent._handle_feature_event(event)

        # Should have one animal event in history
        assert len(agent._interaction_history) == 1
        assert agent._interaction_history[0].source == "animal"
        assert agent._interaction_history[0].cluster_id == 8

    def test_agent_tracks_system_responses(self, valid_bigrams, cluster_context_map):
        """Agent should track system responses in interaction history."""
        config = InteractionAgentConfig(
            cluster_context_map=cluster_context_map,
            valid_bigrams=valid_bigrams,
            enable_ethological_mode=True,
            confidence_threshold=0.3,  # Low to ensure response
        )

        agent = InteractionAgent(config=config)
        agent._last_response_time = 0
        agent.start()  # Initialize session

        from realtime.feature_subscriber import FeatureEvent

        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=8,
            features_112d=np.zeros(112, dtype=np.float32),
            timestamp=time.time(),
            sequence=1,
            emitter_id=1,
            confidence=0.9,
        )

        agent._handle_feature_event(event)

        # Should have both animal and system events
        assert len(agent._interaction_history) >= 1

        # Find system event
        system_events = [e for e in agent._interaction_history if e.source == "system"]
        assert len(system_events) > 0

        agent.stop()


# =============================================================================
# TEST SUITE 5: RAS Integration
# =============================================================================

class TestRASIntegration:
    """Integration tests for RAS tracking."""

    def test_calculate_current_ras(self, valid_bigrams, cluster_context_map):
        """calculate_current_ras() should return current RAS score."""
        config = InteractionAgentConfig(
            cluster_context_map=cluster_context_map,
            valid_bigrams=valid_bigrams,
            enable_ethological_mode=True,
            confidence_threshold=0.3,
        )

        agent = InteractionAgent(config=config)
        agent._last_response_time = 0

        from realtime.feature_subscriber import FeatureEvent

        # Simulate interaction sequence
        # Animal 8 → System 12 → Animal 8 (valid bigram)
        events = [
            FeatureEvent(
                event_type="feature_extraction",
                cluster_id=8,
                features_112d=np.zeros(112, dtype=np.float32),
                timestamp=float(i),
                sequence=i,
                emitter_id=1,
                confidence=0.9,
            )
            for i in range(3)
        ]

        for event in events:
            agent._handle_feature_event(event)

        # Should have tracked events
        ras = agent.calculate_current_ras()
        assert 0.0 <= ras <= 1.0

    def test_get_session_metrics_returns_current_state(self, valid_bigrams, cluster_context_map):
        """get_session_metrics() should return up-to-date metrics."""
        config = InteractionAgentConfig(
            cluster_context_map=cluster_context_map,
            valid_bigrams=valid_bigrams,
            enable_ethological_mode=True,
            confidence_threshold=0.3,
        )

        agent = InteractionAgent(config=config)
        agent._last_response_time = 0
        agent.start()

        metrics = agent.get_session_metrics()

        assert metrics is not None
        assert metrics.session_id == agent.config.session_id
        assert metrics.condition == "full_system"
        assert metrics.duration_seconds >= 0.0

        agent.stop()

    def test_get_stats_includes_ethological_validation(self, valid_bigrams, cluster_context_map):
        """get_stats() should include ethological validation section."""
        config = InteractionAgentConfig(
            cluster_context_map=cluster_context_map,
            valid_bigrams=valid_bigrams,
            enable_ethological_mode=True,
        )

        agent = InteractionAgent(config=config)
        agent.start()

        stats = agent.get_stats()

        assert "ethological_validation" in stats
        assert stats["ethological_validation"]["enabled"] == True
        assert stats["ethological_validation"]["session_id"] is not None
        assert "ras_score" in stats["ethological_validation"]

        agent.stop()

    def test_interaction_history_bounded_size(self, valid_bigrams, cluster_context_map):
        """Interaction history should respect max_interaction_history limit."""
        config = InteractionAgentConfig(
            cluster_context_map=cluster_context_map,
            valid_bigrams=valid_bigrams,
            enable_ethological_mode=True,
            max_interaction_history=5,
        )

        agent = InteractionAgent(config=config)
        agent._last_response_time = 0

        from realtime.feature_subscriber import FeatureEvent

        # Add 10 events
        for i in range(10):
            event = FeatureEvent(
                event_type="feature_extraction",
                cluster_id=8,
                features_112d=np.zeros(112, dtype=np.float32),
                timestamp=float(i),
                sequence=i,
                emitter_id=1,
            )
            agent._handle_feature_event(event)

        # History should be bounded to 5
        assert len(agent._interaction_history) <= 5

    def test_ethological_mode_disabled_skips_tracking(self, cluster_context_map):
        """When ethological mode disabled, no tracking should occur."""
        config = InteractionAgentConfig(
            cluster_context_map=cluster_context_map,
            enable_ethological_mode=False,
        )

        agent = InteractionAgent(config=config)
        agent._last_response_time = 0

        from realtime.feature_subscriber import FeatureEvent

        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=8,
            features_112d=np.zeros(112, dtype=np.float32),
            timestamp=0.0,
            sequence=1,
            emitter_id=1,
        )

        agent._handle_feature_event(event)

        # Should not track
        assert len(agent._interaction_history) == 0
        assert agent.get_session_metrics() is None


# =============================================================================
# TEST SUITE 6: Experimental Conditions
# =============================================================================

class TestExperimentalConditions:
    """Test different experimental condition configurations."""

    def test_baseline_condition(self):
        """Baseline condition should not emit system responses."""
        config = InteractionAgentConfig(
            cluster_context_map={8: "contact"},
            valid_bigrams={(8, 12), (12, 8)},
            enable_ethological_mode=True,
            experimental_condition="baseline",
        )

        assert config.experimental_condition == "baseline"

    def test_conspecific_condition(self):
        """Conspecific condition uses pre-recorded bat vocalizations."""
        config = InteractionAgentConfig(
            cluster_context_map={8: "contact"},
            enable_ethological_mode=True,
            experimental_condition="conspecific",
        )

        assert config.experimental_condition == "conspecific"

    def test_full_system_condition(self):
        """Full system condition enables active interaction."""
        config = InteractionAgentConfig(
            cluster_context_map={8: "contact"},
            valid_bigrams={(8, 12), (12, 8)},
            enable_ethological_mode=True,
            experimental_condition="full_system",
        )

        assert config.experimental_condition == "full_system"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
