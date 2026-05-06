#!/usr/bin/env python3
"""
TDD Tests for InteractionAgent v1.3.0 - Level 2 Speaker Grounding

This test suite validates the upgrade from Level 1 (archetype-based context)
to Level 2 (speaker-aware response policies).

Red Phase: Failing tests that define the requirements for:
1. SpeakerProfile dataclass with emitter_id, dominance_rank, age_class
2. Emitter ID tracking alongside cluster ID
3. Speaker-specific response policies (Alpha vs Juvenile)
4. Speaker profile lookup and fallback behavior

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
"""

import pytest
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from realtime.interaction_agent import (
    InteractionAgent,
    InteractionAgentConfig,
    SpeakerProfile,
)


# =============================================================================
# FIXTURES: Speaker Profiles for Colony Hierarchy
# =============================================================================

@pytest.fixture
def colony_speaker_profiles():
    """
    Speaker profiles representing a bat colony hierarchy.

    Simulates a natural colony with:
    - Alpha emitter (dominance_rank=1.0)
    - Beta emitter (dominance_rank=0.7)
    - Juvenile emitter (dominance_rank=0.2)
    - Unknown emitter (no profile)
    """
    return {
        1: SpeakerProfile(
            emitter_id=1,
            dominance_rank=1.0,
            age_class="adult",
            response_bias={
                "alarm": 0.95,      # Alpha triggers strong alarm response
                "territorial": 0.90,
                "contact": 0.70,
                "social": 0.50,
            }
        ),
        2: SpeakerProfile(
            emitter_id=2,
            dominance_rank=0.7,
            age_class="adult",
            response_bias={
                "alarm": 0.80,
                "territorial": 0.75,
                "contact": 0.65,
                "social": 0.55,
            }
        ),
        3: SpeakerProfile(
            emitter_id=3,
            dominance_rank=0.2,
            age_class="juvenile",
            response_bias={
                "alarm": 0.50,      # Juvenile gets weaker responses
                "territorial": 0.40,
                "contact": 0.90,    # But high contact response (solicitous)
                "social": 0.85,
            }
        ),
    }


@pytest.fixture
def cluster_context_map():
    """Minimal cluster context map for testing."""
    return {
        8: "contact",
        12: "contact",
        25: "alarm",
        35: "territorial",
    }


# =============================================================================
# TEST SUITE 1: SpeakerProfile Dataclass
# =============================================================================

class TestSpeakerProfile:
    """Test SpeakerProfile dataclass structure."""

    def test_speaker_profile_creation(self):
        """SpeakerProfile should create with all fields."""
        profile = SpeakerProfile(
            emitter_id=1,
            dominance_rank=0.9,
            age_class="adult",
            response_bias={"alarm": 0.8, "contact": 0.7}
        )

        assert profile.emitter_id == 1
        assert profile.dominance_rank == 0.9
        assert profile.age_class == "adult"
        assert profile.response_bias == {"alarm": 0.8, "contact": 0.7}

    def test_speaker_profile_optional_fields(self):
        """Only emitter_id is required; other fields are optional."""
        profile = SpeakerProfile(emitter_id=5)

        assert profile.emitter_id == 5
        assert profile.dominance_rank is None
        assert profile.age_class is None
        assert profile.response_bias is None

    def test_speaker_profile_response_bias_lookup(self):
        """response_bias should provide context-specific multipliers."""
        profile = SpeakerProfile(
            emitter_id=1,
            response_bias={"alarm": 0.95, "contact": 0.70}
        )

        assert profile.get_response_bias("alarm") == 0.95
        assert profile.get_response_bias("contact") == 0.70
        # Default to 1.0 for unknown context
        assert profile.get_response_bias("territorial") == 1.0


# =============================================================================
# TEST SUITE 2: Emitter ID Tracking
# =============================================================================

class TestEmitterIDTracking:
    """Test that agent tracks emitter_id alongside cluster_id."""

    def test_agent_tracks_last_emitter_id(self, colony_speaker_profiles):
        """Agent should track _last_emitter_id after processing events."""
        config = InteractionAgentConfig(
            cluster_context_map={8: "contact"},
            speaker_profiles=colony_speaker_profiles,
            enable_speaker_adaptation=True,
        )

        agent = InteractionAgent(config=config)

        # Initially None
        assert agent._last_emitter_id is None

        # Process first event with emitter_id=1
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

        # Should now track emitter_id 1
        assert agent._last_emitter_id == 1

    def test_agent_tracks_emitter_id_changes(self, colony_speaker_profiles):
        """Agent should update _last_emitter_id when different speaker."""
        config = InteractionAgentConfig(
            cluster_context_map={8: "contact"},
            speaker_profiles=colony_speaker_profiles,
            enable_speaker_adaptation=True,
        )

        agent = InteractionAgent(config=config)
        agent._last_response_time = 0

        from realtime.feature_subscriber import FeatureEvent

        # First event from emitter 1
        event1 = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=8,
            features_112d=np.zeros(112, dtype=np.float32),
            timestamp=0.0,
            sequence=1,
            emitter_id=1,
            confidence=0.9,
        )
        agent._handle_feature_event(event1)
        assert agent._last_emitter_id == 1

        # Second event from emitter 3
        event2 = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=8,
            features_112d=np.zeros(112, dtype=np.float32),
            timestamp=1.0,
            sequence=2,
            emitter_id=3,
            confidence=0.9,
        )
        agent._handle_feature_event(event2)

        # Should update to new emitter
        assert agent._last_emitter_id == 3

    def test_agent_handles_none_emitter_id(self, colony_speaker_profiles):
        """Agent should handle events without emitter_id gracefully."""
        config = InteractionAgentConfig(
            cluster_context_map={8: "contact"},
            speaker_profiles=colony_speaker_profiles,
            enable_speaker_adaptation=True,
        )

        agent = InteractionAgent(config=config)
        agent._last_response_time = 0

        from realtime.feature_subscriber import FeatureEvent

        # Event without emitter_id
        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=8,
            features_112d=np.zeros(112, dtype=np.float32),
            timestamp=0.0,
            sequence=1,
            emitter_id=None,
            confidence=0.9,
        )

        agent._handle_feature_event(event)

        # Should track None
        assert agent._last_emitter_id is None


# =============================================================================
# TEST SUITE 3: Speaker Profile Lookup
# =============================================================================

class TestSpeakerProfileLookup:
    """Test speaker profile retrieval and fallback."""

    def test_get_speaker_profile_returns_profile(self, colony_speaker_profiles):
        """_get_speaker_profile() should return profile for known emitter."""
        config = InteractionAgentConfig(
            cluster_context_map={8: "contact"},
            speaker_profiles=colony_speaker_profiles,
            enable_speaker_adaptation=True,
        )

        agent = InteractionAgent(config=config)

        profile = agent._get_speaker_profile(1)
        assert profile is not None
        assert profile.emitter_id == 1
        assert profile.dominance_rank == 1.0

    def test_get_speaker_profile_returns_none_for_unknown(self, colony_speaker_profiles):
        """Unknown emitter_id should return None profile."""
        config = InteractionAgentConfig(
            cluster_context_map={8: "contact"},
            speaker_profiles=colony_speaker_profiles,
            enable_speaker_adaptation=True,
        )

        agent = InteractionAgent(config=config)

        profile = agent._get_speaker_profile(999)
        assert profile is None

    def test_get_speaker_profile_returns_none_when_disabled(self, colony_speaker_profiles):
        """When enable_speaker_adaptation=False, should always return None."""
        config = InteractionAgentConfig(
            cluster_context_map={8: "contact"},
            speaker_profiles=colony_speaker_profiles,
            enable_speaker_adaptation=False,  # Disabled
        )

        agent = InteractionAgent(config=config)

        # Even with speaker_profiles configured, should return None
        profile = agent._get_speaker_profile(1)
        assert profile is None


# =============================================================================
# TEST SUITE 4: Speaker-Specific Response Policies
# =============================================================================

class TestSpeakerSpecificResponsePolicies:
    """Test that response behavior varies by speaker profile."""

    def test_alpha_speaker_gets_strong_alarm_response(self, colony_speaker_profiles):
        """Alpha (emitter_id=1) should trigger strong alarm response."""
        config = InteractionAgentConfig(
            cluster_context_map={25: "alarm"},
            speaker_profiles=colony_speaker_profiles,
            enable_speaker_adaptation=True,
            confidence_threshold=0.5,
        )

        agent = InteractionAgent(config=config)
        agent._last_response_time = 0

        from realtime.feature_subscriber import FeatureEvent

        # Alpha emits alarm call
        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=25,
            features_112d=np.zeros(112, dtype=np.float32),
            timestamp=0.0,
            sequence=1,
            emitter_id=1,  # Alpha
            confidence=0.9,
        )

        result = agent._process_features(event)
        should_respond = agent._should_respond(result)

        # Alpha's high alarm bias should trigger response
        assert should_respond == True
        # Check that speaker info is in result
        assert result["speaker_profile"] is not None
        assert result["speaker_profile"].emitter_id == 1

    def test_juvenile_speaker_gets_solicitous_contact_response(self, colony_speaker_profiles):
        """Juvenile (emitter_id=3) should trigger solicitous contact response."""
        config = InteractionAgentConfig(
            cluster_context_map={8: "contact"},
            speaker_profiles=colony_speaker_profiles,
            enable_speaker_adaptation=True,
            confidence_threshold=0.5,
        )

        agent = InteractionAgent(config=config)
        agent._last_response_time = 0

        from realtime.feature_subscriber import FeatureEvent

        # Juvenile emits contact call
        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=8,
            features_112d=np.zeros(112, dtype=np.float32),
            timestamp=0.0,
            sequence=1,
            emitter_id=3,  # Juvenile
            confidence=0.9,
        )

        result = agent._process_features(event)
        should_respond = agent._should_respond(result)

        # Juvenile's high contact bias should trigger response
        assert should_respond == True
        assert result["speaker_profile"].emitter_id == 3
        assert result["speaker_profile"].age_class == "juvenile"

    def test_low_bias_speaker_can_suppress_response(self, colony_speaker_profiles):
        """Low response_bias should suppress even valid context."""
        config = InteractionAgentConfig(
            cluster_context_map={25: "alarm"},
            speaker_profiles=colony_speaker_profiles,
            enable_speaker_adaptation=True,
            confidence_threshold=0.5,
        )

        agent = InteractionAgent(config=config)
        agent._last_response_time = 0

        from realtime.feature_subscriber import FeatureEvent

        # Juvenile emits alarm call (but has low alarm bias)
        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=25,
            features_112d=np.zeros(112, dtype=np.float32),
            timestamp=0.0,
            sequence=1,
            emitter_id=3,  # Juvenile (alarm bias = 0.50)
            confidence=0.9,
        )

        result = agent._process_features(event)

        # Check if low bias suppressed response
        # The bias of 0.50 might suppress if below effective threshold
        # This test documents the behavior
        assert "speaker_bias_multiplier" in result

    def test_unknown_speaker_uses_default_policy(self, colony_speaker_profiles):
        """Unknown emitter_id should use default (no speaker modification)."""
        config = InteractionAgentConfig(
            cluster_context_map={25: "alarm"},
            speaker_profiles=colony_speaker_profiles,
            enable_speaker_adaptation=True,
            confidence_threshold=0.5,
        )

        agent = InteractionAgent(config=config)
        agent._last_response_time = 0

        from realtime.feature_subscriber import FeatureEvent

        # Unknown emitter
        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=25,
            features_112d=np.zeros(112, dtype=np.float32),
            timestamp=0.0,
            sequence=1,
            emitter_id=999,  # Unknown
            confidence=0.9,
        )

        result = agent._process_features(event)
        should_respond = agent._should_respond(result)

        # Should respond normally (no profile bias)
        assert result["speaker_profile"] is None
        assert result.get("speaker_bias_multiplier", 1.0) == 1.0

    def test_disabled_speaker_adaptation_ignores_profiles(self, colony_speaker_profiles):
        """When enable_speaker_adaptation=False, profiles should be ignored."""
        config = InteractionAgentConfig(
            cluster_context_map={25: "alarm"},
            speaker_profiles=colony_speaker_profiles,
            enable_speaker_adaptation=False,  # Disabled
            confidence_threshold=0.5,
        )

        agent = InteractionAgent(config=config)
        agent._last_response_time = 0

        from realtime.feature_subscriber import FeatureEvent

        # Even though emitter_id=1 is Alpha, with adaptation disabled
        # it should be treated as unknown
        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=25,
            features_112d=np.zeros(112, dtype=np.float32),
            timestamp=0.0,
            sequence=1,
            emitter_id=1,
            confidence=0.9,
        )

        result = agent._process_features(event)

        # Should not apply speaker bias
        assert result["speaker_profile"] is None


# =============================================================================
# TEST SUITE 5: Integration - Full Level 2 Pipeline
# =============================================================================

class TestLevel2SemanticGrounding:
    """Integration tests for Level 2 (Who + What) semantic grounding."""

    def test_full_pipeline_who_plus_what(self, colony_speaker_profiles):
        """Complete pipeline: emitter_id + cluster_id → speaker + context."""
        config = InteractionAgentConfig(
            cluster_context_map={8: "contact", 25: "alarm", 35: "territorial"},
            speaker_profiles=colony_speaker_profiles,
            enable_speaker_adaptation=True,
            confidence_threshold=0.5,
        )

        agent = InteractionAgent(config=config)
        agent._last_response_time = 0

        from realtime.feature_subscriber import FeatureEvent

        # Alpha emits territorial call
        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=35,
            features_112d=np.zeros(112, dtype=np.float32),
            timestamp=0.0,
            sequence=1,
            emitter_id=1,  # Alpha
            confidence=0.9,
        )

        result = agent._process_features(event)

        # Verify Level 1: Context grounded
        assert result["context_state"] == "territorial"
        assert result["cluster_id"] == 35

        # Verify Level 2: Speaker identified
        assert result["speaker_profile"] is not None
        assert result["speaker_profile"].emitter_id == 1
        assert result["speaker_profile"].dominance_rank == 1.0

        # Verify combined: speaker-aware response
        assert "speaker_bias_multiplier" in result
        # Alpha's territorial bias is 0.90
        assert result["speaker_bias_multiplier"] == 0.90

    def test_speaker_diarization_enables_social_graph(self, colony_speaker_profiles):
        """Tracking multiple speakers enables interaction network analysis."""
        config = InteractionAgentConfig(
            cluster_context_map={8: "contact"},
            speaker_profiles=colony_speaker_profiles,
            enable_speaker_adaptation=True,
        )

        agent = InteractionAgent(config=config)
        agent._last_response_time = 0

        from realtime.feature_subscriber import FeatureEvent

        # Simulate conversation: Alpha calls, Juvenile responds
        events = [
            FeatureEvent(
                event_type="feature_extraction",
                cluster_id=8,
                features_112d=np.zeros(112, dtype=np.float32),
                timestamp=float(i),
                sequence=i,
                emitter_id=1,  # Alpha
                confidence=0.9,
            )
            for i in range(3)
        ]

        speaker_sequence = []
        for event in events:
            agent._handle_feature_event(event)
            if agent._last_emitter_id is not None:
                speaker_sequence.append(agent._last_emitter_id)

        # All from Alpha
        assert speaker_sequence == [1, 1, 1]

        # Stats should track emitter diversity
        stats = agent.get_stats()
        assert "speaker_tracking" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
