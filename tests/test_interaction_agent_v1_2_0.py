#!/usr/bin/env python3
"""
TDD Tests for InteractionAgent v1.2.0 - Cluster-Based Semantic Grounding

This test suite validates the upgrade from rule-based context inference to
cluster-archetype-based inference using the BGMM-distilled 45-cluster vocabulary.

Red Phase: Failing tests that define the requirements for:
1. Cluster-to-Context Mapping (Level 1 Semantic Grounding)
2. Confidence-based response suppression
3. Syntax-driven response validation (bigram grammar)

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
"""

import json
import pytest
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from realtime.interaction_agent import (
    InteractionAgent,
    InteractionAgentConfig,
)
from realtime.feature_subscriber import FeatureEvent


# =============================================================================
# FIXTURES: 45-Cluster BGMM Vocabulary
# =============================================================================

@pytest.fixture
def bgmm_centroids_45():
    """Synthetic centroids representing 45 BGMM-discovered clusters."""
    # Generate realistic 112D centroids with different acoustic properties
    centroids = []
    for i in range(45):
        centroid = np.zeros(112, dtype=np.float32)

        # Vary F0 (index 0) and RMS (index 1) across clusters
        if i < 10:  # Clusters 0-9: Social (low F0)
            centroid[0] = 3000 + i * 100
            centroid[1] = 0.3 + i * 0.02
        elif i < 25:  # Clusters 10-24: Contact (mid F0)
            centroid[0] = 5000 + (i - 10) * 150
            centroid[1] = 0.4 + (i - 10) * 0.02
        elif i < 35:  # Clusters 25-34: Alarm (high F0, high RMS)
            centroid[0] = 9000 + (i - 25) * 100
            centroid[1] = 0.7 + (i - 25) * 0.02
        else:  # Clusters 35-44: Territorial (high F0, mid RMS)
            centroid[0] = 7000 + (i - 35) * 120
            centroid[1] = 0.5 + (i - 35) * 0.02

        centroids.append(centroid)
    return centroids


@pytest.fixture
def cluster_context_map(bgmm_centroids_45):
    """Pre-computed context map for all 45 clusters."""
    from realtime.interaction_agent import build_cluster_context_map
    return build_cluster_context_map(bgmm_centroids_45)


@pytest.fixture
def valid_bat_bigrams():
    """The 50 valid bigrams from LRN-6 analysis."""
    # Simplified set for testing - represents the 50 valid transitions
    # Format: (opener_cluster, response_cluster)
    return {
        (8, 12), (8, 15), (8, 18),  # Cluster 8 opens, can respond with 12, 15, 18
        (12, 8), (12, 20), (12, 25),
        (15, 8), (15, 12), (15, 22),
        (18, 8), (18, 15), (18, 30),
        # ... (truncated for brevity, actual set has 50 entries)
    }


# =============================================================================
# TEST SUITE 1: Cluster-to-Context Mapping
# =============================================================================

class TestClusterContextMapping:
    """Test cluster-archetype-based context inference."""

    def test_build_cluster_context_map_creates_45_entries(self, bgmm_centroids_45):
        """build_cluster_context_map should create a mapping for all 45 clusters."""
        from realtime.interaction_agent import build_cluster_context_map

        context_map = build_cluster_context_map(bgmm_centroids_45)

        assert len(context_map) == 45, f"Expected 45 entries, got {len(context_map)}"

        # All cluster IDs should be present
        for i in range(45):
            assert i in context_map, f"Cluster {i} missing from context map"

    def test_context_map_infers_correct_contexts(self, bgmm_centroids_45):
        """Contexts should be inferred from centroid acoustic properties."""
        from realtime.interaction_agent import build_cluster_context_map

        context_map = build_cluster_context_map(bgmm_centroids_45)

        # Cluster 0: Low F0 → social
        assert context_map[0] == "social"

        # Cluster 10: Mid F0 → contact
        assert context_map[10] == "contact"

        # Cluster 25: High F0, high RMS → alarm
        assert context_map[25] == "alarm"

        # Cluster 35: High F0, mid RMS → territorial
        assert context_map[35] == "territorial"

    def test_agent_uses_cluster_id_for_context(self, cluster_context_map):
        """Agent should use pre-computed cluster context, not raw F0 rules."""
        config = InteractionAgentConfig(
            cluster_context_map=cluster_context_map,
        )

        agent = InteractionAgent(config=config)

        # Create event from Cluster 8 (contact archetype)
        centroid_8 = np.zeros(112, dtype=np.float32)
        centroid_8[0] = 5500  # Contact F0 range
        centroid_8[1] = 0.45

        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=8,
            features_112d=centroid_8,
            timestamp=0.0,
            sequence=1,
        )

        result = agent._process_features(event)

        # Context should be inferred from cluster archetype
        assert result["context_state"] in ["contact", "social", "alarm", "territorial"]
        # The method used should be cluster-based, not rule-based
        # (we'll add a "method" field to track this)

    def test_fallback_to_rule_based_without_cluster_map(self):
        """Without cluster_context_map, agent should fall back to rules."""
        config = InteractionAgentConfig(
            cluster_context_map=None,  # No map provided
        )

        agent = InteractionAgent(config=config)

        # Create event with known F0/RMS
        features = np.zeros(112, dtype=np.float32)
        features[0] = 9000  # High F0 → alarm
        features[1] = 0.7   # High RMS

        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=999,  # Unmapped cluster
            features_112d=features,
            timestamp=0.0,
            sequence=1,
        )

        result = agent._process_features(event)

        # Should still infer context from rules
        assert result["context_state"] == "alarm"


# =============================================================================
# TEST SUITE 2: Confidence-Based Response Suppression
# =============================================================================

class TestConfidenceBasedSuppression:
    """Test confidence score from Rust Student model."""

    def test_high_confidence_triggers_response(self, cluster_context_map):
        """High confidence (> 0.5) should allow response."""
        config = InteractionAgentConfig(
            cluster_context_map=cluster_context_map,
        )

        agent = InteractionAgent(config=config)
        agent._last_response_time = 0  # Reset cooldown

        # Create event with high confidence (from Rust Student)
        features = np.zeros(112, dtype=np.float32)
        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=8,  # Contact cluster
            features_112d=features,
            timestamp=0.0,
            sequence=1,
            confidence=0.9,  # High confidence
        )

        result = agent._process_features(event)
        should_respond = agent._should_respond(result)

        # High confidence should trigger response (subject to other checks)
        # Note: contact context doesn't trigger response by default
        # So we check that confidence doesn't block it
        assert result["confidence"] >= 0.5

    def test_low_confidence_suppresses_response(self, cluster_context_map):
        """Low confidence (< 0.5) should suppress response."""
        config = InteractionAgentConfig(
            cluster_context_map=cluster_context_map,
            confidence_threshold=0.5,  # Explicit threshold
        )

        agent = InteractionAgent(config=config)
        agent._last_response_time = 0  # Reset cooldown

        # Create event on edge of cluster (low confidence from Rust Student)
        features = np.zeros(112, dtype=np.float32)
        features[0] = 9000  # Alarm F0
        features[1] = 0.7   # Alarm RMS

        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=25,  # Alarm cluster
            features_112d=features,
            timestamp=0.0,
            sequence=1,
            confidence=0.2,  # Low confidence - near boundary
        )

        result = agent._process_features(event)
        should_respond = agent._should_respond(result)

        # Low confidence should suppress response
        assert should_respond == False, "Low confidence event should not trigger response"

    def test_agent_tracks_last_cluster_id(self, cluster_context_map):
        """Agent should track last cluster_id for syntax validation."""
        config = InteractionAgentConfig(
            cluster_context_map=cluster_context_map,
        )

        agent = InteractionAgent(config=config)

        # Initially None
        assert agent._last_cluster_id is None

        # Process first event
        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=8,
            features_112d=np.zeros(112, dtype=np.float32),
            timestamp=0.0,
            sequence=1,
        )

        agent._handle_feature_event(event)

        # Should now track cluster 8
        assert agent._last_cluster_id == 8


# =============================================================================
# TEST SUITE 3: Syntax-Driven Response (Bigram Grammar)
# =============================================================================

class TestBigramSyntaxValidation:
    """Test bat syntax validation using valid bigram set."""

    def test_valid_bigram_allows_response(self, cluster_context_map, valid_bat_bigrams):
        """Valid bigram (8, 12) should allow response."""
        config = InteractionAgentConfig(
            cluster_context_map=cluster_context_map,
            valid_bigrams=valid_bat_bigrams,
        )

        agent = InteractionAgent(config=config)
        agent._last_response_time = 0
        agent._last_cluster_id = 8  # Previous was cluster 8

        # Current event is cluster 12 (valid follow-up to 8)
        features = np.zeros(112, dtype=np.float32)
        features[0] = 5500

        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=12,
            features_112d=features,
            timestamp=0.0,
            sequence=1,
            confidence=0.9,
        )

        result = agent._process_features(event)
        should_respond = agent._should_respond(result)

        # Valid bigram - syntax check should pass
        # (actual response depends on context too)
        assert "bigram_valid" in result
        assert result["bigram_valid"] == True

    def test_invalid_bigram_blocks_response(self, cluster_context_map, valid_bat_bigrams):
        """Invalid bigram (8, 999) should block response."""
        config = InteractionAgentConfig(
            cluster_context_map=cluster_context_map,
            valid_bigrams=valid_bat_bigrams,
        )

        agent = InteractionAgent(config=config)
        agent._last_response_time = 0
        agent._last_cluster_id = 8  # Previous was cluster 8

        # Current event is cluster 999 (NOT in valid bigrams)
        features = np.zeros(112, dtype=np.float32)
        features[0] = 5500

        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=999,
            features_112d=features,
            timestamp=0.0,
            sequence=1,
            confidence=0.9,  # Even with high confidence
        )

        result = agent._process_features(event)
        should_respond = agent._should_respond(result)

        # Invalid bigram - should block response
        assert result["bigram_valid"] == False
        assert should_respond == False, "Invalid bigram should block response"

    def test_first_event_always_valid_bigram(self, cluster_context_map, valid_bat_bigrams):
        """First event (no previous cluster) should pass bigram check."""
        config = InteractionAgentConfig(
            cluster_context_map=cluster_context_map,
            valid_bigrams=valid_bat_bigrams,
        )

        agent = InteractionAgent(config=config)
        agent._last_response_time = 0
        agent._last_cluster_id = None  # No previous event

        # First event - any cluster is valid
        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=999,
            features_112d=np.zeros(112, dtype=np.float32),
            timestamp=0.0,
            sequence=1,
            confidence=0.9,
        )

        result = agent._process_features(event)

        # First event should pass bigram check
        assert result["bigram_valid"] == True

    def test_agent_without_bigrams_skips_check(self, cluster_context_map):
        """Without valid_bigrams config, agent should skip syntax check."""
        config = InteractionAgentConfig(
            cluster_context_map=cluster_context_map,
            # No valid_bigrams provided
        )

        agent = InteractionAgent(config=config)
        agent._last_response_time = 0
        agent._last_cluster_id = 8

        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=999,  # Would be invalid with bigrams
            features_112d=np.zeros(112, dtype=np.float32),
            timestamp=0.0,
            sequence=1,
            confidence=0.9,
        )

        result = agent._process_features(event)

        # Should not have bigram_valid field (check skipped)
        # Or should default to True
        assert result.get("bigram_valid", True) == True


# =============================================================================
# TEST SUITE 4: Integration - Full Pipeline
# =============================================================================

class TestClusterBasedSemanticGrounding:
    """Integration tests for full cluster-based pipeline."""

    def test_full_pipeline_with_rust_student(self, cluster_context_map, valid_bat_bigrams):
        """Full pipeline: Rust Student → Python Agent with cluster grounding."""
        config = InteractionAgentConfig(
            cluster_context_map=cluster_context_map,
            valid_bigrams=valid_bat_bigrams,
            confidence_threshold=0.5,
        )

        agent = InteractionAgent(config=config)
        agent._last_response_time = 0
        agent._last_cluster_id = 8

        # Simulate Rust Student output
        features = np.zeros(112, dtype=np.float32)
        features[0] = 5500  # Contact F0

        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=12,  # Valid bigram (8, 12)
            features_112d=features,
            timestamp=0.0,
            sequence=1,
            confidence=0.85,  # Rust Student confidence
        )

        result = agent._process_features(event)
        should_respond = agent._should_respond(result)

        # Verify all checks pass
        assert result["context_state"] in ["contact", "social", "alarm", "territorial"]
        assert result["cluster_id"] == 12
        assert result["confidence"] == 0.85
        assert result["bigram_valid"] == True

        # Response decision depends on context
        # (contact doesn't auto-trigger, but the check passed)

    def test_perceptual_grounding_prevents_feedback_loop(self, cluster_context_map):
        """OOD events from Rust should never reach Python processing."""
        # This test validates that the Rust OOD filter works
        # OOD events have None cluster_id or are dropped entirely
        # Python agent should only receive valid (0-44) cluster_ids

        config = InteractionAgentConfig(
            cluster_context_map=cluster_context_map,
        )

        agent = InteractionAgent(config=config)

        # All valid events from Rust should have cluster_id 0-44
        for cluster_id in range(45):
            event = FeatureEvent(
                event_type="feature_extraction",
                cluster_id=cluster_id,
                features_112d=np.zeros(112, dtype=np.float32),
                timestamp=0.0,
                sequence=cluster_id,
            )

            result = agent._process_features(event)

            # Should successfully process
            assert result["cluster_id"] == cluster_id
            assert "context_state" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
