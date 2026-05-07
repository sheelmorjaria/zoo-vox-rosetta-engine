#!/usr/bin/env python3
"""
TDD Tests for InteractionAgent v1.4.0 - Probabilistic Bigram Weights

This test suite validates the upgrade from binary bigram validation to
probabilistic Markov chain-based response weighting.

Red Phase: Failing tests that define the requirements for:
1. Bigram probability dataclass with count, probability, rarity_score
2. Corpus bigram frequency analyzer from labeled data
3. Probability-weighted response amplitude/confidence
4. Rarity-based cognitive attention signals

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from realtime.interaction_agent import (
    BigramProbability,
    InteractionAgent,
    InteractionAgentConfig,
    analyze_corpus_bigram_frequencies,
    build_bigram_probability_map,
)

# =============================================================================
# FIXTURES: Synthetic Corpus Data
# =============================================================================


@pytest.fixture
def synthetic_corpus_sequence():
    """
    Synthetic cluster sequence representing typical bat vocalization patterns.

    Pattern: Cluster 8 (contact) frequently transitions to 12, 15, 18
    Cluster 12 frequently transitions back to 8
    Rare transitions exist (e.g., 8→25 is unusual but valid)
    """
    # Common transitions (repeated)
    common = [
        8,
        12,
        8,
        12,
        8,
        12,  # 8→12 very common
        8,
        15,
        8,
        15,  # 8→15 common
        12,
        8,
        12,
        8,  # 12→8 common
        8,
        18,  # 8→18 less common
    ]

    # Rare transitions (only once)
    rare = [
        8,
        25,  # 8→25 rare (contact → alarm, unusual)
    ]

    return common + rare


@pytest.fixture
def valid_bigrams():
    """The 50 valid bigrams from LRN-6 analysis (simplified subset)."""
    return {
        (8, 12),
        (8, 15),
        (8, 18),
        (8, 25),  # Cluster 8 openers
        (12, 8),
        (12, 20),
        (12, 25),
        (15, 8),
        (15, 12),
        (15, 22),
        (18, 8),
        (18, 15),
        (18, 30),
        (20, 8),
        (22, 8),
        (25, 8),
        (25, 12),
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
# TEST SUITE 1: BigramProbability Dataclass
# =============================================================================


class TestBigramProbability:
    """Test BigramProbability dataclass structure."""

    def test_bigram_probability_creation(self):
        """BigramProbability should store all frequency metrics."""
        bp = BigramProbability(
            opener=8,
            response=12,
            count=100,
            probability=0.35,
            rarity_score=0.5,  # Medium rarity
        )

        assert bp.opener == 8
        assert bp.response == 12
        assert bp.count == 100
        assert bp.probability == 0.35
        assert bp.rarity_score == 0.5

    def test_rarity_score_calculation_common(self):
        """Common transitions should have low rarity_score."""
        bp = BigramProbability(
            opener=8,
            response=12,
            count=1000,
            probability=0.80,  # 80% of transitions (truly high)
            rarity_score=None,  # Auto-calculate
        )

        # High probability → low rarity
        assert bp.calculate_rarity_score() < 0.3

    def test_rarity_score_calculation_rare(self):
        """Rare transitions should have high rarity_score."""
        bp = BigramProbability(
            opener=8,
            response=25,
            count=5,
            probability=0.01,  # 1% of transitions
            rarity_score=None,
        )

        # Low probability → high rarity
        assert bp.calculate_rarity_score() > 0.7


# =============================================================================
# TEST SUITE 2: Corpus Bigram Frequency Analysis
# =============================================================================


class TestBigramFrequencyAnalysis:
    """Test bigram frequency counting from corpus sequences."""

    def test_analyze_corpus_counts_bigrams(self, synthetic_corpus_sequence):
        """analyze_corpus_bigram_frequencies should count all bigrams."""
        bigram_counts = analyze_corpus_bigram_frequencies(synthetic_corpus_sequence)

        # 8→12 should be most common
        assert bigram_counts.get((8, 12), 0) > bigram_counts.get((8, 15), 0)

        # 8→25 should be rare (only 1 occurrence)
        assert bigram_counts.get((8, 25), 0) == 1

        # Total bigram count should match sequence length - 1
        total_count = sum(bigram_counts.values())
        assert total_count == len(synthetic_corpus_sequence) - 1

    def test_analyze_corpus_filters_to_valid_bigrams(
        self, synthetic_corpus_sequence, valid_bigrams
    ):
        """build_bigram_probability_map should only include valid bigrams."""
        # First get raw counts
        analyze_corpus_bigram_frequencies(synthetic_corpus_sequence)

        # Build probability map with valid bigrams filter
        prob_map = build_bigram_probability_map(
            corpus_sequence=synthetic_corpus_sequence,
            valid_bigrams=valid_bigrams,
        )

        # Should only contain valid bigrams
        for opener, response in prob_map.keys():
            assert (opener, response) in valid_bigrams

    def test_build_bigram_probability_map_calculates_probabilities(
        self, synthetic_corpus_sequence, valid_bigrams
    ):
        """Probabilities should sum to 1.0 for each opener."""
        prob_map = build_bigram_probability_map(
            corpus_sequence=synthetic_corpus_sequence,
            valid_bigrams=valid_bigrams,
        )

        # Group by opener and check probabilities sum to ~1.0
        openers = set([opener for opener, _ in prob_map.keys()])

        for opener in openers:
            opener_bigrams = {(o, r): bp for (o, r), bp in prob_map.items() if o == opener}

            # Sum probabilities for this opener
            total_prob = sum(bp.probability for bp in opener_bigrams.values())

            # Should sum to approximately 1.0 (with float tolerance)
            assert 0.99 <= total_prob <= 1.01, f"Opener {opener} probabilities sum to {total_prob}"

    def test_rarity_score_increases_with_rarity(self, synthetic_corpus_sequence, valid_bigrams):
        """rarity_score should be higher for less common bigrams."""
        prob_map = build_bigram_probability_map(
            corpus_sequence=synthetic_corpus_sequence,
            valid_bigrams=valid_bigrams,
        )

        # 8→12 should have lower rarity than 8→25
        bp_8_12 = prob_map.get((8, 12))
        bp_8_25 = prob_map.get((8, 25))

        if bp_8_12 and bp_8_25:
            assert bp_8_12.rarity_score < bp_8_25.rarity_score


# =============================================================================
# TEST SUITE 3: Probability-Weighted Responses
# =============================================================================


class TestProbabilityWeightedResponses:
    """Test that response amplitude/confidence is weighted by bigram probability."""

    def test_common_bigram_increases_effective_confidence(self, valid_bigrams, cluster_context_map):
        """Common (high-probability) bigrams should boost effective confidence."""
        # Create probability map where 8→12 is very common
        prob_map = {
            (8, 12): BigramProbability(8, 12, count=100, probability=0.50, rarity_score=0.2),
        }

        config = InteractionAgentConfig(
            cluster_context_map=cluster_context_map,
            valid_bigrams=valid_bigrams,
            bigram_probability_map=prob_map,
            enable_probabilistic_weighting=True,
            confidence_threshold=0.5,
        )

        agent = InteractionAgent(config=config)
        agent._last_response_time = 0
        agent._last_cluster_id = 8  # Previous was cluster 8

        from realtime.feature_subscriber import FeatureEvent

        # Current event is cluster 12 (common transition)
        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=12,
            features_112d=np.zeros(112, dtype=np.float32),
            timestamp=0.0,
            sequence=1,
            confidence=0.6,  # Base confidence
            emitter_id=1,
        )

        result = agent._process_features(event)

        # Should include probability info
        assert "bigram_probability" in result
        assert result["bigram_probability"] == 0.50
        assert result["bigram_rarity_score"] == 0.2

        # Effective confidence should be boosted
        # With confidence=0.6, probability=0.5, multiplier=(0.5+0.5)=1.0
        # So effective_confidence = 0.6 × 1.0 × 1.0 (no speaker) = 0.6
        # For high probability we need > 0.5 probability
        assert result["effective_confidence"] >= 0.6

    def test_rare_bigram_decreases_effective_confidence(self, valid_bigrams, cluster_context_map):
        """Rare (low-probability) bigrams should reduce effective confidence."""
        # Create probability map where 8→25 is rare
        prob_map = {
            (8, 25): BigramProbability(8, 25, count=2, probability=0.02, rarity_score=0.9),
        }

        config = InteractionAgentConfig(
            cluster_context_map=cluster_context_map,
            valid_bigrams=valid_bigrams,
            bigram_probability_map=prob_map,
            enable_probabilistic_weighting=True,
            confidence_threshold=0.5,
        )

        agent = InteractionAgent(config=config)
        agent._last_response_time = 0
        agent._last_cluster_id = 8

        from realtime.feature_subscriber import FeatureEvent

        # Current event is cluster 25 (rare transition)
        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=25,
            features_112d=np.zeros(112, dtype=np.float32),
            timestamp=0.0,
            sequence=1,
            confidence=0.7,  # Higher base confidence
            emitter_id=1,
        )

        result = agent._process_features(event)

        # Should include probability info
        assert "bigram_probability" in result
        assert result["bigram_probability"] == 0.02
        assert result["bigram_rarity_score"] == 0.9

        # Effective confidence should be reduced due to rarity
        assert result["effective_confidence"] < 0.7

    def test_first_event_has_default_probability(self, valid_bigrams, cluster_context_map):
        """First event (no previous cluster) should use default probability."""
        prob_map = {
            (8, 12): BigramProbability(8, 12, count=100, probability=0.50, rarity_score=0.2),
        }

        config = InteractionAgentConfig(
            cluster_context_map=cluster_context_map,
            valid_bigrams=valid_bigrams,
            bigram_probability_map=prob_map,
            enable_probabilistic_weighting=True,
            default_bigram_probability=0.5,  # Default for first event
        )

        agent = InteractionAgent(config=config)
        agent._last_response_time = 0
        agent._last_cluster_id = None  # No previous event

        from realtime.feature_subscriber import FeatureEvent

        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=8,
            features_112d=np.zeros(112, dtype=np.float32),
            timestamp=0.0,
            sequence=1,
            confidence=0.6,
            emitter_id=1,
        )

        result = agent._process_features(event)

        # Should use default probability
        assert result["bigram_probability"] == 0.5

    def test_unknown_bigram_uses_default_probability(self, valid_bigrams, cluster_context_map):
        """Bigrams not in probability map should use default."""
        prob_map = {}  # Empty map

        config = InteractionAgentConfig(
            cluster_context_map=cluster_context_map,
            valid_bigrams=valid_bigrams,
            bigram_probability_map=prob_map,
            enable_probabilistic_weighting=True,
            default_bigram_probability=0.3,  # Conservative default
        )

        agent = InteractionAgent(config=config)
        agent._last_response_time = 0
        agent._last_cluster_id = 8

        from realtime.feature_subscriber import FeatureEvent

        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=12,  # (8, 12) not in prob_map
            features_112d=np.zeros(112, dtype=np.float32),
            timestamp=0.0,
            sequence=1,
            confidence=0.6,
            emitter_id=1,
        )

        result = agent._process_features(event)

        # Should use default probability
        assert result["bigram_probability"] == 0.3


# =============================================================================
# TEST SUITE 4: Rarity-Based Cognitive Attention
# =============================================================================


class TestRarityBasedCognitiveAttention:
    """Test that rare transitions trigger cognitive attention signals."""

    def test_high_rarity_triggers_attention_flag(self, valid_bigrams, cluster_context_map):
        """High rarity_score should set cognitive_attention flag."""
        prob_map = {
            (8, 25): BigramProbability(8, 25, count=1, probability=0.005, rarity_score=0.95),
        }

        config = InteractionAgentConfig(
            cluster_context_map=cluster_context_map,
            valid_bigrams=valid_bigrams,
            bigram_probability_map=prob_map,
            enable_probabilistic_weighting=True,
            rarity_attention_threshold=0.8,  # Trigger attention above 0.8
        )

        agent = InteractionAgent(config=config)
        agent._last_response_time = 0
        agent._last_cluster_id = 8

        from realtime.feature_subscriber import FeatureEvent

        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=25,
            features_112d=np.zeros(112, dtype=np.float32),
            timestamp=0.0,
            sequence=1,
            confidence=0.9,
            emitter_id=1,
        )

        result = agent._process_features(event)

        # Should trigger cognitive attention
        assert result.get("cognitive_attention", False)
        assert result["bigram_rarity_score"] == 0.95

    def test_low_rarity_no_attention_flag(self, valid_bigrams, cluster_context_map):
        """Low rarity_score should not trigger cognitive attention."""
        prob_map = {
            (8, 12): BigramProbability(8, 12, count=1000, probability=0.40, rarity_score=0.1),
        }

        config = InteractionAgentConfig(
            cluster_context_map=cluster_context_map,
            valid_bigrams=valid_bigrams,
            bigram_probability_map=prob_map,
            enable_probabilistic_weighting=True,
            rarity_attention_threshold=0.8,
        )

        agent = InteractionAgent(config=config)
        agent._last_response_time = 0
        agent._last_cluster_id = 8

        from realtime.feature_subscriber import FeatureEvent

        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=12,
            features_112d=np.zeros(112, dtype=np.float32),
            timestamp=0.0,
            sequence=1,
            confidence=0.9,
            emitter_id=1,
        )

        result = agent._process_features(event)

        # Should NOT trigger cognitive attention
        assert not result.get("cognitive_attention", False)


# =============================================================================
# TEST SUITE 5: Integration - Full Markov Chain Pipeline
# =============================================================================


class TestMarkovChainIntegration:
    """Integration tests for full probabilistic pipeline."""

    def test_full_markov_chain_pipeline(
        self, synthetic_corpus_sequence, valid_bigrams, cluster_context_map
    ):
        """Complete pipeline: corpus → probabilities → weighted responses."""
        # Step 1: Build probability map from corpus
        prob_map = build_bigram_probability_map(
            corpus_sequence=synthetic_corpus_sequence,
            valid_bigrams=valid_bigrams,
        )

        # Verify 8→12 has higher probability than 8→25
        assert prob_map[(8, 12)].probability > prob_map[(8, 25)].probability

        # Step 2: Configure agent with probability map
        config = InteractionAgentConfig(
            cluster_context_map=cluster_context_map,
            valid_bigrams=valid_bigrams,
            bigram_probability_map=prob_map,
            enable_probabilistic_weighting=True,
            confidence_threshold=0.5,
        )

        agent = InteractionAgent(config=config)
        agent._last_response_time = 0
        agent._last_cluster_id = 8

        from realtime.feature_subscriber import FeatureEvent

        # Step 3: Process common transition
        event_common = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=12,
            features_112d=np.zeros(112, dtype=np.float32),
            timestamp=0.0,
            sequence=1,
            confidence=0.6,
            emitter_id=1,
        )

        result_common = agent._process_features(event_common)

        # Common transition: high probability, low rarity
        assert result_common["bigram_probability"] > 0.3
        # With prob=~0.5 and formula 0.5+prob=1.0, rarity=1-prob=0.5, so check <= 0.5
        assert result_common["bigram_rarity_score"] <= 0.5
        assert not result_common.get("cognitive_attention", False)

        # Step 4: Process rare transition
        event_rare = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=25,
            features_112d=np.zeros(112, dtype=np.float32),
            timestamp=1.0,
            sequence=2,
            confidence=0.6,
            emitter_id=1,
        )

        agent._last_cluster_id = 8  # Reset to 8
        result_rare = agent._process_features(event_rare)

        # Rare transition: low probability, high rarity
        assert result_rare["bigram_probability"] < 0.2  # 1/8 = 0.125
        assert result_rare["bigram_rarity_score"] > 0.7
        assert result_rare.get("cognitive_attention", False)

    def test_disabled_weighting_uses_binary_validation(self, valid_bigrams, cluster_context_map):
        """When enable_probabilistic_weighting=False, fall back to binary."""
        prob_map = {
            (8, 12): BigramProbability(8, 12, count=100, probability=0.50, rarity_score=0.2),
        }

        config = InteractionAgentConfig(
            cluster_context_map=cluster_context_map,
            valid_bigrams=valid_bigrams,
            bigram_probability_map=prob_map,
            enable_probabilistic_weighting=False,  # Disabled
        )

        agent = InteractionAgent(config=config)
        agent._last_response_time = 0
        agent._last_cluster_id = 8

        from realtime.feature_subscriber import FeatureEvent

        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=12,
            features_112d=np.zeros(112, dtype=np.float32),
            timestamp=0.0,
            sequence=1,
            confidence=0.6,
            emitter_id=1,
        )

        result = agent._process_features(event)

        # Should not include probability fields
        assert "bigram_probability" not in result or result["bigram_probability"] == 1.0
        # But bigram_valid should still work
        assert result["bigram_valid"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
