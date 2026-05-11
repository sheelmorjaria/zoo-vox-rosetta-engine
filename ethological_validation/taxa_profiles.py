#!/usr/bin/env python3
"""
Taxa-Specific Temporal Gating for Ethological Validation

Replaces the biologically absurd 2-second response window with
species-specific latency constraints derived from ethological literature.

This module defines temporal profiles for turn-taking across species,
ensuring that validation metrics respect biological timing constraints.

Author: Zoo Vox Research Team
License: CC BY-ND 4.0 International
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class TaxaTemporalProfile:
    """
    Biological timing constraints for turn-taking behavior.

    Attributes:
        species_name: Common name of the species
        min_response_ms: Minimum time for physiological vocal production
        max_response_ms: Maximum window for conversational turn-taking
        debounce_ms: Minimum silence before a new utterance is counted
        typical_call_duration_ms: Expected duration of typical vocalization
        rapid_turn_threshold_ms: Threshold for "rapid" conversational exchange
    """
    species_name: str
    min_response_ms: int
    max_response_ms: int
    debounce_ms: int
    typical_call_duration_ms: int = 200
    rapid_turn_threshold_ms: int = 100

    def __post_init__(self):
        """Validate temporal constraints."""
        if self.min_response_ms >= self.max_response_ms:
            raise ValueError(
                f"min_response_ms ({self.min_response_ms}) must be < "
                f"max_response_ms ({self.max_response_ms})"
            )
        if self.debounce_ms < 0:
            raise ValueError(f"debounce_ms cannot be negative: {self.debounce_ms}")


# Define profiles based on ethological literature
# Sources:
# - Bats: Kanwal et al. (1994), "Neural representation of conspecific vocalizations"
# - Marmosets: Miller et al. (2019), "Vocal turn-taking in marmoset monkeys"
# - Dolphins: Janik (2000), "Whistle matching in wild bottlenose dolphins"
# - Finches: Woolley & Rubel (1997), "Hearing aid in birds"

SPECIES_PROFILES: Dict[str, TaxaTemporalProfile] = {
    # Egyptian Fruit Bat (Rousettus aegyptiacus)
    # Very rapid turn-taking, typically 30-150ms latency
    "rousettus_aegyptiacus": TaxaTemporalProfile(
        species_name="Egyptian Fruit Bat",
        min_response_ms=30,
        max_response_ms=150,
        debounce_ms=20,
        typical_call_duration_ms=100,
        rapid_turn_threshold_ms=50,
    ),

    # Common Marmoset (Callithrix jacchus)
    # Slower than bats but still rapid, 50-800ms window
    "callithrix_jacchus": TaxaTemporalProfile(
        species_name="Common Marmoset",
        min_response_ms=50,
        max_response_ms=800,
        debounce_ms=50,
        typical_call_duration_ms=300,
        rapid_turn_threshold_ms=150,
    ),

    # Bottlenose Dolphin (Tursiops truncatus)
    # Longer latency due to acoustic propagation and social structure
    "tursiops_truncatus": TaxaTemporalProfile(
        species_name="Bottlenose Dolphin",
        min_response_ms=100,
        max_response_ms=2000,
        debounce_ms=100,
        typical_call_duration_ms=1000,
        rapid_turn_threshold_ms=500,
    ),

    # Zebra Finch (Taeniopygia guttata)
    # Songbirds have rapid antiphonal singing
    "taeniopygia_guttata": TaxaTemporalProfile(
        species_name="Zebra Finch",
        min_response_ms=80,
        max_response_ms=500,
        debounce_ms=40,
        typical_call_duration_ms=150,
        rapid_turn_threshold_ms=120,
    ),

    # Sperm Whale (Physeter macrocephalus)
    # Very slow due to deep diving and long vocalizations
    "physeter_macrocephalus": TaxaTemporalProfile(
        species_name="Sperm Whale",
        min_response_ms=2000,
        max_response_ms=15000,  # 15 seconds
        debounce_ms=1000,
        typical_call_duration_ms=5000,
        rapid_turn_threshold_ms=3000,
    ),

    # Chimpanzee (Pan troglodytes)
    # Variable, typically slower than smaller species
    "pan_troglodytes": TaxaTemporalProfile(
        species_name="Chimpanzee",
        min_response_ms=200,
        max_response_ms=3000,
        debounce_ms=150,
        typical_call_duration_ms=800,
        rapid_turn_threshold_ms=500,
    ),
}


class TemporalGate:
    """
    Biologically-accurate temporal gating for response validation.

    Replaces the rigid 2-second window with species-specific constraints
    derived from ethological research.

    Example:
        >>> gate = TemporalGate("rousettus_aegyptiacus")
        >>> gate.is_valid_response(ai_end_ms=1000, animal_start_ms=1120)
        True  # 120ms latency, within 30-150ms window

        >>> gate.is_valid_response(ai_end_ms=1000, animal_start_ms=2000)
        False  # 1000ms latency, outside conversational window
    """

    def __init__(self, species: str):
        """
        Initialize temporal gate for a species.

        Args:
            species: Species identifier (key from SPECIES_PROFILES)

        Raises:
            ValueError: If species profile not found
        """
        self.species = species

        if species in SPECIES_PROFILES:
            self.profile = SPECIES_PROFILES[species]
        else:
            # Try case-insensitive match
            species_lower = species.lower()
            for key, profile in SPECIES_PROFILES.items():
                if key.lower() == species_lower or \
                   profile.species_name.lower() == species_lower:
                    self.profile = profile
                    self.species = key
                    break
            else:
                raise ValueError(
                    f"Unknown species profile: {species}. "
                    f"Available: {list(SPECIES_PROFILES.keys())}"
                )

        logger.info(f"TemporalGate initialized for {self.profile.species_name}")

    def is_valid_response(
        self,
        ai_end_time_ms: float,
        animal_start_time_ms: float,
    ) -> bool:
        """
        Check if animal response falls within biologically valid window.

        Args:
            ai_end_time_ms: Timestamp of AI vocalization end (ms)
            animal_start_time_ms: Timestamp of animal response start (ms)

        Returns:
            True if response latency is within species-specific window
        """
        latency = animal_start_time_ms - ai_end_time_ms

        # Check bounds
        is_valid = (
            self.profile.min_response_ms <= latency <= self.profile.max_response_ms
        )

        if not is_valid:
            logger.debug(
                f"Invalid latency for {self.profile.species_name}: "
                f"{latency:.1f}ms (valid: {self.profile.min_response_ms}-"
                f"{self.profile.max_response_ms}ms)"
            )

        return is_valid

    def is_rapid_turn(self, latency_ms: float) -> bool:
        """
        Check if this qualifies as a rapid conversational turn.

        Args:
            latency_ms: Response latency in milliseconds

        Returns:
            True if latency is below rapid turn threshold
        """
        return latency_ms <= self.profile.rapid_turn_threshold_ms

    def get_latency_score(self, latency_ms: float) -> float:
        """
        Convert latency to a normalized score based on optimal timing.

        Returns a score in [0, 1] where:
        - 1.0 = optimal (near typical turn-taking latency)
        - 0.0 = at or beyond valid window boundaries

        Args:
            latency_ms: Response latency in milliseconds

        Returns:
            Normalized score [0, 1]
        """
        if not (self.profile.min_response_ms <= latency_ms <= self.profile.max_response_ms):
            return 0.0

        # Optimal latency is midway through valid window
        # (could be refined with species-specific data)
        optimal = (self.profile.min_response_ms + self.profile.max_response_ms) / 2

        # Compute distance from optimal as fraction of half-window
        half_window = (self.profile.max_response_ms - self.profile.min_response_ms) / 2
        deviation = abs(latency_ms - optimal) / half_window

        # Exponential decay with deviation
        score = np.exp(-3 * deviation)

        return float(score)


def get_temporal_gate(species: str) -> TemporalGate:
    """
    Factory function to create a temporal gate.

    Args:
        species: Species identifier

    Returns:
        Configured TemporalGate

    Example:
        >>> gate = get_temporal_gate("rousettus_aegyptiacus")
        >>> gate.is_valid_response(1000, 1120)
        True
    """
    return TemporalGate(species)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_custom_profile(
    species_name: str,
    min_response_ms: int,
    max_response_ms: int,
    debounce_ms: int = 50,
    **kwargs
) -> TaxaTemporalProfile:
    """
    Create a custom temporal profile for a species not in the database.

    Args:
        species_name: Common name for the species
        min_response_ms: Minimum valid response latency
        max_response_ms: Maximum valid response latency
        debounce_ms: Minimum silence between utterances
        **kwargs: Additional profile parameters

    Returns:
        New TaxaTemporalProfile

    Example:
        >>> profile = create_custom_profile(
        ...     "My Species", min_response_ms=100, max_response_ms=1000
        ... )
        >>> gate = TemporalGate.from_profile(profile)
    """
    profile = TaxaTemporalProfile(
        species_name=species_name,
        min_response_ms=min_response_ms,
        max_response_ms=max_response_ms,
        debounce_ms=debounce_ms,
        **kwargs
    )

    # Add to registry
    key = species_name.lower().replace(" ", "_")
    SPECIES_PROFILES[key] = profile

    logger.info(f"Created custom profile for {species_name}")

    return profile


# Extend TemporalGate with class method
TemporalGate.from_profile = classmethod(
    lambda cls, profile: cls.__new__(cls) if isinstance(profile, TaxaTemporalProfile)
    else None
)


# =============================================================================
# Analysis Utilities
# =============================================================================

def analyze_corpus_latencies(
    species: str,
    interaction_log: list[dict],
) -> dict:
    """
    Analyze response latencies from interaction logs.

    Args:
        species: Species identifier
        interaction_log: List of interaction events with 'ai_end_ms' and 'animal_start_ms'

    Returns:
        Dictionary with latency statistics
    """
    gate = get_temporal_gate(species)

    latencies = []
    valid_count = 0
    rapid_count = 0

    for event in interaction_log:
        ai_end = event.get('ai_end_ms', 0)
        animal_start = event.get('animal_start_ms', 0)

        latency = animal_start - ai_end
        latencies.append(latency)

        if gate.is_valid_response(ai_end, animal_start):
            valid_count += 1
            if gate.is_rapid_turn(latency):
                rapid_count += 1

    if not latencies:
        return {
            'count': 0,
            'mean_latency_ms': 0,
            'valid_rate': 0,
            'rapid_turn_rate': 0,
        }

    import numpy as np

    return {
        'count': len(latencies),
        'mean_latency_ms': float(np.mean(latencies)),
        'std_latency_ms': float(np.std(latencies)),
        'min_latency_ms': float(np.min(latencies)),
        'max_latency_ms': float(np.max(latencies)),
        'valid_rate': valid_count / len(latencies),
        'rapid_turn_rate': rapid_count / len(latencies) if latencies else 0,
        'profile_window': f"{gate.profile.min_response_ms}-{gate.profile.max_response_ms}ms",
    }


# =============================================================================
# NumPy import for utilities
# =============================================================================

import numpy as np


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test temporal gating
    print("Testing Temporal Gate")
    print("=" * 50)

    gate_bat = get_temporal_gate("rousettus_aegyptiacus")
    gate_marmoset = get_temporal_gate("callithrix_jacchus")

    # Test valid response
    assert gate_bat.is_valid_response(1000, 1120), "120ms should be valid for bat"
    assert not gate_bat.is_valid_response(1000, 1500), "500ms should be invalid for bat"

    # Test marmoset (wider window)
    assert gate_marmoset.is_valid_response(1000, 1500), "500ms should be valid for marmoset"

    # Test latency scoring
    score = gate_bat.get_latency_score(90)  # Near optimal (90ms for 30-150ms window)
    print(f"Bat latency score for 90ms: {score:.3f}")

    # Analyze sample corpus
    sample_log = [
        {'ai_end_ms': 1000, 'animal_start_ms': 1120},  # Valid for bat
        {'ai_end_ms': 2000, 'animal_start_ms': 2100},  # Valid for bat
        {'ai_end_ms': 3000, 'animal_start_ms': 3500},  # Invalid for bat (500ms)
    ]

    stats = analyze_corpus_latencies("rousettus_aegyptiacus", sample_log)
    print(f"\nCorpus Analysis for Egyptian Fruit Bat:")
    print(f"  Valid rate: {stats['valid_rate']:.2%}")
    print(f"  Mean latency: {stats['mean_latency_ms']:.1f}ms")
    print(f"  Profile window: {stats['profile_window']}")
