"""
Metadata-First Synthesis Engine
==============================

Replaces Persona-Based routing with Direct Metadata Queries.

KEY INSIGHT:
- Personas = Taxonomic labels (discrete boxes) - for human readability
- Metadata = Genetic coordinates (continuous vectors) - for math power

Architecture:
    OLD: Intent → PersonaRouter → Single Buffer → Synthesis
    NEW: Intent → Vector Query → Multi-Buffer Selection → Granular Morphing

This enables:
- Interpolation BETWEEN personas (not just within)
- "Ghost Word" discovery in the void between clusters
- Cross-pollination of acoustic features across semantic boundaries

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MetadataQuery:
    """Query for the vector space."""
    target_f0_hz: float
    target_duration_ms: float
    f0_tolerance_hz: float = 500.0
    duration_tolerance_ms: float = 20.0

    # Soft constraints (score modifiers, NOT hard filters)
    preferred_contexts: List[str] = field(default_factory=list)
    avoided_contexts: List[str] = field(default_factory=list)
    preferred_clusters: List[int] = field(default_factory=list)

    # Scoring weights
    acoustic_weight: float = 1.0  # F0/duration matching
    context_weight: float = 0.5   # Context preference
    novelty_weight: float = 0.3   # Reward exploration


@dataclass
class PhraseCandidate:
    """A phrase from the database with metadata and scoring."""
    phrase_id: str
    audio_buffer: np.ndarray
    metadata: Dict[str, Any]
    sample_rate: int

    # Computed scores
    acoustic_score: float = 0.0  # Distance from target in vector space
    context_score: float = 0.0   # Context preference match
    novelty_score: float = 0.0   # Exploration bonus
    total_score: float = 0.0     # Weighted combination

    def __post_init__(self):
        """Extract acoustic features from metadata."""
        self.f0_hz = self.metadata.get('mean_f0_hz', 0.0)
        self.duration_ms = self.metadata.get('duration_ms', 0.0)
        self.f0_range_hz = self.metadata.get('f0_range_hz', 0.0)
        self.harmonicity = self.metadata.get('harmonicity', 0.0)
        self.context = self.metadata.get('context', 'unknown')
        self.cluster_id = self.metadata.get('cluster_id', -1)
        self.species = self.metadata.get('species', 'unknown')


@dataclass
class SynthesisRecipe:
    """
    A synthesis recipe with multiple source buffers.

    Unlike persona-based synthesis (single buffer), this can combine
    multiple buffers to create "Ghost Words" in the void between clusters.
    """
    sources: List[Tuple[PhraseCandidate, float]]  # (candidate, weight)
    target_params: Dict[str, float]  # Interpolated target parameters
    synthesis_mode: str = "morph"  # morph, crossfade, alternate

    # Metadata about the synthesis
    is_cross_persona: bool = False
    discovery_potential: float = 0.0  # 0-1, how novel is this?
    reasoning: str = ""


class VectorSpaceQueryEngine:
    """
    Queries the acoustic vector space directly.

    Instead of routing to a persona, we search for phrases that match
    our target acoustic coordinates, using context only as a soft constraint.
    """

    def __init__(self, phrase_database_path: Optional[str] = None):
        """
        Initialize the query engine.

        Args:
            phrase_database_path: Path to JSON with phrase metadata
        """
        self.phrases: List[PhraseCandidate] = []
        self.species_index: Dict[str, List[PhraseCandidate]] = {}
        self.cluster_index: Dict[int, List[PhraseCandidate]] = {}

        if phrase_database_path:
            self.load_phrase_database(phrase_database_path)
        else:
            # Create synthetic phrases for testing
            self._create_synthetic_phrases()

        logger.info(f"VectorSpaceQueryEngine initialized with {len(self.phrases)} phrases")

    def load_phrase_database(self, db_path: str):
        """Load phrase metadata from JSON database."""
        try:
            with open(db_path, 'r') as f:
                json.load(f)

            # For now, create synthetic candidates
            # In production, this would load from actual database
            self._create_synthetic_phrases()

        except Exception as e:
            logger.warning(f"Failed to load phrase database: {e}")
            self._create_synthetic_phrases()

    def _create_synthetic_phrases(self):
        """Create synthetic phrase candidates for testing."""
        # Marmoset phrases
        marmoset_phrases = [
            {
                "phrase_id": "marm_phee_001",
                "species": "marmoset",
                "cluster_id": 0,
                "context": "contact",
                "mean_f0_hz": 6526.0,
                "duration_ms": 76.5,
                "f0_range_hz": 427.0,
                "harmonicity": 0.95,
            },
            {
                "phrase_id": "marm_phee_002",
                "species": "marmoset",
                "cluster_id": 0,
                "context": "contact",
                "mean_f0_hz": 6480.0,
                "duration_ms": 82.0,
                "f0_range_hz": 380.0,
                "harmonicity": 0.93,
            },
            {
                "phrase_id": "marm_alarm_001",
                "species": "marmoset",
                "cluster_id": 1,
                "context": "alarm",
                "mean_f0_hz": 6020.0,
                "duration_ms": 58.1,
                "f0_range_hz": 3722.0,
                "harmonicity": 0.7,
            },
        ]

        # Bat phrases
        bat_phrases = [
            {
                "phrase_id": "bat_midfm_001",
                "species": "egyptian_bat",
                "cluster_id": 1,
                "context": "navigation",
                "mean_f0_hz": 7437.0,
                "duration_ms": 17.4,
                "f0_range_hz": 9755.0,
                "harmonicity": 0.6,
            },
            {
                "phrase_id": "bat_social_001",
                "species": "egyptian_bat",
                "cluster_id": 2,
                "context": "social",
                "mean_f0_hz": 7408.0,
                "duration_ms": 17.4,
                "f0_range_hz": 24.0,
                "harmonicity": 0.85,
            },
            {
                "phrase_id": "bat_roost_001",
                "species": "egyptian_bat",
                "cluster_id": 0,
                "context": "roost",
                "mean_f0_hz": 2884.0,
                "duration_ms": 11.6,
                "f0_range_hz": 11535.0,
                "harmonicity": 0.5,
            },
        ]

        # Create candidates with synthetic audio
        sample_rate = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        for phrase_data in marmoset_phrases + bat_phrases:
            # Generate synthetic audio
            f0 = phrase_data['mean_f0_hz']
            audio = 0.5 * np.sin(2 * np.pi * f0 * t)

            # Add F0 range modulation
            f0_range = phrase_data['f0_range_hz']
            if f0_range > 100:
                modulation = 0.3 * np.sin(2 * np.pi * 20 * t)
                audio = 0.5 * np.sin(2 * np.pi * f0 * t + modulation)

            candidate = PhraseCandidate(
                phrase_id=phrase_data['phrase_id'],
                audio_buffer=audio,
                metadata=phrase_data,
                sample_rate=sample_rate
            )

            self.phrases.append(candidate)

        # Build indexes
        for phrase in self.phrases:
            species = phrase.species
            if species not in self.species_index:
                self.species_index[species] = []
            self.species_index[species].append(phrase)

            cluster = phrase.cluster_id
            if cluster not in self.cluster_index:
                self.cluster_index[cluster] = []
            self.cluster_index[cluster].append(phrase)

        logger.info(f"Created {len(self.phrases)} synthetic phrase candidates")
        logger.info(f"Species index: {list(self.species_index.keys())}")
        logger.info(f"Cluster index: {list(self.cluster_index.keys())}")

    def query_nearest_metadata(
        self,
        query: MetadataQuery,
        species: Optional[str] = None,
        top_k: int = 5
    ) -> List[PhraseCandidate]:
        """
        Query the vector space for nearest neighbors.

        Returns candidates ranked by total_score (acoustic + context + novelty).
        """
        candidates = []

        # Filter by species if specified
        search_space = self.phrases
        if species:
            search_space = self.species_index.get(species, [])

        for phrase in search_space:
            # Calculate acoustic score (distance in vector space)
            f0_distance = abs(phrase.f0_hz - query.target_f0_hz)
            duration_distance = abs(phrase.duration_ms - query.target_duration_ms)

            # Normalize distances (closer = higher score)
            f0_score = 1.0 / (1.0 + f0_distance / query.f0_tolerance_hz)
            duration_score = 1.0 / (1.0 + duration_distance / query.duration_tolerance_ms)
            phrase.acoustic_score = (f0_score + duration_score) / 2.0

            # Calculate context score (soft constraint)
            phrase.context_score = 0.0
            if phrase.context in query.preferred_contexts:
                phrase.context_score += query.context_weight
            if phrase.context in query.avoided_contexts:
                phrase.context_score -= query.context_weight * 0.5

            # Calculate novelty score (reward exploration)
            # Prefer phrases from less-used clusters
            cluster_usage = len(self.cluster_index.get(phrase.cluster_id, []))
            phrase.novelty_score = query.novelty_weight * (1.0 / cluster_usage)

            # Total score (weighted combination)
            phrase.total_score = (
                query.acoustic_weight * phrase.acoustic_score +
                phrase.context_score +
                phrase.novelty_score
            )

            candidates.append(phrase)

        # Sort by total score and return top_k
        candidates.sort(key=lambda p: p.total_score, reverse=True)
        return candidates[:top_k]

    def query_interpolation_targets(
        self,
        query: MetadataQuery,
        num_sources: int = 2,
        species: Optional[str] = None
    ) -> SynthesisRecipe:
        """
        Query for interpolation targets.

        Unlike query_nearest_metadata (which finds single best match),
        this can return multiple sources for blending/morphing.

        This enables "Ghost Word" synthesis - creating sounds that
        exist between clusters in the vector space.
        """
        candidates = self.query_nearest_metadata(
            query, species=species, top_k=num_sources * 2
        )

        # Select diverse sources for interpolation
        sources = []
        used_clusters = set()

        for candidate in candidates:
            # Prefer diverse clusters for interpolation
            if candidate.cluster_id not in used_clusters or len(sources) < num_sources:
                weight = candidate.total_score
                sources.append((candidate, weight))
                used_clusters.add(candidate.cluster_id)

            if len(sources) >= num_sources:
                break

        # Normalize weights
        total_weight = sum(w for _, w in sources)
        sources = [(c, w/total_weight) for c, w in sources]

        # Calculate target parameters (weighted average)
        target_f0 = sum(c.f0_hz * w for c, w in sources)
        target_duration = sum(c.duration_ms * w for c, w in sources)
        target_f0_range = sum(c.f0_range_hz * w for c, w in sources)

        # Determine if this is cross-persona synthesis
        clusters_used = [c.cluster_id for c, _ in sources]
        is_cross_persona = len(set(clusters_used)) > 1

        # Calculate discovery potential (how novel is this?)
        # Higher if we're interpolating between distant clusters
        if is_cross_persona and len(sources) >= 2:
            cluster_distance = abs(sources[0][0].cluster_id - sources[1][0].cluster_id)
            discovery_potential = min(1.0, cluster_distance / 5.0)
        else:
            discovery_potential = 0.0

        # Generate reasoning
        source_names = [c.phrase_id for c, _ in sources]
        clusters_str = ", ".join(f"C{c.cluster_id}" for c, _ in sources)
        reasoning = (
            f"Interpolating {len(sources)} sources: {', '.join(source_names)} "
            f"(clusters: {clusters_str})"
        )

        return SynthesisRecipe(
            sources=sources,
            target_params={
                'mean_f0_hz': target_f0,
                'duration_ms': target_duration,
                'f0_range_hz': target_f0_range,
            },
            synthesis_mode="morph",
            is_cross_persona=is_cross_persona,
            discovery_potential=discovery_potential,
            reasoning=reasoning
        )


class MetadataFirstSynthesizer:
    """
    Metadata-First Synthesis Engine

    Replaces PersonaRouter with direct vector space queries.
    """

    def __init__(self, phrase_database_path: Optional[str] = None):
        self.query_engine = VectorSpaceQueryEngine(phrase_database_path)
        self.sample_rate = 48000

    def synthesize_by_target(
        self,
        target_f0_hz: float,
        target_duration_ms: float,
        species: Optional[str] = None,
        preferred_contexts: Optional[List[str]] = None,
        synthesis_duration_ms: float = 200.0
    ) -> Tuple[np.ndarray, SynthesisRecipe]:
        """
        Synthesize by targeting acoustic coordinates.

        This is the key difference from persona-based synthesis:
        - We don't select a persona first
        - We query the vector space directly
        - We may use multiple source buffers for interpolation
        """

        # Create query
        query = MetadataQuery(
            target_f0_hz=target_f0_hz,
            target_duration_ms=target_duration_ms,
            preferred_contexts=preferred_contexts or [],
        )

        # Get synthesis recipe (may include multiple sources)
        recipe = self.query_engine.query_interpolation_targets(
            query, num_sources=2, species=species
        )

        # Synthesize using recipe
        audio = self._synthesize_from_recipe(recipe, synthesis_duration_ms)

        return audio, recipe

    def synthesize_ghost_word(
        self,
        cluster_a_id: int,
        cluster_b_id: int,
        blend_ratio: float = 0.5,
        species: Optional[str] = None
    ) -> Tuple[np.ndarray, SynthesisRecipe]:
        """
        Synthesize a "Ghost Word" - a sound in the void between two clusters.

        This is the key discovery mechanism: create sounds that don't
        exist in the database but lie on the interpolation line between
        known clusters.
        """

        # Get phrases from both clusters
        phrases_a = self.query_engine.cluster_index.get(cluster_a_id, [])
        phrases_b = self.query_engine.cluster_index.get(cluster_b_id, [])

        if not phrases_a or not phrases_b:
            raise ValueError(f"Cannot find phrases for clusters {cluster_a_id} and {cluster_b_id}")

        # Select best from each cluster
        best_a = max(phrases_a, key=lambda p: p.harmonicity)
        best_b = max(phrases_b, key=lambda p: p.harmonicity)

        # Create recipe
        sources = [
            (best_a, 1.0 - blend_ratio),
            (best_b, blend_ratio)
        ]

        # Calculate target (interpolated between clusters)
        target_f0 = best_a.f0_hz * (1 - blend_ratio) + best_b.f0_hz * blend_ratio
        target_duration = best_a.duration_ms * (1 - blend_ratio) + best_b.duration_ms * blend_ratio

        reasoning = (
            f"Ghost word: Cluster {cluster_a_id} + Cluster {cluster_b_id} "
            f"@ {blend_ratio:.1%} ratio"
        )

        recipe = SynthesisRecipe(
            sources=sources,
            target_params={
                'mean_f0_hz': target_f0,
                'duration_ms': target_duration,
                'f0_range_hz': 0.0,  # Will be calculated
            },
            synthesis_mode="morph",
            is_cross_persona=True,
            discovery_potential=1.0,  # Maximum novelty
            reasoning=reasoning
        )

        audio = self._synthesize_from_recipe(recipe, 200.0)

        return audio, recipe

    def _synthesize_from_recipe(
        self,
        recipe: SynthesisRecipe,
        duration_ms: float
    ) -> np.ndarray:
        """Synthesize audio from a recipe (may use multiple sources)."""

        if len(recipe.sources) == 1:
            # Single source synthesis
            candidate, _ = recipe.sources[0]
            return self._granular_synthesize(
                candidate.audio_buffer,
                duration_ms,
                pitch_shift=recipe.target_params['mean_f0_hz'] / candidate.f0_hz
            )
        else:
            # Multi-source morphing
            return self._morph_sources(recipe, duration_ms)

    def _granular_synthesize(
        self,
        source_buffer: np.ndarray,
        duration_ms: float,
        pitch_shift: float = 1.0,
        grain_size_ms: float = 50.0
    ) -> np.ndarray:
        """Granular synthesis from single source."""
        grain_size = int(grain_size_ms * self.sample_rate / 1000)
        num_samples = int(duration_ms * self.sample_rate / 1000)
        output = np.zeros(num_samples)

        # Simple grain-based synthesis
        position = 0.0
        effective_stride = 1.0 / pitch_shift

        for i in range(num_samples):
            pos_int = int(position) % len(source_buffer)
            pos_frac = position - int(position)

            if pos_int + 1 < len(source_buffer):
                sample = source_buffer[pos_int] * (1 - pos_frac) + source_buffer[pos_int + 1] * pos_frac
            else:
                sample = source_buffer[pos_int]

            # Apply grain window
            grain_pos = int(position % grain_size)
            window = 0.5 * (1 - np.cos(2 * np.pi * grain_pos / grain_size))
            output[i] = sample * window

            position += effective_stride

        return output

    def _morph_sources(
        self,
        recipe: SynthesisRecipe,
        duration_ms: float
    ) -> np.ndarray:
        """Morph between multiple source buffers."""

        grain_size_ms = 50.0
        num_samples = int(duration_ms * self.sample_rate / 1000)
        output = np.zeros(num_samples)

        # Alternating grains from each source
        grain_size = int(grain_size_ms * self.sample_rate / 1000)
        grain_idx = 0

        for source, weight in recipe.sources:
            grain_start = grain_idx * grain_size
            if grain_start >= num_samples:
                break

            grain_end = min(grain_start + grain_size, num_samples)
            grain_audio = self._granular_synthesize(
                source.audio_buffer,
                (grain_end - grain_start) * 1000 / self.sample_rate,
                pitch_shift=recipe.target_params['mean_f0_hz'] / source.f0_hz,
                grain_size_ms=grain_size_ms
            )

            # Mix with weight
            output[grain_start:grain_end] += grain_audio * weight

            grain_idx += 1

        # Normalize
        max_val = np.max(np.abs(output))
        if max_val > 0:
            output = output / max_val * 0.95

        return output


# ============================================================================
# Demo
# ============================================================================

def demonstrate_metadata_first_synthesis():
    """Demonstrate metadata-first synthesis advantages."""

    print("\n" + "="*80)
    print("METADATA-FIRST SYNTHESIS DEMONSTRATION")
    print("="*80)

    synthesizer = MetadataFirstSynthesizer()

    # Demo 1: Direct target query
    print("\n--- Demo 1: Direct Target Query ---")
    print("Query: Find phrases near F0=7000Hz, Duration=50ms")

    audio, recipe = synthesizer.synthesize_by_target(
        target_f0_hz=7000.0,
        target_duration_ms=50.0,
        species="egyptian_bat",
        synthesis_duration_ms=200.0
    )

    print(f"Recipe: {recipe.reasoning}")
    print(f"Sources: {len(recipe.sources)}")
    for candidate, weight in recipe.sources:
        print(f"  - {candidate.phrase_id} (weight={weight:.2f})")
        print(f"    F0={candidate.f0_hz:.0f}Hz, Context={candidate.context}")

    if recipe.is_cross_persona:
        print("⚠️  CROSS-PERSONA SYNTHESIS!")
    print(f"Discovery Potential: {recipe.discovery_potential:.2f}")

    # Demo 2: Ghost Word synthesis
    print("\n--- Demo 2: Ghost Word Discovery ---")
    print("Synthesizing sound between Bat Cluster 1 (Mid-FM) and Cluster 2 (Social)")

    audio, recipe = synthesizer.synthesize_ghost_word(
        cluster_a_id=1,
        cluster_b_id=2,
        blend_ratio=0.5,
        species="egyptian_bat"
    )

    print(f"Recipe: {recipe.reasoning}")
    print(f"Target F0: {recipe.target_params['mean_f0_hz']:.0f}Hz")
    print(f"Discovery Potential: {recipe.discovery_potential:.2f}")
    print("⚠️  This sound exists in the VOID between clusters!")

    # Demo 3: Cross-species query (acoustic-only, ignoring context)
    print("\n--- Demo 3: Acoustic-Only Query ---")
    print("Finding phrases by acoustic features, regardless of semantic context")

    audio, recipe = synthesizer.synthesize_by_target(
        target_f0_hz=6500.0,
        target_duration_ms=75.0,
        # No species filter - let it find best acoustic match
        synthesis_duration_ms=200.0
    )

    print(f"Recipe: {recipe.reasoning}")
    print("Selected by ACOUSTIC match, not by persona label")
    for candidate, weight in recipe.sources:
        print(f"  - {candidate.phrase_id} ({candidate.species})")
        print(f"    F0={candidate.f0_hz:.0f}Hz (target was 6500Hz)")

    print("\n" + "="*80)
    print("\n🎯 KEY INSIGHT:")
    print("   Metadata-First enables interpolation BETWEEN personas,")
    print("   not just within them. This discovers 'Ghost Words' that")
    print("   exist theoretically but not statistically in the dataset.")
    print()


if __name__ == "__main__":
    demonstrate_metadata_first_synthesis()
