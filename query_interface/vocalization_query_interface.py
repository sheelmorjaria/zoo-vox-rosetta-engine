"""
Query Interface for Animal Vocalization Database

This module provides efficient query interfaces for accessing vocalization data
in real-time with various filtering, aggregation, and search capabilities.
"""

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from data_models import (
    GrammarRule,
    Phrase,
    PhraseContext,
    Sentence,
    Species,
    VocalizationDatabase,
    VocalizationModality,
)

logger = logging.getLogger(__name__)


class VocalizationQueryInterface:
    """High-performance query interface for vocalization data"""

    def __init__(self, database_path: str = "./src/vocalization_database.json"):
        """Initialize with database path"""
        self.database_path = Path(database_path)
        self.db = None
        self._load_database()
        self._build_indexes()

    def _load_database(self):
        """Load database from JSON file"""
        try:
            with open(self.database_path, "r") as f:
                data = json.load(f)

            # Reconstruct the database
            self.db = VocalizationDatabase()

            # Import species data
            for species_value, species_data in data["species_data"].items():
                species = Species(species_value)
                species_data_obj = self._create_species_data_from_json(species, species_data)
                self.db.add_species_data(species_data_obj)

        except FileNotFoundError:
            logger.error(f"Database file not found: {self.database_path}")
            self.db = VocalizationDatabase()
        except Exception as e:
            logger.error(f"Error loading database: {e}")
            self.db = VocalizationDatabase()

    def _create_species_data_from_json(self, species: Species, species_data: Dict) -> Any:
        """Create SpeciesData object from JSON data"""
        from data_models import GrammarRule, Phrase, SpeciesData

        species_data_obj = SpeciesData(species=species)

        # Set basic stats
        species_data_obj.total_phrases = species_data.get("total_phrases", 0)
        species_data_obj.total_sentences = species_data.get("total_sentences", 0)
        species_data_obj.total_grammar_rules = species_data.get("total_grammar_rules", 0)
        species_data_obj.vocabulary_size = species_data.get("vocabulary_size", 0)

        # Import phrases
        for phrase_key, phrase_data in species_data["phrases"].items():
            acoustic_features = self._create_acoustic_features_from_json(
                phrase_data["acoustic_features"]
            )

            phrase = Phrase(
                phrase_key=phrase_key,
                signature=phrase_data["signature"],
                species=Species(phrase_data["species"]),
                modality=VocalizationModality(phrase_data["modality"]),
                acoustic_features=acoustic_features,
                total_occurrences=phrase_data["total_occurrences"],
                contexts=[
                    PhraseContext(ctx["context_name"], ctx["count"], ctx.get("percentage", 0))
                    for ctx in phrase_data.get("contexts", [])
                ],
                social_contexts=phrase_data.get("social_contexts", {}),
                is_compositional=phrase_data.get("is_compositional", False),
                phrase_components=phrase_data.get("phrase_components", []),
            )
            species_data_obj.add_phrase(phrase)

        # Import sentences
        for sentence_data in species_data["sentences"]:
            sentence = Sentence(
                sentence_id=sentence_data["sentence_id"],
                species=Species(sentence_data["species"]),
                phrase_sequence=sentence_data["phrase_sequence"],
                context=sentence_data["context"],
                has_ascending_syntax=sentence_data["has_ascending_syntax"],
                has_descending_syntax=sentence_data["has_descending_syntax"],
                has_fm_pattern=sentence_data["has_fm_pattern"],
                syntax_score=sentence_data["syntax_score"],
                total_duration_ms=sentence_data["total_duration_ms"],
                complexity_score=sentence_data["complexity_score"],
            )
            species_data_obj.sentences.append(sentence)

        # Import grammar rules
        for rule_data in species_data["grammar_rules"]:
            rule = GrammarRule(
                rule_id=rule_data["rule_id"],
                from_phrase=rule_data["from_phrase"],
                to_phrase=rule_data["to_phrase"],
                frequency=rule_data["frequency"],
                confidence=rule_data["confidence"],
                contexts=rule_data["contexts"],
                is_bidirectional=rule_data["is_bidirectional"],
                strength_score=rule_data["strength_score"],
            )
            species_data_obj.grammar_rules.append(rule)

        return species_data_obj

    def _create_acoustic_features_from_json(self, features_data: Dict) -> Any:
        """Create AcousticFeatures object from JSON data"""
        from data_models import AcousticFeatures

        return AcousticFeatures(
            mean_f0_hz=features_data["mean_f0_hz"],
            std_f0_hz=features_data["std_f0_hz"],
            min_f0_hz=features_data["min_f0_hz"],
            max_f0_hz=features_data["max_f0_hz"],
            f0_range_hz=features_data["f0_range_hz"],
            duration_frames=features_data["duration_frames"],
            voiced_ratio=features_data["voiced_ratio"],
            f0_slope=features_data["f0_slope"],
            modulation_rate=features_data["modulation_rate"],
            acoustic_variance=features_data["acoustic_variance"],
            mean_duration_ms=features_data["mean_duration_ms"],
        )

    def _build_indexes(self):
        """Build indexes for efficient querying"""
        self.phrase_by_key_index = {}
        self.species_phrase_index = {}
        self.modality_phrase_index = {}
        self.grammar_transition_index = {}

        for species, species_data in self.db.species_data.items():
            # Species phrase index
            if species not in self.species_phrase_index:
                self.species_phrase_index[species] = []
            self.species_phrase_index[species].extend(species_data.phrase_library.keys())

            # Modality index
            for phrase_key, phrase in species_data.phrase_library.items():
                # Global phrase index
                self.phrase_by_key_index[phrase_key] = (species, phrase)

                # Modality index
                modality = phrase.modality
                if modality not in self.modality_phrase_index:
                    self.modality_phrase_index[modality] = {}
                if species not in self.modality_phrase_index[modality]:
                    self.modality_phrase_index[modality][species] = []
                self.modality_phrase_index[modality][species].append(phrase_key)
            # Grammar transition index
            for rule in species_data.grammar_rules:
                if rule.from_phrase not in self.grammar_transition_index:
                    self.grammar_transition_index[rule.from_phrase] = {}
                self.grammar_transition_index[rule.from_phrase][rule.to_phrase] = rule

    def get_phrase_by_key(self, phrase_key: str) -> Optional[Tuple[Species, Phrase]]:
        """Get phrase by key using index"""
        return self.phrase_by_key_index.get(phrase_key)

    def get_phrases_by_species(self, species: Species) -> Dict[str, Phrase]:
        """Get all phrases for a species"""
        species_data = self.db.get_species_data(species)
        return species_data.phrase_library if species_data else {}

    def get_phrases_by_modality(self, modality: VocalizationModality) -> Dict[Species, List[str]]:
        """Get phrases grouped by modality"""
        return self.modality_phrase_index.get(modality, {})

    def get_grammar_transitions(self, from_phrase: str) -> Dict[str, GrammarRule]:
        """Get all grammar transitions from a phrase"""
        return self.grammar_transition_index.get(from_phrase, {})

    def search_phrases_by_f0_range(
        self, min_f0: float, max_f0: float, species: Optional[Species] = None
    ) -> List[Tuple[str, Phrase]]:
        """Search phrases by F0 frequency range"""
        results = []

        if species:
            phrases = self.get_phrases_by_species(species).items()
        else:
            phrases = []
            for species_data in self.db.species_data.values():
                phrases.extend(species_data.phrase_library.items())

        for phrase_key, phrase in phrases:
            if min_f0 <= phrase.acoustic_features.mean_f0_hz <= max_f0:
                results.append((phrase_key, phrase))

        return results

    def search_phrases_by_duration(
        self, min_duration: float, max_duration: float, species: Optional[Species] = None
    ) -> List[Tuple[str, Phrase]]:
        """Search phrases by duration range"""
        results = []

        if species:
            phrases = self.get_phrases_by_species(species).items()
        else:
            phrases = []
            for species_data in self.db.species_data.values():
                phrases.extend(species_data.phrase_library.items())

        for phrase_key, phrase in phrases:
            if min_duration <= phrase.acoustic_features.mean_duration_ms <= max_duration:
                results.append((phrase_key, phrase))

        return results

    def get_similar_phrases(
        self, phrase_key: str, threshold: float = 0.8
    ) -> List[Tuple[float, str, Phrase]]:
        """Find similar phrases based on acoustic features"""
        if phrase_key not in self.phrase_by_key_index:
            return []

        target_species, target_phrase = self.phrase_by_key_index[phrase_key]
        target_f0 = target_phrase.acoustic_features.mean_f0_hz
        target_duration = target_phrase.acoustic_features.mean_duration_ms
        target_range = target_phrase.acoustic_features.f0_range_hz

        similarities = []

        for key, (species, phrase) in self.phrase_by_key_index.items():
            if key == phrase_key:
                continue

            # Calculate similarity based on normalized features
            f0_diff = abs(target_f0 - phrase.acoustic_features.mean_f0_hz)
            duration_diff = abs(target_duration - phrase.acoustic_features.mean_duration_ms)
            range_diff = abs(target_range - phrase.acoustic_features.f0_range_hz)

            # Normalize differences (0-1 scale)
            f0_similarity = 1 - min(f0_diff / 10000, 1)  # Max diff 10kHz
            duration_similarity = 1 - min(duration_diff / 1000, 1)  # Max diff 1000ms
            range_similarity = 1 - min(range_diff / 5000, 1)  # Max diff 5kHz

            # Weighted average similarity
            similarity = f0_similarity * 0.5 + duration_similarity * 0.3 + range_similarity * 0.2

            if similarity >= threshold:
                similarities.append((similarity, key, phrase))

        return sorted(similarities, reverse=True)

    def get_phrase_statistics(self, species: Optional[Species] = None) -> Dict[str, Any]:
        """Get comprehensive statistics about phrases"""
        stats = {
            "total_phrases": 0,
            "species_breakdown": {},
            "modality_breakdown": {},
            "frequency_distribution": {"min": float("inf"), "max": 0, "avg": 0},
            "duration_distribution": {"min": float("inf"), "max": 0, "avg": 0},
            "context_distribution": {},
            "semantic_coverage": 0,
        }

        phrases_to_analyze = []
        if species:
            phrases_to_analyze = list(self.get_phrases_by_species(species).items())
        else:
            for species_data in self.db.species_data.values():
                phrases_to_analyze.extend(species_data.phrase_library.items())

        if not phrases_to_analyze:
            return stats

        total_f0 = 0
        total_duration = 0
        phrase_count = 0

        for phrase_key, phrase in phrases_to_analyze:
            # Species breakdown
            species_name = phrase.species.value
            if species_name not in stats["species_breakdown"]:
                stats["species_breakdown"][species_name] = 0
            stats["species_breakdown"][species_name] += 1

            # Modality breakdown
            modality = phrase.modality.value
            if modality not in stats["modality_breakdown"]:
                stats["modality_breakdown"][modality] = 0
            stats["modality_breakdown"][modality] += 1

            # Frequency distribution
            f0 = phrase.acoustic_features.mean_f0_hz
            stats["frequency_distribution"]["min"] = min(stats["frequency_distribution"]["min"], f0)
            stats["frequency_distribution"]["max"] = max(stats["frequency_distribution"]["max"], f0)
            total_f0 += f0

            # Duration distribution
            duration = phrase.acoustic_features.mean_duration_ms
            stats["duration_distribution"]["min"] = min(
                stats["duration_distribution"]["min"], duration
            )
            stats["duration_distribution"]["max"] = max(
                stats["duration_distribution"]["max"], duration
            )
            total_duration += duration

            # Context distribution
            for ctx in phrase.contexts:
                ctx_name = ctx.context_name
                if ctx_name not in stats["context_distribution"]:
                    stats["context_distribution"][ctx_name] = 0
                stats["context_distribution"][ctx_name] += ctx.count

            phrase_count += 1

        # Calculate averages
        if phrase_count > 0:
            stats["frequency_distribution"]["avg"] = total_f0 / phrase_count
            stats["duration_distribution"]["avg"] = total_duration / phrase_count
            stats["total_phrases"] = phrase_count

        return stats

    def get_grammar_network(self, species: Optional[Species] = None) -> Dict[str, Any]:
        """Get grammar network statistics"""
        network = {
            "nodes": 0,
            "edges": 0,
            "transitions": {},
            "most_connected_phrases": [],
            "strongest_transitions": [],
        }

        transitions_to_analyze = []

        if species:
            species_data = self.db.get_species_data(species)
            if species_data:
                transitions_to_analyze = species_data.grammar_rules
        else:
            for species_data in self.db.species_data.values():
                transitions_to_analyze.extend(species_data.grammar_rules)

        # Build transition map
        transition_map = {}
        phrase_connections = {}

        for rule in transitions_to_analyze:
            from_phrase = rule.from_phrase
            to_phrase = rule.to_phrase

            if from_phrase not in transition_map:
                transition_map[from_phrase] = {}
            transition_map[from_phrase][to_phrase] = rule.frequency

            if from_phrase not in phrase_connections:
                phrase_connections[from_phrase] = 0
            if to_phrase not in phrase_connections:
                phrase_connections[to_phrase] = 0
            phrase_connections[from_phrase] += 1
            phrase_connections[to_phrase] += 1

        # Compile network stats
        network["nodes"] = len(phrase_connections)
        network["edges"] = len(transitions_to_analyze)
        network["transitions"] = transition_map

        # Most connected phrases
        network["most_connected_phrases"] = sorted(
            phrase_connections.items(), key=lambda x: x[1], reverse=True
        )[:10]

        # Strongest transitions
        all_transitions = []
        for from_phrase, to_phrases in transition_map.items():
            for to_phrase, frequency in to_phrases.items():
                all_transitions.append((from_phrase, to_phrase, frequency))

        network["strongest_transitions"] = sorted(
            all_transitions, key=lambda x: x[2], reverse=True
        )[:10]

        return network

    def export_query_results(self, results: List[Any], format: str = "json") -> str:
        """Export query results in specified format"""
        if format == "json":
            return json.dumps(
                [asdict(item) if hasattr(item, "__dict__") else item for item in results], indent=2
            )
        elif format == "csv":
            import csv
            import io

            output = io.StringIO()
            writer = csv.writer(output)

            # Simple CSV export for phrase results
            if results and isinstance(results[0], tuple) and len(results[0]) >= 2:
                writer.writerow(
                    ["Phrase Key", "Species", "Mean F0 (Hz)", "Duration (ms)", "Occurrences"]
                )
                for phrase_key, phrase in results:
                    writer.writerow(
                        [
                            phrase_key,
                            phrase.species.value,
                            phrase.acoustic_features.mean_f0_hz,
                            phrase.acoustic_features.mean_duration_ms,
                            phrase.total_occurrences,
                        ]
                    )

            return output.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format}")

    def refresh_database(self):
        """Reload database from file"""
        logger.info("Refreshing database...")
        self._load_database()
        self._build_indexes()
        logger.info("Database refreshed successfully")

    # ========================================================================
    # 30D Metadata Queries
    # ========================================================================

    def search_by_hnr(
        self, min_hnr: float, max_hnr: float, species: Optional[Species] = None
    ) -> List[Tuple[str, Phrase]]:
        """Search phrases by harmonic-to-noise ratio (Grit Factor)

        Args:
            min_hnr: Minimum HNR in dB (e.g., 20.0 for tonal, 2.0 for gritty)
            max_hnr: Maximum HNR in dB
            species: Optional species filter

        Returns:
            List of (phrase_key, Phrase) tuples matching HNR range
        """
        results = []
        for sp, data in self.db.species_data.items():
            if species and sp != species:
                continue

            for phrase_key, phrase in data.phrase_library.items():
                hnr = phrase.acoustic_features.harmonic_to_noise_ratio
                if min_hnr <= hnr <= max_hnr:
                    results.append((phrase_key, phrase))

        results.sort(key=lambda x: x[1].acoustic_features.harmonic_to_noise_ratio, reverse=True)
        return results

    def search_by_spectral_flatness(
        self, min_flatness: float, max_flatness: float, species: Optional[Species] = None
    ) -> List[Tuple[str, Phrase]]:
        """Search phrases by spectral flatness (Grit Factor)

        Args:
            min_flatness: Minimum flatness (0=tonal, 1=noise-like)
            max_flatness: Maximum flatness
            species: Optional species filter

        Returns:
            List of (phrase_key, Phrase) tuples matching flatness range
        """
        results = []
        for sp, data in self.db.species_data.items():
            if species and sp != species:
                continue

            for phrase_key, phrase in data.phrase_library.items():
                flatness = phrase.acoustic_features.spectral_flatness
                if min_flatness <= flatness <= max_flatness:
                    results.append((phrase_key, phrase))

        results.sort(key=lambda x: x[1].acoustic_features.spectral_flatness)
        return results

    def search_by_attack_time(
        self, min_attack: float, max_attack: float, species: Optional[Species] = None
    ) -> List[Tuple[str, Phrase]]:
        """Search phrases by attack time (Motion Factor)

        Args:
            min_attack: Minimum attack time in ms (fast=sharp, slow=gentle)
            max_attack: Maximum attack time in ms
            species: Optional species filter

        Returns:
            List of (phrase_key, Phrase) tuples matching attack time range
        """
        results = []
        for sp, data in self.db.species_data.items():
            if species and sp != species:
                continue

            for phrase_key, phrase in data.phrase_library.items():
                attack = phrase.acoustic_features.attack_time_ms
                if min_attack <= attack <= max_attack:
                    results.append((phrase_key, phrase))

        results.sort(key=lambda x: x[1].acoustic_features.attack_time_ms)
        return results

    def search_by_jitter(
        self, min_jitter: float, max_jitter: float, species: Optional[Species] = None
    ) -> List[Tuple[str, Phrase]]:
        """Search phrases by jitter (Motion Factor - stability/roughness)

        Args:
            min_jitter: Minimum jitter (0=stable, 1=unstable)
            max_jitter: Maximum jitter
            species: Optional species filter

        Returns:
            List of (phrase_key, Phrase) tuples matching jitter range
        """
        results = []
        for sp, data in self.db.species_data.items():
            if species and sp != species:
                continue

            for phrase_key, phrase in data.phrase_library.items():
                jitter = phrase.acoustic_features.jitter
                if min_jitter <= jitter <= max_jitter:
                    results.append((phrase_key, phrase))

        results.sort(key=lambda x: x[1].acoustic_features.jitter)
        return results

    def search_by_onset_rate(
        self, min_rate: float, max_rate: float, species: Optional[Species] = None
    ) -> List[Tuple[str, Phrase]]:
        """Search phrases by onset rate (Rhythm Factor - pulsed vs continuous)

        Args:
            min_rate: Minimum onset rate in Hz (0=continuous, >10=pulsed)
            max_rate: Maximum onset rate in Hz
            species: Optional species filter

        Returns:
            List of (phrase_key, Phrase) tuples matching onset rate range
        """
        results = []
        for sp, data in self.db.species_data.items():
            if species and sp != species:
                continue

            for phrase_key, phrase in data.phrase_library.items():
                rate = phrase.acoustic_features.onset_rate_hz
                if min_rate <= rate <= max_rate:
                    results.append((phrase_key, phrase))

        results.sort(key=lambda x: x[1].acoustic_features.onset_rate_hz, reverse=True)
        return results

    # ========================================================================
    # Acoustic Persona Queries
    # ========================================================================

    def get_pure_persona_phrases(
        self, species: Optional[Species] = None
    ) -> List[Tuple[str, Phrase]]:
        """Get phrases matching PURE persona (tonal, clean, smooth)

        Characteristics: HNR > 20dB, flatness < 0.1, attack > 20ms, jitter < 0.05

        Args:
            species: Optional species filter

        Returns:
            List of (phrase_key, Phrase) tuples matching PURE persona
        """
        results = []
        for sp, data in self.db.species_data.items():
            if species and sp != species:
                continue

            for phrase_key, phrase in data.phrase_library.items():
                af = phrase.acoustic_features
                if (
                    af.harmonic_to_noise_ratio > 20.0
                    and af.spectral_flatness < 0.1
                    and af.attack_time_ms > 20.0
                    and af.jitter < 0.05
                ):
                    results.append((phrase_key, phrase))

        # Sort by HNR (most tonal first)
        results.sort(key=lambda x: x[1].acoustic_features.harmonic_to_noise_ratio, reverse=True)
        return results

    def get_gritty_persona_phrases(
        self, species: Optional[Species] = None
    ) -> List[Tuple[str, Phrase]]:
        """Get phrases matching GRITTY persona (noisy, rough, sharp)

        Characteristics: HNR < 5dB, flatness > 0.6, attack < 5ms, jitter > 0.1

        Args:
            species: Optional species filter

        Returns:
            List of (phrase_key, Phrase) tuples matching GRITTY persona
        """
        results = []
        for sp, data in self.db.species_data.items():
            if species and sp != species:
                continue

            for phrase_key, phrase in data.phrase_library.items():
                af = phrase.acoustic_features
                if (
                    af.harmonic_to_noise_ratio < 5.0
                    and af.spectral_flatness > 0.6
                    and af.attack_time_ms < 5.0
                    and af.jitter > 0.1
                ):
                    results.append((phrase_key, phrase))

        # Sort by grittiness (lowest HNR, highest flatness first)
        results.sort(
            key=lambda x: (
                x[1].acoustic_features.harmonic_to_noise_ratio,
                -x[1].acoustic_features.spectral_flatness,
            )
        )
        return results

    def get_rhythmic_persona_phrases(
        self, species: Optional[Species] = None
    ) -> List[Tuple[str, Phrase]]:
        """Get phrases matching RHYTHMIC persona (pulsed, regular temporal patterns)

        Characteristics: onset rate > 15Hz, ICI CV < 0.3 (regular)

        Args:
            species: Optional species filter

        Returns:
            List of (phrase_key, Phrase) tuples matching RHYTHMIC persona
        """
        results = []
        for sp, data in self.db.species_data.items():
            if species and sp != species:
                continue

            for phrase_key, phrase in data.phrase_library.items():
                af = phrase.acoustic_features
                if (
                    af.onset_rate_hz > 15.0
                    and af.ici_coefficient_of_variation < 0.3
                    and af.ici_coefficient_of_variation > 0
                ):
                    results.append((phrase_key, phrase))

        # Sort by onset rate (most rhythmic first)
        results.sort(key=lambda x: x[1].acoustic_features.onset_rate_hz, reverse=True)
        return results

    def get_harmonic_persona_phrases(
        self, species: Optional[Species] = None
    ) -> List[Tuple[str, Phrase]]:
        """Get phrases matching HARMONIC persona (continuous tones, no pulses)

        Characteristics: onset rate = 0, median ICI = 0

        Args:
            species: Optional species filter

        Returns:
            List of (phrase_key, Phrase) tuples matching HARMONIC persona
        """
        results = []
        for sp, data in self.db.species_data.items():
            if species and sp != species:
                continue

            for phrase_key, phrase in data.phrase_library.items():
                af = phrase.acoustic_features
                if af.onset_rate_hz == 0 and af.median_ici_ms == 0:
                    results.append((phrase_key, phrase))

        # Sort by HNR (most harmonic first)
        results.sort(key=lambda x: x[1].acoustic_features.harmonic_to_noise_ratio, reverse=True)
        return results

    # ========================================================================
    # 30D Nearest Neighbor Search
    # ========================================================================

    def find_nearest_neighbors_17d(
        self, phrase_key: str, k: int = 5, species: Optional[Species] = None
    ) -> List[Tuple[float, str, Phrase]]:
        """Find k nearest neighbors in 30D metadata space

        Uses Euclidean distance over all 17 micro-dynamics features:
        - Fundamental (3): mean_f0_hz, duration_ms, f0_range_hz
        - Grit Factors (2): harmonic_to_noise_ratio, spectral_flatness
        - Motion Factors (6): attack_time_ms, decay_time_ms, sustain_level,
                            vibrato_rate_hz, vibrato_depth, jitter
        - Fingerprint Factors (5): mfcc_1-4, spectral_contrast
        - Rhythm Factors (3): median_ici_ms, onset_rate_hz, ici_coefficient_of_variation

        Args:
            phrase_key: Target phrase key
            k: Number of neighbors to return
            species: Optional species filter

        Returns:
            List of (distance, phrase_key, Phrase) tuples sorted by distance
        """
        # Get target phrase
        target_species, target_phrase = self.get_phrase_by_key(phrase_key)
        if not target_phrase:
            return []

        # Extract 30D feature vector
        def extract_17d(phrase: Phrase) -> List[float]:
            af = phrase.acoustic_features
            return [
                af.mean_f0_hz,
                af.mean_duration_ms,
                af.f0_range_hz,
                af.harmonic_to_noise_ratio,
                af.spectral_flatness,
                af.attack_time_ms,
                af.decay_time_ms,
                af.sustain_level,
                af.vibrato_rate_hz,
                af.vibrato_depth,
                af.jitter,
                af.mfcc_1,
                af.mfcc_2,
                af.mfcc_3,
                af.mfcc_4,
                af.spectral_contrast,
                af.median_ici_ms,
                af.onset_rate_hz,
                af.ici_coefficient_of_variation,
            ]

        target_vector = extract_17d(target_phrase)

        # Calculate distances to all phrases
        from math import sqrt

        distances = []

        for sp, data in self.db.species_data.items():
            if species and sp != species:
                continue

            for pk, phrase in data.phrase_library.items():
                if pk == phrase_key:
                    continue  # Skip self

                vector = extract_17d(phrase)
                # Euclidean distance
                distance = sqrt(sum((t - v) ** 2 for t, v in zip(target_vector, vector)))
                distances.append((distance, pk, phrase))

        # Sort by distance and return top k
        distances.sort(key=lambda x: x[0])
        return distances[:k]

    # ========================================================================
    # 30D Delta Calculation
    # ========================================================================

    def calculate_17d_delta(self, from_phrase_key: str, to_phrase_key: str) -> Dict[str, float]:
        """Calculate 30D delta (difference) between two phrases

        Returns the transformation needed to go from from_phrase to to_phrase
        in the 30D micro-dynamics feature space.

        Args:
            from_phrase_key: Source phrase key
            to_phrase_key: Target phrase key

        Returns:
            Dictionary with 17 delta values (to - from)
        """
        _, from_phrase = self.get_phrase_by_key(from_phrase_key)
        _, to_phrase = self.get_phrase_by_key(to_phrase_key)

        if not from_phrase or not to_phrase:
            raise ValueError(f"Phrase not found: {from_phrase_key} or {to_phrase_key}")

        from_af = from_phrase.acoustic_features
        to_af = to_phrase.acoustic_features

        return {
            # Fundamental deltas
            "delta_mean_f0_hz": to_af.mean_f0_hz - from_af.mean_f0_hz,
            "delta_duration_ms": to_af.mean_duration_ms - from_af.mean_duration_ms,
            "delta_f0_range_hz": to_af.f0_range_hz - from_af.f0_range_hz,
            # Grit Factor deltas
            "delta_harmonic_to_noise_ratio": to_af.harmonic_to_noise_ratio
            - from_af.harmonic_to_noise_ratio,
            "delta_spectral_flatness": to_af.spectral_flatness - from_af.spectral_flatness,
            # Motion Factor deltas
            "delta_attack_time_ms": to_af.attack_time_ms - from_af.attack_time_ms,
            "delta_decay_time_ms": to_af.decay_time_ms - from_af.decay_time_ms,
            "delta_sustain_level": to_af.sustain_level - from_af.sustain_level,
            "delta_vibrato_rate_hz": to_af.vibrato_rate_hz - from_af.vibrato_rate_hz,
            "delta_vibrato_depth": to_af.vibrato_depth - from_af.vibrato_depth,
            "delta_jitter": to_af.jitter - from_af.jitter,
            # Fingerprint Factor deltas
            "delta_mfcc_1": to_af.mfcc_1 - from_af.mfcc_1,
            "delta_mfcc_2": to_af.mfcc_2 - from_af.mfcc_2,
            "delta_mfcc_3": to_af.mfcc_3 - from_af.mfcc_3,
            "delta_mfcc_4": to_af.mfcc_4 - from_af.mfcc_4,
            "delta_spectral_contrast": to_af.spectral_contrast - from_af.spectral_contrast,
            # Rhythm Factor deltas
            "delta_median_ici_ms": to_af.median_ici_ms - from_af.median_ici_ms,
            "delta_onset_rate_hz": to_af.onset_rate_hz - from_af.onset_rate_hz,
            "delta_ici_coefficient_of_variation": to_af.ici_coefficient_of_variation
            - from_af.ici_coefficient_of_variation,
        }

    # =========================================================================
    # Semantic Search Methods (Human-Guided Context Discovery)
    # =========================================================================

    def search_by_semantic_label(
        self,
        label: str,
        min_confidence: float = 0.0,
        species: Optional[Species] = None,
    ) -> List[Tuple[str, Phrase, float]]:
        """
        Search phrases by semantic label from Human-Guided Dictionary.

        Args:
            label: Semantic label to search for (e.g., "Alarm", "Contact", "Phee")
            min_confidence: Minimum confidence threshold (0.0 - 1.0)
            species: Optional species filter

        Returns:
            List of (phrase_key, phrase, confidence) tuples
        """
        results = []

        species_data_iter = (
            [(species, self.db.species_data[species])]
            if species and species in self.db.species_data
            else self.db.species_data.items()
        )

        for sp, sp_data in species_data_iter:
            for phrase_key, phrase in sp_data.phrase_library.items():
                # Check contexts for matching label
                for context in phrase.contexts:
                    if label.lower() in context.context_name.lower():
                        confidence = context.percentage / 100.0  # Convert to 0-1
                        if confidence >= min_confidence:
                            results.append((phrase_key, phrase, confidence))
                            break  # Only add once per phrase

        # Sort by confidence descending
        results.sort(key=lambda x: x[2], reverse=True)
        return results

    def search_by_intent(
        self,
        intent: str,
        species: Optional[Species] = None,
    ) -> List[Tuple[str, Phrase, str]]:
        """
        Search phrases by inferred intent.

        Intent is derived from semantic label + environmental context:
        - "Phee" + "Windy" → "Long_Range_Contact"
        - "Tsik" + "Storm" → "Emergency_Alert"
        - "Twitter" + Any → "Social_Bonding"

        Args:
            intent: Intent to search for (e.g., "Long_Range_Contact", "Warning")
            species: Optional species filter

        Returns:
            List of (phrase_key, phrase, matched_intent) tuples
        """
        # Intent inference rules
        intent_to_labels = {
            "Long_Range_Contact": ["phee", "contact"],
            "Social_Contact": ["phee", "contact"],
            "Emergency_Alert": ["tsik", "alarm", "protest"],
            "Warning": ["tsik", "alarm", "protest"],
            "Social_Bonding": ["twitter", "grooming", "social"],
            "Affiliative": ["trill", "affiliative"],
            "Solicitation": ["infant_cry", "solicitation"],
            "Aggression": ["fighting", "aggression", "protest"],
            "Reproductive": ["mating", "reproductive"],
            "Territorial_Defense": ["fighting", "territorial"],
        }

        # Find matching labels for the intent
        matching_labels = intent_to_labels.get(intent, [intent.lower()])

        results = []
        species_data_iter = (
            [(species, self.db.species_data[species])]
            if species and species in self.db.species_data
            else self.db.species_data.items()
        )

        for sp, sp_data in species_data_iter:
            for phrase_key, phrase in sp_data.phrase_library.items():
                for context in phrase.contexts:
                    context_lower = context.context_name.lower()
                    if any(label in context_lower for label in matching_labels):
                        results.append((phrase_key, phrase, intent))
                        break

        return results

    def search_by_grading_score(
        self,
        min_score: float = 0.0,
        max_score: float = 1.0,
        label: Optional[str] = None,
        species: Optional[Species] = None,
    ) -> List[Tuple[str, Phrase, float]]:
        """
        Search phrases by grading score (discrete vs graded vocalizations).

        Grading score indicates how much a phrase deviates from its type centroid:
        - Low score (~0.0): Discrete calls (consistent, stereotyped)
        - High score (~1.0): Graded calls (variable, continuous)

        Args:
            min_score: Minimum grading score (0.0 = most discrete)
            max_score: Maximum grading score (1.0 = most graded)
            label: Optional semantic label filter
            species: Optional species filter

        Returns:
            List of (phrase_key, phrase, grading_score) tuples
        """
        results = []

        # Get phrases matching label if specified
        if label:
            base_phrases = self.search_by_semantic_label(label, 0.0, species)
            phrase_iter = [(pk, p) for pk, p, _ in base_phrases]
        else:
            species_data_iter = (
                [(species, self.db.species_data[species])]
                if species and species in self.db.species_data
                else self.db.species_data.items()
            )
            phrase_iter = [
                (pk, p)
                for sp, sp_data in species_data_iter
                for pk, p in sp_data.phrase_library.items()
            ]

        for phrase_key, phrase in phrase_iter:
            # Estimate grading score from acoustic variation
            # High jitter/shimmer = more graded; Low = more discrete
            af = phrase.acoustic_features
            if af is None:
                continue

            # Calculate grading proxy from acoustic features
            jitter = getattr(af, 'jitter', 0.0) or 0.0
            spectral_flatness = getattr(af, 'spectral_flatness', 0.0) or 0.0
            f0_range = getattr(af, 'f0_range_hz', 0.0) or 0.0

            # Normalize to 0-1 range (approximate)
            grading_score = min(1.0, (jitter * 5.0 + spectral_flatness + f0_range / 5000.0) / 3.0)

            if min_score <= grading_score <= max_score:
                results.append((phrase_key, phrase, grading_score))

        # Sort by grading score descending (most graded first)
        results.sort(key=lambda x: x[2], reverse=True)
        return results

    def search_semantic_across_species(
        self,
        label: str,
        min_confidence: float = 0.5,
    ) -> Dict[Species, List[Tuple[str, Phrase, float]]]:
        """
        Find similar semantic contexts across species.

        Example: "alarm" type calls across species:
        - Marmoset: Tsik
        - Egyptian Fruit Bat: Protest
        - Bird: Alarm Call

        Args:
            label: Semantic label to search for
            min_confidence: Minimum confidence threshold

        Returns:
            Dict mapping species to list of matching phrases
        """
        results = {}

        for species in self.db.species_data.keys():
            species_results = self.search_by_semantic_label(label, min_confidence, species)
            if species_results:
                results[species] = species_results

        return results

    def get_semantic_labels(self, species: Optional[Species] = None) -> List[str]:
        """
        Get all unique semantic labels in the database.

        Args:
            species: Optional species filter

        Returns:
            Sorted list of unique semantic labels
        """
        labels = set()

        species_data_iter = (
            [(species, self.db.species_data[species])]
            if species and species in self.db.species_data
            else self.db.species_data.items()
        )

        for sp, sp_data in species_data_iter:
            for phrase in sp_data.phrase_library.values():
                for context in phrase.contexts:
                    labels.add(context.context_name)

        return sorted(labels)

    def get_available_intents(self) -> List[str]:
        """
        Get all available intent categories.

        Returns:
            List of intent names that can be searched
        """
        return [
            "Long_Range_Contact",
            "Social_Contact",
            "Emergency_Alert",
            "Warning",
            "Social_Bonding",
            "Affiliative",
            "Solicitation",
            "Aggression",
            "Reproductive",
            "Territorial_Defense",
        ]

    def get_database_info(self) -> Dict[str, Any]:
        """Get database information and statistics"""
        info = {
            "database_path": str(self.database_path),
            "last_loaded": time.strftime("%Y-%m-%d %H:%M:%S"),
            "species_count": len(self.db.species_data),
            "total_phrases": sum(
                len(data.phrase_library) for data in self.db.species_data.values()
            ),
            "total_sentences": sum(len(data.sentences) for data in self.db.species_data.values()),
            "total_grammar_rules": sum(
                len(data.grammar_rules) for data in self.db.species_data.values()
            ),
            "species_available": [species.value for species in self.db.species_data.keys()],
            "modalities_available": [mod.value for mod in VocalizationModality],
        }
        return info


# Global query interface instance
_query_interface = None


def get_query_interface() -> VocalizationQueryInterface:
    """Get global query interface instance"""
    global _query_interface
    if _query_interface is None:
        _query_interface = VocalizationQueryInterface()
    return _query_interface


def query_phrases_by_f0(
    min_f0: float, max_f0: float, species: Optional[Species] = None
) -> List[Tuple[str, Any]]:
    """Convenience function for F0-based queries"""
    interface = get_query_interface()
    return interface.search_phrases_by_f0_range(min_f0, max_f0, species)


def query_phrases_by_duration(
    min_duration: float, max_duration: float, species: Optional[Species] = None
) -> List[Tuple[str, Any]]:
    """Convenience function for duration-based queries"""
    interface = get_query_interface()
    return interface.search_phrases_by_duration(min_duration, max_duration, species)


def get_phrase_similarities(
    phrase_key: str, threshold: float = 0.8
) -> List[Tuple[float, str, Any]]:
    """Convenience function for similarity queries"""
    interface = get_query_interface()
    return interface.get_similar_phrases(phrase_key, threshold)


def get_database_statistics() -> Dict[str, Any]:
    """Convenience function for getting database statistics"""
    interface = get_query_interface()
    return interface.get_database_info()


# =============================================================================
# Semantic Search Convenience Functions
# =============================================================================

def search_by_semantic_label(
    label: str,
    min_confidence: float = 0.0,
    species: Optional[Species] = None,
) -> List[Tuple[str, Phrase, float]]:
    """Convenience function for semantic label queries"""
    interface = get_query_interface()
    return interface.search_by_semantic_label(label, min_confidence, species)


def search_by_intent(
    intent: str,
    species: Optional[Species] = None,
) -> List[Tuple[str, Phrase, str]]:
    """Convenience function for intent-based queries"""
    interface = get_query_interface()
    return interface.search_by_intent(intent, species)


def search_by_grading_score(
    min_score: float = 0.0,
    max_score: float = 1.0,
    label: Optional[str] = None,
    species: Optional[Species] = None,
) -> List[Tuple[str, Phrase, float]]:
    """Convenience function for grading score queries"""
    interface = get_query_interface()
    return interface.search_by_grading_score(min_score, max_score, label, species)


def search_semantic_across_species(
    label: str,
    min_confidence: float = 0.5,
) -> Dict[Species, List[Tuple[str, Phrase, float]]]:
    """Convenience function for cross-species semantic queries"""
    interface = get_query_interface()
    return interface.search_semantic_across_species(label, min_confidence)