"""
Query Interface for Animal Vocalization Database

This module provides efficient query interfaces for accessing vocalization data
in real-time with various filtering, aggregation, and search capabilities.
"""

import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from dataclasses import asdict
import numpy as np
import logging

from data_models import (
    VocalizationDatabase, Species, VocalizationModality,
    Phrase, Sentence, GrammarRule, PhraseContext
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
            with open(self.database_path, 'r') as f:
                data = json.load(f)

            # Reconstruct the database
            self.db = VocalizationDatabase()

            # Import species data
            for species_value, species_data in data['species_data'].items():
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
        from data_models import SpeciesData, Phrase, Sentence, GrammarRule

        species_data_obj = SpeciesData(species=species)

        # Set basic stats
        species_data_obj.total_phrases = species_data.get('total_phrases', 0)
        species_data_obj.total_sentences = species_data.get('total_sentences', 0)
        species_data_obj.total_grammar_rules = species_data.get('total_grammar_rules', 0)
        species_data_obj.vocabulary_size = species_data.get('vocabulary_size', 0)

        # Import phrases
        for phrase_key, phrase_data in species_data['phrases'].items():
            acoustic_features = self._create_acoustic_features_from_json(phrase_data['acoustic_features'])

            phrase = Phrase(
                phrase_key=phrase_key,
                signature=phrase_data['signature'],
                species=Species(phrase_data['species']),
                modality=VocalizationModality(phrase_data['modality']),
                acoustic_features=acoustic_features,
                total_occurrences=phrase_data['total_occurrences'],
                contexts=[PhraseContext(ctx['context_name'], ctx['count'], ctx.get('percentage', 0)) for ctx in phrase_data.get('contexts', [])],
                social_contexts=phrase_data.get('social_contexts', {}),
                is_compositional=phrase_data.get('is_compositional', False),
                phrase_components=phrase_data.get('phrase_components', [])
            )
            species_data_obj.add_phrase(phrase)

        # Import sentences
        for sentence_data in species_data['sentences']:
            sentence = Sentence(
                sentence_id=sentence_data['sentence_id'],
                species=Species(sentence_data['species']),
                phrase_sequence=sentence_data['phrase_sequence'],
                context=sentence_data['context'],
                has_ascending_syntax=sentence_data['has_ascending_syntax'],
                has_descending_syntax=sentence_data['has_descending_syntax'],
                has_fm_pattern=sentence_data['has_fm_pattern'],
                syntax_score=sentence_data['syntax_score'],
                total_duration_ms=sentence_data['total_duration_ms'],
                complexity_score=sentence_data['complexity_score']
            )
            species_data_obj.sentences.append(sentence)

        # Import grammar rules
        for rule_data in species_data['grammar_rules']:
            rule = GrammarRule(
                rule_id=rule_data['rule_id'],
                from_phrase=rule_data['from_phrase'],
                to_phrase=rule_data['to_phrase'],
                frequency=rule_data['frequency'],
                confidence=rule_data['confidence'],
                contexts=rule_data['contexts'],
                is_bidirectional=rule_data['is_bidirectional'],
                strength_score=rule_data['strength_score']
            )
            species_data_obj.grammar_rules.append(rule)

        
        return species_data_obj

    def _create_acoustic_features_from_json(self, features_data: Dict) -> Any:
        """Create AcousticFeatures object from JSON data"""
        from data_models import AcousticFeatures

        return AcousticFeatures(
            mean_f0_hz=features_data['mean_f0_hz'],
            std_f0_hz=features_data['std_f0_hz'],
            min_f0_hz=features_data['min_f0_hz'],
            max_f0_hz=features_data['max_f0_hz'],
            f0_range_hz=features_data['f0_range_hz'],
            duration_frames=features_data['duration_frames'],
            voiced_ratio=features_data['voiced_ratio'],
            f0_slope=features_data['f0_slope'],
            modulation_rate=features_data['modulation_rate'],
            acoustic_variance=features_data['acoustic_variance'],
            mean_duration_ms=features_data['mean_duration_ms']
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

    def search_phrases_by_f0_range(self, min_f0: float, max_f0: float,
                                 species: Optional[Species] = None) -> List[Tuple[str, Phrase]]:
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

    def search_phrases_by_duration(self, min_duration: float, max_duration: float,
                                 species: Optional[Species] = None) -> List[Tuple[str, Phrase]]:
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

    def get_similar_phrases(self, phrase_key: str, threshold: float = 0.8) -> List[Tuple[float, str, Phrase]]:
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
            similarity = (f0_similarity * 0.5 + duration_similarity * 0.3 + range_similarity * 0.2)

            if similarity >= threshold:
                similarities.append((similarity, key, phrase))

        return sorted(similarities, reverse=True)

    def get_phrase_statistics(self, species: Optional[Species] = None) -> Dict[str, Any]:
        """Get comprehensive statistics about phrases"""
        stats = {
            'total_phrases': 0,
            'species_breakdown': {},
            'modality_breakdown': {},
            'frequency_distribution': {'min': float('inf'), 'max': 0, 'avg': 0},
            'duration_distribution': {'min': float('inf'), 'max': 0, 'avg': 0},
            'context_distribution': {},
            'semantic_coverage': 0
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
            if species_name not in stats['species_breakdown']:
                stats['species_breakdown'][species_name] = 0
            stats['species_breakdown'][species_name] += 1

            # Modality breakdown
            modality = phrase.modality.value
            if modality not in stats['modality_breakdown']:
                stats['modality_breakdown'][modality] = 0
            stats['modality_breakdown'][modality] += 1

            # Frequency distribution
            f0 = phrase.acoustic_features.mean_f0_hz
            stats['frequency_distribution']['min'] = min(stats['frequency_distribution']['min'], f0)
            stats['frequency_distribution']['max'] = max(stats['frequency_distribution']['max'], f0)
            total_f0 += f0

            # Duration distribution
            duration = phrase.acoustic_features.mean_duration_ms
            stats['duration_distribution']['min'] = min(stats['duration_distribution']['min'], duration)
            stats['duration_distribution']['max'] = max(stats['duration_distribution']['max'], duration)
            total_duration += duration

            # Context distribution
            for ctx in phrase.contexts:
                ctx_name = ctx.context_name
                if ctx_name not in stats['context_distribution']:
                    stats['context_distribution'][ctx_name] = 0
                stats['context_distribution'][ctx_name] += ctx.count

            phrase_count += 1

        # Calculate averages
        if phrase_count > 0:
            stats['frequency_distribution']['avg'] = total_f0 / phrase_count
            stats['duration_distribution']['avg'] = total_duration / phrase_count
            stats['total_phrases'] = phrase_count

        return stats

    def get_grammar_network(self, species: Optional[Species] = None) -> Dict[str, Any]:
        """Get grammar network statistics"""
        network = {
            'nodes': 0,
            'edges': 0,
            'transitions': {},
            'most_connected_phrases': [],
            'strongest_transitions': []
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
        network['nodes'] = len(phrase_connections)
        network['edges'] = len(transitions_to_analyze)
        network['transitions'] = transition_map

        # Most connected phrases
        network['most_connected_phrases'] = sorted(
            phrase_connections.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        # Strongest transitions
        all_transitions = []
        for from_phrase, to_phrases in transition_map.items():
            for to_phrase, frequency in to_phrases.items():
                all_transitions.append((from_phrase, to_phrase, frequency))

        network['strongest_transitions'] = sorted(
            all_transitions,
            key=lambda x: x[2],
            reverse=True
        )[:10]

        return network

    def export_query_results(self, results: List[Any], format: str = 'json') -> str:
        """Export query results in specified format"""
        if format == 'json':
            return json.dumps([asdict(item) if hasattr(item, '__dict__') else item
                             for item in results], indent=2)
        elif format == 'csv':
            import csv
            import io
            output = io.StringIO()
            writer = csv.writer(output)

            # Simple CSV export for phrase results
            if results and isinstance(results[0], tuple) and len(results[0]) >= 2:
                writer.writerow(['Phrase Key', 'Species', 'Mean F0 (Hz)', 'Duration (ms)', 'Occurrences'])
                for phrase_key, phrase in results:
                    writer.writerow([
                        phrase_key,
                        phrase.species.value,
                        phrase.acoustic_features.mean_f0_hz,
                        phrase.acoustic_features.mean_duration_ms,
                        phrase.total_occurrences
                    ])

            return output.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format}")

    def refresh_database(self):
        """Reload database from file"""
        logger.info("Refreshing database...")
        self._load_database()
        self._build_indexes()
        logger.info("Database refreshed successfully")

    def get_database_info(self) -> Dict[str, Any]:
        """Get database information and statistics"""
        info = {
            'database_path': str(self.database_path),
            'last_loaded': time.strftime('%Y-%m-%d %H:%M:%S'),
            'species_count': len(self.db.species_data),
            'total_phrases': sum(len(data.phrase_library) for data in self.db.species_data.values()),
            'total_sentences': sum(len(data.sentences) for data in self.db.species_data.values()),
            'total_grammar_rules': sum(len(data.grammar_rules) for data in self.db.species_data.values()),
            'species_available': [species.value for species in self.db.species_data.keys()],
            'modalities_available': [mod.value for mod in VocalizationModality]
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


def query_phrases_by_f0(min_f0: float, max_f0: float,
                       species: Optional[Species] = None) -> List[Tuple[str, Any]]:
    """Convenience function for F0-based queries"""
    interface = get_query_interface()
    return interface.search_phrases_by_f0_range(min_f0, max_f0, species)


def query_phrases_by_duration(min_duration: float, max_duration: float,
                             species: Optional[Species] = None) -> List[Tuple[str, Any]]:
    """Convenience function for duration-based queries"""
    interface = get_query_interface()
    return interface.search_phrases_by_duration(min_duration, max_duration, species)


def get_phrase_similarities(phrase_key: str, threshold: float = 0.8) -> List[Tuple[float, str, Any]]:
    """Convenience function for similarity queries"""
    interface = get_query_interface()
    return interface.get_similar_phrases(phrase_key, threshold)


def get_database_statistics() -> Dict[str, Any]:
    """Convenience function for getting database statistics"""
    interface = get_query_interface()
    return interface.get_database_info()