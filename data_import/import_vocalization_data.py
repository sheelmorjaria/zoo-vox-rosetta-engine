"""
Import Script for Animal Vocalization Analysis Data

This script imports phrase, sentence, and grammar data from analysis results
into the unified data models for production system usage.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import logging

from data_models import (
    VocalizationDatabase, SpeciesData, Phrase, Sentence, GrammarRule,
    AcousticFeatures, PhraseContext, PhraseOccurrence,
    Species, VocalizationModality
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataImporter:
    """Handles importing vocalization data into the database"""

    def __init__(self):
        self.db = VocalizationDatabase()
        self.import_stats = {
            'total_phrases': 0,
            'total_sentences': 0,
            'total_grammar_rules': 0,
            'total_semantic_mappings': 0,
            'species_processed': []
        }

    def import_marmoset_data(self, filepath: str) -> SpeciesData:
        """Import marmoset phrase library data"""
        logger.info(f"Importing marmoset data from {filepath}")

        with open(filepath, 'r') as f:
            data = json.load(f)

        species_data = SpeciesData(species=Species.MARMOSET)
        species_data.analysis_date = datetime.fromtimestamp(os.path.getmtime(filepath))

        # Import phrases
        for phrase_key, phrase_data in data.items():
            # Parse acoustic features
            acoustic_features = AcousticFeatures(
                mean_f0_hz=phrase_data['mean_f0_hz'],
                std_f0_hz=phrase_data.get('std_f0_hz', 0.0),
                min_f0_hz=phrase_data.get('min_f0_hz', 0.0),
                max_f0_hz=phrase_data.get('max_f0_hz', 0.0),
                f0_range_hz=phrase_data['mean_range_hz'],
                mean_duration_ms=phrase_data['mean_duration_ms'],
                duration_frames=int(phrase_data.get('duration_frames', 0)),
                voiced_ratio=phrase_data['voiced_ratio'],
                f0_slope=phrase_data.get('f0_slope', 0.0)
            )

            # Parse contexts
            contexts = []
            context_counts = phrase_data.get('contexts', {})
            for ctx_name, count in context_counts.items():
                contexts.append(PhraseContext(ctx_name, count))

            # Parse occurrences
            occurrences = []
            for occ_data in phrase_data.get('occurrences', []):
                # Parse acoustic features for occurrence
                occ_acoustic = AcousticFeatures(
                    mean_f0_hz=occ_data['mean_f0_hz'],
                    std_f0_hz=occ_data['std_f0_hz'],
                    min_f0_hz=occ_data['min_f0_hz'],
                    max_f0_hz=occ_data['max_f0_hz'],
                    f0_range_hz=occ_data['f0_range_hz'],
                    mean_duration_ms=occ_data['duration_ms'],
                    duration_frames=int(occ_data.get('duration_frames', 0)),
                    voiced_ratio=occ_data.get('voiced_ratio', 0.0)
                )

                occurrence = PhraseOccurrence(
                    phrase_key=occ_data['phrase_key'],
                    f0_values=occ_data['f0_values'],
                    acoustic_features=occ_acoustic,
                    source_file=occ_data['source_file'],
                    source_path=occ_data['source_path'],
                    context=occ_data['context']
                )
                occurrences.append(occurrence)

            # Create phrase object
            phrase = Phrase(
                phrase_key=phrase_key,
                signature=phrase_key,
                species=Species.MARMOSET,
                modality=VocalizationModality.HARMONIC,
                acoustic_features=acoustic_features,
                total_occurrences=phrase_data['total_occurrences'],
                contexts=contexts,
                occurrences=occurrences,
                social_contexts=phrase_data.get('social_contexts', {})
            )

            species_data.add_phrase(phrase)

        logger.info(f"Imported {species_data.total_phrases} marmoset phrases")
        self.import_stats['total_phrases'] += species_data.total_phrases
        self.import_stats['species_processed'].append('marmoset')

        return species_data

    def import_egyptian_bat_data(self, filepath: str) -> SpeciesData:
        """Import Egyptian fruit bat phrase library data"""
        logger.info(f"Importing Egyptian bat data from {filepath}")

        with open(filepath, 'r') as f:
            data = json.load(f)

        species_data = SpeciesData(species=Species.EGYPTIAN_BAT)
        species_data.analysis_date = datetime.fromtimestamp(os.path.getmtime(filepath))

        # Import phrases
        for phrase_key, phrase_data in data.items():
            # Parse acoustic features
            acoustic_features = AcousticFeatures(
                mean_f0_hz=phrase_data['mean_f0_hz'],
                std_f0_hz=phrase_data.get('std_f0_hz', 0.0),
                min_f0_hz=phrase_data.get('min_f0_hz', 0.0),
                max_f0_hz=phrase_data.get('max_f0_hz', 0.0),
                f0_range_hz=phrase_data['mean_range_hz'],
                mean_duration_ms=phrase_data['mean_duration_ms'],
                duration_frames=int(phrase_data.get('duration_frames', 0)),
                voiced_ratio=phrase_data.get('voiced_ratio', 0.0),
                modulation_rate=phrase_data.get('mean_modulation_rate', 0.0)
            )

            # Parse contexts
            contexts = []
            context_counts = phrase_data.get('contexts', {})
            for ctx_name, count in context_counts.items():
                contexts.append(PhraseContext(ctx_name, count))

            # Parse occurrences
            occurrences = []
            for occ_data in phrase_data.get('occurrences', []):
                occ_acoustic = AcousticFeatures(
                    mean_f0_hz=occ_data['mean_f0_hz'],
                    std_f0_hz=occ_data['std_f0_hz'],
                    min_f0_hz=occ_data['min_f0_hz'],
                    max_f0_hz=occ_data['max_f0_hz'],
                    f0_range_hz=occ_data['f0_range_hz'],
                    mean_duration_ms=occ_data['duration_ms'],
                    duration_frames=int(occ_data.get('duration_frames', 0)),
                    voiced_ratio=occ_data.get('voiced_ratio', 0.0)
                )

                occurrence = PhraseOccurrence(
                    phrase_key=occ_data['phrase_key'],
                    f0_values=occ_data['f0_values'],
                    acoustic_features=occ_acoustic,
                    source_file=occ_data['source_file'],
                    source_path=occ_data['source_path'],
                    context=occ_data['context']
                )
                occurrences.append(occurrence)

            # Create phrase object
            phrase = Phrase(
                phrase_key=phrase_key,
                signature=phrase_key,
                species=Species.EGYPTIAN_BAT,
                modality=VocalizationModality.FM_SWEEP,
                acoustic_features=acoustic_features,
                total_occurrences=phrase_data['total_occurrences'],
                contexts=contexts,
                occurrences=occurrences,
                social_contexts=phrase_data.get('social_contexts', {})
            )

            species_data.add_phrase(phrase)

        logger.info(f"Imported {species_data.total_phrases} Egyptian bat phrases")
        self.import_stats['total_phrases'] += species_data.total_phrases
        self.import_stats['species_processed'].append('egyptian_bat')

        return species_data

    def import_dolphin_data(self, filepath: str) -> SpeciesData:
        """Import dolphin phrase library data"""
        logger.info(f"Importing dolphin data from {filepath}")

        with open(filepath, 'r') as f:
            data = json.load(f)

        species_data = SpeciesData(species=Species.DOLPHIN)
        species_data.analysis_date = datetime.fromtimestamp(os.path.getmtime(filepath))

        # Import phrases
        for phrase_key, phrase_data in data.items():
            # Parse acoustic features
            acoustic_features = AcousticFeatures(
                mean_f0_hz=phrase_data['mean_f0_hz'],
                std_f0_hz=phrase_data.get('std_f0_hz', 0.0),
                min_f0_hz=phrase_data.get('min_f0_hz', 0.0),
                max_f0_hz=phrase_data.get('max_f0_hz', 0.0),
                f0_range_hz=phrase_data['mean_range_hz'],
                mean_duration_ms=phrase_data['mean_duration_ms'],
                duration_frames=int(phrase_data.get('duration_frames', 0)),
                voiced_ratio=phrase_data.get('voiced_ratio', 0.0),
                f0_slope=phrase_data.get('mean_slope', 0.0)
            )

            # Create phrase object
            phrase = Phrase(
                phrase_key=phrase_key,
                signature=phrase_key,
                species=Species.DOLPHIN,
                modality=VocalizationModality.WHISTLE,
                acoustic_features=acoustic_features,
                total_occurrences=phrase_data['total_occurrences'],
                contexts=[],  # Will be populated if needed
                social_contexts=phrase_data.get('social_contexts', {})
            )

            species_data.add_phrase(phrase)

        logger.info(f"Imported {species_data.total_phrases} dolphin phrases")
        self.import_stats['total_phrases'] += species_data.total_phrases
        self.import_stats['species_processed'].append('dolphin')

        return species_data

    def import_chimpanzee_data(self, filepath: str) -> SpeciesData:
        """Import chimpanzee phrase library data"""
        logger.info(f"Importing chimpanzee data from {filepath}")

        with open(filepath, 'r') as f:
            data = json.load(f)

        species_data = SpeciesData(species=Species.CHIMPANZEE)
        species_data.analysis_date = datetime.fromtimestamp(os.path.getmtime(filepath))

        # Import phrases
        for phrase_key, phrase_data in data.items():
            # Parse acoustic features
            acoustic_features = AcousticFeatures(
                mean_f0_hz=phrase_data['mean_f0_hz'],
                std_f0_hz=phrase_data.get('std_f0_hz', 0.0),
                min_f0_hz=phrase_data.get('min_f0_hz', 0.0),
                max_f0_hz=phrase_data.get('max_f0_hz', 0.0),
                f0_range_hz=phrase_data['mean_range_hz'],
                mean_duration_ms=phrase_data['mean_duration_ms'],
                duration_frames=int(phrase_data.get('duration_frames', 0)),
                voiced_ratio=phrase_data.get('voiced_ratio', 0.0)
            )

            # Create phrase object
            phrase = Phrase(
                phrase_key=phrase_key,
                signature=phrase_key,
                species=Species.CHIMPANZEE,
                modality=VocalizationModality.HARMONIC,
                acoustic_features=acoustic_features,
                total_occurrences=phrase_data['total_occurrences'],
                contexts=[],  # Will be populated if needed
                social_contexts=phrase_data.get('social_contexts', {})
            )

            species_data.add_phrase(phrase)

        logger.info(f"Imported {species_data.total_phrases} chimpanzee phrases")
        self.import_stats['total_phrases'] += species_data.total_phrases
        self.import_stats['species_processed'].append('chimpanzee')

        return species_data

    
    def import_sentence_data(self, filepath: str, species: Species):
        """Import sentence clustering data"""
        logger.info(f"Importing sentence data from {filepath}")

        with open(filepath, 'r') as f:
            data = json.load(f)

        species_data = self.db.get_species_data(species)
        if not species_data:
            logger.warning(f"No species data found for {species}")
            return

        sentences_data = data.get('sentences', {})
        for cluster_id, sentences in sentences_data.items():
            for sentence_data in sentences:
                sentence = Sentence(
                    sentence_id=f"{species.value}_sentence_{sentence_data.get('sentence_id', cluster_id)}",
                    species=species,
                    phrase_sequence=sentence_data.get('phrase_sequence', []),
                    context=sentence_data.get('context', 'unknown'),
                    complexity_score=sentence_data.get('complexity_score', 0.0)
                )
                species_data.sentences.append(sentence)

        logger.info(f"Imported {len(data.get('sentences', {}))} sentence clusters for {species.value}")
        self.import_stats['total_sentences'] += len(data.get('sentences', {}))

    def import_grammar_rules(self, filepath: str, species: Species):
        """Import grammar transition rules"""
        logger.info(f"Importing grammar rules from {filepath}")

        with open(filepath, 'r') as f:
            data = json.load(f)

        species_data = self.db.get_species_data(species)
        if not species_data:
            logger.warning(f"No species data found for {species}")
            return

        grammar_data = data.get('grammar_rules', {})
        for from_phrase, transitions in grammar_data.items():
            for to_phrase, frequency in transitions.items():
                rule = GrammarRule(
                    rule_id=f"{species.value}_grammar_{from_phrase}_{to_phrase}",
                    from_phrase=from_phrase,
                    to_phrase=to_phrase,
                    frequency=frequency,
                    confidence=1.0  # Default confidence
                )
                species_data.grammar_rules.append(rule)

        logger.info(f"Imported {len(data.get('grammar_rules', {}))} grammar rules for {species.value}")
        self.import_stats['total_grammar_rules'] += len(data.get('grammar_rules', {}))

    def import_all_data(self, base_path: str):
        """Import all available data from the base path"""
        base_path = Path(base_path)
        logger.info(f"Starting import from {base_path}")

        # Import marmoset data
        marmoset_file = base_path / "validation_output" / "marmoset_phrase_library.json"
        if marmoset_file.exists():
            marmoset_data = self.import_marmoset_data(str(marmoset_file))
            self.db.add_species_data(marmoset_data)

            # Import additional marmoset data
            sentences_file = base_path / "validation_output" / "marmoset_sentences.json"
            if sentences_file.exists():
                self.import_sentence_data(str(sentences_file), Species.MARMOSET)

        # Import Egyptian bat data
        bat_file = base_path / "validation_output" / "egyptian_bat_phrase_library.json"
        if bat_file.exists():
            bat_data = self.import_egyptian_bat_data(str(bat_file))
            self.db.add_species_data(bat_data)

            # Import additional bat data
            sentences_file = base_path / "validation_output" / "egyptian_bat_sentences.json"
            if sentences_file.exists():
                self.import_sentence_data(str(sentences_file), Species.EGYPTIAN_BAT)

        # Import dolphin data
        dolphin_file = base_path / "validation_output" / "dolphin_phrase_library.json"
        if dolphin_file.exists():
            dolphin_data = self.import_dolphin_data(str(dolphin_file))
            self.db.add_species_data(dolphin_data)

            
            # Import dolphin grammar
            grammar_file = base_path / "validation_output" / "dolphin_grammar.json"
            if grammar_file.exists():
                self.import_grammar_rules(str(grammar_file), Species.DOLPHIN)

        # Import chimpanzee data
        chimp_file = base_path / "validation_output" / "chimpanzee_phrase_library.json"
        if chimp_file.exists():
            chimp_data = self.import_chimpanzee_data(str(chimp_file))
            self.db.add_species_data(chimp_data)

            # Import chimpanzee grammar
            grammar_file = base_path / "validation_output" / "chimpanzee_grammar.json"
            if grammar_file.exists():
                self.import_grammar_rules(str(grammar_file), Species.CHIMPANZEE)

        # Generate summary statistics
        self._generate_summary()

    def _generate_summary(self):
        """Generate and display import summary"""
        logger.info("\n" + "="*50)
        logger.info("IMPORT SUMMARY")
        logger.info("="*50)

        logger.info(f"Total phrases imported: {self.import_stats['total_phrases']}")
        logger.info(f"Total sentences imported: {self.import_stats['total_sentences']}")
        logger.info(f"Total grammar rules imported: {self.import_stats['total_grammar_rules']}")
        logger.info(f"Species processed: {', '.join(self.import_stats['species_processed'])}")

        logger.info("\nSpecies breakdown:")
        for species, species_data in self.db.species_data.items():
            logger.info(f"  {species.value}: {species_data.total_phrases} phrases, "
                       f"{len(species_data.sentences)} sentences, "
                       f"{len(species_data.grammar_rules)} grammar rules")

        # Find cross-species patterns
        patterns = self.db.find_cross_species_patterns()
        if patterns['common_phrase_types']:
            logger.info(f"\nCross-species patterns found: {len(patterns['common_phrase_types'])}")
            for phrase, species_list in patterns['common_phrase_types'][:5]:  # Show top 5
                logger.info(f"  {phrase}: {', '.join([s.value for s in species_list])}")

    def save_database(self, output_path: str):
        """Save the populated database to file"""
        logger.info(f"Saving database to {output_path}")
        self.db.export_to_json(output_path)
        logger.info("Database saved successfully")


def main():
    """Main function to run the import process"""
    # Set paths
    base_path = Path(".")
    import_output = Path("./src/vocalization_database.json")

    # Create importer and import data
    importer = DataImporter()
    importer.import_all_data(str(base_path))

    # Save the database
    importer.save_database(str(import_output))

    print(f"\nImport complete! Database saved to: {import_output}")


if __name__ == "__main__":
    main()