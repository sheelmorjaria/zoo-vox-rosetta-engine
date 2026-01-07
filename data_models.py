"""
Unified Data Models for Animal Vocalization Analysis

This module provides standardized data structures for importing and managing
phrase, sentence, and grammar data across multiple species.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class Species(Enum):
    """Supported species for vocalization analysis"""

    MARMOSET = "marmoset"
    EGYPTIAN_BAT = "egyptian_bat"
    DOLPHIN = "dolphin"
    CHIMPANZEE = "chimpanzee"
    SPERM_WHALE = "sperm_whale"
    ZEBRA_FINCH = "zebra_finch"


class VocalizationModality(Enum):
    """Types of vocalization modalities"""

    HARMONIC = "harmonic"
    FM_SWEEP = "fm_sweep"
    TRANSIENT = "transient"
    RHYTHMIC = "rhythmic"
    WHISTLE = "whistle"
    TEMPORAL = "temporal"


@dataclass
class AcousticFeatures:
    """
    29-dimensional acoustic feature vector (expanded from 17D/20D)

    NOTE: Named AcousticFeatures for backwards compatibility. Now contains 29 fields:
    - 3 Fundamental features (F0, duration)
    - 3 Grit factors (HNR, flatness, harmonicity)
    - 7 Motion factors (attack, decay, sustain, vibrato, jitter, shimmer)
    - 13 MFCC coefficients (expanded from 4) for formant/timbre analysis
    - 1 Spectral contrast
    - 1 Spectral flux
    - 3 Rhythm factors (ICI, onset rate)

    Total: 3 + 3 + 7 + 13 + 1 + 1 + 3 = 31 fields (including some legacy fields)
    """

    # === Fundamental (3 features) ===
    mean_f0_hz: float
    duration_ms: float
    f0_range_hz: float

    # === Grit Factors (3 features) ===
    harmonic_to_noise_ratio: float = 0.0
    spectral_flatness: float = 0.0
    harmonicity: float = 0.0  # NEW: Degree of periodicity vs noise

    # === Motion Factors (7 features) ===
    attack_time_ms: float = 0.0
    decay_time_ms: float = 0.0
    sustain_level: float = 0.0
    vibrato_rate_hz: float = 0.0
    vibrato_depth: float = 0.0
    jitter: float = 0.0
    shimmer: float = 0.0  # NEW: Amplitude instability

    # === Fingerprint Factors (13 features) - Expanded from 4 ===
    mfcc_1: float = 0.0
    mfcc_2: float = 0.0
    mfcc_3: float = 0.0
    mfcc_4: float = 0.0
    mfcc_5: float = 0.0  # NEW
    mfcc_6: float = 0.0  # NEW
    mfcc_7: float = 0.0  # NEW
    mfcc_8: float = 0.0  # NEW
    mfcc_9: float = 0.0  # NEW
    mfcc_10: float = 0.0  # NEW
    mfcc_11: float = 0.0  # NEW
    mfcc_12: float = 0.0  # NEW
    mfcc_13: float = 0.0  # NEW
    spectral_contrast: float = 0.0

    # === Spectral Dynamics (1 feature) ===
    spectral_flux: float = 0.0  # NEW: Rate of spectral change

    # === Rhythm Factors (3 features) ===
    median_ici_ms: float = 0.0
    onset_rate_hz: float = 0.0
    ici_coefficient_of_variation: float = 0.0

    # === Legacy features (for backward compatibility) ===
    std_f0_hz: float = 0.0
    min_f0_hz: float = 0.0
    max_f0_hz: float = 0.0
    duration_frames: int = 0
    voiced_ratio: float = 0.0
    f0_slope: float = 0.0
    modulation_rate: float = 0.0
    acoustic_variance: float = 0.0
    mean_duration_ms: float = 0.0  # Alias for duration_ms
    spectral_centroid_hz: float = 0.0
    spectral_slope: float = 0.0
    spectral_bandwidth_hz: float = 0.0
    spectral_rolloff_hz: float = 0.0
    mfcc_delta_mean: float = 0.0


# Alias for test compatibility
PhraseSignature = AcousticFeatures


@dataclass
class PhraseOccurrence:
    """Individual occurrence of a phrase"""

    phrase_key: str
    f0_values: str  # JSON string of array
    acoustic_features: AcousticFeatures
    source_file: str
    source_path: str
    context: str
    timestamp: Optional[datetime] = None
    individual_id: Optional[str] = None


@dataclass
class PhraseContext:
    """Context distribution for a phrase"""

    context_name: str
    count: int
    percentage: float = 0.0


@dataclass
class Phrase:
    """Complete phrase definition with metadata"""

    phrase_key: str
    signature: str
    species: Species
    modality: VocalizationModality
    acoustic_features: AcousticFeatures
    total_occurrences: int
    contexts: List[PhraseContext] = field(default_factory=list)
    occurrences: List[PhraseOccurrence] = field(default_factory=list)
    acoustic_variations: List[Dict[str, Any]] = field(default_factory=list)
    social_contexts: Dict[str, int] = field(default_factory=dict)
    related_phrases: List[str] = field(default_factory=list)
    is_compositional: bool = False
    phrase_components: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Calculate derived fields"""
        if self.contexts:
            total = sum(ctx.count for ctx in self.contexts)
            for ctx in self.contexts:
                ctx.percentage = (ctx.count / total) * 100 if total > 0 else 0


@dataclass
class Sentence:
    """Sentence composed of multiple phrases"""

    sentence_id: str
    species: Species
    phrase_sequence: List[str]  # List of phrase keys
    context: str
    has_ascending_syntax: bool = False
    has_descending_syntax: bool = False
    has_fm_pattern: bool = False
    syntax_score: float = 0.0
    total_duration_ms: float = 0.0
    complexity_score: float = 0.0
    social_context: Optional[Dict[str, Any]] = None


@dataclass
class GrammarRule:
    """Grammar transition rule between phrases"""

    rule_id: str
    from_phrase: str
    to_phrase: str
    frequency: int
    confidence: float = 0.0
    contexts: List[str] = field(default_factory=list)
    is_bidirectional: bool = False
    strength_score: float = 0.0


@dataclass
class SpeciesData:
    """Complete vocalization data for a species"""

    species: Species
    phrase_library: Dict[str, Phrase] = field(default_factory=dict)
    sentences: List[Sentence] = field(default_factory=list)
    grammar_rules: List[GrammarRule] = field(default_factory=list)
    total_phrases: int = 0
    total_sentences: int = 0
    total_grammar_rules: int = 0
    analysis_date: datetime = field(default_factory=datetime.now)
    vocabulary_size: int = 0
    modality_distribution: Dict[VocalizationModality, int] = field(default_factory=dict)

    def add_phrase(self, phrase: Phrase):
        """Add a phrase to the species library"""
        self.phrase_library[phrase.phrase_key] = phrase
        self.total_phrases = len(self.phrase_library)
        self.vocabulary_size = len(set(p.phrase_key for p in self.phrase_library.values()))

        # Update modality distribution
        modality = phrase.modality
        self.modality_distribution[modality] = self.modality_distribution.get(modality, 0) + 1

    def get_phrase_by_key(self, phrase_key: str) -> Optional[Phrase]:
        """Retrieve phrase by key"""
        return self.phrase_library.get(phrase_key)

    def get_sentences_with_phrase(self, phrase_key: str) -> List[Sentence]:
        """Find all sentences containing a specific phrase"""
        return [sentence for sentence in self.sentences if phrase_key in sentence.phrase_sequence]


class VocalizationDatabase:
    """Unified database for cross-species vocalization data"""

    def __init__(self):
        self.species_data: Dict[Species, SpeciesData] = {}
        self.cross_species_mappings: Dict[str, List[str]] = {}
        self.universal_grammar_patterns: List[Dict[str, Any]] = []

    def add_species_data(self, species_data: SpeciesData):
        """Add data for a species"""
        self.species_data[species_data.species] = species_data

    def get_species_data(self, species: Species) -> Optional[SpeciesData]:
        """Get data for a specific species"""
        return self.species_data.get(species)

    def find_cross_species_patterns(self) -> Dict[str, Any]:
        """Find patterns common across species"""
        patterns = {
            "common_phrase_types": [],
            "shared_grammar_rules": [],
            "universal_modalities": [],
            "species_specific_features": {},
        }

        # Find common phrase patterns
        all_phrases = {}
        for species, data in self.species_data.items():
            for phrase_key in data.phrase_library.keys():
                if phrase_key not in all_phrases:
                    all_phrases[phrase_key] = []
                all_phrases[phrase_key].append(species)

        patterns["common_phrase_types"] = [
            (phrase, species_list)
            for phrase, species_list in all_phrases.items()
            if len(species_list) > 1
        ]

        return patterns

    def export_to_json(self, filepath: str):
        """Export database to JSON"""
        export_data = {"export_date": datetime.now().isoformat(), "species_data": {}}

        for species, data in self.species_data.items():
            species_dict = {
                "species": species.value,
                "analysis_date": data.analysis_date.isoformat(),
                "total_phrases": data.total_phrases,
                "total_sentences": data.total_sentences,
                "vocabulary_size": data.vocabulary_size,
                "modality_distribution": {
                    mod.value: count for mod, count in data.modality_distribution.items()
                },
                "phrases": {},
                "sentences": [],
                "grammar_rules": [],
            }

            # Export phrases
            for phrase_key, phrase in data.phrase_library.items():
                phrase_dict = {
                    "phrase_key": phrase.phrase_key,
                    "signature": phrase.signature,
                    "species": phrase.species.value,
                    "modality": phrase.modality.value,
                    "acoustic_features": {
                        "mean_f0_hz": phrase.acoustic_features.mean_f0_hz,
                        "std_f0_hz": phrase.acoustic_features.std_f0_hz,
                        "min_f0_hz": phrase.acoustic_features.min_f0_hz,
                        "max_f0_hz": phrase.acoustic_features.max_f0_hz,
                        "f0_range_hz": phrase.acoustic_features.f0_range_hz,
                        "mean_duration_ms": phrase.acoustic_features.mean_duration_ms,
                        "duration_frames": phrase.acoustic_features.duration_frames,
                        "voiced_ratio": phrase.acoustic_features.voiced_ratio,
                        "f0_slope": phrase.acoustic_features.f0_slope,
                        "modulation_rate": phrase.acoustic_features.modulation_rate,
                        "acoustic_variance": phrase.acoustic_features.acoustic_variance,
                        # Timbre features
                        "spectral_centroid_hz": phrase.acoustic_features.spectral_centroid_hz,
                        "spectral_slope": phrase.acoustic_features.spectral_slope,
                        "spectral_bandwidth_hz": phrase.acoustic_features.spectral_bandwidth_hz,
                        "spectral_rolloff_hz": phrase.acoustic_features.spectral_rolloff_hz,
                        # Micro-dynamics features
                        "harmonic_to_noise_ratio": phrase.acoustic_features.harmonic_to_noise_ratio,
                        "spectral_flatness": phrase.acoustic_features.spectral_flatness,
                        "attack_time_ms": phrase.acoustic_features.attack_time_ms,
                        "decay_time_ms": phrase.acoustic_features.decay_time_ms,
                        "sustain_level": phrase.acoustic_features.sustain_level,
                        "vibrato_rate_hz": phrase.acoustic_features.vibrato_rate_hz,
                        "vibrato_depth": phrase.acoustic_features.vibrato_depth,
                        "jitter": phrase.acoustic_features.jitter,
                        "mfcc_1": phrase.acoustic_features.mfcc_1,
                        "mfcc_2": phrase.acoustic_features.mfcc_2,
                        "mfcc_3": phrase.acoustic_features.mfcc_3,
                        "mfcc_4": phrase.acoustic_features.mfcc_4,
                        "mfcc_delta_mean": phrase.acoustic_features.mfcc_delta_mean,
                        "spectral_contrast": phrase.acoustic_features.spectral_contrast,
                        "median_ici_ms": phrase.acoustic_features.median_ici_ms,
                        "onset_rate_hz": phrase.acoustic_features.onset_rate_hz,
                        "ici_coefficient_of_variation": (
                            phrase.acoustic_features.ici_coefficient_of_variation
                        ),
                    },
                    "total_occurrences": phrase.total_occurrences,
                    "contexts": [
                        {
                            "context_name": ctx.context_name,
                            "count": ctx.count,
                            "percentage": ctx.percentage,
                        }
                        for ctx in phrase.contexts
                    ],
                    "social_contexts": phrase.social_contexts,
                    "is_compositional": phrase.is_compositional,
                    "phrase_components": phrase.phrase_components,
                }
                species_dict["phrases"][phrase_key] = phrase_dict

            # Export sentences
            for sentence in data.sentences:
                sentence_dict = {
                    "sentence_id": sentence.sentence_id,
                    "species": sentence.species.value,
                    "phrase_sequence": sentence.phrase_sequence,
                    "context": sentence.context,
                    "has_ascending_syntax": sentence.has_ascending_syntax,
                    "has_descending_syntax": sentence.has_descending_syntax,
                    "has_fm_pattern": sentence.has_fm_pattern,
                    "syntax_score": sentence.syntax_score,
                    "total_duration_ms": sentence.total_duration_ms,
                    "complexity_score": sentence.complexity_score,
                }
                species_dict["sentences"].append(sentence_dict)

            # Export grammar rules
            for rule in data.grammar_rules:
                rule_dict = {
                    "rule_id": rule.rule_id,
                    "from_phrase": rule.from_phrase,
                    "to_phrase": rule.to_phrase,
                    "frequency": rule.frequency,
                    "confidence": rule.confidence,
                    "contexts": rule.contexts,
                    "is_bidirectional": rule.is_bidirectional,
                    "strength_score": rule.strength_score,
                }
                species_dict["grammar_rules"].append(rule_dict)

            export_data["species_data"][species.value] = species_dict

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)

    @classmethod
    def import_from_json(cls, filepath: str) -> "VocalizationDatabase":
        """Import database from JSON"""
        with open(filepath, "r") as f:
            data = json.load(f)

        db = cls()

        for species_value, species_data in data["species_data"].items():
            species = Species(species_value)
            species_data_obj = SpeciesData(species=species)

            # Import phrases
            for phrase_key, phrase_data in species_data["phrases"].items():
                acoustic_features = AcousticFeatures(
                    mean_f0_hz=phrase_data["acoustic_features"]["mean_f0_hz"],
                    std_f0_hz=phrase_data["acoustic_features"].get("std_f0_hz", 0.0),
                    min_f0_hz=phrase_data["acoustic_features"].get("min_f0_hz", 0.0),
                    max_f0_hz=phrase_data["acoustic_features"].get("max_f0_hz", 0.0),
                    f0_range_hz=phrase_data["acoustic_features"].get("f0_range_hz", 0.0),
                    mean_duration_ms=phrase_data["acoustic_features"].get("mean_duration_ms", 0.0),
                    duration_frames=phrase_data["acoustic_features"].get("duration_frames", 0),
                    voiced_ratio=phrase_data["acoustic_features"].get("voiced_ratio", 0.0),
                    f0_slope=phrase_data["acoustic_features"].get("f0_slope", 0.0),
                    modulation_rate=phrase_data["acoustic_features"].get("modulation_rate", 0.0),
                    acoustic_variance=phrase_data["acoustic_features"].get(
                        "acoustic_variance", 0.0
                    ),
                    # Timbre features
                    spectral_centroid_hz=phrase_data["acoustic_features"].get(
                        "spectral_centroid_hz", 0.0
                    ),
                    spectral_slope=phrase_data["acoustic_features"].get("spectral_slope", 0.0),
                    spectral_bandwidth_hz=phrase_data["acoustic_features"].get(
                        "spectral_bandwidth_hz", 0.0
                    ),
                    spectral_rolloff_hz=phrase_data["acoustic_features"].get(
                        "spectral_rolloff_hz", 0.0
                    ),
                    # Micro-dynamics features
                    harmonic_to_noise_ratio=phrase_data["acoustic_features"].get(
                        "harmonic_to_noise_ratio", 0.0
                    ),
                    spectral_flatness=phrase_data["acoustic_features"].get(
                        "spectral_flatness", 0.0
                    ),
                    attack_time_ms=phrase_data["acoustic_features"].get("attack_time_ms", 0.0),
                    decay_time_ms=phrase_data["acoustic_features"].get("decay_time_ms", 0.0),
                    sustain_level=phrase_data["acoustic_features"].get("sustain_level", 0.0),
                    vibrato_rate_hz=phrase_data["acoustic_features"].get("vibrato_rate_hz", 0.0),
                    vibrato_depth=phrase_data["acoustic_features"].get("vibrato_depth", 0.0),
                    jitter=phrase_data["acoustic_features"].get("jitter", 0.0),
                    mfcc_1=phrase_data["acoustic_features"].get("mfcc_1", 0.0),
                    mfcc_2=phrase_data["acoustic_features"].get("mfcc_2", 0.0),
                    mfcc_3=phrase_data["acoustic_features"].get("mfcc_3", 0.0),
                    mfcc_4=phrase_data["acoustic_features"].get("mfcc_4", 0.0),
                    mfcc_delta_mean=phrase_data["acoustic_features"].get("mfcc_delta_mean", 0.0),
                    spectral_contrast=phrase_data["acoustic_features"].get(
                        "spectral_contrast", 0.0
                    ),
                    median_ici_ms=phrase_data["acoustic_features"].get("median_ici_ms", 0.0),
                    onset_rate_hz=phrase_data["acoustic_features"].get("onset_rate_hz", 0.0),
                    ici_coefficient_of_variation=phrase_data["acoustic_features"].get(
                        "ici_coefficient_of_variation", 0.0
                    ),
                )

                phrase = Phrase(
                    phrase_key=phrase_key,
                    signature=phrase_data["signature"],
                    species=species,
                    modality=VocalizationModality(phrase_data["modality"]),
                    acoustic_features=acoustic_features,
                    total_occurrences=phrase_data["total_occurrences"],
                    contexts=[
                        PhraseContext(ctx["context_name"], ctx["count"], ctx["percentage"])
                        for ctx in phrase_data["contexts"]
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
                    species=species,
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

            db.add_species_data(species_data_obj)

        return db
