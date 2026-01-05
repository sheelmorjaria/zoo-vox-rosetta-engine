"""
Semiotic Detection Engine for Animal Communication Analysis

This module implements a sophisticated semiotic analysis system that goes beyond
simple vocalization detection to understand the cognitive dimensions of animal
communication, including deception, emergence, and multi-sensory integration.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
import numpy as np
from dataclasses import asdict
import json
import logging

from data_models import (
    Species, VocalizationModality, Phrase, AcousticFeatures,
    SpeciesData, VocalizationDatabase
)

logger = logging.getLogger(__name__)


class SemioticRelation(Enum):
    """Types of semiotic relationships in animal communication"""
    INDEXICAL = "indexical"  # Direct causal connection (smoke -> fire)
    ICONIC = "iconic"  # Resemblance-based relationship (sound -> meaning)
    SYMBOLIC = "symbolic"  # Arbitrary conventional relationship
    DECEPTIVE = "deceptive"  # Information falsification
    EMERGENT = "emergent"  # Novel meaning from context
    DIRECTED = "directed"  # Intentional communication target


class SemioticState(Enum):
    """Three fundamental states of semiosis"""
    CONSISTENT = "consistent"  # Signifier -> Object -> Interpretant aligned
    DECEPTIVE = "deceptive"  # Signifier -> Object mismatch (deception)
    EMERGENT = "emergent"  # New Interpretant emerges (innovation)


@dataclass
class SemioticContext:
    """Context for semiotic analysis including sensory and social dimensions"""
    species: Species
    acoustic_features: AcousticFeatures
    social_context: Dict[str, Any] = field(default_factory=dict)
    behavioral_context: Dict[str, Any] = field(default_factory=dict)
    temporal_context: Dict[str, Any] = field(default_factory=dict)
    cross_sensory_data: Dict[str, Any] = field(default_factory=dict)
    attention_focus: Optional[str] = None
    communication_target: Optional[str] = None


@dataclass
class SemioticAnalysisResult:
    """Results of semiotic analysis on a vocalization"""
    phrase_key: str
    semiotic_state: SemioticState
    relation_type: SemioticRelation
    confidence: float
    deception_score: float = 0.0
    emergence_score: float = 0.0
    directed_score: float = 0.0
    cross_modal_attention: Dict[str, float] = field(default_factory=dict)
    interpretant_chain: List[str] = field(default_factory=list)
    context_alignment: float = 0.0
    innovation_potential: float = 0.0
    behavioral_correlates: Dict[str, Any] = field(default_factory=dict)
    communication_target: Optional[str] = None


@dataclass
class SemioticPattern:
    """Recurring semiotic patterns across the population"""
    pattern_id: str
    phrase_keys: List[str]
    relation_type: SemioticRelation
    common_contexts: List[str]
    frequency: int
    adaptive_value: float
    cultural_transmission_score: float = 0.0


class SemioticEngine:
    """
    Advanced semiotic analysis engine that transforms vocalization data
    into cognitive intelligence understanding.
    """

    def __init__(self, database_path: str = "./src/vocalization_database.json"):
        """Initialize with database path"""
        self.database_path = database_path
        self.db = None
        self._load_database()
        self._initialize_semiotic_patterns()

        # Thresholds for semiotic state classification
        self.deception_threshold = 0.7
        self.emergence_threshold = 0.6
        self.directed_threshold = 0.8

    def _load_database(self):
        """Load vocalization database"""
        try:
            with open(self.database_path, 'r') as f:
                data = json.load(f)

            # Reconstruct database
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

    def _create_species_data_from_json(self, species: Species, species_data: Dict) -> SpeciesData:
        """Create SpeciesData object from JSON data"""
        from data_models import SpeciesData, Phrase, Sentence, GrammarRule, PhraseContext

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
                contexts=[PhraseContext(ctx['context_name'], ctx['count'], ctx.get('percentage', 0))
                         for ctx in phrase_data.get('contexts', [])],
                social_contexts=phrase_data.get('social_contexts', {}),
                is_compositional=phrase_data.get('is_compositional', False),
                phrase_components=phrase_data.get('phrase_components', [])
            )
            species_data_obj.add_phrase(phrase)

        return species_data_obj

    def _create_acoustic_features_from_json(self, features_data: Dict) -> AcousticFeatures:
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

    def _initialize_semiotic_patterns(self):
        """Initialize known semiotic patterns for each species"""
        self.semiotic_patterns = {
            Species.MARMOSET: [],
            Species.EGYPTIAN_BAT: [],
            Species.DOLPHIN: [],
            Species.CHIMPANZEE: []
        }

        # Add species-specific patterns
        self._add_marmoset_patterns()
        self._add_bat_patterns()
        self._add_dolphin_patterns()
        self._add_chimpanzee_patterns()

    def analyze_semiotics(self, phrase: Phrase, context: SemioticContext) -> SemioticAnalysisResult:
        """
        Analyze the semiotic state of a vocalization in given context.

        Args:
            phrase: The vocalization phrase to analyze
            context: The context in which the vocalization occurs

        Returns:
            SemioticAnalysisResult containing semiotic analysis results
        """
        # Initialize result
        result = SemioticAnalysisResult(
            phrase_key=phrase.phrase_key,
            semiotic_state=SemioticState.CONSISTENT,
            relation_type=SemioticRelation.INDEXICAL,
            confidence=0.0,
            communication_target=context.communication_target
        )

        # 1. Calculate deception score based on context misalignment
        result.deception_score = self._calculate_deception_score(phrase, context)

        # 2. Calculate emergence score for innovative behaviors
        result.emergence_score = self._calculate_emergence_score(phrase, context)

        # 3. Calculate directed communication score
        result.directed_score = self._calculate_directed_score(phrase, context)

        # 4. Analyze cross-modal attention
        result.cross_modal_attention = self._analyze_cross_modal_attention(phrase, context)

        # 5. Calculate context alignment
        result.context_alignment = self._calculate_context_alignment(phrase, context)

        # 6. Determine semiotic state
        if result.deception_score > self.deception_threshold:
            result.semiotic_state = SemioticState.DECEPTIVE
            result.relation_type = SemioticRelation.DECEPTIVE
        elif result.emergence_score > self.emergence_threshold:
            result.semiotic_state = SemioticState.EMERGENT
            result.relation_type = SemioticRelation.EMERGENT
        elif result.directed_score > self.directed_threshold:
            result.semiotic_state = SemioticState.CONSISTENT
            result.relation_type = SemioticRelation.DIRECTED
        else:
            result.semiotic_state = SemioticState.CONSISTENT

        # 7. Calculate confidence based on multiple factors
        result.confidence = self._calculate_confidence(result)

        # 8. Calculate innovation potential
        result.innovation_potential = self._calculate_innovation_potential(phrase, context)

        # 9. Extract behavioral correlates
        result.behavioral_correlates = self._extract_behavioral_correlates(phrase, context, result)

        # 10. Generate interpretant chain
        result.interpretant_chain = self._generate_interpretant_chain(phrase, context, result)

        return result

    def _calculate_deception_score(self, phrase: Phrase, context: SemioticContext) -> float:
        """Calculate deception score based on various indicators"""
        score = 0.0

        # 1. Low occurrence frequency suggests deception
        if phrase.total_occurrences < 10:
            score += 0.4

        # 2. Context mismatch
        if context.social_context.get("no_immediate_threat", False):
            if "predator" in [ctx.context_name for ctx in phrase.contexts]:
                score += 0.4

        # 3. Social deception indicators
        if context.social_context.get("dominance", False):
            if context.social_context.get("resource_competition", False):
                score += 0.3

        # 4. Cross-species deception
        if context.social_context.get("interspecies_target"):
            score += 0.4

        # 5. Acoustic anomalies
        if phrase.acoustic_features.std_f0_hz > 100:
            score += 0.2

        if phrase.acoustic_features.acoustic_variance > 0.5:
            score += 0.2

        return min(score, 1.0)

    def _calculate_emergence_score(self, phrase: Phrase, context: SemioticContext) -> float:
        """Calculate emergence score for innovative behaviors"""
        score = 0.0

        # 1. First occurrence suggests emergence
        if phrase.total_occurrences <= 1:
            score += 0.5

        # 2. Novel context
        if context.social_context.get("novel_situation", False):
            score += 0.3

        # 3. Problem-solving context
        if context.behavioral_context.get("problem_solving", False):
            score += 0.3

        # 4. Social learning context
        if context.social_context.get("social_learning", False):
            score += 0.3

        # 5. Compositional phrases show emergence
        if phrase.is_compositional:
            score += 0.2

        # 6. High observation potential
        if context.social_context.get("observation_potential", 0) > 0.5:
            score += 0.4

        return min(score, 1.0)

    def _calculate_directed_score(self, phrase: Phrase, context: SemioticContext) -> float:
        """Calculate directed communication score"""
        score = 0.0

        # 1. Specific target context
        if context.communication_target:
            score += 0.5

        # 2. Social context indicates directed communication
        if context.social_context.get("communication_type") == "directed":
            score += 0.3

        # 3. Joint attention
        if context.behavioral_context.get("joint_attention", False):
            score += 0.3

        # 4. Specific context patterns
        if "specific_individual" in [ctx.context_name for ctx in phrase.contexts]:
            score += 0.4

        # 5. Bilateral coordination
        if context.behavioral_context.get("bilateral_coordination", False):
            score += 0.3

        return min(score, 1.0)

    def _analyze_cross_modal_attention(self, phrase: Phrase, context: SemioticContext) -> Dict[str, float]:
        """Analyze cross-modal attention patterns"""
        attention_scores = {}

        # Visual attention
        if context.cross_sensory_data.get("visual_attention", 0) > 0:
            attention_scores["visual_attention"] = context.cross_sensory_data["visual_attention"]

        # Acoustic focus
        if context.cross_sensory_data.get("acoustic_focus", 0) > 0:
            attention_scores["acoustic_focus"] = context.cross_sensory_data["acoustic_focus"]

        # Spatial coordination
        if context.cross_sensory_data.get("spatial_coordination", 0) > 0:
            attention_scores["spatial_coordination"] = context.cross_sensory_data["spatial_coordination"]

        # Attention focus strength
        if context.attention_focus:
            attention_scores["attention_focus"] = 0.8

        return attention_scores

    def _calculate_context_alignment(self, phrase: Phrase, context: SemioticContext) -> float:
        """Calculate how well the phrase aligns with context"""
        alignment = 0.0
        total_checks = 0

        # Check context alignment
        for ctx in phrase.contexts:
            if ctx.context_name in context.social_context:
                alignment += ctx.percentage / 100
            total_checks += 1

        # Add behavioral alignment
        if context.behavioral_context.get("current_behavior"):
            behavior_alignment = context.behavioral_context["current_behavior"]
            if behavior_alignment in [ctx.context_name for ctx in phrase.contexts]:
                alignment += 0.3

        return alignment / max(total_checks, 1)

    def _calculate_confidence(self, result: SemioticAnalysisResult) -> float:
        """Calculate confidence score based on multiple factors"""
        confidence = 0.0
        factors = 0

        # Base confidence from context alignment
        confidence += result.context_alignment
        factors += 1

        # Add deception/emergence/directed confidence
        if result.semiotic_state == SemioticState.DECEPTIVE:
            confidence += result.deception_score
        elif result.semiotic_state == SemioticState.EMERGENT:
            confidence += result.emergence_score
        elif result.relation_type == SemioticRelation.DIRECTED:
            confidence += result.directed_score
        factors += 1

        # Cross-modal attention adds confidence
        if result.cross_modal_attention:
            attention_confidence = sum(result.cross_modal_attention.values()) / len(result.cross_modal_attention)
            confidence += attention_confidence
            factors += 1

        return confidence / max(factors, 1)

    def _calculate_innovation_potential(self, phrase: Phrase, context: SemioticContext) -> float:
        """Calculate innovation potential of the vocalization"""
        potential = 0.0

        # Novel acoustic features
        if phrase.acoustic_features.std_f0_hz > 50:
            potential += 0.3

        # Cultural learning context
        if context.social_context.get("social_learning", False):
            potential += 0.3

        # Observation potential
        if context.social_context.get("observation_potential", 0) > 0.5:
            potential += 0.4

        return potential

    def _extract_behavioral_correlates(self, phrase: Phrase, context: SemioticContext, result: SemioticAnalysisResult) -> Dict[str, Any]:
        """Extract behavioral correlates from analysis"""
        correlates = {}

        # Context misalignment
        if result.context_alignment < 0.5:
            correlates["context_misalignment"] = True

        # Acoustic anomalies
        if phrase.acoustic_features.std_f0_hz > 100:
            correlates["acoustic_anomaly"] = True

        # Cross-species behavior
        if context.social_context.get("interspecies_target"):
            correlates["cross_species"] = True

        # Cultural transmission
        if result.innovation_potential > 0.7:
            correlates["cultural_transmission"] = result.innovation_potential

        # Bilateral coordination
        if context.behavioral_context.get("bilateral_coordination", False):
            correlates["bilateral_coordination"] = True

        return correlates

    def _generate_interpretant_chain(self, phrase: Phrase, context: SemioticContext, result: SemioticAnalysisResult) -> List[str]:
        """Generate interpretant chain for the vocalization"""
        chain = []

        # Base phrase interpretation
        chain.append(f"Phonetic: {phrase.phrase_key}")

        # Contextual interpretation
        if context.attention_focus:
            chain.append(f"Attention: {context.attention_focus}")

        # Communication target
        if context.communication_target:
            chain.append(f"Target: {context.communication_target}")

        # State-specific interpretations
        if result.semiotic_state == SemioticState.DECEPTIVE:
            chain.append("Interpretant: Deceptive communication")
        elif result.semiotic_state == SemioticState.EMERGENT:
            chain.append("Interpretant: Emergent meaning")
        elif result.relation_type == SemioticRelation.DIRECTED:
            chain.append("Interpretant: Directed communication")

        return chain

    def _add_marmoset_patterns(self):
        """Add marmoset-specific semiotic patterns"""
        # Alarm call patterns
        self.semiotic_patterns[Species.MARMOSET].append(SemioticPattern(
            pattern_id="marmoset_alarm_calls",
            phrase_keys=["F0_12000_DUR_0_RANGE_0", "F0_11000_DUR_0_RANGE_0"],
            relation_type=SemioticRelation.INDEXICAL,
            common_contexts=["predator", "threat"],
            frequency=156,
            adaptive_value=0.95
        ))

        # Social bonding patterns
        self.semiotic_patterns[Species.MARMOSET].append(SemioticPattern(
            pattern_id="marmoset_bonding_calls",
            phrase_keys=["F0_6400_DUR_5_RANGE_0", "F0_7200_DUR_5_RANGE_0"],
            relation_type=SemioticRelation.EMERGENT,
            common_contexts=["grooming", "proximity"],
            frequency=89,
            adaptive_value=0.87
        ))

    def _add_bat_patterns(self):
        """Add bat-specific semiotic patterns"""
        # Foraging communication
        self.semiotic_patterns[Species.EGYPTIAN_BAT].append(SemioticPattern(
            pattern_id="bat_foraging_calls",
            phrase_keys=["F0_25000_DUR_0_RANGE_15000", "F0_22000_DUR_0_RANGE_12000"],
            relation_type=SemioticRelation.INDEXICAL,
            common_contexts=["hunting", "feeding"],
            frequency=67,
            adaptive_value=0.92
        ))

        # Social calls
        self.semiotic_patterns[Species.EGYPTIAN_BAT].append(SemioticPattern(
            pattern_id="bat_social_interaction",
            phrase_keys=["F0_18000_DUR_0_RANGE_8000", "F0_15000_DUR_0_RANGE_6000"],
            relation_type=SemioticRelation.ICONIC,
            common_contexts=["roost", "group"],
            frequency=43,
            adaptive_value=0.78
        ))

    def _add_dolphin_patterns(self):
        """Add dolphin-specific semiotic patterns"""
        # Signature whistle patterns
        self.semiotic_patterns[Species.DOLPHIN].append(SemioticPattern(
            pattern_id="dolphin_signature_whistles",
            phrase_keys=["F0_10000_DUR_650_RANGE_7500", "F0_12000_DUR_700_RANGE_8000"],
            relation_type=SemioticRelation.SYMBOLIC,
            common_contexts=["identity", "recognition"],
            frequency=124,
            adaptive_value=0.94
        ))

        # Cooperative hunting
        self.semiotic_patterns[Species.DOLPHIN].append(SemioticPattern(
            pattern_id="dolphin_hunting_cooperation",
            phrase_keys=["F0_8000_DUR_0_RANGE_300", "F0_9000_DUR_0_RANGE_400"],
            relation_type=SemioticRelation.DIRECTED,
            common_contexts=["hunting", "cooperation"],
            frequency=78,
            adaptive_value=0.91
        ))

    def _add_chimpanzee_patterns(self):
        """Add chimpanzee-specific semiotic patterns"""
        # Food calls
        self.semiotic_patterns[Species.CHIMPANZEE].append(SemioticPattern(
            pattern_id="chimpanzee_food_calls",
            phrase_keys=["F0_4000_DUR_100_RANGE_500", "F0_4500_DUR_150_RANGE_600"],
            relation_type=SemioticRelation.INDEXICAL,
            common_contexts=["food", "foraging"],
            frequency=92,
            adaptive_value=0.89
        ))

        # Social hierarchy
        self.semiotic_patterns[Species.CHIMPANZEE].append(SemioticPattern(
            pattern_id="chimpanzee_hierarchy_calls",
            phrase_keys=["F0_5500_DUR_0_RANGE_0", "F0_6000_DUR_0_RANGE_0"],
            relation_type=SemioticRelation.DIRECTED,
            common_contexts=["dominance", "subordination"],
            frequency=67,
            adaptive_value=0.85
        ))