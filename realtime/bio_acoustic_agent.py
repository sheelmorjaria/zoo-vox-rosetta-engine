"""
Bio-Acoustic Interaction Agent (Python Layer)

Bridges the RosettaPipeline (understanding) with Granular Synthesis (response).

This is the Python cognitive layer that:
1. Receives ContextEnrichedPhrase events from Rust
2. Decides response strategy
3. Selects source prototype from AcousticInventory
4. Calculates synthesis deltas
5. Validates against Formant Barrier
6. Triggers synthesis output

Integration Flow:
    Listen: RosettaPipeline → ContextEnrichedPhrase
    Decide: Logic Layer → Response Strategy
    Select: AcousticInventory → Source Prototype
    Calculate: ContextDeltaCalculator → MicroDynamicsDelta
    Check: FormantBarrierValidator → Validation
    Synthesize: GranularConcatenativeSynthesizer → Audio Output

Enhanced Architecture (8-Phase):
    Phase 1: Multi-Modal Fusion (data_fusion.py) - Python Slow Path
    Phase 2: Rosetta Pipeline (Rust) - Fast Path
    Phase 3: Semiotic Analysis (semiotic_engine.py) - Python Slow Path [NEW]
    Phase 4: Probabilistic Context (probabilistic_context_machine.py) - Fast Path
    Phase 5: Adaptive Decision (adaptive_context_switcher.py) - Python Slow Path
    Phase 6: Synthesis Planning (bio_acoustic_agent.rs) - Fast Path
    Phase 7: Granular Synthesis (synthesis.rs) - Fast Path
    Phase 8: Online Learning (cognitive_layer.py) - Python Slow Path [NEW]
"""

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Enums (matching Rust)
# =============================================================================


class EnvState(Enum):
    """Environmental state from sensors"""

    QUIET = "Quiet"
    WIND = "Wind"
    RAIN = "Rain"
    STORM = "Storm"
    UNKNOWN = "Unknown"


class InteractionContext(Enum):
    """Context of the vocalization"""

    INITIATOR = "Initiator"
    REPLY = "Reply"
    SOLO = "Solo"
    CHORUS = "Chorus"


class AcousticModality(Enum):
    """Acoustic modality for Formant Barrier checks"""

    HARMONIC = "Harmonic"  # Tonal (e.g., Phee, whistles)
    TRANSIENT = "Transient"  # Short (e.g., Tsik, clicks)
    MIXED = "Mixed"  # Mixed (e.g., trills)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class SourceMetadata:
    """45D source metadata for synthesis control"""

    # Fundamental
    mean_f0_hz: float = 0.0
    duration_ms: float = 0.0
    f0_range_hz: float = 0.0

    # Harmonic
    harmonic_to_noise_ratio: float = 0.0
    entropy: float = 0.0  # Spectral flatness proxy

    # Temporal
    attack_time_ms: float = 0.0
    sustain_level: float = 0.5
    rms_energy: float = 0.0

    # Modulation
    fm_depth_hz: float = 0.0
    am_depth: float = 0.0

    # Micro-dynamics
    jitter: float = 0.0
    shimmer: float = 0.0

    # Psychoacoustic
    loudness: float = 0.5
    sharpness: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "SourceMetadata":
        return cls(
            mean_f0_hz=data.get("mean_f0_hz", 0.0),
            duration_ms=data.get("duration_ms", 0.0),
            f0_range_hz=data.get("f0_range_hz", 0.0),
            harmonic_to_noise_ratio=data.get("harmonic_to_noise_ratio", 0.0),
            entropy=data.get("entropy", 0.0),
            attack_time_ms=data.get("attack_time_ms", 0.0),
            sustain_level=data.get("sustain_level", 0.5),
            rms_energy=data.get("rms_energy", 0.0),
            fm_depth_hz=data.get("fm_depth_hz", 0.0),
            am_depth=data.get("am_depth", 0.0),
            jitter=data.get("jitter", 0.0),
            shimmer=data.get("shimmer", 0.0),
            loudness=data.get("loudness", 0.5),
            sharpness=data.get("sharpness", 0.0),
        )

    def get_modality(self) -> AcousticModality:
        """Determine modality from metadata"""
        if self.harmonic_to_noise_ratio > 15.0 and self.entropy < 0.3:
            return AcousticModality.HARMONIC
        elif (
            self.harmonic_to_noise_ratio < 10.0 and self.entropy > 0.5 and self.duration_ms < 100.0
        ):
            return AcousticModality.TRANSIENT
        return AcousticModality.MIXED


@dataclass
class MicroDynamicsDelta:
    """Delta transformation for synthesis"""

    delta_mean_f0_hz: float = 0.0
    delta_duration_ms: float = 0.0
    delta_f0_range_hz: float = 0.0
    delta_harmonic_to_noise_ratio: float = 0.0
    delta_entropy: float = 0.0
    delta_attack_time_ms: float = 0.0
    delta_sustain_level: float = 0.0
    delta_fm_depth_hz: float = 0.0
    delta_am_depth: float = 0.0
    delta_jitter: float = 0.0
    delta_shimmer: float = 0.0
    delta_loudness: float = 0.0
    delta_sharpness: float = 0.0

    def apply_to(self, source: SourceMetadata) -> SourceMetadata:
        """Apply delta to source metadata"""
        return SourceMetadata(
            mean_f0_hz=source.mean_f0_hz + self.delta_mean_f0_hz,
            duration_ms=source.duration_ms + self.delta_duration_ms,
            f0_range_hz=source.f0_range_hz + self.delta_f0_range_hz,
            harmonic_to_noise_ratio=source.harmonic_to_noise_ratio
            + self.delta_harmonic_to_noise_ratio,
            entropy=max(0.0, min(1.0, source.entropy + self.delta_entropy)),
            attack_time_ms=source.attack_time_ms + self.delta_attack_time_ms,
            sustain_level=max(0.0, min(1.0, source.sustain_level + self.delta_sustain_level)),
            rms_energy=source.rms_energy,
            fm_depth_hz=source.fm_depth_hz + self.delta_fm_depth_hz,
            am_depth=max(0.0, min(1.0, source.am_depth + self.delta_am_depth)),
            jitter=max(0.0, min(1.0, source.jitter + self.delta_jitter)),
            shimmer=max(0.0, min(1.0, source.shimmer + self.delta_shimmer)),
            loudness=max(0.0, min(1.0, source.loudness + self.delta_loudness)),
            sharpness=source.sharpness + self.delta_sharpness,
        )


@dataclass
class AcousticPrototype:
    """Acoustic prototype with audio buffer and metadata"""

    label: str
    audio_buffer: Optional[bytes] = None  # Raw audio bytes
    sample_rate: int = 48000
    metadata: SourceMetadata = field(default_factory=SourceMetadata)
    sample_count: int = 1
    modality: AcousticModality = AcousticModality.MIXED

    @classmethod
    def from_dict(cls, data: Dict) -> "AcousticPrototype":
        return cls(
            label=data["label"],
            sample_rate=data.get("sample_rate", 48000),
            metadata=SourceMetadata.from_dict(data.get("metadata", {})),
            sample_count=data.get("sample_count", 1),
            modality=AcousticModality(data.get("modality", "Mixed")),
        )


@dataclass
class ValidationResult:
    """Result of formant barrier validation"""

    is_valid: bool
    violations: List[str]
    recommended_action: str


@dataclass
class SynthesisPlan:
    """Complete synthesis plan for execution"""

    source_label: str
    source_metadata: SourceMetadata
    delta: MicroDynamicsDelta
    target_metadata: SourceMetadata
    validation: ValidationResult
    description: str


# =============================================================================
# Context Delta Calculator (Acoustic Algebra)
# =============================================================================


class ContextDeltaCalculator:
    """
    Calculator for context-to-delta mapping (Acoustic Algebra)

    | Context | Source | Delta | Result |
    |---------|--------|-------|--------|
    | High Wind | Phee | +Pitch, +Loudness | Long_Range_Contact |
    | Storm | Any | +Entropy, +Sharpness | Broader band |
    | Agitation | Tsik | +Jitter | High_Urgency_Alarm |
    | Reply | Phee | -Pitch | Individual_Identity |
    """

    @staticmethod
    def calculate(env: EnvState, context: InteractionContext) -> MicroDynamicsDelta:
        """Calculate delta based on environmental state and interaction context"""
        delta = MicroDynamicsDelta()

        # Environmental adaptations
        if env == EnvState.WIND:
            # "Long_Range_Contact" - Cut through noise
            delta.delta_mean_f0_hz = 200.0  # Pitch up for propagation
            delta.delta_sustain_level = 0.2  # Louder
            delta.delta_loudness = 0.15  # More energy

        elif env == EnvState.RAIN:
            # Moderate adaptation
            delta.delta_mean_f0_hz = 100.0
            delta.delta_loudness = 0.1

        elif env == EnvState.STORM:
            # Emergency signal - broader band
            delta.delta_entropy = 0.2  # More noise-like
            delta.delta_loudness = 0.25  # Much louder
            delta.delta_sharpness = 0.3  # More cutting

        # Interaction context adaptations
        if context == InteractionContext.REPLY:
            # Individual identity marker
            delta.delta_mean_f0_hz -= 150.0  # Slightly lower pitch

        elif context == InteractionContext.INITIATOR:
            # Clear, strong signal
            delta.delta_sustain_level += 0.1

        return delta

    @staticmethod
    def calculate_for_grading(grading_score: float) -> MicroDynamicsDelta:
        """Calculate delta for emotional intensity (grading score)"""
        delta = MicroDynamicsDelta()

        if grading_score > 0.7:
            # "High_Urgency_Alarm"
            delta.delta_jitter = 0.15
            delta.delta_shimmer = 0.1
            delta.delta_entropy = 0.1
        elif grading_score < 0.3:
            # Discrete, stable call
            delta.delta_jitter = -0.05
            delta.delta_shimmer = -0.05

        return delta

    @staticmethod
    def combine(deltas: List[MicroDynamicsDelta]) -> MicroDynamicsDelta:
        """Combine multiple deltas"""
        combined = MicroDynamicsDelta()
        for delta in deltas:
            combined.delta_mean_f0_hz += delta.delta_mean_f0_hz
            combined.delta_duration_ms += delta.delta_duration_ms
            combined.delta_f0_range_hz += delta.delta_f0_range_hz
            combined.delta_harmonic_to_noise_ratio += delta.delta_harmonic_to_noise_ratio
            combined.delta_entropy += delta.delta_entropy
            combined.delta_attack_time_ms += delta.delta_attack_time_ms
            combined.delta_sustain_level += delta.delta_sustain_level
            combined.delta_fm_depth_hz += delta.delta_fm_depth_hz
            combined.delta_am_depth += delta.delta_am_depth
            combined.delta_jitter += delta.delta_jitter
            combined.delta_shimmer += delta.delta_shimmer
            combined.delta_loudness += delta.delta_loudness
            combined.delta_sharpness += delta.delta_sharpness
        return combined


# =============================================================================
# Formant Barrier Validator
# =============================================================================


class FormantBarrierValidator:
    """
    Prevents "Semantic Violations" - attempting synthesis beyond physical limits.

    Key rule: cannot cross from Harmonic to Transient via warping alone.
    """

    MAX_HNR_DELTA = 15.0  # Max HNR change (dB)
    MAX_ENTROPY_DELTA = 0.4  # Max spectral flatness change

    @classmethod
    def validate(cls, source: SourceMetadata, target: SourceMetadata) -> ValidationResult:
        """Validate if synthesis from source to target is physically possible"""
        violations = []

        # Check HNR change
        hnr_delta = abs(target.harmonic_to_noise_ratio - source.harmonic_to_noise_ratio)
        if hnr_delta > cls.MAX_HNR_DELTA:
            violations.append(
                f"HNR change too large: {hnr_delta:.1f}dB (max {cls.MAX_HNR_DELTA}dB)"
            )

        # Check entropy change
        entropy_delta = abs(target.entropy - source.entropy)
        if entropy_delta > cls.MAX_ENTROPY_DELTA:
            violations.append(
                f"Entropy change too large: {entropy_delta:.2f} (max {cls.MAX_ENTROPY_DELTA})"
            )

        # Check modality crossing
        source_modality = source.get_modality()
        target_modality = target.get_modality()

        if source_modality != target_modality:
            if (
                source_modality == AcousticModality.HARMONIC
                and target_modality == AcousticModality.TRANSIENT
            ):
                violations.append(
                    "FORMANT BARRIER: Cannot create Transient from Harmonic via warping"
                )
            elif (
                source_modality == AcousticModality.TRANSIENT
                and target_modality == AcousticModality.HARMONIC
            ):
                violations.append(
                    "FORMANT BARRIER: Cannot create Harmonic from Transient via warping"
                )

        # Determine recommended action
        if not violations:
            recommended = "Proceed with synthesis"
        elif any("FORMANT BARRIER" in v for v in violations):
            recommended = "Switch source buffer to match target modality - do NOT warp"
        else:
            recommended = "Reduce delta magnitude or use smaller transformations"

        return ValidationResult(
            is_valid=len(violations) == 0,
            violations=violations,
            recommended_action=recommended,
        )


# =============================================================================
# COGNITIVE GLUE: Semiotic & Context Enhancement (Phase 3 & 4)
# =============================================================================


class ResponseModification(Enum):
    """How the response should be modified based on semiotic analysis"""

    NORMAL = "normal"  # Standard response
    DECEPTION_ACKNOWLEDGE = "deception_ack"  # Acknowledge but don't echo deception
    DECEPTION_IGNORE = "deception_ignore"  # Ignore deceptive signal
    EMERGENCE_LOG = "emergence_log"  # Log novel phrase for review
    EMERGENCE_ECHO = "emergence_echo"  # Echo novel behavior for observation
    DIRECTED_REPLY = "directed_reply"  # Reply to specific target
    URGENCY_BOOST = "urgency_boost"  # Boost response intensity
    URGENCY_REDUCE = "urgency_reduce"  # Reduce response intensity (calming)


@dataclass
class SemioticEnrichment:
    """
    Semiotic analysis results for enhancing synthesis decisions.

    Populated by SemioticEnhancer (Phase 3) - Python Slow Path (10-20 Hz)
    """

    # Core semiotic scores
    deception_score: float = 0.0  # 0.0-1.0: Is this call deceptive?
    emergence_score: float = 0.0  # 0.0-1.0: Is this a novel behavior?
    directed_score: float = 0.0  # 0.0-1.0: Is this intentionally directed?

    # Classification flags
    deception_detected: bool = False
    emergence_detected: bool = False
    directed_communication: bool = False

    # Context
    communication_target: Optional[str] = None
    context_alignment: float = 0.0
    innovation_potential: float = 0.0

    # Recommended action
    response_modification: ResponseModification = ResponseModification.NORMAL

    # Confidence
    confidence: float = 0.0

    # Timing
    analysis_time_ms: float = 0.0


@dataclass
class ProbabilisticContextState:
    """
    Current probabilistic context state.

    Managed by Rust Fast Path, updated by Python Slow Path priors.
    """

    current_context: str = "neutral"  # silence, contact, alarm, food, neutral
    context_confidence: float = 0.0
    predicted_next_context: str = "neutral"
    transition_probability: float = 0.0

    # State history for smoothing
    context_history: List[str] = field(default_factory=list)
    confidence_history: List[float] = field(default_factory=list)

    # Timestamp
    last_update_ms: float = 0.0


@dataclass
class EffectivenessScore:
    """
    Tracks effectiveness of responses for learning.

    Populated by EffectivenessTracker (Phase 8) - Python Slow Path
    """

    input_label: str
    response_label: str
    context: str

    # Proxy metrics
    animal_stayed: bool = False  # Did animal stay in area?
    expected_response: bool = False  # Did animal respond with expected call?
    looked_at_speaker: bool = False  # Did animal look at speaker?

    # Computed effectiveness
    effectiveness: float = 0.0
    timestamp: float = 0.0


class SemioticEnhancer:
    """
    Phase 3: Semiotic Analysis Integration

    Bridges bio_acoustic_agent with semiotic_engine for:
    - Deception detection
    - Emergence tracking
    - Directed communication scoring

    Runs at 10-20 Hz (Python Slow Path)
    """

    # Thresholds matching semiotic_engine.py
    DECEPTION_THRESHOLD = 0.7
    EMERGENCE_THRESHOLD = 0.6
    DIRECTED_THRESHOLD = 0.8

    def __init__(self, database_path: str = "./src/vocalization_database.json"):
        self._engine = None
        self._database_path = database_path
        self._initialized = False
        self._analysis_times: deque = deque(maxlen=100)  # Track latency

    def _lazy_init(self):
        """Lazily initialize the semiotic engine"""
        if self._initialized:
            return

        try:
            from semiotics.semiotic_engine import SemioticEngine, SemioticContext

            self._engine = SemioticEngine(self._database_path)
            self._SemioticContext = SemioticContext
            self._initialized = True
            logger.info("SemioticEngine initialized successfully")
        except ImportError as e:
            logger.warning(f"SemioticEngine not available: {e}")
            self._initialized = False
        except Exception as e:
            logger.error(f"Failed to initialize SemioticEngine: {e}")
            self._initialized = False

    def analyze(
        self,
        semantic_label: str,
        inferred_intent: str,
        social_context: Dict[str, Any] = None,
        behavioral_context: Dict[str, Any] = None,
        cross_sensory_data: Dict[str, Any] = None,
    ) -> SemioticEnrichment:
        """
        Analyze semiotic state of a vocalization.

        Args:
            semantic_label: The semantic label from RosettaPipeline (e.g., "Phee")
            inferred_intent: The inferred intent (e.g., "Contact")
            social_context: Social context dict (dominance, resource_competition, etc.)
            behavioral_context: Behavioral context dict (current_behavior, joint_attention, etc.)
            cross_sensory_data: Cross-modal data (visual_attention, etc.)

        Returns:
            SemioticEnrichment with deception/emergence scores
        """
        start_time = time.perf_counter()

        self._lazy_init()

        enrichment = SemioticEnrichment()

        # If semiotic engine not available, use heuristic analysis
        if not self._initialized or self._engine is None:
            return self._heuristic_analysis(
                semantic_label, inferred_intent, social_context, enrichment
            )

        try:
            # Build semiotic context
            context = self._SemioticContext(
                species=self._infer_species(semantic_label),
                acoustic_features=self._get_placeholder_features(),
                social_context=social_context or {},
                behavioral_context=behavioral_context or {},
                cross_sensory_data=cross_sensory_data or {},
            )

            # Get phrase from database (or use placeholder)
            phrase = self._get_phrase_by_label(semantic_label)
            if phrase is None:
                return self._heuristic_analysis(
                    semantic_label, inferred_intent, social_context, enrichment
                )

            # Run semiotic analysis
            result = self._engine.analyze_semiotics(phrase, context)

            # Map results to enrichment
            enrichment.deception_score = result.deception_score
            enrichment.emergence_score = result.emergence_score
            enrichment.directed_score = result.directed_score
            enrichment.deception_detected = result.deception_score > self.DECEPTION_THRESHOLD
            enrichment.emergence_detected = result.emergence_score > self.EMERGENCE_THRESHOLD
            enrichment.directed_communication = result.directed_score > self.DIRECTED_THRESHOLD
            enrichment.communication_target = result.communication_target
            enrichment.context_alignment = result.context_alignment
            enrichment.innovation_potential = result.innovation_potential
            enrichment.confidence = result.confidence

            # Determine response modification
            enrichment.response_modification = self._determine_response_modification(enrichment)

        except Exception as e:
            logger.error(f"Semiotic analysis failed: {e}")
            return self._heuristic_analysis(
                semantic_label, inferred_intent, social_context, enrichment
            )

        # Track latency
        enrichment.analysis_time_ms = (time.perf_counter() - start_time) * 1000
        self._analysis_times.append(enrichment.analysis_time_ms)

        return enrichment

    def _heuristic_analysis(
        self,
        semantic_label: str,
        inferred_intent: str,
        social_context: Dict[str, Any],
        enrichment: SemioticEnrichment,
    ) -> SemioticEnrichment:
        """Fallback heuristic analysis when semiotic engine unavailable"""

        # Heuristic deception detection
        # Alarm call without threat context = potential deception
        if semantic_label in ["Tsik", "Alarm"] and inferred_intent == "Warning":
            if social_context and not social_context.get("immediate_threat", True):
                enrichment.deception_score = 0.6
                enrichment.deception_detected = True
                enrichment.response_modification = ResponseModification.DECEPTION_ACKNOWLEDGE

        # Heuristic emergence detection
        # Novel context = potential emergence
        if social_context and social_context.get("novel_situation", False):
            enrichment.emergence_score = 0.7
            enrichment.emergence_detected = True
            enrichment.response_modification = ResponseModification.EMERGENCE_LOG

        # Default confidence for heuristic
        enrichment.confidence = 0.5
        enrichment.analysis_time_ms = 1.0  # Fast fallback

        return enrichment

    def _determine_response_modification(
        self, enrichment: SemioticEnrichment
    ) -> ResponseModification:
        """Determine how to modify response based on semiotic analysis"""

        if enrichment.deception_detected:
            # Don't echo deceptive signals
            if enrichment.deception_score > 0.85:
                return ResponseModification.DECEPTION_IGNORE
            else:
                return ResponseModification.DECEPTION_ACKNOWLEDGE

        if enrichment.emergence_detected:
            # Log novel behaviors
            if enrichment.innovation_potential > 0.7:
                return ResponseModification.EMERGENCE_ECHO
            else:
                return ResponseModification.EMERGENCE_LOG

        if enrichment.directed_communication:
            return ResponseModification.DIRECTED_REPLY

        return ResponseModification.NORMAL

    def _infer_species(self, semantic_label: str) -> Any:
        """Infer species from semantic label"""
        # Default to marmoset - in production would use actual detection
        try:
            from data_models import Species

            return Species.MARMOSET
        except ImportError:
            return "marmoset"

    def _get_placeholder_features(self) -> Any:
        """Get placeholder acoustic features"""
        try:
            from data_models import AcousticFeatures

            return AcousticFeatures()
        except ImportError:
            return None

    def _get_phrase_by_label(self, label: str) -> Optional[Any]:
        """Get phrase from database by semantic label"""
        # Placeholder - in production would query database
        return None

    def get_avg_latency_ms(self) -> float:
        """Get average analysis latency in milliseconds"""
        if not self._analysis_times:
            return 0.0
        return sum(self._analysis_times) / len(self._analysis_times)


class ProbabilisticContextAdapter:
    """
    Phase 4: Probabilistic Context Machine Adapter

    Interfaces with probabilistic_context_machine.py for context detection.
    Rust holds state (Fast Path), Python updates priors (Slow Path).

    Runs at audio block rate (Rust) with Python updates at 10-20 Hz
    """

    def __init__(self, history_length: int = 5, confidence_threshold: float = 0.7):
        self._machine = None
        self._initialized = False
        self._confidence_threshold = confidence_threshold

        # Local state cache (synced with Rust)
        self._current_state = ProbabilisticContextState()

    def _lazy_init(self):
        """Lazily initialize the probabilistic context machine"""
        if self._initialized:
            return

        try:
            from realtime.probabilistic_context_machine import (
                ProbabilisticContextMachine,
                ContextState,
            )

            self._machine = ProbabilisticContextMachine(
                history_length=5,
                confidence_threshold=self._confidence_threshold,
            )
            self._ContextState = ContextState
            self._initialized = True
            logger.info("ProbabilisticContextMachine initialized successfully")
        except ImportError as e:
            logger.warning(f"ProbabilisticContextMachine not available: {e}")
            self._initialized = False
        except Exception as e:
            logger.error(f"Failed to initialize ProbabilisticContextMachine: {e}")
            self._initialized = False

    def detect_context(self, audio_features: Dict[str, float]) -> ProbabilisticContextState:
        """
        Detect current context from audio features.

        Args:
            audio_features: Dict with keys like 'rms', 'spectral_centroid', 'f0', etc.

        Returns:
            ProbabilisticContextState with detected context
        """
        self._lazy_init()

        if not self._initialized or self._machine is None:
            return self._heuristic_context(audio_features)

        try:
            from realtime.probabilistic_context_machine import AudioFeatures
            import numpy as np

            # Build AudioFeatures from dict
            features = AudioFeatures(
                rms=audio_features.get("rms", 0.0),
                spectral_centroid=audio_features.get("spectral_centroid", 0.0),
                bandwidth=audio_features.get("bandwidth", 0.0),
                zero_crossing_rate=audio_features.get("zero_crossing_rate", 0.0),
                harmonic_ratio=audio_features.get("harmonic_ratio", 0.0),
                fundamental_freq=audio_features.get("f0", 0.0),
                spectral_flatness=audio_features.get("spectral_flatness", 0.0),
                temporal_envelope=np.array([audio_features.get("rms", 0.0)]),
                mfcc_features=np.zeros(13),
            )

            # Detect context
            state, confidence = self._machine.detect_context(features)

            # Update local state
            self._current_state.current_context = state.value
            self._current_state.context_confidence = confidence
            self._current_state.context_history.append(state.value)
            self._current_state.confidence_history.append(confidence)
            self._current_state.last_update_ms = time.perf_counter() * 1000

            # Predict next context
            predicted = self._machine.predict_next_context()
            if predicted:
                self._current_state.predicted_next_context = predicted.value

        except Exception as e:
            logger.error(f"Context detection failed: {e}")
            return self._heuristic_context(audio_features)

        return self._current_state

    def _heuristic_context(self, audio_features: Dict[str, float]) -> ProbabilisticContextState:
        """Fallback heuristic context detection"""

        state = ProbabilisticContextState()
        f0 = audio_features.get("f0", 0.0)
        rms = audio_features.get("rms", 0.0)

        # Simple heuristic based on F0 and energy
        if rms < 0.01:
            state.current_context = "silence"
            state.context_confidence = 0.8
        elif f0 > 8000:
            state.current_context = "alarm"  # High pitch often alarm
            state.context_confidence = 0.5
        elif 6000 < f0 < 8000:
            state.current_context = "contact"  # Mid-range often contact
            state.context_confidence = 0.5
        else:
            state.current_context = "neutral"
            state.context_confidence = 0.3

        state.last_update_ms = time.perf_counter() * 1000
        return state

    def update_priors(self, context_priors: Dict[str, float]):
        """
        Update context priors from Python Slow Path.

        This is called periodically (10-20 Hz) to adjust the
        probabilistic model based on higher-level cognitive analysis.
        """
        if self._machine is None:
            return

        # Update the machine's prior probabilities
        # (Implementation depends on probabilistic_context_machine.py API)
        pass

    @property
    def current_state(self) -> ProbabilisticContextState:
        """Get current cached state"""
        return self._current_state


class EffectivenessTracker:
    """
    Phase 8: Track effectiveness of responses for online learning.

    Uses proxy metrics:
    - Did the animal stay in the area?
    - Did it respond with the expected call?
    - Did it look at the speaker?

    Runs at 10-20 Hz (Python Slow Path)
    """

    def __init__(self, history_size: int = 100):
        self._history: deque = deque(maxlen=history_size)
        self._effectiveness_by_strategy: Dict[str, List[float]] = {}

    def record_interaction(
        self,
        input_label: str,
        response_label: str,
        context: str,
        animal_reaction: Dict[str, Any],
    ) -> EffectivenessScore:
        """
        Record an interaction and compute effectiveness.

        Args:
            input_label: The input phrase label (e.g., "Tsik")
            response_label: The response phrase label (e.g., "Phee")
            context: The context (e.g., "alarm", "contact")
            animal_reaction: Dict with reaction metrics:
                - stayed: bool
                - expected_response: bool
                - looked_at_speaker: bool

        Returns:
            EffectivenessScore with computed effectiveness
        """
        score = EffectivenessScore(
            input_label=input_label,
            response_label=response_label,
            context=context,
            animal_stayed=animal_reaction.get("stayed", False),
            expected_response=animal_reaction.get("expected_response", False),
            looked_at_speaker=animal_reaction.get("looked_at_speaker", False),
            timestamp=time.time(),
        )

        # Compute effectiveness (weighted combination)
        effectiveness = 0.0
        if score.animal_stayed:
            effectiveness += 0.3
        if score.expected_response:
            effectiveness += 0.5
        if score.looked_at_speaker:
            effectiveness += 0.2

        score.effectiveness = effectiveness
        self._history.append(score)

        # Track by strategy
        strategy_key = f"{input_label}->{response_label}"
        if strategy_key not in self._effectiveness_by_strategy:
            self._effectiveness_by_strategy[strategy_key] = []
        self._effectiveness_by_strategy[strategy_key].append(effectiveness)

        return score

    def get_strategy_effectiveness(self, input_label: str, response_label: str) -> float:
        """Get average effectiveness for a strategy"""
        strategy_key = f"{input_label}->{response_label}"
        scores = self._effectiveness_by_strategy.get(strategy_key, [])
        if not scores:
            return 0.5  # Neutral default
        return sum(scores) / len(scores)

    def get_best_response(self, input_label: str, candidate_responses: List[str]) -> str:
        """Get the best response for an input based on effectiveness history"""
        best_response = candidate_responses[0] if candidate_responses else "Phee"
        best_effectiveness = 0.0

        for response in candidate_responses:
            effectiveness = self.get_strategy_effectiveness(input_label, response)
            if effectiveness > best_effectiveness:
                best_effectiveness = effectiveness
                best_response = response

        return best_response

    @property
    def recent_effectiveness(self) -> float:
        """Get average effectiveness of recent interactions"""
        if not self._history:
            return 0.5
        return sum(s.effectiveness for s in self._history) / len(self._history)


# =============================================================================
# Acoustic Inventory
# =============================================================================


class AcousticInventory:
    """
    Upgraded semantic dictionary with audio prototypes.

    This is the "Source Library" for the synthesizer, mapping semantic
    labels to acoustic prototypes (golden samples).
    """

    def __init__(self, species: str = "unknown"):
        self.species = species
        self.prototypes: Dict[str, AcousticPrototype] = {}
        self.response_strategies: Dict[str, str] = {}

    def add_prototype(self, prototype: AcousticPrototype) -> None:
        """Add a prototype to the inventory"""
        self.prototypes[prototype.label] = prototype

    def get_prototype(self, label: str) -> Optional[AcousticPrototype]:
        """Get prototype by semantic label"""
        return self.prototypes.get(label)

    def available_labels(self) -> List[str]:
        """Get all available labels"""
        return list(self.prototypes.keys())

    def set_response_strategy(self, input_label: str, response_label: str) -> None:
        """Set default response for an input label"""
        self.response_strategies[input_label] = response_label

    def get_response_label(self, input_label: str) -> Optional[str]:
        """Get recommended response for an input"""
        return self.response_strategies.get(input_label)

    def save(self, path: Path) -> None:
        """Save inventory to JSON"""
        data = {
            "species": self.species,
            "prototypes": {
                label: {
                    "label": p.label,
                    "sample_rate": p.sample_rate,
                    "metadata": {
                        "mean_f0_hz": p.metadata.mean_f0_hz,
                        "duration_ms": p.metadata.duration_ms,
                        "f0_range_hz": p.metadata.f0_range_hz,
                        "harmonic_to_noise_ratio": p.metadata.harmonic_to_noise_ratio,
                        "entropy": p.metadata.entropy,
                        "attack_time_ms": p.metadata.attack_time_ms,
                        "sustain_level": p.metadata.sustain_level,
                        "jitter": p.metadata.jitter,
                        "shimmer": p.metadata.shimmer,
                        "loudness": p.metadata.loudness,
                    },
                    "sample_count": p.sample_count,
                    "modality": p.modality.value,
                }
                for label, p in self.prototypes.items()
            },
            "response_strategies": self.response_strategies,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "AcousticInventory":
        """Load inventory from JSON"""
        with open(path, "r") as f:
            data = json.load(f)

        inventory = cls(species=data.get("species", "unknown"))

        for label, pdata in data.get("prototypes", {}).items():
            prototype = AcousticPrototype.from_dict(pdata)
            inventory.prototypes[label] = prototype

        inventory.response_strategies = data.get("response_strategies", {})
        return inventory


# =============================================================================
# Bio-Acoustic Interaction Agent
# =============================================================================


class BioAcousticAgent:
    """
    The complete Bio-Acoustic Interaction Agent.

    Bridges the RosettaPipeline (understanding) with Granular Synthesis (response).

    Usage:
        agent = BioAcousticAgent(inventory, synthesizer)

        # Plan synthesis for a request
        plan = agent.plan_synthesis("Phee", environment=EnvState.WIND)

        # Or use the complete interaction loop
        response = agent.generate_response(input_phrase)
    """

    def __init__(
        self,
        inventory: AcousticInventory,
        synthesizer: Any = None,  # GranularConcatenativeSynthesizer
    ):
        self.inventory = inventory
        self.synthesizer = synthesizer

    def plan_synthesis(
        self,
        label: str,
        environment: EnvState = EnvState.UNKNOWN,
        context: InteractionContext = InteractionContext.SOLO,
        grading: Optional[float] = None,
        pitch_offset: Optional[float] = None,
    ) -> SynthesisPlan:
        """
        Plan synthesis for a request.

        This is the main entry point that:
        1. Selects source prototype from semantic label
        2. Calculates context-based deltas
        3. Validates against Formant Barrier
        4. Returns complete synthesis plan
        """
        # Step 1: Retrieve prototype
        prototype = self.inventory.get_prototype(label)
        if prototype is None:
            raise ValueError(f"No prototype found for label: {label}")

        source_metadata = prototype.metadata

        # Step 2: Calculate deltas
        deltas = [
            ContextDeltaCalculator.calculate(environment, context),
        ]

        if grading is not None:
            deltas.append(ContextDeltaCalculator.calculate_for_grading(grading))

        combined_delta = ContextDeltaCalculator.combine(deltas)

        if pitch_offset is not None:
            combined_delta.delta_mean_f0_hz += pitch_offset

        # Step 3: Calculate target metadata
        target_metadata = combined_delta.apply_to(source_metadata)

        # Step 4: Validate against Formant Barrier
        validation = FormantBarrierValidator.validate(source_metadata, target_metadata)

        # Step 5: Build description
        if validation.is_valid:
            desc = f"Synthesize '{label}' with {environment.value} environment, {context.value} context. Valid."
        else:
            desc = f"Synthesize '{label}' with {environment.value} environment. WARNING: {validation.recommended_action}"

        return SynthesisPlan(
            source_label=label,
            source_metadata=source_metadata,
            delta=combined_delta,
            target_metadata=target_metadata,
            validation=validation,
            description=desc,
        )

    def generate_response(
        self,
        input_label: str,
        environment: EnvState = EnvState.UNKNOWN,
        grading_score: float = 0.5,
    ) -> Tuple[SynthesisPlan, str]:
        """
        Generate a response for an input phrase.

        Args:
            input_label: Semantic label of input phrase
            environment: Current environmental state
            grading_score: Emotional intensity of input

        Returns:
            (synthesis_plan, response_label)
        """
        # Step 1: Select response strategy
        response_label = self.inventory.get_response_label(input_label)
        if response_label is None:
            # Default: reply with same type
            response_label = input_label

        # Step 2: Plan synthesis
        plan = self.plan_synthesis(
            label=response_label,
            environment=environment,
            context=InteractionContext.REPLY,
            grading=grading_score,
        )

        return plan, response_label

    def quick_synthesize(
        self,
        label: str,
        environment: EnvState = EnvState.UNKNOWN,
    ) -> SynthesisPlan:
        """Quick synthesis: label + environment -> plan"""
        return self.plan_synthesis(
            label=label,
            environment=environment,
            context=InteractionContext.REPLY,
        )

    def execute_synthesis(self, plan: SynthesisPlan) -> Optional[bytes]:
        """
        Execute a synthesis plan.

        This requires the synthesizer to be set.
        Returns the synthesized audio bytes or None if not possible.
        """
        if self.synthesizer is None:
            logger.warning("No synthesizer configured")
            return None

        if not plan.validation.is_valid:
            logger.warning(f"Synthesis validation failed: {plan.validation.violations}")
            if any("FORMANT BARRIER" in v for v in plan.validation.violations):
                logger.error("Cannot execute synthesis - would cross Formant Barrier")
                return None

        # Load source
        prototype = self.inventory.get_prototype(plan.source_label)
        if prototype is None or prototype.audio_buffer is None:
            logger.warning(f"No audio buffer for {plan.source_label}")
            return None

        # Apply deltas via synthesizer
        # (This would call the actual synthesizer implementation)
        # synth.load_source_with_metadata(prototype.audio_buffer, plan.source_metadata)
        # synth.shift_pitch_by_hz(plan.delta.delta_mean_f0_hz)
        # synth.adjust_loudness(plan.delta.delta_loudness)
        # etc.

        logger.info(f"Synthesis planned: {plan.description}")
        return prototype.audio_buffer  # Placeholder - would return transformed audio


# =============================================================================
# Enhanced Bio-Acoustic Agent with Cognitive Glue
# =============================================================================


@dataclass
class EnhancedSynthesisPlan:
    """
    Enhanced synthesis plan with semiotic enrichment.

    This is the output of the EnhancedBioAcousticAgent, combining:
    - Base synthesis plan (from Rust Fast Path)
    - Semiotic enrichment (from Python Slow Path)
    - Context state (from ProbabilisticContextMachine)
    """

    # Base synthesis plan
    base_plan: SynthesisPlan

    # Semiotic enrichment (Phase 3)
    semiotic: SemioticEnrichment = field(default_factory=SemioticEnrichment)

    # Probabilistic context (Phase 4)
    context_state: ProbabilisticContextState = field(default_factory=ProbabilisticContextState)

    # Response modification
    response_modification: ResponseModification = ResponseModification.NORMAL

    # Adjusted response label (may differ from base_plan if modified)
    actual_response_label: str = ""

    # Timing
    total_planning_time_ms: float = 0.0

    @property
    def should_respond(self) -> bool:
        """Whether we should respond at all"""
        # Don't respond to high-confidence deception
        if self.semiotic.deception_score > 0.85:
            return False
        # Don't respond in silence context
        if self.context_state.current_context == "silence":
            return False
        return True

    @property
    def should_echo(self) -> bool:
        """Whether we should echo the input or use a different response"""
        if self.semiotic.deception_detected:
            return False  # Don't echo deceptive signals
        if self.semiotic.emergence_detected:
            return True  # Echo novel behaviors for observation
        return True

    @property
    def description(self) -> str:
        """Human-readable description"""
        parts = [self.base_plan.description]

        if self.semiotic.deception_detected:
            parts.append(f"DECEPTION({self.semiotic.deception_score:.2f})")
        if self.semiotic.emergence_detected:
            parts.append(f"EMERGENCE({self.semiotic.emergence_score:.2f})")
        if self.response_modification != ResponseModification.NORMAL:
            parts.append(f"MOD:{self.response_modification.value}")

        return " | ".join(parts)


class EnhancedBioAcousticAgent:
    """
    Enhanced Bio-Acoustic Agent with Cognitive Glue Integration.

    Implements the 8-Phase Architecture:
    - Phase 1: Multi-Modal Fusion (data_fusion.py) - external
    - Phase 2: Rosetta Pipeline (Rust) - external input
    - Phase 3: Semiotic Analysis (semiotic_engine.py) - integrated
    - Phase 4: Probabilistic Context (probabilistic_context_machine.py) - integrated
    - Phase 5: Adaptive Decision (adaptive_context_switcher.py) - integrated
    - Phase 6: Synthesis Planning (bio_acoustic_agent.rs) - delegated
    - Phase 7: Granular Synthesis (synthesis.rs) - external
    - Phase 8: Online Learning (cognitive_layer.py) - integrated

    Hybrid Execution:
    - Python Slow Path (10-20 Hz): Phases 1, 3, 5, 8
    - Rust Fast Path (audio rate): Phases 2, 4, 6, 7
    """

    def __init__(
        self,
        inventory: AcousticInventory,
        database_path: str = "./src/vocalization_database.json",
    ):
        self.base_agent = BioAcousticAgent(inventory)

        # Cognitive enhancement modules
        self.semiotic_enhancer = SemioticEnhancer(database_path)
        self.context_adapter = ProbabilisticContextAdapter()
        self.effectiveness_tracker = EffectivenessTracker()

        # Configuration
        self.enable_semiotic = True
        self.enable_context = True
        self.enable_learning = True

        logger.info("EnhancedBioAcousticAgent initialized with Cognitive Glue")

    def plan_enhanced_synthesis(
        self,
        semantic_label: str,
        inferred_intent: str = "",
        environment: EnvState = EnvState.QUIET,
        context: InteractionContext = InteractionContext.REPLY,
        grading: float = 0.5,
        pitch_offset: float = 0.0,
        social_context: Dict[str, Any] = None,
        behavioral_context: Dict[str, Any] = None,
        audio_features: Dict[str, float] = None,
    ) -> EnhancedSynthesisPlan:
        """
        Plan synthesis with full cognitive enhancement.

        This is the main entry point for the Python Slow Path.

        Args:
            semantic_label: The semantic label from RosettaPipeline
            inferred_intent: The inferred intent (e.g., "Contact", "Warning")
            environment: Environmental state
            context: Interaction context
            grading: Emotional intensity (0.0-1.0)
            pitch_offset: Additional pitch offset in Hz
            social_context: Social context for semiotic analysis
            behavioral_context: Behavioral context for semiotic analysis
            audio_features: Audio features for context detection

        Returns:
            EnhancedSynthesisPlan with semiotic and context enrichment
        """
        start_time = time.perf_counter()

        # Step 1: Get base synthesis plan (Phase 6)
        base_plan = self.base_agent.plan_synthesis(
            label=semantic_label,
            environment=environment,
            context=context,
            grading=grading,
            pitch_offset=pitch_offset,
        )

        enhanced = EnhancedSynthesisPlan(
            base_plan=base_plan,
            actual_response_label=base_plan.source_label,
        )

        # Step 2: Semiotic Analysis (Phase 3) - Python Slow Path
        if self.enable_semiotic:
            enhanced.semiotic = self.semiotic_enhancer.analyze(
                semantic_label=semantic_label,
                inferred_intent=inferred_intent,
                social_context=social_context,
                behavioral_context=behavioral_context,
            )
            enhanced.response_modification = enhanced.semiotic.response_modification

        # Step 3: Probabilistic Context (Phase 4) - with Python cache update
        if self.enable_context and audio_features:
            enhanced.context_state = self.context_adapter.detect_context(audio_features)

        # Step 4: Apply response modification (Phase 5 - Adaptive Decision)
        enhanced.actual_response_label = self._apply_response_modification(
            base_label=base_plan.source_label,
            input_label=semantic_label,
            semiotic=enhanced.semiotic,
            context=enhanced.context_state,
        )

        # Step 5: If response was modified, recalculate base plan
        if enhanced.actual_response_label != base_plan.source_label:
            enhanced.base_plan = self.base_agent.plan_synthesis(
                label=enhanced.actual_response_label,
                environment=environment,
                context=context,
                grading=self._adjusted_grading(grading, enhanced.semiotic),
            )

        # Track timing
        enhanced.total_planning_time_ms = (time.perf_counter() - start_time) * 1000

        return enhanced

    def _apply_response_modification(
        self,
        base_label: str,
        input_label: str,
        semiotic: SemioticEnrichment,
        context: ProbabilisticContextState,
    ) -> str:
        """
        Apply response modification based on semiotic and context analysis.

        This is Phase 5: Adaptive Decision.
        """

        # If deception detected, use calming response
        if semiotic.deception_detected:
            # Get calming response from inventory
            calming = self.base_agent.inventory.get_response_label(input_label)
            if calming and calming != input_label:
                logger.info(f"Deception detected, using calming response: {calming}")
                return calming

        # If emergence detected, log and potentially echo
        if semiotic.emergence_detected:
            logger.info(f"Emergence detected: {semiotic.emergence_score:.2f}")
            # Echo the novel behavior
            return base_label

        # If directed communication, reply to target
        if semiotic.directed_communication:
            logger.info(f"Directed communication to: {semiotic.communication_target}")
            return base_label

        # Check effectiveness history for better response
        if self.enable_learning:
            available = self.base_agent.inventory.available_labels()
            best = self.effectiveness_tracker.get_best_response(input_label, available)
            if best != base_label:
                logger.debug(f"Using effectiveness-optimized response: {best}")
                return best

        return base_label

    def _adjusted_grading(self, original_grading: float, semiotic: SemioticEnrichment) -> float:
        """Adjust grading based on semiotic analysis"""

        # Reduce intensity for deceptive signals
        if semiotic.deception_detected:
            return original_grading * 0.5

        # Increase intensity for urgency
        if semiotic.response_modification == ResponseModification.URGENCY_BOOST:
            return min(1.0, original_grading * 1.3)

        # Reduce intensity for calming
        if semiotic.response_modification == ResponseModification.URGENCY_REDUCE:
            return original_grading * 0.7

        return original_grading

    def record_effectiveness(
        self,
        input_label: str,
        response_label: str,
        animal_reaction: Dict[str, Any],
    ):
        """
        Record effectiveness of a response for learning.

        This is Phase 8: Online Learning.

        Args:
            input_label: The input phrase label
            response_label: The response phrase label
            animal_reaction: Dict with reaction metrics:
                - stayed: bool (did animal stay in area?)
                - expected_response: bool (did animal respond as expected?)
                - looked_at_speaker: bool (did animal look at speaker?)
        """
        if not self.enable_learning:
            return

        context = self.context_adapter.current_state.current_context
        score = self.effectiveness_tracker.record_interaction(
            input_label=input_label,
            response_label=response_label,
            context=context,
            animal_reaction=animal_reaction,
        )

        logger.debug(
            f"Recorded effectiveness: {input_label}->{response_label} = {score.effectiveness:.2f}"
        )

    @property
    def inventory(self) -> AcousticInventory:
        """Access to underlying inventory"""
        return self.base_agent.inventory

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            "semiotic_latency_ms": self.semiotic_enhancer.get_avg_latency_ms(),
            "recent_effectiveness": self.effectiveness_tracker.recent_effectiveness,
            "current_context": self.context_adapter.current_state.current_context,
            "context_confidence": self.context_adapter.current_state.context_confidence,
        }


# =============================================================================
# Factory Functions
# =============================================================================


def create_default_marmoset_inventory() -> AcousticInventory:
    """Create a default marmoset inventory with typical call types"""
    inventory = AcousticInventory(species="marmoset")

    # Phee - contact call
    inventory.add_prototype(
        AcousticPrototype(
            label="Phee",
            metadata=SourceMetadata(
                mean_f0_hz=7000.0,
                duration_ms=300.0,
                f0_range_hz=500.0,
                harmonic_to_noise_ratio=20.0,
                entropy=0.15,
                attack_time_ms=20.0,
                sustain_level=0.7,
                jitter=0.02,
                loudness=0.6,
            ),
            modality=AcousticModality.HARMONIC,
        )
    )

    # Tsik - alarm call
    inventory.add_prototype(
        AcousticPrototype(
            label="Tsik",
            metadata=SourceMetadata(
                mean_f0_hz=9000.0,
                duration_ms=80.0,
                f0_range_hz=200.0,
                harmonic_to_noise_ratio=8.0,
                entropy=0.6,
                attack_time_ms=5.0,
                sustain_level=0.4,
                jitter=0.15,
                loudness=0.8,
            ),
            modality=AcousticModality.TRANSIENT,
        )
    )

    # Twitter - social bonding
    inventory.add_prototype(
        AcousticPrototype(
            label="Twitter",
            metadata=SourceMetadata(
                mean_f0_hz=8000.0,
                duration_ms=200.0,
                f0_range_hz=1500.0,
                harmonic_to_noise_ratio=15.0,
                entropy=0.3,
                attack_time_ms=10.0,
                sustain_level=0.5,
                jitter=0.05,
                loudness=0.5,
            ),
            modality=AcousticModality.MIXED,
        )
    )

    # Set response strategies
    inventory.set_response_strategy("Tsik", "Phee")  # Calm alarm with contact
    inventory.set_response_strategy("Phee", "Phee")  # Reply to contact

    return inventory


def create_agent(species: str = "marmoset") -> BioAcousticAgent:
    """Create a Bio-Acoustic Agent for a species"""
    if species == "marmoset":
        inventory = create_default_marmoset_inventory()
    else:
        inventory = AcousticInventory(species=species)

    return BioAcousticAgent(inventory)


def create_enhanced_agent(
    species: str = "marmoset",
    database_path: str = "./src/vocalization_database.json",
) -> EnhancedBioAcousticAgent:
    """Create an Enhanced Bio-Acoustic Agent with Cognitive Glue"""
    if species == "marmoset":
        inventory = create_default_marmoset_inventory()
    else:
        inventory = AcousticInventory(species=species)

    return EnhancedBioAcousticAgent(inventory, database_path)


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    import sys

    use_enhanced = "--enhanced" in sys.argv

    if use_enhanced:
        # =====================================================================
        # Enhanced Agent Demo with Cognitive Glue
        # =====================================================================
        agent = create_enhanced_agent("marmoset")

        print("=" * 80)
        print("     Enhanced Bio-Acoustic Interaction Agent (with Cognitive Glue)")
        print("=" * 80)
        print()

        print("8-Phase Architecture:")
        print("  Phase 1: Multi-Modal Fusion     (data_fusion.py)      - Python Slow Path")
        print("  Phase 2: Rosetta Pipeline       (Rust)                 - Fast Path")
        print("  Phase 3: Semiotic Analysis      (semiotic_engine.py)   - Python Slow Path")
        print("  Phase 4: Probabilistic Context  (context_machine.py)   - Fast Path")
        print("  Phase 5: Adaptive Decision      (context_switcher.py)  - Python Slow Path")
        print("  Phase 6: Synthesis Planning     (bio_acoustic_agent)   - Fast Path")
        print("  Phase 7: Granular Synthesis     (synthesis.rs)         - Fast Path")
        print("  Phase 8: Online Learning        (cognitive_layer.py)   - Python Slow Path")
        print()

        # Test 1: Normal contact call
        print("Test 1: Normal Contact Call (Phee)")
        print("-" * 40)
        plan = agent.plan_enhanced_synthesis(
            semantic_label="Phee",
            inferred_intent="Contact",
            environment=EnvState.WIND,
            audio_features={"f0": 7000.0, "rms": 0.3},
        )
        print(f"  Should respond: {plan.should_respond}")
        print(f"  Response: {plan.actual_response_label}")
        print(
            f"  Deception: {plan.semiotic.deception_detected} ({plan.semiotic.deception_score:.2f})"
        )
        print(f"  Context: {plan.context_state.current_context}")
        print(f"  Planning time: {plan.total_planning_time_ms:.2f}ms")
        print()

        # Test 2: Potential deception (alarm without threat)
        print("Test 2: Potential Deception (Tsik without threat)")
        print("-" * 40)
        plan = agent.plan_enhanced_synthesis(
            semantic_label="Tsik",
            inferred_intent="Warning",
            environment=EnvState.QUIET,
            social_context={"immediate_threat": False},  # No actual threat!
            audio_features={"f0": 9000.0, "rms": 0.5},
        )
        print(f"  Should respond: {plan.should_respond}")
        print(f"  Response: {plan.actual_response_label}")
        print(
            f"  Deception: {plan.semiotic.deception_detected} ({plan.semiotic.deception_score:.2f})"
        )
        print(f"  Modification: {plan.response_modification.value}")
        print(f"  Context: {plan.context_state.current_context}")
        print()

        # Test 3: Emergence detection (novel situation)
        print("Test 3: Emergence Detection (novel situation)")
        print("-" * 40)
        plan = agent.plan_enhanced_synthesis(
            semantic_label="Twitter",
            inferred_intent="Social",
            social_context={"novel_situation": True},
            audio_features={"f0": 8000.0, "rms": 0.4},
        )
        print(f"  Should respond: {plan.should_respond}")
        print(f"  Response: {plan.actual_response_label}")
        print(
            f"  Emergence: {plan.semiotic.emergence_detected} ({plan.semiotic.emergence_score:.2f})"
        )
        print(f"  Innovation potential: {plan.semiotic.innovation_potential:.2f}")
        print()

        # Test 4: Record effectiveness for learning
        print("Test 4: Record Effectiveness (Phase 8 Learning)")
        print("-" * 40)
        agent.record_effectiveness(
            input_label="Tsik",
            response_label="Phee",
            animal_reaction={
                "stayed": True,
                "expected_response": True,
                "looked_at_speaker": True,
            },
        )
        stats = agent.get_performance_stats()
        print(f"  Recent effectiveness: {stats['recent_effectiveness']:.2f}")
        print(f"  Current context: {stats['current_context']}")
        print()

        print("=" * 80)
        print("Enhanced Agent Demo Complete")
        print("=" * 80)

    else:
        # =====================================================================
        # Standard Agent Demo
        # =====================================================================
        agent = create_agent("marmoset")

        print("=== Bio-Acoustic Interaction Agent Demo ===\n")

        # Show available labels
        print(f"Available labels: {agent.inventory.available_labels()}\n")

        # Test 1: Plan synthesis in windy conditions
        print("Test 1: Synthesize 'Phee' in windy conditions")
        plan = agent.quick_synthesize("Phee", EnvState.WIND)
        print(f"  Description: {plan.description}")
        print(
            f"  Delta: +{plan.delta.delta_mean_f0_hz:.0f}Hz pitch, +{plan.delta.delta_loudness:.2f} loudness"
        )
        print(f"  Valid: {plan.validation.is_valid}\n")

        # Test 2: Plan synthesis with high emotional intensity
        print("Test 2: Synthesize 'Tsik' with high urgency (grading=0.9)")
        plan = agent.plan_synthesis("Tsik", grading=0.9)
        print(f"  Description: {plan.description}")
        print(
            f"  Delta: +{plan.delta.delta_jitter:.2f} jitter, +{plan.delta.delta_shimmer:.2f} shimmer"
        )
        print(f"  Valid: {plan.validation.is_valid}\n")

        # Test 3: Response generation
        print("Test 3: Generate response to 'Tsik' alarm")
        plan, response_label = agent.generate_response("Tsik", EnvState.WIND)
        print(f"  Response label: {response_label}")
        print(f"  Description: {plan.description}")
        print(f"  Valid: {plan.validation.is_valid}\n")

        # Test 4: Formant Barrier validation
        print("Test 4: Attempting to cross Formant Barrier")
        source = SourceMetadata(harmonic_to_noise_ratio=25.0, entropy=0.1)
        target = SourceMetadata(harmonic_to_noise_ratio=5.0, entropy=0.8)
        validation = FormantBarrierValidator.validate(source, target)
        print(f"  Valid: {validation.is_valid}")
        print(f"  Violations: {validation.violations}")
        print(f"  Recommended: {validation.recommended_action}")

        print("\n  Run with --enhanced flag for Cognitive Glue demo")
