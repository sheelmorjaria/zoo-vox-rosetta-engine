"""
Data Fusion Module
==================

Implements cross-modal data fusion that integrates visual and audio information
to enhance response prediction. The module applies a 20% boost to response
probability when high visual attention is detected during contact calls.

Key Features:
- Visual + Vocalization fusion with 20% attention boost
- Species-specific cross-modal weighting
- Attention ensemble combination
- Context-aware fusion logic

Architecture:
```
DataFusionSystem
├── CrossModalFusion
│   ├── VisualAttentionCalculator
│   ├── AuditoryFeatureExtractor
│   └── AttentionEnsemble
├── SpeciesSpecificWeights
│   ├── Marmoset (30% visual, 70% audio)
│   ├── Dolphin (10% visual, 90% audio)
│   └── Human (60% visual, 40% audio)
└── ResponseBoostLogic
    ├── ContactCallAttentionBoost (20%)
    └── ContextAwareModulation
```

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

# Import visual fusion components
try:
    from cognitive_intelligence.visual_fusion import VisualAttentionLevel, VisualFeatures
except ImportError:
    # Create mock classes if visual fusion not available
    class VisualAttentionLevel(Enum):
        LOW = "Low"
        MODERATE = "Moderate"
        HIGH = "High"
        VERY_HIGH = "Very High"

    @dataclass
    class VisualFeatures:
        """Visual features container (mock implementation)"""

        attention_level: VisualAttentionLevel = VisualAttentionLevel.LOW
        gaze_direction: Optional[str] = None
        movement_intensity: float = 0.0
        hand_gestures: List[str] = None
        confidence: float = 0.0
        timestamp: float = time.time()

        def __post_init__(self):
            if self.hand_gestures is None:
                self.hand_gestures = []


@dataclass
class AudioFeatures:
    """Audio features container"""

    rms: float  # Root mean square energy
    f0: float  # Fundamental frequency
    spectral_centroid: float
    bandwidth: float
    context: str  # Behavioral context
    response_probability: float
    confidence: float = 0.0
    timestamp: float = time.time()


@dataclass
class FusionConfig:
    """Configuration for Data Fusion System"""

    attention_boost_factor: float = 0.2  # 20% boost for high attention
    species_weights: Dict[str, Dict[str, float]] = None
    min_attention_threshold: VisualAttentionLevel = VisualAttentionLevel.MODERATE
    enable_cross_modal_fusion: bool = True
    context_aware_boosting: bool = True

    def __post_init__(self):
        if self.species_weights is None:
            self.species_weights = {
                "marmoset": {"visual_weight": 0.3, "audio_weight": 0.7},
                "dolphin": {"visual_weight": 0.1, "audio_weight": 0.9},
                "human": {"visual_weight": 0.6, "audio_weight": 0.4},
                "default": {"visual_weight": 0.5, "audio_weight": 0.5},
            }


class VisualAttentionCalculator:
    """Calculate visual attention scores from visual features"""

    def __init__(self, config: FusionConfig):
        self.config = config

    def calculate_attention_score(self, visual_features: VisualFeatures) -> float:
        """Calculate normalized attention score (0.0 to 1.0)"""
        base_score = 0.0

        # Base score from attention level
        if visual_features.attention_level == VisualAttentionLevel.VERY_HIGH:
            base_score = 1.0
        elif visual_features.attention_level == VisualAttentionLevel.HIGH:
            base_score = 0.8
        elif visual_features.attention_level == VisualAttentionLevel.MODERATE:
            base_score = 0.5
        elif visual_features.attention_level == VisualAttentionLevel.LOW:
            base_score = 0.2

        # Apply gaze direction modifier
        gaze_modifier = 1.0
        if visual_features.gaze_direction == "towards_camera":
            gaze_modifier = 1.2
        elif visual_features.gaze_direction == "away":
            gaze_modifier = 0.5

        # Apply movement intensity modifier
        movement_modifier = min(visual_features.movement_intensity, 1.0)

        # Apply confidence modifier
        confidence_modifier = visual_features.confidence

        # Final score with all modifiers
        final_score = base_score * gaze_modifier * movement_modifier * confidence_modifier
        return min(final_score, 1.0)  # Clamp to [0, 1]


class AuditoryFeatureExtractor:
    """Extract and normalize auditory features"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def normalize_audio_features(self, audio_features: AudioFeatures) -> np.ndarray:
        """Normalize audio features to [0, 1] range"""
        features = np.array(
            [
                min(audio_features.rms / 0.5, 1.0),  # Normalize RMS
                min(audio_features.f0 / 20000.0, 1.0),  # Normalize F0 (max 20kHz)
                min(audio_features.spectral_centroid / 10000.0, 1.0),  # Normalize spectral centroid
                min(audio_features.bandwidth / 10000.0, 1.0),  # Normalize bandwidth
                audio_features.response_probability,  # Already in [0, 1]
            ]
        )
        return features

    def extract_context_features(self, audio_features: AudioFeatures) -> Dict[str, float]:
        """Extract context-specific features"""
        context_features = {}

        # Contact call modifier
        if audio_features.context.lower() == "contact_call":
            context_features["contact_call_boost"] = 1.2
        else:
            context_features["contact_call_boost"] = 1.0

        # Alarm call modifier
        if audio_features.context.lower() == "alarm_call":
            context_features["alarm_call_modifier"] = 0.8
        else:
            context_features["alarm_call_modifier"] = 1.0

        return context_features


class AttentionEnsemble:
    """Combine visual and auditory attention signals"""

    def __init__(self, config: FusionConfig):
        self.config = config

    def combine_attention_signals(
        self, visual_attention: float, auditory_attention: float, species: str = "default"
    ) -> float:
        """Combine visual and auditory attention signals using species-specific weights"""
        # Get species-specific weights, fall back to default
        if species in self.config.species_weights:
            weights = self.config.species_weights[species]
        elif "default" in self.config.species_weights:
            weights = self.config.species_weights["default"]
        else:
            # Fallback to equal weights if no default available
            weights = {"visual_weight": 0.5, "audio_weight": 0.5}

        # Weighted combination
        combined_attention = (
            visual_attention * weights["visual_weight"]
            + auditory_attention * weights["audio_weight"]
        )

        # Apply context modulation if enabled
        if self.config.context_aware_boosting:
            # Additional boost for high attention in social contexts
            if visual_attention > 0.7 and auditory_attention > 0.7:
                combined_attention *= 1.1

        return min(combined_attention, 1.0)


class CrossModalFusion:
    """Cross-modal fusion of visual and audio information"""

    def __init__(self, config: FusionConfig):
        self.config = config
        self.visual_calculator = VisualAttentionCalculator(config)
        self.auditory_extractor = AuditoryFeatureExtractor()
        self.attention_ensemble = AttentionEnsemble(config)
        self.logger = logging.getLogger(__name__)

    def fuse_modalities(
        self,
        visual_features: VisualFeatures,
        audio_features: AudioFeatures,
        species: str = "default",
    ) -> Dict[str, Any]:
        """Fuse visual and auditory modalities"""
        # Calculate individual attention scores
        visual_attention = self.visual_calculator.calculate_attention_score(visual_features)
        auditory_attention = audio_features.response_probability  # Direct from audio

        # Combine attention signals
        combined_attention = self.attention_ensemble.combine_attention_signals(
            visual_attention, auditory_attention, species
        )

        # Apply species-specific fusion logic
        fusion_result = self._apply_species_fusion_logic(
            visual_features, audio_features, combined_attention, species
        )

        fusion_result.update(
            {
                "visual_attention_score": visual_attention,
                "auditory_attention_score": auditory_attention,
                "combined_attention": combined_attention,
                "fusion_timestamp": time.time(),
                "species": species,
            }
        )

        return fusion_result

    def _apply_species_fusion_logic(
        self,
        visual_features: VisualFeatures,
        audio_features: AudioFeatures,
        combined_attention: float,
        species: str,
    ) -> Dict[str, Any]:
        """Apply species-specific fusion logic"""
        result = {"response_probability": combined_attention}

        # Marmoset-specific logic
        if species == "marmoset":
            if audio_features.context.lower() == "contact_call":
                result["enhanced_social_bonding"] = True
                result["call_sharpness_boost"] = 0.1

        # Dolphin-specific logic
        elif species == "dolphin":
            if audio_features.context.lower() == "contact_call":
                result["whistle_modulation_enhanced"] = True

        # Human-specific logic
        elif species == "human":
            result["verbal_hinting_enabled"] = True

        return result


class ResponseBoostLogic:
    """Apply attention-based response probability boosting"""

    def __init__(self, config: FusionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def calculate_attention_boost(
        self, visual_features: VisualFeatures, audio_features: AudioFeatures
    ) -> float:
        """Calculate attention-based response probability boost"""
        # Only apply boost for contact calls with high attention
        if audio_features.context.lower() == "contact_call" and self._is_attention_level_sufficient(
            visual_features.attention_level
        ):
            base_boost = self.config.attention_boost_factor

            # Boost scaling based on attention level
            if visual_features.attention_level == VisualAttentionLevel.VERY_HIGH:
                boost_factor = base_boost * 1.5  # 1.5x boost for very high attention
            elif visual_features.attention_level == VisualAttentionLevel.HIGH:
                boost_factor = base_boost  # Full boost for high
            elif visual_features.attention_level == VisualAttentionLevel.MODERATE:
                boost_factor = base_boost * 0.5  # Half boost for moderate
            else:
                boost_factor = 0.0

            return boost_factor

        return 0.0

    def _is_attention_level_sufficient(self, attention_level: VisualAttentionLevel) -> bool:
        """Check if attention level meets minimum threshold"""
        threshold_value = self.config.min_attention_threshold.value
        current_value = attention_level.value

        # Handle both string and enum values
        if isinstance(current_value, str):
            # Map string values to numeric for comparison
            level_mapping = {"Low": 1, "Moderate": 2, "High": 3, "VERY_HIGH": 4, "Very High": 4}
            threshold_num = level_mapping.get(threshold_value, 2)
            current_num = level_mapping.get(current_value, 1)
            return current_num >= threshold_num
        else:
            # Handle enum comparison
            return attention_level.value >= threshold_value

    def apply_response_boost(
        self, audio_features: AudioFeatures, visual_features: VisualFeatures
    ) -> AudioFeatures:
        """Apply attention boost to audio features"""
        # Calculate boost amount
        boost_amount = self.calculate_attention_boost(visual_features, audio_features)

        if boost_amount > 0:
            # Apply boost to response probability
            original_probability = audio_features.response_probability
            boosted_probability = min(original_probability + boost_amount, 1.0)

            # Create updated audio features
            boosted_features = AudioFeatures(
                rms=audio_features.rms,
                f0=audio_features.f0,
                spectral_centroid=audio_features.spectral_centroid,
                bandwidth=audio_features.bandwidth,
                context=audio_features.context,
                response_probability=boosted_probability,
                confidence=max(audio_features.confidence, boost_amount),
                timestamp=time.time(),
            )

            self.logger.info(
                f"Applied {boost_amount:.2f} boost to response probability: "
                f"{original_probability:.2f} -> {boosted_probability:.2f}"
            )

            return boosted_features

        return audio_features


class DataFusionSystem:
    """Main data fusion system coordinating cross-modal integration"""

    def __init__(self, config: FusionConfig):
        self.config = config
        self.cross_modal_fusion = CrossModalFusion(config)
        self.response_boost_logic = ResponseBoostLogic(config)
        self.logger = logging.getLogger(__name__)

        # Performance tracking
        self.fusion_count = 0
        self.boost_count = 0
        self.last_fusion_time = None

    def integrate_with_audio(
        self,
        audio_features: Dict[str, Any],
        visual_features: VisualFeatures,
        species: str = "default",
    ) -> Dict[str, Any]:
        """Integrate visual features with audio features"""
        # Convert dict to AudioFeatures if needed
        if isinstance(audio_features, dict):
            try:
                audio_features = AudioFeatures(**audio_features)
            except TypeError:
                # Handle invalid dict gracefully
                self.logger.warning(f"Invalid audio features dict: {audio_features}")
                return {"error": "Invalid audio features", "fusion_timestamp": time.time()}

        # Perform cross-modal fusion
        fusion_result = self.cross_modal_fusion.fuse_modalities(
            visual_features, audio_features, species
        )

        # Apply attention-based response boost
        boosted_features = self.response_boost_logic.apply_response_boost(
            audio_features, visual_features
        )

        # Update fusion result with boosted features
        fusion_result.update(
            {
                "original_response_probability": audio_features.response_probability,
                "boosted_response_probability": boosted_features.response_probability,
                "boost_applied": boosted_features.response_probability
                > audio_features.response_probability,
                "boost_amount": boosted_features.response_probability
                - audio_features.response_probability,
                "visual_context": {
                    "attention_level": visual_features.attention_level.value,
                    "gaze_direction": visual_features.gaze_direction,
                    "movement_intensity": visual_features.movement_intensity,
                    "hand_gestures": visual_features.hand_gestures,
                },
            }
        )

        # Update performance tracking
        self.fusion_count += 1
        if fusion_result["boost_applied"]:
            self.boost_count += 1

        # Log fusion result
        self.logger.info(
            f"Fusion completed: Visual attention {fusion_result['visual_attention_score']:.2f}, "
            f"Boost applied: {fusion_result['boost_applied']}, "
            f"Response prob: {audio_features.response_probability:.2f} -> "
            f"{boosted_features.response_probability:.2f}"
        )

        return fusion_result

    def create_visual_attention_score(self, visual_features: VisualFeatures) -> float:
        """Create a unified visual attention score (0.0 to 1.0)"""
        return self.cross_modal_fusion.visual_calculator.calculate_attention_score(visual_features)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the fusion system"""
        return {
            "fusion_count": self.fusion_count,
            "boost_count": self.boost_count,
            "boost_rate": self.boost_count / max(self.fusion_count, 1),
            "config": {
                "attention_boost_factor": self.config.attention_boost_factor,
                "min_attention_threshold": self.config.min_attention_threshold.value,
                "enable_cross_modal_fusion": self.config.enable_cross_modal_fusion,
            },
        }


# Test utility function
def create_test_data_fusion_system() -> DataFusionSystem:
    """Create a DataFusionSystem for testing"""
    config = FusionConfig(
        attention_boost_factor=0.2,
        species_weights={
            "marmoset": {"visual_weight": 0.3, "audio_weight": 0.7},
            "dolphin": {"visual_weight": 0.1, "audio_weight": 0.9},
            "human": {"visual_weight": 0.6, "audio_weight": 0.4},
        },
    )
    return DataFusionSystem(config)
