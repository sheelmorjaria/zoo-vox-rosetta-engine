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
"""

import json
import logging
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
    HARMONIC = "Harmonic"   # Tonal (e.g., Phee, whistles)
    TRANSIENT = "Transient"  # Short (e.g., Tsik, clicks)
    MIXED = "Mixed"          # Mixed (e.g., trills)


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
    def from_dict(cls, data: Dict[str, float]) -> 'SourceMetadata':
        return cls(
            mean_f0_hz=data.get('mean_f0_hz', 0.0),
            duration_ms=data.get('duration_ms', 0.0),
            f0_range_hz=data.get('f0_range_hz', 0.0),
            harmonic_to_noise_ratio=data.get('harmonic_to_noise_ratio', 0.0),
            entropy=data.get('entropy', 0.0),
            attack_time_ms=data.get('attack_time_ms', 0.0),
            sustain_level=data.get('sustain_level', 0.5),
            rms_energy=data.get('rms_energy', 0.0),
            fm_depth_hz=data.get('fm_depth_hz', 0.0),
            am_depth=data.get('am_depth', 0.0),
            jitter=data.get('jitter', 0.0),
            shimmer=data.get('shimmer', 0.0),
            loudness=data.get('loudness', 0.5),
            sharpness=data.get('sharpness', 0.0),
        )

    def get_modality(self) -> AcousticModality:
        """Determine modality from metadata"""
        if self.harmonic_to_noise_ratio > 15.0 and self.entropy < 0.3:
            return AcousticModality.HARMONIC
        elif self.harmonic_to_noise_ratio < 10.0 and self.entropy > 0.5 and self.duration_ms < 100.0:
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
            harmonic_to_noise_ratio=source.harmonic_to_noise_ratio + self.delta_harmonic_to_noise_ratio,
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
    def from_dict(cls, data: Dict) -> 'AcousticPrototype':
        return cls(
            label=data['label'],
            sample_rate=data.get('sample_rate', 48000),
            metadata=SourceMetadata.from_dict(data.get('metadata', {})),
            sample_count=data.get('sample_count', 1),
            modality=AcousticModality(data.get('modality', 'Mixed')),
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
            delta.delta_mean_f0_hz = 200.0      # Pitch up for propagation
            delta.delta_sustain_level = 0.2     # Louder
            delta.delta_loudness = 0.15         # More energy

        elif env == EnvState.RAIN:
            # Moderate adaptation
            delta.delta_mean_f0_hz = 100.0
            delta.delta_loudness = 0.1

        elif env == EnvState.STORM:
            # Emergency signal - broader band
            delta.delta_entropy = 0.2           # More noise-like
            delta.delta_loudness = 0.25         # Much louder
            delta.delta_sharpness = 0.3         # More cutting

        # Interaction context adaptations
        if context == InteractionContext.REPLY:
            # Individual identity marker
            delta.delta_mean_f0_hz -= 150.0     # Slightly lower pitch

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

    MAX_HNR_DELTA = 15.0       # Max HNR change (dB)
    MAX_ENTROPY_DELTA = 0.4    # Max spectral flatness change

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
            if source_modality == AcousticModality.HARMONIC and target_modality == AcousticModality.TRANSIENT:
                violations.append(
                    "FORMANT BARRIER: Cannot create Transient from Harmonic via warping"
                )
            elif source_modality == AcousticModality.TRANSIENT and target_modality == AcousticModality.HARMONIC:
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
            'species': self.species,
            'prototypes': {
                label: {
                    'label': p.label,
                    'sample_rate': p.sample_rate,
                    'metadata': {
                        'mean_f0_hz': p.metadata.mean_f0_hz,
                        'duration_ms': p.metadata.duration_ms,
                        'f0_range_hz': p.metadata.f0_range_hz,
                        'harmonic_to_noise_ratio': p.metadata.harmonic_to_noise_ratio,
                        'entropy': p.metadata.entropy,
                        'attack_time_ms': p.metadata.attack_time_ms,
                        'sustain_level': p.metadata.sustain_level,
                        'jitter': p.metadata.jitter,
                        'shimmer': p.metadata.shimmer,
                        'loudness': p.metadata.loudness,
                    },
                    'sample_count': p.sample_count,
                    'modality': p.modality.value,
                }
                for label, p in self.prototypes.items()
            },
            'response_strategies': self.response_strategies,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'AcousticInventory':
        """Load inventory from JSON"""
        with open(path, 'r') as f:
            data = json.load(f)

        inventory = cls(species=data.get('species', 'unknown'))

        for label, pdata in data.get('prototypes', {}).items():
            prototype = AcousticPrototype.from_dict(pdata)
            inventory.prototypes[label] = prototype

        inventory.response_strategies = data.get('response_strategies', {})
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
# Factory Functions
# =============================================================================

def create_default_marmoset_inventory() -> AcousticInventory:
    """Create a default marmoset inventory with typical call types"""
    inventory = AcousticInventory(species="marmoset")

    # Phee - contact call
    inventory.add_prototype(AcousticPrototype(
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
    ))

    # Tsik - alarm call
    inventory.add_prototype(AcousticPrototype(
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
    ))

    # Twitter - social bonding
    inventory.add_prototype(AcousticPrototype(
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
    ))

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


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Create agent
    agent = create_agent("marmoset")

    print("=== Bio-Acoustic Interaction Agent Demo ===\n")

    # Show available labels
    print(f"Available labels: {agent.inventory.available_labels()}\n")

    # Test 1: Plan synthesis in windy conditions
    print("Test 1: Synthesize 'Phee' in windy conditions")
    plan = agent.quick_synthesize("Phee", EnvState.WIND)
    print(f"  Description: {plan.description}")
    print(f"  Delta: +{plan.delta.delta_mean_f0_hz:.0f}Hz pitch, +{plan.delta.delta_loudness:.2f} loudness")
    print(f"  Valid: {plan.validation.is_valid}\n")

    # Test 2: Plan synthesis with high emotional intensity
    print("Test 2: Synthesize 'Tsik' with high urgency (grading=0.9)")
    plan = agent.plan_synthesis("Tsik", grading=0.9)
    print(f"  Description: {plan.description}")
    print(f"  Delta: +{plan.delta.delta_jitter:.2f} jitter, +{plan.delta.delta_shimmer:.2f} shimmer")
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
