"""
Context-Aware Synthesis with Encoding Support
============================================

Advanced synthesis system using vertical/horizontal/combination encoding
with behavioral semantics as control dimensions.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import time

# Import GPU components
from .jetson_accelerated_core import JetsonAccelerator
from .gpu_phase_vocoder import GPUPhaseVocoder

# Import context and phrase components
try:
    from .probabilistic_context_machine import ContextState
    from .gpu_phrase_integration import GPUPhraseSegment, AtomicWord
    CONTEXT_AVAILABLE = True
except ImportError:
    # Create stubs
    class ContextState(Enum):
        SILENCE = 'silence'
        CONTACT = 'contact'
        ALARM = 'alarm'
        FOOD = 'food'
        NEUTRAL = 'neutral'
        UNCERTAIN = 'uncertain'

    @dataclass
    class GPUPhraseSegment:
        start_time: float
        end_time: float
        audio: np.ndarray
        f0: float
        duration: float
        encoding_type: str
        context: ContextState
        confidence: float

    @dataclass
    class AtomicWord:
        f0: float
        duration: float
        frequency_range: Tuple[float, float]
        encoding_type: str
        count: int

    CONTEXT_AVAILABLE = False

logger = logging.getLogger(__name__)


class BehavioralDimension(Enum):
    """Behavioral semantic dimensions for synthesis control"""
    AGGRESSION = 'aggression'
    FEEDING = 'feeding'
    CONTACT = 'contact'
    NEUTRAL = 'neutral'
    PLAYFUL = 'playful'
    THREATENING = 'threatening'
    SUBMISSIVE = 'submissive'


@dataclass
class SynthesisParams:
    """Parameters for context-aware synthesis"""
    base_f0: float = 6000.0
    base_duration: float = 0.1
    amplitude: float = 0.3
    encoding_type: str = 'horizontal'
    behavioral_dim: BehavioralDimension = BehavioralDimension.NEUTRAL
    modulation_depth: float = 0.1
    vibrato_rate: float = 5.0
    noise_floor: float = 0.01


class ContextAwareSynthesizer:
    """
    GPU-accelerated synthesizer with behavioral semantics and encoding support.

    Implements your original vertical/horizontal/combination encoding methodology
    with behavioral context as a primary control dimension.
    """

    def __init__(self, sr: int = 44100):
        """
        Initialize context-aware synthesizer.

        Args:
            sr: Sample rate for synthesis
        """
        self.sr = sr
        self.accelerator = JetsonAccelerator()
        self.vocoder = GPUPhaseVocoder()

        # Behavioral parameter mappings
        self.behavioral_mappings = {
            BehavioralDimension.AGGRESSION: {
                'f0_range': (7000, 9000),
                'duration_range': (0.05, 0.15),
                'amplitude_range': (0.4, 0.8),
                'encoding_preference': ['combination', 'horizontal'],
                'modulation_depth': 0.2,
                'noise_floor': 0.05
            },
            BehavioralDimension.FEEDING: {
                'f0_range': (5000, 6500),
                'duration_range': (0.1, 0.3),
                'amplitude_range': (0.3, 0.5),
                'encoding_preference': ['vertical', 'combination'],
                'modulation_depth': 0.1,
                'noise_floor': 0.02
            },
            BehavioralDimension.CONTACT: {
                'f0_range': (4000, 6000),
                'duration_range': (0.1, 0.2),
                'amplitude_range': (0.2, 0.4),
                'encoding_preference': ['horizontal'],
                'modulation_depth': 0.05,
                'noise_floor': 0.01
            },
            BehavioralDimension.NEUTRAL: {
                'f0_range': (5500, 6500),
                'duration_range': (0.1, 0.15),
                'amplitude_range': (0.25, 0.35),
                'encoding_preference': ['horizontal', 'vertical'],
                'modulation_depth': 0.08,
                'noise_floor': 0.02
            },
            BehavioralDimension.PLAYFUL: {
                'f0_range': (6000, 8000),
                'duration_range': (0.08, 0.2),
                'amplitude_range': (0.3, 0.6),
                'encoding_preference': ['combination', 'horizontal'],
                'modulation_depth': 0.15,
                'noise_floor': 0.03
            },
            BehavioralDimension.THREATENING: {
                'f0_range': (7500, 10000),
                'duration_range': (0.03, 0.1),
                'amplitude_range': (0.5, 0.9),
                'encoding_preference': ['combination'],
                'modulation_depth': 0.3,
                'noise_floor': 0.1
            },
            BehavioralDimension.SUBMISSIVE: {
                'f0_range': (3500, 5000),
                'duration_range': (0.15, 0.3),
                'amplitude_range': (0.1, 0.3),
                'encoding_preference': ['vertical'],
                'modulation_depth': 0.05,
                'noise_floor': 0.01
            }
        }

        # Encoding type synthesizers
        self.encoding_synthesizers = {
            'horizontal': self._synthesize_horizontal,
            'vertical': self._synthesize_vertical,
            'combination': self._synthesize_combination
        }

        logger.info(f"ContextAwareSynthesizer initialized at {sr}Hz")

    def synthesize_behavioral_vocalization(
        self,
        context: ContextState,
        encoding_type: str,
        behavioral_dim: BehavioralDimension,
        num_variants: int = 3
    ) -> List[Tuple[np.ndarray, SynthesisParams]]:
        """
        Synthesize vocalizations with behavioral and encoding control.

        Args:
            context: Current context state
            encoding_type: 'horizontal', 'vertical', or 'combination'
            behavioral_dim: Behavioral semantic dimension
            num_variants: Number of variants to generate

        Returns:
            List of (audio, params) tuples
        """
        try:
            variants = []

            for _ in range(num_variants):
                # Generate synthesis parameters based on context and behavior
                params = self._generate_synthesis_params(context, encoding_type, behavioral_dim)

                # Synthesize using specified encoding type
                audio = self.encoding_synthesizers[encoding_type](params)

                # Apply behavioral-specific modifications
                audio = self._apply_behavioral_modifications(audio, params, behavioral_dim)

                variants.append((audio, params))

            logger.debug(f"Generated {num_variants} {behavioral_dim.value} vocalizations "
                        f"with {encoding_type} encoding")

            return variants

        except Exception as e:
            logger.error(f"Error in behavioral synthesis: {e}")
            return []

    def _generate_synthesis_params(
        self,
        context: ContextState,
        encoding_type: str,
        behavioral_dim: BehavioralDimension
    ) -> SynthesisParams:
        """Generate synthesis parameters based on context and behavior."""
        # Get behavioral mapping
        mapping = self.behavioral_mappings[behavioral_dim]

        # Base parameters from behavior
        f0_range = mapping['f0_range']
        duration_range = mapping['duration_range']
        amplitude_range = mapping['amplitude_range']

        # Adjust based on context
        context_adjustments = self._get_context_adjustments(context)

        # Apply context modifications
        adjusted_f0 = np.random.uniform(*f0_range) * context_adjustments['f0_multiplier']
        adjusted_duration = np.random.uniform(*duration_range) * context_adjustments['duration_multiplier']
        adjusted_amplitude = np.random.uniform(*amplitude_range) * context_adjustments['amplitude_multiplier']

        return SynthesisParams(
            base_f0=adjusted_f0,
            base_duration=adjusted_duration,
            amplitude=adjusted_amplitude,
            encoding_type=encoding_type,
            behavioral_dim=behavioral_dim,
            modulation_depth=mapping['modulation_depth'],
            noise_floor=mapping['noise_floor']
        )

    def _get_context_adjustments(self, context: ContextState) -> Dict[str, float]:
        """Get parameter adjustments based on context."""
        adjustments = {
            'f0_multiplier': 1.0,
            'duration_multiplier': 1.0,
            'amplitude_multiplier': 1.0
        }

        if context == ContextState.ALARM:
            adjustments['f0_multiplier'] = 1.2  # Higher pitch for alarm
            adjustments['amplitude_multiplier'] = 1.3  # Louder for alarm
        elif context == ContextState.FOOD:
            adjustments['f0_multiplier'] = 0.9  # Slightly lower for feeding
            adjustments['duration_multiplier'] = 1.2  # Longer for food
        elif context == ContextState.CONTACT:
            adjustments['f0_multiplier'] = 1.0  # Normal pitch
            adjustments['duration_multiplier'] = 1.1  # Slightly longer for contact
        elif context == ContextState.SILENCE:
            adjustments['amplitude_multiplier'] = 0.8  # Quieter for silence

        return adjustments

    def _synthesize_horizontal(self, params: SynthesisParams) -> np.ndarray:
        """
        Horizontal encoding: F0 height encoding.

        Your original methodology: primarily encoded through F0 variations
        with relatively constant duration.
        """
        try:
            # Generate base tone
            duration = params.base_duration
            t = np.linspace(0, duration, int(self.sr * duration))

            # Create F0 contour with behavioral characteristics
            f0_contour = self._create_f0_contour(params, t)

            # Generate audio
            audio = np.zeros_like(t)
            for i, freq in enumerate(f0_contour):
                if i < len(t):
                    audio[i] += params.amplitude * np.sin(2 * np.pi * freq * t[i])

            # Add natural harmonics
            audio = self._add_harmonics(audio, params.base_f0, params.amplitude)

            # Apply noise for naturalness
            noise = np.random.normal(0, params.noise_floor, len(audio))
            audio += noise

            return audio

        except Exception as e:
            logger.error(f"Error in horizontal synthesis: {e}")
            return np.zeros(int(self.sr * params.base_duration))

    def _synthesize_vertical(self, params: SynthesisParams) -> np.ndarray:
        """
        Vertical encoding: Duration encoding.

        Your original methodology: primarily encoded through duration variations
        with relatively constant F0.
        """
        try:
            # Generate base tone with fixed F0
            duration = params.base_duration
            t = np.linspace(0, duration, int(self.sr * duration))

            # Constant F0 with slight modulation
            f0 = params.base_f0 * (1 + 0.05 * np.sin(2 * np.pi * params.vibrato_rate * t))

            # Generate audio
            audio = params.amplitude * np.sin(2 * np.pi * f0 * t)

            # Add amplitude envelope for duration encoding
            envelope = self._create_duration_envelope(t, params.behavioral_dim)
            audio *= envelope

            # Add harmonics
            audio = self._add_harmonics(audio, params.base_f0, params.amplitude * 0.8)

            # Apply noise
            noise = np.random.normal(0, params.noise_floor, len(audio))
            audio += noise

            return audio

        except Exception as e:
            logger.error(f"Error in vertical synthesis: {e}")
            return np.zeros(int(self.sr * params.base_duration))

    def _synthesize_combination(self, params: SynthesisParams) -> np.ndarray:
        """
        Combination encoding: Both F0 and duration vary.

        Your original methodology: encoded through both F0 height and duration
        variations for maximum expressiveness.
        """
        try:
            # Extended duration for combination encoding
            duration = params.base_duration * 1.5
            t = np.linspace(0, duration, int(self.sr * duration))

            # Complex F0 contour
            f0_contour = self._create_f0_contour(params, t, complexity=2)

            # Generate audio
            audio = np.zeros_like(t)
            for i, freq in enumerate(f0_contour):
                if i < len(t):
                    audio[i] += params.amplitude * np.sin(2 * np.pi * freq * t[i])

            # Add amplitude modulation
            am_envelope = self._create_combination_envelope(t, params)
            audio *= am_envelope

            # Add rich harmonic content
            audio = self._add_harmonics(audio, params.base_f0, params.amplitude * 1.2)

            # Add vibrato for naturalness
            vibrato = 0.02 * np.sin(2 * np.pi * 5.0 * t)
            audio *= (1 + vibrato)

            # Apply noise
            noise = np.random.normal(0, params.noise_floor * 1.5, len(audio))
            audio += noise

            return audio

        except Exception as e:
            logger.error(f"Error in combination synthesis: {e}")
            return np.zeros(int(self.sr * params.base_duration))

    def _create_f0_contour(self, params: SynthesisParams, t: np.ndarray, complexity: int = 1) -> np.ndarray:
        """Create F0 contour based on behavioral parameters."""
        # Base F0 with behavioral modulation
        f0_contour = np.full_like(t, params.base_f0)

        # Add behavioral-specific modulation
        if params.behavioral_dim == BehavioralDimension.AGGRESSION:
            # Rapid F0 modulation for aggression
            f0_contour += 500 * np.sin(2 * np.pi * 10 * t) * params.modulation_depth
        elif params.behavioral_dim == BehavioralDimension.FEEDING:
            # Slow F0 modulation for feeding
            f0_contour += 200 * np.sin(2 * np.pi * 3 * t) * params.modulation_depth
        elif params.behavioral_dim == BehavioralDimension.PLAYFUL:
            # Playful F0 jumps
            f0_contour += 300 * np.sin(2 * np.pi * 8 * t) * params.modulation_depth

        # Add complexity
        for i in range(complexity):
            freq = 20 * (i + 1)
            f0_contour += 100 / (i + 1) * np.sin(2 * np.pi * freq * t)

        return f0_contour

    def _create_duration_envelope(self, t: np.ndarray, behavioral_dim: BehavioralDimension) -> np.ndarray:
        """Create amplitude envelope for duration encoding."""
        if behavioral_dim == BehavioralDimension.SUBMISSIVE:
            # Slow attack, long decay
            attack_time = 0.2
            decay_time = 0.8
        elif behavioral_dim == BehavioralDimension.THREATENING:
            # Fast attack, fast decay
            attack_time = 0.05
            decay_time = 0.3
        else:
            # Balanced envelope
            attack_time = 0.1
            decay_time = 0.4

        attack_samples = int(attack_time * len(t))
        decay_samples = int(decay_time * len(t))

        envelope = np.ones_like(t)

        # Attack phase
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)

        # Decay phase
        if decay_samples > 0:
            envelope[-decay_samples:] = np.linspace(1, 0.3, decay_samples)

        return envelope

    def _create_combination_envelope(self, t: np.ndarray, params: SynthesisParams) -> np.ndarray:
        """Create complex envelope for combination encoding."""
        # Multi-layer envelope for combination encoding
        envelope = np.ones_like(t)

        # Primary envelope (duration-based)
        primary_envelope = self._create_duration_envelope(t, params.behavioral_dim)
        envelope *= primary_envelope

        # Secondary modulation (F0-based)
        secondary_mod = 1 + 0.1 * np.sin(2 * np.pi * 4 * t)
        envelope *= secondary_mod

        # Tertiary noise texture
        noise_texture = 1 + 0.05 * np.random.normal(0, 1, len(t))
        envelope *= noise_texture

        return envelope

    def _add_harmonics(self, audio: np.ndarray, fundamental_f0: float, amplitude: float) -> np.ndarray:
        """Add harmonic content for natural vocal quality."""
        if len(audio) == 0:
            return audio

        # Add first few harmonics with decreasing amplitude
        harmonic_audio = audio.copy()
        for harmonic in range(2, 6):  # 2nd to 5th harmonics
            harmonic_freq = fundamental_f0 * harmonic
            harmonic_amplitude = amplitude / (harmonic * 2)

            # Create harmonic component
            t = np.linspace(0, len(audio) / self.sr, len(audio))
            harmonic_component = harmonic_amplitude * np.sin(2 * np.pi * harmonic_freq * t)

            # Apply same phase for coherent harmonics
            harmonic_audio += harmonic_component

        return harmonic_audio

    def _apply_behavioral_modifications(
        self,
        audio: np.ndarray,
        params: SynthesisParams,
        behavioral_dim: BehavioralDimension
    ) -> np.ndarray:
        """Apply behavioral-specific modifications to synthesized audio."""
        try:
            # Apply dimension-specific modifications
            if behavioral_dim == BehavioralDimension.AGGRESSION:
                # Add harshness for aggression
                if len(audio) > 0:
                    # Add slight distortion
                    audio = np.tanh(audio * 1.5)

                    # Increase high frequency content
                    fft = np.fft.rfft(audio)
                    fft[len(fft)//4:] *= 1.3  # Boost high frequencies
                    audio = np.fft.irfft(fft)

            elif behavioral_dim == BehavioralDimension.FEEDING:
                # Add softness for feeding
                if len(audio) > 0:
                    # Apply gentle low-pass filter
                    from scipy import signal
                    try:
                        b, a = signal.butter(4, 8000/(self.sr/2), 'low')
                        audio = signal.filtfilt(b, a, audio)
                    except ImportError:
                        pass  # Skip if scipy not available

            elif behavioral_dim == BehavioralDimension.SUBMISSIVE:
                # Add weakness for submissive
                if len(audio) > 0:
                    audio *= 0.7  # Reduce overall amplitude
                    # Add slight tremor
                    t = np.linspace(0, len(audio) / self.sr, len(audio))
                    tremor = 1 + 0.1 * np.sin(2 * np.pi * 8 * t)
                    audio *= tremor

            return audio

        except Exception as e:
            logger.error(f"Error in behavioral modifications: {e}")
            return audio

    def synthesize_phrase_from_atomic_words(
        self,
        atomic_words: List[AtomicWord],
        context: ContextState,
        behavioral_dim: BehavioralDimension
    ) -> Tuple[np.ndarray, SynthesisParams]:
        """
        Synthesize phrase from atomic word sequence.

        Args:
            atomic_words: Sequence of atomic words
            context: Current context
            behavioral_dim: Behavioral dimension

        Returns:
            Tuple of (phrase_audio, final_params)
        """
        try:
            if not atomic_words:
                return np.array([]), SynthesisParams()

            # Synthesize each atomic word
            word_audios = []
            for atomic_word in atomic_words:
                params = SynthesisParams(
                    base_f0=atomic_word.f0,
                    base_duration=atomic_word.duration,
                    amplitude=0.3,
                    encoding_type=atomic_word.encoding_type,
                    behavioral_dim=behavioral_dim
                )

                # Use appropriate synthesizer
                audio = self.encoding_synthesizers[atomic_word.encoding_type](params)
                word_audios.append(audio)

            # Concatenate with appropriate gaps
            phrase_audio = np.concatenate([
                np.pad(audio, (0, int(0.02 * self.sr)), 'constant')  # 20ms gap
                for audio in word_audios
            ])[:-int(0.02 * self.sr)]  # Remove last gap

            return phrase_audio, SynthesisParams()

        except Exception as e:
            logger.error(f"Error in phrase synthesis: {e}")
            return np.array([]), SynthesisParams()

    def get_encoding_summary(self) -> Dict[str, Any]:
        """Get summary of encoding methodology."""
        return {
            'horizontal_encoding': 'F0 height variations with constant duration',
            'vertical_encoding': 'Duration variations with constant F0',
            'combination_encoding': 'Both F0 and duration variations for maximum expressiveness',
            'behavioral_dimensions': [dim.value for dim in BehavioralDimension],
            'context_mappings': {
                ctx.value: {
                    'f0_adjustment': 'multiplier',
                    'duration_adjustment': 'multiplier',
                    'amplitude_adjustment': 'multiplier'
                } for ctx in ContextState
            }
        }