"""
Universal Synthesizer - Phrase and Sentence Generation

This module synthesizes new vocalizations based on discovered vocabulary
and grammatical rules, enabling the system to generate novel "sentences"
for interaction.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
import logging

try:
    # Try relative import first
    from .universal_rosetta_stone import Modality, PhraseSignature
except ImportError:
    try:
        # Try absolute import
        from analysis.rosetta_stone.universal_rosetta_stone import Modality, PhraseSignature
    except ImportError:
        # Fallback for running as script
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from universal_rosetta_stone import Modality, PhraseSignature

import warnings
warnings.filterwarnings('ignore')


class UniversalSynthesizer:
    """
    Synthesizes novel vocalizations from discovered vocabulary and grammar.

    This system uses Markov chain generation to create plausible sequences
    of phrases that follow the discovered syntactic rules.
    """

    def __init__(self, vocabulary: Dict[int, PhraseSignature], grammar: Counter):
        """
        Initialize synthesizer with discovered vocabulary and grammar.

        Args:
            vocabulary: Dictionary mapping cluster IDs to phrase signatures
            grammar: Counter of transition probabilities
        """
        self.vocabulary = vocabulary
        self.grammar = grammar
        self.logger = logging.getLogger(__name__)

        # Configure logging
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def generate_sequence(self, num_phrases: int,
                         start_phrase: Optional[int] = None,
                         temperature: float = 1.0) -> List[int]:
        """
        Generate a sequence of phrase IDs using Markov chain generation.

        Args:
            num_phrases: Number of phrases to generate
            start_phrase: Starting phrase ID (random if None)
            temperature: Sampling temperature (higher = more random)

        Returns:
            List of phrase IDs representing the generated sequence
        """
        if not self.vocabulary:
            raise ValueError("Vocabulary is empty. Cannot generate sequence.")

        if num_phrases <= 0:
            return []

        # Choose starting phrase
        if start_phrase is None:
            current = np.random.choice(list(self.vocabulary.keys()))
        else:
            if start_phrase not in self.vocabulary:
                self.logger.warning(f"Start phrase {start_phrase} not in vocabulary")
                current = np.random.choice(list(self.vocabulary.keys()))
            else:
                current = start_phrase

        sequence = [current]

        for _ in range(num_phrases - 1):
            # Get possible transitions from current phrase
            transitions = {k: v for k, v in self.grammar.items() if k[0] == current}

            if not transitions:
                # No known transitions, choose randomly
                if len(self.vocabulary) > 1:
                    remaining = [pid for pid in self.vocabulary.keys() if pid != current]
                    current = np.random.choice(remaining)
                else:
                    current = sequence[0]  # Repeat the only phrase
            else:
                # Apply temperature to transition probabilities
                transition_pairs = list(transitions.keys())
                raw_probs = np.array([transitions[pair] for pair in transition_pairs])

                # Temperature scaling
                if temperature != 1.0:
                    raw_probs = raw_probs ** (1.0 / temperature)
                    raw_probs = raw_probs / np.sum(raw_probs)

                # Choose next phrase
                probs = raw_probs / np.sum(raw_probs)
                next_idx = np.random.choice(len(transition_pairs), p=probs)
                current = transition_pairs[next_idx][1]

            sequence.append(current)

        self.logger.info(f"Generated sequence of {len(sequence)} phrases: {sequence}")
        return sequence

    def synthesize_audio(self, sequence: List[int],
                       phrase_duration_ms: float = 100.0,
                       gap_ms: float = 50.0,
                       sample_rate: int = 48000) -> np.ndarray:
        """
        Convert a sequence of phrase IDs into audio.

        Args:
            sequence: List of phrase IDs
            phrase_duration_ms: Duration for each synthesized phrase
            gap_ms: Silence gap between phrases
            sample_rate: Audio sample rate

        Returns:
            Synthesized audio signal
        """
        if not sequence:
            return np.array([])

        duration_samples = int(phrase_duration_ms * sample_rate / 1000)
        gap_samples = int(gap_ms * sample_rate / 1000)

        # Start with silence
        audio = np.zeros(duration_samples)

        for i, phrase_id in enumerate(sequence):
            if phrase_id not in self.vocabulary:
                self.logger.warning(f"Phrase ID {phrase_id} not found in vocabulary")
                continue

            phrase_signature = self.vocabulary[phrase_id]

            # Synthesize phrase based on modality
            synthesized_phrase = self._synthesize_phrase(
                phrase_signature, duration_samples, sample_rate
            )

            # Add gap between phrases (except first)
            if i > 0:
                audio = np.concatenate([audio, np.zeros(gap_samples)])

            # Add synthesized phrase
            audio = np.concatenate([audio, synthesized_phrase])

        self.logger.info(f"Synthesized audio: {len(audio)/sample_rate:.2f}s")
        return audio

    def _synthesize_phrase(self, phrase_signature: PhraseSignature,
                          duration_samples: int, sample_rate: int) -> np.ndarray:
        """
        Synthesize audio for a single phrase based on its signature.

        Args:
            phrase_signature: Phrase signature to synthesize
            duration_samples: Target duration in samples
            sample_rate: Audio sample rate

        Returns:
            Synthesized phrase audio
        """
        t = np.linspace(0, duration_samples / sample_rate, duration_samples)

        if phrase_signature.modality == Modality.HARMONIC:
            # Synthesize harmonic tone with discovered pitch
            f0 = phrase_signature.features.get('f0_mean', 6000)  # Default to 6kHz
            if f0 <= 0:
                f0 = 6000  # Fallback

            # Add harmonics for richer sound
            audio = np.zeros_like(t)
            for harmonic in [1, 2, 3]:  # First 3 harmonics
                amplitude = 1.0 / harmonic
                phase = np.random.uniform(0, 2 * np.pi)  # Random phase
                audio += amplitude * np.sin(2 * np.pi * f0 * harmonic * t + phase)

            # Apply amplitude envelope
            envelope = self._create_envelope(t, 'exponential')
            audio = audio * envelope

        elif phrase_signature.modality == Modality.FM_SWEEP:
            # Synthesize FM sweep with discovered parameters
            start_freq = phrase_signature.features.get('start_freq', 4000)
            end_freq = phrase_signature.features.get('end_freq', 6000)
            mean_freq = phrase_signature.features.get('mean_freq', 5000)
            freq_slope = phrase_signature.features.get('freq_slope', 2000)

            # Linear sweep
            instantaneous_freq = start_freq + (end_freq - start_freq) * t / (duration_samples / sample_rate)
            audio = np.sin(2 * np.pi * cumulative_sum(instantaneous_freq) * (1/sample_rate))

            # Apply envelope
            envelope = self._create_envelope(t, 'linear')
            audio = audio * envelope

        elif phrase_signature.modality == Modality.TRANSIENT:
            # Synthesize transient/click
            centroid = phrase_signature.features.get('spectral_centroid', 5000)
            kurtosis = phrase_signature.features.get('kurtosis', 3.0)

            # Create damped sinusoid
            decay = np.exp(-t * 5)  # Exponential decay
            carrier = np.sin(2 * np.pi * centroid * t)
            audio = decay * carrier

            # Apply envelope for click-like sound
            envelope = np.zeros_like(t)
            envelope[:len(t)//10] = np.ones(len(t)//10)  # Short duration
            audio = audio * envelope

        elif phrase_signature.modality == Modality.RHYTHMIC:
            # Synthesize rhythmic pattern
            tempo = phrase_signature.features.get('tempo', 120)  # BPM
            strength = phrase_signature.features.get('rhythmic_strength', 0.5)

            # Generate rhythm based on tempo
            beat_duration = 60.0 / tempo  # seconds per beat
            beat_samples = int(beat_duration * sample_rate)

            audio = np.zeros_like(t)
            for beat in range(int(duration_samples / beat_samples)):
                beat_start = beat * beat_samples
                beat_end = min(beat_start + beat_samples//4, len(audio))  # Quarter note
                audio[beat_start:beat_end] = strength * np.sin(2 * np.pi * 1000 * t[beat_start:beat_end])

        else:
            # Default: simple sine wave
            audio = np.sin(2 * np.pi * 440 * t)

        # Normalize amplitude
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.5  # Normalize to 50% max

        return audio

    def _create_envelope(self, t: np.ndarray, envelope_type: str = 'exponential') -> np.ndarray:
        """
        Create amplitude envelope for audio synthesis.

        Args:
            t: Time array
            envelope_type: Type of envelope ('exponential', 'linear', 'gaussian')

        Returns:
            Amplitude envelope
        """
        duration = t[-1]

        if envelope_type == 'exponential':
            # Exponential rise and fall (natural for vocalizations)
            attack_time = duration * 0.1
            decay_time = duration * 0.3
            sustain_level = 0.7
            release_time = duration * 0.1

            envelope = np.zeros_like(t)

            # Attack
            attack_idx = np.where(t <= attack_time)[0]
            if len(attack_idx) > 0:
                envelope[attack_idx] = np.power(t[attack_idx] / attack_time, 2)

            # Decay
            decay_idx = np.where((t > attack_time) & (t <= attack_time + decay_time))[0]
            if len(decay_idx) > 0:
                decay_progress = (t[decay_idx] - attack_time) / decay_time
                envelope[decay_idx] = 1.0 - (1.0 - sustain_level) * decay_progress

            # Sustain
            sustain_idx = np.where((t > attack_time + decay_time) &
                                 (t <= duration - release_time))[0]
            if len(sustain_idx) > 0:
                envelope[sustain_idx] = sustain_level

            # Release
            release_idx = np.where(t > duration - release_time)[0]
            if len(release_idx) > 0:
                release_progress = (t[release_idx] - (duration - release_time)) / release_time
                envelope[release_idx] = sustain_level * (1.0 - release_progress)

        elif envelope_type == 'linear':
            # Linear fade in/out
            fade_time = duration * 0.1
            envelope = np.ones_like(t)

            # Fade in
            fade_in_idx = np.where(t <= fade_time)[0]
            if len(fade_in_idx) > 0:
                envelope[fade_in_idx] = t[fade_in_idx] / fade_time

            # Fade out
            fade_out_idx = np.where(t >= duration - fade_time)[0]
            if len(fade_out_idx) > 0:
                envelope[fade_out_idx] = (duration - t[fade_out_idx]) / fade_time

        elif envelope_type == 'gaussian':
            # Gaussian envelope (smooth)
            center = duration / 2
            std = duration / 4
            envelope = np.exp(-0.5 * ((t - center) / std) ** 2)

        else:
            envelope = np.ones_like(t)

        return envelope

    def generate_variations(self, base_sequence: List[int],
                          num_variations: int = 5,
                          temperature_range: Tuple[float, float] = (0.5, 2.0)) -> List[List[int]]:
        """
        Generate variations of a base sequence.

        Args:
            base_sequence: Original sequence of phrase IDs
            num_variations: Number of variations to generate
            temperature_range: Range of temperature values for sampling

        Returns:
            List of varied sequences
        """
        variations = []

        temperatures = np.linspace(temperature_range[0], temperature_range[1], num_variations)

        for temp in temperatures:
            variation = self.generate_sequence(
                num_phrases=len(base_sequence),
                start_phrase=base_sequence[0] if base_sequence else None,
                temperature=temp
            )
            variations.append(variation)

        self.logger.info(f"Generated {len(variations)} sequence variations")
        return variations

    def get_statistics(self) -> Dict[str, Any]:
        """Get synthesizer statistics."""
        if not self.grammar:
            return {'status': 'no_grammar'}

        stats = {
            'vocabulary_size': len(self.vocabulary),
            'grammar_rules': len(self.grammar),
            'most_common_transitions': self.grammar.most_common(5),
            'phrase_coverage': len(set([pid for seq in self.generate_sequence(10) for pid in seq])) / len(self.vocabulary)
        }

        return stats

    def __repr__(self) -> str:
        return (f"UniversalSynthesizer(vocabulary_size={len(self.vocabulary)}, "
                f"grammar_rules={len(self.grammar)})")


def cumulative_sum(array: np.ndarray) -> np.ndarray:
    """Compute cumulative sum of an array (for FM synthesis)."""
    return np.cumsum(array)