"""
Probabilistic Context State Machine
===================================

Advanced probabilistic state machine for context detection using
Bayesian inference and temporal modeling.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import collections
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np

# Try to import scipy for advanced signal processing
try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class ContextState(Enum):
    """Context states for the state machine"""
    SILENCE = 'silence'
    CONTACT = 'contact'
    ALARM = 'alarm'
    FOOD = 'food'
    NEUTRAL = 'neutral'
    UNCERTAIN = 'uncertain'


@dataclass
class AudioFeatures:
    """Audio features for context detection"""
    rms: float
    spectral_centroid: float
    bandwidth: float
    zero_crossing_rate: float
    harmonic_ratio: float
    fundamental_freq: float
    spectral_flatness: float
    temporal_envelope: np.ndarray
    mfcc_features: np.ndarray

    @property
    def feature_vector(self) -> np.ndarray:
        """Get flattened feature vector"""
        return np.array([
            self.rms,
            self.spectral_centroid,
            self.bandwidth,
            self.zero_crossing_rate,
            self.harmonic_ratio,
            self.fundamental_freq,
            self.spectral_flatness,
            *self.temporal_envelope,
            *self.mfcc_features
        ])


class ProbabilisticContextMachine:
    """
    Probabilistic state machine for context detection.

    Uses Bayesian inference with temporal smoothing and confidence
    estimation for robust context detection.
    """

    def __init__(self,
                 context_states: List[ContextState] = None,
                 history_length: int = 5,
                 confidence_threshold: float = 0.7,
                 transition_memory: int = 3):
        """
        Initialize the probabilistic context machine.

        Args:
            context_states: List of context states to track
            history_length: Length of feature history for temporal analysis
            confidence_threshold: Minimum confidence for state commitment
            transition_memory: Memory for state transition modeling
        """
        self.context_states = context_states or [ContextState.SILENCE, ContextState.CONTACT,
                                                ContextState.ALARM, ContextState.FOOD, ContextState.NEUTRAL]

        self.history_length = history_length
        self.confidence_threshold = confidence_threshold
        self.transition_memory = transition_memory

        # Initialize state tracking
        self.current_state = ContextState.SILENCE
        self.state_history = collections.deque(maxlen=history_length)
        self.transition_matrix = np.ones((len(self.context_states), len(self.context_states))) / len(self.context_states)

        # Feature history for temporal analysis
        self.feature_history = collections.deque(maxlen=history_length)
        self.confidence_history = collections.deque(maxlen=history_length)

        # Context probability models (simplified Gaussian Mixture Models)
        self.context_models = {}
        self._initialize_context_models()

        # Temporal smoothing parameters
        self.smoothing_factor = 0.7
        self.prediction_weight = 0.2

        logger.info(f"ProbabilisticContextMachine initialized with {len(self.context_states)} states")

    def _initialize_context_models(self):
        """Initialize statistical models for each context state."""
        # Simplified feature distributions for each context
        # In production, these would be learned from training data

        self.context_models = {
            ContextState.SILENCE: {
                'rms_range': (0.0, 0.01),
                'freq_range': (0, 1000),
                'zcr_range': (0.0, 0.1)
            },
            ContextState.CONTACT: {
                'rms_range': (0.05, 0.3),
                'freq_range': (4000, 7000),  # Extended range
                'zcr_range': (0.01, 0.2)
            },
            ContextState.ALARM: {
                'rms_range': (0.1, 0.4),
                'freq_range': (6000, 10000),  # Extended range
                'zcr_range': (0.01, 0.25)
            },
            ContextState.FOOD: {
                'rms_range': (0.05, 0.2),
                'freq_range': (4500, 7500),  # Extended range
                'zcr_range': (0.01, 0.2)
            },
            ContextState.NEUTRAL: {
                'rms_range': (0.02, 0.15),
                'freq_range': (2000, 6000),
                'zcr_range': (0.01, 0.25)
            }
        }

    def extract_features(self, audio: np.ndarray, sr: int = 44100) -> AudioFeatures:
        """
        Extract comprehensive audio features for context analysis.

        Args:
            audio: Input audio array
            sr: Sample rate

        Returns:
            AudioFeatures object containing all extracted features
        """
        try:
            # Basic features
            rms = np.sqrt(np.mean(audio ** 2))

            # Spectral features
            fft = np.fft.rfft(audio)
            magnitudes = np.abs(fft)
            freqs = np.fft.rfftfreq(len(audio), 1/sr)

            # Remove DC component
            if len(magnitudes) > 1:
                magnitudes = magnitudes[1:]
                freqs = freqs[1:]

            # Spectral centroid
            spectral_centroid = np.sum(freqs * magnitudes) / np.sum(magnitudes) if np.sum(magnitudes) > 0 else 0

            # Spectral bandwidth
            weighted_freqs = freqs - spectral_centroid
            bandwidth = np.sqrt(np.sum(weighted_freqs**2 * magnitudes) / np.sum(magnitudes)) if np.sum(magnitudes) > 0 else 0

            # Zero crossing rate
            zcr = np.sum(np.diff(np.sign(audio)) != 0) / (2 * len(audio))

            # Harmonic ratio (simplified)
            harmonic_ratio = self._estimate_harmonic_ratio(magnitudes)

            # Fundamental frequency (simplified)
            fundamental_freq = self._estimate_fundamental_freq(audio, sr)

            # Spectral flatness (noisiness measure)
            spectral_flatness = self._calculate_spectral_flatness(magnitudes)

            # Temporal envelope
            if SCIPY_AVAILABLE:
                try:
                    envelope = np.abs(signal.hilbert(audio))
                except Exception:
                    envelope = np.abs(audio)
            else:
                # Fallback to simple absolute value
                envelope = np.abs(audio)

            temporal_envelope = np.array([
                np.mean(envelope),
                np.std(envelope) if len(envelope) > 1 else 0,
                np.max(envelope),
                np.min(envelope)
            ])

            # MFCC features (simplified - first 13 coefficients)
            mfcc_features = self._calculate_mfcc(audio, sr)

            return AudioFeatures(
                rms=rms,
                spectral_centroid=spectral_centroid,
                bandwidth=bandwidth,
                zero_crossing_rate=zcr,
                harmonic_ratio=harmonic_ratio,
                fundamental_freq=fundamental_freq,
                spectral_flatness=spectral_flatness,
                temporal_envelope=temporal_envelope,
                mfcc_features=mfcc_features
            )

        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            # Return default features
            return self._get_default_features()

    def _estimate_harmonic_ratio(self, magnitudes: np.ndarray) -> float:
        """Estimate harmonic-to-noise ratio."""
        if len(magnitudes) < 3:
            return 0.0

        # Find peaks (potential harmonics)
        peaks = []
        for i in range(1, len(magnitudes)-1):
            if magnitudes[i] > magnitudes[i-1] and magnitudes[i] > magnitudes[i+1]:
                peaks.append(i)

        # Calculate harmonic ratio based on peak strength
        if peaks:
            peak_energy = sum(magnitudes[p] for p in peaks[:3])  # Top 3 peaks
            total_energy = np.sum(magnitudes)
            return peak_energy / total_energy if total_energy > 0 else 0.0

        return 0.0

    def _estimate_fundamental_freq(self, audio: np.ndarray, sr: int) -> float:
        """Estimate fundamental frequency using autocorrelation."""
        if len(audio) < 64:
            return 0.0

        # Use shorter segment for efficiency
        segment = audio[:4096] if len(audio) >= 4096 else audio

        # Autocorrelation
        correlation = np.correlate(segment, segment, mode='full')
        correlation = correlation[len(correlation)//2:]

        # Find first peak
        for i in range(1, len(correlation)-1):
            if correlation[i] > correlation[i-1] and correlation[i] > correlation[i+1]:
                if correlation[i] > 0.1 * np.max(correlation):
                    return sr / i

        return 0.0

    def _calculate_spectral_flatness(self, magnitudes: np.ndarray) -> float:
        """Calculate spectral flatness (measure of noisiness)."""
        if len(magnitudes) == 0 or np.sum(magnitudes) == 0:
            return 1.0

        geometric_mean = np.exp(np.mean(np.log(magnitudes + 1e-10)))
        arithmetic_mean = np.mean(magnitudes)

        return geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 1.0

    def _calculate_mfcc(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Calculate simplified MFCC features."""
        # Simplified MFCC calculation (production would use librosa)
        if len(audio) < 256:
            return np.zeros(13)

        # Frame the audio
        frame_length = 256
        hop_length = 128
        frames = []

        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            frames.append(frame)

        if not frames:
            return np.zeros(13)

        # Calculate energy in each frame
        frame_energies = [np.mean(frame ** 2) for frame in frames]

        # Simple MFCC approximation
        mfcc = np.array(frame_energies[:13])  # Simplified

        # Normalize
        if np.std(mfcc) > 0:
            mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)

        return mfcc

    def _get_default_features(self) -> AudioFeatures:
        """Get default features for error cases."""
        return AudioFeatures(
            rms=0.0,
            spectral_centroid=0.0,
            bandwidth=0.0,
            zero_crossing_rate=0.0,
            harmonic_ratio=0.0,
            fundamental_freq=0.0,
            spectral_flatness=1.0,
            temporal_envelope=np.zeros(4),
            mfcc_features=np.zeros(13)
        )

    def calculate_context_probabilities(self, features: AudioFeatures) -> Dict[ContextState, float]:
        """
        Calculate probabilities for each context state.

        Args:
            features: Extracted audio features

        Returns:
            Dictionary of state probabilities
        """
        probabilities = {}

        for state in self.context_states:
            # Calculate likelihood based on feature similarity to state model
            likelihood = self._calculate_likelihood(features, state)

            # Apply temporal smoothing based on state history
            smoothing = self._apply_temporal_smoothing(state, likelihood)

            # Apply transition matrix
            transition_prob = self._get_transition_probability(state)

            # Combine likelihoods
            probabilities[state] = likelihood * smoothing * transition_prob

        # Normalize probabilities
        total = sum(probabilities.values())
        if total > 0:
            probabilities = {state: prob/total for state, prob in probabilities.items()}

        return probabilities

    def _calculate_likelihood(self, features: AudioFeatures, state: ContextState) -> float:
        """Calculate likelihood of features given state model."""
        model = self.context_models.get(state, self.context_models[ContextState.NEUTRAL])

        # Check feature ranges
        rms_match = self._feature_match(features.rms, model['rms_range'])
        freq_match = self._feature_match(features.spectral_centroid, model['freq_range'])
        zcr_match = self._feature_match(features.zero_crossing_rate, model['zcr_range'])

        # Weight frequency match more heavily for sine waves
        freq_weight = 2.0
        rms_weight = 0.5
        zcr_weight = 0.5

        # Combine matches (weighted geometric mean)
        weighted_matches = [
            np.log(rms_match + 1e-10) * rms_weight,
            np.log(freq_match + 1e-10) * freq_weight,
            np.log(zcr_match + 1e-10) * zcr_weight
        ]
        return np.exp(np.mean(weighted_matches))

    def _feature_match(self, value: float, range_bounds: Tuple[float, float]) -> float:
        """Calculate how well a value matches expected range."""
        lower, upper = range_bounds

        if lower <= value <= upper:
            return 1.0
        elif value < lower:
            # Value too low - exponential decay
            return np.exp(-(lower - value) / (lower + 1e-10)) if lower > 0 else 0.0
        else:
            # Value too high - gentler decay
            return np.exp(-(value - upper) / (upper + 10.0))

    def _apply_temporal_smoothing(self, state: ContextState, likelihood: float) -> float:
        """Apply temporal smoothing based on state history."""
        if not self.state_history:
            return likelihood

        # Count recent occurrences of this state
        recent_count = sum(1 for s in list(self.state_history)[-self.transition_memory:] if s == state)

        # Adjust likelihood based on temporal consistency
        temporal_factor = 1.0 + (recent_count * self.smoothing_factor / self.transition_memory)

        return likelihood * temporal_factor

    def _get_transition_probability(self, target_state: ContextState) -> float:
        """Get transition probability based on transition matrix."""
        try:
            state_idx = list(self.context_states).index(target_state)
            return self.transition_matrix[state_idx][state_idx]  # Self-transition probability
        except (ValueError, IndexError):
            return 1.0 / len(self.context_states)

    def update_state_machine(self, audio: np.ndarray, sr: int = 44100) -> Tuple[ContextState, float, Dict[ContextState, float]]:
        """
        Update the state machine with new audio input.

        Args:
            audio: Input audio array
            sr: Sample rate

        Returns:
            Tuple of (predicted_state, confidence, probabilities)
        """
        # Extract features
        features = self.extract_features(audio, sr)
        self.feature_history.append(features)

        # Calculate context probabilities
        probabilities = self.calculate_context_probabilities(features)

        # Select best state with confidence threshold
        best_state = max(probabilities, key=probabilities.get)
        best_confidence = probabilities[best_state]

        # Apply confidence threshold
        if best_confidence < self.confidence_threshold:
            predicted_state = ContextState.UNCERTAIN
            confidence = best_confidence
        else:
            predicted_state = best_state
            confidence = best_confidence

            # Update transition matrix for learning
            self._update_transition_matrix(self.current_state, predicted_state)

        # Update state history
        self.state_history.append(predicted_state)
        self.current_state = predicted_state

        # Store confidence for temporal analysis
        self.confidence_history.append(confidence)

        # Debug logging
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Context probabilities: {[(s.value, p) for s, p in probabilities.items()]}")
            logger.debug(f"Predicted state: {predicted_state.value}, confidence: {confidence:.3f}")

        return predicted_state, confidence, probabilities

    def _update_transition_matrix(self, from_state: ContextState, to_state: ContextState):
        """Update transition matrix for learning (simplified)."""
        try:
            from_idx = list(self.context_states).index(from_state)
            to_idx = list(self.context_states).index(to_state)

            # Simple learning rate
            learning_rate = 0.01

            # Update transition probability
            current_prob = self.transition_matrix[from_idx][to_idx]
            self.transition_matrix[from_idx][to_idx] = min(1.0, current_prob + learning_rate)

            # Normalize row
            row_sum = np.sum(self.transition_matrix[from_idx])
            if row_sum > 0:
                self.transition_matrix[from_idx] /= row_sum

        except (ValueError, IndexError):
            pass  # Skip if states not found

    def get_state_history(self) -> List[ContextState]:
        """Get state history for analysis."""
        return list(self.state_history)

    def get_confidence_trend(self) -> List[float]:
        """Get confidence history for trend analysis."""
        return list(self.confidence_history)

    def reset(self):
        """Reset the state machine."""
        self.current_state = ContextState.SILENCE
        self.state_history.clear()
        self.feature_history.clear()
        self.confidence_history.clear()
        logger.info("ProbabilisticContextMachine reset")
