#!/usr/bin/env python3
"""
Neural Vocoder for Acoustic Synthesis (Direction 6)
====================================================

Neural network-based vocoder that generates audio directly from 112D features.

Replaces granular concatenation with:
1. NeuralVocoder - Generate audio from feature sequences
2. FeatureInterpolator - Smooth interpolation between features
3. ProsodicModifier - Modify pitch, duration, amplitude

Uses lightweight numpy implementation for testing, with PyTorch fallback
for production training.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging
import pickle
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


class NeuralVocoder:
    """
    Neural vocoder for generating audio from 112D features.

    Uses a lightweight neural network to synthesize audio directly
    from feature vectors, enabling smooth interpolation and
    prosodic modification.
    """

    def __init__(
        self,
        model_type: str = "simple",
        sample_rate: int = 48000,
        frame_size_ms: float = 10.0,
        hop_size_ms: float = 2.5,
    ):
        """
        Initialize neural vocoder.

        Args:
            model_type: Type of vocoder model ("simple", "hifi_gan", "waveglow")
            sample_rate: Output audio sample rate
            frame_size_ms: Analysis frame size in milliseconds
            hop_size_ms: Analysis hop size in milliseconds
        """
        self.model_type = model_type
        self.sample_rate = sample_rate
        self.frame_size_ms = frame_size_ms
        self.hop_size_ms = hop_size_ms

        # Calculate frame/hop sizes in samples
        self.frame_size = int(sample_rate * frame_size_ms / 1000)
        self.hop_size = int(sample_rate * hop_size_ms / 1000)

        # Initialize simple model parameters (for testing)
        self._init_simple_model()

        logger.info(
            f"NeuralVocoder initialized: model={model_type}, "
            f"sr={sample_rate}, frame={frame_size_ms}ms, hop={hop_size_ms}ms"
        )

    def _init_simple_model(self):
        """Initialize simple vocoder parameters."""
        # Feature to audio mapping (simplified)
        # In production, this would be a trained neural network
        np.random.seed(42)
        self.feature_weights = np.random.randn(112, self.frame_size).astype(np.float32) * 0.01

    def train(
        self,
        features: List[np.ndarray],
        audio: List[np.ndarray],
        epochs: int = 10,
        learning_rate: float = 0.001,
    ) -> List[float]:
        """
        Train the vocoder on feature-audio pairs.

        Args:
            features: List of 112D feature sequences
            audio: List of corresponding audio buffers
            epochs: Number of training epochs
            learning_rate: Learning rate

        Returns:
            List of losses per epoch
        """
        losses = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            count = 0

            for feat_seq, audio_buf in zip(features, audio):
                # Simplified training: adjust weights to minimize error
                loss = self._train_step(feat_seq, audio_buf, learning_rate)
                epoch_loss += loss
                count += 1

            avg_loss = epoch_loss / max(count, 1)
            losses.append(avg_loss)

            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch}, loss: {avg_loss:.4f}")

        return losses

    def _train_step(self, features: np.ndarray, audio: np.ndarray, learning_rate: float) -> float:
        """Single training step."""
        # Simplified: nudge weights toward better reconstruction
        # In production, this would use backpropagation
        n_frames = min(len(features), len(audio) // self.hop_size)

        loss = 0.0
        for i in range(n_frames):
            # Get audio frame
            start = i * self.hop_size
            end = min(start + self.frame_size, len(audio))
            if end - start < self.frame_size:
                continue

            audio_frame = audio[start:end]

            # Simple gradient: move weights toward target
            if len(audio_frame) == self.frame_size:
                # Compute error (simplified)
                predicted = features[i] @ self.feature_weights
                error = predicted - audio_frame

                # Update weights
                grad = np.outer(features[i], error)
                self.feature_weights -= learning_rate * grad

                loss += float(np.mean(error**2))

        return loss / max(n_frames, 1)

    def synthesize(self, features: np.ndarray) -> np.ndarray:
        """
        Generate audio from feature sequence.

        Args:
            features: Feature sequence of shape (n_frames, 112)

        Returns:
            Audio buffer at sample_rate
        """
        if len(features) == 0:
            return np.array([], dtype=np.float32)

        n_frames = len(features)

        # Calculate output length
        output_length = (n_frames - 1) * self.hop_size + self.frame_size
        output = np.zeros(output_length, dtype=np.float32)

        # Generate audio frame by frame
        for i, frame_features in enumerate(features):
            start = i * self.hop_size
            end = start + self.frame_size

            if end > len(output):
                end = len(output)

            # Synthesize frame
            audio_frame = self._synthesize_frame(frame_features)

            # Overlap-add
            frame_len = min(len(audio_frame), end - start)
            output[start : start + frame_len] += audio_frame[:frame_len]

        # Normalize to prevent clipping
        max_val = np.max(np.abs(output))
        if max_val > 0:
            output = output / max_val * 0.95

        return output

    def _synthesize_frame(self, features: np.ndarray) -> np.ndarray:
        """Synthesize single audio frame from features."""
        # Simple synthesis: feature weights
        audio = features @ self.feature_weights

        # Apply envelope for smoothness
        envelope = np.hanning(len(audio))
        audio = audio * envelope

        return audio.astype(np.float32)

    def synthesize_batch(self, features_list: List[np.ndarray]) -> List[np.ndarray]:
        """
        Synthesize multiple feature sequences efficiently.

        Args:
            features_list: List of feature sequences

        Returns:
            List of audio buffers
        """
        results = []
        for features in features_list:
            audio = self.synthesize(features)
            results.append(audio)
        return results

    def get_metadata(self) -> dict:
        """Get model metadata."""
        return {
            "version": "0.1.0",
            "model_type": self.model_type,
            "sample_rate": self.sample_rate,
            "frame_size_ms": self.frame_size_ms,
            "hop_size_ms": self.hop_size_ms,
        }

    def save(self, path: str) -> None:
        """Save model to file."""
        data = {
            "model_type": self.model_type,
            "sample_rate": self.sample_rate,
            "frame_size_ms": self.frame_size_ms,
            "hop_size_ms": self.hop_size_ms,
            "feature_weights": self.feature_weights.tolist(),
            "metadata": self.get_metadata(),
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Vocoder saved to {path}")

    @classmethod
    def load(cls, path: str) -> "NeuralVocoder":
        """Load model from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        vocoder = cls(
            model_type=data["model_type"],
            sample_rate=data["sample_rate"],
            frame_size_ms=data["frame_size_ms"],
            hop_size_ms=data["hop_size_ms"],
        )
        vocoder.feature_weights = np.array(data["feature_weights"], dtype=np.float32)

        logger.info(f"Vocoder loaded from {path}")
        return vocoder


class FeatureInterpolator:
    """Interpolate between feature vectors for smooth transitions."""

    @staticmethod
    def linear(f1: np.ndarray, f2: np.ndarray, t: float) -> np.ndarray:
        """
        Linear interpolation between feature vectors.

        Args:
            f1: First feature vector
            f2: Second feature vector
            t: Interpolation parameter in [0, 1]

        Returns:
            Interpolated feature vector
        """
        t = np.clip(t, 0.0, 1.0)
        return (1 - t) * f1 + t * f2

    @staticmethod
    def slerp(f1: np.ndarray, f2: np.ndarray, t: float) -> np.ndarray:
        """
        Spherical linear interpolation for normalized features.

        Preserves the norm of feature vectors during interpolation.

        Args:
            f1: First feature vector (normalized)
            f2: Second feature vector (normalized)
            t: Interpolation parameter in [0, 1]

        Returns:
            Interpolated feature vector (normalized)
        """
        t = np.clip(t, 0.0, 1.0)

        # Normalize inputs
        f1_norm = f1 / (np.linalg.norm(f1) + 1e-8)
        f2_norm = f2 / (np.linalg.norm(f2) + 1e-8)

        # Compute angle
        dot = np.clip(np.dot(f1_norm, f2_norm), -1.0, 1.0)
        theta = np.arccos(dot)

        if theta < 1e-6:
            # Vectors are parallel, use linear interpolation
            return FeatureInterpolator.linear(f1, f2, t)

        sin_theta = np.sin(theta)

        # Slerp formula
        w1 = np.sin((1 - t) * theta) / sin_theta
        w2 = np.sin(t * theta) / sin_theta

        result = w1 * f1 + w2 * f2

        # Preserve norm
        norm = np.linalg.norm(result)
        if norm > 1e-8:
            result = result / norm * np.linalg.norm(f1)

        return result.astype(np.float32)

    @staticmethod
    def interpolate_sequence(features: np.ndarray, n_interp: int = 2) -> np.ndarray:
        """
        Interpolate between consecutive feature frames.

        Args:
            features: Feature sequence of shape (n_frames, n_features)
            n_interp: Number of interpolated frames between each pair

        Returns:
            Interpolated feature sequence
        """
        if len(features) < 2:
            return features

        result = []

        for i in range(len(features) - 1):
            result.append(features[i])

            # Add interpolated frames
            for j in range(1, n_interp + 1):
                t = j / (n_interp + 1)
                interp = FeatureInterpolator.linear(features[i], features[i + 1], t)
                result.append(interp)

        result.append(features[-1])

        return np.array(result, dtype=np.float32)


class ProsodicModifier:
    """Modify prosody of synthesized audio via feature manipulation."""

    @staticmethod
    def adjust_pitch(features: np.ndarray, shift_semitones: float) -> np.ndarray:
        """
        Pitch shift by modifying F0-related features.

        Args:
            features: Feature sequence (n_frames, 112)
            shift_semitones: Pitch shift in semitones (+/-)

        Returns:
            Pitch-shifted features
        """
        result = features.copy()

        # F0 conversion factor: 2^(semitones/12)
        factor = 2.0 ** (shift_semitones / 12.0)

        # Assume F0 is in first dimension
        # Adjust F0
        result[:, 0] = features[:, 0] * factor

        # Also adjust related spectral features
        # Higher pitch = compressed spectrum, lower = expanded
        for col in range(2, min(20, features.shape[1])):
            if shift_semitones > 0:
                # Shift up: move features to earlier bins
                result[:, col] = features[:, col] * (1.0 + shift_semitones * 0.01)
            else:
                # Shift down: move features to later bins
                result[:, col] = features[:, col] * (1.0 + shift_semitones * 0.01)

        return result.astype(np.float32)

    @staticmethod
    def adjust_duration(features: np.ndarray, speed_factor: float) -> np.ndarray:
        """
        Time stretch by resampling feature sequence.

        Args:
            features: Feature sequence (n_frames, 112)
            speed_factor: Speed factor (>1 = faster, <1 = slower)

        Returns:
            Time-stretched features
        """
        if speed_factor == 1.0:
            return features

        # Calculate new length
        new_len = int(len(features) / speed_factor)
        if new_len < 1:
            new_len = 1

        # Resample using linear interpolation
        result = np.zeros((new_len, features.shape[1]), dtype=np.float32)

        for i in range(new_len):
            # Source position
            src_pos = i * speed_factor
            src_idx = int(src_pos)
            src_frac = src_pos - src_idx

            if src_idx < len(features) - 1:
                # Linear interpolation between frames
                result[i] = (1 - src_frac) * features[src_idx] + src_frac * features[src_idx + 1]
            elif src_idx < len(features):
                result[i] = features[src_idx]

        return result

    @staticmethod
    def adjust_amplitude(features: np.ndarray, gain_db: float) -> np.ndarray:
        """
        Amplitude scaling via energy-related features.

        Args:
            features: Feature sequence (n_frames, 112)
            gain_db: Gain in decibels (+/-)

        Returns:
            Amplitude-modified features
        """
        result = features.copy()

        # dB to linear: 10^(dB/20) for amplitude
        factor = 10.0 ** (gain_db / 20.0)

        # Assume RMS energy is in second dimension
        result[:, 1] = features[:, 1] * factor

        # Also adjust related energy features
        for col in range(2, min(10, features.shape[1])):
            result[:, col] = features[:, col] * factor

        return result.astype(np.float32)


def create_vocoder(model_type: str = "simple", sample_rate: int = 48000) -> NeuralVocoder:
    """
    Create a neural vocoder instance.

    Args:
        model_type: Type of vocoder model
        sample_rate: Output audio sample rate

    Returns:
        Configured NeuralVocoder
    """
    return NeuralVocoder(model_type=model_type, sample_rate=sample_rate)


def main():
    """Command-line interface for neural vocoder."""
    import argparse

    parser = argparse.ArgumentParser(description="Neural Vocoder for Direction 6")
    parser.add_argument("--model", type=str, default="simple", help="Model type")
    parser.add_argument("--sample-rate", type=int, default=48000, help="Sample rate")
    parser.add_argument("--train", type=str, help="Training data file")
    parser.add_argument("--output", type=str, help="Output model file")
    parser.add_argument("--synthesize", action="store_true", help="Synthesis mode")
    parser.add_argument("--features", type=str, help="Input features file")
    parser.add_argument("--audio", type=str, help="Output audio file")

    args = parser.parse_args()

    if args.train:
        # Load training data
        with open(args.train, "rb") as f:
            data = pickle.load(f)
        features = data["features"]
        audio = data["audio"]

        # Create and train vocoder
        vocoder = NeuralVocoder(model_type=args.model, sample_rate=args.sample_rate)
        vocoder.train(features, audio, epochs=20)
        vocoder.save(args.output)
        print(f"Model saved to {args.output}")

    elif args.synthesize:
        # Load model
        vocoder = NeuralVocoder.load(args.output)

        # Load features
        with open(args.features, "rb") as f:
            data = pickle.load(f)
        features = data["features"]

        # Synthesize
        audio = vocoder.synthesize(features)

        # Save audio
        import soundfile as sf

        sf.write(args.audio, audio, vocoder.sample_rate)
        print(f"Audio saved to {args.audio}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
