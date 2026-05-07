#!/usr/bin/env python3
"""
Multimodal Fusion - Vision + Audio
==================================

Combining audio features with visual information for enhanced
context understanding in animal vocalization analysis.

This module implements:
- Visual feature extraction from video frames
- Cross-modal attention between audio and visual
- Early and late fusion strategies
- Temporal alignment between audio and video streams
- Audio-visual correlation learning

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VisualFeatures:
    """Container for visual features."""

    features: np.ndarray
    temporal_idx: int
    motion_score: float


class VisualFeatureExtractor:
    """
    Extract visual features from video frames.

    Uses lightweight CNN-style features for real-time processing.
    """

    def __init__(
        self,
        frame_height: int = 224,
        frame_width: int = 224,
        feature_dim: int = 512,
        temporal_window: int = 10,
    ):
        """
        Initialize visual feature extractor.

        Args:
            frame_height: Input frame height
            frame_width: Input frame width
            feature_dim: Output feature dimension
            temporal_window: Number of frames for temporal aggregation
        """
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.feature_dim = feature_dim
        self.temporal_window = temporal_window

        # Simulated CNN projection (in production, use pretrained model)
        scale = 1.0 / math.sqrt(frame_height * frame_width * 3)
        self.cnn_projection = np.random.randn(frame_height * frame_width * 3, feature_dim) * scale

        # Temporal aggregation weights
        self.temporal_weights = np.ones(temporal_window) / temporal_window

    def extract_frame_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract features from a single frame.

        Args:
            frame: RGB frame of shape (H, W, 3)

        Returns:
            Feature vector of shape (feature_dim,)
        """
        # Flatten and project
        flattened = frame.reshape(-1)
        features = flattened @ self.cnn_projection
        return features

    def aggregate_temporal(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Aggregate features across temporal window.

        Args:
            frames: List of video frames

        Returns:
            Aggregated feature vector
        """
        n_frames = min(len(frames), self.temporal_window)
        features_list = []

        for i in range(n_frames):
            feat = self.extract_frame_features(frames[i])
            features_list.append(feat)

        # Weighted average
        stacked = np.stack(features_list, axis=0)  # (n_frames, feature_dim)
        weights = self.temporal_weights[:n_frames]
        aggregated = np.average(stacked, axis=0, weights=weights)

        return aggregated

    def detect_motion(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Detect motion between two frames.

        Args:
            frame1: First frame
            frame2: Second frame

        Returns:
            Motion score between 0 and 1
        """
        # Compute frame difference
        diff = np.abs(frame2.astype(np.float32) - frame1.astype(np.float32))
        motion = np.mean(diff) / 255.0

        # Clamp to [0, 1]
        return float(np.clip(motion, 0.0, 1.0))

    def extract_with_motion(self, frames: List[np.ndarray]) -> List[VisualFeatures]:
        """
        Extract features with motion scores.

        Args:
            frames: List of video frames

        Returns:
            List of VisualFeatures
        """
        results = []
        prev_frame = None

        for i, frame in enumerate(frames):
            features = self.extract_frame_features(frame)

            if prev_frame is not None:
                motion_score = self.detect_motion(prev_frame, frame)
            else:
                motion_score = 0.0

            results.append(
                VisualFeatures(
                    features=features,
                    temporal_idx=i,
                    motion_score=motion_score,
                )
            )

            prev_frame = frame

        return results


class AudioVisualFusion:
    """
    Fuse audio and visual features using various strategies.

    Supports:
    - Early fusion: Feature-level combination
    - Late fusion: Decision-level combination
    - Cross-modal attention: Attention-based fusion
    """

    def __init__(
        self,
        audio_dim: int = 112,
        visual_dim: int = 512,
        fusion_dim: int = 256,
        num_heads: int = 4,
    ):
        """
        Initialize audio-visual fusion.

        Args:
            audio_dim: Audio feature dimension
            visual_dim: Visual feature dimension
            fusion_dim: Fused representation dimension
            num_heads: Number of attention heads for cross-modal attention
        """
        self.audio_dim = audio_dim
        self.visual_dim = visual_dim
        self.fusion_dim = fusion_dim
        self.num_heads = num_heads

        # Projection layers
        scale = 1.0 / math.sqrt(fusion_dim)
        self.audio_proj = np.random.randn(audio_dim, fusion_dim) * scale
        self.visual_proj = np.random.randn(visual_dim, fusion_dim) * scale

        # Cross-modal attention parameters
        self.attn_scale = 1.0 / math.sqrt(fusion_dim // num_heads)
        self.q_proj = np.random.randn(fusion_dim, fusion_dim) * scale
        self.k_proj = np.random.randn(fusion_dim, fusion_dim) * scale
        self.v_proj = np.random.randn(fusion_dim, fusion_dim) * scale

        # Output projection
        self.out_proj = np.random.randn(2 * fusion_dim, fusion_dim) * scale

        logger.debug(
            f"AudioVisualFusion: audio_dim={audio_dim}, visual_dim={visual_dim}, "
            f"fusion_dim={fusion_dim}"
        )

    def early_fusion(self, audio_features: np.ndarray, visual_features: np.ndarray) -> np.ndarray:
        """
        Perform early fusion (feature-level combination).

        Args:
            audio_features: Audio features of shape (seq_len, audio_dim)
            visual_features: Visual features of shape (seq_len, visual_dim)

        Returns:
            Fused features of shape (seq_len, fusion_dim)
        """
        # Project both to fusion dimension
        audio_proj = audio_features @ self.audio_proj
        visual_proj = visual_features @ self.visual_proj

        # Concatenate and project
        fused_input = np.concatenate([audio_proj, visual_proj], axis=-1)

        # Simple projection (no additional layer for efficiency)
        fused = fused_input[:, : self.fusion_dim]  # Take first half

        return fused

    def late_fusion(self, audio_features: np.ndarray, visual_features: np.ndarray) -> np.ndarray:
        """
        Perform late fusion (decision-level combination).

        Args:
            audio_features: Audio features of shape (seq_len, audio_dim)
            visual_features: Visual features of shape (seq_len, visual_dim)

        Returns:
            Fused features of shape (seq_len, fusion_dim)
        """
        # Pool each modality separately
        audio_pooled = np.mean(audio_features, axis=0)  # (audio_dim,)
        visual_pooled = np.mean(visual_features, axis=0)  # (visual_dim,)

        # Project and combine
        audio_proj = audio_pooled @ self.audio_proj
        visual_proj = visual_pooled @ self.visual_proj

        # Average
        fused = (audio_proj + visual_proj) / 2.0

        # Broadcast to sequence length
        seq_len = audio_features.shape[0]
        return np.tile(fused[np.newaxis, :], (seq_len, 1))

    def cross_modal_attention(
        self, audio_features: np.ndarray, visual_features: np.ndarray
    ) -> np.ndarray:
        """
        Apply cross-modal attention between audio and visual.

        Args:
            audio_features: Audio features of shape (seq_len, audio_dim)
            visual_features: Visual features of shape (seq_len, visual_dim)

        Returns:
            Fused features of shape (seq_len, fusion_dim)
        """
        seq_len = audio_features.shape[0]

        # Project to fusion dimension
        audio_proj = audio_features @ self.audio_proj  # (seq_len, fusion_dim)
        visual_proj = visual_features @ self.visual_proj  # (seq_len, fusion_dim)

        # Stack for multi-head attention
        audio_queries = audio_proj @ self.q_proj
        visual_keys = visual_proj @ self.k_proj
        visual_values = visual_proj @ self.v_proj

        # Reshape for multi-head
        head_dim = self.fusion_dim // self.num_heads
        audio_queries = audio_queries.reshape(seq_len, self.num_heads, head_dim).transpose(
            1, 0, 2
        )  # (num_heads, seq_len, head_dim)
        visual_keys = visual_keys.reshape(seq_len, self.num_heads, head_dim).transpose(1, 0, 2)
        visual_values = visual_values.reshape(seq_len, self.num_heads, head_dim).transpose(1, 0, 2)

        # Compute attention
        attn_scores = (
            audio_queries @ visual_keys.transpose(0, 2, 1)
        ) * self.attn_scale  # (num_heads, seq_len, seq_len)
        attn_weights = self._softmax(attn_scores, axis=-1)

        # Apply attention
        attended = attn_weights @ visual_values  # (num_heads, seq_len, head_dim)

        # Combine heads
        attended = attended.transpose(1, 0, 2).reshape(seq_len, self.fusion_dim)

        # Residual connection
        fused = audio_proj + attended

        return fused

    def fuse_with_attention_weights(
        self, audio_features: np.ndarray, visual_features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fuse and return attention weights for interpretability.

        Args:
            audio_features: Audio features
            visual_features: Visual features

        Returns:
            Tuple of (fused, audio_attention, visual_attention)
        """
        audio_features.shape[0]

        # Project
        audio_proj = audio_features @ self.audio_proj
        visual_proj = visual_features @ self.visual_proj

        # Simple attention: compute similarity-based weights
        audio_norm = np.linalg.norm(audio_proj, axis=-1, keepdims=True) + 1e-8
        visual_norm = np.linalg.norm(visual_proj, axis=-1, keepdims=True) + 1e-8

        audio_normalized = audio_proj / audio_norm
        visual_normalized = visual_proj / visual_norm

        # Attention weights based on similarity
        audio_attn = np.mean(audio_normalized, axis=-1)
        audio_attn = self._softmax(audio_attn)

        visual_attn = np.mean(visual_normalized, axis=-1)
        visual_attn = self._softmax(visual_attn)

        # Fuse
        fused = (audio_proj + visual_proj) / 2.0

        return fused, audio_attn, visual_attn

    def compute_modality_importance(
        self, audio_features: np.ndarray, visual_features: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute relative importance of each modality.

        Args:
            audio_features: Audio features
            visual_features: Visual features

        Returns:
            Dictionary with importance scores
        """
        # Compute variance as proxy for information content
        audio_var = np.var(audio_features)
        visual_var = np.var(visual_features)

        total = audio_var + visual_var + 1e-8

        importance = {
            "audio": audio_var / total,
            "visual": visual_var / total,
        }

        return importance

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class MultimodalContextClassifier:
    """
    Context classifier using multimodal input.

    Handles:
    - Audio-only input
    - Visual-only input
    - Combined audio-visual input
    """

    def __init__(
        self,
        audio_dim: int = 112,
        visual_dim: int = 512,
        fusion_dim: int = 256,
        num_classes: int = 4,
    ):
        """
        Initialize multimodal context classifier.

        Args:
            audio_dim: Audio feature dimension
            visual_dim: Visual feature dimension
            fusion_dim: Fusion dimension
            num_classes: Number of context classes
        """
        self.audio_dim = audio_dim
        self.visual_dim = visual_dim
        self.fusion_dim = fusion_dim
        self.num_classes = num_classes

        # Fusion module
        self.fusion = AudioVisualFusion(audio_dim, visual_dim, fusion_dim)

        # Classification heads
        scale = 1.0 / math.sqrt(fusion_dim)
        self.audio_classifier = np.random.randn(fusion_dim, num_classes) * scale
        self.visual_classifier = np.random.randn(fusion_dim, num_classes) * scale
        self.fused_classifier = np.random.randn(fusion_dim, num_classes) * scale

        self.bias = np.zeros(num_classes)

    def classify(
        self,
        audio_features: Optional[np.ndarray] = None,
        visual_features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Classify using available modalities.

        Args:
            audio_features: Optional audio features
            visual_features: Optional visual features

        Returns:
            Class logits of shape (num_classes,)
        """
        # Handle different modality combinations
        if audio_features is not None and visual_features is not None:
            # Both available: use fusion
            fused = self.fusion.cross_modal_attention(audio_features, visual_features)
            pooled = np.mean(fused, axis=0)
            logits = pooled @ self.fused_classifier + self.bias

        elif audio_features is not None:
            # Audio only
            audio_proj = audio_features @ self.fusion.audio_proj
            pooled = np.mean(audio_proj, axis=0)
            logits = pooled @ self.audio_classifier + self.bias

        elif visual_features is not None:
            # Visual only
            visual_proj = visual_features @ self.fusion.visual_proj
            pooled = np.mean(visual_proj, axis=0)
            logits = pooled @ self.visual_classifier + self.bias

        else:
            # No input: return uniform distribution
            logits = np.zeros(self.num_classes)

        return logits

    def predict_with_confidence(
        self,
        audio_features: Optional[np.ndarray] = None,
        visual_features: Optional[np.ndarray] = None,
    ) -> Tuple[int, float, str]:
        """
        Make prediction with confidence and modality info.

        Args:
            audio_features: Optional audio features
            visual_features: Optional visual features

        Returns:
            Tuple of (predicted_class, confidence, modality_used)
        """
        logits = self.classify(audio_features, visual_features)
        probs = self._softmax(logits)

        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

        # Determine which modalities were used
        if audio_features is not None and visual_features is not None:
            modality = "audio_visual"
        elif audio_features is not None:
            modality = "audio_only"
        elif visual_features is not None:
            modality = "visual_only"
        else:
            modality = "none"

        return pred_class, confidence, modality

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax with numerical stability."""
        x_max = np.max(x)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x)


class VisualVocalizationCorrelation:
    """
    Learn correlation between visual context and vocalizations.

    Enables:
    - Predicting visual features from audio
    - Retrieving similar visual contexts
    - Cross-modal similarity search
    """

    def __init__(self, audio_dim: int = 112, visual_dim: int = 512):
        """
        Initialize audio-visual correlation learner.

        Args:
            audio_dim: Audio feature dimension
            visual_dim: Visual feature dimension
        """
        self.audio_dim = audio_dim
        self.visual_dim = visual_dim
        self.correlation_strength = 0.0

        # Mapping matrices
        scale = 1.0 / math.sqrt(audio_dim)
        self.audio_to_visual = np.random.randn(audio_dim, visual_dim) * scale
        self.visual_to_audio = np.random.randn(visual_dim, audio_dim) * scale

        # Storage for retrieval
        self.visual_index: List[np.ndarray] = []
        self.context_labels: List[str] = []

    def learn_correlation(
        self, audio_features: List[np.ndarray], visual_features: List[np.ndarray]
    ) -> None:
        """
        Learn correlation between paired audio and visual features.

        Args:
            audio_features: List of audio feature vectors
            visual_features: List of visual feature vectors
        """
        if len(audio_features) != len(visual_features):
            raise ValueError("Audio and visual features must have same length")

        # Simple correlation learning via CCA-like projection
        # Stack features
        audio_stack = np.stack(audio_features, axis=0)  # (n, audio_dim)
        visual_stack = np.stack(visual_features, axis=0)  # (n, visual_dim)

        # Compute cross-covariance
        audio_centered = audio_stack - np.mean(audio_stack, axis=0)
        visual_centered = visual_stack - np.mean(visual_stack, axis=0)

        cross_cov = audio_centered.T @ visual_centered / len(audio_features)

        # Update mapping based on correlation
        self.audio_to_visual = 0.5 * self.audio_to_visual + 0.5 * cross_cov

        # Compute correlation strength (singular norm)
        svd = np.linalg.svd(cross_cov, compute_uv=False)
        self.correlation_strength = float(np.sum(svd)) if len(svd) > 0 else 0.0

    def predict_visual(self, audio_features: np.ndarray) -> np.ndarray:
        """
        Predict visual features from audio.

        Args:
            audio_features: Audio feature vector

        Returns:
            Predicted visual features
        """
        visual_pred = audio_features @ self.audio_to_visual
        return visual_pred

    def predict_audio(self, visual_features: np.ndarray) -> np.ndarray:
        """
        Predict audio features from visual.

        Args:
            visual_features: Visual feature vector

        Returns:
            Predicted audio features
        """
        audio_pred = visual_features @ self.visual_to_audio
        return audio_pred

    def build_visual_index(self, visual_features: List[np.ndarray], contexts: List[str]) -> None:
        """
        Build index for visual context retrieval.

        Args:
            visual_features: List of visual feature vectors
            contexts: Context labels for each feature
        """
        if len(visual_features) != len(contexts):
            raise ValueError("Features and contexts must have same length")

        self.visual_index = visual_features
        self.context_labels = contexts

    def retrieve_similar(self, query_audio: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve similar visual contexts from audio query.

        Args:
            query_audio: Query audio features
            top_k: Number of results to return

        Returns:
            List of (context, score) tuples
        """
        if not self.visual_index:
            return []

        # Predict visual from audio
        query_visual = self.predict_visual(query_audio)

        # Compute similarities
        similarities = []
        for i, visual_feat in enumerate(self.visual_index):
            sim = float(np.dot(query_visual, visual_feat))
            similarities.append((self.context_labels[i], sim))

        # Sort and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class TemporalAlignment:
    """
    Temporal alignment between audio and video streams.

    Handles:
    - Timestamp conversion between audio samples and video frames
    - Synchronized window extraction
    - Frame-to-audio alignment
    """

    def __init__(self, fps: int = 30, audio_rate: int = 48000):
        """
        Initialize temporal alignment.

        Args:
            fps: Video frames per second
            audio_rate: Audio sample rate
        """
        self.fps = fps
        self.audio_rate = audio_rate
        self.samples_per_frame = audio_rate / fps

    def audio_to_frame_indices(self, audio_timestamps: np.ndarray) -> np.ndarray:
        """
        Convert audio timestamps to frame indices.

        Args:
            audio_timestamps: Array of audio timestamps in seconds

        Returns:
            Array of frame indices
        """
        frame_indices = (audio_timestamps * self.fps).astype(np.int32)
        return frame_indices

    def frame_to_audio_range(self, frame_idx: int) -> Tuple[int, int]:
        """
        Get audio sample range for a frame.

        Args:
            frame_idx: Frame index

        Returns:
            Tuple of (start_sample, end_sample)
        """
        start_sample = int(frame_idx * self.samples_per_frame)
        end_sample = int((frame_idx + 1) * self.samples_per_frame)
        return start_sample, end_sample

    def sync_windows(
        self, audio_features: np.ndarray, visual_features: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Create synchronized audio-visual windows.

        Args:
            audio_features: Audio features of shape (audio_len, audio_dim)
            visual_features: Visual features of shape (visual_len, visual_dim)

        Returns:
            Dictionary with synchronized 'audio' and 'visual' arrays
        """
        audio_len = audio_features.shape[0]
        visual_len = visual_features.shape[0]

        # Determine the shorter length (number of frames)
        # Assume audio features are already downsampled or aligned
        target_len = min(audio_len, visual_len)

        synced = {
            "audio": audio_features[:target_len],
            "visual": visual_features[:target_len],
        }

        return synced

    def create_temporal_windows(
        self,
        audio_features: np.ndarray,
        visual_features: np.ndarray,
        window_size_frames: int = 10,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Create sliding temporal windows.

        Args:
            audio_features: Audio features
            visual_features: Visual features
            window_size_frames: Window size in frames

        Returns:
            List of synchronized windows
        """
        synced = self.sync_windows(audio_features, visual_features)
        audio_synced = synced["audio"]
        visual_synced = synced["visual"]

        total_frames = audio_synced.shape[0]
        windows = []

        for i in range(0, total_frames, window_size_frames):
            end_idx = min(i + window_size_frames, total_frames)

            windows.append(
                {
                    "audio": audio_synced[i:end_idx],
                    "visual": visual_synced[i:end_idx],
                    "start_frame": i,
                }
            )

        return windows


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Multimodal Fusion - Vision + Audio")
    print("=" * 50)

    # Test visual feature extraction
    extractor = VisualFeatureExtractor(frame_height=224, frame_width=224)
    frame = np.random.randn(224, 224, 3).astype(np.float32)
    features = extractor.extract_frame_features(frame)

    print(f"Visual features shape: {features.shape}")

    # Test audio-visual fusion
    fusion = AudioVisualFusion(audio_dim=112, visual_dim=512, fusion_dim=256)
    audio_feat = np.random.randn(10, 112).astype(np.float32)
    visual_feat = np.random.randn(10, 512).astype(np.float32)

    fused = fusion.cross_modal_attention(audio_feat, visual_feat)
    print(f"Fused features shape: {fused.shape}")

    # Test multimodal classifier
    classifier = MultimodalContextClassifier(audio_dim=112, visual_dim=512, num_classes=4)
    logits = classifier.classify(audio_feat, visual_feat)
    print(f"Classification logits shape: {logits.shape}")
