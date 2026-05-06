#!/usr/bin/env python3
"""
Speaker Embeddings (Direction 3)
==============================

Extract fixed-length speaker embeddings from audio or features.
Enroll, verify, and identify speakers across recordings.
Enable speaker-adaptive synthesis.

This module implements:
1. SpeakerEmbeddingExtractor - Extract embeddings from audio/features
2. SpeakerDatabase - Manage known speakers with verification/ID
3. SpeakerAdaptiveSynthesis - Generate output in target speaker's voice

Uses lightweight neural network approach that works with 112D features.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import json
import logging
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SpeakerProfile:
    """Profile for a known speaker."""

    speaker_id: str
    embedding: np.ndarray  # L2-normalized embedding
    enrollment_count: int = 1
    first_seen: float = 0.0  # timestamp
    last_seen: float = 0.0  # timestamp

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "speaker_id": self.speaker_id,
            "embedding": self.embedding.tolist(),
            "enrollment_count": self.enrollment_count,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SpeakerProfile":
        """Create from dictionary."""
        return cls(
            speaker_id=data["speaker_id"],
            embedding=np.array(data["embedding"], dtype=np.float32),
            enrollment_count=data.get("enrollment_count", 1),
            first_seen=data.get("first_seen", 0.0),
            last_seen=data.get("last_seen", 0.0),
        )


@dataclass
class VerificationResult:
    """Result of speaker verification."""

    is_match: bool
    confidence: float
    distance: float


class SpeakerEmbeddingExtractor:
    """
    Extract fixed-length speaker embeddings from audio or features.

    Uses a lightweight neural network approach:
    - For audio: Mel-spectrogram → CNN → embedding
    - For features: 112D → MLP → embedding (lightweight, no audio needed)

    Embeddings are L2-normalized for cosine similarity comparison.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        model_type: str = "feature_based",
        random_state: int = 42,
    ):
        """
        Initialize the speaker embedding extractor.

        Args:
            embedding_dim: Dimension of output embedding
            model_type: "feature_based" for 112D features, "audio" for raw audio
            random_state: Random seed for reproducibility
        """
        self.embedding_dim = embedding_dim
        self.model_type = model_type
        self.random_state = random_state

        # Initialize projection matrix for feature-based extraction
        # In production, this would be a trained neural network
        np.random.seed(random_state)
        self._projection = np.random.randn(112, embedding_dim).astype(np.float32)
        self._projection /= np.linalg.norm(self._projection, axis=0)

        # For audio-based extraction, would use a trained model
        # For now, we use a simpler approach that extracts from features
        logger.info(
            f"SpeakerEmbeddingExtractor initialized: dim={embedding_dim}, type={model_type}"
        )

    def extract_from_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract embedding from raw audio.

        For production use, this would use a trained model like ECAPAT-DNN.
        For now, we extract features first, then embed.

        Args:
            audio: Audio samples (float32, normalized to [-1, 1])
            sr: Sample rate in Hz

        Returns:
            L2-normalized embedding vector of shape (embedding_dim,)
        """
        import time

        # In production: mel spectrogram → CNN → embedding
        # For now: extract simple features and project
        if len(audio) < sr:
            # Pad if too short
            padded = np.zeros(sr)
            padded[: len(audio)] = audio
            audio = padded

        # Extract simple spectral features
        # Use rolling window statistics
        frame_size = min(2048, len(audio) // 10)
        n_frames = len(audio) // frame_size

        features = []
        for i in range(n_frames):
            frame = audio[i * frame_size : (i + 1) * frame_size]
            if len(frame) == frame_size:
                # Simple features: energy, zero crossings, spectral centroid approximation
                energy = np.mean(frame**2)
                zcr = np.mean(np.abs(np.diff(np.sign(frame))))
                features.extend([energy, zcr])

        # Pad or truncate to 112 features
        features = np.array(features, dtype=np.float32)
        if len(features) < 112:
            features = np.pad(features, (0, 112 - len(features)))
        else:
            features = features[:112]

        # Project to embedding space
        embedding = np.dot(features, self._projection)

        # L2 normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        return embedding

    def extract_from_features(self, features_112d: np.ndarray) -> np.ndarray:
        """
        Extract embedding from 112D feature vector.

        This is a lightweight method that works directly with the
        RosettaFeatures output, no audio needed.

        Args:
            features_112d: 112D feature vector from RosettaFeatures

        Returns:
            L2-normalized embedding vector of shape (embedding_dim,)
        """
        # Ensure we have 112 features
        if len(features_112d) != 112:
            raise ValueError(f"Expected 112 features, got {len(features_112d)}")

        features = features_112d.astype(np.float32)

        # Project to embedding space
        embedding = np.dot(features, self._projection)

        # L2 normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        return embedding


class SpeakerDatabase:
    """
    Database of known speakers with their embeddings.

    Supports enrollment, verification, identification, and clustering.
    """

    def __init__(self, similarity_threshold: float = 0.8):
        """
        Initialize speaker database.

        Args:
            similarity_threshold: Default threshold for verification
        """
        self.speakers: Dict[str, SpeakerProfile] = {}
        self.similarity_threshold = similarity_threshold
        logger.info("SpeakerDatabase initialized")

    def enroll(
        self, speaker_id: str, embedding: np.ndarray, timestamp: Optional[float] = None
    ) -> bool:
        """
        Add or update a speaker in the database.

        Args:
            speaker_id: Unique identifier for the speaker
            embedding: L2-normalized speaker embedding
            timestamp: Optional timestamp for when this embedding was captured

        Returns:
            True if enrollment successful
        """
        import time

        if timestamp is None:
            timestamp = time.time()

        # Ensure embedding is normalized
        embedding = embedding.astype(np.float32)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        if speaker_id in self.speakers:
            # Update existing speaker (moving average)
            existing = self.speakers[speaker_id]
            count = existing.enrollment_count
            # Exponential moving average
            alpha = 1.0 / (count + 1)
            new_embedding = existing.embedding * (1 - alpha) + embedding * alpha
            new_embedding = new_embedding / (np.linalg.norm(new_embedding) + 1e-8)

            self.speakers[speaker_id] = replace(
                existing,
                embedding=new_embedding,
                enrollment_count=count + 1,
                last_seen=timestamp,
            )
            logger.debug(f"Updated speaker {speaker_id} (enrollment #{count + 1})")
        else:
            # New speaker
            self.speakers[speaker_id] = SpeakerProfile(
                speaker_id=speaker_id,
                embedding=embedding,
                enrollment_count=1,
                first_seen=timestamp,
                last_seen=timestamp,
            )
            logger.info(f"Enrolled new speaker: {speaker_id}")

        return True

    def verify(
        self, speaker_id: str, embedding: np.ndarray, threshold: Optional[float] = None
    ) -> VerificationResult:
        """
        Check if embedding matches known speaker.

        Args:
            speaker_id: Speaker to verify against
            embedding: Query embedding (L2-normalized)
            threshold: Similarity threshold (uses default if None)

        Returns:
            VerificationResult with is_match, confidence, and distance
        """
        if threshold is None:
            threshold = self.similarity_threshold

        if speaker_id not in self.speakers:
            return VerificationResult(is_match=False, confidence=0.0, distance=1.0)

        # Ensure embedding is normalized
        embedding = embedding.astype(np.float32)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        stored = self.speakers[speaker_id].embedding

        # Cosine similarity (since embeddings are L2-normalized)
        similarity = float(np.dot(stored, embedding))
        distance = 1.0 - similarity

        is_match = similarity >= threshold

        return VerificationResult(
            is_match=is_match, confidence=max(0.0, similarity), distance=distance
        )

    def identify(
        self, embedding: np.ndarray, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find most similar speakers in database.

        Args:
            embedding: Query embedding (L2-normalized)
            top_k: Number of top results to return

        Returns:
            List of (speaker_id, similarity) tuples, sorted by similarity
        """
        if not self.speakers:
            return []

        # Ensure embedding is normalized
        embedding = embedding.astype(np.float32)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        # Compute similarities to all speakers
        similarities = []
        for speaker_id, profile in self.speakers.items():
            sim = float(np.dot(profile.embedding, embedding))
            similarities.append((speaker_id, sim))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top-k
        return similarities[:top_k]

    def cluster_speakers(
        self, embeddings: List[np.ndarray], n_clusters: Optional[int] = None
    ) -> List[int]:
        """
        Discover speakers in unlabeled data using clustering.

        Args:
            embeddings: List of embeddings to cluster
            n_clusters: Number of clusters (auto-detected if None)

        Returns:
            List of cluster assignments (same length as embeddings)
        """
        if not embeddings:
            return []

        # Ensure all embeddings are normalized
        normalized = []
        for emb in embeddings:
            emb = emb.astype(np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            normalized.append(emb)

        X = np.array(normalized)

        # Convert to distance matrix (1 - cosine similarity)
        # Since embeddings are normalized, cosine similarity is dot product
        similarities = np.dot(X, X.T)
        distances = 1.0 - similarities

        # Use agglomerative clustering with precomputed distances
        if n_clusters is None:
            # Auto-detect number of clusters using distance threshold
            # Typical threshold: 0.5 distance (0.5 similarity)
            clustering = AgglomerativeClustering(
                n_clusters=None, distance_threshold=0.5, linkage="average",
                metric="precomputed"
            )
        else:
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters, linkage="average",
                metric="precomputed"
            )

        labels = clustering.fit_predict(distances)

        return labels.tolist()

    def save(self, path: str) -> None:
        """
        Save speaker database to JSON.

        Args:
            path: Path to save database
        """
        data = {
            "similarity_threshold": self.similarity_threshold,
            "speakers": [profile.to_dict() for profile in self.speakers.values()],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved speaker database with {len(self.speakers)} speakers to {path}")

    def load(self, path: str) -> None:
        """
        Load speaker database from JSON.

        Args:
            path: Path to load database from
        """
        with open(path, "r") as f:
            data = json.load(f)

        self.similarity_threshold = data.get("similarity_threshold", 0.8)
        self.speakers = {}
        for speaker_data in data.get("speakers", []):
            profile = SpeakerProfile.from_dict(speaker_data)
            self.speakers[profile.speaker_id] = profile

        logger.info(f"Loaded speaker database with {len(self.speakers)} speakers from {path}")

    def get_stats(self) -> dict:
        """Get database statistics."""
        return {
            "num_speakers": len(self.speakers),
            "similarity_threshold": self.similarity_threshold,
            "total_enrollments": sum(
                p.enrollment_count for p in self.speakers.values()
            ),
        }


class SpeakerAdaptiveSynthesis:
    """
    Adapt synthesis output to target speaker's voice.

    Uses speaker embeddings to condition the synthesis model.
    """

    def __init__(self, base_model, speaker_db: SpeakerDatabase, tokenizer: Optional[Any] = None):
        """
        Initialize speaker-adaptive synthesizer.

        Args:
            base_model: Base synthesis model (will be conditioned on speaker)
            speaker_db: Database of known speakers
            tokenizer: Optional tokenizer for converting tokens to features
        """
        self.base_model = base_model
        self.speaker_db = speaker_db
        self.tokenizer = tokenizer
        logger.info("SpeakerAdaptiveSynthesis initialized")

    def synthesize_as_speaker(
        self, tokens: List[int], target_speaker: str, **kwargs
    ) -> Optional[np.ndarray]:
        """
        Generate audio in target speaker's voice.

        Args:
            tokens: Token sequence to synthesize
            target_speaker: Speaker ID to imitate
            **kwargs: Additional arguments for base model

        Returns:
            Synthesized audio, or None if speaker unknown and no fallback
        """
        # Get speaker embedding
        if target_speaker in self.speaker_db.speakers:
            speaker_embedding = self.speaker_db.speakers[target_speaker].embedding
        else:
            logger.warning(f"Unknown speaker: {target_speaker}, using default synthesis")
            speaker_embedding = None

        # Call base model with speaker conditioning
        if hasattr(self.base_model, "synthesize_with_speaker"):
            # Model supports speaker conditioning
            return self.base_model.synthesize_with_speaker(
                tokens, speaker_embedding=speaker_embedding, **kwargs
            )
        else:
            # Check if we need to convert tokens to features
            # NeuralVocoder expects features (n_frames, 112), not token IDs
            if self.tokenizer is not None:
                # Convert tokens to features
                features = np.array([self.tokenizer.detokenize(t) for t in tokens])
                return self.base_model.synthesize(features, **kwargs)
            else:
                # Direct synthesis - tokens may be incompatible depending on model
                # Try to synthesize, but log if it might fail
                try:
                    return self.base_model.synthesize(tokens, **kwargs)
                except (ValueError, TypeError, IndexError) as e:
                    logger.warning(
                        f"Base model {type(self.base_model).__name__} may not accept "
                        f"token IDs directly. Provide a tokenizer for conversion: {e}"
                    )
                    return None


def create_speaker_pipeline(embedding_dim: int = 256) -> Tuple[
    SpeakerEmbeddingExtractor, SpeakerDatabase
]:
    """
    Create a complete speaker identification pipeline.

    Args:
        embedding_dim: Dimension for speaker embeddings

    Returns:
        Tuple of (extractor, database)
    """
    extractor = SpeakerEmbeddingExtractor(embedding_dim=embedding_dim)
    database = SpeakerDatabase()
    return extractor, database


def main():
    """Command-line interface for speaker embeddings."""
    import argparse

    parser = argparse.ArgumentParser(description="Speaker Embeddings for Direction 3")
    parser.add_argument(
        "--enroll",
        type=str,
        help="Enroll a speaker from audio file",
    )
    parser.add_argument(
        "--speaker-id",
        type=str,
        help="Speaker ID for enrollment",
    )
    parser.add_argument(
        "--verify",
        type=str,
        help="Verify speaker from audio file",
    )
    parser.add_argument(
        "--target-speaker",
        type=str,
        help="Target speaker ID for verification",
    )
    parser.add_argument(
        "--database",
        type=str,
        default="speaker_database.json",
        help="Path to speaker database",
    )
    parser.add_argument(
        "--identify",
        type=str,
        help="Identify speaker from audio file",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top results for identification",
    )

    args = parser.parse_args()

    # Create components
    extractor = SpeakerEmbeddingExtractor()
    db = SpeakerDatabase()

    # Load database if exists
    db_path = Path(args.database)
    if db_path.exists():
        db.load(args.database)
        print(f"Loaded database with {len(db.speakers)} speakers")

    # Enrollment
    if args.enroll and args.speaker_id:
        import librosa

        audio, sr = librosa.load(args.enroll, sr=48000)
        embedding = extractor.extract_from_audio(audio, sr)
        db.enroll(args.speaker_id, embedding)
        db.save(args.database)
        print(f"Enrolled speaker: {args.speaker_id}")

    # Verification
    elif args.verify and args.target_speaker:
        import librosa

        audio, sr = librosa.load(args.verify, sr=48000)
        embedding = extractor.extract_from_audio(audio, sr)
        result = db.verify(args.target_speaker, embedding)
        print(f"Verification: {'MATCH' if result.is_match else 'NO MATCH'}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Distance: {result.distance:.3f}")

    # Identification
    elif args.identify:
        import librosa

        audio, sr = librosa.load(args.identify, sr=48000)
        embedding = extractor.extract_from_audio(audio, sr)
        matches = db.identify(embedding, top_k=args.top_k)
        print(f"Identification results (top {len(matches)}):")
        for speaker_id, score in matches:
            print(f"  {speaker_id}: {score:.3f}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
