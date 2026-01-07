"""
Online Phrase Discovery Agent for Field Deployment
==================================================

Real-time phrase discovery for field scenarios where DBSCAN's offline
batch processing is not feasible.

Uses KNN/thresholding instead of DBSCAN:
- Lab: DBSCAN (re-processes all data offline)
- Field: KNN/Thresholding (incremental, real-time)

Workflow: Cold Storage → Repetition Validation → Hot Swap
1. Detect unknown phrase (KNN distance > threshold)
2. Cold store immediately with UNKNOWN_XXX label
3. Monitor for repetition (statistical validation)
4. Promote to Hot Swap when confirmed (load into Rust)

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import asyncio
import json
import logging
import pickle
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy.spatial.distance import cdist

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Data Structures
# ============================================================================


class PhraseState(Enum):
    """Phrase lifecycle states."""

    CANDIDATE = "CANDIDATE"  # Unknown, waiting for validation
    ACTIVE = "ACTIVE"  # Validated, available for synthesis
    ARCHIVED = "ARCHIVED"  # Old, rarely used


@dataclass
class PhraseCandidate:
    """
    A phrase candidate being tracked for validation.
    """

    id: str
    source_file: str
    feature_vector: np.ndarray  # 17-dimensional feature vector
    count: int = 1
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    state: PhraseState = PhraseState.CANDIDATE

    # Audio buffer (kept in memory for hot swapping)
    audio_buffer: Optional[np.ndarray] = None
    sample_rate: int = 22050

    # Metadata
    context: Optional[str] = None
    quality_score: float = 0.0

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "source_file": self.source_file,
            "feature_vector": self.feature_vector.tolist(),
            "count": self.count,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "state": self.state.value,
            "context": self.context,
            "quality_score": self.quality_score,
        }


@dataclass
class DiscoveryConfig:
    """Configuration for online phrase discovery."""

    # KNN Thresholds
    known_phrase_threshold: float = 2.0  # Z-score distance < 2.0 = known
    unknown_phrase_threshold: float = 2.0  # Z-score distance >= 2.0 = unknown

    # Validation
    confidence_threshold: int = 3  # Repetitions needed for promotion
    validation_window_sec: float = 300.0  # Max time between repetitions

    # Cold Storage
    temp_dir: str = "/field_data/session/temp"
    min_phrase_duration_ms: float = 20.0
    max_phrase_duration_ms: float = 500.0

    # Hot Swap
    enable_rust_bridge: bool = True
    rust_async_load_timeout: float = 1.0  # Max seconds to wait for Rust load

    # Babble Prevention
    max_candidate_ratio: float = 0.3  # Max 30% candidates vs active phrases
    candidate_selection_weight: float = 0.2  # Lower weight for candidates in synthesis


# ============================================================================
# Feature Extractor (17-dimensional)
# ============================================================================


class FeatureExtractor:
    """
    Extract 17-dimensional feature vectors from audio.

    Compatible with audio_aware_grammar_discovery.py features.
    """

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    def extract(self, audio: np.ndarray, sr: Optional[int] = None) -> np.ndarray:
        """
        Extract 17-dimensional feature vector.

        Returns:
            np.ndarray: Feature vector (17,)
        """
        sr = sr or self.sample_rate
        features = np.zeros(17)

        # F0 (fundamental frequency)
        import librosa

        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
            )
            # Use median of voiced frames
            voiced_f0 = f0[voiced_flag]
            if len(voiced_f0) > 0:
                features[0] = np.median(voiced_f0)
        except Exception:
            features[0] = 0.0

        # Duration
        features[1] = len(audio) / sr * 1000  # ms

        # Attack/decay (proxy with ZCR)
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features[2] = np.mean(zcr) * 10  # attack_ms proxy
        features[3] = np.std(zcr) * 50  # decay_ms proxy

        # F0 range (local std)
        if len(voiced_f0) > 1:
            features[4] = np.std(voiced_f0)

        # RMS energy
        rms = librosa.feature.rms(y=audio)[0]
        features[15] = 20 * np.log10(np.mean(rms) + 1e-6)  # rms_db
        features[16] = np.max(rms)  # peak_amplitude

        # Harmonicity (voicing probability)
        if len(voiced_probs) > 0:
            features[9] = np.mean(voiced_probs) * 20

        # Spectral features
        spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        spec_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        spec_flatness = librosa.feature.spectral_flatness(y=audio)[0]

        features[10] = np.mean(spec_flatness)
        features[11] = np.mean(spec_centroid)
        features[12] = np.mean(spec_rolloff)
        features[13] = np.mean(spec_bandwidth)

        # Placeholder defaults for features requiring more context
        features[5] = 0.0  # vibrato_rate
        features[6] = 0.0  # vibrato_depth
        features[7] = np.std(zcr) * 0.1  # jitter
        features[8] = np.std(rms) * 0.1  # shimmer
        features[14] = -8.0  # slope

        return features


# ============================================================================
# KNN Phrase Search
# ============================================================================


class KNNPhraseSearch:
    """
    Fast KNN search for phrase matching.

    Uses pre-computed statistics for Z-score normalization.
    """

    def __init__(
        self,
        phrase_library_path: Optional[str] = None,
        feature_mean: Optional[np.ndarray] = None,
        feature_std: Optional[np.ndarray] = None,
    ):
        """
        Initialize KNN search with phrase library.

        Args:
            phrase_library_path: Path to .pkl phrase library
            feature_mean: Pre-computed mean for normalization
            feature_std: Pre-computed std for normalization
        """
        self.phrases = {}  # phrase_key -> feature_vector
        self.feature_mean = feature_mean
        self.feature_std = feature_std

        if phrase_library_path:
            self.load_library(phrase_library_path)

    def load_library(self, path: str):
        """Load phrase library from .pkl file."""
        with open(path, "rb") as f:
            library = pickle.load(f)

        # Extract feature vectors from phrase segments
        for phrase_key, segments in library["phrase_segments"].items():
            # Use first segment as canonical representation
            if segments:
                seg = segments[0]
                vector = np.array(
                    [
                        seg["mean_f0_hz"],
                        seg["mean_duration_ms"],
                        0.0,  # attack_ms (not stored)
                        0.0,  # decay_ms (not stored)
                        seg["mean_range_hz"],  # f0_range_hz
                        0.0,  # vibrato_rate
                        0.0,  # vibrato_depth
                        0.0,  # jitter
                        0.0,  # shimmer
                        seg.get("snr_db", 0.0),  # harmonicity_hnr
                        0.0,  # spectral_flatness
                        0.0,  # spectral_centroid
                        0.0,  # spectral_rolloff
                        0.0,  # bandwidth
                        -8.0,  # slope
                        0.0,  # rms_db
                        1.0,  # peak_amplitude
                    ]
                )
                self.phrases[phrase_key] = vector

        # Compute normalization statistics if not provided
        if self.feature_mean is None:
            matrix = np.stack(list(self.phrases.values()))
            self.feature_mean = np.mean(matrix, axis=0)
            self.feature_std = np.std(matrix, axis=0)
            self.feature_std[self.feature_std == 0] = 1.0

        logger.info(f"KNN Search loaded {len(self.phrases)} phrases")

    def search(
        self, query_vector: np.ndarray, k: int = 1, normalize: bool = True
    ) -> Tuple[Optional[str], float]:
        """
        Find nearest phrase.

        Args:
            query_vector: 17-dimensional feature vector
            k: Number of nearest neighbors
            normalize: Whether to apply z-score normalization

        Returns:
            Tuple of (phrase_key, distance)
        """
        if not self.phrases:
            return None, float("inf")

        # Normalize query
        if normalize and self.feature_mean is not None:
            query_norm = (query_vector - self.feature_mean) / self.feature_std
        else:
            query_norm = query_vector

        # Normalize library
        if normalize:
            matrix = np.stack(list(self.phrases.values()))
            matrix_norm = (matrix - self.feature_mean) / self.feature_std
        else:
            matrix_norm = np.stack(list(self.phrases.values()))

        # Compute distances
        distances = cdist([query_norm], matrix_norm, metric="euclidean")[0]

        # Get nearest
        nearest_idx = np.argmin(distances)
        nearest_key = list(self.phrases.keys())[nearest_idx]
        nearest_distance = distances[nearest_idx]

        return nearest_key, nearest_distance


# ============================================================================
# Rust Bridge for Async Hot Swapping
# ============================================================================


class RustSynthesizerBridge:
    """
    Bridge to Rust synthesis engine for async hot swapping.

    Simulates async loading in production. In deployment, this would
    use ZeroMQ or PyO3 to communicate with Rust engine.
    """

    def __init__(self, config: DiscoveryConfig):
        self.config = config
        self.active_phrases = set()
        self._rust_lock = threading.Lock()

    async def load_source_async(
        self, phrase_id: str, file_path: str, audio_buffer: Optional[np.ndarray] = None
    ) -> bool:
        """
        Asynchronously load phrase into Rust engine.

        Args:
            phrase_id: Phrase identifier
            file_path: Path to audio file
            audio_buffer: Optional in-memory audio buffer

        Returns:
            True if loaded successfully
        """
        logger.info(f"Hot swapping {phrase_id} into Rust engine...")

        # Simulate async file loading (in production, this is actual Rust call)
        if audio_buffer is None and Path(file_path).exists():
            audio_buffer, sr = sf.read(file_path)
        elif audio_buffer is None:
            logger.warning(f"Audio file not found: {file_path}")
            return False

        # Simulate async operation (in production: ZeroMQ/PyO3 call)
        await asyncio.sleep(0.01)  # Simulate 10ms load time

        # Update active set (thread-safe)
        with self._rust_lock:
            self.active_phrases.add(phrase_id)

        logger.info(f"✓ Hot swap complete: {phrase_id} now active in Rust")
        return True

    def is_active(self, phrase_id: str) -> bool:
        """Check if phrase is loaded in Rust engine."""
        with self._rust_lock:
            return phrase_id in self.active_phrases

    def get_active_phrases(self) -> set:
        """Get all active phrases."""
        with self._rust_lock:
            return self.active_phrases.copy()


# ============================================================================
# Online Phrase Discovery Agent
# ============================================================================


class OnlinePhraseDiscoveryAgent:
    """
    Real-time phrase discovery for field deployment.

    Key features:
    - KNN-based phrase detection (not DBSCAN)
    - Cold storage for unknown phrases
    - Repetition-based validation
    - Async hot swapping to Rust
    - Smart babble prevention
    """

    def __init__(self, phrase_library_path: str, config: Optional[DiscoveryConfig] = None):
        """
        Initialize online discovery agent.

        Args:
            phrase_library_path: Path to initial .pkl phrase library
            config: Discovery configuration
        """
        self.config = config or DiscoveryConfig()

        # Initialize components
        self.feature_extractor = FeatureExtractor(sample_rate=22050)
        self.knn_search = KNNPhraseSearch(phrase_library_path)
        self.rust_bridge = (
            RustSynthesizerBridge(self.config) if self.config.enable_rust_bridge else None
        )

        # Phrase tracking
        self.candidates: Dict[str, PhraseCandidate] = {}  # Temp phrase storage
        self.active_phrases: Dict[str, PhraseCandidate] = {}  # Validated phrases
        self.unknown_counter = 0

        # Statistics
        self.total_processed = 0
        self.known_detected = 0
        self.unknown_detected = 0
        self.promotions = 0

        # Create temp directory
        Path(self.config.temp_dir).mkdir(parents=True, exist_ok=True)

        # Background monitoring
        self._monitor_running = False
        self._monitor_thread: Optional[threading.Thread] = None

        logger.info("Online Phrase Discovery Agent initialized")
        logger.info(f"  Initial library: {len(self.knn_search.phrases)} phrases")
        logger.info(f"  Threshold: {self.config.known_phrase_threshold} Z-score")
        logger.info(f"  Validation: {self.config.confidence_threshold} repetitions")

    def process_live_audio(
        self, audio_buffer: np.ndarray, sr: int, context: Optional[str] = None
    ) -> str:
        """
        Process live audio buffer and detect phrase.

        Args:
            audio_buffer: Audio samples
            sr: Sample rate
            context: Optional behavioral context

        Returns:
            Detected phrase ID or "UNKNOWN_DETECTED"
        """
        self.total_processed += 1

        # Extract features
        vector = self.feature_extractor.extract(audio_buffer, sr)

        # KNN search
        best_match, distance = self.knn_search.search(vector, k=1, normalize=True)

        # Classify
        if distance < self.config.known_phrase_threshold:
            # Known phrase
            self.known_detected += 1
            return best_match
        else:
            # Unknown phrase - trigger cold storage
            self.unknown_detected += 1
            return self._handle_unknown_phrase(audio_buffer, sr, vector, context, distance)

    def _handle_unknown_phrase(
        self,
        audio_buffer: np.ndarray,
        sr: int,
        vector: np.ndarray,
        context: Optional[str],
        distance: float,
    ) -> str:
        """
        Handle unknown phrase detection (Cold Storage).

        Args:
            audio_buffer: Audio samples
            sr: Sample rate
            vector: Feature vector
            context: Behavioral context
            distance: KNN distance

        Returns:
            "UNKNOWN_DETECTED"
        """
        # Assign temporary ID
        temp_id = f"UNKNOWN_{self.unknown_counter:04d}"
        self.unknown_counter += 1

        # Save audio to cold storage
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{temp_id}_{timestamp}.wav"
        save_path = Path(self.config.temp_dir) / filename
        sf.write(save_path, audio_buffer, sr)

        # Create shadow entry (candidate)
        candidate = PhraseCandidate(
            id=temp_id,
            source_file=str(save_path),
            feature_vector=vector,
            audio_buffer=audio_buffer.copy(),
            sample_rate=sr,
            context=context,
            state=PhraseState.CANDIDATE,
            quality_score=max(0.0, 1.0 - distance / 5.0),  # Higher distance = lower quality
        )

        self.candidates[temp_id] = candidate

        logger.info(f"Unknown phrase detected: {temp_id} (distance={distance:.2f})")
        logger.info(f"  Cold stored: {save_path}")
        logger.info(f"  Total candidates: {len(self.candidates)}")

        return "UNKNOWN_DETECTED"

    def monitor_candidates(self) -> int:
        """
        Check for phrase repetition and promote to active.

        Returns:
            Number of promotions this cycle
        """
        promotions = 0
        current_time = time.time()

        # Group candidates by similarity (for repetition detection)
        candidate_ids = list(self.candidates.keys())
        to_promote = []

        for candidate_id in candidate_ids:
            candidate = self.candidates[candidate_id]

            # Check age
            age = current_time - candidate.first_seen
            if age > self.config.validation_window_sec:
                # Too old, remove
                del self.candidates[candidate_id]
                continue

            # Check for similar candidates (repetition detection)
            similar = self._find_similar_candidates(candidate)
            if similar >= self.config.confidence_threshold:
                # Promote!
                to_promote.append(candidate_id)

        # Execute promotions
        for candidate_id in to_promote:
            if self._promote_candidate(candidate_id):
                promotions += 1

        return promotions

    def _find_similar_candidates(self, target: PhraseCandidate) -> int:
        """
        Find candidates similar to target (repetition counting).

        Args:
            target: Target candidate

        Returns:
            Count of similar candidates (including target)
        """
        count = target.count

        target_vector = target.feature_vector

        for other_id, other in self.candidates.items():
            if other_id == target.id:
                continue

            # Compute distance
            distance = np.linalg.norm(target_vector - other.feature_vector)

            # Similar if very close (< 0.5 Z-score)
            if distance < 0.5:
                count += 1

        return count

    def _promote_candidate(self, candidate_id: str) -> bool:
        """
        Promote candidate to active phrase (Hot Swap).

        Args:
            candidate_id: Candidate to promote

        Returns:
            True if promoted successfully
        """
        if candidate_id not in self.candidates:
            return False

        candidate = self.candidates[candidate_id]

        # Generate final ID
        final_id = f"DISCOVERED_{len(self.active_phrases):04d}"

        # Update candidate
        old_id = candidate.id
        candidate.id = final_id
        candidate.state = PhraseState.ACTIVE

        # Move to active
        self.active_phrases[final_id] = candidate
        del self.candidates[old_id]

        # Hot swap to Rust
        if self.rust_bridge:
            # In production, this runs in background thread
            asyncio.create_task(
                self.rust_bridge.load_source_async(
                    final_id, candidate.source_file, candidate.audio_buffer
                )
            )

        # Update KNN search with new phrase
        self.knn_search.phrases[final_id] = candidate.feature_vector

        self.promotions += 1

        logger.info(f"✓ PROMOTED: {old_id} → {final_id}")
        logger.info(f"  Repetitions: {candidate.count}")
        logger.info(f"  Active phrases: {len(self.active_phrases)}")

        return True

    def start_background_monitor(self, interval_sec: float = 5.0):
        """
        Start background monitoring thread.

        Args:
            interval_sec: Check interval
        """
        if self._monitor_running:
            logger.warning("Monitor already running")
            return

        self._monitor_running = True

        def monitor_loop():
            while self._monitor_running:
                time.sleep(interval_sec)
                try:
                    promotions = self.monitor_candidates()
                    if promotions > 0:
                        logger.info(f"Background monitor: {promotions} promotions")
                except Exception as e:
                    logger.error(f"Monitor error: {e}")

        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()

        logger.info(f"Background monitor started (interval={interval_sec}s)")

    def stop_background_monitor(self):
        """Stop background monitoring thread."""
        self._monitor_running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Background monitor stopped")

    def get_synthesis_candidates(
        self, intent: str, max_candidates: int = 10
    ) -> List[PhraseCandidate]:
        """
        Get candidates for synthesis with babble prevention.

        Prioritizes validated phrases over candidates.

        Args:
            intent: Behavioral intent (for future filtering)
            max_candidates: Maximum candidates to return

        Returns:
            List of phrase candidates
        """
        # Collect all phrases
        all_phrases = []

        # Active phrases (high priority)
        for phrase in self.active_phrases.values():
            all_phrases.append((phrase, 1.0))  # Weight 1.0

        # Candidate phrases (low priority - babble prevention)
        candidate_ratio = len(self.candidates) / max(len(self.active_phrases), 1)
        if candidate_ratio < self.config.max_candidate_ratio:
            for phrase in self.candidates.values():
                all_phrases.append((phrase, self.config.candidate_selection_weight))

        # Shuffle and return
        import random

        random.shuffle(all_phrases)

        # Select by weighted lottery
        selected = []
        for phrase, weight in all_phrases[:max_candidates]:
            if random.random() < weight:
                selected.append(phrase)

        return selected

    def save_session(self, output_path: str):
        """
        Save session state to disk.

        Args:
            output_path: Output file path
        """
        session_data = {
            "timestamp": datetime.now().isoformat(),
            "statistics": {
                "total_processed": self.total_processed,
                "known_detected": self.known_detected,
                "unknown_detected": self.unknown_detected,
                "promotions": self.promotions,
                "active_count": len(self.active_phrases),
                "candidate_count": len(self.candidates),
            },
            "active_phrases": {k: v.to_dict() for k, v in self.active_phrases.items()},
            "candidates": {k: v.to_dict() for k, v in self.candidates.items()},
        }

        with open(output_path, "w") as f:
            json.dump(session_data, f, indent=2)

        logger.info(f"Session saved to {output_path}")

    def get_statistics(self) -> Dict:
        """Get discovery statistics."""
        return {
            "total_processed": self.total_processed,
            "known_detected": self.known_detected,
            "unknown_detected": self.unknown_detected,
            "promotions": self.promotions,
            "active_phrases": len(self.active_phrases),
            "candidates": len(self.candidates),
            "detection_rate": self.known_detected / max(self.total_processed, 1),
            "discovery_rate": self.promotions / max(self.unknown_detected, 1),
        }


# ============================================================================
# Demo
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Online Phrase Discovery Agent")
    parser.add_argument("library_path", type=str, help="Path to .pkl phrase library")
    parser.add_argument("--audio", type=str, help="Path to test audio file")
    parser.add_argument(
        "--threshold", type=float, default=2.0, help="KNN distance threshold for unknown detection"
    )
    parser.add_argument(
        "--confidence", type=int, default=3, help="Repetitions needed for promotion"
    )

    args = parser.parse_args()

    # Create config
    config = DiscoveryConfig(
        known_phrase_threshold=args.threshold, confidence_threshold=args.confidence
    )

    # Create agent
    agent = OnlinePhraseDiscoveryAgent(args.library_path, config)

    # Start background monitor
    agent.start_background_monitor()

    if args.audio:
        # Process test audio
        import librosa

        audio, sr = librosa.load(args.audio, sr=22050)

        print(f"\nProcessing {args.audio}...")
        print(f"Audio duration: {len(audio) / sr:.2f}s")

        # Simulate streaming (chunk into 100ms segments)
        chunk_size = int(0.1 * sr)
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i : i + chunk_size]
            if len(chunk) < chunk_size:
                break

            result = agent.process_live_audio(chunk, sr)
            print(f"  Chunk {i // chunk_size}: {result}")

        # Wait for monitor to process
        import time

        time.sleep(2)

        # Show statistics
        stats = agent.get_statistics()
        print("\nStatistics:")
        print(f"  Processed: {stats['total_processed']}")
        print(f"  Known: {stats['known_detected']}")
        print(f"  Unknown: {stats['unknown_detected']}")
        print(f"  Promoted: {stats['promotions']}")
        print(f"  Active phrases: {stats['active_phrases']}")
        print(f"  Candidates: {stats['candidates']}")

        # Save session
        session_path = "online_discovery_session.json"
        agent.save_session(session_path)
        print(f"\nSession saved: {session_path}")

    agent.stop_background_monitor()
