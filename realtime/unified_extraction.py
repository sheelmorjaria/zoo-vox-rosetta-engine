"""
Unified Phrase/Sentence/Grammar Extraction Pipeline

This module implements a complete extraction pipeline that:
1. Segments audio into sentences (PELT change point detection)
2. Extracts phrase candidates (sliding window within sentences)
3. Clusters phrases into atomic units (DBSCAN on 29D features)
4. Tests atomicity (intra vs inter cluster similarity)
5. Tests compositionality (phrase reuse across sentences)
6. Builds grammar rules from observed transitions
7. Exports segmented audio and context associations

Input: Audio directory + annotations file
Output: Phrases, sentences, grammar, segmented audio

Algorithm:
- PELT (Pruned Exact Linear Time) for sentence segmentation
- Sliding window for phrase candidate extraction
- DBSCAN for phrase clustering (29D feature space)
- Silhouette analysis for atomicity validation
- Co-occurrence analysis for compositionality

Architecture: Audio Directory → 29D Features → Phrases → Sentences → Grammar

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN

# =============================================================================
# Data Models
# =============================================================================


@dataclass
class PhraseCandidate:
    """Candidate phrase from sliding window"""

    audio_segment: np.ndarray
    start_sample: int
    end_sample: int
    features_29d: Dict[str, float]
    source_sentence_idx: int
    window_id: int


@dataclass
class AtomicPhrase:
    """Validated atomic phrase (clustered)"""

    phrase_id: str
    cluster_id: int
    features_29d: Dict[str, float]
    member_candidates: List[PhraseCandidate]
    intra_cluster_similarity: float
    inter_cluster_similarity: float
    is_atomic: bool


@dataclass
class Sentence:
    """Sentence (sequence of phrases)"""

    sentence_id: str
    audio_path: str
    start_sample: int
    end_sample: int
    phrases: List[AtomicPhrase]
    context: str
    compositionality_score: float


@dataclass
class GrammarRule:
    """Transition rule between phrases"""

    from_phrase_id: str
    to_phrase_id: str
    frequency: int
    probability: float
    contexts: List[str] = field(default_factory=list)


@dataclass
class ExtractionResult:
    """Complete extraction result"""

    sentences: List[Sentence]
    phrases: List[AtomicPhrase]
    grammar_rules: List[GrammarRule]
    audio_segments_dir: Path
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Step 1: PELT Sentence Segmentation
# =============================================================================


def segment_sentences_pelt(
    audio: np.ndarray, sr: int, penalty: float = 10.0, min_segment_length_sec: float = 0.3
) -> List[int]:
    """
    Segment audio into sentences using PELT (Pruned Exact Linear Time).

    PELT detects change points in the signal by optimizing:
    sum of segment costs + penalty * number_of_change_points

    Args:
        audio: Audio samples
        sr: Sample rate
        penalty: Penalty for each change point (higher = fewer segments)
        min_segment_length_sec: Minimum segment length in seconds

    Returns:
        List of change point sample indices
    """
    try:
        import ruptures as rpt

        # Extract features for change point detection
        # Use multiple features for robustness
        features = []

        # 1. MFCCs (13 coefficients)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features.append(mfcc.T)  # (time, n_mfcc)

        # 2. Spectral contrast
        spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        features.append(spec_contrast.T)

        # 3. Chroma
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        features.append(chroma.T)

        # Concatenate features
        feature_matrix = np.concatenate(features, axis=1)  # (time, n_features)

        # Normalize features
        feature_mean = np.mean(feature_matrix, axis=0)
        feature_std = np.std(feature_matrix, axis=0) + 1e-8
        feature_matrix = (feature_matrix - feature_mean) / feature_std

        # Apply PELT
        # Calculate min_size in frames
        hop_length = 512
        min_size = max(2, int(min_segment_length_sec * sr / hop_length))

        # Initialize and fit PELT
        algo = rpt.Pelt(model="rbf", min_size=min_size, jump=1).fit(feature_matrix)
        change_points = algo.fit_predict(feature_matrix, penalty)

        # Convert frame indices to sample indices
        n_samples = len(audio)
        n_frames = feature_matrix.shape[0]
        samples_per_frame = n_samples // n_frames

        # Filter out endpoint
        change_point_samples = [
            int(cp * samples_per_frame) for cp in change_points if cp < n_frames - 1
        ]

        return change_point_samples

    except ImportError:
        # Fallback: Use energy-based change point detection
        print("Warning: ruptures not installed, using fallback method")
        return _segment_by_energy(audio, sr, min_segment_length_sec)


def _segment_by_energy(
    audio: np.ndarray, sr: int, min_segment_length_sec: float = 0.3
) -> List[int]:
    """
    Fallback method: Segment by energy changes.

    Uses local minima in energy envelope to detect boundaries.
    """
    # Compute energy envelope
    hop_length = 512
    frame_length = 2048

    # RMS energy
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]

    # Smooth
    from scipy.ndimage import gaussian_filter1d

    rms_smooth = gaussian_filter1d(rms, sigma=5)

    # Find local minima
    from scipy.signal import find_peaks

    minima, _ = find_peaks(-rms_smooth, distance=int(min_segment_length_sec * sr / hop_length))

    # Convert to samples
    change_points = [int(m * hop_length) for m in minima]

    return change_points


# =============================================================================
# Step 2: Sliding Window Phrase Extraction
# =============================================================================


def extract_phrase_candidates(
    audio: np.ndarray,
    sr: int,
    min_window_ms: int = 50,
    max_window_ms: int = 500,
    hop_ms: int = 25,
    window_sizes_ms: Optional[List[int]] = None,
) -> List[PhraseCandidate]:
    """
    Extract phrase candidates using sliding window.

    Uses multiple window sizes to capture phrases of different durations.

    Args:
        audio: Audio samples
        sr: Sample rate
        min_window_ms: Minimum window size
        max_window_ms: Maximum window size
        hop_ms: Hop size between windows
        window_sizes_ms: Specific window sizes to use (if None, uses range)

    Returns:
        List of phrase candidates
    """
    candidates = []

    # Determine window sizes
    if window_sizes_ms is None:
        # Use geometric progression from min to max
        window_sizes_ms = []
        size = min_window_ms
        while size <= max_window_ms:
            window_sizes_ms.append(size)
            size = int(size * 1.5)

    hop_samples = int(hop_ms * sr / 1000)

    window_id = 0
    for window_ms in window_sizes_ms:
        window_samples = int(window_ms * sr / 1000)

        # Slide window across audio
        for start in range(0, len(audio) - window_samples + 1, hop_samples):
            end = start + window_samples

            # Extract segment
            segment = audio[start:end]

            # Skip if too short or too quiet
            if len(segment) < window_samples // 2:
                continue
            if np.max(np.abs(segment)) < 0.01:
                continue

            # Extract 29D features
            features = extract_29d_features(segment, sr)

            candidate = PhraseCandidate(
                audio_segment=segment,
                start_sample=start,
                end_sample=end,
                features_29d=features,
                source_sentence_idx=-1,  # Will be assigned later
                window_id=window_id,
            )
            candidates.append(candidate)
            window_id += 1

    return candidates


def extract_29d_features(audio: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Extract 29-dimensional acoustic features.

    Returns:
        Dictionary with 29 feature values
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        features = {}

        try:
            # === Fundamental (3 features) ===
            # Extract F0 using librosa
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y=audio,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sr,
                threshold=0.1,
            )

            # Use only voiced frames
            voiced_f0 = f0[voiced_flag]

            if len(voiced_f0) > 0:
                features["mean_f0_hz"] = float(np.mean(voiced_f0))
                features["f0_range_hz"] = float(np.max(voiced_f0) - np.min(voiced_f0))
            else:
                features["mean_f0_hz"] = 0.0
                features["f0_range_hz"] = 0.0

            features["duration_ms"] = len(audio) / sr * 1000.0

            # === Grit Factors (3 features) ===
            # Harmonic-to-noise ratio
            harmonic, percussive = librosa.effects.hpss(audio)
            energy_harmonic = np.mean(harmonic**2)
            energy_total = np.mean(audio**2) + 1e-10
            features["harmonic_to_noise_ratio"] = float(
                10 * np.log10(energy_harmonic / (energy_total - energy_harmonic + 1e-10))
            )

            # Spectral flatness
            spectral_flatness = librosa.feature.spectral_flatness(y=audio)
            features["spectral_flatness"] = float(np.mean(spectral_flatness))

            # Harmonicity (using harmonic/percussive separation)
            features["harmonicity"] = float(np.clip(energy_harmonic / energy_total, 0.0, 1.0))

            # === Motion Factors (7 features) ===
            # Envelope for attack/decay
            envelope = np.abs(audio)
            peak_idx = np.argmax(envelope)
            peak_amp = envelope[peak_idx]

            # Attack time (to 90% of peak)
            attack_threshold = 0.9 * peak_amp
            attack_idx = np.where(envelope[:peak_idx] > attack_threshold)[0]
            features["attack_time_ms"] = float(
                (attack_idx[0] if len(attack_idx) > 0 else 0) / sr * 1000 if peak_idx > 0 else 0.0
            )

            # Decay time (to 10% of peak)
            decay_threshold = 0.1 * peak_amp
            decay_idx = np.where(envelope[peak_idx:] < decay_threshold)[0]
            features["decay_time_ms"] = float(
                (decay_idx[0] if len(decay_idx) > 0 else len(envelope) - peak_idx) / sr * 1000
                if peak_idx < len(envelope)
                else 0.0
            )

            # Sustain level
            features["sustain_level"] = float(np.mean(envelope[peak_idx:]) / (peak_amp + 1e-10))

            # Vibrato (using autocorrelation of envelope)
            if len(envelope) > 1024:
                autocorr = np.correlate(envelope, envelope, mode="full")
                autocorr = autocorr[len(autocorr) // 2 :]
                # Find peaks in autocorr
                from scipy.signal import find_peaks

                peaks, _ = find_peaks(autocorr, distance=int(sr * 0.05))
                if len(peaks) >= 2:
                    # Vibrato rate from peak spacing
                    periods = np.diff(peaks)
                    if len(periods) > 0:
                        vibrato_period_samples = np.mean(periods)
                        features["vibrato_rate_hz"] = sr / vibrato_period_samples
                        features["vibrato_depth"] = float(
                            np.std(envelope) / (np.mean(envelope) + 1e-10)
                        )
                    else:
                        features["vibrato_rate_hz"] = 0.0
                        features["vibrato_depth"] = 0.0
                else:
                    features["vibrato_rate_hz"] = 0.0
                    features["vibrato_depth"] = 0.0
            else:
                features["vibrato_rate_hz"] = 0.0
                features["vibrato_depth"] = 0.0

            # Jitter (frequency instability)
            if len(voiced_f0) > 1:
                features["jitter"] = float(
                    np.std(np.diff(voiced_f0)) / (np.mean(voiced_f0) + 1e-10)
                )
            else:
                features["jitter"] = 0.0

            # Shimmer (amplitude instability)
            zero_crossings = np.where(np.diff(np.sign(audio)))[0]
            if len(zero_crossings) > 4:
                peak_amps = []
                for i in range(0, len(zero_crossings) - 2, 2):
                    start = zero_crossings[i]
                    end = zero_crossings[min(i + 2, len(zero_crossings) - 1)]
                    if end > start:
                        peak_amps.append(np.max(np.abs(audio[start:end])))
                if len(peak_amps) > 1:
                    features["shimmer"] = float(np.std(peak_amps) / (np.mean(peak_amps) + 1e-10))
                else:
                    features["shimmer"] = 0.0
            else:
                features["shimmer"] = 0.0

            # === MFCCs (13 features) ===
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f"mfcc_{i + 1}"] = float(np.mean(mfcc[i]))

            # Spectral contrast
            spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            features["spectral_contrast"] = float(np.mean(spec_contrast))

            # Spectral flux
            S = np.abs(librosa.stft(audio + 1e-8))
            flux = np.linalg.norm(S[:, 1:] - S[:, :-1], axis=0)
            features["spectral_flux"] = float(np.mean(flux) * 100)

            # === Rhythm Factors (3 features) ===
            # Onset detection
            onset_frames = librosa.onset.onset_detect(y=audio, sr=sr, wait=1)
            features["onset_rate_hz"] = float(len(onset_frames) / (len(audio) / sr))

            # ICI (inter-onset interval)
            if len(onset_frames) > 1:
                onset_times = librosa.frames_to_time(onset_frames, sr=sr)
                icis = np.diff(onset_times) * 1000  # Convert to ms
                features["median_ici_ms"] = float(np.median(icis))
                features["ici_coefficient_of_variation"] = float(
                    np.std(icis) / (np.mean(icis) + 1e-10)
                )
            else:
                features["median_ici_ms"] = 0.0
                features["ici_coefficient_of_variation"] = 0.0

        except Exception:
            # Fallback: return zeros for all features
            default_features = {
                "mean_f0_hz": 0.0,
                "duration_ms": len(audio) / sr * 1000.0,
                "f0_range_hz": 0.0,
                "harmonic_to_noise_ratio": 0.0,
                "spectral_flatness": 0.0,
                "harmonicity": 0.0,
                "attack_time_ms": 0.0,
                "decay_time_ms": 0.0,
                "sustain_level": 0.0,
                "vibrato_rate_hz": 0.0,
                "vibrato_depth": 0.0,
                "jitter": 0.0,
                "shimmer": 0.0,
                "mfcc_1": 0.0,
                "mfcc_2": 0.0,
                "mfcc_3": 0.0,
                "mfcc_4": 0.0,
                "mfcc_5": 0.0,
                "mfcc_6": 0.0,
                "mfcc_7": 0.0,
                "mfcc_8": 0.0,
                "mfcc_9": 0.0,
                "mfcc_10": 0.0,
                "mfcc_11": 0.0,
                "mfcc_12": 0.0,
                "mfcc_13": 0.0,
                "spectral_contrast": 0.0,
                "spectral_flux": 0.0,
                "onset_rate_hz": 0.0,
                "median_ici_ms": 0.0,
                "ici_coefficient_of_variation": 0.0,
            }
            features = default_features

        return features


# =============================================================================
# Step 3: DBSCAN Phrase Clustering
# =============================================================================


def cluster_phrases_dbscan(
    candidates: List[PhraseCandidate], eps: float = 0.5, min_samples: int = 5
) -> List[AtomicPhrase]:
    """
    Cluster phrase candidates using DBSCAN.

    DBSCAN is ideal because:
    - No need to specify number of clusters
    - Handles noise/outliers
    - Can find arbitrary cluster shapes

    Args:
        candidates: List of phrase candidates
        eps: Maximum distance between samples in same cluster
        min_samples: Minimum samples in a neighborhood to form a cluster

    Returns:
        List of atomic phrases (one per cluster, excluding noise)
    """
    if len(candidates) == 0:
        return []

    # Extract feature vectors
    feature_names = list(candidates[0].features_29d.keys())
    X = np.array([[c.features_29d[name] for name in feature_names] for c in candidates])

    # Normalize features
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-8
    X_normalized = (X - X_mean) / X_std

    # Apply DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(X_normalized)

    # Create atomic phrases for each cluster (excluding noise)
    phrases = []
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)  # Exclude noise

    for cluster_id in unique_labels:
        # Get candidates in this cluster
        member_indices = np.where(labels == cluster_id)[0]
        member_candidates = [candidates[i] for i in member_indices]

        # Compute representative features (median)
        member_features = X_normalized[member_indices]
        representative_features_normalized = np.median(member_features, axis=0)
        representative_features = representative_features_normalized * X_std + X_mean

        features_dict = {
            name: representative_features_normalized[i] for i, name in enumerate(feature_names)
        }

        # Calculate atomicity metrics
        intra_sim, inter_sim = _calculate_cluster_similarities(member_indices, labels, X_normalized)

        phrase = AtomicPhrase(
            phrase_id=f"phrase_{cluster_id}",
            cluster_id=cluster_id,
            features_29d=features_dict,
            member_candidates=member_candidates,
            intra_cluster_similarity=intra_sim,
            inter_cluster_similarity=inter_sim,
            is_atomic=intra_sim > 0.7,  # Atomic if high intra-cluster similarity
        )
        phrases.append(phrase)

    return phrases


def _calculate_cluster_similarities(
    member_indices: np.ndarray, labels: np.ndarray, X: np.ndarray
) -> Tuple[float, float]:
    """Calculate intra and inter cluster similarity."""
    if len(member_indices) == 0:
        return 0.0, 0.0

    # Intra-cluster similarity (1 - normalized avg distance)
    if len(member_indices) > 1:
        member_points = X[member_indices]
        distances = pdist(member_points, metric="euclidean")
        intra_distance = np.mean(distances)
        intra_sim = 1.0 / (1.0 + intra_distance)  # Convert to similarity
    else:
        intra_sim = 1.0  # Single point = perfectly similar

    # Inter-cluster similarity (avg distance to other clusters)
    other_indices = np.where(labels != labels[member_indices[0]])[0]
    if len(other_indices) > 0:
        other_points = X[other_indices]
        member_points = X[member_indices]
        # Compute min distance from each member to any other cluster
        inter_distances = []
        for mp in member_points:
            dists = np.linalg.norm(other_points - mp, axis=1)
            inter_distances.append(np.min(dists))
        inter_distance = np.mean(inter_distances)
        inter_sim = 1.0 / (1.0 + inter_distance)
    else:
        inter_sim = 0.0

    return float(intra_sim), float(inter_sim)


# =============================================================================
# Step 4: Atomicity Testing
# =============================================================================


def calculate_phrase_atomicity(
    candidates: List[PhraseCandidate], all_candidates: List[PhraseCandidate]
) -> Tuple[float, float, bool]:
    """
    Calculate atomicity of a phrase cluster.

    A phrase is atomic if:
    - High intra-cluster similarity (members are similar)
    - Low inter-cluster similarity (distinct from other phrases)

    Args:
        candidates: Members of this phrase
        all_candidates: All candidates (for inter-cluster comparison)

    Returns:
        (intra_cluster_similarity, inter_cluster_similarity, is_atomic)
    """
    if len(candidates) == 0:
        return 0.0, 0.0, False

    # Extract features
    feature_names = list(candidates[0].features_29d.keys())
    X = np.array([[c.features_29d[name] for name in feature_names] for c in all_candidates])

    # Normalize
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-8
    X_normalized = (X - X_mean) / X_std

    # Get indices for this cluster (use window_id for comparison)
    member_indices = []
    for c in candidates:
        # Find matching candidate by window_id and source_sentence_idx
        for i, ac in enumerate(all_candidates):
            if ac.window_id == c.window_id and ac.source_sentence_idx == c.source_sentence_idx:
                member_indices.append(i)
                break

    intra_sim, inter_sim = _calculate_cluster_similarities_from_indices(
        member_indices, X_normalized
    )

    # Atomic if high intra and low inter
    # Note: Thresholds are more lenient for high-dimensional space
    is_atomic = (intra_sim > 0.2) and (inter_sim < 0.6)

    return float(intra_sim), float(inter_sim), is_atomic


def _calculate_cluster_similarities_from_indices(
    member_indices: List[int], X: np.ndarray
) -> Tuple[float, float]:
    """Calculate similarities from indices."""
    if len(member_indices) == 0:
        return 0.0, 0.0

    member_points = X[member_indices]

    # Intra-cluster
    if len(member_indices) > 1:
        distances = pdist(member_points, metric="euclidean")
        intra_distance = np.mean(distances)
        intra_sim = 1.0 / (1.0 + intra_distance)
    else:
        intra_sim = 1.0

    # Inter-cluster
    other_indices = [i for i in range(len(X)) if i not in member_indices]
    if len(other_indices) > 0:
        other_points = X[other_indices]
        inter_distances = []
        for mp in member_points:
            dists = np.linalg.norm(other_points - mp, axis=1)
            inter_distances.append(np.min(dists))
        inter_distance = np.mean(inter_distances)
        inter_sim = 1.0 / (1.0 + inter_distance)
    else:
        inter_sim = 0.0

    return float(intra_sim), float(inter_sim)


# =============================================================================
# Step 5: Compositionality Testing
# =============================================================================


def detect_compositionality(sentences: List[Sentence]) -> Dict[str, float]:
    """
    Detect phrase reuse (compositionality) across sentences.

    A phrase is compositional if it appears in multiple contexts.

    Args:
        sentences: List of sentences with phrase sequences

    Returns:
        Dictionary mapping phrase_id to compositionality score
    """
    # Count phrase occurrences across sentences
    phrase_sentence_counts: Dict[str, set] = {}

    for sentence in sentences:
        seen_phrases = set()
        for phrase in sentence.phrases:
            if phrase.phrase_id not in seen_phrases:
                if phrase.phrase_id not in phrase_sentence_counts:
                    phrase_sentence_counts[phrase.phrase_id] = set()
                phrase_sentence_counts[phrase.phrase_id].add(sentence.sentence_id)
                seen_phrases.add(phrase.phrase_id)

    # Calculate compositionality scores
    scores = {}
    for phrase_id, sentence_ids in phrase_sentence_counts.items():
        # Compositionality = number of sentences containing phrase / total sentences
        score = len(sentence_ids) / len(sentences)
        scores[phrase_id] = float(score)

    return scores


# =============================================================================
# Step 6: Grammar Rule Building
# =============================================================================


def build_grammar_rules(sentences: List[Sentence]) -> List[GrammarRule]:
    """
    Build grammar rules from observed phrase transitions.

    Args:
        sentences: List of sentences with phrase sequences

    Returns:
        List of grammar rules
    """
    transition_counts: Dict[Tuple[str, str], int] = {}
    transition_contexts: Dict[Tuple[str, str], set] = {}
    phrase_counts: Dict[str, int] = {}

    # Count transitions
    for sentence in sentences:
        for i in range(len(sentence.phrases) - 1):
            from_phrase = sentence.phrases[i].phrase_id
            to_phrase = sentence.phrases[i + 1].phrase_id

            key = (from_phrase, to_phrase)
            transition_counts[key] = transition_counts.get(key, 0) + 1

            if key not in transition_contexts:
                transition_contexts[key] = set()
            transition_contexts[key].add(sentence.context)

            phrase_counts[from_phrase] = phrase_counts.get(from_phrase, 0) + 1

    # Calculate probabilities and create rules
    rules = []
    for (from_phrase, to_phrase), count in transition_counts.items():
        # P(to | from) = count(from->to) / count(from)
        probability = count / phrase_counts.get(from_phrase, 1)

        rule = GrammarRule(
            from_phrase_id=from_phrase,
            to_phrase_id=to_phrase,
            frequency=count,
            probability=probability,
            contexts=list(transition_contexts[(from_phrase, to_phrase)]),
        )
        rules.append(rule)

    # Sort by frequency
    rules.sort(key=lambda r: r.frequency, reverse=True)

    return rules


# =============================================================================
# Complete Pipeline
# =============================================================================


def extract_phrases_sentences_grammar(
    audio_dir: Path,
    annotations_file: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    pelt_penalty: float = 10.0,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
) -> ExtractionResult:
    """
    Complete extraction pipeline.

    Args:
        audio_dir: Directory containing audio files
        annotations_file: Optional annotations file with context labels
        output_dir: Optional output directory for segments and metadata
        pelt_penalty: PELT penalty parameter
        dbscan_eps: DBSCAN epsilon parameter
        dbscan_min_samples: DBSCAN min_samples parameter

    Returns:
        ExtractionResult with sentences, phrases, grammar rules
    """
    # Create output directory
    if output_dir is None:
        output_dir = audio_dir.parent / "extraction_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_segments_dir = output_dir / "audio_segments"
    audio_segments_dir.mkdir(exist_ok=True)

    # Load annotations if provided
    annotations = {}
    if annotations_file and annotations_file.exists():
        with open(annotations_file, "r") as f:
            annotations = json.load(f)

    # Process each audio file
    all_sentences = []
    all_candidates = []
    sentence_id_counter = 0

    audio_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3"))

    for audio_file in audio_files:
        print(f"Processing {audio_file.name}...")

        # Load audio
        audio, sr = librosa.load(str(audio_file), sr=None)

        # Get context from annotations
        context = annotations.get(audio_file.stem, {}).get("context", "unknown")

        # Step 1: Segment into sentences
        change_points = segment_sentences_pelt(audio, sr, penalty=pelt_penalty)

        # Split into sentences
        prev_change = 0
        for i, change_point in enumerate(change_points + [len(audio)]):
            if i < len(change_points):
                next_change = change_points[i]
            else:
                next_change = len(audio)

            sentence_audio = audio[prev_change:next_change]

            # Skip if too short
            if len(sentence_audio) < sr * 0.1:
                prev_change = next_change
                continue

            # Step 2: Extract phrase candidates
            candidates = extract_phrase_candidates(sentence_audio, sr)

            # Assign sentence index
            for candidate in candidates:
                candidate.source_sentence_idx = sentence_id_counter

            all_candidates.extend(candidates)

            # Create sentence placeholder (phrases will be assigned after clustering)
            sentence = Sentence(
                sentence_id=f"sentence_{sentence_id_counter}",
                audio_path=str(audio_file),
                start_sample=prev_change,
                end_sample=next_change,
                phrases=[],
                context=context,
                compositionality_score=0.0,
            )
            all_sentences.append(sentence)
            sentence_id_counter += 1

            prev_change = next_change

    # Step 3: Cluster all candidates into atomic phrases
    print(f"Clustering {len(all_candidates)} candidates...")
    phrases = cluster_phrases_dbscan(all_candidates, eps=dbscan_eps, min_samples=dbscan_min_samples)

    print(f"Found {len(phrases)} atomic phrases")

    # Assign phrases to sentences
    for sentence in all_sentences:
        sentence_phrases = []
        for candidate in all_candidates:
            if candidate.source_sentence_idx == int(sentence.sentence_id.split("_")[1]):
                # Find which cluster this candidate belongs to
                for phrase in phrases:
                    if candidate in phrase.member_candidates:
                        sentence_phrases.append((candidate.start_sample, phrase))
                        break

        # Sort by position and deduplicate
        sentence_phrases.sort(key=lambda x: x[0])
        seen_phrases = set()
        unique_phrases = []
        for _, phrase in sentence_phrases:
            if phrase.phrase_id not in seen_phrases:
                unique_phrases.append(phrase)
                seen_phrases.add(phrase.phrase_id)

        sentence.phrases = unique_phrases

    # Step 5: Detect compositionality
    compositionality_scores = detect_compositionality(all_sentences)

    # Update sentences with compositionality scores
    for sentence in all_sentences:
        if sentence.phrases:
            avg_score = np.mean(
                [compositionality_scores.get(p.phrase_id, 0.0) for p in sentence.phrases]
            )
            sentence.compositionality_score = float(avg_score)

    # Step 6: Build grammar
    grammar_rules = build_grammar_rules(all_sentences)

    # Export audio segments
    print(f"Exporting {len(phrases)} phrase audio segments...")
    for phrase in phrases:
        # Export one representative audio segment
        if phrase.member_candidates:
            representative = phrase.member_candidates[0]
            segment_path = audio_segments_dir / f"{phrase.phrase_id}.wav"
            sf.write(str(segment_path), representative.audio_segment, 48000)

    # Create metadata
    metadata = {
        "num_sentences": len(all_sentences),
        "num_phrases": len(phrases),
        "num_grammar_rules": len(grammar_rules),
        "parameters": {
            "pelt_penalty": pelt_penalty,
            "dbscan_eps": dbscan_eps,
            "dbscan_min_samples": dbscan_min_samples,
        },
    }

    # Export metadata
    metadata_path = output_dir / "extraction_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Export grammar
    grammar_path = output_dir / "grammar_rules.json"
    with open(grammar_path, "w") as f:
        json.dump(
            [
                {
                    "from": r.from_phrase_id,
                    "to": r.to_phrase_id,
                    "frequency": r.frequency,
                    "probability": r.probability,
                    "contexts": r.contexts,
                }
                for r in grammar_rules
            ],
            f,
            indent=2,
        )

    print("\n✓ Extraction complete!")
    print(f"  Sentences: {len(all_sentences)}")
    print(f"  Atomic phrases: {len(phrases)}")
    print(f"  Grammar rules: {len(grammar_rules)}")
    print(f"  Output: {output_dir}")

    return ExtractionResult(
        sentences=all_sentences,
        phrases=phrases,
        grammar_rules=grammar_rules,
        audio_segments_dir=audio_segments_dir,
        metadata=metadata,
    )


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python unified_extraction.py <audio_dir> [annotations_file] [output_dir]")
        sys.exit(1)

    audio_dir = Path(sys.argv[1])
    annotations_file = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    output_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else None

    result = extract_phrases_sentences_grammar(
        audio_dir=audio_dir, annotations_file=annotations_file, output_dir=output_dir
    )
