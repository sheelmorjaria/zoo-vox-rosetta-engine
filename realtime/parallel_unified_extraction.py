"""
Parallel Unified Extraction Pipeline for 16-Core 32-Thread Machine

Processes the Egyptian fruit bat dataset in parallel to extract:
- Sentences (each vocalization)
- Phrases (atomic units via PELT + sliding window + DBSCAN)
- Grammar rules (phrase transitions)
- Segmented audio

Optimized for 16-core CPU with multiprocessing.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import json

# Audio processing
import librosa

# Change point detection
import ruptures as rpt

# Clustering
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Multiprocessing
import multiprocessing as mp


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class Sentence:
    """A single vocalization (treated as a "sentence")"""
    sentence_id: str
    audio_file: str
    context: int
    emitter: int
    addressee: int
    duration_sec: float
    change_points: List[int] = field(default_factory=list)
    phrases: List[str] = field(default_factory=list)  # phrase_ids


@dataclass
class PhraseCandidate:
    """Candidate phrase from sliding window extraction"""
    audio_segment: np.ndarray
    start_sample: int
    end_sample: int
    features_29d: Dict[str, float]
    source_sentence_id: str
    window_id: int
    context: int


@dataclass
class AtomicPhrase:
    """Validated atomic phrase (clustered from candidates)"""
    phrase_id: str
    cluster_id: int
    features_29d: Dict[str, float]
    member_candidates: List[Dict[str, Any]]
    intra_cluster_similarity: float
    inter_cluster_similarity: float
    is_atomic: bool
    contexts: List[int] = field(default_factory=list)


@dataclass
class GrammarRule:
    """Grammar rule from phrase transitions"""
    antecedent: str  # phrase_id
    consequent: str  # phrase_id
    transition_count: int
    probability: float
    contexts: List[int] = field(default_factory=list)


@dataclass
class ExtractionResult:
    """Complete extraction results"""
    sentences: List[Sentence]
    phrases: List[AtomicPhrase]
    grammar_rules: List[GrammarRule]
    total_candidates: int
    total_atomic_phrases: int
    processing_time_sec: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Feature Extraction (29D)
# =============================================================================

def extract_29d_features(audio: np.ndarray, sr: int) -> Dict[str, float]:
    """Extract 29-dimensional acoustic features with micro-dynamics."""
    features = {}

    # === Fundamental (3 features) ===
    # Extract pitch using pyin with ultrasonic support
    # Adjust fmin/fmax based on sample rate for bat vocalizations
    if sr >= 200000:  # Ultrasonic (bats: 250kHz)
        fmin_hz = 10000.0  # 10kHz minimum for bats
        fmax_hz = 100000.0  # 100kHz maximum for bats
        frame_len = 4096  # Larger frame for ultrasonic
    elif sr >= 96000:  # High quality
        fmin_hz = librosa.note_to_hz('C2')
        fmax_hz = librosa.note_to_hz('C7')
        frame_len = 2048
    else:  # Standard audio
        fmin_hz = librosa.note_to_hz('C2')
        fmax_hz = librosa.note_to_hz('C7')
        frame_len = 2048

    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            sr=sr,
            fmin=fmin_hz,
            fmax=fmax_hz,
            frame_length=frame_len,
            hop_length=512
        )
    except Exception:
        # Fallback: use zero array if pyin fails
        f0 = np.array([0.0])
        voiced_flag = np.array([False])
        voiced_probs = np.array([0.0])

    voiced_f0 = f0[voiced_flag]

    if len(voiced_f0) > 0:
        features['mean_f0_hz'] = float(np.mean(voiced_f0))
        features['f0_range_hz'] = float(np.max(voiced_f0) - np.min(voiced_f0))
    else:
        features['mean_f0_hz'] = 0.0
        features['f0_range_hz'] = 0.0

    features['duration_ms'] = len(audio) / sr * 1000.0

    # === Grit Factors (3 features) ===
    # Harmonic-to-noise ratio (adjust for ultrasonic)
    if sr >= 200000:
        hnr_fmin, hnr_fmax = 10000.0, 100000.0
    else:
        hnr_fmin, hnr_fmax = 200.0, 16000.0

    try:
        harmonicity = librosa.pyin(audio, sr=sr, fmin=hnr_fmin, fmax=hnr_fmax, frame_length=frame_len)[2]
        features['harmonic_to_noise_ratio'] = float(np.mean(harmonicity[~np.isnan(harmonicity)]))
    except Exception:
        features['harmonic_to_noise_ratio'] = 0.0

    # Spectral flatness
    spectral_flatness = librosa.feature.spectral_flatness(y=audio, hop_length=512)[0]
    features['spectral_flatness'] = float(np.mean(spectral_flatness))

    # Harmonicity (alternative measure)
    features['harmonicity'] = float(np.mean(voiced_probs[~np.isnan(voiced_probs)]))

    # === Motion Factors (7 features) ===
    # Envelope extraction
    envelope = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=512)

    # Attack time
    onset_frames = librosa.onset.onset_detect(onset_envelope=envelope, sr=sr, hop_length=512)
    if len(onset_frames) > 0:
        features['attack_time_ms'] = float(onset_frames[0] * 512 / sr * 1000)
    else:
        features['attack_time_ms'] = 0.0

    # Decay time (time to 10% of peak)
    peak_idx = np.argmax(envelope)
    peak_val = envelope[peak_idx]
    decay_threshold = 0.1 * peak_val
    decay_frames = np.where(envelope[peak_idx:] < decay_threshold)[0]
    if len(decay_frames) > 0:
        features['decay_time_ms'] = float((peak_idx + decay_frames[0]) * 512 / sr * 1000)
    else:
        features['decay_time_ms'] = 0.0

    # Sustain level (normalized)
    features['sustain_level'] = float(np.mean(envelope) / (peak_val + 1e-6))

    # Vibrato
    if len(voiced_f0) > 10:
        # Autocorrelation to find vibrato rate
        autocorr = np.correlate(voiced_f0 - np.mean(voiced_f0),
                                voiced_f0 - np.mean(voiced_f0), mode='full')
        autocorr = autocorr[len(autocorr)//2:]

        # Find peaks in autocorrelation
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(autocorr[1:50])  # Look for 2-50 frame lag

        if len(peaks) > 0:
            vibrato_period_frames = peaks[0] + 1
            features['vibrato_rate_hz'] = float(sr / 512 / vibrato_period_frames)
        else:
            features['vibrato_rate_hz'] = 0.0

        # Vibrato depth
        features['vibrato_depth'] = float(np.std(voiced_f0) / (np.mean(voiced_f0) + 1e-6))
    else:
        features['vibrato_rate_hz'] = 0.0
        features['vibrato_depth'] = 0.0

    # Jitter (frequency perturbation)
    if len(voiced_f0) > 2:
        f0_diff = np.diff(voiced_f0)
        features['jitter'] = float(np.mean(np.abs(f0_diff)) / (np.mean(voiced_f0) + 1e-6))
    else:
        features['jitter'] = 0.0

    # Shimmer (amplitude perturbation)
    if len(envelope) > 2:
        env_diff = np.diff(envelope)
        features['shimmer'] = float(np.mean(np.abs(env_diff)) / (np.mean(envelope) + 1e-6))
    else:
        features['shimmer'] = 0.0

    # === Fingerprint Factors (13 MFCCs) ===
    mfcc_frame_based = librosa.feature.mfcc(
        y=audio.astype(np.float32),
        sr=sr,
        n_mfcc=13,
        n_fft=2048,
        hop_length=512
    )

    for i in range(13):
        features[f'mfcc_{i+1}'] = float(np.mean(mfcc_frame_based[i]))

    # Spectral contrast
    spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    features['spectral_contrast'] = float(np.mean(spec_contrast))

    # === Spectral Dynamics (1 feature) ===
    # Spectral flux
    spectral_flux = librosa.onset.onset_strength(y=audio, sr=sr)
    features['spectral_flux'] = float(np.mean(spectral_flux))

    # === Rhythm Factors (3 features) ===
    # Onset detection
    onsets = librosa.onset.onset_detect(y=audio, sr=sr, hop_length=512, backtrack=True)

    if len(onsets) > 1:
        # Inter-onset intervals
        intervals = np.diff(onsets) * 512 / sr * 1000  # Convert to ms
        features['median_ici_ms'] = float(np.median(intervals))
        features['onset_rate_hz'] = float(len(onsets) / (len(audio) / sr))
        features['ici_coefficient_of_variation'] = float(np.std(intervals) / (np.mean(intervals) + 1e-6))
    else:
        features['median_ici_ms'] = 0.0
        features['onset_rate_hz'] = 0.0
        features['ici_coefficient_of_variation'] = 0.0

    return features


# =============================================================================
# PELT Sentence Segmentation
# =============================================================================

def segment_sentences_pelt(
    audio: np.ndarray,
    sr: int,
    penalty: float = 10.0,
    min_segment_length_sec: float = 0.3
) -> List[int]:
    """
    Segment audio into sentences using PELT change point detection.

    Returns list of sample indices where sentences change.
    """
    # Extract features for change point detection
    n_samples = len(audio)

    # If audio is too short, return single sentence
    if n_samples < sr * min_segment_length_sec:
        return [0, n_samples]

    # Multi-resolution feature extraction
    # MFCCs (13)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=512)

    # Spectral contrast (7)
    spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, hop_length=512)

    # Chroma (12)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=512)

    # Combine features
    feature_matrix = np.concatenate([mfcc.T, spec_contrast.T, chroma.T], axis=1)

    # Normalize features
    feature_matrix = (feature_matrix - np.mean(feature_matrix, axis=0)) / (np.std(feature_matrix, axis=0) + 1e-6)

    # Convert penalty to feature frame space
    samples_per_frame = 512
    n_frames = feature_matrix.shape[0]
    min_size = int(min_segment_length_sec * sr / samples_per_frame)

    # Apply PELT
    try:
        algo = rpt.Pelt(model="rbf", min_size=min_size, jump=1).fit(feature_matrix)
        change_point_indices = algo.predict(penalty)

        # Convert frame indices to sample indices
        change_points = [int(cp * samples_per_frame) for cp in change_point_indices if cp < n_frames - 1]

        # Ensure start and end are included
        if len(change_points) == 0 or change_points[0] != 0:
            change_points = [0] + change_points
        if change_points[-1] != n_samples:
            change_points.append(n_samples)

        return change_points

    except Exception as e:
        # Fallback: return single sentence
        return [0, n_samples]


# =============================================================================
# Sliding Window Phrase Extraction
# =============================================================================

def extract_phrase_candidates(
    audio: np.ndarray,
    sr: int,
    sentence_id: str,
    context: int,
    window_sizes_sec: List[float] = None
) -> List[PhraseCandidate]:
    """
    Extract phrase candidates using multi-scale sliding windows.

    Uses multiple window sizes (50ms to 500ms) to capture phrases
    of different durations.
    """
    if window_sizes_sec is None:
        window_sizes_sec = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

    candidates = []
    window_id = 0

    for window_sec in window_sizes_sec:
        window_size = int(window_sec * sr)
        hop_size = window_size // 2  # 50% overlap

        for start in range(0, len(audio) - window_size + 1, hop_size):
            end = start + window_size
            segment = audio[start:end]

            # Skip very quiet segments
            rms = np.sqrt(np.mean(segment**2))
            if rms < 0.001:
                continue

            # Extract 29D features
            features = extract_29d_features(segment, sr)

            candidate = PhraseCandidate(
                audio_segment=segment,
                start_sample=start,
                end_sample=end,
                features_29d=features,
                source_sentence_id=sentence_id,
                window_id=window_id,
                context=context
            )

            candidates.append(candidate)
            window_id += 1

    return candidates


# =============================================================================
# DBSCAN Phrase Clustering
# =============================================================================

def cluster_phrases_dbscan(
    candidates: List[PhraseCandidate],
    eps: float = 0.5,
    min_samples: int = 5
) -> List[AtomicPhrase]:
    """
    Cluster phrase candidates using DBSCAN.

    Returns list of atomic phrases with cluster assignments.
    """
    if len(candidates) < min_samples:
        return []

    # Extract feature matrix
    feature_names = list(candidates[0].features_29d.keys())
    X = np.array([[c.features_29d[name] for name in feature_names] for c in candidates])

    # Normalize features
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    # Apply DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(X_normalized)

    # Group candidates by cluster
    phrases = []
    unique_labels = set(labels)

    for cluster_id in unique_labels:
        if cluster_id == -1:  # Noise
            continue

        # Get members of this cluster
        member_indices = np.where(labels == cluster_id)[0]
        cluster_members = [candidates[i] for i in member_indices]

        # Calculate cluster centroid
        centroid_features = np.mean(X_normalized[member_indices], axis=0)

        # Calculate similarities
        intra_sim = _calculate_intra_cluster_similarity(X_normalized[member_indices])
        inter_sim = _calculate_inter_cluster_similarity(X_normalized, member_indices, labels, cluster_id)

        # Check atomicity
        is_atomic = (intra_sim > 0.2) and (inter_sim < 0.6)

        # Collect contexts
        contexts = list(set([c.context for c in cluster_members]))

        phrase = AtomicPhrase(
            phrase_id=f"phrase_{cluster_id}",
            cluster_id=cluster_id,
            features_29d=dict(zip(feature_names, scaler.inverse_transform([centroid_features])[0])),
            member_candidates=[{
                'source_sentence_id': c.source_sentence_id,
                'window_id': c.window_id,
                'start_sample': c.start_sample,
                'end_sample': c.end_sample,
                'context': c.context
            } for c in cluster_members],
            intra_cluster_similarity=intra_sim,
            inter_cluster_similarity=inter_sim,
            is_atomic=is_atomic,
            contexts=contexts
        )

        phrases.append(phrase)

    return phrases


def _calculate_intra_cluster_similarity(cluster_features: np.ndarray) -> float:
    """Calculate average pairwise similarity within cluster."""
    if len(cluster_features) < 2:
        return 1.0

    # Calculate pairwise cosine similarities
    n = len(cluster_features)
    similarities = []

    for i in range(n):
        for j in range(i + 1, n):
            # Cosine similarity
            dot = np.dot(cluster_features[i], cluster_features[j])
            norm_i = np.linalg.norm(cluster_features[i])
            norm_j = np.linalg.norm(cluster_features[j])
            if norm_i > 0 and norm_j > 0:
                sim = dot / (norm_i * norm_j)
                similarities.append(sim)

    return float(np.mean(similarities)) if similarities else 0.0


def _calculate_inter_cluster_similarity(
    all_features: np.ndarray,
    cluster_indices: np.ndarray,
    labels: np.ndarray,
    cluster_id: int
) -> float:
    """Calculate average similarity to nearest other cluster."""
    other_indices = np.where(labels != cluster_id)[0]

    if len(other_indices) == 0:
        return 0.0

    cluster_members = all_features[cluster_indices]
    other_members = all_features[other_indices]

    # Calculate centroid of this cluster
    centroid = np.mean(cluster_members, axis=0)

    # Calculate similarities to other cluster members
    similarities = []
    for other in other_members:
        dot = np.dot(centroid, other)
        norm_centroid = np.linalg.norm(centroid)
        norm_other = np.linalg.norm(other)
        if norm_centroid > 0 and norm_other > 0:
            sim = dot / (norm_centroid * norm_other)
            similarities.append(sim)

    return float(np.mean(similarities)) if similarities else 0.0


# =============================================================================
# Compositionality Testing
# =============================================================================

def detect_compositionality(
    sentences: List[Sentence],
    phrases: List[AtomicPhrase]
) -> Dict[str, Any]:
    """
    Detect phrase reuse patterns (compositionality).

    Returns statistics on phrase reuse across sentences.
    """
    # Count phrase occurrences across sentences
    phrase_usage = {}

    for sentence in sentences:
        for phrase_id in sentence.phrases:
            if phrase_id not in phrase_usage:
                phrase_usage[phrase_id] = {
                    'sentence_count': 0,
                    'contexts': set()
                }
            phrase_usage[phrase_id]['sentence_count'] += 1
            phrase_usage[phrase_id]['contexts'].add(sentence.context)

    # Calculate statistics
    reusable_phrases = [p for p, stats in phrase_usage.items() if stats['sentence_count'] > 1]

    return {
        'total_unique_phrases': len(phrase_usage),
        'reusable_phrases': len(reusable_phrases),
        'compositionality_ratio': len(reusable_phrases) / len(phrase_usage) if phrase_usage else 0.0,
        'phrase_usage': phrase_usage
    }


# =============================================================================
# Grammar Rule Extraction
# =============================================================================

def extract_grammar_rules(sentences: List[Sentence]) -> List[GrammarRule]:
    """
    Extract grammar rules from phrase transitions.

    Rules are of the form: phrase_A -> phrase_B
    """
    transitions = {}

    for sentence in sentences:
        phrases = sentence.phrases
        for i in range(len(phrases) - 1):
            antecedent = phrases[i]
            consequent = phrases[i + 1]

            key = (antecedent, consequent)
            if key not in transitions:
                transitions[key] = {
                    'count': 0,
                    'contexts': set()
                }

            transitions[key]['count'] += 1
            transitions[key]['contexts'].add(sentence.context)

    # Convert to rules with probabilities
    rules = []
    antecedent_totals = {}

    for (ant, cons), data in transitions.items():
        if ant not in antecedent_totals:
            antecedent_totals[ant] = 0
        antecedent_totals[ant] += data['count']

    for (ant, cons), data in transitions.items():
        probability = data['count'] / antecedent_totals[ant]
        rule = GrammarRule(
            antecedent=ant,
            consequent=cons,
            transition_count=data['count'],
            probability=probability,
            contexts=list(data['contexts'])
        )
        rules.append(rule)

    # Sort by transition count
    rules.sort(key=lambda r: r.transition_count, reverse=True)

    return rules


# =============================================================================
# Single File Processing (for parallelization)
# =============================================================================

def process_single_vocalization(args: Tuple[str, Dict[str, Any], Path, float, float, int]) -> Dict[str, Any]:
    """
    Process a single vocalization file.

    Args:
        args: Tuple of (audio_filename, annotation, audio_dir, penalty, eps, min_samples)

    Returns:
        Dictionary with sentence, candidates, phrases, and metadata
    """
    audio_filename, annotation, audio_dir, penalty, eps, min_samples = args

    try:
        # Load audio
        audio_path = audio_dir / audio_filename
        audio, sr = librosa.load(str(audio_path), sr=None)

        # Create sentence ID
        sentence_id = audio_filename.replace('.wav', '')

        # Step 1: Segment into sentences (for this bat dataset, each file is a sentence)
        # We still run PELT to check for internal structure
        change_points = segment_sentences_pelt(audio, sr, penalty=penalty)

        # Step 2: Extract phrase candidates
        candidates = extract_phrase_candidates(
            audio,
            sr,
            sentence_id=sentence_id,
            context=annotation['context']
        )

        # Create sentence object
        sentence = Sentence(
            sentence_id=sentence_id,
            audio_file=audio_filename,
            context=annotation['context'],
            emitter=annotation['emitter'],
            addressee=annotation['addressee'],
            duration_sec=len(audio) / sr,
            change_points=change_points
        )

        return {
            'success': True,
            'sentence': sentence,
            'candidates': candidates,
            'audio': audio,
            'sr': sr,
            'num_candidates': len(candidates)
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'audio_filename': audio_filename
        }


# =============================================================================
# Main Parallel Pipeline
# =============================================================================

def extract_phrases_sentences_grammar_parallel(
    audio_dir: Path,
    annotations_file: Path,
    output_dir: Path,
    num_workers: int = 16,
    pelt_penalty: float = 10.0,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
    max_files: Optional[int] = None
) -> ExtractionResult:
    """
    Parallel unified extraction pipeline optimized for 16-core machine.

    Pipeline:
    1. Load annotations
    2. Process audio files in parallel (16 workers)
    3. Collect all phrase candidates
    4. Cluster phrases (DBSCAN)
    5. Test atomicity and compositionality
    6. Extract grammar rules
    7. Export results

    Args:
        audio_dir: Directory containing audio files
        annotations_file: CSV file with annotations
        output_dir: Directory for exports
        num_workers: Number of parallel workers (default 16)
        pelt_penalty: PELT penalty parameter
        dbscan_eps: DBSCAN epsilon parameter
        dbscan_min_samples: DBSCAN min_samples parameter
        max_files: Optional limit on files to process (for testing)

    Returns:
        ExtractionResult with all extracted data
    """
    import time
    start_time = time.time()

    print("=" * 80)
    print("PARALLEL UNIFIED EXTRACTION PIPELINE")
    print("=" * 80)
    print(f"Audio directory: {audio_dir}")
    print(f"Annotations file: {annotations_file}")
    print(f"Output directory: {output_dir}")
    print(f"Workers: {num_workers}")
    print("=" * 80)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    audio_segments_dir = output_dir / "audio_segments"
    audio_segments_dir.mkdir(exist_ok=True)

    # ========================================================================
    # Step 1: Load annotations
    # ========================================================================
    print("\n[1/7] Loading annotations...")
    df = pd.read_csv(annotations_file)

    # Parse annotations
    annotations = []
    for _, row in df.iterrows():
        # Get filename from 'File Name' column
        audio_filename = str(row['File Name'])
        annotations.append({
            'filename': audio_filename,
            'context': int(row['Context']),
            'emitter': int(row['Emitter']),
            'addressee': int(row['Addressee'])
        })

    # Filter to only those with context (user requirement)
    # All have context 0-12, so no filtering needed
    print(f"  Loaded {len(annotations)} vocalizations")
    print(f"  All have context labels")

    # Limit files if specified
    if max_files:
        annotations = annotations[:max_files]
        print(f"  Limited to {max_files} files for testing")

    # ========================================================================
    # Step 2: Process audio files in parallel
    # ========================================================================
    print(f"\n[2/7] Processing {len(annotations)} audio files with {num_workers} workers...")

    # Prepare arguments for parallel processing
    process_args = [
        (ann['filename'], ann, audio_dir, pelt_penalty, dbscan_eps, dbscan_min_samples)
        for ann in annotations
    ]

    all_results = []
    all_candidates = []
    sentences = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all jobs
        futures = {
            executor.submit(process_single_vocalization, args): args[0]
            for args in process_args
        }

        # Process completed jobs with progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing audio"):
            audio_filename = futures[future]
            try:
                result = future.result()
                if result['success']:
                    all_results.append(result)
                    all_candidates.extend(result['candidates'])
                    sentences.append(result['sentence'])
                else:
                    print(f"  ERROR processing {audio_filename}: {result.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"  EXCEPTION processing {audio_filename}: {e}")

    print(f"  Successfully processed: {len(all_results)}/{len(annotations)}")
    print(f"  Total phrase candidates: {len(all_candidates)}")

    # ========================================================================
    # Step 3: Cluster all phrase candidates
    # ========================================================================
    print(f"\n[3/7] Clustering {len(all_candidates)} phrase candidates (DBSCAN)...")
    print(f"  eps={dbscan_eps}, min_samples={dbscan_min_samples}")

    phrases = cluster_phrases_dbscan(
        all_candidates,
        eps=dbscan_eps,
        min_samples=dbscan_min_samples
    )

    print(f"  Found {len(phrases)} phrase clusters")
    atomic_phrases = [p for p in phrases if p.is_atomic]
    print(f"  Atomic phrases: {len(atomic_phrases)}/{len(phrases)}")

    # ========================================================================
    # Step 4: Assign phrases to sentences
    # ========================================================================
    print("\n[4/7] Assigning phrases to sentences...")

    # Build spatial index for candidates
    candidate_map = {}
    for candidate in all_candidates:
        key = (candidate.source_sentence_id, candidate.window_id)
        candidate_map[key] = candidate

    # For each sentence, find which phrases its candidates belong to
    for sentence in sentences:
        assigned_phrases = []

        # Find all candidates from this sentence
        sentence_candidates = [
            c for c in all_candidates
            if c.source_sentence_id == sentence.sentence_id
        ]

        # Assign each candidate to its cluster
        for candidate in sentence_candidates:
            # Find which cluster this candidate belongs to
            for phrase in phrases:
                # Check if candidate is in this phrase's members
                for member in phrase.member_candidates:
                    if (member['source_sentence_id'] == candidate.source_sentence_id and
                        member['window_id'] == candidate.window_id):
                        assigned_phrases.append(phrase.phrase_id)
                        break

        # Update sentence with assigned phrases
        sentence.phrases = list(set(assigned_phrases))  # Remove duplicates

    print(f"  Assigned phrases to {len(sentences)} sentences")

    # ========================================================================
    # Step 5: Test compositionality
    # ========================================================================
    print("\n[5/7] Testing compositionality (phrase reuse)...")

    compositionality = detect_compositionality(sentences, phrases)

    print(f"  Total unique phrases: {compositionality['total_unique_phrases']}")
    print(f"  Reusable phrases: {compositionality['reusable_phrases']}")
    print(f"  Compositionality ratio: {compositionality['compositionality_ratio']:.3f}")

    # ========================================================================
    # Step 6: Extract grammar rules
    # ========================================================================
    print("\n[6/7] Extracting grammar rules...")

    grammar_rules = extract_grammar_rules(sentences)

    print(f"  Found {len(grammar_rules)} grammar rules")
    if grammar_rules:
        print(f"  Top rule: {grammar_rules[0].antecedent} -> {grammar_rules[0].consequent} "
              f"(count={grammar_rules[0].transition_count}, prob={grammar_rules[0].probability:.3f})")

    # ========================================================================
    # Step 7: Export results
    # ========================================================================
    print("\n[7/7] Exporting results...")

    # Save sentences
    sentences_file = output_dir / "sentences.json"
    with open(sentences_file, 'w') as f:
        sentences_data = []
        for s in sentences:
            sentence_dict = asdict(s)
            sentences_data.append(sentence_dict)
        json.dump(sentences_data, f, indent=2)
    print(f"  Saved {len(sentences)} sentences to {sentences_file}")

    # Save phrases
    phrases_file = output_dir / "phrases.json"
    with open(phrases_file, 'w') as f:
        phrases_data = []
        for p in phrases:
            phrase_dict = asdict(p)
            phrases_data.append(phrase_dict)
        json.dump(phrases_data, f, indent=2)
    print(f"  Saved {len(phrases)} phrases to {phrases_file}")

    # Save grammar rules
    rules_file = output_dir / "grammar_rules.json"
    with open(rules_file, 'w') as f:
        rules_data = [asdict(r) for r in grammar_rules]
        json.dump(rules_data, f, indent=2)
    print(f"  Saved {len(grammar_rules)} grammar rules to {rules_file}")

    # Save metadata
    metadata = {
        'total_vocalizations': len(annotations),
        'successfully_processed': len(all_results),
        'total_candidates': len(all_candidates),
        'total_phrases': len(phrases),
        'atomic_phrases': len(atomic_phrases),
        'grammar_rules': len(grammar_rules),
        'compositionality_ratio': compositionality['compositionality_ratio'],
        'parameters': {
            'pelt_penalty': pelt_penalty,
            'dbscan_eps': dbscan_eps,
            'dbscan_min_samples': dbscan_min_samples,
            'num_workers': num_workers
        }
    }

    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata to {metadata_file}")

    # ========================================================================
    # Final timing
    # ========================================================================
    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Processing time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"Throughput: {len(all_results)/elapsed:.1f} vocalizations/second")
    print(f"\nResults:")
    print(f"  Sentences: {len(sentences)}")
    print(f"  Phrase candidates: {len(all_candidates)}")
    print(f"  Atomic phrases: {len(atomic_phrases)}")
    print(f"  Grammar rules: {len(grammar_rules)}")
    print(f"  Compositionality: {compositionality['compositionality_ratio']:.3f}")
    print(f"\nOutput directory: {output_dir}")
    print("=" * 80)

    return ExtractionResult(
        sentences=sentences,
        phrases=phrases,
        grammar_rules=grammar_rules,
        total_candidates=len(all_candidates),
        total_atomic_phrases=len(atomic_phrases),
        processing_time_sec=elapsed,
        metadata=metadata
    )


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Parallel Unified Extraction Pipeline for Egyptian Fruit Bat Dataset"
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        default="/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/audio",
        help="Directory containing audio files"
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default="/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/annotations.csv",
        help="Annotations CSV file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/extraction_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel workers (default: 16)"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit files to process (for testing)"
    )
    parser.add_argument(
        "--penalty",
        type=float,
        default=10.0,
        help="PELT penalty parameter"
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.5,
        help="DBSCAN epsilon parameter"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="DBSCAN min_samples parameter"
    )

    args = parser.parse_args()

    # Run pipeline
    result = extract_phrases_sentences_grammar_parallel(
        audio_dir=Path(args.audio_dir),
        annotations_file=Path(args.annotations),
        output_dir=Path(args.output_dir),
        num_workers=args.workers,
        pelt_penalty=args.penalty,
        dbscan_eps=args.eps,
        dbscan_min_samples=args.min_samples,
        max_files=args.max_files
    )
