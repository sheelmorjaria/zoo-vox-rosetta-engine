"""
Optimized Parallel Extraction Pipeline for Egyptian Fruit Bat Dataset

Key optimizations:
1. Skip PELT (each file is already a "sentence" - one vocalization)
2. Pre-compute MFCCs for entire audio, then segment (avoid redundant computation)
3. Reduced window sizes (3 instead of 7)
4. Faster feature extraction (avoid expensive PYIN for initial clustering)
5. Use soundfile instead of librosa for faster audio loading

Expected speedup: 5-10x faster than original

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import warnings

warnings.filterwarnings("ignore")

import json
import pickle

# Import optimized feature extraction
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf  # Faster than librosa
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

# Import librosa only for feature extraction
import librosa
from parallel_unified_extraction import (
    ExtractionResult,
    Sentence,
    cluster_phrases_dbscan,
    detect_compositionality,
    extract_grammar_rules,
)

# =============================================================================
# Data Models
# =============================================================================


@dataclass
class PhraseCandidate:
    """Lightweight phrase candidate - audio_segment NOT stored (memory optimization)"""

    start_sample: int
    end_sample: int
    features_29d: Dict[str, float]
    source_sentence_id: str
    window_id: int
    context: int


# =============================================================================
# Optimized Feature Extraction
# =============================================================================


def extract_features_fast(
    audio: np.ndarray,
    sr: int,
    mfcc_full: Optional[np.ndarray] = None,
    spec_centroid_full: Optional[np.ndarray] = None,
    spec_bandwidth_full: Optional[np.ndarray] = None,
    spec_rolloff_full: Optional[np.ndarray] = None,
    zcr_full: Optional[np.ndarray] = None,
    rms_full: Optional[np.ndarray] = None,
    start_sample: int = 0,
) -> Dict[str, float]:
    """
    Fast 29D feature extraction using pre-computed features.

    Optimizations:
    - Avoids expensive PYIN pitch tracking
    - Uses pre-computed MFCCs and spectral features
    - Simple statistical aggregations
    """
    features = {}

    # Duration
    features["duration_ms"] = len(audio) / sr * 1000.0

    # Compute hop-based indices for segment
    hop_length = 512
    start_frame = start_sample // hop_length
    end_frame = (start_sample + len(audio)) // hop_length + 1

    # Extract segment features from pre-computed arrays
    if mfcc_full is not None:
        mfcc_seg = mfcc_full[:, start_frame:end_frame]
        for i in range(min(13, mfcc_seg.shape[0])):
            features[f"mfcc_{i + 1}"] = float(np.mean(mfcc_seg[i]))

    if spec_centroid_full is not None:
        features["spectral_centroid"] = float(np.mean(spec_centroid_full[start_frame:end_frame]))
        features["mean_f0_hz"] = features["spectral_centroid"]  # Proxy for F0
        features["f0_range_hz"] = float(np.std(spec_centroid_full[start_frame:end_frame]))

    if spec_bandwidth_full is not None:
        features["spectral_bandwidth"] = float(np.mean(spec_bandwidth_full[start_frame:end_frame]))

    if spec_rolloff_full is not None:
        features["spectral_rolloff"] = float(np.mean(spec_rolloff_full[start_frame:end_frame]))

    if zcr_full is not None:
        features["zcr"] = float(np.mean(zcr_full[start_frame:end_frame]))

    if rms_full is not None:
        rms_seg = rms_full[start_frame:end_frame]
        features["rms"] = float(np.mean(rms_seg))
        features["spectral_flatness"] = float(np.exp(np.mean(np.log(rms_seg + 1e-6))))  # Proxy

    # Additional spectral features
    stft = librosa.stft(audio, hop_length=hop_length)
    mag = np.abs(stft)

    # Spectral contrast
    spec_contrast = librosa.feature.spectral_contrast(S=mag, sr=sr)
    features["spectral_contrast"] = float(np.mean(spec_contrast))

    # Spectral flux
    spec_flux = np.diff(mag, axis=1)
    features["spectral_flux"] = float(np.mean(np.sqrt(np.mean(spec_flux**2, axis=0))))

    # Harmonic-to-noise ratio (simplified)
    harmonic = (
        librosa.yin(audio, sr=sr, fmin=10000, fmax=100000, frame_length=4096)
        if sr >= 200000
        else librosa.yin(audio, sr=sr, fmin=200, fmax=16000)
    )
    features["harmonic_to_noise_ratio"] = (
        float(np.mean(harmonic[~np.isnan(harmonic)])) if len(harmonic) > 0 else 0.0
    )

    # Fill missing features with defaults
    defaults = {
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
        "mean_f0_hz": 0.0,
        "f0_range_hz": 0.0,
        "spectral_centroid": 0.0,
        "spectral_bandwidth": 0.0,
        "spectral_rolloff": 0.0,
        "spectral_flatness": 0.0,
        "spectral_contrast": 0.0,
        "spectral_flux": 0.0,
        "zcr": 0.0,
        "rms": 0.0,
        "harmonic_to_noise_ratio": 0.0,
        "harmonicity": 0.0,
        "attack_time_ms": 0.0,
        "decay_time_ms": 0.0,
        "sustain_level": 0.0,
        "vibrato_rate_hz": 0.0,
        "vibrato_depth": 0.0,
        "jitter": 0.0,
        "shimmer": 0.0,
        "median_ici_ms": 0.0,
        "onset_rate_hz": 0.0,
        "ici_coefficient_of_variation": 0.0,
    }

    for key, val in defaults.items():
        if key not in features:
            features[key] = val

    # Motion factors (simplified)
    features["attack_time_ms"] = 5.0
    features["decay_time_ms"] = 20.0
    features["sustain_level"] = 0.7
    features["vibrato_rate_hz"] = 7.0
    features["vibrato_depth"] = 0.02
    features["jitter"] = 0.01
    features["shimmer"] = 0.015
    features["harmonicity"] = 0.75

    # Rhythm factors
    features["median_ici_ms"] = 15.0
    features["onset_rate_hz"] = 50.0
    features["ici_coefficient_of_variation"] = 0.3

    return features


def extract_phrase_candidates_fast(
    audio: np.ndarray,
    sr: int,
    sentence_id: str,
    context: int,
    mfcc_full: np.ndarray,
    spec_centroid_full: np.ndarray,
    spec_bandwidth_full: np.ndarray,
    spec_rolloff_full: np.ndarray,
    zcr_full: np.ndarray,
    rms_full: np.ndarray,
) -> List[PhraseCandidate]:
    """
    Fast phrase extraction using pre-computed features.

    Optimizations:
    - Only 3 window sizes instead of 7
    - Uses pre-computed features
    - 75% overlap instead of 50%
    """
    window_sizes_sec = [0.1, 0.2, 0.4]  # 3 sizes instead of 7
    candidates = []
    window_id = 0

    for window_sec in window_sizes_sec:
        window_size = int(window_sec * sr)
        hop_size = window_size // 4  # 75% overlap (was 50%)

        for start in range(0, len(audio) - window_size + 1, hop_size):
            end = start + window_size
            segment = audio[start:end]

            # Skip quiet segments
            rms = np.sqrt(np.mean(segment**2))
            if rms < 0.001:
                continue

            # Extract features using pre-computed arrays
            features = extract_features_fast(
                segment,
                sr,
                mfcc_full=mfcc_full,
                spec_centroid_full=spec_centroid_full,
                spec_bandwidth_full=spec_bandwidth_full,
                spec_rolloff_full=spec_rolloff_full,
                zcr_full=zcr_full,
                rms_full=rms_full,
                start_sample=start,
            )

            candidate = PhraseCandidate(
                start_sample=start,
                end_sample=end,
                features_29d=features,
                source_sentence_id=sentence_id,
                window_id=window_id,
                context=context,
            )

            candidates.append(candidate)
            window_id += 1

    return candidates


# =============================================================================
# Single File Processing (Optimized)
# =============================================================================


def process_single_vocalization_fast(args: Tuple[str, Dict[str, Any], Path]) -> Dict[str, Any]:
    """
    Process a single vocalization with optimizations.

    Optimizations:
    - Skip PELT (each file is already a sentence)
    - Pre-compute features once, then segment
    - Use soundfile for faster loading
    """
    audio_filename, annotation, audio_dir = args

    try:
        # Load audio faster with soundfile
        audio_path = audio_dir / audio_filename
        audio, sr = sf.read(str(audio_path))

        # Convert to mono if needed
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Create sentence ID
        sentence_id = audio_filename.replace(".wav", "")

        # Pre-compute features for entire audio (KEY OPTIMIZATION)
        hop_length = 512

        # MFCCs
        mfcc_full = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=hop_length)

        # Spectral features
        spec_centroid_full = librosa.feature.spectral_centroid(
            y=audio, sr=sr, hop_length=hop_length
        )[0]
        spec_bandwidth_full = librosa.feature.spectral_bandwidth(
            y=audio, sr=sr, hop_length=hop_length
        )[0]
        spec_rolloff_full = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=hop_length)[
            0
        ]
        zcr_full = librosa.feature.zero_crossing_rate(audio, hop_length=hop_length)[0]
        rms_full = librosa.feature.rms(y=audio, hop_length=hop_length)[0]

        # Create sentence (NO PELT - each file is already a sentence)
        sentence = Sentence(
            sentence_id=sentence_id,
            audio_file=audio_filename,
            context=annotation["context"],
            emitter=annotation["emitter"],
            addressee=annotation["addressee"],
            duration_sec=len(audio) / sr,
            change_points=[0, len(audio)],  # Simple: entire file is one sentence
        )

        # Extract phrase candidates using pre-computed features
        candidates = extract_phrase_candidates_fast(
            audio,
            sr,
            sentence_id=sentence_id,
            context=annotation["context"],
            mfcc_full=mfcc_full,
            spec_centroid_full=spec_centroid_full,
            spec_bandwidth_full=spec_bandwidth_full,
            spec_rolloff_full=spec_rolloff_full,
            zcr_full=zcr_full,
            rms_full=rms_full,
        )

        return {
            "success": True,
            "sentence": sentence,
            "candidates": candidates,
            "num_candidates": len(candidates),
            "audio_filename": audio_filename,
        }

    except Exception as e:
        import traceback

        return {
            "success": False,
            "error": str(e),
            "audio_filename": audio_filename,
            "traceback": traceback.format_exc(),
        }


# =============================================================================
# Checkpoint Manager (from previous version)
# =============================================================================


class CheckpointManager:
    """Manages incremental checkpointing."""

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"
        self.progress_file = self.checkpoint_dir / "checkpoint_progress.json"
        self.results_dir = self.checkpoint_dir / "partial_results"
        self.results_dir.mkdir(exist_ok=True)
        self.metadata = self._load_metadata()
        self.progress = self._load_progress()

    def _load_metadata(self) -> Dict[str, Any]:
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        return {
            "created_at": datetime.now().isoformat(),
            "last_updated": None,
            "total_files": 0,
            "processed_files": 0,
            "parameters": {},
        }

    def _load_progress(self) -> Dict[str, Any]:
        if self.progress_file.exists():
            with open(self.progress_file, "r") as f:
                return json.load(f)
        return {
            "processed_files": [],
            "failed_files": [],
            "current_batch": 0,
            "total_candidates": 0,
        }

    def save_metadata(self, total_files: int, parameters: Dict[str, Any]):
        self.metadata["last_updated"] = datetime.now().isoformat()
        self.metadata["total_files"] = total_files
        self.metadata["processed_files"] = len(self.progress["processed_files"])
        self.metadata["parameters"] = parameters
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def save_progress(
        self,
        processed_files: List[str],
        failed_files: List[str],
        total_candidates: int,
        batch_id: int,
    ):
        self.progress["processed_files"] = processed_files
        self.progress["failed_files"] = failed_files
        self.progress["total_candidates"] = total_candidates
        self.progress["current_batch"] = batch_id
        with open(self.progress_file, "w") as f:
            json.dump(self.progress, f, indent=2)

    def save_batch_results(
        self, batch_id: int, sentences: List[Sentence], candidates_data: List[Dict[str, Any]]
    ):
        sentences_file = self.results_dir / f"batch_{batch_id:04d}_sentences.pkl"
        with open(sentences_file, "wb") as f:
            pickle.dump(sentences, f)
        candidates_file = self.results_dir / f"batch_{batch_id:04d}_candidates.pkl"
        with open(candidates_file, "wb") as f:
            pickle.dump(candidates_data, f)

    def load_batch_results(self, batch_id: int) -> Tuple[List[Sentence], List[Dict[str, Any]]]:
        sentences_file = self.results_dir / f"batch_{batch_id:04d}_sentences.pkl"
        candidates_file = self.results_dir / f"batch_{batch_id:04d}_candidates.pkl"
        sentences = []
        candidates_data = []
        if sentences_file.exists():
            with open(sentences_file, "rb") as f:
                sentences = pickle.load(f)
        if candidates_file.exists():
            with open(candidates_file, "rb") as f:
                candidates_data = pickle.load(f)
        return sentences, candidates_data

    def get_processed_files(self) -> set:
        return set(self.progress["processed_files"])

    def load_all_results(self) -> Tuple[List[Sentence], List[Dict[str, Any]]]:
        all_sentences = []
        all_candidates = []
        for batch_file in sorted(self.results_dir.glob("batch_*_sentences.pkl")):
            batch_id = int(batch_file.stem.split("_")[1])
            sentences, candidates = self.load_batch_results(batch_id)
            all_sentences.extend(sentences)
            all_candidates.extend(candidates)
        return all_sentences, all_candidates


# =============================================================================
# Main Optimized Pipeline
# =============================================================================


def extract_optimized(
    audio_dir: Path,
    annotations_file: Path,
    output_dir: Path,
    num_workers: int = 4,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
    max_files: Optional[int] = None,
    batch_size: int = 100,
    resume: bool = True,
) -> ExtractionResult:
    """
    Optimized parallel extraction pipeline.

    Key optimizations:
    1. Skip PELT (each file is already a sentence)
    2. Pre-compute features once, then segment
    3. Only 3 window sizes (vs 7)
    4. Faster audio loading (soundfile)

    Expected speedup: 5-10x faster
    """
    start_time = time.time()

    print("=" * 80)
    print("OPTIMIZED PARALLEL EXTRACTION PIPELINE")
    print("=" * 80)
    print("Optimizations:")
    print("  - Skip PELT (each file is already a sentence)")
    print("  - Pre-compute MFCCs once, then segment")
    print("  - 3 window sizes (vs 7)")
    print("  - soundfile loading (vs librosa)")
    print(f"Audio directory: {audio_dir}")
    print(f"Workers: {num_workers}")
    print(f"Batch size: {batch_size}")
    print("=" * 80)

    # Create checkpoint directory
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_manager = CheckpointManager(checkpoint_dir)

    # ========================================================================
    # Step 1: Load annotations
    # ========================================================================
    print("\n[1/5] Loading annotations...")
    df = pd.read_csv(annotations_file)
    annotations = []
    for _, row in df.iterrows():
        annotations.append(
            {
                "filename": str(row["File Name"]),
                "context": int(row["Context"]),
                "emitter": int(row["Emitter"]),
                "addressee": int(row["Addressee"]),
            }
        )
    print(f"  Loaded {len(annotations)} vocalizations")

    if max_files:
        annotations = annotations[:max_files]
        print(f"  Limited to {max_files} files")

    processed_files = checkpoint_manager.get_processed_files() if resume else set()
    if resume and processed_files:
        remaining_annotations = [a for a in annotations if a["filename"] not in processed_files]
        total_files = len(annotations)
        processed_count = len(processed_files)
        remaining_count = len(remaining_annotations)
        print(
            f"  Resuming: {total_files} total, {processed_count} processed, "
            f"{remaining_count} remaining"
        )
        annotations = remaining_annotations

    checkpoint_manager.save_metadata(
        total_files=max_files if max_files else len(annotations),
        parameters={
            "dbscan_eps": dbscan_eps,
            "dbscan_min_samples": dbscan_min_samples,
            "num_workers": num_workers,
            "optimized": True,
        },
    )

    # ========================================================================
    # Step 2: Process audio files in batches
    # ========================================================================
    print(f"\n[2/5] Processing {len(annotations)} audio files (optimized)...")

    all_sentences = []
    all_candidates = []
    failed_files = []
    num_batches = (len(annotations) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(annotations))
        batch_annotations = annotations[batch_start:batch_end]

        print(f"\n  Batch {batch_idx + 1}/{num_batches} (files {batch_start + 1}-{batch_end})")

        process_args = [(ann["filename"], ann, audio_dir) for ann in batch_annotations]

        batch_sentences = []
        batch_candidates = []

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(process_single_vocalization_fast, args): args[0]
                for args in process_args
            }

            for future in tqdm(
                as_completed(futures), total=len(futures), desc=f"  Batch {batch_idx + 1}"
            ):
                audio_filename = futures[future]
                try:
                    result = future.result()
                    if result["success"]:
                        batch_sentences.append(result["sentence"])
                        batch_candidates.extend(result["candidates"])
                        all_sentences.append(result["sentence"])
                        all_candidates.extend(result["candidates"])
                    else:
                        failed_files.append(audio_filename)
                        if "traceback" in result:
                            print(f"\n  ERROR: {audio_filename}\n{result['traceback']}")
                        else:
                            print(f"\n  ERROR: {audio_filename} - {result.get('error', 'Unknown')}")
                except Exception as e:
                    failed_files.append(audio_filename)
                    print(f"\n  EXCEPTION: {audio_filename} - {e}")

        # Save batch checkpoint
        # Convert candidates to lightweight format
        candidates_lightweight = []
        for c in batch_candidates:
            candidates_lightweight.append(
                {
                    "start_sample": c.start_sample,
                    "end_sample": c.end_sample,
                    "features_29d": c.features_29d,
                    "source_sentence_id": c.source_sentence_id,
                    "window_id": c.window_id,
                    "context": c.context,
                }
            )

        checkpoint_manager.save_batch_results(batch_idx, batch_sentences, candidates_lightweight)

        processed_filenames = [s.audio_file for s in all_sentences]
        checkpoint_manager.save_progress(
            processed_files=processed_filenames,
            failed_files=failed_files,
            total_candidates=len(all_candidates),
            batch_id=batch_idx,
        )

        sentences_count = len(batch_sentences)
        candidates_count = len(batch_candidates)
        print(f"  Batch complete: {sentences_count} sentences, {candidates_count} candidates")
        print(
            f"  Progress: {len(processed_filenames)}/{checkpoint_manager.metadata['total_files']} "
            f"({len(processed_filenames) / checkpoint_manager.metadata['total_files'] * 100:.1f}%)"
        )

    # ========================================================================
    # Step 3: Load all results from checkpoints
    # ========================================================================
    print("\n[3/5] Loading all results from checkpoints...")
    all_sentences, all_candidates = checkpoint_manager.load_all_results()
    print(f"  Loaded {len(all_sentences)} sentences")
    print(f"  Loaded {len(all_candidates):,} phrase candidates")

    # ========================================================================
    # Step 4: Cluster phrases
    # ========================================================================
    print(f"\n[4/5] Clustering {len(all_candidates):,} candidates (DBSCAN)...")

    # Convert back to objects
    def candidate_dict_to_obj(c_dict):
        return PhraseCandidate(
            start_sample=c_dict["start_sample"],
            end_sample=c_dict["end_sample"],
            features_29d=c_dict["feature_29d"]
            if "feature_29d" in c_dict
            else c_dict["features_29d"],
            source_sentence_id=c_dict["source_sentence_id"],
            window_id=c_dict["window_id"],
            context=c_dict["context"],
        )

    candidate_objs = [candidate_dict_to_obj(c) for c in all_candidates]
    phrases = cluster_phrases_dbscan(candidate_objs, eps=dbscan_eps, min_samples=dbscan_min_samples)
    atomic_phrases = [p for p in phrases if p.is_atomic]
    print(f"  Found {len(phrases)} clusters ({len(atomic_phrases)} atomic)")

    # ========================================================================
    # Step 5: Export results
    # ========================================================================
    print("\n[5/5] Exporting results...")

    # Assign phrases to sentences
    candidate_lookup = {}
    for i, c_dict in enumerate(all_candidates):
        key = (c_dict["source_sentence_id"], c_dict["window_id"])
        candidate_lookup[key] = c_dict

    for sentence in all_sentences:
        assigned = []
        for c_dict in all_candidates:
            if c_dict["source_sentence_id"] == sentence.sentence_id:
                for phrase in phrases:
                    for member in phrase.member_candidates:
                        if (
                            member["source_sentence_id"] == c_dict["source_sentence_id"]
                            and member["window_id"] == c_dict["window_id"]
                        ):
                            assigned.append(phrase.phrase_id)
                            break
        sentence.phrases = list(set(assigned))

    grammar_rules = extract_grammar_rules(all_sentences)
    compositionality = detect_compositionality(all_sentences, phrases)

    # Helper to convert numpy types for JSON serialization
    def convert_to_json_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        return obj

    # Save final results
    sentences_file = output_dir / "sentences_final.json"
    with open(sentences_file, "w") as f:
        sentences_serializable = [convert_to_json_serializable(asdict(s)) for s in all_sentences]
        json.dump(sentences_serializable, f, indent=2)

    phrases_file = output_dir / "phrases_final.json"
    with open(phrases_file, "w") as f:
        phrases_serializable = [convert_to_json_serializable(asdict(p)) for p in phrases]
        json.dump(phrases_serializable, f, indent=2)

    rules_file = output_dir / "grammar_rules_final.json"
    with open(rules_file, "w") as f:
        rules_serializable = [convert_to_json_serializable(asdict(r)) for r in grammar_rules]
        json.dump(rules_serializable, f, indent=2)

    metadata = {
        "total_vocalizations": checkpoint_manager.metadata.get("total_files", len(annotations)),
        "successfully_processed": len(all_sentences),
        "failed_files": len(failed_files),
        "total_candidates": len(all_candidates),
        "total_phrases": len(phrases),
        "atomic_phrases": len(atomic_phrases),
        "grammar_rules": len(grammar_rules),
        "compositionality_ratio": compositionality["compositionality_ratio"],
        "processing_time_sec": time.time() - start_time,
        "parameters": checkpoint_manager.metadata.get("parameters", {}),
        "optimized": True,
    }

    with open(output_dir / "metadata_final.json", "w") as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Processing time: {elapsed:.1f}s ({elapsed / 60:.1f}min)")
    print(f"Throughput: {len(all_sentences) / elapsed:.1f} files/sec")
    print("\nResults:")
    print(f"  Sentences: {len(all_sentences)}")
    print(f"  Candidates: {len(all_candidates):,}")
    print(f"  Atomic phrases: {len(atomic_phrases)}")
    print(f"  Grammar rules: {len(grammar_rules)}")
    print(f"  Compositionality: {compositionality['compositionality_ratio']:.3f}")
    print("=" * 80)

    return ExtractionResult(
        sentences=all_sentences,
        phrases=phrases,
        grammar_rules=grammar_rules,
        total_candidates=len(all_candidates),
        total_atomic_phrases=len(atomic_phrases),
        processing_time_sec=elapsed,
        metadata=metadata,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optimized Parallel Extraction")
    parser.add_argument(
        "--audio-dir", type=str, default="/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/audio"
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default="/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/annotations.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/extraction_results_optimized",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--no-resume", action="store_true")

    args = parser.parse_args()

    result = extract_optimized(
        audio_dir=Path(args.audio_dir),
        annotations_file=Path(args.annotations),
        output_dir=Path(args.output_dir),
        num_workers=args.workers,
        dbscan_eps=args.eps,
        dbscan_min_samples=args.min_samples,
        max_files=args.max_files,
        batch_size=args.batch_size,
        resume=not args.no_resume,
    )
