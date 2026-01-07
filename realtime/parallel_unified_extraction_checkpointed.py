"""
Parallel Unified Extraction Pipeline with Checkpointing

Processes the Egyptian fruit bat dataset in parallel with incremental checkpointing.
Supports resuming from interruptions and preliminary analysis.

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
import pickle
import time
from datetime import datetime
import hashlib

# Audio processing
import librosa

# Change point detection
import ruptures as rpt

# Clustering
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Import feature extraction and other functions from the base module
import sys
sys.path.insert(0, str(Path(__file__).parent))

from parallel_unified_extraction import (
    Sentence, PhraseCandidate, AtomicPhrase, GrammarRule, ExtractionResult,
    extract_29d_features, segment_sentences_pelt, extract_phrase_candidates,
    cluster_phrases_dbscan, detect_compositionality, extract_grammar_rules,
    _calculate_intra_cluster_similarity, _calculate_inter_cluster_similarity
)


# =============================================================================
# Checkpoint Manager
# =============================================================================

class CheckpointManager:
    """Manages incremental checkpointing for long-running pipelines."""

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint files
        self.metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"
        self.progress_file = self.checkpoint_dir / "checkpoint_progress.json"
        self.results_dir = self.checkpoint_dir / "partial_results"
        self.results_dir.mkdir(exist_ok=True)

        # Load existing checkpoint if available
        self.metadata = self._load_metadata()
        self.progress = self._load_progress()

    def _load_metadata(self) -> Dict[str, Any]:
        """Load checkpoint metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            'created_at': datetime.now().isoformat(),
            'last_updated': None,
            'total_files': 0,
            'processed_files': 0,
            'parameters': {}
        }

    def _load_progress(self) -> Dict[str, Any]:
        """Load processing progress."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            'processed_files': [],
            'failed_files': [],
            'current_batch': 0,
            'total_candidates': 0
        }

    def save_metadata(self, total_files: int, parameters: Dict[str, Any]):
        """Save checkpoint metadata."""
        self.metadata['last_updated'] = datetime.now().isoformat()
        self.metadata['total_files'] = total_files
        self.metadata['processed_files'] = len(self.progress['processed_files'])
        self.metadata['parameters'] = parameters

        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def save_progress(self, processed_files: List[str], failed_files: List[str],
                     total_candidates: int, batch_id: int):
        """Save processing progress."""
        self.progress['processed_files'] = processed_files
        self.progress['failed_files'] = failed_files
        self.progress['total_candidates'] = total_candidates
        self.progress['current_batch'] = batch_id

        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)

    def save_batch_results(self, batch_id: int, sentences: List[Sentence],
                          candidates_data: List[Dict[str, Any]]):
        """Save results for a single batch."""
        # Save sentences
        sentences_file = self.results_dir / f"batch_{batch_id:04d}_sentences.pkl"
        with open(sentences_file, 'wb') as f:
            pickle.dump(sentences, f)

        # Save candidates (lightweight version - no audio data)
        candidates_file = self.results_dir / f"batch_{batch_id:04d}_candidates.pkl"
        with open(candidates_file, 'wb') as f:
            pickle.dump(candidates_data, f)

    def load_batch_results(self, batch_id: int) -> Tuple[List[Sentence], List[Dict[str, Any]]]:
        """Load results for a single batch."""
        sentences_file = self.results_dir / f"batch_{batch_id:04d}_sentences.pkl"
        candidates_file = self.results_dir / f"batch_{batch_id:04d}_candidates.pkl"

        sentences = []
        candidates_data = []

        if sentences_file.exists():
            with open(sentences_file, 'rb') as f:
                sentences = pickle.load(f)

        if candidates_file.exists():
            with open(candidates_file, 'rb') as f:
                candidates_data = pickle.load(f)

        return sentences, candidates_data

    def get_processed_files(self) -> set:
        """Get set of already processed file names."""
        return set(self.progress['processed_files'])

    def get_status_summary(self) -> Dict[str, Any]:
        """Get checkpoint status summary."""
        return {
            'total_files': self.metadata.get('total_files', 0),
            'processed_files': len(self.progress['processed_files']),
            'failed_files': len(self.progress['failed_files']),
            'total_candidates': self.progress['total_candidates'],
            'progress_percent': (
                len(self.progress['processed_files']) / self.metadata.get('total_files', 1) * 100
                if self.metadata.get('total_files', 0) > 0 else 0
            ),
            'last_updated': self.metadata.get('last_updated'),
            'current_batch': self.progress['current_batch']
        }

    def load_all_results(self) -> Tuple[List[Sentence], List[Dict[str, Any]]]:
        """Load all batch results."""
        all_sentences = []
        all_candidates = []

        for batch_file in sorted(self.results_dir.glob("batch_*_sentences.pkl")):
            batch_id = int(batch_file.stem.split('_')[1])
            sentences, candidates = self.load_batch_results(batch_id)
            all_sentences.extend(sentences)
            all_candidates.extend(candidates)

        return all_sentences, all_candidates


# =============================================================================
# Batch Processing with Checkpointing
# =============================================================================

def process_single_vocalization_lightweight(args: Tuple[str, Dict[str, Any], Path, float]) -> Dict[str, Any]:
    """
    Process a single vocalization file (lightweight version for checkpointing).

    Returns lightweight results without audio data (to save checkpoint space).
    """
    audio_filename, annotation, audio_dir, penalty = args

    try:
        # Load audio
        audio_path = audio_dir / audio_filename
        audio, sr = librosa.load(str(audio_path), sr=None)

        # Create sentence ID
        sentence_id = audio_filename.replace('.wav', '')

        # Step 1: Segment into sentences
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

        # Lightweight candidate data (no audio segments)
        candidates_lightweight = []
        for c in candidates:
            candidates_lightweight.append({
                'start_sample': c.start_sample,
                'end_sample': c.end_sample,
                'features_29d': c.features_29d,
                'source_sentence_id': c.source_sentence_id,
                'window_id': c.window_id,
                'context': c.context
            })

        return {
            'success': True,
            'sentence': sentence,
            'candidates': candidates_lightweight,
            'num_candidates': len(candidates),
            'audio_filename': audio_filename
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'audio_filename': audio_filename
        }


# =============================================================================
# Main Pipeline with Checkpointing
# =============================================================================

def extract_with_checkpointing(
    audio_dir: Path,
    annotations_file: Path,
    output_dir: Path,
    num_workers: int = 16,
    pelt_penalty: float = 10.0,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
    max_files: Optional[int] = None,
    batch_size: int = 1000,
    checkpoint_interval: int = 100,
    resume: bool = True
) -> ExtractionResult:
    """
    Parallel unified extraction pipeline with incremental checkpointing.

    Pipeline with checkpointing:
    1. Load annotations
    2. Process audio files in batches with checkpointing
    3. Each batch is saved incrementally
    4. Can resume from interruptions
    5. Supports preliminary analysis on partial results

    Args:
        audio_dir: Directory containing audio files
        annotations_file: CSV file with annotations
        output_dir: Directory for final results
        num_workers: Number of parallel workers (default 16)
        pelt_penalty: PELT penalty parameter
        dbscan_eps: DBSCAN epsilon parameter
        dbscan_min_samples: DBSCAN min_samples parameter
        max_files: Optional limit on files to process
        batch_size: Files per batch for checkpointing
        checkpoint_interval: Checkpoint every N files
        resume: Resume from existing checkpoint

    Returns:
        ExtractionResult with all extracted data
    """
    import time
    start_time = time.time()

    print("=" * 80)
    print("PARALLEL UNIFIED EXTRACTION WITH CHECKPOINTING")
    print("=" * 80)
    print(f"Audio directory: {audio_dir}")
    print(f"Annotations file: {annotations_file}")
    print(f"Output directory: {output_dir}")
    print(f"Workers: {num_workers}")
    print(f"Batch size: {batch_size}")
    print(f"Resume: {resume}")
    print("=" * 80)

    # Create checkpoint directory
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_manager = CheckpointManager(checkpoint_dir)

    # ========================================================================
    # Step 1: Load annotations
    # ========================================================================
    print("\n[1/6] Loading annotations...")
    df = pd.read_csv(annotations_file)

    # Parse annotations
    annotations = []
    for _, row in df.iterrows():
        audio_filename = str(row['File Name'])
        annotations.append({
            'filename': audio_filename,
            'context': int(row['Context']),
            'emitter': int(row['Emitter']),
            'addressee': int(row['Addressee'])
        })

    print(f"  Loaded {len(annotations)} vocalizations")

    # Limit files if specified
    if max_files:
        annotations = annotations[:max_files]
        print(f"  Limited to {max_files} files")

    # Filter out already processed files if resuming
    processed_files = checkpoint_manager.get_processed_files() if resume else set()

    if resume and processed_files:
        remaining_annotations = [a for a in annotations if a['filename'] not in processed_files]
        print(f"  Resuming: {len(annotations)} total, {len(processed_files)} already processed, {len(remaining_annotations)} remaining")
        annotations = remaining_annotations

    # Save initial metadata
    checkpoint_manager.save_metadata(
        total_files=max_files if max_files else len(annotations),
        parameters={
            'pelt_penalty': pelt_penalty,
            'dbscan_eps': dbscan_eps,
            'dbscan_min_samples': dbscan_min_samples,
            'num_workers': num_workers
        }
    )

    # ========================================================================
    # Step 2: Process audio files in batches with checkpointing
    # ========================================================================
    print(f"\n[2/6] Processing {len(annotations)} audio files with checkpointing...")

    all_sentences = []
    all_candidates = []
    failed_files = []

    # Process in batches
    num_batches = (len(annotations) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(annotations))
        batch_annotations = annotations[batch_start:batch_end]

        print(f"\n  Batch {batch_idx + 1}/{num_batches} (files {batch_start + 1}-{batch_end})")

        # Prepare arguments
        process_args = [
            (ann['filename'], ann, audio_dir, pelt_penalty)
            for ann in batch_annotations
        ]

        batch_sentences = []
        batch_candidates = []

        # Process batch in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(process_single_vocalization_lightweight, args): args[0]
                for args in process_args
            }

            for future in tqdm(as_completed(futures), total=len(futures),
                              desc=f"  Batch {batch_idx + 1}"):
                audio_filename = futures[future]
                try:
                    result = future.result()
                    if result['success']:
                        batch_sentences.append(result['sentence'])
                        batch_candidates.extend(result['candidates'])
                        all_sentences.append(result['sentence'])
                        all_candidates.extend(result['candidates'])
                    else:
                        failed_files.append(audio_filename)
                        print(f"\n  ERROR: {audio_filename} - {result.get('error', 'Unknown error')}")
                except Exception as e:
                    failed_files.append(audio_filename)
                    print(f"\n  EXCEPTION: {audio_filename} - {e}")

        # Save batch checkpoint
        checkpoint_manager.save_batch_results(batch_idx, batch_sentences, batch_candidates)

        # Update progress checkpoint
        processed_filenames = [s.audio_file for s in all_sentences]
        checkpoint_manager.save_progress(
            processed_files=processed_filenames,
            failed_files=failed_files,
            total_candidates=len(all_candidates),
            batch_id=batch_idx
        )

        # Print status
        status = checkpoint_manager.get_status_summary()
        print(f"  Batch complete: {len(batch_sentences)} sentences, {len(batch_candidates)} candidates")
        print(f"  Progress: {status['processed_files']}/{status['total_files']} ({status['progress_percent']:.1f}%)")
        print(f"  Total candidates so far: {status['total_candidates']:,}")

        # Preliminary analysis checkpoint
        if (batch_idx + 1) % (num_batches // 5) == 0:  # Every 20% of batches
            print(f"\n  [PRELIMINARY ANALYSIS - {status['progress_percent']:.0f}% COMPLETE]")
            _run_preliminary_analysis(all_sentences, all_candidates, output_dir,
                                      dbscan_eps, dbscan_min_samples, batch_idx)

    # ========================================================================
    # Step 3: Load all results from checkpoints
    # ========================================================================
    print("\n[3/6] Loading all results from checkpoints...")

    # Reload everything from checkpoints for consistency
    all_sentences, all_candidates = checkpoint_manager.load_all_results()

    print(f"  Loaded {len(all_sentences)} sentences")
    print(f"  Loaded {len(all_candidates):,} phrase candidates")

    # ========================================================================
    # Step 4: Cluster all phrase candidates
    # ========================================================================
    print(f"\n[4/6] Clustering {len(all_candidates):,} phrase candidates (DBSCAN)...")
    print(f"  eps={dbscan_eps}, min_samples={dbscan_min_samples}")

    # Convert lightweight candidates back to PhraseCandidate objects
    def candidate_dict_to_obj(c_dict: Dict[str, Any]) -> PhraseCandidate:
        return PhraseCandidate(
            audio_segment=np.array([]),  # Empty - we don't have audio anymore
            start_sample=c_dict['start_sample'],
            end_sample=c_dict['end_sample'],
            features_29d=c_dict['features_29d'],
            source_sentence_id=c_dict['source_sentence_id'],
            window_id=c_dict['window_id'],
            context=c_dict['context']
        )

    candidate_objs = [candidate_dict_to_obj(c) for c in all_candidates]

    phrases = cluster_phrases_dbscan(
        candidate_objs,
        eps=dbscan_eps,
        min_samples=dbscan_min_samples
    )

    print(f"  Found {len(phrases)} phrase clusters")
    atomic_phrases = [p for p in phrases if p.is_atomic]
    print(f"  Atomic phrases: {len(atomic_phrases)}/{len(phrases)}")

    # ========================================================================
    # Step 5: Assign phrases to sentences and extract grammar
    # ========================================================================
    print("\n[5/6] Assigning phrases and extracting grammar...")

    # Build candidate lookup
    candidate_lookup = {}
    for i, c_dict in enumerate(all_candidates):
        key = (c_dict['source_sentence_id'], c_dict['window_id'])
        candidate_lookup[key] = c_dict

    # Assign phrases to sentences
    for sentence in all_sentences:
        assigned_phrases = []

        # Find candidates from this sentence
        for c_dict in all_candidates:
            if c_dict['source_sentence_id'] == sentence.sentence_id:
                # Find which cluster this candidate belongs to
                for phrase in phrases:
                    for member in phrase.member_candidates:
                        if (member['source_sentence_id'] == c_dict['source_sentence_id'] and
                            member['window_id'] == c_dict['window_id']):
                            assigned_phrases.append(phrase.phrase_id)
                            break

        sentence.phrases = list(set(assigned_phrases))

    print(f"  Assigned phrases to {len(all_sentences)} sentences")

    # Extract grammar rules
    grammar_rules = extract_grammar_rules(all_sentences)
    print(f"  Found {len(grammar_rules)} grammar rules")

    # Test compositionality
    compositionality = detect_compositionality(all_sentences, phrases)
    print(f"  Compositionality ratio: {compositionality['compositionality_ratio']:.3f}")

    # ========================================================================
    # Step 6: Export final results
    # ========================================================================
    print("\n[6/6] Exporting final results...")

    # Save final results
    sentences_file = output_dir / "sentences_final.json"
    with open(sentences_file, 'w') as f:
        sentences_data = []
        for s in all_sentences:
            sentence_dict = asdict(s)
            sentences_data.append(sentence_dict)
        json.dump(sentences_data, f, indent=2)
    print(f"  Saved {len(all_sentences)} sentences")

    phrases_file = output_dir / "phrases_final.json"
    with open(phrases_file, 'w') as f:
        phrases_data = []
        for p in phrases:
            phrase_dict = asdict(p)
            phrases_data.append(phrase_dict)
        json.dump(phrases_data, f, indent=2)
    print(f"  Saved {len(phrases)} phrases")

    rules_file = output_dir / "grammar_rules_final.json"
    with open(rules_file, 'w') as f:
        rules_data = [asdict(r) for r in grammar_rules]
        json.dump(rules_data, f, indent=2)
    print(f"  Saved {len(grammar_rules)} grammar rules")

    # Final metadata
    elapsed = time.time() - start_time
    metadata = {
        'total_vocalizations': checkpoint_manager.metadata.get('total_files', len(annotations)),
        'successfully_processed': len(all_sentences),
        'failed_files': len(failed_files),
        'total_candidates': len(all_candidates),
        'total_phrases': len(phrases),
        'atomic_phrases': len(atomic_phrases),
        'grammar_rules': len(grammar_rules),
        'compositionality_ratio': compositionality['compositionality_ratio'],
        'processing_time_sec': elapsed,
        'parameters': checkpoint_manager.metadata.get('parameters', {})
    }

    metadata_file = output_dir / "metadata_final.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata")

    # ========================================================================
    # Final Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Processing time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"Throughput: {len(all_sentences)/elapsed:.1f} vocalizations/second")
    print(f"\nResults:")
    print(f"  Sentences: {len(all_sentences)}")
    print(f"  Phrase candidates: {len(all_candidates):,}")
    print(f"  Atomic phrases: {len(atomic_phrases)}")
    print(f"  Grammar rules: {len(grammar_rules)}")
    print(f"  Compositionality: {compositionality['compositionality_ratio']:.3f}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print("=" * 80)

    return ExtractionResult(
        sentences=all_sentences,
        phrases=phrases,
        grammar_rules=grammar_rules,
        total_candidates=len(all_candidates),
        total_atomic_phrases=len(atomic_phrases),
        processing_time_sec=elapsed,
        metadata=metadata
    )


def _run_preliminary_analysis(sentences: List[Sentence], candidates: List[Dict[str, Any]],
                             output_dir: Path, eps: float, min_samples: int,
                             batch_id: int):
    """Run preliminary analysis on partial results."""
    print(f"  Running preliminary clustering on {len(candidates):,} candidates...")

    try:
        # Convert to objects
        from parallel_unified_extraction import PhraseCandidate

        candidate_objs = []
        for c_dict in candidates[:50000]:  # Limit for speed
            candidate_objs.append(PhraseCandidate(
                audio_segment=np.array([]),
                start_sample=c_dict['start_sample'],
                end_sample=c_dict['end_sample'],
                features_29d=c_dict['features_29d'],
                source_sentence_id=c_dict['source_sentence_id'],
                window_id=c_dict['window_id'],
                context=c_dict['context']
            ))

        # Quick clustering
        phrases = cluster_phrases_dbscan(candidate_objs, eps=eps, min_samples=min_samples)

        # Save preliminary results
        prelim_dir = output_dir / "preliminary"
        prelim_dir.mkdir(exist_ok=True)

        prelim_file = prelim_dir / f"analysis_batch_{batch_id:04d}.json"
        with open(prelim_file, 'w') as f:
            json.dump({
                'batch_id': batch_id,
                'num_sentences': len(sentences),
                'num_candidates': len(candidates),
                'num_phrases': len(phrases),
                'num_atomic': sum(1 for p in phrases if p.is_atomic),
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)

        print(f"  -> Found {len(phrases)} phrases ({sum(1 for p in phrases if p.is_atomic)} atomic)")

    except Exception as e:
        print(f"  -> Preliminary analysis failed: {e}")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Parallel Unified Extraction with Checkpointing"
    )
    parser.add_argument("--audio-dir", type=str, default="/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/audio")
    parser.add_argument("--annotations", type=str, default="/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/annotations.csv")
    parser.add_argument("--output-dir", type=str, default="/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/extraction_results_checkpointed")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--penalty", type=float, default=10.0)
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from checkpoint")

    args = parser.parse_args()

    result = extract_with_checkpointing(
        audio_dir=Path(args.audio_dir),
        annotations_file=Path(args.annotations),
        output_dir=Path(args.output_dir),
        num_workers=args.workers,
        pelt_penalty=args.penalty,
        dbscan_eps=args.eps,
        dbscan_min_samples=args.min_samples,
        max_files=args.max_files,
        batch_size=args.batch_size,
        resume=not args.no_resume
    )
