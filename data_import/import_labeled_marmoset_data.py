#!/usr/bin/env python3
"""
Import Marmoset Vocalizations with Behavioral Context Annotations

This script imports labeled marmoset vocalizations and creates a phrase database
with proper behavioral context associations.

Data Sources:
- Annotations.tsv: 871,045 labeled vocalizations (Tsik, Trill, Twitter, Phee, Seep, Infant, Vocalization)
- ~/birdsong_analysis/data/Vocalizations/: Audio files organized by date folders

Process:
1. Load annotations with behavioral context labels
2. Load audio files and extract micro-dynamics features (URS)
3. Group similar audio into phrase types
4. Associate phrases with behavioral contexts
5. Export to vocalization_database.json

Behavioral Contexts (Call Types):
- Tsik: Short alarm/alert calls
- Trill: Social/courtship calls with frequency modulation
- Twitter: Excitement/social calls
- Phee: Long-distance contact calls
- Seep: Quiet contact calls
- Infant_cry: Infant vocalizations
- Vocalization: General/unclassified
"""

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

# Add URS path
urs_path = str(Path(__file__).parent.parent / "analysis" / "rosetta_stone")
sys.path.insert(0, urs_path)

from universal_rosetta_stone import Modality, PhraseSignature

# Configuration
ANNOTATIONS_PATH = "/home/sheel/birdsong_analysis/Annotations.tsv"
VOCALIZATIONS_DIR = "/home/sheel/birdsong_analysis/data/Vocalizations"
OUTPUT_PATH = "/home/sheel/birdsong_analysis/src/vocalization_database_with_contexts.json"
SAMPLE_RATE = 22050

# Context label mapping (normalize labels)
CONTEXT_MAPPING = {
    "Tsik": "tsik",
    "Trill": "trill",
    "Twitter": "twitter",
    "Phee": "phee",
    "Seep": "seep",
    "Infant": "infant",
    "Infant_cry": "infant",
    "Vocalization": "vocalization",
}


def load_annotations(annotations_path: str, max_per_label: int = None) -> pd.DataFrame:
    """Load annotations and filter by label counts."""
    print(f"Loading annotations from {annotations_path}...")

    df = pd.read_csv(annotations_path, sep="\t")

    print(f"✅ Loaded {len(df)} annotations")

    # Show label distribution
    print("\n📊 Label Distribution:")
    label_counts = df["label"].value_counts()
    for label, count in label_counts.items():
        print(f"   {label:<20} {count:>8}")

    # Optionally sample to limit processing time
    if max_per_label:
        print(f"\n⚠️  Sampling to {max_per_label} per label...")
        sampled_df = (
            df.groupby("label")
            .apply(lambda x: x.sample(n=min(max_per_label, len(x)), random_state=42))
            .reset_index(drop=True)
        )
        df = sampled_df
        print(f"✅ Sampled to {len(df)} annotations")

    return df


def load_audio_file(file_path: str) -> Tuple[np.ndarray, int]:
    """Load audio file and return audio data + sample rate."""
    try:
        audio, sr = sf.read(file_path)

        # Convert to mono if needed
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Resample if needed
        if sr != SAMPLE_RATE:
            # Simple resampling (for production, use librosa or scipy)
            from scipy import signal

            num_samples = int(len(audio) * SAMPLE_RATE / sr)
            audio = signal.resample(audio, num_samples)

        return audio, SAMPLE_RATE

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None


def extract_features_from_audio(audio: np.ndarray) -> Dict[str, float]:
    """Extract micro-dynamics features from audio using URS."""
    try:
        sig = PhraseSignature(modality=Modality.HARMONIC, data=audio, sample_rate=SAMPLE_RATE)
        return sig.features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return {}


def generate_phrase_key(features: Dict[str, float]) -> str:
    """Generate phrase key from acoustic features."""
    # Get F0 statistics
    mean_f0 = features.get("f0_mean", 0)
    f0_range = features.get("f0_range", 0)
    duration_ms = features.get("duration_ms", 0)

    # Round to meaningful precision
    f0_bucket = int(mean_f0 / 100) * 100  # Round to nearest 100 Hz
    dur_bucket = int(duration_ms / 5) * 5  # Round to nearest 5 ms
    range_bucket = int(f0_range / 100) * 100  # Round to nearest 100 Hz

    return f"F0_{f0_bucket}_DUR_{dur_bucket}_RANGE_{range_bucket}"


def import_labeled_vocalizations(
    annotations: pd.DataFrame, vocalizations_dir: str, max_files: int = None
) -> Dict:
    """
    Import labeled vocalizations and create phrase database with contexts.

    Returns:
        Dictionary with species data including phrases with context associations
    """
    print("\n" + "=" * 80)
    print("IMPORTING LABELED MARMOSET VOCALIZATIONS")
    print("=" * 80)

    # Group phrases by key and aggregate contexts
    phrase_library = defaultdict(
        lambda: {
            "audio_segments": [],
            "contexts": Counter(),
            "total_occurrences": 0,
            "features_list": [],
        }
    )

    processed = 0
    skipped = 0
    errors = 0

    # Process each annotation
    for idx, row in annotations.iterrows():
        if max_files and processed >= max_files:
            break

        parent_name = str(row["parent_name"]).replace(" ", "_")
        file_name = str(row["file_name"])
        label = str(row["label"])

        # Normalize label
        context_name = CONTEXT_MAPPING.get(label, label.lower())

        # Build file path
        file_path = Path(vocalizations_dir) / parent_name / file_name

        if not file_path.exists():
            skipped += 1
            continue

        # Load audio
        audio, sr = load_audio_file(str(file_path))

        if audio is None or len(audio) == 0:
            errors += 1
            continue

        # Extract features
        features = extract_features_from_audio(audio)

        if not features:
            errors += 1
            continue

        # Generate phrase key
        phrase_key = generate_phrase_key(features)

        # Store
        phrase_library[phrase_key]["audio_segments"].append(audio)
        phrase_library[phrase_key]["contexts"][context_name] += 1
        phrase_library[phrase_key]["total_occurrences"] += 1
        phrase_library[phrase_key]["features_list"].append(features)

        processed += 1

        if processed % 100 == 0:
            print(f"  Processed {processed} files...")

    print(f"\n✅ Processed {processed} audio files")
    print(f"   Skipped: {skipped} (not found)")
    print(f"   Errors: {errors}")

    # Create export structure
    species_data = {
        "species": "marmoset",
        "analysis_date": datetime.now().isoformat(),
        "total_phrases": len(phrase_library),
        "total_sentences": 0,
        "vocabulary_size": len(phrase_library),
        "modality_distribution": {"harmonic": len(phrase_library)},
        "phrases": {},
    }

    # Export phrases
    print("\n📊 Creating phrase database...")

    for phrase_key, phrase_data in phrase_library.items():
        # Use first occurrence's features as representative
        if phrase_data["features_list"]:
            features = phrase_data["features_list"][0]
        else:
            continue

        # Calculate modality based on features
        spectral_flatness = features.get("spectral_flatness", 0)
        if spectral_flatness > 0.5:
            modality = "transient"
        elif features.get("vibrato_rate_hz", 0) > 5:
            modality = "rhythmic"
        else:
            modality = "harmonic"

        # Create context list
        contexts = []
        total_context_count = sum(phrase_data["contexts"].values())

        for ctx_name, count in phrase_data["contexts"].most_common():
            contexts.append(
                {
                    "context_name": ctx_name,
                    "count": count,
                    "percentage": (count / total_context_count * 100)
                    if total_context_count > 0
                    else 0,
                }
            )

        # Build acoustic features dict
        acoustic_features = {
            "mean_f0_hz": features.get("f0_mean", 0),
            "std_f0_hz": features.get("f0_std", 0),
            "min_f0_hz": features.get("f0_mean", 0) - features.get("f0_range", 0) / 2,
            "max_f0_hz": features.get("f0_mean", 0) + features.get("f0_range", 0) / 2,
            "f0_range_hz": features.get("f0_range", 0),
            "duration_frames": int(features.get("duration_ms", 0) * SAMPLE_RATE / 1000),
            "voiced_ratio": features.get("harmonicity", 0),
            "f0_slope": 0,
            "modulation_rate": features.get("vibrato_rate_hz", 0),
            "acoustic_variance": features.get("envelope_cv", 0),
            "mean_duration_ms": features.get("duration_ms", 0),
            # Timbre features
            "spectral_centroid_hz": features.get("spectral_centroid_hz", 0),
            "spectral_slope": features.get("spectral_slope", 0),
            "spectral_bandwidth_hz": features.get("spectral_bandwidth_hz", 0),
            "spectral_rolloff_hz": features.get("spectral_rolloff_hz", 0),
            # Micro-dynamics features
            "harmonic_to_noise_ratio": features.get("harmonic_to_noise_ratio", 0),
            "attack_time_ms": features.get("attack_time_ms", 0),
            "decay_time_ms": features.get("decay_time_ms", 0),
            "sustain_level": features.get("sustain_level", 0),
            "vibrato_rate_hz": features.get("vibrato_rate_hz", 0),
            "vibrato_depth": features.get("vibrato_depth", 0),
            "jitter": features.get("jitter", 0),
            "mfcc_1": features.get("mfcc_1", 0),
            "mfcc_2": features.get("mfcc_2", 0),
            "mfcc_3": features.get("mfcc_3", 0),
            "mfcc_4": features.get("mfcc_4", 0),
            "mfcc_delta_mean": features.get("mfcc_delta_mean", 0),
            "spectral_contrast": features.get("spectral_contrast", 0),
            "median_ici_ms": features.get("median_ici_ms", 0),
            "onset_rate_hz": features.get("onset_rate_hz", 0),
            "ici_coefficient_of_variation": features.get("ici_coefficient_of_variation", 0),
        }

        species_data["phrases"][phrase_key] = {
            "phrase_key": phrase_key,
            "signature": f"{modality}_{phrase_key}",
            "species": "marmoset",
            "modality": modality,
            "acoustic_features": acoustic_features,
            "total_occurrences": phrase_data["total_occurrences"],
            "contexts": contexts,
            "social_contexts": {},
            "is_compositional": False,
            "phrase_components": [],
        }

    # Print statistics
    print("\n📊 DATABASE STATISTICS:")
    print(f"   Total phrases: {len(species_data['phrases'])}")
    print(
        f"   Total occurrences: {sum(p['total_occurrences'] for p in species_data['phrases'].values())}"
    )

    # Show context distribution
    all_contexts = Counter()
    for phrase in species_data["phrases"].values():
        for ctx in phrase["contexts"]:
            all_contexts[ctx["context_name"]] += ctx["count"]

    print("\n📊 CONTEXT DISTRIBUTION:")
    for ctx, count in all_contexts.most_common():
        percentage = (count / sum(all_contexts.values())) * 100
        print(f"   {ctx:<20} {count:>8} ({percentage:>5.1f}%)")

    return species_data


def main():
    """Main import function."""
    print("=" * 80)
    print("MARMOSET VOCALIZATION IMPORT WITH BEHAVIORAL CONTEXTS")
    print("=" * 80)

    # Load annotations
    annotations = load_annotations(ANNOTATIONS_PATH, max_per_label=1000)

    # Import labeled vocalizations
    species_data = import_labeled_vocalizations(
        annotations,
        VOCALIZATIONS_DIR,
        max_files=5000,  # Limit for demonstration
    )

    # Create export structure
    export_data = {
        "export_date": datetime.now().isoformat(),
        "species_data": {"marmoset": species_data},
    }

    # Save
    print(f"\n💾 Saving to {OUTPUT_PATH}...")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(export_data, f, indent=2)

    print("✅ Saved!")

    print("\n" + "=" * 80)
    print("✅ IMPORT COMPLETE!")
    print("=" * 80)

    print("\n🎯 Next steps:")
    print("   1. Run persona-context association analysis:")
    print("      python analysis/rosetta_stone/real_context_association.py \\")
    print(f"         --db {OUTPUT_PATH}")
    print("   2. Discover semantic meanings of acoustic personas")
    print("   3. Validate with statistical tests")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
