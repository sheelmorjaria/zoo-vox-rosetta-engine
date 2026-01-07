#!/usr/bin/env python3
"""
Update vocalization_database.json with Micro-Dynamics Features

This script updates the vocalization database to include the NEW micro-dynamics
features for atomic phrase discovery:

1. Grit Factors: Harmonic-to-Noise Ratio (HNR), Spectral Flatness
2. Motion Factors: Attack Time, Decay Time, Sustain Level, Vibrato, Jitter
3. Fingerprint Factors: MFCCs (1-4), MFCC Delta, Spectral Contrast
4. Rhythm Factors: Inter-Click Interval, Onset Rate, ICI CV

Features: 16 new acoustic dimensions for discovering smallest units of meaning
"""

import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# Add URS path
urs_path = str(Path(__file__).parent.parent / "analysis" / "rosetta_stone")
sys.path.insert(0, urs_path)

from universal_rosetta_stone import Modality, PhraseSignature

# All micro-dynamics features
MICRO_DYNAMICS_FEATURES = [
    # Grit factors
    "harmonic_to_noise_ratio",
    "spectral_flatness",
    # Motion factors
    "attack_time_ms",
    "decay_time_ms",
    "sustain_level",
    "vibrato_rate_hz",
    "vibrato_depth",
    "jitter",
    # Fingerprint factors
    "mfcc_1",
    "mfcc_2",
    "mfcc_3",
    "mfcc_4",
    "mfcc_delta_mean",
    "spectral_contrast",
    # Rhythm factors
    "median_ici_ms",
    "onset_rate_hz",
    "ici_coefficient_of_variation",
]

print("=" * 80)
print("UPDATING VOCALIZATION_DATABASE.JSON WITH MICRO-DYNAMICS FEATURES")
print("=" * 80)

# Load phrase segments
pickle_path = "/home/sheel/birdsong_analysis/phrase_audio_database_full/phrase_segments.pkl"
print(f"\nLoading phrase segments from {pickle_path}...")

if not Path(pickle_path).exists():
    print(f"❌ Phrase segments file not found: {pickle_path}")
    print("   Micro-dynamics features will be set to 0 for all phrases")
    phrase_segments = {}
else:
    with open(pickle_path, "rb") as f:
        phrase_segments = pickle.load(f)

    print(f"✅ Loaded {len(phrase_segments)} phrase types")
    print(f"Total audio segments: {sum(len(segs) for segs in phrase_segments.values()):,}")

# Load existing database
db_path = "/home/sheel/birdsong_analysis/src/vocalization_database.json"
print(f"\nLoading existing database from {db_path}...")

with open(db_path, "r") as f:
    db = json.load(f)

print("✅ Loaded database")

# Extract micro-dynamics features from phrase segments
print("\nExtracting micro-dynamics features...")

micro_dynamics_map = {}  # phrase_key -> features

if phrase_segments:
    for i, (phrase_key, segments) in enumerate(phrase_segments.items()):
        if not segments or len(segments) == 0:
            continue

        # Get first segment
        audio = segments[0]

        if len(audio) == 0:
            continue

        # Sample rate for marmoset data
        sample_rate = 22050

        try:
            # Extract micro-dynamics using PhraseSignature (includes _extract_common_features)
            sig = PhraseSignature(modality=Modality.HARMONIC, data=audio, sample_rate=sample_rate)

            # Extract micro-dynamics features from .features dict
            micro_dynamics = {f: sig.features.get(f, 0.0) for f in MICRO_DYNAMICS_FEATURES}
            micro_dynamics_map[phrase_key] = micro_dynamics

        except Exception as e:
            print(f"  Error processing {phrase_key}: {e}")
            continue

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(phrase_segments)} phrase types...")

    print(f"\n✅ Extracted micro-dynamics for {len(micro_dynamics_map)} phrases")
else:
    print("⚠️  No phrase segments available - all features will be set to 0")


# Function to find matching phrase key (flexible matching)
def find_matching_phrase_key(target_key, available_keys):
    """Find matching phrase key, ignoring duration differences."""
    # Extract F0 and RANGE from target
    match = re.match(r"F0_(\d+)_DUR_\d+_RANGE_(\d+)", target_key)
    if not match:
        return None

    f0 = match.group(1)
    range_val = match.group(2)

    # Look for keys with same F0 and RANGE
    pattern = re.compile(f"F0_{f0}_DUR_\\d+_RANGE_{range_val}")
    matches = [k for k in available_keys if pattern.match(k)]

    return matches[0] if matches else None


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


# Update database with micro-dynamics features
print("\nUpdating database...")

marmoset_phrases = db["species_data"]["marmoset"]["phrases"]
updated_count = 0
not_found_count = 0

for phrase_key, phrase_data in marmoset_phrases.items():
    # Initialize acoustic_features dict if needed
    if "acoustic_features" not in phrase_data:
        phrase_data["acoustic_features"] = {}

    af = phrase_data["acoustic_features"]

    # Try direct match first
    if phrase_key in micro_dynamics_map:
        micro_dynamics = micro_dynamics_map[phrase_key]
    else:
        # Try flexible matching
        match = find_matching_phrase_key(phrase_key, micro_dynamics_map.keys())
        if match:
            micro_dynamics = micro_dynamics_map[match]
        else:
            # No micro-dynamics data available
            micro_dynamics = {f: 0.0 for f in MICRO_DYNAMICS_FEATURES}
            not_found_count += 1

    # Update acoustic features with micro-dynamics
    for feature, value in micro_dynamics.items():
        # Convert numpy types to Python floats for JSON serialization
        af[feature] = float(value)

    updated_count += 1

    if (updated_count % 200) == 0:
        print(f"  Updated {updated_count}/{len(marmoset_phrases)} phrases...")

print(f"\n✅ Updated {updated_count} phrases")
print(f"⚠️  {not_found_count} phrases had no micro-dynamics data (set to 0)")

# Save updated database
output_path = "/home/sheel/birdsong_analysis/src/vocalization_database_with_micro_dynamics.json"
print(f"\nSaving to {output_path}...")

# Convert all numpy types to Python native types
db = convert_numpy_types(db)

with open(output_path, "w") as f:
    json.dump(db, f, indent=2)

print("✅ Saved!")

# Sample micro-dynamics values
print("\n📊 SAMPLE MICRO-DYNAMICS VALUES:")

sample_count = 0
for phrase_key, phrase_data in marmoset_phrases.items():
    af = phrase_data["acoustic_features"]
    if af.get("harmonic_to_noise_ratio", 0) > 0:  # Only show non-zero features
        print(f"  {phrase_key}:")
        print(f"    HNR: {af.get('harmonic_to_noise_ratio', 0):.2f}")
        print(f"    Attack: {af.get('attack_time_ms', 0):.1f}ms")
        print(f"    Decay: {af.get('decay_time_ms', 0):.1f}ms")
        print(f"    Vibrato Rate: {af.get('vibrato_rate_hz', 0):.2f} Hz")
        print(f"    MFCC-1: {af.get('mfcc_1', 0):.2f}")
        print(f"    Spectral Contrast: {af.get('spectral_contrast', 0):.2f}")
        sample_count += 1
        if sample_count >= 5:
            break

print("\n" + "=" * 80)
print("✅ DATABASE UPDATE COMPLETE!")
print("=" * 80)
print("\n📊 SUMMARY:")
print(f"  Total phrases: {updated_count}")
print(f"  Phrases with micro-dynamics: {updated_count - not_found_count}")
print(f"  Phrases without micro-dynamics: {not_found_count}")
print(f"\n🎯 Features added: {len(MICRO_DYNAMICS_FEATURES)} new dimensions")
print("  Grit Factors: HNR, Spectral Flatness")
print("  Motion Factors: Attack/Decay, Sustain, Vibrato, Jitter")
print("  Fingerprint Factors: MFCCs (1-4), Delta, Spectral Contrast")
print("  Rhythm Factors: ICI, Onset Rate, ICI CV")
print("\n🎯 Next steps:")
print("  1. Backup old database:")
print("     mv /home/sheel/birdsong_analysis/src/vocalization_database.json \\")
print("        /home/sheel/birdsong_analysis/src/vocalization_database_before_micro_dynamics.json")
print("  2. Replace with new database:")
print(f"     mv {output_path} /home/sheel/birdsong_analysis/src/vocalization_database.json")
print("  3. Test atomic phrase discovery:")
print(
    "     python analysis/rosetta_stone/acoustic_similarity_for_atomic_phrase_candidates.py --demo"
)
print("=" * 80)
