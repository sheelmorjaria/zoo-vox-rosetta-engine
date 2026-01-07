#!/usr/bin/env python3
"""
Update vocalization_database.json with Timbre Features (Improved Version)

This script:
1. Loads phrase_segments.pkl (984 phrase types with audio)
2. Extracts timbre features for each phrase
3. Updates vocalization_database.json with timbre features
4. Handles phrase key mismatches intelligently
"""

import json
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Add URS path
urs_path = str(Path(__file__).parent.parent / "analysis" / "rosetta_stone")
sys.path.insert(0, urs_path)

from universal_rosetta_stone import UniversalRosettaStone

TIMBRE_FEATURES = [
    "spectral_centroid_hz",
    "spectral_slope",
    "spectral_bandwidth_hz",
    "spectral_rolloff_hz",
]

print("=" * 80)
print("UPDATING VOCALIZATION_DATABASE.JSON WITH TIMBRE FEATURES")
print("=" * 80)

# Load phrase segments
pickle_path = "/home/sheel/birdsong_analysis/phrase_audio_database_full/phrase_segments.pkl"
print(f"\nLoading phrase segments from {pickle_path}...")

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

# Extract timbre features from phrase segments
print("\nExtracting timbre features...")

timbre_map = {}  # phrase_key -> timbre features

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
        # Extract timbre using URS
        analyzer = UniversalRosettaStone(sample_rate=sample_rate)
        features = analyzer._extract_modality_features(audio)

        # Extract timbre features
        timbre = {f: features.get(f, 0.0) for f in TIMBRE_FEATURES}
        timbre_map[phrase_key] = timbre

    except Exception as e:
        print(f"  Error processing {phrase_key}: {e}")
        continue

    if (i + 1) % 100 == 0:
        print(f"  Processed {i + 1}/{len(phrase_segments)} phrase types...")

print(f"\n✅ Extracted timbre for {len(timbre_map)} phrases")


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


# Update database with timbre features
print("\nUpdating database...")

marmoset_phrases = db["species_data"]["marmoset"]["phrases"]
updated_count = 0
not_found_count = 0

for phrase_key, phrase_data in marmoset_phrases.items():
    # Try direct match first
    if phrase_key in timbre_map:
        timbre = timbre_map[phrase_key]
    else:
        # Try flexible matching
        match = find_matching_phrase_key(phrase_key, timbre_map.keys())
        if match:
            timbre = timbre_map[match]
        else:
            # No timbre data available
            timbre = {f: 0.0 for f in TIMBRE_FEATURES}
            not_found_count += 1

    # Update acoustic features
    phrase_data["acoustic_features"]["spectral_centroid_hz"] = timbre["spectral_centroid_hz"]
    phrase_data["acoustic_features"]["spectral_slope"] = timbre["spectral_slope"]
    phrase_data["acoustic_features"]["spectral_bandwidth_hz"] = timbre["spectral_bandwidth_hz"]
    phrase_data["acoustic_features"]["spectral_rolloff_hz"] = timbre["spectral_rolloff_hz"]

    updated_count += 1

    if (updated_count % 200) == 0:
        print(f"  Updated {updated_count}/{len(marmoset_phrases)} phrases...")

print(f"\n✅ Updated {updated_count} phrases")
print(f"⚠️  {not_found_count} phrases had no timbre data (set to 0)")

# Save updated database
output_path = "/home/sheel/birdsong_analysis/src/vocalization_database_with_timbre.json"
print(f"\nSaving to {output_path}...")

with open(output_path, "w") as f:
    json.dump(db, f, indent=2)

print("✅ Saved!")

# Sample timbre values
print("\n📊 SAMPLE TIMBRE VALUES:")

sample_count = 0
for phrase_key, phrase_data in marmoset_phrases.items():
    af = phrase_data["acoustic_features"]
    if af["spectral_centroid_hz"] > 0:  # Only show non-zero timbre
        print(f"  {phrase_key}:")
        print(f"    spectral_centroid_hz: {af['spectral_centroid_hz']:.1f}")
        print(f"    spectral_slope: {af['spectral_slope']:.4f}")
        print(f"    spectral_bandwidth_hz: {af['spectral_bandwidth_hz']:.1f}")
        print(f"    spectral_rolloff_hz: {af['spectral_rolloff_hz']:.1f}")
        sample_count += 1
        if sample_count >= 5:
            break

print("\n" + "=" * 80)
print("✅ DATABASE UPDATE COMPLETE!")
print("=" * 80)
print("\n📊 SUMMARY:")
print(f"  Total phrases: {updated_count}")
print(f"  Phrases with timbre: {updated_count - not_found_count}")
print(f"  Phrases without timbre: {not_found_count}")
print("\n🎯 Next steps:")
print("  1. Backup old database:")
print("     mv /home/sheel/birdsong_analysis/src/vocalization_database.json \\")
print("        /home/sheel/birdsong_analysis/src/vocalization_database_old.json")
print("  2. Replace with new database:")
print(f"     mv {output_path} /home/sheel/birdsong_analysis/src/vocalization_database.json")
print("=" * 80)
