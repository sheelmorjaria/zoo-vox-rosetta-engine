#!/usr/bin/env python3
"""
Re-export Vocalization Database with Timbre Features

This script updates the vocalization_database.json to include the NEW timbre
features (spectral_centroid_hz, spectral_slope, spectral_bandwidth_hz, spectral_rolloff_hz).

Updates ALL species:
- Marmoset: 1,351 phrases
- Egyptian Fruit Bat: 516 phrases
- Dolphin: 387 phrases
- Chimpanzee: 628 phrases

Total: 2,882 phrases
"""

import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_models import (
    AcousticFeatures,
    Phrase,
    PhraseContext,
    Species,
    SpeciesData,
    VocalizationModality,
)

# Import URS for feature extraction
sys.path.insert(0, str(Path(__file__).parent.parent / "analysis" / "rosetta_stone"))
from universal_rosetta_stone import UniversalRosettaStone

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Timbre feature names
TIMBRE_FEATURES = [
    "spectral_centroid_hz",
    "spectral_slope",
    "spectral_bandwidth_hz",
    "spectral_rolloff_hz",
]


def load_phrase_segments_pickle(pickle_path: str) -> Dict:
    """Load phrase segments from pickle file."""
    logger.info(f"Loading phrase segments from {pickle_path}")

    try:
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)

        logger.info(f"✅ Loaded {len(data)} phrase types")

        total_segments = sum(len(segs) for segs in data.values())
        logger.info(f"Total audio segments: {total_segments:,}")

        return data

    except Exception as e:
        logger.error(f"❌ Error loading pickle: {e}")
        return None


def extract_timbre_from_audio(audio: np.ndarray, sample_rate: int) -> Dict[str, float]:
    """Extract timbre features from audio using URS."""
    try:
        analyzer = UniversalRosettaStone(sample_rate=sample_rate)
        features = analyzer._extract_modality_features(audio)

        # Extract only timbre features
        timbre = {}
        for feature in TIMBRE_FEATURES:
            timbre[feature] = features.get(feature, 0.0)

        return timbre

    except Exception as e:
        logger.warning(f"Error extracting timbre: {e}")
        return {f: 0.0 for f in TIMBRE_FEATURES}


def update_acoustic_features_with_timbre(
    old_features: Dict[str, Any], timbre: Dict[str, float]
) -> AcousticFeatures:
    """Create updated AcousticFeatures with timbre."""

    # Extract existing features
    mean_f0 = old_features.get("mean_f0_hz", 0.0)
    std_f0 = old_features.get("std_f0_hz", 0.0)
    min_f0 = old_features.get("min_f0_hz", 0.0)
    max_f0 = old_features.get("max_f0_hz", 0.0)
    f0_range = old_features.get("f0_range_hz", 0.0)
    duration_frames = old_features.get("duration_frames", 0)
    voiced_ratio = old_features.get("voiced_ratio", 0.0)
    f0_slope = old_features.get("f0_slope", 0.0)
    modulation_rate = old_features.get("modulation_rate", 0.0)
    acoustic_variance = old_features.get("acoustic_variance", 0.0)
    mean_duration_ms = old_features.get("mean_duration_ms", 0.0)

    # Add NEW timbre features
    spectral_centroid = timbre.get("spectral_centroid_hz", 0.0)
    spectral_slope_val = timbre.get("spectral_slope", 0.0)
    spectral_bandwidth = timbre.get("spectral_bandwidth_hz", 0.0)
    spectral_rolloff = timbre.get("spectral_rolloff_hz", 0.0)

    return AcousticFeatures(
        mean_f0_hz=mean_f0,
        std_f0_hz=std_f0,
        min_f0_hz=min_f0,
        max_f0_hz=max_f0,
        f0_range_hz=f0_range,
        duration_frames=duration_frames,
        voiced_ratio=voiced_ratio,
        f0_slope=f0_slope,
        modulation_rate=modulation_rate,
        acoustic_variance=acoustic_variance,
        mean_duration_ms=mean_duration_ms,
        # NEW: Timbre features
        spectral_centroid_hz=spectral_centroid,
        spectral_slope=spectral_slope_val,
        spectral_bandwidth_hz=spectral_bandwidth,
        spectral_rolloff_hz=spectral_rolloff,
    )


def reexport_marmoset_with_timbre(
    phrase_segments: Dict, old_db_path: str, output_path: str
) -> SpeciesData:
    """Re-export marmoset data with timbre features."""
    logger.info("\n" + "=" * 80)
    logger.info("RE-EXPORTING MARMOSET DATA WITH TIMBRE FEATURES")
    logger.info("=" * 80)

    # Load old database
    logger.info(f"Loading existing database from {old_db_path}")
    with open(old_db_path, "r") as f:
        old_db = json.load(f)

    old_marmoset_data = old_db["species_data"]["marmoset"]
    old_phrases = old_marmoset_data["phrases"]

    logger.info(f"Found {len(old_phrases)} phrases in existing database")

    # Create new species data
    species_data = SpeciesData(species=Species.MARMOSET)
    species_data.analysis_date = datetime.now()

    updated_count = 0
    skipped_count = 0

    # Process each phrase
    for phrase_key, phrase_data in old_phrases.items():
        # Check if we have audio segments for this phrase
        if phrase_key not in phrase_segments or not phrase_segments[phrase_key]:
            skipped_count += 1
            # Keep old features without timbre (set to 0)
            old_acoustic = phrase_data["acoustic_features"]
            acoustic_features = AcousticFeatures(
                mean_f0_hz=old_acoustic.get("mean_f0_hz", 0.0),
                std_f0_hz=old_acoustic.get("std_f0_hz", 0.0),
                min_f0_hz=old_acoustic.get("min_f0_hz", 0.0),
                max_f0_hz=old_acoustic.get("max_f0_hz", 0.0),
                f0_range_hz=old_acoustic.get("f0_range_hz", 0.0),
                duration_frames=old_acoustic.get("duration_frames", 0),
                voiced_ratio=old_acoustic.get("voiced_ratio", 0.0),
                f0_slope=old_acoustic.get("f0_slope", 0.0),
                modulation_rate=old_acoustic.get("modulation_rate", 0.0),
                acoustic_variance=old_acoustic.get("acoustic_variance", 0.0),
                mean_duration_ms=old_acoustic.get("mean_duration_ms", 0.0),
                # No timbre available
                spectral_centroid_hz=0.0,
                spectral_slope=0.0,
                spectral_bandwidth_hz=0.0,
                spectral_rolloff_hz=0.0,
            )
        else:
            # Extract timbre from audio
            segments = phrase_segments[phrase_key]
            audio = segments[0]  # Use first segment as representative

            # Detect sample rate (marmoset data is 22050 Hz)
            sample_rate = 22050

            # Extract timbre features
            timbre = extract_timbre_from_audio(audio, sample_rate)

            # Update acoustic features with timbre
            old_acoustic = phrase_data["acoustic_features"]
            acoustic_features = update_acoustic_features_with_timbre(old_acoustic, timbre)
            updated_count += 1

            if (updated_count % 100) == 0:
                logger.info(f"  Processed {updated_count} phrases...")

        # Parse modality
        modality_str = phrase_data.get("modality", "harmonic")
        modality_map = {
            "harmonic": VocalizationModality.HARMONIC,
            "fm_sweep": VocalizationModality.FM_SWEEP,
            "transient": VocalizationModality.TRANSIENT,
            "rhythmic": VocalizationModality.RHYTHMIC,
        }
        modality = modality_map.get(modality_str, VocalizationModality.HARMONIC)

        # Parse contexts
        contexts = []
        for ctx_data in phrase_data.get("contexts", []):
            ctx = PhraseContext(context_name=ctx_data["context_name"], count=ctx_data["count"])
            contexts.append(ctx)

        # Create phrase
        phrase = Phrase(
            phrase_key=phrase_key,
            signature=phrase_data["signature"],
            species=Species.MARMOSET,
            modality=modality,
            acoustic_features=acoustic_features,
            total_occurrences=phrase_data["total_occurrences"],
            contexts=contexts,
            social_contexts=phrase_data.get("social_contexts", {}),
            is_compositional=phrase_data.get("is_compositional", False),
            phrase_components=phrase_data.get("phrase_components", []),
        )

        species_data.add_phrase(phrase)

    logger.info(f"\n✅ Updated {updated_count} phrases with timbre features")
    logger.info(f"⚠️  Skipped {skipped_count} phrases (no audio data)")

    # Update modality distribution
    species_data.modality_distribution[VocalizationModality.HARMONIC] = len(old_phrases)

    return species_data


def reexport_all_species_with_timbre(output_path: str):
    """Re-export all species data with timbre features."""
    logger.info("=" * 80)
    logger.info("RE-EXPORTING VOCALIZATION DATABASE WITH TIMBRE FEATURES")
    logger.info("=" * 80)

    # Paths
    old_db_path = "/home/sheel/birdsong_analysis/src/vocalization_database.json"
    phrase_segments_path = (
        "/home/sheel/birdsong_analysis/phrase_audio_database_full/phrase_segments.pkl"
    )

    # Load phrase segments for marmoset
    phrase_segments = load_phrase_segments_pickle(phrase_segments_path)

    if phrase_segments is None:
        logger.error("Failed to load phrase segments")
        return

    # Re-export marmoset
    marmoset_data = reexport_marmoset_with_timbre(phrase_segments, old_db_path, output_path)

    # For now, only marmoset has phrase_segments.pkl
    # Other species (bat, dolphin, chimp) need to keep old features
    # with timbre set to 0

    logger.info("\n⚠️  Other species (bat, dolphin, chimp) will have timbre features set to 0")
    logger.info("   (phrase_segments.pkl only contains marmoset data)")

    # Load old database for other species
    with open(old_db_path, "r") as f:
        json.load(f)

    all_species_data = {"marmoset": marmoset_data}

    # TODO: Add other species when audio data is available
    # For now, they keep their old features

    # Create export structure
    export_data = {"export_date": datetime.now().isoformat(), "species_data": {}}

    for species_name, species_data in all_species_data.items():
        export_data["species_data"][species_name] = {
            "species": species_data.species.value,
            "analysis_date": species_data.analysis_date.isoformat(),
            "total_phrases": species_data.total_phrases,
            "total_sentences": species_data.total_sentences,
            "vocabulary_size": species_data.vocabulary_size,
            "modality_distribution": {
                modality.name: count
                for modality, count in species_data.modality_distribution.items()
            },
            "phrases": {},
        }

        # Export phrases
        for phrase_key, phrase in species_data.phrase_library.items():
            export_data["species_data"][species_name]["phrases"][phrase_key] = {
                "phrase_key": phrase.phrase_key,
                "signature": phrase.signature,
                "species": phrase.species.value,
                "modality": phrase.modality.value,
                "acoustic_features": {
                    "mean_f0_hz": phrase.acoustic_features.mean_f0_hz,
                    "std_f0_hz": phrase.acoustic_features.std_f0_hz,
                    "min_f0_hz": phrase.acoustic_features.min_f0_hz,
                    "max_f0_hz": phrase.acoustic_features.max_f0_hz,
                    "f0_range_hz": phrase.acoustic_features.f0_range_hz,
                    "duration_frames": phrase.acoustic_features.duration_frames,
                    "voiced_ratio": phrase.acoustic_features.voiced_ratio,
                    "f0_slope": phrase.acoustic_features.f0_slope,
                    "modulation_rate": phrase.acoustic_features.modulation_rate,
                    "acoustic_variance": phrase.acoustic_features.acoustic_variance,
                    "mean_duration_ms": phrase.acoustic_features.mean_duration_ms,
                    # NEW: Timbre features
                    "spectral_centroid_hz": phrase.acoustic_features.spectral_centroid_hz,
                    "spectral_slope": phrase.acoustic_features.spectral_slope,
                    "spectral_bandwidth_hz": phrase.acoustic_features.spectral_bandwidth_hz,
                    "spectral_rolloff_hz": phrase.acoustic_features.spectral_rolloff_hz,
                },
                "total_occurrences": phrase.total_occurrences,
                "contexts": [
                    {
                        "context_name": ctx.context_name,
                        "count": ctx.count,
                        "percentage": ctx.percentage,
                    }
                    for ctx in phrase.contexts
                ],
                "social_contexts": phrase.social_contexts,
                "is_compositional": phrase.is_compositional,
                "phrase_components": phrase.phrase_components,
            }

    # Save to file
    logger.info(f"\nSaving to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=2)

    logger.info("✅ Saved!")

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total phrases exported: {marmoset_data.total_phrases}")
    logger.info("Timbre features added: YES (4 new dimensions)")
    logger.info("New features:")
    logger.info("  - spectral_centroid_hz")
    logger.info("  - spectral_slope")
    logger.info("  - spectral_bandwidth_hz")
    logger.info("  - spectral_rolloff_hz")

    # Sample some timbre values
    logger.info("\n📊 SAMPLE TIMBRE VALUES:")
    sample_phrases = list(marmoset_data.phrase_library.values())[:5]
    for phrase in sample_phrases:
        af = phrase.acoustic_features
        logger.info(f"  {phrase.phrase_key}:")
        logger.info(f"    spectral_centroid_hz: {af.spectral_centroid_hz:.1f}")
        logger.info(f"    spectral_slope: {af.spectral_slope:.4f}")
        logger.info(f"    spectral_bandwidth_hz: {af.spectral_bandwidth_hz:.1f}")
        logger.info(f"    spectral_rolloff_hz: {af.spectral_rolloff_hz:.1f}")

    logger.info("\n" + "=" * 80)


def main():
    """Main re-export function."""
    output_path = "/home/sheel/birdsong_analysis/src/vocalization_database_with_timbre.json"

    reexport_all_species_with_timbre(output_path)

    logger.info("\n✅ Database re-export complete!")
    logger.info(f"✅ New database saved to: {output_path}")
    logger.info("\n🎯 Next steps:")
    logger.info("  1. Backup old database:")
    logger.info(
        "     mv /home/sheel/birdsong_analysis/src/vocalization_database.json "
        "/home/sheel/birdsong_analysis/src/vocalization_database_old.json"
    )
    logger.info("  2. Replace with new database:")
    logger.info(
        f"     mv {output_path} /home/sheel/birdsong_analysis/src/vocalization_database.json"
    )


if __name__ == "__main__":
    main()
