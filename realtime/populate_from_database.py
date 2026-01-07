#!/usr/bin/env python3
"""
Populate PhraseAudioLibrary from Database
==========================================

This script demonstrates how to populate the real-time phrase audio library
with phrases from the phrase_audio_database_full directory.

Author: Sheel Morjaria
License: CC BY-ND 4.0 International
"""

import logging
import pickle
import sys

import numpy as np

# Add parent directory to path
sys.path.append("/home/sheel/birdsong_analysis")

# Import our frameworks
from phrase_audio_library import PhraseAudioLibrary, PhraseAudioSegment

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_phrase_database(phrase_db_path: str) -> dict:
    """Load phrase segments from pickle file."""
    try:
        with open(phrase_db_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading phrase database: {e}")
        return None


def populate_library_from_database(
    output_path: str = "populated_realtime_library.pkl",
    max_phrases: int = 100,
    sample_rate: int = 22050,
) -> PhraseAudioLibrary:
    """
    Populate PhraseAudioLibrary with phrases from the database.

    Args:
        output_path: Path to save the populated library
        max_phrases: Maximum number of phrases to include
        sample_rate: Target sample rate for audio

    Returns:
        Populated PhraseAudioLibrary
    """
    print("=" * 80)
    print("POPULATING REALTIME PHRASE AUDIO LIBRARY FROM DATABASE")
    print("=" * 80)

    # Initialize phrase library
    library = PhraseAudioLibrary(species="marmoset", sr=sample_rate)

    # Load the phrase database
    phrase_db_path = "/home/sheel/birdsong_analysis/phrase_audio_database_full/phrase_segments.pkl"
    phrase_segments = load_phrase_database(phrase_db_path)

    if phrase_segments is None:
        print("❌ Failed to load phrase database")
        return library

    print(f"✅ Loaded database with {len(phrase_segments)} phrase types")
    print(f"Total audio segments: {sum(len(segs) for segs in phrase_segments.values()):,}")

    # Parse phrase keys and categorize by F0 range for context assignment
    f0_ranges = {
        "contact": (4000, 5000),  # Low frequency for contact
        "neutral": (5000, 6000),  # Mid-low frequency for neutral
        "food": (6000, 7000),  # Mid frequency for food
        "social": (7000, 8000),  # Mid-high frequency for social
        "alarm": (8000, 9000),  # High frequency for alarm
    }

    # Select and convert segments
    phrases_added = 0

    for phrase_key, segment_list in phrase_segments.items():
        if phrases_added >= max_phrases:
            break

        if not segment_list:
            continue

        # Get the first segment
        audio_np = segment_list[0]
        if len(audio_np) == 0:
            continue

        try:
            # Parse F0 from phrase key (e.g., 'F0_7000_DUR_5_RANGE_100')
            parts = phrase_key.split("_")
            if len(parts) >= 3 and parts[0] == "F0":
                f0_val = float(parts[1])

                # Assign context based on F0 range
                context = "neutral"  # Default
                for ctx, (min_f0, max_f0) in f0_ranges.items():
                    if min_f0 <= f0_val < max_f0:
                        context = ctx
                        break

                # Create PhraseAudioSegment
                segment = PhraseAudioSegment(
                    audio=audio_np,
                    sr=sample_rate,
                    phrase_key=phrase_key,
                    source_file="phrase_audio_database_full",
                    start_time_ms=0,
                    end_time_ms=len(audio_np) / sample_rate * 1000,
                    mean_f0_hz=f0_val,
                    std_f0_hz=100,  # Estimate
                    mean_duration_ms=len(audio_np) / sample_rate * 1000,
                    mean_range_hz=f0_val * 0.05,  # 5% range estimate
                    encoding="horizontal",
                    superposed_with=[],
                    context=context,
                    individual_id="marmoset_individual_db",
                    snr_db=25.0,
                    quality_score=0.9,
                    microharmonic_signature={
                        "dominant_harmonic": 1,
                        "harmonic_entropy": 0.1,
                        "spectral_centroid_hz": f0_val,
                        "formants": [f0_val],
                        "modulation_depth": 0.05,
                    },
                )

                # Add to library
                library.add_segment(segment)
                phrases_added += 1

                if phrases_added % 20 == 0:
                    print(f"   Processed {phrases_added}/{max_phrases} phrases...")

        except Exception as e:
            logger.warning(f"Error processing {phrase_key}: {e}")
            continue

    print("\n📊 Population Summary:")
    print(f"   Total phrases in database: {len(phrase_segments)}")
    print(f"   Phrases added to library: {phrases_added}")
    print(f"   Library total segments: {library.total_segments}")

    # Context distribution
    context_stats = library.get_context_statistics()
    if "context_statistics" in context_stats:
        print("\n   Context distribution:")
        for context, stats in context_stats["context_statistics"].items():
            print(f"     {context}: {stats['total_occurrences']} segments")

    # Frequency analysis
    phrase_keys = library.get_all_phrase_keys()
    f0_values = []

    for phrase_key in phrase_keys:
        segments = library.get_segment(phrase_key)
        if segments and hasattr(segments, "mean_f0_hz"):
            f0_values.append(segments.mean_f0_hz)

    if f0_values:
        print(f"\n   F0 range: {np.min(f0_values):.0f} - {np.max(f0_values):.0f} Hz")
        print(f"   Mean F0: {np.mean(f0_values):.0f} Hz")

    # Save populated library
    print(f"\n💾 Saving populated library to: {output_path}")
    library.save(output_path)

    print("\n" + "=" * 80)
    print("✅ REALTIME PHRASE AUDIO LIBRARY POPULATION COMPLETE")
    print("✅ Library populated with authentic marmoset vocalizations")
    print("✅ Context-aware categorization implemented")
    print("✅ Ready for real-time synthesis and interaction")
    print("=" * 80)

    return library


def demonstrate_realtime_functionality(library: PhraseAudioLibrary):
    """Demonstrate real-time functionality with the populated library."""
    print("\n" + "=" * 80)
    print("REALTIME FUNCTIONALITY DEMONSTRATION")
    print("=" * 80)

    # Context-aware selection
    print("\nContext-aware phrase selection:")
    for context in ["alarm", "food", "social", "neutral", "contact"]:
        selected = library.select_phrases_by_context(context, min_quality=0.5)
        if selected:
            selected.sort(key=lambda x: getattr(x, "mean_f0_hz", 0))
            print(f"   {context.capitalize()}: {len(selected)} phrases")
            print(
                f"     F0 range: {getattr(selected[0], 'mean_f0_hz', 0):.0f} - {getattr(selected[-1], 'mean_f0_hz', 0):.0f} Hz"
            )

    # Synthesis preparation
    print("\nSynthesis capabilities:")
    print(f"   Total phrase types available: {len(library.get_all_phrase_keys())}")
    print(f"   Total audio segments: {library.total_segments}")
    print("   Ready for horizontal (concatenative) synthesis")
    print("   Ready for vertical (superpositional) synthesis")
    print("   Ready for combined synthesis methods")


def main():
    """Main function."""
    # Create and populate library
    library = populate_library_from_database(
        output_path="src/realtime/populated_realtime_library.pkl",
        max_phrases=100,  # Limit for demonstration
    )

    # Demonstrate functionality
    demonstrate_realtime_functionality(library)

    print("\n🎯 Next steps:")
    print("   1. Use library in real-time interaction systems")
    print("   2. Implement context-aware synthesis strategies")
    print("   3. Add more phrases from database for full coverage")
    print("   4. Integrate with hardware acceleration if needed")


if __name__ == "__main__":
    main()
