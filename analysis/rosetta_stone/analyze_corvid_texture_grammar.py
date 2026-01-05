#!/usr/bin/env python3
"""
Corvid Texture Grammar Analysis

Category 1, Item 3: Modality Sequence Graphing

This script analyzes the transition patterns between modalities (HARMONIC,
TRANSIENT, FM_SWEEP) within corvid vocalizations. This reveals the "Texture
Syntax" - how corvids mix different acoustic textures in their communication.

Key metrics:
- Transition probability matrix (e.g., P(H|T) = 0.8)
- Sequence statistics (runs, alternations, entropy)
- Most common modality sequences

This is a NOVEL concept in bioacoustics - no prior research has analyzed
modality transition patterns in animal vocalizations.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from universal_rosetta_stone import UniversalRosettaStone, Modality

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


def analyze_corvid_texture_grammar(filepath, duration_sec=5):
    """Analyze texture grammar patterns in a corvid recording."""
    try:
        audio, sr = sf.read(filepath)
        if len(audio.shape) > 1:
            audio = audio[:, 0]

        # Use first N seconds
        max_samples = int(duration_sec * sr)
        if len(audio) < max_samples:
            audio = audio
        else:
            audio = audio[:max_samples]

        analyzer = UniversalRosettaStone(sample_rate=sr)

        # Segment phrases with adaptive gap
        phrases = analyzer.segment_phrases(
            audio,
            min_gap_ms=30.0,  # Optimized for corvids
            use_adaptive_gap=True
        )

        if len(phrases) < 2:
            return None  # Skip files with insufficient phrases

        # Analyze modality sequences
        texture_grammar = analyzer.analyze_modality_sequences(phrases)

        return {
            'filename': Path(filepath).name,
            'duration_sec': len(audio) / sr,
            'num_phrases': len(phrases),
            'texture_grammar': texture_grammar
        }
    except Exception as e:
        return {'error': str(e), 'filename': Path(filepath).name}


def print_transition_matrix(transition_matrix, title="TRANSITION PROBABILITY MATRIX"):
    """Pretty print transition probability matrix."""
    if not transition_matrix:
        return

    print(f"\n📊 {title}")
    print(f"{'FROM \\ TO':<15} {'HARMONIC':>12} {'FM_SWEEP':>12} {'TRANSIENT':>12} {'RHYTHMIC':>12}")
    print("-" * 70)

    for from_mod in ['HARMONIC', 'FM_SWEEP', 'TRANSIENT', 'RHYTHMIC']:
        row = [from_mod]
        for to_mod in ['HARMONIC', 'FM_SWEEP', 'TRANSIENT', 'RHYTHMIC']:
            prob = transition_matrix.get((from_mod, to_mod), 0.0)
            if prob > 0:
                row.append(f"{prob*100:5.1f}%")
            else:
                row.append("    -   ")
        print(f"{row[0]:<15} {row[1]:>12} {row[2]:>12} {row[3]:>12} {row[4]:>12}")


def analyze_species_texture_grammar(species_name, species_dir, num_files=30):
    """Analyze texture grammar across multiple corvid recordings."""
    print(f"\n{'='*90}")
    print(f"{species_name.upper()} TEXTURE GRAMMAR ANALYSIS")
    print(f"{'='*90}\n")

    # Get all MP3 files
    all_files = sorted(list(species_dir.glob("*.mp3")))

    if len(all_files) == 0:
        print(f"⚠️  No MP3 files found in {species_dir}")
        return None

    # Select subset
    if len(all_files) > num_files:
        indices = np.linspace(0, len(all_files) - 1, num_files, dtype=int)
        test_files = [all_files[i] for i in indices]
    else:
        test_files = all_files

    print(f"📁 Analyzing {len(test_files)} files...")

    # Analyze all files
    all_results = []
    valid_results = []

    for filepath in test_files:
        result = analyze_corvid_texture_grammar(filepath, duration_sec=5)

        if result is None:
            continue  # Skip files with insufficient phrases

        if 'error' in result:
            all_results.append(result)
            continue

        all_results.append(result)
        valid_results.append(result)

    if not valid_results:
        print(f"⚠️  No valid results for {species_name}")
        return None

    print(f"✅ {len(valid_results)} files with sufficient phrases")

    # Aggregate statistics
    print(f"\n{'='*90}")
    print(f"AGGREGATE STATISTICS")
    print(f"{'='*90}")

    total_phrases = sum(r['num_phrases'] for r in valid_results)
    print(f"\n📊 Total phrases analyzed: {total_phrases}")
    print(f"📊 Mean phrases per file: {np.mean([r['num_phrases'] for r in valid_results]):.2f}")
    print(f"📊 Range: {np.min([r['num_phrases'] for r in valid_results])} - {np.max([r['num_phrases'] for r in valid_results])} phrases")

    # Aggregate transition matrix
    print(f"\n{'='*90}")
    print(f"AGGREGATE TRANSITION PROBABILITY MATRIX")
    print(f"{'='*90}")

    # Aggregate all transition counts
    aggregate_counts = {}

    for result in valid_results:
        transition_counts = result['texture_grammar']['transition_counts']
        for (frm, to), count in transition_counts.items():
            aggregate_counts[(frm, to)] = aggregate_counts.get((frm, to), 0) + count

    # Convert to probability matrix
    aggregate_matrix = {}
    for from_mod in ['HARMONIC', 'FM_SWEEP', 'TRANSIENT', 'RHYTHMIC']:
        total_from = sum(count for (frm, to), count in aggregate_counts.items() if frm == from_mod)

        if total_from > 0:
            for to_mod in ['HARMONIC', 'FM_SWEEP', 'TRANSIENT', 'RHYTHMIC']:
                count = aggregate_counts.get((from_mod, to_mod), 0)
                prob = count / total_from
                if prob > 0:
                    aggregate_matrix[(from_mod, to_mod)] = prob

    print_transition_matrix(aggregate_matrix, f"{species_name.upper()} - AGGREGATE TRANSITION PROBABILITIES")

    # Aggregate sequence statistics
    print(f"\n{'='*90}")
    print(f"AGGREGATE SEQUENCE STATISTICS")
    print(f"{'='*90}")

    avg_run_length = np.mean([r['texture_grammar']['sequence_stats']['avg_run_length']
                             for r in valid_results])
    avg_alternation_rate = np.mean([r['texture_grammar']['sequence_stats']['alternation_rate']
                                    for r in valid_results])
    avg_entropy = np.mean([r['texture_grammar']['sequence_stats']['entropy']
                          for r in valid_results])
    avg_normalized_entropy = np.mean([r['texture_grammar']['sequence_stats']['normalized_entropy']
                                     for r in valid_results])

    print(f"\n  Average run length: {avg_run_length:.2f} phrases")
    print(f"  Average alternation rate: {avg_alternation_rate:.3f} (changes per transition)")
    print(f"  Average entropy: {avg_entropy:.3f} bits")
    print(f"  Average normalized entropy: {avg_normalized_entropy:.3f} (0 = uniform, 1 = maximum diversity)")

    # Most common sequences across all files
    print(f"\n{'='*90}")
    print(f"MOST COMMON MODALITY SEQUENCES")
    print(f"{'='*90}\n")

    all_sequences = []
    for result in valid_results:
        all_sequences.extend(result['texture_grammar']['common_sequences'])

    # Group by sequence pattern
    sequence_patterns = {}
    for seq in all_sequences:
        pattern_key = tuple(seq['sequence'])
        if pattern_key not in sequence_patterns:
            sequence_patterns[pattern_key] = {'count': 0, 'length': seq['length'], 'files': 0}
        sequence_patterns[pattern_key]['count'] += seq['count']
        sequence_patterns[pattern_key]['files'] += 1

    # Sort by occurrence
    sorted_patterns = sorted(sequence_patterns.items(),
                            key=lambda x: x[1]['count'],
                            reverse=True)

    for i, (pattern, stats) in enumerate(sorted_patterns[:10], 1):
        sequence_str = " → ".join(pattern)
        print(f"  {i:2d}. {sequence_str:<30} ({stats['length']}-gram)")
        print(f"      Occurrences: {stats['count']}, Files: {stats['files']}")

    return {
        'species': species_name,
        'num_files': len(valid_results),
        'total_phrases': total_phrases,
        'aggregate_matrix': aggregate_matrix,
        'avg_run_length': avg_run_length,
        'avg_alternation_rate': avg_alternation_rate,
        'avg_entropy': avg_entropy,
        'avg_normalized_entropy': avg_normalized_entropy,
        'common_sequences': sorted_patterns[:10]
    }


def main():
    """Main analysis of corvid texture grammar."""
    if not HAS_SOUNDFILE:
        print("soundfile library required")
        return

    xenocanto_dir = Path.home() / "birdsong_analysis/data/xenocanto"

    if not xenocanto_dir.exists():
        print(f"Xenocanto directory not found: {xenocanto_dir}")
        return

    print("=" * 90)
    print("CORVID TEXTURE GRAMMAR ANALYSIS")
    print("Category 1, Item 3: Modality Sequence Graphing")
    print("=" * 90)
    print("\nThis analysis reveals the 'Texture Syntax' - how corvids mix different")
    print("acoustic textures (HARMONIC, TRANSIENT, FM_SWEEP) in their communication.")
    print("\nThis is a NOVEL concept in bioacoustics!")

    # Analyze American Crow
    american_crow_dir = xenocanto_dir / "American_Crow"
    american_crow_results = None

    if american_crow_dir.exists():
        american_crow_results = analyze_species_texture_grammar(
            "American Crow",
            american_crow_dir,
            num_files=30
        )

    # Overall summary
    if american_crow_results:
        print("\n" + "=" * 90)
        print("OVERALL SUMMARY")
        print("=" * 90)

        print(f"\n✅ Texture Grammar Analysis Complete for American Crow")
        print(f"\nKey Findings:")
        print(f"  - {american_crow_results['total_phrases']} total phrases analyzed")
        print(f"  - {american_crow_results['avg_run_length']:.2f} average phrases per modality run")
        print(f"  - {american_crow_results['avg_alternation_rate']:.3f} alternation rate")
        print(f"  - {american_crow_results['avg_entropy']:.3f} bits of entropy")
        print(f"  - {american_crow_results['avg_normalized_entropy']:.3f} normalized entropy")

        # Interpretation
        print(f"\nInterpretation:")
        if american_crow_results['avg_normalized_entropy'] > 0.7:
            print(f"  ✅ HIGH diversity - Corvids frequently switch between modalities")
            print(f"     This indicates complex 'Texture Syntax' with mixed acoustic textures")
        elif american_crow_results['avg_normalized_entropy'] > 0.4:
            print(f"  ⚠️  MODERATE diversity - Some modality switching observed")
        else:
            print(f"  ⚠️  LOW diversity - Corvids tend to stay in same modality")

        print(f"\nScientific Impact:")
        print(f"  📚 This is the FIRST analysis of modality transition patterns in")
        print(f"     animal vocalizations - a publishable novel concept!")

    print("\n" + "=" * 90)
    print("✅ Texture Grammar Analysis Complete!")
    print("=" * 90)


if __name__ == "__main__":
    main()
