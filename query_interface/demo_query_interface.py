"""
Demo Script for Vocalization Query Interface

This script demonstrates how to use the query interface for various
search, analysis, and data retrieval operations.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time

from data_models import Species, VocalizationModality
from query_interface.vocalization_query_interface import (
    get_phrase_similarities,
    get_query_interface,
    query_phrases_by_duration,
    query_phrases_by_f0,
)


def demo_basic_queries():
    """Demonstrate basic query operations"""
    print("="*50)
    print("BASIC QUERY DEMONSTRATION")
    print("="*50)

    # Get query interface
    interface = get_query_interface()

    # Get database info
    print("\n1. Database Information:")
    db_info = interface.get_database_info()
    for key, value in db_info.items():
        print(f"   {key}: {value}")

    # Get phrases for each species
    print("\n2. Phrases by Species:")
    for species in Species:
        phrases = interface.get_phrases_by_species(species)
        print(f"   {species.value}: {len(phrases)} phrases")

    # Get phrases by modality
    print("\n3. Phrases by Modality:")
    for modality in VocalizationModality:
        phrases_by_species = interface.get_phrases_by_modality(modality)
        total = sum(len(phrases) for phrases in phrases_by_species.values())
        print(f"   {modality.value}: {total} phrases")


def demo_search_operations():
    """Demonstrate search capabilities"""
    print("\n" + "="*50)
    print("SEARCH OPERATIONS DEMONSTRATION")
    print("="*50)

    interface = get_query_interface()

    # Search by F0 range
    print("\n1. Searching phrases by F0 range (5000-10000 Hz):")
    f0_results = query_phrases_by_f0(5000, 10000)
    print(f"   Found {len(f0_results)} phrases")
    for i, (phrase_key, phrase) in enumerate(f0_results[:5]):  # Show first 5
        print(f"   {i+1}. {phrase_key}: {phrase.acoustic_features.mean_f0_hz:.1f} Hz "
              f"({phrase.species.value}, {phrase.total_occurrences} occurrences)")

    # Search by duration
    print("\n2. Searching phrases by duration (50-150 ms):")
    duration_results = query_phrases_by_duration(50, 150)
    print(f"   Found {len(duration_results)} phrases")
    for i, (phrase_key, phrase) in enumerate(duration_results[:5]):  # Show first 5
        print(f"   {i+1}. {phrase_key}: {phrase.acoustic_features.mean_duration_ms:.1f} ms "
              f"({phrase.species.value})")

    # Find similar phrases
    print("\n3. Finding similar phrases:")
    if interface.get_phrases_by_species(Species.MARMOSET):
        # Get a sample phrase from marmosets
        sample_phrase_key = list(interface.get_phrases_by_species(Species.MARMOSET).keys())[0]
        print(f"   Finding similar phrases to: {sample_phrase_key}")
        similar_phrases = get_phrase_similarities(sample_phrase_key, threshold=0.6)
        print(f"   Found {len(similar_phrases)} similar phrases")
        for similarity, phrase_key, phrase in similar_phrases[:3]:
            print(f"   {similarity:.3f} - {phrase_key} ({phrase.species.value})")


def demo_statistics_analysis():
    """Demonstrate statistical analysis capabilities"""
    print("\n" + "="*50)
    print("STATISTICAL ANALYSIS DEMONSTRATION")
    print("="*50)

    interface = get_query_interface()

    # Get overall statistics
    print("\n1. Overall Database Statistics:")
    stats = interface.get_phrase_statistics()
    print(f"   Total phrases: {stats['total_phrases']}")
    print(f"   Frequency range: {stats['frequency_distribution']['min']:.1f} - "
          f"{stats['frequency_distribution']['max']:.1f} Hz "
          f"(avg: {stats['frequency_distribution']['avg']:.1f} Hz)")
    print(f"   Duration range: {stats['duration_distribution']['min']:.1f} - "
          f"{stats['duration_distribution']['max']:.1f} ms "
          f"(avg: {stats['duration_distribution']['avg']:.1f} ms)")

    print("\n   Species breakdown:")
    for species, count in stats['species_breakdown'].items():
        percentage = (count / stats['total_phrases']) * 100 if stats['total_phrases'] > 0 else 0
        print(f"     {species}: {count} ({percentage:.1f}%)")

    print("\n   Modality breakdown:")
    for modality, count in stats['modality_breakdown'].items():
        percentage = (count / stats['total_phrases']) * 100 if stats['total_phrases'] > 0 else 0
        print(f"     {modality}: {count} ({percentage:.1f}%)")

    # Get species-specific statistics
    print("\n2. Marmoset-specific Statistics:")
    marmoset_stats = interface.get_phrase_statistics(Species.MARMOSET)
    if marmoset_stats['total_phrases'] > 0:
        print(f"   Total phrases: {marmoset_stats['total_phrases']}")
        print(f"   Average F0: {marmoset_stats['frequency_distribution']['avg']:.1f} Hz")
        print(f"   Average duration: {marmoset_stats['duration_distribution']['avg']:.1f} ms")

    # Get grammar network analysis
    print("\n3. Grammar Network Analysis:")
    grammar_network = interface.get_grammar_network()
    print(f"   Nodes (phrase types): {grammar_network['nodes']}")
    print(f"   Edges (grammar rules): {grammar_network['edges']}")

    if grammar_network['most_connected_phrases']:
        print("   Most connected phrases:")
        for phrase, connections in grammar_network['most_connected_phrases'][:5]:
            print(f"     {phrase}: {connections} connections")

    if grammar_network['strongest_transitions']:
        print("   Strongest transitions:")
        for from_phrase, to_phrase, strength in grammar_network['strongest_transitions'][:5]:
            print(f"     {from_phrase} → {to_phrase}: {strength} occurrences")


def demo_cross_species_analysis():
    """Demonstrate cross-species analysis capabilities"""
    print("\n" + "="*50)
    print("CROSS-SPECIES ANALYSIS DEMONSTRATION")
    print("="*50)

    interface = get_query_interface()

    # Find common patterns across species
    print("\n1. Common frequency ranges:")
    for species in Species:
        stats = interface.get_phrase_statistics(species)
        if stats['total_phrases'] > 0:
            f0_avg = stats['frequency_distribution']['avg']
            f0_min = stats['frequency_distribution']['min']
            f0_max = stats['frequency_distribution']['max']
            print(f"   {species.value}: {f0_min:.0f}-{f0_max:.0f} Hz (avg: {f0_avg:.0f} Hz)")

    # Compare modalities across species
    print("\n2. Modality distribution by species:")
    for species in Species:
        phrases = interface.get_phrases_by_species(species)
        modality_counts = {}
        for phrase in phrases.values():
            modality = phrase.modality.value
            modality_counts[modality] = modality_counts.get(modality, 0) + 1

        print(f"   {species.value}:")
        for modality, count in modality_counts.items():
            print(f"     {modality}: {count} phrases")


def demo_performance_benchmarks():
    """Demonstrate query performance"""
    print("\n" + "="*50)
    print("PERFORMANCE BENCHMARKS")
    print("="*50)

    interface = get_query_interface()

    # Test query performance
    print("\n1. Query performance tests:")

    # F0 query
    start_time = time.time()
    results = query_phrases_by_f0(0, 50000)  # All frequencies
    f0_time = time.time() - start_time
    print(f"   F0 query (all phrases): {len(results)} results in {f0_time:.4f} seconds")

    # Duration query
    start_time = time.time()
    results = query_phrases_by_duration(0, 2000)  # All durations
    duration_time = time.time() - start_time
    print(f"   Duration query (all phrases): {len(results)} results in {duration_time:.4f} seconds")

    # Similarity query
    if interface.get_phrases_by_species(Species.MARMOSET):
        sample_phrase_key = list(interface.get_phrases_by_species(Species.MARMOSET).keys())[0]
        start_time = time.time()
        results = get_phrase_similarities(sample_phrase_key, threshold=0.5)
        similarity_time = time.time() - start_time
        print(f"   Similarity query: {len(results)} results in {similarity_time:.4f} seconds")

    # Grammar network generation
    start_time = time.time()
    interface.get_grammar_network()
    grammar_time = time.time() - start_time
    print(f"   Grammar network generation: {grammar_time:.4f} seconds")


def main():
    """Run all demo functions"""
    print("🎵 Vocalization Query Interface Demo")
    print("====================================")

    try:
        demo_basic_queries()
        demo_search_operations()
        demo_statistics_analysis()
        demo_cross_species_analysis()
        demo_performance_benchmarks()

        print("\n" + "="*50)
        print("DEMO COMPLETE")
        print("="*50)
        print("\nThe query interface is ready for use!")
        print("\nTo use in your code:")
        print("  from src.vocalization_query_interface import get_query_interface")
        print("  interface = get_query_interface()")
        print("  results = interface.search_phrases_by_f0_range(5000, 10000)")

    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
