#!/usr/bin/env python3
"""
Monitor 112D extraction and automatically proceed with PCFG analysis when complete.
"""

import json
import time
from pathlib import Path

EXTRACTION_LOG = (
    "/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/extraction_112d/extraction.log"
)
EXTRACTION_RESULT = "/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/extraction_112d/extraction_112d_results.json"
PROGRESS_FILE = "/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/extraction_112d/progress.txt"


def check_progress():
    """Check extraction progress from log."""
    try:
        with open(EXTRACTION_LOG, "r") as f:
            lines = f.readlines()

        for line in reversed(lines[-50:]):
            if "Processing" in line and "/91080]" in line:
                # Extract progress: [XXXXX/91080]
                parts = line.split("[")[1].split("]")
                current = int(parts[0].split("/")[0])
                return current, 91080
        return 0, 91080
    except:
        return 0, 91080


def is_complete():
    """Check if extraction is complete."""
    result_path = Path(EXTRACTION_RESULT)
    if result_path.exists():
        try:
            with open(result_path, "r") as f:
                data = json.load(f)
            return data.get("total_segments", 0) > 0
        except:
            return False
    return False


def run_pcfg_analysis():
    """Run PCFG analysis on 112D extraction results."""
    print("\n" + "=" * 70)
    print("112D Extraction Complete! Running PCFG Analysis...")
    print("=" * 70)

    # Check results
    with open(EXTRACTION_RESULT, "r") as f:
        data = json.load(f)

    print("\nExtraction Summary:")
    print(f"  Total files: {data['total_files']}")
    print(f"  Total segments: {data['total_segments']}")
    print(f"  Feature dimension: {data['feature_dimension']}")
    print(f"  Clusters found: {data['cluster_count']}")
    print(f"  Noise points: {data['noise_count']}")

    # Next: Generate sequences by cluster for PCFG analysis
    print("\nGenerating sequences by cluster for PCFG analysis...")

    # Group segments by cluster
    cluster_segments = {}
    for segment in data["segments"]:
        cluster_id = segment.get("cluster_id", -1)
        if cluster_id not in cluster_segments:
            cluster_segments[cluster_id] = []
        cluster_segments[cluster_id].append(segment)

    # Create sequential data from file order
    sequences_by_context = {}

    # Group by file and create sequences
    file_sequences = {}
    for segment in data["segments"]:
        file_name = segment["file_name"]
        cluster_id = segment.get("cluster_id", -1)

        if file_name not in file_sequences:
            file_sequences[file_name] = []
        file_sequences[file_name].append((segment["start_sample"], cluster_id))

    # Sort by start_sample and extract sequences
    for file_name, segments in file_sequences.items():
        segments.sort()  # Sort by start_sample
        # Create context from file name
        context_id = f"context_{file_name}"

        if context_id not in sequences_by_context:
            sequences_by_context[context_id] = []

        # Extract sequence (filter noise)
        sequence = [cluster_id for _, cluster_id in segments if cluster_id >= 0]
        if len(sequence) > 1:
            sequences_by_context[context_id].append(sequence)

    # Save sequences for PCFG analysis
    output_path = "/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/extraction_112d/sequences_by_context.json"
    with open(output_path, "w") as f:
        json.dump(sequences_by_context, f, indent=2)

    print(
        f"  Saved {len(sequences_by_context)} contexts with {sum(len(s) for s in sequences_by_context.values())} total sequences"
    )
    print(f"  Output: {output_path}")

    print("\n" + "=" * 70)
    print("Next Step: Run PCFG analysis on 112D-based sequences")
    print("  python3 src/analysis/bat_pcfg_syntax_analysis_112d.py")
    print("=" * 70)


def main():
    print("Monitoring 112D extraction progress...")
    print("Press Ctrl+C to stop monitoring\n")

    last_progress = 0
    while True:
        current, total = check_progress()
        progress_pct = (current / total) * 100

        if current != last_progress:
            print(f"\rProgress: {current:,}/{total:,} ({progress_pct:.1f}%)", end="", flush=True)
            last_progress = current

        if is_complete():
            print("\n\nExtraction complete!")
            run_pcfg_analysis()
            break

        time.sleep(30)  # Check every 30 seconds


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
