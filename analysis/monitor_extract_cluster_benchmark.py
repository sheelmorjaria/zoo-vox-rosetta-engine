#!/usr/bin/env python3
"""
Monitor extraction → clustering → benchmark pipeline
"""

import subprocess
import time
from pathlib import Path

EXTRACTION_LOG = (
    "/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/extraction_112d/extraction_no_cluster.log"
)
EXTRACTION_RESULT = "/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/extraction_112d/extraction_112d_no_cluster.json"
BENCHMARK_SCRIPT = "/mnt/c/Users/sheel/Desktop/src/analysis/cluster_benchmark_suite.py"
BENCHMARK_RESULT = "/mnt/c/Users/sheel/Desktop/src/analysis/results/cluster_benchmark_112d.json"


def check_file_size(path: str, min_mb: float = 100) -> bool:
    """Check if file exists and meets minimum size."""
    p = Path(path)
    if not p.exists():
        return False
    size_mb = p.stat().st_size / (1024 * 1024)
    return size_mb > min_mb


def get_progress() -> str:
    """Get current extraction progress from log."""
    try:
        with open(EXTRACTION_LOG, "r") as f:
            content = f.read()
        # Find last progress line
        lines = content.split("\n")
        for line in reversed(lines[-20:]):
            if "Processing..." in line and "/91080]" in line:
                return line.strip()
    except:
        pass
    return ""


def is_extraction_complete() -> bool:
    """Check if extraction is complete."""
    return check_file_size(EXTRACTION_RESULT, min_mb=100)


def is_benchmark_complete() -> bool:
    """Check if benchmark is complete."""
    return check_file_size(BENCHMARK_RESULT, min_mb=1)


def run_benchmark():
    """Run clustering benchmark on extracted features."""
    print("\n" + "=" * 70)
    print("Extraction Complete! Running Clustering Benchmark Suite...")
    print("=" * 70)

    # Run benchmark on extracted features (suite does clustering internally)
    result = subprocess.run(
        [
            "python3",
            "-c",
            f"""
import sys
sys.path.insert(0, '.')

# Import and run benchmark
from analysis.cluster_benchmark_suite import ClusterBenchmarkSuite
import json
import numpy as np

# Load extracted features (raw 112D, no pre-clustering)
print("Loading extracted features for benchmark...")
with open('{EXTRACTION_RESULT}', 'r') as f:
    data = json.load(f)

# Sample for benchmark (100k segments max for comparison)
n_samples = min(100000, len(data['segments']))
print(f"Sampling {{n_samples:,}} segments...")

features_list = []
for seg in data['segments'][:n_samples]:
    features_list.append(seg['features_112d'])

features_112d = np.array(features_list, dtype=np.float32)

# Create dummy sequences for N-gram metrics (file order)
sequences = []
current_seq = []
current_file = None

for seg in data['segments'][:n_samples]:
    if current_file != seg['file_name']:
        if current_seq:
            sequences.append(current_seq)
        current_file = seg['file_name']
        current_seq = []
    current_seq.append(0)  # Placeholder - will be replaced by cluster labels

if current_seq:
    sequences.append(current_seq)

# Run benchmark comparing multiple clustering methods
print("\\nRunning clustering comparison: K-Means, UMAP+HDBSCAN, Bayesian GMM...")
suite = ClusterBenchmarkSuite()
results = suite.run(features_112d, sequences, methods=['kmeans', 'umap_hdbscan', 'bgmm'])

# Export results
output_path = '{BENCHMARK_RESULT}'
suite.export_results(results, output_path)
print(f"\\nBenchmark complete! Results saved to {{output_path}}")
""",
        ],
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    return result.returncode == 0


def main():
    """Main monitoring loop."""
    print("╔═══════════════════════════════════════════════════════════════════════════╗")
    print("║     Pipeline Monitor: Extract → Benchmark (with clustering comparison)   ║")
    print("╚═══════════════════════════════════════════════════════════════════════════╝")

    while True:
        # Check extraction
        if is_extraction_complete():
            print("\n✓ Extraction complete!")

            # Check if benchmark already done
            if not is_benchmark_complete():
                if not run_benchmark():
                    print("Benchmark failed!")
                    break

                print("\n" + "=" * 70)
                print("Pipeline Complete!")
                print("=" * 70)
                break
            else:
                print("\n✓ Benchmark already complete!")
                break
        else:
            # Show progress
            progress = get_progress()
            if progress:
                print(f"\r{progress}", end="", flush=True)

        time.sleep(60)  # Check every minute


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
