#!/usr/bin/env python3
"""
Monitor extraction completion and automatically run clustering.
"""

import subprocess
import time
from pathlib import Path

EXTRACTION_LOG = (
    "/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/extraction_112d/extraction_no_cluster.log"
)
EXTRACTION_RESULT = "/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/extraction_112d/extraction_112d_no_cluster.json"
CLUSTER_SCRIPT = "/mnt/c/Users/sheel/Desktop/src/analysis/graded_cluster_112d.py"


def is_complete():
    """Check if extraction is complete."""
    result_path = Path(EXTRACTION_RESULT)
    if result_path.exists():
        # Check file size is reasonable (should be several GB)
        size_mb = result_path.stat().st_size / (1024 * 1024)
        return size_mb > 100  # At least 100 MB
    return False


def check_progress():
    """Check extraction progress from log."""
    try:
        with open(EXTRACTION_LOG, "r") as f:
            content = f.read()

        # Look for progress indicators
        if "Processing..." in content:
            lines = content.split("\n")
            for line in reversed(lines[-50:]):
                if "[1000/91080]" in line or "[5000/91080]" in line or "[10000/91080]" in line:
                    return line.strip()
        return None
    except:
        return None


def run_clustering():
    """Run clustering on extracted features."""
    print("\n" + "=" * 70)
    print("Extraction Complete! Running Clustering...")
    print("=" * 70)

    result = subprocess.run(["python3", CLUSTER_SCRIPT], capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    print("\n" + "=" * 70)
    print("Clustering Complete!")
    print("=" * 70)


def main():
    """Main monitoring loop."""
    print("Monitoring extraction progress...")
    print("Press Ctrl+C to stop\n")

    while True:
        if is_complete():
            print("\nExtraction complete!")
            run_clustering()
            break

        progress = check_progress()
        if progress:
            print(f"\r{progress}", end="", flush=True)

        time.sleep(30)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
