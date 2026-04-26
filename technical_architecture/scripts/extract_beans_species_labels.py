#!/usr/bin/env python3
"""Extract proper species labels from BEANS-Zero manifest.

The BEANS-Zero manifest has species information in the 'output' field,
but it needs to be extracted properly for each task type.
"""

import json
from collections import Counter, defaultdict


def extract_species(sample: dict) -> str:
    """Extract species label from a BEANS-Zero sample."""
    labels = sample.get("labels", {})
    output = labels.get("output")
    task = labels.get("task", "unknown")

    if output and output != "None" and output.strip():
        # Clean up the output
        output = output.strip()

        # Handle multi-species labels (comma-separated)
        if "," in output:
            # Take the first species (primary)
            species = output.split(",")[0].strip()
        else:
            species = output

        # Handle long caption-style descriptions
        if len(species) > 50:
            # Extract species from captions like "The sound of a Yellow-rumped Warbler..."
            if "sound of" in species.lower():
                # Extract from "The sound of a Yellow-rumped Warbler in a Chinese tallow tree."
                parts = species.lower().split("sound of")
                if len(parts) > 1:
                    species_part = parts[1].strip().split()[0:4]  # Take a few words
                    species = " ".join(species_part).title()

        # Handle taxonomic labels (long scientific names)
        if "Chordata" in species or "Arthropoda" in species:
            # Extract species from taxonomic path
            parts = species.split()
            if len(parts) >= 2:
                # Take genus species (last two words)
                species = parts[-2] + " " + parts[-1]

        return species

    # Fallback to task name if no output
    return task


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract species labels from BEANS-Zero")
    parser.add_argument("--input", default="beans_zero_full_manifest.json", help="Input manifest")
    parser.add_argument(
        "--output",
        default="beans_zero_species_manifest.json",
        help="Output manifest with species labels",
    )
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    with open(args.input) as f:
        manifest = json.load(f)

    samples = manifest.get("samples", [])
    print(f"Total samples: {len(samples)}")

    # Extract species for all samples
    species_counts = Counter()
    task_species = defaultdict(Counter)

    for sample in samples:
        species = extract_species(sample)
        sample["labels"]["species"] = species
        species_counts[species] += 1
        task = sample["labels"].get("task", "unknown")
        task_species[task][species] += 1

    print(f"\nTotal unique species: {len(species_counts)}")

    print("\n=== Species by task ===")
    for task in sorted(task_species.keys()):
        species_list = task_species[task]
        print(f"\n{task} ({len(species_list)} species, {sum(species_list.values())} samples):")
        for species, count in species_list.most_common(5):
            print(f"  {species[:50]}: {count}")
        if len(species_list) > 5:
            print(f"  ... and {len(species_list) - 5} more species")

    # Save updated manifest
    print(f"\nSaving to {args.output}...")
    with open(args.output, "w") as f:
        json.dump(manifest, f)

    # Create a subset manifest with only samples that have actual species labels
    # (not task names as fallback)
    labeled_samples = [s for s in samples if s["labels"].get("species") != s["labels"].get("task")]

    print(f"\nSamples with actual species labels: {len(labeled_samples)}")

    # Save subset manifest
    subset_manifest = {
        "dataset": manifest.get("dataset", "BEANS-Zero-Species"),
        "n_samples": len(labeled_samples),
        "samples": labeled_samples,
    }

    subset_path = args.output.replace(".json", "_subset.json")
    with open(subset_path, "w") as f:
        json.dump(subset_manifest, f)
    print(f"Saved subset to {subset_path}")


if __name__ == "__main__":
    main()
