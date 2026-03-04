#!/usr/bin/env python3
"""
Test Phrase Typing Ensemble on Marmoset Vocalizations
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

# Configuration
MATCH_THRESHOLD = 0.85
MIN_CLUSTER_SIZE = 2
CLUSTER_THRESHOLD = 0.3

# Paths
wav_dir = Path("test_marmoset_wav")
results_dir = Path("test_marmoset_results")
results_dir.mkdir(exist_ok=True)


class PhraseLibrary:
    """Phrase library for template matching"""

    def __init__(self):
        self.phrases = {}

    def find_best_match(self, features):
        """Find best matching phrase"""
        best_id = None
        best_score = 0.0

        for id, template in self.phrases.items():
            score = cosine_similarity(features, template["features"])
            if score > best_score:
                best_score = score
                best_id = id

        if best_id:
            return best_id, best_score
        return None

    def add_phrase(self, id, features):
        """Add new phrase to library"""
        self.phrases[id] = {"features": features, "count": 1}

    def update_phrase(self, id, features):
        """Update existing phrase with EMA"""
        if id in self.phrases:
            alpha = 0.1
            old = self.phrases[id]["features"]
            updated = old * (1 - alpha) + np.array(features) * alpha
            self.phrases[id]["features"] = updated
            self.phrases[id]["count"] += 1


class PhraseClusterer:
    """Simple clustering for phrase discovery"""

    def __init__(self, min_size=2):
        self.min_size = min_size

    def cluster(self, features_list):
        """Cluster features using distance threshold"""
        n = len(features_list)
        if n == 0:
            return []

        # Compute distance matrix
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = 1.0 - cosine_similarity(features_list[i], features_list[j])
                distances[i, j] = d
                distances[j, i] = d

        # Simple threshold clustering
        labels = [None] * n
        cluster_id = 0

        for i in range(n):
            if labels[i] is not None:
                continue

            # Find neighbors
            neighbors = [i]
            for j in range(n):
                if j != i and distances[i, j] < CLUSTER_THRESHOLD:
                    neighbors.append(j)

            if len(neighbors) >= self.min_size:
                for j in neighbors:
                    labels[j] = cluster_id
                cluster_id += 1

        return labels


def cosine_similarity(a, b):
    """Compute cosine similarity"""
    a = np.array(a)
    b = np.array(b)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a > 0 and norm_b > 0:
        return dot / (norm_a * norm_b)
    return 0.0


def extract_features(audio, sr):
    """Extract simplified 105D features"""
    features = np.zeros(105, dtype=np.float32)

    if len(audio) < 64:
        return features

    # Duration
    features[0] = len(audio) / sr * 1000

    # RMS energy
    features[1] = np.sqrt(np.mean(audio**2))

    # Spectral features
    n_fft = min(2048, len(audio))
    spec = np.abs(np.fft.rfft(audio[:n_fft]))
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)

    # Spectral centroid
    if np.sum(spec) > 0:
        features[2] = np.sum(freqs * spec) / np.sum(spec)

    # ZCR
    features[3] = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))

    # Stats
    features[4] = np.mean(audio)
    features[5] = np.std(audio)

    # Spectral bands
    n_bands = min(20, len(spec) // 10)
    for i in range(n_bands):
        start = i * 10
        end = min((i + 1) * 10, len(spec))
        features[10 + i] = np.mean(spec[start:end])

    return features


def main():
    wav_files = list(wav_dir.glob("*.wav"))
    print("=" * 60)
    print("Testing Phrase Typing Ensemble on Marmoset Vocalizations")
    print("=" * 60)
    print(f"\nProcessing {len(wav_files)} audio files...")

    # Initialize ensemble
    library = PhraseLibrary()
    clusterer = PhraseClusterer(min_size=MIN_CLUSTER_SIZE)

    # Extract features
    segments = []
    for wav_file in wav_files:
        audio, sr = sf.read(str(wav_file))
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        features = extract_features(audio, sr)
        call_type = wav_file.stem.split("_")[0]

        segments.append(
            {
                "id": wav_file.stem,
                "features": features,
                "type": call_type,
                "duration_ms": len(audio) / sr * 1000,
            }
        )

    print(f"Extracted features from {len(segments)} segments")

    # Stage A: Template Matching
    print(f"\nStage A: Template Matching (threshold={MATCH_THRESHOLD})")
    matched = []
    pending = []

    for seg in segments:
        match = library.find_best_match(seg["features"])
        if match and match[1] > MATCH_THRESHOLD:
            matched.append(
                {
                    "id": seg["id"],
                    "label": f"known:{match[0]}",
                    "type": seg["type"],
                    "score": match[1],
                }
            )
            library.update_phrase(match[0], seg["features"])
        else:
            pending.append(seg)

    print(f"  Matched: {len(matched)}")

    # Stage B: Clustering (Discovery)
    print(f"\nStage B: Clustering (threshold={CLUSTER_THRESHOLD})")
    discovered = []
    noise = []

    if pending:
        features_list = [s["features"] for s in pending]
        labels = clusterer.cluster(features_list)

        for i, seg in enumerate(pending):
            if labels[i] is not None:
                phrase_id = f"phrase_{labels[i]}"
                library.add_phrase(phrase_id, seg["features"])
                discovered.append(
                    {"id": seg["id"], "label": f"discovered:{phrase_id}", "type": seg["type"]}
                )
            else:
                noise.append({"id": seg["id"], "label": "noise", "type": seg["type"]})

    print(f"  Discovered: {len(discovered)}")
    print(f"  Noise: {len(noise)}")

    # Combine results
    results = matched + discovered + noise

    # Save results
    results_file = results_dir / "phrase_labels.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print("\nClassification Results:")
    print(f"  Known (Stable):     {len(matched):3d}")
    print(f"  Discovered (New):   {len(discovered):3d}")
    print(f"  Noise (Rejected):   {len(noise):3d}")
    print(f"  Library Phrases:    {len(library.phrases):3d}")

    print("\nBy Call Type:")
    by_type = defaultdict(lambda: {"known": 0, "discovered": 0, "noise": 0})
    for r in results:
        if r["label"].startswith("known:"):
            by_type[r["type"]]["known"] += 1
        elif r["label"].startswith("discovered:"):
            by_type[r["type"]]["discovered"] += 1
        else:
            by_type[r["type"]]["noise"] += 1

    for call_type, counts in sorted(by_type.items()):
        total = sum(counts.values())
        print(
            f"  {call_type:15s}: known={counts['known']:2d} "
            f"discovered={counts['discovered']:2d} "
            f"noise={counts['noise']:2d} (n={total})"
        )

    print("\nDiscovered Phrases:")
    phrase_types = defaultdict(list)
    for r in results:
        if r["label"].startswith("discovered:"):
            phrase_types[r["label"]].append(r["type"])

    for phrase, types in sorted(phrase_types.items()):
        print(f"  {phrase}: {types}")

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
