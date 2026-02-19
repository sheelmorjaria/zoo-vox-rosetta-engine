#!/usr/bin/env python3
"""Analyze feature variance from checkpoint data to understand clustering behavior."""

import json
from pathlib import Path

import numpy as np

# Load checkpoint data
checkpoint_path = Path("/tmp/bat_checkpoint/candidates_checkpoint.json")

print("📂 Loading checkpoint data...")
with open(checkpoint_path, "r") as f:
    data = json.load(f)

# Extract features (data might be a list directly)
candidates = data if isinstance(data, list) else data.get("candidates", [])
print(f"📊 Total candidates: {len(candidates)}")

if not candidates:
    print("❌ No candidates found in checkpoint!")
    exit(1)

# Convert to numpy array
features_list = []
for c in candidates:
    if "features" in c and c["features"]:
        features_list.append(c["features"])

features = np.array(features_list)
print(f"📐 Feature matrix shape: {features.shape}")

# Analyze variance per dimension
print("\n" + "=" * 70)
print("30D FEATURE VARIANCE ANALYSIS")
print("=" * 70)

# Variance per dimension
variances = np.var(features, axis=0)
means = np.mean(features, axis=0)

print("\n📈 Variance Statistics:")
print(f"   Mean variance: {np.mean(variances):.6f}")
print(f"   Median variance: {np.median(variances):.6f}")
print(f"   Min variance: {np.min(variances):.6f}")
print(f"   Max variance: {np.max(variances):.6f}")
print(f"   Std of variances: {np.std(variances):.6f}")

# Dimensions with highest variance
top_indices = np.argsort(variances)[-10:][::-1]
print("\n🔝 Top 10 Dimensions by Variance:")
for i, idx in enumerate(top_indices):
    print(f"   {i + 1:2}. Dimension {idx:2}: var={variances[idx]:.6f}, mean={means[idx]:.6f}")

# Dimensions with lowest variance
bottom_indices = np.argsort(variances)[:10]
print("\n🔻 Bottom 10 Dimensions by Variance:")
for i, idx in enumerate(bottom_indices):
    print(f"   {i + 1:2}. Dimension {idx:2}: var={variances[idx]:.6f}, mean={means[idx]:.6f}")

# Scale analysis (for DBSCAN eps selection)
# Calculate average pairwise distance for a sample
sample_size = min(1000, len(features))
sample_indices = np.random.choice(len(features), sample_size, replace=False)
sample_features = features[sample_indices]

# Calculate distances using a subset for speed
from sklearn.metrics.pairwise import euclidean_distances

print(f"\n📏 Distance Statistics (sample of {sample_size} points):")
# Sample subset for distance calculation
dist_sample_size = min(100, sample_size)
dist_sample = sample_features[:dist_sample_size]
distances = euclidean_distances(dist_sample)

# Get upper triangle (excluding diagonal)
upper_tri = distances[np.triu_indices_from(distances, k=1)]

print(f"   Mean distance: {np.mean(upper_tri):.6f}")
print(f"   Median distance: {np.median(upper_tri):.6f}")
print(f"   Min distance: {np.min(upper_tri):.6f}")
print(f"   Max distance: {np.max(upper_tri):.6f}")
print(f"   10th percentile: {np.percentile(upper_tri, 10):.6f}")
print(f"   25th percentile: {np.percentile(upper_tri, 25):.6f}")
print(f"   75th percentile: {np.percentile(upper_tri, 75):.6f}")
print(f"   90th percentile: {np.percentile(upper_tri, 90):.6f}")

# Feature correlation analysis
print("\n🔗 Feature Correlation Analysis:")
# Sample for correlation
corr_sample_size = min(5000, len(features))
corr_sample = features[:corr_sample_size]
corr_matrix = np.corrcoef(corr_sample.T)

# Get upper triangle of correlation matrix
upper_corr = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]

print(f"   Mean correlation: {np.mean(upper_corr):.6f}")
print(f"   Median correlation: {np.median(upper_corr):.6f}")
print(f"   Min correlation: {np.min(upper_corr):.6f}")
print(f"   Max correlation: {np.max(upper_corr):.6f}")

# High correlations (>0.8)
high_corr_count = np.sum(np.abs(upper_corr) > 0.8)
print(f"   High correlations (>0.8): {high_corr_count} pairs")

# Recommendations
print("\n" + "=" * 70)
print("DBSCAN PARAMETER RECOMMENDATIONS")
print("=" * 70)

# eps recommendation: fraction of median distance
median_dist = np.median(upper_tri)
eps_25 = np.percentile(upper_tri, 25)
eps_10 = np.percentile(upper_tri, 10)

print("\n📊 Current Results:")
print("   Current eps=0.5 -> 205 clusters (mostly noise)")
print("   Target: ~5,833 clusters (Python result)")

print("\n🎯 Suggested DBSCAN Parameters:")
print(f"   eps=0.5 (current): {median_dist:.6f} = median distance")
print(f"   eps=0.35: {eps_25:.6f} = 25th percentile distance")
print(f"   eps=0.25: {eps_10:.6f} = 10th percentile distance")

print("\n💡 Recommendation:")
print("   Try eps=0.30-0.35 with min_samples=10-15")
print("   This should create tighter, more discriminative clusters")

# Check for zero/low variance dimensions
low_variance_dims = np.sum(variances < 0.001)
print("\n⚠️  Low Variance Warning:")
print(f"   Dimensions with variance < 0.001: {low_variance_dims}/30")
if low_variance_dims > 0:
    print("   These dimensions contribute little to clustering!")

print("\n" + "=" * 70)
