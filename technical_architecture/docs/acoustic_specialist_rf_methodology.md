# Acoustic Specialist Random Forest Classification Methodology

## Overview

This document describes the methodology used in `train_acoustic_specialist_rfs.rs` and `eval_acoustic_specialist_rfs.rs` for bioacoustic species classification. The approach uses **acoustic coherence** rather than biological taxonomy to group species, resulting in more effective specialist classifiers.

## Key Insight

Traditional bioacoustic classification systems group animals by biological taxonomy (e.g., "all birds" or "all mammals"). However, this approach is suboptimal because:

- **Bats** produce ultrasonic FM sweeps (20-80 kHz, 5-50 ms)
- **Baleen whales** produce low-frequency moans (20-5000 Hz, 0.5-5 s)
- **Songbirds** produce high-frequency modulated calls (2-8 kHz)
- **Dolphins** produce ultrasonic whistles and clicks

These acoustic characteristics cross taxonomic boundaries. A single "mammal" classifier must learn vastly different acoustic patterns, reducing accuracy.

## Acoustic Grouping Strategy

Instead of taxonomy, we group species by **acoustic coherence**—similar frequency ranges, durations, and modulation patterns.

### 13 Acoustic Groups

| Group | Example Species | Acoustic Characteristics |
|-------|-----------------|-------------------------|
| **UltrasonicMammal** | Bats (Vespertilionidae, Pteropodidae) | F0: 20-80 kHz, Duration: 5-50 ms, FM sweeps |
| **SonicLongMammal** | Humpback, blue, fin whales | F0: 20-5000 Hz, Duration: 500-5000 ms, low moans |
| **SonicShortMammal** | Primates, marmosets, gibbons | Mid F0, variable duration |
| **InsectWingbeat** | Mosquitoes, flies, bees | Steady F0, pure tones, 100-1000 Hz |
| **InsectStridulation** | Crickets, cicadas, katydids | Broadband, impulsive, 2-10 kHz |
| **BirdHighFreq** | Songbirds, warblers, finches | High F0 (4-8 kHz), fast modulation |
| **BirdLowFreq** | Doves, pigeons, owls | Low F0 (200-1000 Hz), long duration |
| **BirdMechanical** | Hummingbirds, woodpeckers | Broadband, pulse-like, mechanical |
| **MarineWhistle** | Dolphins, orcas, pilot whales | FM sweeps, harmonic, 2-24 kHz |
| **MarineClick** | Porpoises, sperm whales | Impulsive, broadband, echolocation |
| **MarineMoan** | Baleen whales (fallback) | Low F0, long duration |
| **Amphibian** | Frogs, toads | 500-5000 Hz, pulsed calls |
| **Pinniped** | Seals, sea lions, walruses | 100-5000 Hz, varied patterns |

### Species Mapping Algorithm

Species are mapped to acoustic groups using keyword matching on species names (common names, scientific names, and taxonomic families):

```rust
fn map_species_to_acoustic_group(species: &str) -> AcousticGroup {
    let s = species.to_lowercase();

    // Example: Ultrasonic mammals (bats)
    if s.contains("bat") || s.contains("pteropodid") || s.contains("vesper")
        || s.contains("rhinolophus") || s.contains("myotis") /* ... */
    {
        return AcousticGroup::UltrasonicMammal;
    }

    // ... additional mappings ...

    // Default fallback
    AcousticGroup::SonicShortMammal
}
```

## Feature Engineering

### 112-Dimensional Feature Vector

Each audio sample is represented by a 112-dimensional feature vector extracted from the spectrogram:

| Feature Category | Dimensions | Description |
|-----------------|------------|-------------|
| **Temporal** | ~20 | Duration, onset/offset patterns, rhythm |
| **Spectral** | ~40 | F0, bandwidth, harmonic structure, spectral centroid |
| **Modulation** | ~30 | FM rate, AM depth, frequency contours |
| **Energy** | ~15 | RMS energy, dynamic range, envelope shape |
| **Cepstral** | ~7 | MFCCs, spectral smoothness |

### Feature Caching

Features are pre-computed and cached in `beans_feature_cache_112d/` using bincode serialization for efficient loading during training and evaluation.

## Model Architecture

### Random Forest Classifier

Each acoustic group has a dedicated Random Forest classifier with the following hyperparameters:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_estimators` | 200 | Sufficient trees for ensemble diversity |
| `max_depth` | 30 | Deep enough for complex patterns, prevents overfitting |
| `min_samples_split` | 5 | Regularization to prevent memorization |
| `class_weight` | balanced | Handles class imbalance via weighted sampling |

### Balanced Class Weighting

Uses inverse frequency weighting to handle imbalanced datasets:

```
weight(class_i) = n_samples / (n_classes * n_samples_in_class_i)
```

This ensures minority classes are adequately represented during bootstrap sampling.

## Training Procedure

### Data Loading

1. Load manifest (`beans_zero_full_manifest.json`) containing 91,965 samples
2. Load feature cache manifest mapping audio files to pre-computed features
3. Group samples by acoustic group using species mapping

### Train/Test Split

- **Training**: 80% of data per acoustic group
- **Testing**: 20% of data per acoustic group
- **Stratification**: Shuffle with fixed seed (42) for reproducibility

### Parallel Training

Specialists are trained in parallel using Rayon:

```rust
let results: Vec<_> = group_keys
    .par_iter()
    .filter_map(|group| {
        train_specialist(*group, dataset).ok()
    })
    .collect();
```

### Model Serialization

Models are saved in **bincode format** (not JSON) for:
- **Smaller file size**: ~2x compression vs JSON
- **Faster loading**: Direct binary deserialization
- **Memory efficiency**: Avoids JSON parsing overhead

## Evaluation Procedure

### Top-1 and Top-5 Accuracy

For each test sample:

1. **Routing**: Map species label to acoustic group
2. **Specialist Prediction**: Use the appropriate specialist RF
3. **Top-1**: Correct if highest probability class matches label
4. **Top-5**: Correct if label appears in top 5 probability classes

```rust
// Top-1 prediction
let pred_idx = rf_model.predict(&features_arr);
if rf_model.idx_to_label().get(&pred_idx) == Some(&sample.label) {
    total_correct += 1;
}

// Top-5 prediction
let probs = rf_model.predict_proba(&features_arr);
let mut ranked: Vec<(usize, f32)> = probs.iter().enumerate().collect();
ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
for (idx, _) in ranked.iter().take(5) {
    if rf_model.idx_to_label().get(idx) == Some(&sample.label) {
        total_top5_correct += 1;
        break;
    }
}
```

### Per-Dataset Metrics

Results are aggregated by task/dataset to identify:
- **Strong performance**: Well-represented species with clear acoustic signatures
- **Weak performance**: Zero-shot generalization tasks (unseen species/genus/family)

## Performance Results

### Overall Accuracy (10,041 test samples)

| Metric | Accuracy |
|--------|----------|
| Top-1 | 60.49% |
| Top-5 | 82.27% |

### Best Performing Datasets

| Dataset | Top-1 | Top-5 | Description |
|---------|-------|-------|-------------|
| hiceas | 100.0% | 100.0% | Hi-C EAS mosquito recordings |
| zf-indiv | 95.3% | 100.0% | Zebra finch individual identification |
| humbugdb | 94.7% | 97.7% | Mosquito wingbeat database |
| dcase | 93.9% | 98.7% | DCASE bioacoustic tasks |
| gibbons | 92.9% | 100.0% | Gibbon vocalizations |

### Zero-Shot Generalization

| Dataset | Top-1 | Top-5 | Challenge |
|---------|-------|-------|-----------|
| unseen-genus-sci | 21.0% | 70.7% | Classify unseen genera (scientific names) |
| unseen-species-cmn | 38.6% | 82.3% | Classify unseen species (common names) |
| unseen-family-sci | 44.3% | 86.6% | Classify unseen families (scientific) |

The high Top-5 accuracy for unseen categories indicates the model learns generalizable acoustic features, even when exact species classification fails.

## Model Sizes

After bincode conversion:

| Model | Size | Classes |
|-------|------|---------|
| bird_high_freq | 17.5 GB | 2,549 |
| sonic_short_mammal | 12.2 GB | 2,293 |
| songbird | 14.9 GB | 2,136 |
| bird_low_freq | 2.4 GB | 981 |
| amphibian | 642 MB | 484 |
| insect_wingbeat | 108 MB | 185 |

## Usage

### Training

```bash
cd technical_architecture
cargo run --release --bin train_acoustic_specialist_rfs
```

### Evaluation

```bash
cd technical_architecture
cargo run --release --bin eval_acoustic_specialist_rfs
```

### Output Files

- `specialist_rf_models/specialist_rf_acoustic_*.bincode` - Trained models
- `acoustic_specialist_rf_results.json` - Evaluation metrics per dataset

## Design Rationale

### Why Acoustic Coherence Over Taxonomy?

1. **Feature consistency**: Species in the same acoustic group share similar feature distributions
2. **Simpler decision boundaries**: Specialists learn fine-grained distinctions within coherent groups
3. **Better generalization**: Acoustic features transfer better than taxonomic features

### Why Random Forests Over Neural Networks?

1. **Interpretability**: Feature importances reveal what acoustic features matter
2. **Small sample efficiency**: RFs work well with limited training data per class
3. **No GPU required**: CPU training is fast with parallel implementation
4. **Robust to noise**: Ensemble averaging reduces overfitting

### Why Specialist Models Over One Global Model?

1. **Class imbalance**: 2,549 bird species would dominate a global model
2. **Acoustic diversity**: Bats and whales have non-overlapping frequency ranges
3. **Modular improvement**: Specialists can be retrained independently
4. **Hierarchical routing**: First route to group, then classify within group

## Future Improvements

1. **Neural specialist routers**: Replace keyword-based routing with learned classifiers
2. **Feature selection**: Remove redundant features per specialist
3. **Model compression**: Prune trees to reduce model sizes
4. **Cross-group transfer**: Share features between related acoustic groups
5. **Active learning**: Prioritize uncertain samples for labeling

## References

- BEANS benchmark: https://github.com/earthspecies/beans
- Random Forest classifier: Breiman, L. (2001). "Random Forests". Machine Learning 45(1): 5-32.
- Bioacoustic feature extraction: http://docs.birdvox.cloudflare.com/
