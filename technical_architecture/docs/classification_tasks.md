# Classification Tasks Architecture

## Overview

The `HierarchicalEnsembleRouter` implements a **two-stage hierarchical classification system** for bioacoustic species identification. It solves the "Resolution Paradox" by using coarse features for taxonomy detection and fine features for species discrimination.

## Architecture Diagram

```
        INPUT: 112D Feature Vector
             │
             ▼
    ┌────────────────────────────────────┐
    │     STAGE 1: GROUP DETECTION       │
    │      (Taxonomy / Context)          │
    ├────────────────────────────────────┤
    │  ┌──────────────────┐ ┌───────────┐│
    │  │ RF Gatekeeper    │ │ NN Block 1││
    │  │ (Physics 76D)    │ │ (Physics) ││
    │  └────────┬─────────┘ └─────┬─────┘│
    │           │                 │      │
    │           └───────┬─────────┘      │
    │                   ▼                │
    │          [Ensemble Voter]          │
    │           (5% NN / 95% RF)         │
    └───────────────┬────────────────────┘
                    │
            PREDICTION: "Bat"
                    │
       ┌────────────┴────────────┐
       │     FEATURE REWEIGHTING │
       │  (Boost FM/ICI for Bat) │
       └────────────┬────────────┘
                    │
                    ▼
    ┌────────────────────────────────────┐
    │   STAGE 2: SPECIES DISCRIMINATION  │
    │        (Specialist Models)         │
    ├────────────────────────────────────┤
    │  ┌──────────────────┐ ┌───────────┐│
    │  │ RF Specialist    │ │ NN Block 2││
    │  │ (Bat Expert)     │ │ (Unfreeze)││
    │  └────────┬─────────┘ └─────┬─────┘│
    │           │                 │      │
    │           └───────┬─────────┘      │
    │                   ▼                │
    │          [Ensemble Voter]          │
    │           (50% NN / 50% RF)        │
    └───────────────┬────────────────────┘
                    │
                    ▼
            FINAL PREDICTION
          "Species: Bat #42"
```

---

## Feature Stack Architecture (112D)

The input is a 112-dimensional feature vector organized in three layers:

| Layer | Indices | Dimension | Description |
|-------|---------|-----------|-------------|
| **Layer 1: Base Physics** | 0-45 | 46D | F0, Duration, RMS Energy, MFCCs, Spectral features |
| **Layer 2: Macro Texture** | 46-75 | 30D | Harmonic Density, Pitch Geometry, GLCM Roughness |
| **Layer 3: Micro Texture** | 76-111 | 36D | FM Bins, ICI Bins, Dynamics, Rhythm patterns |

### Divide and Conquer Inputs

| Input Type | Composition | Dimension | Purpose |
|------------|-------------|-----------|---------|
| **Gatekeeper Input** | Base Physics + Macro Texture | 76D (46 + 30) | Stage 1: Taxonomy detection |
| **Species Expert Input** | Base Physics + Micro Texture | 82D (46 + 36) | Stage 2: Species discrimination |

---

## Taxonomic Groups

### Detailed Taxon (8 classes)

Used by specialist RF models for fine-grained classification:

```rust
enum Taxon {
    Cetacean,      // Toothed whales (dolphins, porpoises) - clicks and whistles
    Mysticete,     // Baleen whales (humpback, blue) - songs and moans
    Songbird,      // Songbirds (passerines) - complex syntax
    NonPasserine,  // Non-passerine birds (parrots, owls) - simple calls
    Amphibian,     // Frogs and toads - pulse trains and trills
    Pinniped,      // Seals and sea lions - grunts and barks
    Insect,        // Insects - rigid tempo patterns
    Mammal,        // General mammals (bats, primates) - FM sweeps and formants
    Unknown,       // Unclassified
}
```

### Consolidated Taxon (6 classes)

Used by Gatekeeper RF for coarse taxonomy (groups rare classes for better accuracy):

```rust
enum ConsolidatedTaxon {
    Bird,          // Songbird + NonPasserine
    Mammal,        // Terrestrial mammals (bats, primates)
    MarineMammal,  // Cetacean + Mysticete + Pinniped
    Insect,        // Crickets, mosquitoes, cicadas
    Amphibian,     // Frogs, toads
    Unknown,       // Unclassified
}
```

---

## Stage 1: Group Detection

**Purpose**: Determine the broad taxonomic group using coarse features.

### Components

1. **RF Gatekeeper** (95% weight)
   - Input: 76D (Physics + Macro Texture)
   - Output: Probability distribution over 6 consolidated taxonomic classes
   - High accuracy for coarse taxonomy using physics features

2. **NN Block 1** (5% weight)
   - Input: Full 112D features (Physics block only active initially)
   - Output: Group probability distribution
   - Curriculum learning: starts with physics, progressively adds texture

### Ensemble Voting

```rust
// Default weights
STAGE1_RF_WEIGHT: 0.95  // 95% trust in RF for taxonomy
STAGE1_NN_WEIGHT: 0.05  // 5% NN contribution

// Combined probability
combined_proba[i] = (0.95 * rf_proba[i]) + (0.05 * nn_proba[i])
```

### Confidence Threshold

```rust
MIN_STAGE1_CONFIDENCE: 0.3  // 30% minimum to proceed to Stage 2
```

---

## Feature Reweighting

Between Stage 1 and Stage 2, features are reweighted based on the predicted taxonomic group. This applies **biologically-guided attention** to emphasize discriminative features.

### Taxonomic Weight Patterns

| Taxon | Emphasized Features | Weight |
|-------|---------------------|--------|
| **Cetacean** | ICI Bins (inter-click interval) | 3.0x |
| | FM Bins (frequency modulation) | 2.5x |
| | Spectral Centroid | 2.0x |
| **Mysticete** | Duration, Harmonicity | 2.5x |
| | Low F0 features | 2.0x |
| **Songbird** | Spectral Derivatives | 2.0x |
| | Rhythm features | 2.5x |
| **Bat/Mammal** | FM Bins (sweep rate) | 3.0x |
| | Dynamics bins | 2.0x |
| **Insect** | Rhythm features | 3.5x |
| | Pulse rate | 2.5x |
| **Amphibian** | Pulse train features | 2.5x |

### Example: Bat Feature Reweighting

```rust
Taxon::Mammal => {
    // FM slope critical for bat echolocation
    for i in FM_BINS_START..FM_BINS_END {
        weights[i] = 3.0;
    }
    // Amplitude dynamics
    for i in DYNAMICS_BINS_START..DYNAMICS_BINS_END {
        weights[i] = 2.0;
    }
}
```

---

## Stage 2: Species Discrimination

**Purpose**: Fine-grained species identification within the predicted taxonomic group.

### Specialist RF Registry

Each taxonomic group has a dedicated Random Forest specialist:

| Specialist | Coverage | Training Data |
|------------|----------|---------------|
| `rf_cetacean` | Toothed whales | Only cetacean samples |
| `rf_mysticete` | Baleen whales | Only mysticete samples |
| `rf_songbird` | Passerines | Only songbird samples |
| `rf_non_passerine` | Other birds | Only non-passerine samples |
| `rf_amphibian` | Frogs/toads | Only amphibian samples |
| `rf_pinniped` | Seals/sea lions | Only pinniped samples |
| `rf_insect` | Insects | Only insect samples |
| `rf_mammal` | Bats/primates | Only mammal samples |
| `rf_fallback` | All species | Full dataset (fallback) |

### Specialist Selection Logic

```rust
fn get_best_specialist_for_consolidated(&self, taxon: ConsolidatedTaxon) -> Option<&RFModel> {
    match taxon {
        ConsolidatedTaxon::Bird => {
            // Try Songbird first (most common), then NonPasserine
            self.rf_songbird.as_ref()
                .or(self.rf_non_passerine.as_ref())
        }
        ConsolidatedTaxon::MarineMammal => {
            // Try Cetacean first, then Mysticete, then Pinniped
            self.rf_cetacean.as_ref()
                .or(self.rf_mysticete.as_ref())
                .or(self.rf_pinniped.as_ref())
        }
        ConsolidatedTaxon::Mammal => self.rf_mammal.as_ref(),
        // ...
    }
}
```

### Ensemble Voting (Stage 2)

```rust
// Default weights - balanced for species discrimination
STAGE2_RF_WEIGHT: 0.50  // 50% specialist RF
STAGE2_NN_WEIGHT: 0.50  // 50% NN contribution

// Agreement boost
if rf_pred == nn_pred {
    combined_conf = (rf_conf + nn_conf) / 2.0 * 1.1;  // 10% boost
}
```

---

## Configuration Options

### RouterConfig

```rust
pub struct RouterConfig {
    // Stage 1 weights (must sum to 1.0)
    pub stage1_rf_weight: f32,      // Default: 0.95
    pub stage1_nn_weight: f32,      // Default: 0.05

    // Stage 2 weights (must sum to 1.0)
    pub stage2_rf_weight: f32,      // Default: 0.50
    pub stage2_nn_weight: f32,      // Default: 0.50

    // Thresholds
    pub min_stage1_confidence: f32, // Default: 0.30 (30%)
    pub max_candidates: usize,      // Default: 10

    // Feature processing
    pub apply_feature_reweighting: bool,  // Default: true
    pub enable_nn: bool,                  // Default: true

    // Fallback
    pub fallback_taxon: ConsolidatedTaxon, // Default: Unknown
}
```

### Preset Configurations

```rust
// RF-only mode (no neural network)
RouterConfig::rf_only()      // stage1: 100% RF, stage2: 100% RF

// NN-only mode (no random forest)
RouterConfig::nn_only()      // stage1: 100% NN, stage2: 100% NN

// Balanced mode
RouterConfig::balanced()     // stage1: 50/50, stage2: 50/50

// Default (optimized for taxonomy)
RouterConfig::default()      // stage1: 95/5, stage2: 50/50
```

---

## Result Types

### Stage1Result

```rust
pub struct Stage1Result {
    pub predicted_group: ConsolidatedTaxon,  // Final group prediction
    pub confidence: f32,                      // Combined confidence (0.0-1.0)
    pub rf_prediction: ConsolidatedTaxon,     // RF's prediction
    pub rf_confidence: f32,                   // RF's confidence
    pub rf_proba: Vec<f32>,                   // RF probability distribution
    pub nn_prediction: Option<ConsolidatedTaxon>, // NN's prediction (if enabled)
    pub nn_confidence: Option<f32>,           // NN's confidence (if enabled)
    pub nn_proba: Option<Vec<f32>>,           // NN probability distribution
    pub rf_weight: f32,                       // Effective RF weight used
    pub nn_weight: f32,                       // Effective NN weight used
}
```

### Stage2Result

```rust
pub struct Stage2Result {
    pub species: String,                      // Final species prediction
    pub confidence: f32,                      // Species confidence (0.0-1.0)
    pub taxon: Taxon,                         // Detailed taxonomic group
    pub candidates: Vec<SpeciesCandidate>,    // Top-N candidates with scores
    pub rf_prediction: String,                // Specialist RF's prediction
    pub rf_confidence: f32,                   // RF confidence
    pub nn_prediction: Option<String>,        // NN's prediction (if enabled)
    pub nn_confidence: Option<f32>,           // NN confidence
    pub reweighting_applied: bool,            // Whether features were reweighted
}
```

### RouterResult

```rust
pub struct RouterResult {
    pub species: String,                      // Final species label
    pub confidence: f32,                      // Final confidence (0.0-1.0)
    pub predicted_group: ConsolidatedTaxon,   // Stage 1 group
    pub detailed_taxon: Taxon,                // Detailed taxon from species
    pub stage1: Stage1Result,                 // Full Stage 1 details
    pub stage2: Stage2Result,                 // Full Stage 2 details
    pub processing_time_us: u64,              // Total processing time
    pub is_reliable: bool,                    // Reliability flag
    pub warnings: Vec<String>,                // Warning messages
}
```

### Reliability Criteria

```rust
is_reliable = (stage1.confidence >= 0.30) && (stage2.confidence >= 0.30)
```

---

## Usage Examples

### Basic Classification

```rust
use technical_architecture::hierarchical_ensemble_router::{
    HierarchicalEnsembleRouter, RouterConfig, RouterResult
};

// Create router with default configuration
let mut router = HierarchicalEnsembleRouter::new()?;

// Load models (see Model Loading section below)
router.load_gatekeeper(gatekeeper_rf);
router.load_specialist(Taxon::Mammal, bat_specialist_rf);
router.load_nn(Box::new(curriculum_nn));

// Classify a sample
let features = vec![0.0; 112]; // 112D feature vector
let result = router.classify(&features)?;

println!("Species: {}", result.species);
println!("Confidence: {:.2}%", result.confidence * 100.0);
println!("Group: {:?}", result.predicted_group);
println!("Reliable: {}", result.is_reliable);
```

### Batch Classification

```rust
let features_batch: Vec<Vec<f32>> = vec![
    sample1_features,
    sample2_features,
    sample3_features,
];

let results = router.classify_batch(&features_batch);
for (i, result) in results.iter().enumerate() {
    if let Ok(r) = result {
        println!("Sample {}: {} ({:.1}%)", i, r.species, r.confidence * 100.0);
    }
}
```

### Evaluation

```rust
let metrics = router.evaluate(&test_features, &test_labels);

println!("Species Accuracy: {:.2}%", metrics.species_accuracy);
println!("Group Accuracy: {:.2}%", metrics.group_accuracy);
println!("Reliable Samples: {}/{}", metrics.reliable_samples, metrics.total_samples);
println!("Reliable Accuracy: {:.2}%", metrics.reliable_accuracy);
println!("RF/NN Agreement: {:.2}%", metrics.agreement_rate);
```

---

## Model Loading

### Loading RF Gatekeeper

```rust
use technical_architecture::rf_stacking_ensemble::RFModel;

// Load from JSON file
let gatekeeper_rf = RFModel::load_from_json("gatekeeper_rf.json")?;
router.load_gatekeeper(gatekeeper_rf);
```

### Loading Specialist RFs

```rust
// Load specialists for each taxonomic group
let bat_specialist = RFModel::load_from_json("specialists/bat_rf.json")?;
let cetacean_specialist = RFModel::load_from_json("specialists/cetacean_rf.json")?;
let songbird_specialist = RFModel::load_from_json("specialists/songbird_rf.json")?;

router.load_specialist(Taxon::Mammal, bat_specialist);
router.load_specialist(Taxon::Cetacean, cetacean_specialist);
router.load_specialist(Taxon::Songbird, songbird_specialist);
```

### Loading NN Model

```rust
// Implement NeuralNetworkModel trait for your NN
impl NeuralNetworkModel for CurriculumNN {
    fn predict(&self, features: &[f32]) -> (usize, String) { ... }
    fn predict_proba(&self, features: &[f32]) -> Vec<f32> { ... }
    fn predict_group_proba(&self, features: &[f32]) -> Vec<f32> { ... }
    fn class_labels(&self) -> &[String] { ... }
    fn n_classes(&self) -> usize { ... }
}

router.load_nn(Box::new(curriculum_nn));
```

---

## Key Concepts

### 1. Resolution Paradox

Using high-resolution micro-texture features for coarse taxonomy (distinguishing "Mouse" from "Whale") is wasteful. Physics features suffice for that. Micro-texture should only be used for fine species discrimination within a known group.

### 2. Curriculum Learning

The NN is trained in phases:
1. **Phase 1**: Learn physics features (46D)
2. **Phase 2**: Add macro texture (76D total)
3. **Phase 3**: Add micro texture (112D total)

This prevents overfitting to high-variance micro features.

### 3. Feature Reweighting

After Stage 1 predicts a group, taxonomic priors boost features known to be discriminative:
- **Cetaceans**: ICI (inter-click intervals) critical
- **Bats**: FM slope (frequency modulation rate) critical
- **Insects**: Rhythm patterns critical
- **Birds**: Spectral derivatives critical

### 4. Specialist Models

Stage 2 uses group-specific RFs trained only on samples from that taxonomic group. This allows finding subtle splits that would be washed out in a global model.

---

## Performance Metrics

### RouterMetrics

```rust
pub struct RouterMetrics {
    pub species_accuracy: f32,        // Overall species classification accuracy (%)
    pub group_accuracy: f32,          // Taxonomic group accuracy (%)
    pub total_samples: usize,         // Total samples evaluated
    pub correct_species: usize,       // Correct species predictions
    pub correct_group: usize,         // Correct group predictions
    pub reliable_samples: usize,      // Predictions meeting reliability threshold
    pub reliable_accuracy: f32,       // Accuracy on reliable predictions only (%)
    pub rf_only_count: usize,         // Predictions using RF only
    pub nn_only_count: usize,         // Predictions using NN only
    pub agreement_count: usize,       // RF/NN agreement count
    pub agreement_rate: f32,          // RF/NN agreement rate (%)
}
```

### Expected Performance

| Metric | Target |
|--------|--------|
| Group Accuracy | >95% |
| Species Accuracy | >85% |
| Reliable Accuracy | >90% |
| Agreement Rate | >80% |

---

## File Locations

| File | Purpose |
|------|---------|
| `src/hierarchical_ensemble_router.rs` | Main router implementation |
| `src/taxonomic_router.rs` | Taxonomic weights and feature slicing |
| `src/rf_stacking_ensemble.rs` | Random Forest implementation |
| `src/bin/train_curriculum_nn_112d.rs` | NN training with curriculum learning |
| `src/bin/train_specialist_rfs.rs` | Specialist RF training |
| `src/bin/eval_curriculum_nn_112d.rs` | NN evaluation |
| `src/bin/eval_rf_stacking_ensemble.rs` | RF ensemble evaluation |

---

## Testing

Run the test suite:

```bash
cd technical_architecture
cargo test hierarchical_ensemble_router
```

Test coverage includes:
- Configuration validation
- Router creation and initialization
- Specialist registry operations
- Stage 1 ensemble voting
- Stage 2 ensemble voting
- Feature reweighting
- Batch classification
- Metrics calculation
