# Archival Plan - Component Cleanup

This document tracks components that have been superseded by newer implementations
in the 5-Stage Synthesis Pipeline (NBD → 112D → Corpus → Synthesis).

## Components Superseded by NBD (Neural Boundary Detection)

| Component | Location | Status | Superseded By |
|-----------|----------|--------|---------------|
| Old Segmentation Logic | `analysis/rosetta_stone/universal_rosetta_stone.py` | Keep for reference | `technical_architecture/src/neural_boundary.rs` |

## Components Superseded by 112D Feature Stack

| Component | Location | Status | Superseded By |
|-----------|----------|--------|---------------|
| 30D AcousticFeatures | `src/data_models.py` | Keep (Python compatibility) | `RosettaFeatures` (112D) in Rust |
| 45D SourceMetadata | `technical_architecture/src/bio_acoustic_agent.rs` | Upgraded to 112D | `SourceMetadata112D` in `semantic_reconstruction.rs` |

## Components Superseded by Voting Ensemble

| Component | Location | Status | Superseded By |
|-----------|----------|--------|---------------|
| Single-Model Classifiers | `cognitive_intelligence/train_asteroid_*.py` | Keep for training | `VotingEnsemble` (RF+NN) |

## Components Superseded by Granular Synthesis

| Component | Location | Status | Superseded By |
|-----------|----------|--------|---------------|
| Simple Additive Synthesis | `technical_architecture/src/synthesis.rs` | Still in use (granular mode) | `CachedGranularSynthesizer` |

## Current Architecture (Post-Upgrade)

```
┌──────────────────────────────────────────────────────────────────┐
│                     PIPELINE CONTROLLER                          │
│                  (Rust: manifest_bridge.rs)                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [1. NBD]     Load Raw Audio --> Segment --> Save to Cache      │
│       │         (Rust: neural_boundary.rs)                       │
│       ▼                                                          │
│  [2. 112D]    Load Segments --> Extract 112D --> Save .json     │
│       │         (Rust: micro_dynamics_extractor.rs)              │
│       ▼                                                          │
│  ╔═══════════════════════════════════════════════════════════╗  │
│  ║ [3. CORPUS ANALYSIS] (Python Bridge)                      ║  │
│  ║  - Load segments_manifest.json                            ║  │
│  ║  - Run Clustering (k=1020)                                ║  │
│  ║  - Output: clusters.json {id: [112D_mean, best_wav]}      ║  │
│  ╚═══════════════════════════════════════════════════════════╝  │
│       │                                                          │
│       ▼                                                          │
│  [4. SYNTHESIS] Load best_wav + 112D_mean --> Granular Synth    │
│       │         (Rust: semantic_reconstruction.rs)               │
│       ▼                                                          │
│  [5. PLAYBACK] Output Audio                                      │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Key Files

### Rust (Execution Layer)
- `manifest_bridge.rs` - Pipeline Controller + JSON bridge
- `neural_boundary.rs` - NBD Segmentation (Stage 1)
- `micro_dynamics_extractor.rs` - 112D Feature Extraction (Stage 2)
- `semantic_reconstruction.rs` - ExemplarManager + CachedGranularSynthesizer (Stage 4)
- `synthesis.rs` - Audio output (Stage 5)

### Python (Logic Layer)
- `analysis/rosetta_stone/exemplar_manager.py` - Clustering + Exemplar Selection (Stage 3)
- `cognitive_intelligence/` - Voting Ensemble, classification

### Data Models
- `RosettaFeatures` (Rust) - 112D feature stack
- `SourceMetadata112D` (Rust) - Wraps RosettaFeatures with cluster_id
- `AcousticFeatures` (Python) - 30D for backward compatibility

## Test Results

- Rust: 1564+ tests passing
- Python: 14 tests passing (exemplar_manager)

## Decision: No Files Archived

After analysis, all components are still in active use or provide backward compatibility.
The upgrade to 112D has been completed without breaking existing functionality.

- Python `AcousticFeatures` (30D) remains for Python analysis scripts
- Rust `RosettaFeatures` (112D) is the new standard for synthesis
- Single-model trainers remain useful for ensemble component training
- Old segmentation logic in `universal_rosetta_stone.py` provides reference implementation
