# Acoustic Algebra Integration Guide

## Overview

**High-Dimensional Acoustic Algebra** transforms the phrase discovery pipeline from:
- **Discrete Retrieval**: Binary choice (Aggressive vs. Not Aggressive)
- **Continuous Generation**: Gradient choice (0%, 25%, 50%, 75%, 100% Aggressive)

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Audio + Annotations                                   │
│  Input: WAV files + ELAN/Praat Labels                          │
└────────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Phrase Discovery + Contextual Map                      │
│  DBSCAN Clustering + Annotation Association                    │
│                                                              │
│  🆕 ALGEBRA ROLE 1: DEFINING SEMANTIC VECTORS                │
│  • Calculate "Context Centroids"                                  │
│    Vector_Aggression = Mean(17D vectors for "Agg" phrases)        │
│  • Calculate "Context Variance"                                    │
│    How spread out is "Aggression?"                                 │
└────────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │  CONTEXTUAL VECTOR MAP      │
         │ (e.g., Aggression =          │
         │  +0.5 Jitter, -10ms Duration)│
         └───────────────┬───────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: Context-Aware Synthesis                                │
│  Granular Concatenative Engine                                   │
│                                                              │
│  🆕 ALGEBRA ROLE 2: GRADIENT GENERATION                      │
│  • Input: Intent="Aggression", Intensity=0.7                      │
│  • Math: V_target = V_neutral + (V_agg - V_neutral) * 0.7         │
│  • Output: "Virtual Phrase" (70% Aggressive)                     │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Discovery Phase: Calculate Context Centroids

```python
from analysis.rosetta_stone.contextual_map import ContextualMap
from analysis.rosetta_stone.high_dimensional_acoustic_algebra import AcousticFeatureVector17

# Load your annotated phrases
phrase_vectors = {
    'phrase_001': AcousticFeatureVector17(...),  # 17D features
    'phrase_002': AcousticFeatureVector17(...),
    # ...
}

context_labels = {
    'phrase_001': 'contact',
    'phrase_002': 'aggression',
    'phrase_003': 'food',
    # ...
}

# Create contextual map
map_obj = ContextualMap()
centroids = map_obj.calculate_context_centroids(phrase_vectors, context_labels)

# View centroids
map_obj.summarize()
```

### 2. Synthesis Phase: Generate Graded Phrases

```python
# Generate "30% Aggressive" virtual phrase
virtual_phrase = map_obj.generate_graded_phrase(
    target_context='aggression',
    intensity=0.3  # 30% aggression
)

# Find nearest real phrase (for synthesis)
nearest_key, nearest_vector, distance = map_obj.find_nearest_real_phrase(
    virtual_phrase,
    phrase_vectors
)

# Use nearest phrase as source buffer
synth.set_source(nearest_phrase.audio_buffer)
synth.synthesize()
```

## API Reference

### ContextualMap

#### `calculate_context_centroids(phrase_vectors, context_labels)`

Calculate semantic centroids for each context.

**Parameters:**
- `phrase_vectors`: Dict[str, AcousticFeatureVector17] - Phrase keys → 17D vectors
- `context_labels`: Dict[str, str] - Phrase keys → Context labels

**Returns:**
- Dict[str, ContextCentroid] - Context names → Centroids

#### `generate_graded_phrase(target_context, intensity, baseline_context=None)`

Generate a "Virtual Phrase" at specified intensity.

**Parameters:**
- `target_context`: str - Target context (e.g., "aggression")
- `intensity`: float - 0.0 (baseline) to 1.0 (full target)
- `baseline_context`: Optional[str] - Override baseline

**Returns:**
- AcousticFeatureVector17 - Interpolated virtual phrase

#### `find_nearest_real_phrase(virtual_vector, phrase_vectors)`

Find nearest real phrase to a virtual vector.

**Parameters:**
- `virtual_vector`: AcousticFeatureVector17 - Virtual (interpolated) vector
- `phrase_vectors`: Dict[str, AcousticFeatureVector17] - Available phrases

**Returns:**
- Tuple[str, AcousticFeatureVector17, float] - (phrase_key, vector, distance)

#### `calculate_context_delta(context_a, context_b)`

Calculate difference between two contexts.

**Parameters:**
- `context_a`: str - First context
- `context_b`: str - Second context

**Returns:**
- AcousticFeatureVector17 - Difference vector (A - B)

## Scientific Application: The Threshold Test

### Hypothesis

Animals perceive emotion as a **continuous continuum**, not discrete states.

### Experiment Design

```
Condition A (Baseline):  Intensity 0.0  → Contact
Condition B (Midpoint):    Intensity 0.5  → Mild Aggression
Condition C (Full):       Intensity 1.0  → Full Aggression
```

### Measurement

Plot behavioral response (looking time, flight initiation) vs. Intensity %:

- **If Linear**: Animal perceives a **GRADIENT** → Proof of Acoustic Continuum
- **If Step Function**: Animal perceives a **CATEGORY** → Proof of Discrete Semantics

### Why This Was Impossible Before

- **Old system**: Only 3 discrete levels (contact, aggression, food)
- **New system**: Infinite precision via acoustic algebra

## Files

- `high_dimensional_acoustic_algebra.py` - Core algebra engine
- `contextual_map.py` - Contextual map and gradient generation
- `demo_acoustic_algebra_integration.py` - Full integration demo

## Running the Demo

```bash
cd analysis/rosetta_stone
python3 demo_acoustic_algebra_integration.py
```

## Key Features

| Feature | Without Algebra | With Algebra |
|---------|-----------------|--------------|
| **Synthesis** | Retrieval (Play a file) | Generation (Create a vector) |
| **Nuance** | Low (3 discrete levels) | High (Infinite precision) |
| **Discovery** | Finds Phrases | Finds Contextual Axes |
| **Experiments** | Binary choice tests | Threshold tests (continuum) |
