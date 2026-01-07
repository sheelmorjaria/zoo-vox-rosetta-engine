# Technical Architecture - Rust Execution Layer

High-performance Rust execution layer for the animal vocalization analysis framework. Provides safety-critical audio processing, environmental monitoring, and production deployment capabilities.

## Table of Contents

1. [Overview](#overview)
2. [Complete Research Workflow](#complete-research-workflow)
3. [Features](#features)
4. [Build](#build)
5. [Test Coverage](#test-coverage)
6. [17D Metadata Synthesis](#17d-metadata-synthesis)
7. [Query Interface - 17D Metadata Queries](#query-interface---17d-metadata-queries)
8. [Granular Concatenative Synthesis](#granular-concatenative-synthesis)
9. [Granular Synthesis Limitations: The Formant Barrier](#granular-synthesis-limitations-the-formant-barrier)
10. [Architecture](#architecture)
11. [Deployment](#deployment)
12. [Performance](#performance)
13. [Dependencies](#dependencies)
14. [Documentation](#documentation)
15. [License](#license)
16. [Author](#author)
17. [Scientific Context](#scientific-context)

---

## Overview

This is the **Rust Execution Layer** of a hybrid Python/Rust architecture where:
- **Rust** handles time-critical operations, signal processing, hardware access, and safety
- **Python** handles cognitive intelligence, decision making, and context interpretation

The system follows a **"Fail Open to Safety"** design principle - if Python crashes, Rust immediately mutes audio and continues in safe Passthrough Mode.

---

## Complete Research Workflow

### Phase 1: Data Import & Feature Extraction

**Step 1: Import Vocalization Database**
```bash
python3 ../src/data_import/import_vocalization_data.py
```
- Loads 2,882 phrases across 4 species
- Extracts micro-dynamics features (17D metadata)
- Builds query indexes for fast search

**Step 2: Acoustic Feature Extraction**
- Fundamental: mean_f0_hz, duration_ms, f0_range_hz
- Grit Factors: harmonic_to_noise_ratio, spectral_flatness
- Motion Factors: attack_time_ms, decay_time_ms, sustain_level, vibrato_rate_hz, vibrato_depth, jitter
- Fingerprint Factors: mfcc_1-4, spectral_contrast
- Rhythm Factors: median_ici_ms, onset_rate_hz, ici_coefficient_of_variation

### Phase 2: Metadata-Driven Synthesis

**Step 3: Load Source with Metadata**
```python
from technical_architecture import GranularConcatenativeSynthesizer, SourceMetadata

# Create synthesizer
synth = GranularConcatenativeSynthesizer(sample_rate=22050)

# Load source audio with 17D metadata
metadata = SourceMetadata(
    mean_f0_hz=6800.0,
    duration_ms=50.0,
    f0_range_hz=400.0,
    harmonic_to_noise_ratio=20.0,
    spectral_flatness=0.1,
    # ... all 17 features
)

synth.load_source_with_metadata(audio_buffer, metadata)
```

**Step 4: Vector Delta Commands**
```python
# Shift pitch by relative amount (+200Hz)
synth.shift_pitch_by_hz(200.0)

# Shift duration by relative amount (-10ms)
synth.shift_duration_by_ms(-10.0)

# Apply complete 17D delta transformation
target_metadata = SourceMetadata(
    mean_f0_hz=7500.0,
    duration_ms=60.0,
    # ... all 17 features
)
synth.apply_micro_dynamics_delta(target_metadata)
```

### Phase 3: Acoustic Validation

**Step 5: Bio-Acoustic Validation**
```bash
# Run validation script
python3 tests/test_17d_metadata_synthesis.py

# Expected: 8/8 tests passing
# - Full 17D construction
# - Builder pattern with partial metadata
# - GRITTY vs PURE persona differentiation
# - Rhythmic vs harmonic call differentiation
# - Backward compatibility with 3D API
```

**Step 6: t-SNE Validation**
```bash
python3 ../realtime/granular_synthesis_validation.py
```
- Validates t-SNE distance < 7.0 (currently 6.452)
- Confirms formant structure preservation
- 76.1% improvement over additive synthesis

### Phase 4: Scientific Testing with Personas

**Step 7: Acoustic Persona Validation**

Personas are used for validation, not as a primary discovery approach. They provide scientifically meaningful acoustic profiles:

| Persona | Acoustic Characteristics | Use Case |
|---------|------------------------|----------|
| **PURE** | HNR > 20dB, flatness < 0.1, slow attack (>20ms) | Validate tonal synthesis quality |
| **GRITTY** | HNR < 5dB, flatness > 0.6, fast attack (<5ms) | Validate texture/noise reproduction |
| **RHYTHMIC** | Onset rate > 15Hz, regular ICI (CV < 0.3) | Validate temporal pattern synthesis |
| **HARMONIC** | Zero onset rate, zero ICI | Validate continuous tone synthesis |

**Step 8: Bio-Acoustic Turing Test**
```bash
python3 ../realtime/demo_bio_acoustic_turing_test.py
```
- Live animal testing
- Determines if animals distinguish natural vs. synthesized
- Validates behavioral relevance

### Phase 5: Field Deployment

**Step 9: Production Deployment**
```bash
# Deploy Rust engine and Python agent
sudo cp deployment/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start rust-field-engine.service
sudo systemctl start python-cognitive-agent.service
```

**Step 10: Field Validation**
- Environmental monitoring (rain, temperature, light)
- Power management (battery, solar optimization)
- Wildlife sentry (background species detection)
- Offline data synchronization

---

## Features

### Core Modules
- **Synthesis** - Granular, concatenative, superpositional synthesis engines
  - **Granular Concatenative Synthesis**: t-SNE distance 6.452 (< 7.0 target) ✅
  - 76.1% improvement over additive synthesis (distance 27.0)
  - Preserves formant structure while enabling pitch/time manipulation
  - See `GRANULAR_SYNTHESIS_FINDINGS.md` in parent directory
- **Source Separation** - Conv-TasNet via ONNX/Tract
- **PTP Clock** - IEEE 1588 precision timing (nanosecond accuracy)
- **Safety Monitor** - Watchdog timers and safety limits
- **Thermal Management** - Temperature monitoring and throttling
- **Provenance Logging** - Deterministic audit trails

### Production Deployment (All with comprehensive tests)
| Module | Tests | Description |
|--------|-------|-------------|
| IACUC Compliance | 29 | Legal animal research protocol enforcement |
| Time-Series Archive | 24 | High-frequency data storage and querying |
| Auto-Calibration | 17 | Self-health checks with drift detection |
| Shadow Model Monitoring | 26 | AI concept drift detection and rollback |
| Remote Web Dashboard | 25 | HTTPS/WebSocket monitoring and control |
| Multi-Node Coordination | 21 | PTP election and TDMA scheduling |

### Field Deployment
| Module | Tests | Description |
|--------|-------|-------------|
| Environmental Monitor | 46 | Rain, temperature, light sensing |
| Power Manager | 54 | Battery/solar optimization |
| Wildlife Sentry | 24 | Background species detection |
| Data Synchronizer | 20 | Offline black box queuing |
| Acoustic Simulator | 43 | TDD test fixture |

---

## Build

```bash
# Debug build
cargo build

# Release build (optimized)
cargo build --release

# Run tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run specific test
cargo test test_name

# Run benchmarks
cargo run --example benchmark_peer_controller --release
```

---

## Test Coverage

**466 tests passing** - Comprehensive coverage of all functionality

```
Rust Execution Layer: 466 tests passing
├── Rust Tests: 454 tests
│   ├── Core Modules: 187 tests
│   ├── Production Deployment: 142 tests
│   ├── Field Deployment: 187 tests
│   ├── 17D Metadata: 6 tests
│   └── Granular Synthesis: 5 tests
└── Python Tests: 12 tests (NEW - Formant Barrier Validation)
    ├── test_harmonic_cannot_become_transient ✅
    ├── test_transient_cannot_become_harmonic ✅
    ├── test_harmonic_preserves_spectral_flatness ✅
    ├── test_transient_preserves_spectral_flatness ✅
    ├── test_nearest_neighbor_preserves_modality ✅
    ├── test_persona_router_logic ✅
    ├── test_persona_switching_for_modality ✅
    ├── test_small_grains_create_temporal_artifacts ✅
    └── ...and more
```

---

## Python Integration

This library can be used as a Python module via PyO3 bindings:

```python
import technical_architecture

# Create technical architect
architect = technical_architecture.TechnicalArchitect()

# Get operation mode
mode = architect.get_operation_mode()
```

See `CLAUDE.md` for detailed API documentation.

---

## 17D Metadata Synthesis

### Overview

The 17-dimensional metadata synthesis system enables precise control over acoustic features through vector delta commands. Instead of absolute targets like "set F0 to 7000Hz", you use relative shifts like "increase F0 by +200Hz".

### The 17 Micro-Dynamics Features

**1. Fundamental (3 features)**
- `mean_f0_hz`: Mean fundamental frequency (Hz)
- `duration_ms`: Temporal extent (ms)
- `f0_range_hz`: Pitch modulation range (Hz)

**2. Grit Factors (2 features)**
- `harmonic_to_noise_ratio`: Harmonic purity vs noise (dB)
- `spectral_flatness`: Noise-like vs tonal (0-1)

**3. Motion Factors (6 features)**
- `attack_time_ms`: Onset speed (fast=sharp, slow=gentle)
- `decay_time_ms`: Release speed (ms)
- `sustain_level`: Steady-state amplitude (0-1)
- `vibrato_rate_hz`: Pitch modulation frequency (Hz)
- `vibrato_depth`: Pitch modulation depth (Hz)
- `jitter`: Micro-perturbations/instability (0-1)

**4. Fingerprint Factors (5 features)**
- `mfcc_1` through `mfcc_4`: Mel-frequency cepstral coefficients
- `spectral_contrast`: Formant structure strength

**5. Rhythm Factors (3 features)**
- `median_ici_ms`: Inter-click interval (ms)
- `onset_rate_hz`: Click/event rate (Hz)
- `ici_coefficient_of_variation`: Rhythm regularity (0=regular, 1=irregular)

### Python API

```python
from technical_architecture import GranularConcatenativeSynthesizer, SourceMetadata

# Create synthesizer
synth = GranularConcatenativeSynthesizer(sample_rate=22050)

# Method 1: Full 17D metadata
metadata = SourceMetadata(
    mean_f0_hz=7000.0,
    duration_ms=50.0,
    f0_range_hz=400.0,
    harmonic_to_noise_ratio=20.0,
    spectral_flatness=0.1,
    attack_time_ms=10.0,
    decay_time_ms=15.0,
    sustain_level=0.7,
    vibrato_rate_hz=8.0,
    vibrato_depth=50.0,
    jitter=0.02,
    mfcc_1=-500.0,
    mfcc_2=-100.0,
    mfcc_3=-50.0,
    mfcc_4=-20.0,
    spectral_contrast=20.0,
    median_ici_ms=0.0,
    onset_rate_hz=0.0,
    ici_coefficient_of_variation=0.0,
)

# Method 2: Builder pattern (partial metadata)
metadata = SourceMetadata.builder() \
    .mean_f0_hz(6500.0) \
    .duration_ms(60.0) \
    .jitter(0.05) \
    .build()  # Remaining fields use defaults

# Load source with metadata
synth.load_source_with_metadata(audio_buffer, metadata)

# Now use vector delta commands!
synth.shift_pitch_by_hz(200.0)        # Relative pitch shift
synth.shift_duration_by_ms(-10.0)     # Relative time shift
synth.shift_f0_range_by_hz(100.0)     # Relative F0 range shift
```

### Acoustic Persona Validation

Personas provide scientifically meaningful profiles for validation:

**PURE Persona (Tonal, clean):**
```python
pure_metadata = SourceMetadata(
    mean_f0_hz=7000.0,
    harmonic_to_noise_ratio=25.0,    # High (pure)
    spectral_flatness=0.05,          # Low (focused)
    attack_time_ms=25.0,             # Slow (smooth)
    jitter=0.01,                     # Low (stable)
    spectral_contrast=20.0,          # High formant structure
)
```

**GRITTY Persona (Noisy, rough):**
```python
gritty_metadata = SourceMetadata(
    mean_f0_hz=7000.0,
    harmonic_to_noise_ratio=2.0,     # Low (gritty)
    spectral_flatness=0.8,           # High (noise-like)
    attack_time_ms=3.0,              # Fast (sharp)
    jitter=0.15,                     # High (rough)
    spectral_contrast=5.0,           # Low formant structure
)
```

**Delta Calculation:**
```python
# Calculate transformation from PURE to GRITTY
delta = gritty_metadata.delta_from(pure_metadata)

# Apply delta to any source
synth.apply_micro_dynamics_delta(delta)
```

### Rust Implementation

The 17D metadata system is implemented in `src/synthesis.rs`:

```rust
pub struct SourceMetadata {
    // === Fundamental (3 features) ===
    pub mean_f0_hz: f32,
    pub duration_ms: f32,
    pub f0_range_hz: f32,

    // === Grit Factors (2 features) ===
    pub harmonic_to_noise_ratio: f32,
    pub spectral_flatness: f32,

    // === Motion Factors (6 features) ===
    pub attack_time_ms: f32,
    pub decay_time_ms: f32,
    pub sustain_level: f32,
    pub vibrato_rate_hz: f32,
    pub vibrato_depth: f32,
    pub jitter: f32,

    // === Fingerprint Factors (5 features) ===
    pub mfcc_1: f32,
    pub mfcc_2: f32,
    pub mfcc_3: f32,
    pub mfcc_4: f32,
    pub spectral_contrast: f32,

    // === Rhythm Factors (3 features) ===
    pub median_ici_ms: f32,
    pub onset_rate_hz: f32,
    pub ici_coefficient_of_variation: f32,
}

pub struct MicroDynamicsDelta {
    // 17 delta fields (differences between two metadata sets)
}

pub struct SourceMetadataBuilder {
    // Fluent builder API for partial metadata construction
}
```

### Testing

Run the 17D metadata tests:

```bash
# Python tests
python3 tests/test_17d_metadata_synthesis.py

# Rust tests
cargo test --lib metadata

# Expected: All 14 tests passing (6 Rust + 8 Python)
```

---

## Query Interface - 17D Metadata Queries

### Overview

The query interface (`vocalization_query_interface.py`) provides fast 17D metadata queries over the vocalization database, enabling researchers to find phrases with specific acoustic characteristics, filter by acoustic personas, and calculate transformation deltas.

### 17D Metadata Search Methods

Search by specific acoustic features:

```python
from query_interface import get_query_interface
from data_models import Species

qi = get_query_interface()

# === Grit Factor Queries ===
# Find tonal phrases (high harmonic-to-noise ratio)
tonal_phrases = qi.search_by_hnr(20.0, 50.0, Species.MARMOSET)
for phrase_key, phrase in tonal_phrases:
    print(f"{phrase_key}: HNR={phrase.acoustic_features.harmonic_to_noise_ratio:.1f}dB")

# Find noisy phrases (high spectral flatness)
noisy_phrases = qi.search_by_spectral_flatness(0.5, 1.0)

# === Motion Factor Queries ===
# Find sharp onset phrases (fast attack)
sharp_phrases = qi.search_by_attack_time(0.0, 5.0)

# Find stable phrases (low jitter)
stable_phrases = qi.search_by_jitter(0.0, 0.05)

# === Rhythm Factor Queries ===
# Find pulsed phrases (high onset rate)
pulsed_phrases = qi.search_by_onset_rate(15.0, 50.0)

# Find continuous tones (zero onset rate)
continuous_phrases = qi.search_by_onset_rate(0.0, 0.0)
```

### Acoustic Persona Queries

Find phrases matching scientifically meaningful acoustic personas:

```python
# PURE Persona (tonal, clean, smooth)
# Characteristics: HNR > 20dB, flatness < 0.1, attack > 20ms, jitter < 0.05
pure_phrases = qi.get_pure_persona_phrases()
print(f"Found {len(pure_phrases)} PURE persona phrases")
for phrase_key, phrase in pure_phrases[:5]:
    print(f"  {phrase_key}: HNR={phrase.acoustic_features.harmonic_to_noise_ratio:.1f}dB")

# GRITTY Persona (noisy, rough, sharp)
# Characteristics: HNR < 5dB, flatness > 0.6, attack < 5ms, jitter > 0.1
gritty_phrases = qi.get_gritty_persona_phrases()
print(f"Found {len(gritty_phrases)} GRITTY persona phrases")

# RHYTHMIC Persona (pulsed, regular temporal patterns)
# Characteristics: onset rate > 15Hz, ICI CV < 0.3
rhythmic_phrases = qi.get_rhythmic_persona_phrases()
print(f"Found {len(rhythmic_phrases)} RHYTHMIC persona phrases")

# HARMONIC Persona (continuous tones, no pulses)
# Characteristics: onset rate = 0, median ICI = 0
harmonic_phrases = qi.get_harmonic_persona_phrases()
print(f"Found {len(harmonic_phrases)} HARMONIC persona phrases")
```

### 17D Nearest Neighbor Search

Find acoustically similar phrases in 17D feature space:

```python
# Find 5 nearest neighbors to a target phrase
neighbors = qi.find_nearest_neighbors_17d("marmoset_phee_001", k=5)

print("Nearest neighbors in 17D space:")
for distance, phrase_key, phrase in neighbors:
    af = phrase.acoustic_features
    print(f"  {phrase_key}:")
    print(f"    Distance: {distance:.2f}")
    print(f"    F0: {af.mean_f0_hz:.0f}Hz, HNR: {af.harmonic_to_noise_ratio:.1f}dB")
    print(f"    Attack: {af.attack_time_ms:.1f}ms, Jitter: {af.jitter:.3f}")
```

### 17D Delta Calculation

Calculate transformation deltas between phrases for synthesis:

```python
# Calculate delta from source to target
delta = qi.calculate_17d_delta("source_phrase_001", "target_phrase_001")

print("17D Delta (transformation needed):")
print(f"  Pitch shift: {delta['delta_mean_f0_hz']:+.1f} Hz")
print(f"  Duration change: {delta['delta_duration_ms']:+.1f} ms")
print(f"  F0 range change: {delta['delta_f0_range_hz']:+.1f} Hz")
print(f"  HNR change: {delta['delta_harmonic_to_noise_ratio']:+.1f} dB")
print(f"  Flatness change: {delta['delta_spectral_flatness']:+.3f}")
print(f"  Attack time change: {delta['delta_attack_time_ms']:+.1f} ms")
print(f"  Jitter change: {delta['delta_jitter']:+.3f}")
print(f"  Vibrato rate change: {delta['delta_vibrato_rate_hz']:+.1f} Hz")
print(f"  Onset rate change: {delta['delta_onset_rate_hz']:+.1f} Hz")

# Use delta with Rust synthesizer
from technical_architecture import GranularConcatenativeSynthesizer

synth = GranularConcatenativeSynthesizer(sample_rate=22050)
synth.load_source_with_metadata(source_audio, source_metadata)

# Apply delta transformations
synth.shift_pitch_by_hz(delta['delta_mean_f0_hz'])
synth.shift_duration_by_ms(delta['delta_duration_ms'])
```

### Complete Workflow Example

Find source phrase, calculate delta to target, synthesize:

```python
from query_interface import get_query_interface
from technical_architecture import GranularConcatenativeSynthesizer, SourceMetadata

# Step 1: Find target persona phrases
qi = get_query_interface()
pure_phrases = qi.get_pure_persona_phrases(Species.MARMOSET)
gritty_phrases = qi.get_gritty_persona_phrases(Species.MARMOSET)

if pure_phrases and gritty_phrases:
    pure_key, pure_phrase = pure_phrases[0]
    gritty_key, gritty_phrase = gritty_phrases[0]

    # Step 2: Calculate 17D delta
    delta = qi.calculate_17d_delta(pure_key, gritty_key)

    # Step 3: Load source with metadata
    synth = GranularConcatenativeSynthesizer(sample_rate=22050)

    # Create metadata from source phrase
    source_metadata = SourceMetadata(
        mean_f0_hz=pure_phrase.acoustic_features.mean_f0_hz,
        duration_ms=pure_phrase.acoustic_features.mean_duration_ms,
        f0_range_hz=pure_phrase.acoustic_features.f0_range_hz,
        harmonic_to_noise_ratio=pure_phrase.acoustic_features.harmonic_to_noise_ratio,
        spectral_flatness=pure_phrase.acoustic_features.spectral_flatness,
        attack_time_ms=pure_phrase.acoustic_features.attack_time_ms,
        decay_time_ms=pure_phrase.acoustic_features.decay_time_ms,
        sustain_level=pure_phrase.acoustic_features.sustain_level,
        vibrato_rate_hz=pure_phrase.acoustic_features.vibrato_rate_hz,
        vibrato_depth=pure_phrase.acoustic_features.vibrato_depth,
        jitter=pure_phrase.acoustic_features.jitter,
        mfcc_1=pure_phrase.acoustic_features.mfcc_1,
        mfcc_2=pure_phrase.acoustic_features.mfcc_2,
        mfcc_3=pure_phrase.acoustic_features.mfcc_3,
        mfcc_4=pure_phrase.acoustic_features.mfcc_4,
        spectral_contrast=pure_phrase.acoustic_features.spectral_contrast,
        median_ici_ms=pure_phrase.acoustic_features.median_ici_ms,
        onset_rate_hz=pure_phrase.acoustic_features.onset_rate_hz,
        ici_coefficient_of_variation=pure_phrase.acoustic_features.ici_coefficient_of_variation,
    )

    synth.load_source_with_metadata(source_audio, source_metadata)

    # Step 4: Apply delta to transform from PURE to GRITTY
    output = synth.apply_micro_dynamics_delta(delta)

    print(f"Transformed {pure_key} → {gritty_key}")
    print(f"Applied {delta['delta_harmonic_to_noise_ratio']:+.1f}dB HNR change")
    print(f"Applied {delta['delta_spectral_flatness']:+.3f} flatness change")
```

### API Reference

**17D Search Methods:**
- `search_by_hnr(min_hnr, max_hnr, species=None)` - Search by harmonic-to-noise ratio
- `search_by_spectral_flatness(min_flatness, max_flatness, species=None)` - Search by spectral flatness
- `search_by_attack_time(min_attack, max_attack, species=None)` - Search by attack time
- `search_by_jitter(min_jitter, max_jitter, species=None)` - Search by jitter
- `search_by_onset_rate(min_rate, max_rate, species=None)` - Search by onset rate

**Persona Queries:**
- `get_pure_persona_phrases(species=None)` - Get PURE persona phrases (tonal, clean)
- `get_gritty_persona_phrases(species=None)` - Get GRITTY persona phrases (noisy, rough)
- `get_rhythmic_persona_phrases(species=None)` - Get RHYTHMIC persona phrases (pulsed)
- `get_harmonic_persona_phrases(species=None)` - Get HARMONIC persona phrases (continuous)

**Advanced Methods:**
- `find_nearest_neighbors_17d(phrase_key, k=5, species=None)` - Find k nearest neighbors in 17D space
- `calculate_17d_delta(from_phrase_key, to_phrase_key)` - Calculate 17D transformation delta

---

## Granular Concatenative Synthesis

### Overview

Granular Concatenative Synthesis enables high-fidelity animal vocalization synthesis by manipulating real audio samples through grain windows. This approach preserves formant structure (vocal tract resonances) while enabling systematic pitch and time manipulation.

### Scientific Validation

**t-SNE Distance Results:**
- Natural ↔ Granular: **6.452** ✅ (target < 7.0)
- Natural ↔ Concatenative: 4.208 (baseline)
- Natural ↔ Additive: 27.052 (failed)

**Key Finding**: 76.1% improvement over additive synthesis proves that bio-acoustic complexity lies in the spectral envelope (formants), which mathematical sine waves cannot replicate.

### Python API

```python
from technical_architecture import GranularConcatenativeSynthesizer

# Create synthesizer
synth = GranularConcatenativeSynthesizer(sample_rate=22050)

# Load source audio (real animal vocalization)
synth.load_source(audio_buffer)

# Configure parameters
synth.set_pitch_shift(0.9)      # Lower pitch (0.9 = 10% lower)
synth.set_grain_size_ms(20.0)   # Grain size in milliseconds

# Synthesize output
output_audio = synth.synthesize(duration_ms=100.0)
```

### When to Use Granular Synthesis

**Use Granular Synthesis when:**
1. **Pitch Continuum Testing**: Need pitches not in database (7500Hz, 7600Hz, 7700Hz...)
2. **Controlling Confounds**: Same phrase, different pitches, constant duration
3. **Acoustic Feature Boundaries**: Just-noticeable-difference (JND) measurements
4. **Novel Stimuli**: Creating hybrid calls that don't exist naturally

**Use Concatenative Synthesis when:**
- You have exact audio segments you need (perfect fidelity, low flexibility)
- No parameter variation required
- Playing back natural recordings

### Rust Implementation

The granular synthesis engine is implemented in `src/synthesis.rs`:

```rust
pub struct GrainWindow {
    samples: Vec<f32>,
}

pub struct GranularVoice {
    source_buffer: Vec<f32>,
    sample_rate: usize,
    grain_size_ms: f32,
    window: Vec<f32>,
    position: f32,
    pitch_shift_ratio: f32,
}

pub struct GranularConcatenativeSynthesizer {
    sample_rate: usize,
    voices: Vec<GranularVoice>,
}
```

**Algorithm:**
1. Read sample from source buffer at current position (linear interpolation)
2. Apply grain window envelope based on position within grain
3. Advance position based on pitch shift ratio
4. Wrap around source buffer (looping)

### Test-Driven Development

All components implemented using TDD methodology:

```bash
# Run granular synthesis tests
cargo test --lib -- test_granular test_grain_window -- --nocapture
```

**5 tests, all passing:**
- `test_grain_window_hanning` ✅
- `test_granular_voice_pitch_shift` ✅
- `test_granular_voice_time_stretch` ✅
- `test_granular_morpher_overlap` ✅
- `test_granular_concatenative_synthesizer` ✅

### Validation

Run the granular synthesis validation script:

```bash
python3 ../realtime/granular_synthesis_validation.py
```

This validates that granular synthesis achieves t-SNE distance < 7.0, confirming bio-acoustic validity.

### Bio-Acoustic Turing Test

With granular synthesis validated, the next step is live animal testing using the Bio-Acoustic Turing Test framework:

```bash
python3 ../realtime/demo_bio_acoustic_turing_test.py
```

This determines if live animals can distinguish between natural and granular-synthesized vocalizations.

### Documentation

- **Scientific Findings**: `../realtime/GRANULAR_SYNTHESIS_FINDINGS.md`
- **Validation Script**: `../realtime/granular_synthesis_validation.py`
- **Turing Test Framework**: `../realtime/bio_acoustic_turing_test.py`

---

## Granular Synthesis Limitations: The Formant Barrier

### Overview

**Granular Concatenative Synthesis is a "Warping" technology, not a "Creation" technology.**

It is mathematically bound to the **Spectral Envelope** (DNA) of the source audio you load. You can stretch, squash, and shift the pitch of a "Harmonic" sound, but it will never lose its "Harmonic" nature to become a "Transient."

### The "Formant" Barrier (Why Transmutation Fails)

Granular synthesis works by chopping audio into tiny windows ("grains") and playing them back. It preserves the **Resonance** (Formants) of the original vocal tract.

**The Constraint:**
- **Source:** "Harmonic Phee" (Sine-like, smooth)
- **Operation:** Granular Time Stretch / Pitch Shift
- **Result:** A **"Different" Harmonic Phee** (Deeper/Longer)

**What happens to the "Transient" traits?**
- **Spectral Flatness:** Remains low (Tone-like)
- **Harmonic-to-Noise Ratio (HNR):** Remains high (Pure)
- **Attack Slope:** Remains smooth (Gradual)

**Verdict:**
You **cannot** synthesize a "Click" (Transient) from a "Whistle" (Harmonic) using Granular Synthesis. The smoothness is "baked in" to the source audio buffer.

### What Works vs. What Doesn't

| Goal | Possible? | Technique |
| :--- | :--- | :--- |
| **Harmonic → Deeper Harmonic** | ✅ **Yes** | Granular Pitch Shift |
| **Harmonic → Longer Harmonic** | ✅ **Yes** | Granular Time Stretch |
| **FM Sweep → Harmonic** | ✅ **Yes** (Limited) | Pitch Freezing / Large Grains |
| **Harmonic → Transient Click** | ❌ **No** | Impossible (DNA Mismatch) |
| **Transient → Harmonic** | ❌ **No** | Impossible (DNA Mismatch) |
| **Harmonic → Rhythmic Pulse** | ⚠️ **Artificial** | Granular Cloud (Sounds synthetic) |

### The Exception: FM Sweep → Harmonic ("Freezing" Effect)

There is one case where you **can** alter modality: **Reducing Frequency Modulation.**

If you synthesize an FM Sweep using **Large Grains** or apply **Synchronous Pitch Shifting**, you can "freeze" the frequency movement, effectively turning an FM Sweep into a pseudo-Harmonic tone.

**How it works:**
1. **Source:** Bat "FM Sweep" (7kHz → 9kHz)
2. **Granular Config:** Large grain size (covers the whole sweep) OR aggressive pitch averaging
3. **Synthesis:** The engine smooths out the frequency curve
4. **Result:** A "Choral" or "Warbled" tone that is closer to Harmonic than FM

**Use Case:**
Generating a "Calming" variant of a "Highly Modulated" call.

### The "Rhythmic" Illusion (Granular Artifacts)

You *can* make a Harmonic sound "Rhythmic" by reducing the grain size and randomizing the trigger (Granular Cloud).

**The Result:**
A "Sparkly" or "Bubbly" texture
- **Is it Rhythmic?** Yes, it has a temporal beat
- **Is it Biological?** Usually not. It sounds artificial/synthetic

**Verdict:**
Useful for generating **"Machine"** or **"Anomalous"** stimuli (testing Semiotic Emergence), but not for mimicking natural biological calls.

### The Solution: "Persona Switching" (Source Selection)

Since Granular Synthesis is **Source-Dependent**, you handle Modality via your **Persona Router**, not the Synthesis Engine itself.

**The Logic:**
Different Modalities = Different Source Buffers.

| Target Modality | Granular Config | Source Buffer (Persona) | Result |
| :--- | :--- | :--- | :--- |
| **Harmonic** | 20ms Grain, Pitch Shift | `marmoset_phee.wav` | **Warped Phee** (Preserves Tone) |
| **FM Sweep** | 10ms Grain, Preserve Time | `bat_fm_sweep.wav` | **Warped Sweep** (Preserves Modulation) |
| **Transient** | 2ms Grain (Chopper) | `bat_click.wav` | **Warped Click** (Preserves Attack) |

**Implementation:**
Your `ContextualAgent` handles the "Switch."
```python
def select_modality_source(self, intent_modality):
    if intent_modality == "HARMONIC":
        synth.load_source(self.personas.get("MARMOSET_PHEE"))
    elif intent_modality == "TRANSIENT":
        synth.load_source(self.personas.get("BAT_CLICK"))
    # Granular Engine just applies the "Warping" (Pitch/Time) to the selected buffer
```

### Acoustic Algebra Limitation (High-Dim Vectors)

This impacts your **Acoustic Algebra**.

If you calculate a **Target Vector** that is "50% Harmonic and 50% Transient," your **`find_nearest_real_phrase()`** function will always pick the **closest Neighbor**.

**Scenario:** Target is 50% Transient
- **Neighbor A:** 0% Transient (Distance: 0.4)
- **Neighbor B:** 100% Transient (Distance: 0.6)
- **Engine Choice:** It picks **Neighbor A**
- **Result:** You synthesize a **"Gritty Harmonic"** sound (using Transient source is too risky/distant)

**Conclusion:**
Acoustic Algebra can **nuance** a modality (make a Harmonic sound slightly harsher), but it cannot **cross the bridge** (Harmonic → Transient) because the **Spectral Envelope** prevents it.

### TDD Validation

All formant barrier constraints are validated by TDD tests in `tests/test_granular_synthesis_limitations.py`:

```bash
python3 -m pytest tests/test_granular_synthesis_limitations.py -v

# Expected: 12/12 tests passing
# - test_harmonic_cannot_become_transient_via_granular_synthesis ✅
# - test_transient_cannot_become_harmonic_via_granular_synthesis ✅
# - test_harmonic_preserves_spectral_flatness_after_pitch_shift ✅
# - test_transient_preserves_spectral_flatness ✅
# - test_nearest_neighbor_preserves_modality ✅
# - test_persona_router_logic ✅
# - test_persona_switching_for_modality ✅
# - test_small_grains_create_temporal_artifacts ✅
# - ...and more
```

### Summary

**Key Principle:** Rely on your **Persona Layer** (Source Buffers) to handle Modality. Rely on **Granular Synthesis** to handle **Gradient Intensity** (Emotional nuance) within that Modality.

| Aspect | Responsibility |
| :--- | :--- |
| **Modality Selection** | Persona Router (Source Selection) |
| **Pitch Warping** | Granular Synthesis (Pitch Shift) |
| **Time Warping** | Granular Synthesis (Time Stretch) |
| **Emotional Nuance** | 17D Metadata Deltas (within modality) |
| **Cross-Modality Synthesis** | ❌ Impossible (Formant Barrier) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Systemd Supervisor                        │
│  ┌──────────────────────────┐  ┌──────────────────────────┐     │
│  │  rust-field-engine       │  │  python-cognitive-agent  │     │
│  │  (Technical Architect)   │  │  (Logic Layer)           │     │
│  │                          │  │                          │     │
│  │  - Safety Critical       │  │  - Decision Making       │     │
│  │  - Audio Processing      │◄─┤  - Phrase Selection      │     │
│  │  - Hardware Control      │  │  - Learning              │     │
│  │  - Heartbeat Monitor     │  │  - Intent Generation     │     │
│  │                          │  │                          │     │
│  │  ZeroMQ SUB (Heartbeat)  │◄─┤  ZeroMQ PUB (Heartbeat)  │     │
│  └──────────────────────────┘  └──────────────────────────┘     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Deployment

See `deployment/` directory for systemd service files:

```bash
# Copy systemd files
sudo cp deployment/*.service /etc/systemd/system/
sudo systemctl daemon-reload

# Enable services
sudo systemctl enable rust-field-engine.service
sudo systemctl enable python-cognitive-agent.service

# Start services
sudo systemctl start rust-field-engine.service
sudo systemctl start python-cognitive-agent.service
```

---

## Performance

Key performance metrics from benchmarking:

- **Message Processing**: > 10M ops/sec
- **Heartbeat Latency**: < 1μs average
- **Mode Switching**: < 1μs (immediate flag update)
- **Timeout Detection**: < 1μs (near-instantaneous)
- **Concurrent Access**: Lock-free atomic operations

Run benchmarks:
```bash
cargo run --example benchmark_peer_controller --release
```

---

## Dependencies

See `Cargo.toml` for full dependencies. Key dependencies:

- `tokio` - Async runtime
- `tract-onnx` - ONNX ML inference
- `serde` - Serialization
- `zmq` - ZeroMQ for inter-process communication
- `chrono` - Time handling
- `ndarray` - Numerical computing

---

## Documentation

- `CLAUDE.md` - Comprehensive developer guide
- `TDD_PLAN_FIELD_FEATURES.md` - Field deployment implementation plan (COMPLETE)
- `TDD_PLAN_PRODUCTION_FEATURES.md` - Production features plan (COMPLETE)
- `deployment/README.md` - Deployment instructions

---

## License

**CC BY-ND 4.0 International** - See main project license for details.

---

## Author

Sheel Morjaria (sheelmorjaria@gmail.com)

---

## Scientific Context

This framework transforms animal communication research by:
1. **Metadata-Driven Synthesis**: 17-dimensional acoustic feature control
2. **Vector Delta Commands**: Relative transformations for precise manipulation
3. **Bio-Acoustic Validation**: t-SNE distance < 7.0 confirms naturalness
4. **Acoustic Personas**: Scientific profiles for validation (PURE, GRITTY, RHYTHMIC, HARMONIC)
5. **Cross-Species Analysis**: Marmoset, Bat, Dolphin, Chimpanzee, Sperm Whale, Zebra Finch
6. **Deception Detection**: Identifying intentional communication
7. **Emergent Behavior Tracking**: Cultural change detection

The research impact focuses on understanding animal intelligence through vocalization patterns, enabled by high-fidelity synthesis that preserves bio-acoustic complexity.
