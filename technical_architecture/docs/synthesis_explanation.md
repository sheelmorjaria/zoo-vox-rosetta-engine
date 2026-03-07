# Synthesis Module Explanation

## Overview

The `synthesis.rs` module is a comprehensive audio synthesis system for generating realistic animal vocalizations using granular and parametric synthesis techniques. It implements multiple synthesis engines with species-specific parameter mapping.

## Architecture

### Core Components

```
synthesis.rs
├── AudioFeatures (5D) - Legacy feature representation
├── DynamicMicroharmonicParams - 11D synthesis parameters
├── RosettaGrainParameters - 112D feature-to-synthesis mapping (NEW)
├── Grain - Basic granular synthesis unit
├── RosettaGrain - Enhanced grain with 112D state (NEW)
├── GranularSynthesizer - Basic granular engine
├── DynamicMicroharmonicSynthesizer - Micro-dynamics synthesis
├── RosettaGrainSynthesizer - 112D-aware synthesis (NEW)
├── ConcatenativeSynthesizer - Phrase concatenation
├── SuperpositionalSynthesizer - Layered synthesis
└── CombinedSynthesizer - Unified interface
```

## Key Structures

### 1. DynamicMicroharmonicParams (11D)

Maps micro-dynamics features to synthesis-controllable values:

```rust
pub struct DynamicMicroharmonicParams {
    pub f0_base: f32,              // Base fundamental frequency
    pub duration_ms: f32,          // Grain duration
    pub attack_ms: f32,            // Attack time
    pub decay_ms: f32,             // Decay time
    pub sustain_level: f32,        // Sustain level (0-1)
    pub vibrato_rate_hz: f32,      // Vibrato speed
    pub vibrato_depth_cents: f32,  // Vibrato extent
    pub jitter_amount: f32,        // Phase perturbation
    pub shimmer_amount: f32,       // Amplitude perturbation
    pub spectral_tilt: f32,        // Harmonic rolloff (dB/octave)
    pub hnr_db: f32,               // Harmonic-to-noise ratio
}
```

**Species Defaults:**
- `marmoset_default(f0_hz, duration_ms)` - 8kHz fundamental, moderate vibrato
- `bat_default(f0_hz, duration_ms)` - 40kHz fundamental, large FM depth

### 2. RosettaGrainParameters (28D) - NEW

Maps 112D RosettaFeatures to synthesis-controllable values:

```rust
pub struct RosettaGrainParameters {
    // Pitch Control (Layer 1)
    pub base_f0_hz: f32,
    pub pitch_variation_hz: f32,
    pub vibrato_depth_cents: f32,
    pub vibrato_rate_hz: f32,

    // Timing Control (Layer 1)
    pub grain_duration_ms: f32,
    pub attack_ms: f32,
    pub decay_ms: f32,
    pub sustain_level: f32,
    pub release_ms: f32,

    // Dynamics Control (Layer 1 + 3)
    pub rms_energy: f32,
    pub peak_amplitude: f32,
    pub dynamics_rise_rate: f32,
    pub dynamics_fall_rate: f32,

    // Timbre Control (Layer 1 + 2)
    pub hnr_db: f32,
    pub spectral_centroid: f32,
    pub spectral_tilt: f32,
    pub spectral_flatness: f32,

    // Texture Control (Layer 1 + 3)
    pub jitter_amount: f32,     // Clamped 0.0-0.1
    pub shimmer_amount: f32,    // Clamped 0.0-0.1
    pub fm_depth_hz: f32,
    pub am_depth: f32,

    // Advanced Texture (Layer 2 + 3)
    pub pitch_complexity: f32,
    pub harmonic_density: f32,
    pub granularity: f32,
    pub rhythm_complexity: f32,

    // Spatial Control (derived)
    pub pan: f32,              // -1.0 to 1.0
    pub spatial_width: f32,
}
```

**Key Methods:**
- `from_rosetta_features(features: &RosettaFeatures) -> Self` - Extract params from 112D
- `to_microharmonic_params() -> DynamicMicroharmonicParams` - Backward compatibility
- `default_marmoset() -> Self` - 8kHz, moderate vibrato, 60ms grains
- `default_bat() -> Self` - 40kHz, large FM, 30ms grains

### 3. Grain

Basic granular synthesis unit with Hanning envelope:

```rust
struct Grain {
    samples: VecDeque<f32>,   // Audio samples
    position: usize,          // Current playback position
    envelope: Vec<f32>,       // Hanning window
    amplitude: f32,           // Overall amplitude
    rate: f32,                // Playback rate
    pan: f32,                 // Stereo position
}
```

### 4. RosettaGrain (NEW)

Enhanced grain with 112D-aware synthesis state:

```rust
pub struct RosettaGrain {
    samples: VecDeque<f32>,
    position: usize,
    envelope: Vec<f32>,
    amplitude: f32,
    rate: f32,
    pan: f32,
    rosetta_params: RosettaGrainParameters,  // 112D-derived state
    phase: f64,                              // Phase accumulator
    instant_f0: f32,                         // Instantaneous F0 for FM
    envelope_state: GrainEnvelopeState,      // ADSR tracking
}
```

## Synthesis Engines

### 1. GranularSynthesizer

Basic granular synthesis with source buffer:

```rust
let config = SynthesisConfig::default();
let mut synth = GranularSynthesizer::new(config).await?;
synth.load_source(segment).await?;
let audio = synth.synthesize(100.0).await?;  // 100ms duration
```

### 2. DynamicMicroharmonicSynthesizer

Natural-sounding synthesis using micro-dynamics:

```rust
let synth = DynamicMicroharmonicSynthesizer::new(48000);
let params = DynamicMicroharmonicParams::marmoset_default(8000.0, 50.0);
let audio = synth.synthesize_phrase(&params);
```

**Core synthesis algorithm:**
1. Calculate instantaneous F0 with vibrato
2. Apply jitter (random phase perturbation)
3. Generate additive harmonic stack (8 harmonics)
4. Apply spectral tilt rolloff
5. Apply shimmer (random amplitude variation)
6. Add noise based on HNR
7. Apply ADSR envelope

### 3. RosettaGrainSynthesizer (NEW)

112D-aware synthesis with biologically-informed grain generation:

```rust
let config = SynthesisConfig::default();
let mut synth = RosettaGrainSynthesizer::new(config).await?;

// From 112D features
let features = extractor.extract_rosetta(&audio)?;
let synthesized = synth.generate_from_rosetta(&features, 10).await?;

// Backward compatible with 5D AudioFeatures
let audio_features = AudioFeatures { f0: 8000.0, ... };
let synthesized = synth.generate_from_audio_features(&audio_features).await?;
```

### 4. ConcatenativeSynthesizer

Phrase concatenation with crossfade:

```rust
let synth = ConcatenativeSynthesizer::new(config).await?;
let phrases = vec![phrase1, phrase2, phrase3];
let audio = synth.concatenate(&phrases, SynthesisMode::Horizontal).await?;
```

### 5. SuperpositionalSynthesizer

Layered synthesis for chordal/vertical structures:

```rust
let synth = SuperpositionalSynthesizer::new(config).await?;
let layers = vec![layer1, layer2];
let audio = synth.superpose(&layers).await?;
```

## ADSR Envelope

The ADSR (Attack-Decay-Sustain-Release) envelope shapes amplitude over time:

```rust
fn calculate_adsr_envelope(time: f32, total: f32, attack: f32, decay: f32, sustain: f32) -> f32 {
    if time < attack {
        (time / attack).powf(3.0)  // Logarithmic attack
    } else if time < (total - decay) {
        sustain                     // Sustain phase
    } else if total > decay {
        sustain * (1.0 - (current / decay).powf(0.5))  // Logarithmic decay
    } else {
        0.0
    }
}
```

## Feature-to-Parameter Mapping

### 112D RosettaFeatures → RosettaGrainParameters

| Synthesis Parameter | RosettaFeature Source | Layer |
|---------------------|----------------------|-------|
| base_f0_hz | mean_f0_hz | Base Physics |
| grain_duration_ms | duration_ms | Base Physics |
| attack_ms | attack_time_ms | Base Physics |
| vibrato_rate_hz | vibrato_rate_hz | Base Physics |
| jitter_amount | jitter (clamped 0-0.1) | Base Physics |
| fm_depth_hz | fm_depth_hz | Base Physics |
| spectral_tilt | harmonic_slope | Macro Texture |
| pitch_complexity | pitch_complexity | Macro Texture |
| harmonic_density | harmonic_density | Macro Texture |
| granularity | granularity | Micro Texture |
| rhythm_complexity | rhythm_complexity | Micro Texture |
| pan | spectral_skewness / 3.0 | Derived |
| spatial_width | spectral_spread / 5000.0 | Derived |

## Species-Specific Profiles

### Marmoset (Callithrix jacchus)
- **F0 Range**: 6-12 kHz (mean ~8 kHz)
- **Grain Duration**: 50-80ms
- **Attack**: 5-15ms
- **Vibrato**: Moderate (6-8 Hz, 20-40 cents)
- **Jitter/Shimmer**: Moderate (0.02-0.03)
- **FM Depth**: 200-500 Hz
- **Spectral Tilt**: -5 to -6 dB/octave

### Egyptian Fruit Bat (Rousettus aegyptiacus)
- **F0 Range**: 20-100 kHz (mean ~40 kHz)
- **Grain Duration**: 20-50ms
- **Attack**: 3-10ms
- **Vibrato**: Fast (8-12 Hz)
- **FM Depth**: 2000-10000 Hz (large sweeps)
- **Spectral Tilt**: -6 to -10 dB/octave
- **Harmonic Density**: Lower (CF-FM calls)

## Synthesis Modes

```rust
pub enum SynthesisMode {
    Horizontal,   // Sequential/phrasal (concatenative)
    Vertical,     // Simultaneous/chordal (superpositional)
    Combined,     // Mixed encoding
}
```

## Safety Features

### Emergency Stop

All synthesizers implement `emergency_stop()` for immediate halt:

```rust
pub fn emergency_stop(&mut self) -> Result<()> {
    self.grains.clear();
    self.output_buffer.clear();
    self.read_position = 0.0;
    Ok(())
}
```

### Output Normalization

Automatic normalization prevents clipping:

```rust
let max_amplitude = output.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
if max_amplitude > 0.0 {
    let scale = config.output_gain / max_amplitude;
    for sample in &mut output {
        *sample *= scale;
    }
}
```

## Usage Examples

### Basic Grain Synthesis

```rust
let config = SynthesisConfig {
    sample_rate: 48000,
    grain_size_ms: 50.0,
    grain_overlap: 0.5,
    max_grains: 100,
    ..Default::default()
};

let mut synth = GranularSynthesizer::new(config).await?;
let audio = synth.synthesize(1000.0).await?;  // 1 second
```

### Microharmonic Synthesis with Species Defaults

```rust
let synth = DynamicMicroharmonicSynthesizer::new(48000);

// Marmoset vocalization
let params = DynamicMicroharmonicParams::marmoset_default(8500.0, 60.0);
let marmoset_audio = synth.synthesize_phrase(&params);

// Bat echolocation
let params = DynamicMicroharmonicParams::bat_default(45000.0, 30.0);
let bat_audio = synth.synthesize_phrase(&params);
```

### 112D Rosetta Synthesis

```rust
// Extract features from recording
let extractor = MicroDynamicsExtractor::new(48000);
let features = extractor.extract_rosetta(&recorded_audio)?;

// Generate similar vocalization
let config = SynthesisConfig::default();
let mut synth = RosettaGrainSynthesizer::new(config).await?;
let synthesized = synth.generate_from_rosetta(&features, 20).await?;
```

## Performance Considerations

- **Grain Size**: 20-100ms optimal for animal vocalizations
- **Max Grains**: 50-100 for real-time, higher for offline
- **Sample Rate**: 48kHz minimum, 96kHz for bats (ultrasonic)
- **Buffer Size**: Use power-of-2 sizes for FFT efficiency

## Dependencies

```toml
[dependencies]
anyhow = "1.0"
log = "0.4"
lru = "0.12"
parking_lot = "0.12"
rand = "0.8"
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1.0", features = ["full"] }
```

## Testing

```bash
# Run synthesis tests
cargo test synthesis

# Run specific synthesizer tests
cargo test granular_synthesizer
cargo test dynamic_microharmonic
cargo test rosetta_grain

# Full CI
cargo test --all
```

## References

1. Roads, C. (2001). Microsound. MIT Press.
2. Keller, D. (2000). Granular Synthesis. Cambridge University Press.
3. Universal Rosetta Stone Methodology - Cross-species vocalization analysis framework.
