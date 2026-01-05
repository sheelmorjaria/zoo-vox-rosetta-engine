# Technical Architecture - Rust Execution Layer

High-performance Rust execution layer for the animal vocalization analysis framework. Provides safety-critical audio processing, environmental monitoring, and production deployment capabilities.

## Overview

This is the **Rust Execution Layer** of a hybrid Python/Rust architecture where:
- **Rust** handles time-critical operations, signal processing, hardware access, and safety
- **Python** handles cognitive intelligence, decision making, and context interpretation

The system follows a **"Fail Open to Safety"** design principle - if Python crashes, Rust immediately mutes audio and continues in safe Passthrough Mode.

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

## Test Coverage

**415 tests passing** - Comprehensive coverage of all functionality

```
Rust Execution Layer: 415 tests passing
├── Core Modules: 179 tests
├── Production Deployment: 142 tests
└── Field Deployment: 187 tests
```

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

## Documentation

- `CLAUDE.md` - Comprehensive developer guide
- `TDD_PLAN_FIELD_FEATURES.md` - Field deployment implementation plan (COMPLETE)
- `TDD_PLAN_PRODUCTION_FEATURES.md` - Production features plan (COMPLETE)
- `deployment/README.md` - Deployment instructions

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

## Dependencies

See `Cargo.toml` for full dependencies. Key dependencies:

- `tokio` - Async runtime
- `tract-onnx` - ONNX ML inference
- `serde` - Serialization
- `zmq` - ZeroMQ for inter-process communication
- `chrono` - Time handling
- `ndarray` - Numerical computing

## License

**CC BY-ND 4.0 International** - See main project license for details.

## Author

Sheel Morjaria (sheelmorjaria@gmail.com)

## Scientific Context

This framework transforms animal communication research by:
1. Moving beyond simple classification to cognitive understanding
2. Enabling deception detection in animal communication
3. Tracking emergent cultural behaviors
4. Recognizing intentional, targeted communication
5. Providing comparative analysis across species

The research impact focuses on understanding animal intelligence through vocalization patterns.
