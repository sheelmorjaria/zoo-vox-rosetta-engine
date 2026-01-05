# Phase 1: The Reflex Arc (Rust Core)

This module implements the foundational "spinal cord" of the audio processing system using pure Rust. It provides basic audio I/O, safety systems, and feature extraction capabilities.

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                 Audio Processor                      │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │ Audio I/O   │  │ Safety System│  │ Watchdog    │  │
│  │ (cpal)      │  │ (SPL Limit) │  │ Timer       │  │
│  └─────────────┘  └──────────────┘  └─────────────┘  │
│                                                     │
│  ┌─────────────────────────────────────────────────┐  │
│  │                    FFT Engine                    │  │
│  │  (rustfft)  →  STFT/RMS Extraction → Features  │  │
│  └─────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## Key Components

### 1. AudioProcessor
- **Purpose**: Core audio processing pipeline
- **Features**: STFT computation, RMS calculation, feature extraction
- **Performance**: Optimized for real-time processing (< 5ms latency)

### 2. SafetySystem
- **Purpose**: Protects hearing equipment from SPL overload
- **Features**: SPL monitoring, automatic muting, threshold enforcement
- **Default**: 90 dB SPL threshold (configurable)

### 3. WatchdogTimer
- **Purpose**: System health monitoring and crash detection
- **Features**: Timeout detection, recovery mechanisms
- **Default**: 100ms timeout (configurable)

## Entry Criteria

To complete Phase 1, the following tests must pass:

- ✅ `test_audio_loopback.rs` - Audio I/O and loopback latency < 5ms
- ✅ `test_spl_protection.rs` - Safety system properly mutes loud audio
- ✅ `test_watchdog_timer.rs` - Watchdog detects hung threads
- ✅ `test_cpu_benchmark.rs` - FFT performance < 1ms per computation

## Usage

### Basic Audio Processing

```rust
use reflex_arc::AudioProcessor;

// Initialize processor
let mut processor = AudioProcessor::new(48000, 512);

// Process audio buffer
let input_signal = vec![0.1; 512];
let features = processor.process_buffer(&input_signal);
```

### Safety System

```rust
use reflex_arc::SafetySystem;

let mut safety = SafetySystem::new(90.0); // 90 dB threshold
let result = safety.check_spl(&audio_buffer);

if result.should_mute {
    safety.apply_mute(&mut audio_buffer);
}
```

### Watchdog Timer

```rust
use reflex_arc::WatchdogTimer;
use std::time::Duration;

let mut watchdog = WatchdogTimer::new(Duration::from_millis(100));

// Regular updates from audio thread
watchdog.update();

if watchdog.should_trigger() {
    // Handle system recovery
}
```

## Performance Targets

- **Audio Loopback Latency**: < 5ms
- **FFT Processing Time**: < 1ms (for 1024 samples)
- **SPL Check Overhead**: < 0.1ms
- **Watchdog Overhead**: < 0.01ms

## Dependencies

- `cpal`: Cross-platform audio I/O
- `rustfft`: Fast Fourier Transform implementation
- `criterion`: Performance benchmarking

## Next Steps

Once Phase 1 is complete and all tests pass, proceed to:

### Phase 2: The Neural Bridge (Hybrid Architecture)
- Implement ZeroMQ IPC between Rust and Python
- Test message serialization/deserialization
- Validate GIL handling and thread safety

## Testing

Run the test suite:
```bash
cargo test
```

Run benchmarks:
```bash
cargo bench
```

Run the demo:
```bash
cargo run --example hello_audio
```