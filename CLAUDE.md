# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a sophisticated animal vocalization analysis framework that combines scientific research with advanced software engineering. The system implements the Universal Rosetta Stone methodology for cross-species communication analysis, featuring a **hybrid Python/Rust architecture** with peer-to-peer supervision and cognitive intelligence capabilities.

### Core Architecture: Execution vs. Logic Split

The framework follows a **hybrid architecture** with clear separation of concerns:

- **Rust (Execution Layer)**: Time-critical operations, signal processing, hardware access, safety
  - Location: `technical_architecture/`
  - Zero-copy operations, memory safety, deterministic performance
  - Safety-critical with automatic fail-open

- **Python (Logic Layer)**: Cognitive intelligence, decision making, learning, context interpretation
  - Location: `cognitive_intelligence/`, `realtime/`, `semiotics/`
  - Rapid development, scientific computing, ML frameworks
  - Can crash safely - Rust continues in Passthrough Mode

### Peer-to-Peer Supervision

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

**Key Principle**: Fail open to safety. If Python crashes, Rust immediately mutes audio and continues in Passthrough Mode.

## Common Commands

### Essential Development Commands

```bash
# Run all tests
python3 -m pytest tests/ -v

# Run specific test files
python3 -m pytest tests/test_rosetta_stone_base.py -v
python3 -m pytest tests/test_realtime_system_population.py -v
python3 -m pytest tests/test_zero_copy_rust.py -v

# Import vocalization data (required before running demos)
python3 src/data_import/import_vocalization_data.py

# Run demos
python3 src/query_interface/demo_query_interface.py
python3 src/semiotics/demo_semiotic_engine.py
```

### Rust Components

```bash
# Build Rust components (in technical_architecture/)
cd technical_architecture && cargo build --release

# Run all Rust tests
cd technical_architecture && cargo test

# Run specific Rust tests
cd technical_architecture && cargo test peer_controller
cd technical_architecture && cargo test master_controller

# Run zero-copy example
cd technical_architecture && cargo run --example zero_copy_example

# Run specific tests with Rust integration
python3 -m pytest tests/test_zero_copy_rust.py -v
```

### Deployment (Systemd Services)

```bash
# Copy systemd files
sudo cp technical_architecture/deployment/*.service /etc/systemd/system/
sudo systemctl daemon-reload

# Enable services
sudo systemctl enable rust-field-engine.service
sudo systemctl enable python-cognitive-agent.service

# Start both services
sudo systemctl start rust-field-engine.service
sudo systemctl start python-cognitive-agent.service

# View logs
sudo journalctl -u rust-field-engine.service -f
sudo journalctl -u python-cognitive-agent.service -f

# Test heartbeat client
python3 technical_architecture/deployment/python_heartbeat_client.py
```

### Performance and Validation

```bash
# Run system TDD suite
python3 -m pytest tests/test_system_tdd_suite.py -v

# Run clustering validation
python3 -m pytest tests/test_clustering.py -v

# Run cross-species comparison tests
python3 -m pytest tests/test_cross_species_comparison.py -v
```

## Architecture Overview

### Core Design Philosophy

The framework follows a **hybrid architecture** combining:
- **Python (Logic Layer)**: Cognitive intelligence, decision making, learning, context interpretation
- **Rust (Execution Layer)**: Time-critical operations, signal processing, hardware access, safety
- **ZeroMQ Heartbeat**: Peer-to-peer supervision with automatic fail-safe
- **Systemd**: External process supervision with "Let It Crash" philosophy

### Main Components

#### 1. Rust Execution Layer (`technical_architecture/`)

**Core Modules:**
- **Master Controller** (`master_controller.rs`) - Deterministic Intent-Reality Mediator
  - Translates Python intents into physical Rust actions
  - Enforces thermal, safety, and hardware constraints
  - Watchdog monitoring with crash isolation

- **Peer Controller** (`peer_controller.rs`) - ZeroMQ heartbeat monitoring
  - Non-blocking heartbeat polling (0ms timeout)
  - Automatic mode switching (Passthrough ↔ Interactive)
  - 100ms timeout (5 missed heartbeats = disconnect)

- **Synthesis** (`synthesis.rs`) - Audio synthesis engines
  - Granular, concatenative, superpositional synthesis
  - Emergency stop functionality

- **Source Separation** (`source_separation.rs`) - Conv-TasNet separator
- **Thermal Management** (`thermal.rs`) - Temperature monitoring and throttling
- **Safety Monitoring** (`safety.rs`) - Watchdog timers, safety limits
- **PTP Clock** (`ptp.rs`) - IEEE 1588 precision timing (nanosecond)
- **Provenance Logging** (`logging.rs`) - Deterministic audit trails

**Field Deployment Modules:**
- **Environmental Monitor** (`environmental_monitor.rs`) - Rain, temperature, light sensing
  - Session viability assessment (Viable/Marginal/Infeasible)
  - Solar forecasting integration
  - Forces Passthrough Mode in adverse conditions

- **Power Manager** (`power_manager.rs`) - Battery/solar monitoring and throttling
  - Power modes: Normal (>80%), Medium (50-80%), Low (20-50%), Critical (<20%)
  - Solar prediction for task deferral decisions
  - Atomic flags for FPGA, source separation, synthesis throttling

- **Wildlife Sentry** (`wildlife_sentry.rs`) - Background species detection
  - FFT-based vocalization detection
  - Species signatures: marmoset, dolphin, bat, finch
  - Wake trigger generation with urgency levels
  - Debounce mechanism for rapid successive calls

- **Data Synchronizer** (`data_synchronizer.rs`) - Offline black box queuing
  - Priority-based sync (Critical > High > Normal > Low)
  - Bandwidth throttling
  - Multi-storage backend (SSD, USB, SD Card)
  - Compression support

- **Acoustic Simulator** (`acoustic_simulator.rs`) - TDD test fixture
  - Environmental noise generation (rain, wind, insects, birds)
  - SNR mixing for testing
  - Environment simulation (jungle, rainforest, open field)

#### 2. Python Logic Layer

**Cognitive Intelligence (`cognitive_intelligence/`):**
- `data_fusion.py` - Multi-modal data fusion
- `visual_fusion.py` - Cross-modal attention
- `siamese_network.py` - Similarity learning
- `train_asteroid_model.py` - Source separation training

**Real-time Processing (`realtime/` - Logic Layer Only):**
- `cognitive_layer.py` - Cognitive intelligence and decision making
- `adaptive_context_switcher.py` - Context interpretation
- `adaptive_resonance.py` - Adaptive resonance theory
- `deep_reinforcement_learning.py` - ML training
- `context_aware_synthesis.py` - Phrase selection logic
- `probabilistic_context_machine.py` - Decision making
- `phrase_audio_library.py` - Data management
- `unified_database.py` - Data access
- `task_management.py` - Orchestration

**Note:** 35 execution-layer Python files previously in `realtime/` have been archived. See `realtime/archive/ARCHIVE.md`.

#### 3. Data Layer (`data_models.py`)
- Unified data structures for cross-species compatibility
- Species: Marmoset, Egyptian Fruit Bat, Dolphin, Chimpanzee, Sperm Whale, Zebra Finch
- Key classes: `Phrase`, `AcousticFeatures`, `GrammarRule`, `VocalizationDatabase`

#### 4. Query Interface (`query_interface/`)
- High-performance query system with pre-built indexes
- Real-time search capabilities: F0 range, duration, similarity
- Grammar network analysis and cross-species comparisons
- Main entry: `VocalizationQueryInterface` and `get_query_interface()`

#### 5. Semiotic Analysis (`semiotics/`)
- Advanced cognitive intelligence capabilities
- Deception detection and innovation tracking
- Directed communication analysis
- Cross-modal attention fusion

### Key Architectural Patterns

1. **Execution vs. Logic Split**: Rust for time-critical, Python for cognitive
2. **Peer-to-Peer Supervision**: ZeroMQ heartbeats with systemd management
3. **Intent-Reality Mediation**: Master Controller translates intents to actions
4. **Universal Rosetta Stone**: Cross-species translation methodology
5. **Zero-Copy Architecture**: Minimizes data copying between Python/Rust
6. **Index-First Design**: Pre-built indexes for fast queries
7. **Fail-Open to Safety**: Python crashes trigger safe Passthrough Mode
8. **Field Survival Layer**: Environmental monitoring, power management, wildlife detection, offline queuing

### Operation Modes

**Passthrough Mode** (Safe Default):
- Python disconnected or heartbeats stopped
- Audio muted
- Raw audio recording continues
- Passive monitoring

**Interactive Mode** (Active):
- Python connected and sending heartbeats
- Processing intents from Python
- Synthesizing responses
- Full cognitive interaction

## Database and Data

### Database Structure
- Location: `src/vocalization_database.json` (2.5MB)
- Contains 2,882 phrases across 4 species
- Species distribution:
  - Marmoset: 1,351 phrases (harmonic communication)
  - Egyptian Fruit Bat: 516 phrases (FM sweep communication)
  - Dolphin: 387 phrases (whistle communication)
  - Chimpanzee: 628 phrases (mixed communication)

### Data Import
- Run `import_vocalization_data.py` to populate the database
- Validates species-specific acoustic features
- Builds query indexes automatically

## Development Guidelines

### Working with Rust Components
1. Always build with `--release` flag for performance
2. Rust is the **Execution Layer** - handle time-critical operations only
3. Python is the **Logic Layer** - cognitive decisions and learning
4. Prefer zero-copy patterns for large datasets
5. Run `cargo test` after changes to verify all tests pass

### Working with Peer Controller
1. Heartbeat interval: 20ms (50Hz)
2. Timeout: 100ms (5 missed heartbeats)
3. Always test mode switching behavior
4. Use systemd for production deployment
5. Monitor logs for heartbeat reception

### Performance Considerations
- Query interface uses pre-built indexes for speed
- Large datasets load in milliseconds
- Grammar network analysis is memory-efficient
- Use species-specific analyzers for optimal performance
- Rust operations are zero-copy where possible

### Testing Strategy
- Comprehensive test coverage with 266 Rust tests
- Field deployment features: 187 dedicated tests
  - Environmental Monitor: 46 tests
  - Power Manager: 54 tests
  - Wildlife Sentry: 24 tests
  - Data Synchronizer: 20 tests
  - Acoustic Simulator: 43 tests
- 50+ Python test files
- Separate test suites for each major component
- Integration tests for cross-component functionality
- Performance benchmarks for critical paths

### Important File Locations
- Core models: `src/data_models.py`
- Query interface: `src/query_interface/vocalization_query_interface.py`
- Semiotic engine: `src/semiotics/semiotic_engine.py`
- Rust execution layer: `technical_architecture/`
- Master controller: `technical_architecture/src/master_controller.rs`
- Peer controller: `technical_architecture/src/peer_controller.rs`
- **Field Deployment Modules:**
  - `technical_architecture/src/environmental_monitor.rs`
  - `technical_architecture/src/power_manager.rs`
  - `technical_architecture/src/wildlife_sentry.rs`
  - `technical_architecture/src/data_synchronizer.rs`
  - `technical_architecture/src/acoustic_simulator.rs`
- Deployment files: `technical_architecture/deployment/`
- TDD Plan: `technical_architecture/TDD_PLAN_FIELD_FEATURES.md`
- Database: `src/vocalization_database.json`
- Python tests: `tests/` (organized by component)
- Rust tests: `technical_architecture/src/` (inline with modules)

## Archive Information

The codebase has undergone significant cleanup to remove redundant and deprecated code.

### Root Archive (`/src/archive/`)
Contains deprecated directories superseded by active implementations:
- `jungle-monitoring-system/` - Superseded by `cognitive_intelligence/`
- `audio_engine/` - Unused Rust implementation
- `cognition/` - Superseded by `cognitive_intelligence/`
- `hybrid/` - Unused neural bridge
- `test_cache/` - Temporary cache files
- `duplicate_tests/` - 28 backup test files with `_1.py` suffix
- See `archive/ARCHIVE.md` for details

### Realtime Archive (`/src/realtime/archive/`)
Contains 35 execution-layer Python files moved to Rust:
- `advanced_synthesis_methods.py` → `technical_architecture/src/synthesis.rs`
- `enhanced_microharmonic_synthesizer.py` → Rust implementation
- `safety_manager.py` → `technical_architecture/src/safety.rs`
- `thermal_throttling_prevention.py` → `technical_architecture/src/thermal.rs`
- See `realtime/archive/ARCHIVE.md` for complete mappings

## Scientific Context

This framework enables:
1. **Deception detection** in animal communication
2. **Emergent behavior** identification and tracking
3. **Cross-modal analysis** (audio + visual + contextual)
4. **Universal translation** across species boundaries
5. **Cognitive modeling** of animal intelligence

The research impact focuses on understanding animal intelligence through vocalization patterns, moving beyond simple classification to cognitive understanding.

## Python API Usage

```python
from src import (
    Species, VocalizationModality,
    Phrase, AcousticFeatures,
    get_query_interface,
    SemioticEngine
)

# Query interface example
interface = get_query_interface()
results = interface.search_phrases_by_f0_range(5000, 10000)

# Semiotic analysis example
engine = SemioticEngine()
context = SemioticContext(species=Species.MARMOSET, ...)
result = engine.analyze_semiotics(phrase, context)
```

## Rust API Usage

```rust
use technical_architecture::{
    TechnicalArchitect, PeerController,
    OperationMode, PeerControllerConfig,

    // Field deployment modules
    EnvironmentalMonitor, PowerManager, WildlifeSentry,
    DataSynchronizer, AcousticSimulator,
};

// Create technical architect
let config = TechArchConfig::default();
let architect = TechnicalArchitect::new(config).await?;

// Create peer controller for heartbeat monitoring
let config = PeerControllerConfig::default();
let mut controller = PeerController::new(config)?;

// Create environmental monitor
let env_config = EnvironmentalMonitorConfig::default();
let env_monitor = EnvironmentalMonitor::new(env_config)?;

// Create power manager
let power_config = PowerManagerConfig::default();
let power_manager = PowerManager::new(power_config)?;

// Check if environmental conditions force passthrough
if env_monitor.forces_passthrough() {
    // Enter safe mode due to rain/temperature/light
}

// Check power state
let budget = power_manager.calculate_power_budget();
if power_manager.should_defer_heavy_tasks() {
    // Defer non-critical operations
}

// Create wildlife sentry for background detection
let sentry_config = WildlifeSentryConfig::default();
let sentry = WildlifeSentry::new(sentry_config);

// Process audio for wildlife detection
let audio = vec![0.0; 4800]; // 100ms at 48kHz
if let Some(trigger) = sentry.generate_wake_trigger(&audio)? {
    // Wake Python agent with detection results
    match trigger.urgency {
        TriggerUrgency::Critical => /* Immediate wake */ ,
        TriggerUrgency::High => /* Prioritized wake */ ,
        _ => /* Normal wake */ ,
    }
}

// Create data synchronizer for offline queuing
let sync_config = SyncConfig::default();
let sync = DataSynchronizer::new(sync_config)?;

// Queue log entry
let entry = LogEntry {
    timestamp: PtpTimestamp::from(chrono::Utc::now()),
    level: "INFO".to_string(),
    category: "detection".to_string(),
    message: "Marmoset detected".to_string(),
    data: None,
};
sync.queue_entry(entry, SyncPriority::High)?;

// Sync when network available
if sync.should_sync() {
    sync.sync()?;
}

// Main loop
loop {
    let mode = controller.tick()?;

    match mode {
        OperationMode::Passthrough => {
            // Safe mode - recording only
        }
        OperationMode::Interactive => {
            // Active mode - process Python intents
        }
    }
}
```

## Field Deployment Features

The Rust Execution Layer includes 5 critical field deployment modules that enable autonomous operation in harsh environments:

### 1. Environmental Monitor
Monitors environmental conditions and forces Passthrough Mode when conditions are unsuitable.

```rust
use technical_architecture::{EnvironmentalMonitor, EnvironmentalMonitorConfig};

let config = EnvironmentalMonitorConfig::default();
let monitor = EnvironmentalMonitor::new(config)?;

// Check current conditions
let conditions = monitor.current_conditions();
let viability = monitor.assess_session_viability();

match viability {
    SessionViability::Viable => /* Proceed with interaction */ ,
    SessionViability::Marginal => /* Proceed with caution */ ,
    SessionViability::Infeasible => /* Force Passthrough Mode */ ,
}

// Get optimal interaction windows from solar forecast
let windows = monitor.optimal_interaction_windows(2.0); // 2-hour minimum
```

**Key Features:**
- Rain intensity classification (None → Storm)
- Temperature classification (Freezing → Extreme)
- Light level classification (Dark → Night)
- Solar forecasting with optimal window calculation
- Forces Passthrough Mode in Heavy/Storm rain or Extreme temperatures

### 2. Power Manager
Monitors battery/solar state and throttles system power consumption to extend deployment time.

```rust
use technical_architecture::{PowerManager, PowerManagerConfig, PowerMode};

let config = PowerManagerConfig::default();
let manager = PowerManager::new(config)?;

// Get current power mode
let mode = manager.power_mode();
match mode {
    PowerMode::Normal => /* All features enabled (>80%) */ ,
    PowerMode::Medium => /* FPGA disabled (50-80%) */ ,
    PowerMode::Low => /* Conv-TasNet disabled (20-50%) */ ,
    PowerMode::Critical => /* Detection only (<20%) */ ,
}

// Get atomic flags for sharing with other modules
let fpga_enabled = manager.fpga_enabled_flag();
let separation_enabled = manager.source_separation_enabled_flag();
let synthesis_enabled = manager.synthesis_enabled_flag();

// Calculate power budget
let budget = manager.calculate_power_budget();
println!("Estimated runtime: {} hours", budget.estimated_runtime_hours);

// Check if should defer heavy tasks
if manager.should_defer_heavy_tasks() {
    // Defer non-critical operations
}
```

**Key Features:**
- Battery state tracking with health estimation
- Power mode management with automatic transitions
- Solar prediction integration for task deferral
- Atomic flags for module enable/disable (FPGA, separation, synthesis)
- Power budget calculation with runtime estimation

### 3. Wildlife Sentry
Low-power background detector that wakes the Python agent when target species are detected.

```rust
use technical_architecture::{WildlifeSentry, WildlifeSentryConfig};

let config = WildlifeSentryConfig::default();
let sentry = WildlifeSentry::new(config);

// Process audio buffer (100ms at 48kHz)
let audio = vec![0.0; 4800];
if let Some(trigger) = sentry.generate_wake_trigger(&audio)? {
    println!("Detected {} species", trigger.detections.len());
    println!("Urgency: {:?}", trigger.urgency);
    println!("Suggested response duration: {}ms", trigger.suggested_response_duration_ms);

    // Wake Python agent with detection results
}

// Get detection statistics
let (detections, triggers) = sentry.detection_stats();
```

**Key Features:**
- FFT-based frequency analysis for species detection
- Species signatures: marmoset (7-12kHz), dolphin (2-24kHz), bat (20-100kHz), finch (2-8kHz)
- Wake trigger generation with 4 urgency levels (Low, Medium, High, Critical)
- Debounce mechanism (500ms default) to prevent rapid successive triggers
- Multi-species simultaneous detection support

### 4. Data Synchronizer
Reliable offline "black box" queuing for intermittent network connections.

```rust
use technical_architecture::{DataSynchronizer, SyncConfig, LogEntry, SyncPriority};

let config = SyncConfig::default();
let sync = DataSynchronizer::new(config)?;

// Queue log entry with priority
let entry = LogEntry {
    timestamp: PtpTimestamp::from(chrono::Utc::now()),
    level: "INFO".to_string(),
    category: "detection".to_string(),
    message: "Marmoset detected".to_string(),
    data: None,
};
sync.queue_entry(entry, SyncPriority::Critical)?;

// Check sync status
let status = sync.sync_status();
println!("Queue size: {}", status.queue_size);
println!("Pending upload: {}", status.pending_upload);
println!("Total bytes queued: {}", status.total_bytes_queued);

// Sync when network available
if sync.should_sync() {
    let result = sync.sync()?;
    println!("Synced {} bytes", result.total_bytes_synced);
}
```

**Key Features:**
- Priority-based queue (Critical > High > Normal > Low)
- Bandwidth throttling (configurable kbps limit)
- Multi-storage backend support (LocalSSD, USBDrive, SDCard, NetworkMount)
- Compression support (bincode serialization)
- Automatic retry with configurable max retry count
- Queue persistence across process restarts

### 5. Acoustic Simulator
Test fixture for generating realistic environmental noise for TDD testing.

```rust
use technical_architecture::{AcousticSimulator, AcousticEnvironment};

let simulator = AcousticSimulator::new(48000, 42); // sample_rate, seed

// Generate rain noise
let rain = simulator.generate_rain_noise(10.0, 4800); // 10mm/h, 100ms

// Generate synthetic vocalization
let vocalization = simulator.generate_synthetic_vocalization(
    9000.0, // frequency_hz
    200.0,  // duration_ms
    Some(50.0) // modulation_rate_hz
);

// Simulate environment
let environment = AcousticEnvironment {
    environment_type: EnvironmentType::JungleDense,
    temperature_celsius: 28.0,
    humidity_percent: 85.0,
    wind_speed_m_s: 2.0,
    rain_intensity_mm_h: 5.0,
};

let simulated = simulator.simulate_environment(&vocalization, &environment)?;
```

**Key Features:**
- Noise generation (white, pink, brown, blue spectral colors)
- Environmental effects (rain, thunder, wind, insect chorus, bird chorus)
- SNR mixing for signal-to-noise ratio testing
- Environment simulation (10 predefined environments)
- Reproducible results with seed-based RNG
- Synthetic vocalization generation

### Field Deployment Test Results

```
running 266 tests
test result: ok. 266 passed; 0 failed; 0 ignored; 0 measured
```

All field deployment features are fully tested and ready for production use.
