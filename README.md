# Zoo Vox Rosetta Engine

**Universal Rosetta Stone Methodology for Cross-Species Vocalization Translation**

A revolutionary bioacoustic analysis framework that decodes animal communication through advanced acoustic algebra, cognitive intelligence, and cross-species pattern discovery. The Zoo Vox Rosetta Engine enables true translation between species by mapping vocalizations to a universal 30-dimensional feature space, revealing hidden semantic structures and enabling bidirectional communication synthesis.

**Core Innovation**: Beyond simple classification, Zoo Vox discovers the **grammar, syntax, and semantics** of animal languages through mathematical analysis of acoustic micro-dynamics, enabling translation between marmoset phee-calls, dolphin whistles, bat echolocation, and chimpanzee vocalizations.

## Architecture Overview

### Execution vs. Logic Split

The Zoo Vox Rosetta Engine follows a **hybrid architecture** combining Python and Rust:

- **Rust (Execution Layer)**: Time-critical operations, signal processing, hardware access, safety
  - Location: `technical_architecture/`
  - Zero-copy operations, memory safety, deterministic performance
  - **Field Survival**: Environmental monitoring, power management, wildlife detection, offline queuing

- **Python (Logic Layer)**: Cognitive intelligence, decision making, learning, context interpretation
  - Location: `cognitive_intelligence/`, `realtime/`, `semiotics/`
  - Rapid development, scientific computing, ML frameworks

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

---

## Project Structure

```
src/
├── __init__.py                        # Main package exports
├── data_models.py                     # Unified data structures (30D features)
├── vocalization_database.json         # Main database (2.5MB, 2,882 phrases)
├── CLAUDE.md                          # Project instructions for Claude Code
├── README.md                          # This file
├── pyproject.toml                     # Python package configuration
├── cleanup_technical_architecture.py  # Utility: Clean Rust build artifacts
├── compositional_validation.py        # Utility: Validate composition operations
├── consolidate_tests.py               # Utility: Consolidate duplicate tests
├── harmonic_affirmation.py            # Utility: Harmonic analysis
│
├── analysis/                          # ⭐ STEP 1: Acoustic-First Analysis
│   ├── sweet_spot_synthesis.py        # [NEW] Cross-species sweet spot analysis
│   └── rosetta_stone/                 #     Universal Rosetta Stone Engine
│       ├── universal_rosetta_stone.py #     Core acoustic analysis
│       ├── universal_synthesizer.py   #     Audio synthesis
│       ├── persona_mapping.py          # [NEW] PersonaRouter system
│       ├── persona_invariants_analysis.py # [NEW] Micro-dynamics profiling
│       ├── acoustic_algebra.py        # [NEW] Continuous acoustic field
│       ├── investigate_marmoset_cluster1.py # [NEW] Alarm vs Juvenile test
│       ├── demo_hybrid_persona_analysis.py # [NEW] Hybrid analysis demo
│       ├── demo_concatenative_vs_granular.py # [NEW] Synthesis comparison
│       ├── complete_extraction_pipeline.py # [NEW] Complete extraction
│       ├── VECTOR_DELTA_INTEGRATION.md # [NEW] Vector delta guide
│       └── demo_unknown_species.py    #     Demo for new species
│
├── technical_architecture/            # ✅ STEP 6: Rust Execution Layer (Active)
│   ├── src/
│   │   ├── synthesis.rs              # Audio synthesis engines (3 modes)
│   │   ├── source_separation.rs      # Conv-TasNet separator
│   │   ├── thermal.rs                # Thermal management
│   │   ├── safety.rs                 # Safety monitoring & watchdog
│   │   ├── ptp.rs                    # IEEE 1588 PTP timing
│   │   ├── logging.rs                # Provenance logging
│   │   ├── master_controller.rs     # Intent-Reality mediator
│   │   ├── peer_controller.rs       # ZeroMQ peer supervision
│   │   ├── island_hopping.rs         # 30D Vector Math, Navigation, Safety Clamping
│   │   ├── web_dashboard.rs          # Remote monitoring dashboard
│   │   ├── environmental_monitor.rs # Field: Rain/temp/light sensing
│   │   ├── power_manager.rs          # Field: Battery/solar management
│   │   ├── wildlife_sentry.rs        # Field: Background species detection
│   │   ├── data_synchronizer.rs      # Field: Offline black box queue
│   │   ├── acoustic_simulator.rs     # Field: TDD test fixture
│   │   ├── iacuc_compliance.rs       # IACUC protocol enforcement
│   │   ├── auto_calibration.rs       # Auto tone calibration
│   │   ├── shadow_model_monitor.rs   # ML drift detection
│   │   ├── multi_node_coordination.rs # Multi-node clusters
│   │   └── time_series_archive.rs    # Parquet time-series storage
│   ├── deployment/                   # Systemd deployment files
│   │   ├── rust-field-engine.service
│   │   ├── python-cognitive-agent.service
│   │   ├── python_heartbeat_client.py
│   │   └── README.md
│   └── Cargo.toml
│
├── cognitive_intelligence/            # ✅ Python Logic Layer (Active)
│   ├── data_fusion.py                # Multi-modal data fusion
│   ├── visual_fusion.py              # Cross-modal attention
│   ├── siamese_network.py            # Similarity learning
│   ├── train_asteroid_base.py        # Base training template
│   ├── train_asteroid_marmoset.py    # Marmoset-specific model (4-8kHz)
│   ├── train_asteroid_bat.py         # Bat-specific model (100-17000Hz)
│   ├── train_asteroid_dolphin.py     # Dolphin-specific model (500-16000Hz)
│   ├── train_asteroid_chimpanzee.py  # Chimpanzee-specific model (100-1900Hz)
│   ├── train_asteroid_multispecies.py # Multi-species training
│   └── ASTEROID_TRAINING_README.md   # Training guide
│
├── realtime/                          # ✅ Active (Logic Layer Only)
│   ├── audio_aware_grammar_discovery.py # [NEW] Grain-based phrase discovery & .pkl library
│   ├── online_phrase_discovery_agent.py # [NEW] Real-time KNN discovery for field deployment
│   ├── annotation_loader.py          # [NEW] Load ELAN/Praat/JSON/CSV behavioral annotations
│   ├── phrase_library_segment_extractor.py # [NEW] Extract audio from .pkl metadata
│   ├── cognitive_layer.py            # Cognitive intelligence
│   ├── adaptive_context_switcher.py # Context interpretation
│   ├── adaptive_resonance.py         # Adaptive resonance theory
│   ├── deep_reinforcement_learning.py # ML training
│   ├── context_aware_synthesis.py    # Phrase selection logic
│   ├── probabilistic_context_machine.py # Decision making
│   ├── phrase_audio_library.py       # Data management
│   ├── unified_database.py           # Data access
│   ├── task_management.py            # Orchestration
│   ├── persona_router.py             # [NEW] Persona selection from JSON
│   ├── persona_voice_synthesis_engine.py # [NEW] Voice switching synthesis
│   ├── metadata_synthesizer.py       # [NEW] Metadata-first synthesis (Ghost Words)
│   ├── hybrid_persona_synthesizer.py # [NEW] Hybrid persona generation
│   └── archive/                      # Archived execution-layer Python
│       └── ARCHIVE.md                # (35 files moved to Rust)
│
├── query_interface/                   # ✅ Active
│   ├── vocalization_query_interface.py
│   └── demo_query_interface.py
│
├── semiotics/                         # ✅ Active
│   ├── semiotic_engine.py
│   ├── demo_semiotic_engine.py
│   └── SEMIOTIC_DETECTION_GUIDE.md
│
├── synthesis/                         # ✅ Active
│   ├── advanced_harmonic_extensions.py
│   ├── advanced_phrase_synthesizer.py
│   └── __init__.py
│
├── data_import/                        # ✅ Active
│   ├── import_vocalization_data.py
│   └── __init__.py
│
├── scientific_validation/              # ✅ Active
│   ├── ab_testing_controller.py
│   └── provenance_tracer.py
│
├── system/                             # ✅ [NEW] Self-Healing System
│   ├── state_persistor.py              # Checkpoint/recovery persistence
│   ├── self_heal.py                    # Autonomous crash recovery
│   └── __init__.py
│
├── tests/                              # ✅ Active (41 test files, 500+ tests)
│   ├── test_acoustic_algebra.py       # Acoustic algebra tests
│   ├── test_vector_delta_synthesis.py # Vector delta synthesis tests
│   ├── test_30d_metadata_synthesis.py # 30D metadata synthesis tests
│   ├── test_rust_island_hopping.py    # Python-Rust integration tests
│   ├── test_state_persistor.py        # Checkpoint/recovery system tests
│   ├── test_self_heal.py              # Self-healing system tests
│   ├── test_hybrid_persona_architecture.py # Hybrid persona tests
│   ├── test_rosetta_stone_base.py
│   ├── test_realtime_system_population.py
│   ├── test_zero_copy_rust.py
│   ├── test_visual_fusion.py          # MediaPipe + OpenCV integration
│   ├── test_cognitive_interaction_engine.py
│   ├── test_hybrid_cognitive_stack.py
│   ├── test_*.py                      # 40+ additional test files
│   └── test_island_hopping_integration.rs # Rust integration tests
│
├── analysis_output/                   # [NEW] Analysis outputs & visualizations
│   ├── persona_invariants.json        # Persona statistical profiles
│   ├── persona_database.json          # Persona definitions
│   ├── sweet_spot_comparison_7khz.png # Cross-species spectrogram
│   ├── sweet_spot_metrics.json        # Quantitative metrics
│   ├── hybrid_marmoset_contact_alarm.wav # Hybrid audio
│   ├── hybrid_bat_nav_social.wav     # Hybrid audio
│   ├── hybrid_marmoset_feature_blend.wav # Hybrid audio
│   ├── marmoset_7khz_phee.wav         # Sweet spot synthesis
│   └── bat_7khz_call.wav              # Sweet spot synthesis
│
├── archive/                            # ✅ Archived Content
│   ├── deprecated_python_fallbacks/   # Python code superseded by Rust
│   │   ├── INTERPOLATION_EXTRAPOLATION_DEPRECATION.md
│   │   └── ARCHIVE.md
│   ├── jungle-monitoring-system/      # Deprecated duplicate
│   ├── audio_engine/                   # Unused Rust implementation
│   ├── cognition/                      # Superseded by cognitive_intelligence
│   ├── hybrid/                         # Unused neural bridge
│   ├── test_cache/                     # Temporary cache files
│   ├── duplicate_tests/                # Backup test files
│   ├── experimental_analysis/          # Experimental analysis code
│   ├── experimental_realtime/          # Experimental realtime code
│   └── old_reports/                    # Archived reports
│
├── .github/                            # GitHub CI/CD configuration
│   └── workflows/
│       └── ci.yml                      # Python + Rust testing pipeline
│
└── docs/                              # Additional documentation
    ├── PERSONA_INVARIANTS_SUMMARY.md   # Persona pipeline documentation
    └── VECTOR_DELTA_INTEGRATION.md    # Vector delta integration guide
```

---

## Technical Architecture

The Zoo Vox Rosetta Engine follows a **hybrid architecture** with clear separation of concerns:

### Architecture Principles

| Layer | Responsibility | Technology | Performance |
|-------|---------------|------------|-------------|
| **Rust Execution Layer** | Time-critical operations, safety, hardware control | Rust | Deterministic, zero-copy |
| **Python Logic Layer** | Cognitive intelligence, learning, context interpretation | Python | Rapid development, ML frameworks |
| **Communication** | Peer-to-peer supervision via ZeroMQ heartbeats | ZeroMQ + Systemd | Fail-safe coordination |

---

## Rust Execution Layer (`technical_architecture/`)

The Rust execution layer provides deterministic performance, memory safety, and hardware-level control for time-critical operations. It can continue operating safely even if Python crashes (Passthrough Mode).

### Core Modules

#### 1. Audio Processing (`synthesis.rs`)

**Three Synthesis Modes:**

| Mode | Description | Use Case | t-SNE Score |
|------|-------------|----------|-------------|
| **Concatenative** | Direct phrase concatenation | Natural playback | 4.208 |
| **Granular** | Grain-based with parameter variation | Systematic experiments | 6.452 ✅ |
| **Superpositional** | Vertical layering | Complex harmonies | Variable |
| **Combined** | Mixed horizontal/vertical | Full encoding | Flexible |

**Vector Delta Commands**

The Rust synthesis engine now supports **Vector Delta commands** for integration with Acoustic Algebra:

```rust
use technical_architecture::{GranularConcatenativeSynthesizer, SourceMetadata};

// Create synthesizer
let synth = GranularConcatenativeSynthesizer::new(22050);

// Load source with metadata (enables delta commands)
let metadata = SourceMetadata {
    mean_f0_hz: 6800.0,
    duration_ms: 50.0,
    f0_range_hz: 400.0,
};
synth.load_source_with_metadata(audio_buffer, metadata);

// Apply VECTOR DELTA commands (relative to source!)
synth.shift_pitch_by_hz(200.0);   // 6800 + 200 = 7000Hz ✅
synth.shift_duration_by_ms(-10.0); // 50 - 10 = 40ms ✅
synth.apply_vector_delta(200.0, -10.0, 100.0); // All shifts at once

let output = synth.synthesize(50.0);
```

**Why Delta Commands Matter:**

```
Bad (Absolute): "Set pitch to 7000Hz"
  ❌ Ignores that source started at 6800Hz
  ❌ Ignores that source started at 7200Hz

Good (Delta): "Shift pitch by +200Hz"
  ✅ 6800Hz + 200Hz = 7000Hz
  ✅ 7200Hz - 200Hz = 7000Hz
  (Same target, different delta!)
```

**Concatenative vs Granular Comparison** [NEW]:

Both methods can create the same phrase sequences for validation:

| Aspect | Concatenative | Granular | Improvement |
|--------|---------------|----------|-------------|
| **Fidelity** | Perfect (t-SNE: 4.208) | Near-perfect (t-SNE: 6.452) | 76.1% better than additive |
| **Flexibility** | None | High (any pitch/duration) | Enables acoustic algebra |
| **Formant Preservation** | ✅ Perfect | ✅ Excellent | Both preserve spectral envelope |
| **Use Case** | Baseline validation | Delta-based synthesis | Scientific comparison |

**Validation Workflow:**

```rust
// Scenario: Virtual phrase F0=6750Hz, nearest real F0=6500Hz
let delta_f0 = 6750.0 - 6500.0; // +250Hz

// Method 1: Concatenative (baseline)
let concat_output = load_and_play("nearest_phrase.wav"); // F0=6500Hz
// Error from target: 250Hz

// Method 2: Granular (delta-based)
synth.load_source_with_metadata(audio, metadata);
synth.shift_pitch_by_hz(250.0); // Apply delta
let granular_output = synth.synthesize(50.0); // F0≈6750Hz
// Error from target: 0.5% BETTER than concatenative! ✅
```

See `analysis/rosetta_stone/VECTOR_DELTA_INTEGRATION.md` for complete integration guide.

**Dynamic Microharmonic Synthesis** (NEW):
- Real-time F0 modulation for expressive synthesis
- Combines horizontal sequencing with vertical harmonic layering
- Configurable harmonic sweep rates and depth

#### 2. Source Separation (`source_separation.rs`)

Species-specific Conv-TasNet models via ONNX/Tract:

```rust
use technical_architecture::{ConvTasNetSeparator, SeparatorConfig};

let config = SeparatorConfig {
    model_path: "models/checkpoints/marmoset/conv_tasnet_marmoset.onnx",
    filter_range: (2800, 10400),  // Bandpass for marmoset
    ..Default::default()
};

let mut separator = ConvTasNetSeparator::new(config)?;
let separated = separator.separate(&mixed_audio)?;
```

**Multi-Species Auto-Detection:**
```rust
impl SpeciesSeparator {
    pub fn detect_species(&self, audio: &[f32]) -> Option<String> {
        match dominant_f0 {
            100..=1900 => Some("chimpanzee"),
            2800..=10400 => Some("marmoset"),
            100..=22100 => Some("egyptian_bat"),
            350..=20800 => Some("dolphin"),
            _ => None,
        }
    }
}
```

#### 3. Web Dashboard (`web_dashboard.rs`) [NEW]

Remote monitoring dashboard for field deployments with secure HTTPS/WebSocket support:

**Features:**
- Real-time system monitoring (battery, temperature, uptime)
- Manual override and emergency stop commands
- IACUC compliance status tracking
- Calibration status and health monitoring
- Live spectrogram streaming
- Command audit logging for all interventions
- JWT-based authentication

**Dashboard State:**
```rust
pub struct DashboardState {
    pub operation_mode: DashboardOperationMode,  // Passthrough, Interactive, Maintenance, Emergency
    pub battery_level: f32,                       // 0-100%
    pub temperature_celsius: f32,
    pub uptime_seconds: u64,
    pub active_connections: usize,
    pub iacuc_status: IacucStatus,                // Compliant, Warning, Violation
    pub calibration_status: CalibrationDashboardStatus,
    pub last_updated: PtpTimestamp,
}
```

**WebSocket Messages:**
```rust
pub enum WsMessage {
    Spectrogram { data: Vec<f32>, sample_rate: u32 },
    GaugeUpdate { name: String, value: f32, unit: String },
    StatusUpdate { status: DashboardState },
    Error { message: String },
    Info { message: String },
}
```

**Dashboard Commands:**
```rust
pub enum DashboardCommand {
    EmergencyStop,
    ManualOverride { intent: String },
    SetParameter { name: String, value: serde_json::Value },
    RunCalibration,
    GetStatus,
    SubscribeSpectrogram,
    UnsubscribeSpectrogram,
}
```

**Usage:**
```rust
use technical_architecture::{WebDashboard, DashboardConfig};

let config = DashboardConfig {
    bind_address: "0.0.0.0:8443".to_string(),
    tls_cert_path: PathBuf::from("/etc/dashboard/cert.pem"),
    tls_key_path: PathBuf::from("/etc/dashboard/key.pem"),
    auth_secret: "production_secret".to_string(),
    token_expiry_hours: 24,
    max_connections: 10,
    enable_tls: true,
};

let dashboard = WebDashboard::new(config);
dashboard.start().await?;

// Authenticate client
let token = dashboard.authenticate("admin", "password")?;

// Connect WebSocket client
dashboard.connect_client("client_123", "192.168.1.100", &token.token)?;

// Process commands
let result = dashboard.process_command(
    DashboardCommand::EmergencyStop,
    "admin",
    "192.168.1.100"
);
```

#### 4. IACUC Compliance (`iacuc_compliance.rs`) [NEW]

Enforces ethical research protocols for field deployments:

**Features:**
- Time-based operation windows (no nighttime calls)
- Daily interaction limits per species
- Emergency contact integration
- Policy violation detection and logging
- Protocol approval tracking

**Compliance Checks:**
```rust
pub struct IacucComplianceEngine {
    pub protocols: HashMap<String, IacucProtocol>,
    pub time_windows: HashMap<String, TimeWindow>,
    pub daily_limits: DailyLimits,
    pub emergency_contacts: Vec<EmergencyContact>,
}

impl IacucComplianceEngine {
    pub fn check_intent(&self, intent: &IacucIntent) -> ComplianceCheck {
        // Check time window
        if !self.is_within_allowed_hours() {
            return ComplianceCheck::Rejected {
                reason: "Outside approved operation hours".to_string()
            };
        }

        // Check daily limits
        if self.exceeds_daily_limit(&intent.species) {
            return ComplianceCheck::Rejected {
                reason: "Daily interaction limit exceeded".to_string()
            };
        }

        ComplianceCheck::Approved
    }
}
```

#### 5. Auto Calibration (`auto_calibration.rs`) [NEW]

Automated speaker calibration using onboard test tones:

**Features:**
- Multi-frequency calibration sweeps (1kHz, 5kHz, 10kHz)
- Gain adjustment calculation
- Frequency response curve measurement
- Drift detection and compensation
- Health status monitoring

**Calibration Process:**
```rust
pub struct CalibrationEngine {
    pub config: CalibrationConfig,
    pub calibration_history: Vec<CalibrationResult>,
}

impl CalibrationEngine {
    pub async fn run_calibration(&mut self) -> Result<CalibrationResult> {
        // Generate calibration tones
        let tones = vec![
            CalibrationTone { frequency: 1000.0, duration_ms: 500.0 },
            CalibrationTone { frequency: 5000.0, duration_ms: 500.0 },
            CalibrationTone { frequency: 10000.0, duration_ms: 500.0 },
        ];

        // Play and record each tone
        for tone in tones {
            let recording = self.play_and_record(tone).await?;
            let frequency_response = self.analyze_response(&recording);
            let gain_adjustment = self.calculate_gain_adjustment(&frequency_response);
            self.apply_gain(gain_adjustment)?;
        }

        Ok(CalibrationResult {
            success: true,
            drift_db: Some(measured_drift),
            health_status: "Healthy".to_string(),
        })
    }
}
```

#### 6. Shadow Model Monitor (`shadow_model_monitor.rs`) [NEW]

Detects ML model drift using shadow model comparison:

**Features:**
- Real-time prediction comparison (active vs shadow)
- Statistical drift detection (KL divergence)
- Alert system for significant drift
- Model version tracking

**Drift Detection:**
```rust
pub struct ShadowModelMonitor {
    pub active_model: Box<dyn InferenceModel>,
    pub shadow_model: Box<dyn InferenceModel>,
    pub config: ShadowModelConfig,
}

impl ShadowModelMonitor {
    pub fn monitor_prediction(&mut self, input: &InputFeatures) -> DriftAlert {
        let active_pred = self.active_model.predict(input);
        let shadow_pred = self.shadow_model.predict(input);

        let comparison = ModelComparison {
            active_prediction: active_pred,
            shadow_prediction: shadow_pred,
            difference: (active_pred - shadow_pred).abs(),
        };

        if comparison.difference > self.config.alert_threshold {
            DriftAlert::SignificantDrift {
                difference: comparison.difference,
                confidence: self.calculate_drift_confidence(&comparison),
            }
        } else {
            DriftAlert::NoDrift
        }
    }
}
```

#### 7. Multi-Node Coordination (`multi_node_coordination.rs`) [NEW]

Manages clusters of field devices with synchronized timing:

**Features:**
- IEEE 1588 PTP clock synchronization
- TDMA slot allocation for interference avoidance
- Fused detection data from multiple nodes
- Location estimation using TDOA
- Leader election for autonomous operation

**Cluster Configuration:**
```rust
pub struct MultiNodeCoordinator {
    pub config: ClusterConfig,
    pub nodes: HashMap<NodeId, NodeInfo>,
    pub schedule: TdmaSchedule,
}

impl MultiNodeCoordinator {
    pub async fn elect_leader(&mut self) -> ElectionResult {
        // PTP-based leader election
        let best_clock = self.nodes
            .values()
            .filter(|n| n.is_active())
            .min_by_key(|n| n.clock_accuracy);

        ElectionResult::NewLeader { node_id: best_clock.id }
    }

    pub fn estimate_location(&self, detections: &[FusedDetectionData]) -> LocationEstimate {
        // Time Difference of Arrival (TDOA) localization
        let tdoa_values = self.calculate_tdoa(detections);
        LocationEstimate::from_tdoa(tdoa_values)
    }
}
```

#### 8. Time-Series Archive (`time_series_archive.rs`) [NEW]

Efficient Parquet-based time-series storage for long-term data retention:

**Features:**
- Parquet compression for storage efficiency
- Configurable retention policies
- Storage quota management
- Batch writing for performance
- Query optimization by time range

**Archive Configuration:**
```rust
pub struct TimeSeriesArchiver {
    pub config: TimeSeriesConfig,
    pub storage_stats: StorageStats,
}

impl TimeSeriesArchiver {
    pub fn archive_batch(&mut self, batch: TimeSeriesBatch) -> Result<()> {
        // Check quota
        if self.storage_stats.used_bytes + batch.estimated_size() > self.config.max_storage_bytes {
            self.apply_retention_policy()?;
        }

        // Write to Parquet
        let parquet_path = format!("{}/{}.parquet",
            self.config.storage_path,
            batch.timestamp.format("%Y-%m-%d")
        );

        self.write_parquet(&batch, &parquet_path)?;
        Ok(())
    }
}
```

### Field Survival Modules

These modules enable autonomous operation in remote field environments:

#### Environmental Monitor
- Rain detection (light, moderate, heavy intensity)
- Temperature classification (freezing to extreme heat)
- Light level detection (day/night/twilight)
- Session viability assessment

#### Power Manager
- Battery state tracking (voltage, current, capacity)
- Solar power prediction
- Adaptive power throttling
- Power budget allocation

#### Wildlife Sentry
- Background species detection (to avoid masking target species)
- Wake trigger urgency classification
- Species-specific acoustic signatures

#### Data Synchronizer
- Offline black box queue (up to 24 hours of data)
- Prioritized synchronization (critical > normal > low)
- Multiple storage backends (local, cloud, edge)
- Resume capability after network restoration

### PyO3 Integration

All Rust modules are accessible from Python via PyO3 bindings. The framework provides **17 Python classes** for Rust-Python integration:

```python
from technical_architecture import (
    # Core Components
    TechnicalArchitect,
    DynamicMicroharmonicSynthesizer,
    GranularConcatenativeSynthesizer,
    SourceMetadata,              # Vector delta synthesis metadata

    # Safety-Critical Components
    OperationMode,              # Passthrough/Interactive modes
    PeerController,             # Heartbeat supervision
    PeerControllerConfig,
    ThermalState,               # Thermal state management

    # Environmental Monitoring
    EnvironmentalMonitor,        # Field deployment environment monitoring
    EnvironmentalMonitorConfig,
    EnvironmentalConditions,    # Current conditions data
    SessionViability,           # Viable/Marginal/Infeasible
    RainIntensity,              # Rain classification
    TemperatureClassification,   # Temperature classification

    # Visual Recording
    VisualRecorder,
    VisualRecorderConfig,
    VisualMetadata,
    RecordingStatistics,
    AudioSyncEvent,
)

# === Safety-Critical Components ===

# PeerController: Heartbeat supervision for "Fail-Open to Safety"
controller = PeerController(PeerControllerConfig(
    heartbeat_timeout_ms=100,  # 100ms timeout
    poll_interval_ms=10,
))

mode = controller.tick()
if mode.is_passthrough():
    # Python crashed or heartbeat stopped - audio muted
    print("In safe Passthrough mode")
elif mode.is_interactive():
    # Python alive and sending heartbeats - full operation
    print("In Interactive mode")

# ThermalState: Thermal constraint checking
thermal_state = ThermalState.critical()
if thermal_state.requires_throttling():
    # Block synthesis to prevent overheating
    print("Synthesis blocked - thermal throttling required")

# EnvironmentalMonitor: Field deployment environmental sensing
monitor = EnvironmentalMonitor.for_testing()

# Check if conditions force safe mode
if monitor.forces_passthrough():
    print("Environmental conditions force Passthrough mode")

# Simulate storm conditions
storm = EnvironmentalConditions(
    temperature_celsius=22.0,
    rain_intensity_mm_h=60.0,  # Storm
)
monitor.set_conditions(storm)

if monitor.forces_passthrough():
    print("Storm detected - switching to safe mode")

# === Synthesis ===

# Dynamic Microharmonic Synthesizer (direct synthesis)
synth = DynamicMicroharmonicSynthesizer(sample_rate=48000)
audio = synth.synthesize_phrase(
    f0_base=6000.0,
    duration_ms=100.0,
    attack_ms=5.0,
    decay_ms=10.0,
    sustain_level=0.8,
)

# Granular Concatenative Synthesizer with Vector Delta Commands
# Use this for Acoustic Algebra integration (delta-based shifts)

# === 30D Micro-Dynamics Metadata (Synthesis API) ===
# Load source audio with 30D metadata for vector delta commands
from technical_architecture import SourceMetadata

metadata = SourceMetadata(
    # Fundamental (3 features)
    mean_f0_hz=6800.0,
    duration_ms=50.0,
    f0_range_hz=400.0,
    # Grit Factors (3 features) - Timbre texture
    harmonic_to_noise_ratio=20.0,   # 20 dB HNR (tonal)
    spectral_flatness=0.1,            # 0.1 (very tonal)
    harmonicity=0.8,                  # 0.8 (highly harmonic)
    # Motion Factors (7 features) - Envelope dynamics
    attack_time_ms=10.0,
    decay_time_ms=15.0,
    sustain_level=0.7,
    vibrato_rate_hz=8.0,
    vibrato_depth=50.0,
    jitter=0.02,
    shimmer=0.03,                     # NEW: Amplitude micro-variations
    # Fingerprint Factors (14 features) - Spectral shape
    mfcc_1=-500.0,
    mfcc_2=-100.0,
    mfcc_3=-50.0,
    mfcc_4=-20.0,
    mfcc_5=-0.5,                      # NEW
    mfcc_6=-0.3,                      # NEW
    mfcc_7=-0.2,                      # NEW
    mfcc_8=-0.1,                      # NEW
    mfcc_9=0.0,                       # NEW
    mfcc_10=0.1,                      # NEW
    mfcc_11=0.2,                      # NEW
    mfcc_12=0.3,                      # NEW
    mfcc_13=0.4,                      # NEW
    spectral_flux=0.5,                # REPLACES spectral_contrast
    # Rhythm Factors (3 features) - Temporal patterns
    median_ici_ms=0.0,              # Not applicable for harmonic calls
    onset_rate_hz=0.0,              # Not applicable for harmonic calls
    ici_coefficient_of_variation=0.0,
)
granular_synth.load_source_with_metadata(audio_buffer, metadata)

# === Builder Pattern (Partial Metadata) ===
# For partial specification, use builder pattern (other features use defaults)
contact_metadata = SourceMetadata.builder() \
    .mean_f0_hz(7000.0) \
    .duration_ms(70.0) \
    .harmonic_to_noise_ratio(25.0) \
    .jitter(0.01) \
    .build()

# === Vector Delta Commands ===
# Apply delta (relative to source!)
# This integrates with Acoustic Algebra: delta = virtual - nearest_real
granular_synth.shift_pitch_by_hz(200.0)      # 6800 + 200 = 7000Hz
granular_synth.shift_duration_by_ms(-10.0)   # 50 - 10 = 40ms

# Or apply all shifts at once (legacy 3D method)
granular_synth.apply_vector_delta(
    delta_f0_hz=200.0,         # Pitch shift
    delta_duration_ms=-10.0,   # Duration shift
    delta_f0_range_hz=100.0    # F0 range shift
)

# Synthesize at target duration
output = granular_synth.synthesize(duration_ms=40.0)

# === Visual Recording ===

# Create visual recorder
recorder = VisualRecorder(VisualRecorderConfig(
    camera_id=0,
    resolution=(1920, 1080),
    fps=30.0,
))

# Start recording session
session_id = recorder.start_session("test_session")
recorder.register_audio_event(AudioSyncEvent(
    timestamp_ns=time.time_ns(),
    event_type="PhraseStart",
    phrase_key="F0_7400",
    frame_index=0,
))

recorder.stop_session()
stats = recorder.get_statistics()
```

**Safety-Critical Features**:
- **PeerController**: ZeroMQ heartbeat monitoring with automatic mode switching
- **ThermalState**: Four-state thermal classification (Normal/Warning/Throttling/Critical)
- **EnvironmentalMonitor**: Rain, temperature, and light sensing with automatic Passthrough triggering

---

## Python Logic Layer

The Python logic layer handles cognitive intelligence, decision making, learning, and context interpretation. It can crash safely without affecting Rust operations.

### Core Modules

#### 1. Cognitive Intelligence (`cognitive_intelligence/`)

**Multi-modal data fusion and machine learning:**

| Module | Purpose | Key Features |
|--------|---------|--------------|
| `data_fusion.py` | Multi-modal sensor fusion | Audio + visual + contextual integration |
| `visual_fusion.py` | Cross-modal attention | MediaPipe integration for gaze/gesture tracking |
| `siamese_network.py` | Similarity learning | Metric learning for phrase comparison |
| `train_asteroid_*.py` | Source separation training | Species-specific Conv-TasNet models |

**Visual Fusion with MediaPipe:**
```python
from cognitive_intelligence.visual_fusion import VisualFusionSystem, VisualFusionConfig

# Create visual fusion system
config = VisualFusionConfig(
    camera_resolution=(640, 480),
    fps=30,
    use_mediapipe=True,
    separate_thread=True,
)
fusion_system = VisualFusionSystem(config)

# Process frames for attention level
visual_features = fusion_system.process_frame(frame)

# Integrate with audio features
audio_features = {"rms": 0.1, "f0": 6000.0, "context": "contact_call"}
fused_result = fusion_system.integrate_with_audio(audio_features, visual_features)
```

#### 2. Real-time Processing (`realtime/`)

**Cognitive layer and decision making:**

| Module | Purpose | Key Features |
|--------|---------|--------------|
| `cognitive_layer.py` | Central cognitive intelligence | Decision making, intent generation |
| `adaptive_context_switcher.py` | Context interpretation | Adaptive context-aware processing |
| `context_aware_synthesis.py` | Phrase selection logic | Context-aware response generation |
| `phrase_audio_library.py` | Data management | Phrase library with metadata |
| `unified_database.py` | Data access | Unified database interface |
| `persona_router.py` | Persona selection | JSON-based persona routing |
| `metadata_synthesizer.py` | Metadata-first synthesis | 30D Ghost Word synthesis |
| `hybrid_persona_synthesizer.py` | Hybrid persona generation | Multi-persona blending |

**Cognitive Layer Example:**
```python
from realtime.cognitive_layer import CognitiveLayer
from realtime.context_aware_synthesis import ContextAwareSynthesis

# Initialize cognitive layer
cognitive = CognitiveLayer()
synthesizer = ContextAwareSynthesis()

# Process incoming vocalization
context = cognitive.interpret_context(audio_features, visual_context)

# Generate response phrase
response = synthesizer.generate_response(
    context=context,
    target_species="marmoset",
    intent="contact_call"
)
```

#### 3. Semiotic Analysis (`semiotics/`)

**Advanced cognitive capabilities:**

| Module | Purpose | Key Features |
|--------|---------|--------------|
| `semiotic_engine.py` | Semiotic analysis | Deception detection, innovation tracking |
| `demo_semiotic_engine.py` | Demo & examples | Usage examples |

**Semiotic Engine Capabilities:**
```python
from semiotics import SemioticEngine, SemioticContext

# Initialize semiotic engine
engine = SemioticEngine()

# Analyze semiotics
context = SemioticContext(
    species=Species.MARMOSET,
    behavioral_context="foraging",
    social_rank="subordinate",
)
result = engine.analyze_semiotics(phrase, context)

# Detect deception
if result.deception_probability > 0.7:
    print(f"Deception detected: {result.deception_type}")
```

#### 4. Query Interface (`query_interface/`)

**High-performance query system:**

```python
from query_interface import get_query_interface

# Get query interface (auto-initializes)
interface = get_query_interface()

# Search by F0 range
results = interface.search_phrases_by_f0_range(5000, 10000)

# Search by duration
results = interface.search_phrases_by_duration(40, 80)

# Find similar phrases
similar = interface.find_similar_phrases(target_phrase, threshold=0.8)

# Grammar network analysis
grammar_network = interface.build_grammar_network()
centrality = interface.calculate_centrality_measures(grammar_network)
```

#### 5. Analysis (`analysis/`)

**Acoustic-first analysis and Rosetta Stone engine:**

| Module | Purpose | Key Features |
|--------|---------|--------------|
| `rosetta_stone/universal_rosetta_stone.py` | Core acoustic analysis | Phrase segmentation, grammar discovery |
| `rosetta_stone/acoustic_algebra.py` | Continuous acoustic field | 30D vector space operations |
| `rosetta_stone/persona_mapping.py` | PersonaRouter system | Persona-to-phrase mapping |
| `rosetta_stone/persona_invariants_analysis.py` | Micro-dynamics profiling | Statistical persona profiling |
| `sweet_spot_synthesis.py` | Cross-species analysis | Sweet spot detection |

**Universal Rosetta Stone:**
```python
from analysis.rosetta_stone import UniversalRosettaStone

# Initialize analyzer
analyzer = UniversalRosettaStone(sample_rate=48000)

# Process raw audio → phrases
phrases = analyzer.segment_phrases(audio_data)

# Build vocabulary (atomic units)
vocabulary = analyzer.build_vocabulary(phrases, f0_bin_size=200)

# Discover grammar rules
grammar = analyzer.discover_grammar(phrases)

# Detect sentences (phrase sequences)
sentences = analyzer.discover_sentences(phrases, gaps)
```

### Data Models (`data_models.py`)

**Unified 30D data structures for cross-species compatibility:**

```python
from src import Species, VocalizationModality, Phrase, AcousticFeatures

# Create phrase with 30D features
phrase = Phrase(
    phrase_key="F0_6400_DUR_50_RANGE_0",
    species=Species.MARMOSET,
    modality=VocalizationModality.HARMONIC,
    features=AcousticFeatures(
        # Fundamental (3)
        mean_f0_hz=6400.0,
        duration_ms=50.0,
        f0_range_hz=0.0,
        # Grit Factors (3)
        harmonic_to_noise_ratio=20.0,
        spectral_flatness=0.1,
        harmonicity=0.8,
        # Motion Factors (7)
        attack_time_ms=10.0,
        decay_time_ms=15.0,
        sustain_level=0.7,
        vibrato_rate_hz=8.0,
        vibrato_depth=50.0,
        jitter=0.02,
        shimmer=0.03,
        # Fingerprint Factors (14)
        mfcc_1=-500.0,
        mfcc_2=-100.0,
        # ... mfcc_3 through mfcc_13
        spectral_flux=0.5,
        # Rhythm Factors (3)
        median_ici_ms=0.0,
        onset_rate_hz=0.0,
        ici_coefficient_of_variation=0.0,
    ),
)
```

---

## Complete Research Workflow

The framework follows a **six-step workflow** from raw audio to cognitive intelligence:

### STEP 1: Acoustic-First Analysis (`analysis/rosetta_stone/`)

**The Starting Point** - Extract phrase, sentence, and grammar information from raw audio.

```python
from analysis.rosetta_stone import UniversalRosettaStone

# Initialize analyzer
analyzer = UniversalRosettaStone(sample_rate=48000)

# Process raw audio → phrases
phrases = analyzer.segment_phrases(audio_data)

# Build vocabulary (atomic units)
vocabulary = analyzer.build_vocabulary(phrases, f0_bin_size=200)

# Discover grammar rules
grammar = analyzer.discover_grammar(phrases)

# Detect sentences (phrase sequences)
sentences = analyzer.discover_sentences(phrases, gaps)
```

**Output Database Format:**
- **Phrase Keys**: `F0_6400_DUR_50_RANGE_0` (binned acoustic features)
- **Sentences**: Groups of phrases with timestamps
- **Grammar**: Transition patterns between phrase types

**Key Classes:**
- `PhraseSignature` - Acoustic phrase representation with modality detection
- `Sentence` - Individual vocalization containing phrases
- `UniversalRosettaStone` - Main analysis engine

---

### STEP 1.5: Atomic Word Discovery Using Micro-Dynamics

**Enhanced Method** - Discover smallest semantic units (atomic words) using multi-dimensional acoustic features beyond simple F0 binning.

#### Traditional Approach vs. Micro-Dynamics

| Approach | Features | Phrase Key Example | Limitation |
|----------|----------|-------------------|------------|
| **Traditional** | F0, Duration, Range | `F0_7400_DUR_50_RANGE_300` | Groups dissimilar sounds with same F0 |
| **Micro-Dynamics** | 30 acoustic features | Multi-dimensional persona matching | Requires feature extraction |

#### Micro-Dynamics Feature Categories

**1. Grit Factors** (Timbre texture)
- `harmonic_to_noise_ratio` - Harmonic purity vs noise
- `spectral_flatness` - Noise-like vs tonal

**2. Motion Factors** (Envelope dynamics)
- `attack_time_ms` - Onset speed (fast = sharp, slow = gentle)
- `decay_time_ms` - Release speed
- `sustain_level` - Steady-state amplitude
- `vibrato_rate_hz` - Pitch modulation frequency
- `vibrato_depth` - Pitch modulation depth
- `jitter` - Micro-perturbations (instability vs stability)

**3. Fingerprint Factors** (Spectral shape)
- `mfcc_1` through `mfcc_4` - Mel-frequency cepstral coefficients
- `spectral_contrast` - Formant structure strength

**4. Rhythm Factors** (Temporal patterns)
- `median_ici_ms` - Inter-click interval
- `onset_rate_hz` - Click/event rate
- `ici_coefficient_of_variation` - Rhythm regularity

#### Acoustic Personas for Semantic Discovery

The framework defines **6 acoustic personas** that map acoustic features to semantic meaning:

| Persona | Semantic Category | Key Features | Example Context |
|---------|-----------------|--------------|-----------------|
| **GRITTY** | Aggressive alerts | Low HNR, high flatness, fast attack | Threat, confrontation |
| **PURE** | Contact/affiliation | High HNR, low flatness, slow attack | Food sharing, bonding |
| **BOUNCY** | Courtship/play | High vibrato, low jitter, pulsed | Mating, social play |
| **SHARP** | Alarm/startle | Very fast attack/decay, high contrast | Predator detection |
| **SUSTAINED** | Territory/long-range | Slow attack/decay, high sustain | Territorial claims |
| **TRANSIENT** | Rhythmic/mechanical | High onset rate, regular ICI | Mechanical sounds |

#### Usage

```python
from analysis.rosetta_stone.acoustic_similarity_for_atomic_phrase_candidates import (
    find_atomic_phrases_by_persona,
    find_similar_phrases_multi_dimensional,
    ACOUSTIC_PERSONAS
)

# Find "GRITTY" phrases (aggressive alerts)
gritty_phrases = find_atomic_phrases_by_persona(
    db=vocalization_database,
    persona_name='gritty',
    species='marmoset',
    top_n=20,
    min_score=0.4
)

# Each result: (phrase_key, features_dict, score)
for phrase_key, features, score in gritty_phrases:
    print(f"{phrase_key}: HNR={features['harmonic_to_noise_ratio']:.2f}, "
          f"Attack={features['attack_time_ms']:.1f}ms")

# Find acoustically similar phrases (beyond F0)
similar_phrases = find_similar_phrases_multi_dimensional(
    db=vocalization_database,
    query_phrase_key='F0_7400_DUR_50_RANGE_300',
    species='marmoset',
    top_n=10
)
```

#### Hybrid Architecture: DBSCAN + Persona Mapping

The **Universal Rosetta Stone** now implements a **3-tier hybrid architecture** combining unsupervised clustering with semantic persona mapping:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    HYBRID PERSONA ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  TIER 1: Unsupervised DBSCAN Clustering (Data-Driven Discovery)         │
│  ─────────────────────────────────────────────────────────────────      │
│  • Groups phrases by acoustic similarity using all 17 micro-dynamics    │
│  • No prescriptive categories - lets data speak for itself              │
│  • Preserves novel patterns and edge cases                              │
│                                                                          │
│  TIER 2: Acoustic Persona Mapping (Semantic Interpretation)             │
│  ────────────────────────────────────────────────────────────────       │
│  • Post-hoc assignment of semantic labels to DBSCAN clusters            │
│  • Enables "semantic search": find all aggressive alert phrases         │
│  • Facilitates cross-species generalization and hypothesis testing      │
│                                                                          │
│  TIER 3: Contextual Validation (Ground Truth Verification)             │
│  ────────────────────────────────────────────────────────────────       │
│  • External validation with behavioral context, video, etc.             │
│  • Iterative refinement of persona definitions                          │
│  • Confirms or rejects semantic hypotheses                             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**New Methods in `UniversalRosettaStone`:**

```python
from analysis.rosetta_stone.universal_rosetta_stone import UniversalRosettaStone

rosetta = UniversalRosettaStone(sample_rate=48000)

# Build hybrid vocabulary (Tiers 1 + 2)
hybrid_clusters = rosetta.build_vocabulary_with_personas(
    phrases=phrase_list,
    eps=0.3,
    min_samples=2,
    enable_persona_mapping=True
)

# Returns:
# {
#     cluster_id: {
#         'phrases': [phrase1, phrase2, ...],
#         'dominant_persona': 'pure' | 'gritty' | 'bouncy' | ... | 'unclassified',
#         'persona_scores': {'gritty': 0.2, 'pure': 0.8, ...},
#         'cluster_size': int,
#         'mean_features': {...}
#     }
# }

# Semantic phrase search by persona
matches = rosetta.find_phrases_by_persona(
    clusters=hybrid_clusters,
    persona_name='pure',  # Contact/affiliation calls
    min_score=0.3
)
# Returns: [(cluster_id, phrases, score), ...]

# Get persona distribution summary
summary = rosetta.get_persona_summary(hybrid_clusters)
# Returns:
# {
#     'pure': {'cluster_count': 5, 'total_phrases': 42, 'avg_score': 0.75},
#     'gritty': {'cluster_count': 3, 'total_phrases': 18, 'avg_score': 0.62},
#     ...
# }

# Compute persona score for a specific cluster
score = rosetta.compute_cluster_persona_score(
    phrases=cluster_phrases,
    persona_name='bouncy'
)
```

**Benefits of Hybrid Approach:**

1. **Scientific Rigor**: DBSCAN discovers natural groupings without bias
2. **Interpretability**: Persona labels provide semantic meaning for clusters
3. **Flexibility**: Can disable persona mapping for pure unsupervised discovery
4. **Cross-Species**: Personas defined by acoustic features, not species-specific
5. **Iterative**: Can refine persona definitions based on behavioral validation

**When to Use Hybrid Architecture:**

- Use `build_vocabulary()` for pure data-driven discovery (no semantic bias)
- Use `build_vocabulary_with_personas()` for semantic interpretation + discovery
- Use `find_phrases_by_persona()` for hypothesis testing ("are alarm calls sharp?")
- Use `get_persona_summary()` for comparative analysis across species/recording contexts

---

#### Why Micro-Dynamics Matter

**Example**: Two phrases with identical F0 (7400 Hz) but different meanings:

| Phrase | F0 | Attack | HNR | Vibrato | Persona | Meaning |
|--------|-----|--------|-----|---------|---------|---------|
| A | 7400 Hz | 5 ms | 2.0 | 0 Hz | GRITTY | Alarm |
| B | 7400 Hz | 50 ms | 25.0 | 8 Hz | PURE | Contact |

**Traditional approach**: Groups A and B together (`F0_7400`) ❌
**Micro-dynamics**: Separates A (GRITTY) from B (PURE) ✅

#### Scientific Validation

Persona-based discovery enables:
1. **Fine-grained semantic categories** - Distinguish subtle behavioral contexts
2. **Cross-F0 similarity search** - Find "acoustic siblings" with different pitch
3. **Context-aware synthesis** - Generate context-appropriate vocalizations
4. **Quantified semantic meaning** - Score-based matching instead of binary inclusion

#### Command-Line Interface

```bash
# Find GRITTY phrases
cd analysis/rosetta_stone
python3 -c "
from acoustic_similarity_for_atomic_phrase_candidates import find_atomic_phrases_by_persona
from data_import import get_vocalization_database

db = get_vocalization_database()
results = find_atomic_phrases_by_persona(db, 'gritty', 'marmoset', top_n=10)
for key, feats, score in results:
    print(f'{key}: score={score:.2f}, HNR={feats[\"harmonic_to_noise_ratio\"]:.1f}')
"
```

---

### STEP 1.6: Acoustic Algebra - Continuous Semantic Generation [NEW]

**Breakthrough Innovation** - Transform the pipeline from **Discrete Retrieval** to **Continuous Generation**.

Instead of binary choice (Aggressive vs. Not Aggressive), generate **graded intensities** (30% Aggressive, 50% Aggressive, 70% Aggressive).

#### The Semantic Gradient Engine

**Without Algebra (Discrete Retrieval):**
```
Request: "I want an Aggressive call"
Action: Pick random phrase from "aggression" bucket
Result: You get FULL aggression (cannot get "mildly annoyed")
```

**With Algebra (Continuous Generation):**
```
Request: "I want 30% Aggression"
Action: Interpolate between Contact and Aggression vectors
Result: You get a nuanced "30% Aggressive" virtual phrase
```

#### Integration Architecture

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
│    Vector_Aggression = Mean(30D vectors for "Agg" phrases)        │
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

#### Usage: Discovery Phase

```python
from analysis.rosetta_stone.contextual_map import ContextualMap
from analysis.rosetta_stone.high_dimensional_acoustic_algebra import AcousticFeatureVector30

# Load annotated phrases
phrase_vectors = {
    'phrase_001': AcousticFeatureVector30(
        mean_f0_hz=6500, duration_ms=70, attack_ms=0.010,
        # ... 30 total features
    ),
    # ...
}

context_labels = {
    'phrase_001': 'contact',
    'phrase_002': 'aggression',
    'phrase_003': 'food',
    # ...
}

# Calculate semantic centroids
map_obj = ContextualMap()
centroids = map_obj.calculate_context_centroids(phrase_vectors, context_labels)

# View what each context "means"
map_obj.summarize()
# Output:
#   CONTACT (baseline): F0=6500Hz, Dur=70ms, Attack=10ms, HNR=20dB
#   AGGRESSION: F0=6050Hz, Dur=54ms, Attack=5ms, HNR=5.5dB
#   FOOD: F0=6325Hz, Dur=63ms, Attack=8ms, HNR=15.5dB

# Analyze context delta
delta = map_obj.calculate_context_delta('aggression', 'contact')
# Delta shows what makes aggression different:
#   - Attack: -5ms (faster onset)
#   - HNR: -14.5dB (more noise/harshness)
#   - Jitter: +0.06 (more instability)
```

#### Usage: Synthesis Phase

```python
# Generate "30% Aggressive" virtual phrase
virtual = map_obj.generate_graded_phrase(
    target_context='aggression',
    intensity=0.3  # 30% aggression
)
# Returns: AcousticFeatureVector30 with interpolated features

# Find nearest real phrase (for synthesis)
nearest_key, nearest_vec, distance = map_obj.find_nearest_real_phrase(
    virtual,
    phrase_vectors
)

# Use nearest phrase as source buffer
synth.set_source(nearest_phrase.audio_buffer)
synth.synthesize()
```

#### Scientific Application: The Threshold Test

**Hypothesis**: Animals perceive emotion as a **continuous continuum**, not discrete states.

**Experiment Design**:
```
Condition A (Baseline):  Intensity 0.0  → Contact
Condition B (Midpoint):    Intensity 0.5  → Mild Aggression
Condition C (Full):       Intensity 1.0  → Full Aggression
```

**Measurement**: Plot behavioral response (looking time, flight initiation) vs. Intensity %

- **If Linear**: Animal perceives a **GRADIENT** → Proof of Acoustic Continuum
- **If Step Function**: Animal perceives a **CATEGORY** → Proof of Discrete Semantics

**Why This Was Impossible Before**:
- Old system: Only 3 discrete levels (contact, aggression, food)
- New system: **Infinite precision** via acoustic algebra

#### Key Features

| Feature | Without Algebra | With Algebra |
|---------|-----------------|--------------|
| **Synthesis** | Retrieval (Play file) | Generation (Create vector) |
| **Nuance** | Low (3 discrete levels) | High (Infinite precision) |
| **Discovery** | Finds phrases | Finds contextual axes |
| **Experiments** | Binary choice tests | Threshold tests (continuum) |

#### Files

- `analysis/rosetta_stone/high_dimensional_acoustic_algebra.py` - 30D algebra engine
- `analysis/rosetta_stone/contextual_map.py` - Contextual map and gradient generation
- `analysis/rosetta_stone/demo_acoustic_algebra_integration.py` - Full workflow demo
- `analysis/rosetta_stone/ACOUSTIC_ALGEBRA_README.md` - Quick reference guide

#### Running the Demo

```bash
cd analysis/rosetta_stone
python3 demo_acoustic_algebra_integration.py
```

---

#### Command-Line Interface

```bash
# Find GRITTY phrases
python analysis/rosetta_stone/acoustic_similarity_for_atomic_phrase_candidates.py \
    --persona gritty --species marmoset --top-n 20

# Find phrases similar to specific phrase
python analysis/rosetta_stone/acoustic_similarity_for_atomic_phrase_candidates.py \
    --query F0_7400_DUR_50_RANGE_300 --species marmoset

# Analyze persona distribution
python analysis/rosetta_stone/acoustic_similarity_for_atomic_phrase_candidates.py \
    --analyze-distribution

# Check feature coverage in database
python analysis/rosetta_stone/acoustic_similarity_for_atomic_phrase_candidates.py \
    --analyze-coverage
```

---

### STEP 1.6: Persona Invariants Analysis [NEW]

**Statistical profiling within persona clusters** - Defines the "character" of each persona through micro-dynamics feature distributions.

While DBSCAN clustering (Tier 1) groups similar phrases together, **Persona Invariants** (Tier 2+) quantifies **what makes each cluster unique** by computing statistical profiles for all micro-dynamics features within each cluster.

**The Question**: What defines the "Character" of PERSONA_BAT_SOCIAL_LOW?
- Is it Attack Time? (Do they start with a "click"?)
- Is it Spectral Slope? (Is it "bright" or "dark"?)
- Is it Harmonicity? (Tonal vs noisy?)

**The Answer**: Statistical invariants.

```python
from analysis.rosetta_stone.persona_invariants_analysis import (
    PersonaInvariantsExtractor,
    PersonaProfile,
    PersonaInvariant
)

# Extract invariants for marmoset
extractor = PersonaInvariantsExtractor(sample_rate=48000)
clusters, profiles = extractor.extract_personas_from_db(
    db_path='vocalization_database.json',
    species='marmoset',
    max_phrases=600,
    eps=0.25,
    min_samples=3
)

# Get persona profile
profile = profiles['MARMOSET_PHEE']
print(f"Cluster: {profile.cluster_id}, Size: {profile.cluster_size}")

# Access invariants
f0_inv = profile.get_invariant('mean_f0_hz')
print(f"F0: {f0_inv.mean:.1f} ± {f0_inv.std:.1f} Hz (CV: {f0_inv.cv:.3f})")

# Output:
# Cluster: 0, Size: 576
# F0: 6525.8 ± 935.4 Hz (CV: 0.143)
```

**Persona Invariant Structure:**

```python
@dataclass
class PersonaInvariant:
    feature_name: str
    mean: float          # Central tendency
    std: float           # Variability
    median: float        # Robust central value
    min: float           # Range bounds
    max: float
    q25: float           # Quartiles
    q75: float
    count: int           # Sample size
    cv: float            # Coefficient of variation (std/mean)
```

**Statistical Profiles by Persona:**

| Persona | F0 (Hz) | F0 Range (Hz) | Duration (ms) | Key Characteristics |
|---------|---------|---------------|---------------|---------------------|
| **MARMOSET_PHEE** | 6526 ± 935 | 427 ± 399 | 76.5 ± 57.6 | Stable pitch (CV=0.14), narrow range |
| **MARMOSET_ALARM** | 6020 ± 701 | 3722 ± 163 | 58.1 ± 0.0 | Wide modulation (CV=0.04), urgent |
| **BAT_MID_FM** | 7437 ± 1232 | 9755 ± 2583 | 17.4 ± 0.0 | Wide FM sweeps, navigation |
| **BAT_SOCIAL_US** | 7408 ± 1383 | 24 ± 22 | 17.4 ± 0.0 | Narrow range (CV=0.93), stable pitch |
| **BAT_LOW_SOCIAL** | 2884 ± 1161 | 11535 ± 4154 | 11.6 ± 0.0 | Low-frequency, high-energy |

**Key Discriminators (Cohen's d > 0.8 = Large Effect):**

| Feature | MARMOSET_PHEE | MARMOSET_ALARM | Effect Size | Interpretation |
|---------|---------------|----------------|-------------|----------------|
| **F0 Range** | 427 Hz | 3722 Hz | **8.71** | VERY LARGE - Alarm signature |
| **Duration** | 76.5 ms | 58.1 ms | **Large** | Shorter = Urgency |
| **F0 Mean** | 6526 Hz | 6020 Hz | Moderate | Slight pitch shift |

**Scientific Validation - Alarm vs Juvenile:**

Question: Is marmoset Cluster 1 (2% outliers) "Alarm" or "Juvenile" variety?

**Evidence**:
- F0 Range Ratio: **8.71x** wider (3,722 Hz vs 427 Hz)
- Duration: **24% shorter** (58 ms vs 76 ms)
- Alarm Score: **4/4** → **HIGH CONFIDENCE**

**Conclusion**: ✅ **ALARM VARIETY** (Not Juvenile)

**Why This Matters:**

1. **Targeted Synthesis**: Generate phrases matching specific feature constraints
2. **Hybrid Personas**: Blend characteristics from different personas
3. **Validation**: Identify discriminators for behavioral testing
4. **Cross-Species**: Compare persona strategies across species

**Files:**
- `analysis/rosetta_stone/persona_invariants_analysis.py` - Extraction engine
- `analysis/rosetta_stone/persona_mapping.py` - PersonaRouter system
- `analysis_output/persona_invariants.json` - Statistical profiles

---

### STEP 1.7: Hybrid Persona Generation [NEW]

**Blend characteristics from different personas** to create novel but naturalistic vocalizations.

The "Acoustic Algebra" capability: Generate a sound that is **50% PERSONA_BAT_LOW (frequency profile)** but **50% PERSONA_BAT_HIGH (texture profile)**.

**Synthesis Strategies:**

| Strategy | Method | Use Case |
|----------|--------|----------|
| **Buffer Crossfade** | Weighted mix of audio buffers | Smooth hybrid transition |
| **Granular Alternate** | Alternate grains from different personas | Textural blend |
| **Feature Interpolate** | Interpolate feature targets | Precise control |
| **Spectral Shape** | Apply EQ to match profile | Timbre blending |

**Example: Marmoset Hybrid Contact-Alarm**

```python
from realtime.hybrid_persona_synthesizer import (
    HybridPersonaSynthesizer,
    HybridSynthesisRequest,
    HybridStrategy
)

synthesizer = HybridPersonaSynthesizer(
    invariants_path='analysis_output/persona_invariants.json',
    sample_rate=48000
)

# 70% Phee + 30% Alarm
request = HybridSynthesisRequest(
    hybrid_id='MARMOSET_HYBRID_CONTACT_ALARM',
    source_personas=[
        ('MARMOSET_PHEE', 0.7),    # 70% stable contact
        ('MARMOSET_ALARM', 0.3)    # 30% alarm characteristics
    ],
    strategy=HybridStrategy.GRANULAR_ALTERNATE,
    duration_ms=500.0,
    grain_size_ms=50.0
)

audio, metadata = synthesizer.synthesize_hybrid(request)

# Interpolated Features:
#   F0: 6323.67 Hz (between 6526 and 6020)
#   Range: 1745.12 Hz (between 427 and 3722)
#   Duration: 69.13 ms (between 76 and 58)
```

**Feature Interpolation Example:**

```python
# Target specific feature values
request = HybridSynthesisRequest(
    hybrid_id='MARMOSET_FEATURE_BLEND',
    source_personas=[
        ('MARMOSET_PHEE', 0.6),
        ('MARMOSET_ALARM', 0.4)
    ],
    strategy=HybridStrategy.FEATURE_INTERPOLATE,
    target_features={
        'mean_f0_hz': 6300.0,  # Target F0
        'f0_range_hz': 1500.0,  # Target modulation
        'mean_duration_ms': 70.0  # Target duration
    }
)
```

**Applications:**

1. **Scientific Hypothesis Testing**
   - "What percept emerges from 50% alarm + 50% contact?"
   - Test receiver responses to hybrid signals
   - Map perceptual boundaries between categories

2. **Bio-Inspired Sonification**
   - Naturalistic communication interfaces
   - Non-threatening alert systems
   - Aesthetic applications (sound art, installations)

3. **Ethical Field Deployment**
   - Naturalistic but novel (prevents habituation)
   - Species-appropriate but not identical
   - Avoids playback contamination of wild populations

**Generated Outputs:**
- `hybrid_marmoset_contact_alarm.wav` - 70/30 Phee-Alarm hybrid
- `hybrid_bat_nav_social.wav` - 50/50 Navigation-Social hybrid
- `hybrid_marmoset_feature_blend.wav` - Feature-interpolated blend

**Files:**
- `realtime/hybrid_persona_synthesizer.py` - Hybrid generation engine
- `analysis_output/persona_invariants.json` - Persona profiles (input)
- `analysis_output/hybrid_*.wav` - Generated audio

---

### STEP 1.8: Acoustic Algebra - Continuous Acoustic Field [NEW]

**Mathematical operations on vocalizations as vectors** - Navigate the continuous acoustic space enabled by Universal Rosetta Stone.

**The Core Insight**: URS decodes vocalizations into **mathematical vectors** (F0, Duration, Range, Timbre) rather than opaque "audio blobs," enabling movement beyond retrieval into **navigation** through the acoustic field.

**Acoustic Vector Representation:**

```python
from analysis.rosetta_stone.acoustic_algebra import (
    AcousticVector,
    ContextVector,
    AcousticAlgebraEngine
)

# A phrase as a 7-dimensional vector
neutral_phee = AcousticVector(
    f0_hz=6000.0,           # Fundamental frequency
    duration_ms=50.0,        # Temporal extent
    f0_range_hz=300.0,      # Pitch modulation
    harmonicity=0.95,        # Tonal quality (0=noise, 1=pure)
    spectral_flatness=0.1,  # Timbre (0=tonal, 1=noise)
    jitter=0.0,             # Frequency instability
    shimmer=0.0             # Amplitude instability
)
```

**The Acoustic Algebra Operations:**

| Operation | Formula | Method | Application |
|-----------|---------|--------|-------------|
| **Identity** | `Phrase_A` | `engine.identity(v)` | Retrieval |
| **Addition** | `Phrase + Context` | `engine.add(v, ctx)` | Extrapolation |
| **Subtraction** | `Phrase_A - Phrase_B` | `engine.subtract(a, b)` | Feature delta |
| **Scalar Mult** | `Phrase * 1.5` | `engine.multiply(v, 1.5)` | Intensity scaling |
| **Average** | `(A + B) / 2` | `engine.average(a, b)` | 50% Interpolation |
| **Composition** | `Sentence + Context` | `engine.compose(sent, ctx)` | Grammar warp |

**1. Phrase-Level Interpolation**

Mathematical blend between two phrases:

```python
# Interpolate at 50% (midpoint)
hybrid = engine.interpolate(
    phrase_a=neutral_phee,
    phrase_b=excited_phee,
    alpha=0.5
)

# Result: F0=6500 Hz (between 6000 and 7000)
#         Duration=40 ms (between 50 and 30)
```

**2. Contextual Extrapolation**

Create novel phrases by applying context vectors:

```python
# Define aggression context
aggression = ContextVector(
    name='aggression',
    f0_multiplier=1.2,      # +20% pitch
    duration_ratio=0.8,     # -20% duration (shorter = urgent)
    f0_range_multiplier=1.5 # +50% modulation (more excited)
)

# Extrapolate: Neutral + Aggression
aggressive_phrase = engine.extrapolate(neutral_phee, aggression)

# Result: F0=7200 Hz (6000 * 1.2)
#         Duration=40 ms (50 * 0.8)
```

**Predefined Context Vectors:**

```python
from analysis.rosetta_stone.acoustic_algebra import MARMOSET_CONTEXTS

# Available contexts
MARMOSET_CONTEXTS['neutral']      # Baseline
MARMOSET_CONTEXTS['contact']      # +5% F0, -5% duration
MARMOSET_CONTEXTS['aggression']   # +20% F0, -20% duration, +50% jitter
MARMOSET_CONTEXTS['alarm']        # +15% F0, -30% duration, +100% jitter
MARMOSET_CONTEXTS['submission']   # -10% F0, +20% duration
```

**3. Sentence-Level Extrapolation**

Warp entire sentences (phrases + pauses) with context:

```python
from analysis.rosetta_stone.acoustic_algebra import (
    SENTENCE_TEMPLATE,
    GRAMMAR_CONTEXT
)

# Neutral sentence: [Phrase A] -> [Pause 50ms] -> [Phrase B]
neutral_sentence = SENTENCE_TEMPLATE(
    phrases=[phrase_a, phrase_b],
    pauses_ms=[50.0],
    modality="harmonic"
)

# Urgency context
urgency = GRAMMAR_CONTEXT(
    name='urgency',
    phrase_f0_multiplier=1.15,
    phrase_duration_ratio=0.7,    # Compress phrases
    phrase_pause_ratio=0.2        # Compress pauses heavily
)

# Extrapolate sentence
urgent_sentence = engine.compose(neutral_sentence, urgency)

# Result: [Phrase 28ms] -> [Pause 10ms] -> [Phrase 31ms]
# (Preserves structure, warps timing)
```

**4. Semantic Continuum Generation**

Generate gradual transitions for categorical perception testing:

```python
# Generate 10-step continuum: Neutral → Aggressive
continuum = engine.generate_continuum(
    start=neutral_phee,
    end_context=aggression,
    num_steps=10
)

# Output:
# Step 0: F0=6000 Hz, Dur=50 ms   (Neutral)
# Step 1: F0=6133 Hz, Dur=47 ms
# Step 2: F0=6267 Hz, Dur=43 ms
# ...
# Step 9: F0=7200 Hz, Dur=40 ms   (Fully aggressive)

# Monotonic progression verified ✅
```

**Scientific Applications:**

**1. Categorical Perception Test**

Question: Do animals perceive categories as **gradient** or **binary**?

```python
# Generate 100-step continuum between contact and alarm
continuum = engine.generate_interpolation_continuum(
    contact_call,
    alarm_call,
    num_steps=100
)

# Play each step to animal
# Measure: At which step does behavior switch?
# Result: Validates acoustic space is biologically relevant
```

**2. "Impossible Context" Synthesis**

Generate dangerous-to-record vocalizations mathematically:

```python
# Aggressive alarm from neutral (dangerous to record in wild)
synthetic_alarm = engine.extrapolate(
    base=neutral_phee,
    context=aggression_context
)

# Test: Do other animals respond to synthetic signal?
# If yes: Validates vector space captures biological meaning
```

**3. Perceptual Boundary Mapping**

Find exact switching point between categories:

```python
continuum = engine.generate_interpolation_continuum(
    contact_call,
    alarm_call,
    num_steps=100
)

# Index 50: F0=6495 Hz (approximate boundary)
# Index 51: F0=6505 Hz (possible switch point)
# Index 52: F0=6515 Hz (definite alarm perception)
```

**Complete Workflow:**

```python
# 1. Start with neutral phrase
neutral = AcousticVector(f0_hz=6000.0, duration_ms=50.0, ...)

# 2. Define context trajectory
contexts = [neutral_ctx, aggression_ctx, alarm_ctx]

# 3. Extrapolate through trajectory
for ctx in contexts:
    phrase = engine.extrapolate(neutral, ctx)
    # Play to animal, measure response
    # Map perceptual boundary
```

**Benefits of Acoustic Algebra:**

1. **Continuous Space**: Treats vocalizations as mathematical points, not discrete blobs
2. **Navigation**: Move through acoustic space by interpolation/extrapolation
3. **Prediction**: Generate novel vocalizations beyond dataset
4. **Validation**: Test if vector space is biologically relevant

**Files:**
- `analysis/rosetta_stone/acoustic_algebra.py` - Core implementation
- `tests/test_acoustic_algebra.py` - **33/33 tests passing** ✅

**Test Results:**
```
========================= 33 passed in 0.57s ==========================

✅ Vector operations (7 tests)
✅ Context application (3 tests)
✅ Interpolation (6 tests)
✅ Extrapolation (4 tests)
✅ Sentence warping (4 tests)
✅ Algebra engine (6 tests)
✅ Continuum generation (3 tests)
```

---

### STEP 1.9: Metadata-First Synthesis Architecture [NEW]

**The Fundamental Insight: Direct Metadata is significantly better for Interpolation and Extrapolation.**

Using "Persona" (Cluster Labels) for synthesis creates a **Discrete Abstraction Layer** that limits the power of continuous math. The metadata-first approach enables true interpolation and extrapolation in the acoustic feature space.

#### **Problem with Persona-Based Routing:**

| Approach | Method | Limitation |
|----------|--------|------------|
| **Persona-Based** | Select persona → Load buffer → Synthesize | ❌ Can only interpolate WITHIN one persona |
| **Metadata-Based** | Query vector space → Select multiple buffers → Morph | ✅ Can interpolate BETWEEN personas |

**Why Metadata Wins:**

> "Personas are too 'lumpy' for interpolation. You cannot interpolate 'Box A' and 'Box B'. You can only interpolate the coordinates inside them."
>
> — **Key Architectural Insight**

#### **Complete Workflow: Metadata-Driven Synthesis + Persona Validation**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SYNTHESIS WORKFLOW                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. SYNTHESIS (Metadata-First)                                             │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │ Query Vector Space by F0, Duration, F0 Range (continuous)      │    │
│     │ → Select multiple source phrases from ANY persona              │    │
│     │ → Interpolate/extrapolate between them                          │    │
│     │ → Generate audio                                                │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│                                    ↓                                       │
│  2. VALIDATION (Persona Semantic Zones)                                  │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │ Extract features from synthesized audio                         │    │
│     │ → Calculate Mahalanobis distance to each persona cluster      │    │
│     │ → Check if within 2-sigma boundary (95% confidence)            │    │
│     │ → Assign persona + confidence score                            │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Separation:**
- **Synthesis**: Metadata-first (explore continuous space freely)
- **Validation**: Persona-based (check semantic boundaries)

#### **Architectural Shift:**

```python
# OLD: Persona Routing (Discrete)
persona = router.select_persona(species="marmoset", context="alarm")
buffer = load_buffer(persona)
audio = synthesize(buffer)  # Single source, ONE persona

# NEW: Metadata Query (Continuous)
query = MetadataQuery(
    target_f0_hz=7500,
    target_duration_ms=50,
    target_f0_range_hz=2000,
    preferred_contexts=["alarm"],  # SOFT constraint (boosts score)
    avoided_contexts=["contact"]    # SOFT constraint (penalty)
)
recipe = engine.query_interpolation_targets(query, num_sources=2)
# May return: 0.6 * alarm_buffer + 0.4 * threat_buffer
audio = morph_sources(recipe.sources)  # Multi-source morphing

# THEN: Validate output
validator = PersonaSemanticZoneValidator()
result = validator.validate_vocalization(
    features=extract_features(audio),
    species='marmoset'
)
# Returns: persona_id, confidence, is_outlier, warnings
```

#### **Soft Constraints vs Hard Filters:**

| Constraint Type | Behavior | Discovery Potential |
|----------------|----------|---------------------|
| **Hard Filter** | "Context MUST be alarm" | ❌ Finds only existing alarms |
| **Soft Constraint** | "Context=alarm gets +1.0 score" | ✅ May find acoustic match from different context |

#### **Ghost Word Discovery:**

The key scientific advantage of metadata-first synthesis is the ability to create **"Ghost Words"** - sounds that exist theoretically in the vector space but not statistically in the dataset.

```python
from realtime.metadata_synthesizer import MetadataFirstSynthesizer

synthesizer = MetadataFirstSynthesizer(
    phrase_segments=phrase_library,
    sample_rate=48000
)

# Create sound in the void between clusters
audio, recipe = synthesizer.synthesize_ghost_word(
    cluster_a_id=1,  # Mid-FM (7.4 kHz, wide FM)
    cluster_b_id=2,  # Social (7.4 kHz, narrow range)
    blend_ratio=0.5  # Exactly between
)
# Result: 7.4 kHz, semi-modulated - a "Ghost Word"!
# Discovery Potential: 1.0 (maximum - between distant clusters)
```

**Ghost Words enable:**

1. **Hypothesis Testing**: "What if there was a call between these two clusters?"
2. **Perceptual Boundaries**: Map categorical perception by testing ghost words
3. **Feature Discovery**: Find which acoustic features transcend semantic contexts
4. **Cross-Pollination**: Discover that "urgency" is a timbre, not just a context

#### **Discovery Potential Metric:**

```python
recipe.discovery_potential  # 0.0 to 1.0
# 0.0 = Within existing cluster (low novelty)
# 0.2 = Between nearby clusters (moderate novelty)
# 1.0 = Between distant clusters (maximum novelty)
```

#### **Scientific Applications:**

| Application | Method | Value |
|------------|--------|-------|
| **Hypothesis Testing** | Ghost word at midpoint | Test acoustic boundaries |
| **Perceptual Mapping** | Interpolate across continuum | Find categorical thresholds |
| **Context Extrapolation** | Vector space + constraints | Generate "urgency" variants |
| **Cross-Species Translation** | Match features across species | Preserve "meaning" across F0 |

**Files:**
- `realtime/metadata_synthesizer.py` - Metadata-first synthesis engine with 30D features
- `realtime/persona_semantic_zone_validator.py` - Persona-based validation
- `tests/test_metadata_first_synthesis.py` - 23 tests passing ✅
- `tests/test_persona_semantic_zone_validation.py` - 23 tests passing ✅
- `tests/test_30d_metadata_synthesis.py` - 11/11 tests passing ✅ (30D features)

**Demo Results:**
```
--- Demo 1: Direct Target Query ---
Query: F0=7000Hz, Duration=50ms
Recipe: bat_social_001 (56%) + bat_midfm_001 (44%)
⚠️  CROSS-PERSONA SYNTHESIS!
Discovery Potential: 0.20

--- Demo 2: Ghost Word Discovery ---
Between Bat Cluster 1 (Mid-FM) and Cluster 2 (Social)
Target F0: 6714Hz
Discovery Potential: 1.00
⚠️  This sound exists in the VOID between clusters!
Validation: Within 1.5σ of BOTH clusters (Ghost Word confirmed)
```

**Usage Recommendations:**

| Task | Strategy | Why? |
|------|----------|------|
| **Synthesis** | **Metadata-First** | Requires continuous math (F0/Dur) |
| **Synthesis (Extrapolation)** | **Metadata + Constraints** | Vector space + guardrails |
| **Validation** | **Persona/Cluster** | Check if output is in "semantic zone" |
| **Routing** | **Metadata** | "Find me something that sounds like X" |

**Key Insight:**
> Metadata-First enables interpolation BETWEEN personas, not just within them. This discovers 'Ghost Words' that exist theoretically but not statistically in the dataset. Personas then VALIDATE whether the output falls within acceptable "semantic zones."

**Note:** This framework implements two separate vector space systems:
- **Python 30D** (`metadata_synthesizer.py`): Logic-layer metadata synthesis with full micro-dynamics features
- **Rust 30D** (`island_hopping.rs`): Execution-layer navigation with SIMD-optimized vector math

---

### STEP 1.9.1: 30D Micro-Dynamics Features [NEW]

**Full 30-Dimensional Feature Space for Python Logic Layer and Rust Execution Layer**

The Python metadata synthesizer and Rust island_hopping module both support **complete 30D micro-dynamics features**, enabling precise vector space queries and interpolation for synthesis target generation. The Python implementation is used for logic-layer synthesis planning, while the Rust implementation is used for execution-layer navigation with SIMD-optimized performance.

#### **30D Feature Vector Composition:**

```python
# The 30 features are organized into 6 groups:
# 1. Fundamental (3): mean_f0_hz, f0_range_hz, duration_ms
# 2. Grit Factors (3): harmonic_to_noise_ratio, spectral_flatness, harmonicity
# 3. Motion Factors (7): attack_time_ms, decay_time_ms, sustain_level,
#                        vibrato_rate_hz, vibrato_depth, jitter, shimmer
# 4. Fingerprint Factors (13 MFCCs): mfcc_1 through mfcc_13
# 5. Spectral Dynamics (1): spectral_flux
# 6. Rhythm Factors (3): median_ici_ms, onset_rate_hz, ici_coefficient_of_variation
```

#### **30D Feature Extraction:**

```python
from realtime.metadata_synthesizer import PhraseCandidate

# Extract all 30 features from a phrase
candidate = PhraseCandidate(phrase_id="marmoset_phee_001")

# Get the full 30D feature vector
feature_vector = candidate.get_feature_vector()
# Returns: np.ndarray of shape (30,)

# Access individual features
print(f"F0: {candidate.mean_f0_hz} Hz")
print(f"Duration: {candidate.duration_ms} ms")
print(f"HNR: {candidate.harmonic_to_noise_ratio}")
print(f"MFCC-1: {candidate.mfcc_1}")
print(f"Spectral Flux: {candidate.spectral_flux}")
print(f"Onset Rate: {candidate.onset_rate_hz} Hz")
```

#### **30D Vector Space Queries:**

```python
from realtime.metadata_synthesizer import VectorSpaceQueryEngine

# Create query engine with 30D distance calculation
query_engine = VectorSpaceQueryEngine()

# Define 30D target parameters
target_params = {
    # Fundamental (3)
    'mean_f0_hz': 7000.0,
    'f0_range_hz': 400.0,
    'duration_ms': 50.0,
    # Grit Factors (3)
    'harmonic_to_noise_ratio': 25.0,
    'spectral_flatness': 0.15,
    'harmonicity': 0.9,
    # Motion Factors (7)
    'attack_time_ms': 10.0,
    'decay_time_ms': 20.0,
    'sustain_level': 0.7,
    'vibrato_rate_hz': 8.0,
    'vibrato_depth': 50.0,
    'jitter': 0.02,
    'shimmer': 0.03,
    # Fingerprint Factors (13 MFCCs)
    'mfcc_1': -12.5, 'mfcc_2': 3.2, 'mfcc_3': 0.8,
    'mfcc_4': -0.5, 'mfcc_5': 1.2, 'mfcc_6': 0.3,
    'mfcc_7': -0.8, 'mfcc_8': 0.5, 'mfcc_9': 0.1,
    'mfcc_10': -0.3, 'mfcc_11': 0.2, 'mfcc_12': -0.1,
    'mfcc_13': 0.05,
    # Spectral Dynamics (1)
    'spectral_flux': 0.5,
    # Rhythm Factors (3)
    'median_ici_ms': 15.0,
    'onset_rate_hz': 8.0,
    'ici_coefficient_of_variation': 0.3,
}

# Query nearest neighbors using 30D Euclidean distance
candidates = query_engine.query_nearest_metadata(
    target_params=target_params,
    k=5,
    species='marmoset'
)

# Results are ranked by 30D acoustic similarity
for cand in candidates:
    print(f"{cand.phrase_id}: {cand.acoustic_score:.3f}")
```

#### **30D Interpolation:**

```python
from realtime.metadata_synthesizer import interpolate_30d_features

# Interpolate between two 30D feature vectors
candidate_a = PhraseCandidate(phrase_id="bat_social_001")
candidate_b = PhraseCandidate(phrase_id="bat_midfm_001")

# 50/50 blend (0.5 = 50% of B, 50% of A)
interpolated = interpolate_30d_features(
    candidate_a, candidate_b, blend_ratio=0.5
)
# Returns: np.ndarray of shape (30,)

# Create synthesis recipe with interpolated 30D target
recipe = query_engine.create_synthesis_recipe(
    target_params=interpolated,
    num_phrases=3
)
```

#### **30D Synthesis Recipes:**

```python
# Synthesis recipes now include 30D target parameters
recipe = query_engine.create_synthesis_recipe(
    target_params=target_params,  # 30D target
    num_phrases=3
)

print(f"Target 30D Vector: {recipe.target_30d_vector}")
print(f"Interpolation Weights: {recipe.interpolation_weights}")
print(f"Discovery Potential: {recipe.discovery_potential:.2f}")

# Use the recipe for synthesis
synthesizer = MetadataFirstSynthesizer()
result = synthesizer.synthesize_from_recipe(recipe)
```

#### **Scientific Applications:**

| Application | 30D Advantage | Example |
|------------|---------------|---------|
| **Precise Interpolation** | 30 dimensions capture subtle timbral variations | Interpolate between 30 vocal qualities |
| **Ghost Word Discovery** | Find sounds in 30D void between clusters | Discover novel vocalizations |
| **Cross-Species Mapping** | Match 30 features across species | Preserve "meaning" across F0 range |
| **Timbre Transfer** | Control all 30 acoustic dimensions | Apply "urgency" timbre to new context |

#### **Comparison: Python 30D vs. Rust 30D:**

| Aspect | Python 30D | Rust 30D |
|--------|------------|----------|
| **Purpose** | Logic-layer synthesis targets | Execution-layer navigation |
| **Location** | `realtime/metadata_synthesizer.py` | `technical_architecture/src/island_hopping.rs` |
| **Features** | 30 dimensions (full micro-dynamics) | 30 dimensions (full micro-dynamics) |
| **Performance** | Flexible, numpy-based | SIMD-optimized, 10-100x faster |
| **Use Case** | Query planning, recipe generation | Real-time trajectory navigation |
| **Test Coverage** | 11/11 tests passing ✅ | 33/33 tests passing ✅ |

#### **Files:**
- `realtime/metadata_synthesizer.py` - Python 30D metadata synthesis engine
- `tests/test_30d_metadata_synthesis.py` - 11/11 tests passing ✅
- `technical_architecture/src/island_hopping.rs` - Rust 30D navigation (SIMD-optimized)

#### **Test Coverage:**
- ✅ 30D feature extraction from phrase metadata
- ✅ 30D vector space queries with normalized distance
- ✅ 30D interpolation between candidates
- ✅ Synthesis recipe creation with 30D targets
- ✅ Backward compatibility with 4D metadata
- ✅ MFCC feature extraction (13 dimensions)
- ✅ Rhythm feature extraction (3 dimensions)
- ✅ Spectral dynamics extraction (1 dimension)
- ✅ Complete 30D feature vector construction

---

### STEP 1.10: Persona Semantic Zone Validation [NEW]

**Post-Synthesis Validation Using Persona Clusters**

After synthesis with metadata-first approach, we validate whether the output is scientifically valid by checking if it falls within acceptable "semantic zones" defined by persona clusters.

#### **The Validation Question:**

> "Is this synthesized sound something that could actually exist in nature given this species' vocal repertoire?"

This is answered by checking if the synthesized features fall within the statistical boundaries (typically 2-sigma, 95% confidence) of known persona clusters.

#### **How It Works:**

```python
from realtime.persona_semantic_zone_validator import PersonaSemanticZoneValidator

validator = PersonaSemanticZoneValidator()

# 1. Extract features from synthesized audio
features = {
    'mean_f0_hz': 6526,
    'duration_ms': 76.5,
    'f0_range_hz': 427,
    'harmonicity': 0.95,
    'spectral_flatness': 0.1,
    'jitter': 0.02,
    'shimmer': 0.03
}

# 2. Validate against semantic zones
result = validator.validate_vocalization(
    features=features,
    species='marmoset'
)

# 3. Check result
print(f"Persona: {result.persona_id}")           # 'MARMOSET_PHEE'
print(f"Semantic Label: {result.semantic_label}") # 'contact'
print(f"Confidence: {result.confidence:.2%}")    # 97.3%
print(f"In Zone: {result.passed}")               # True
print(f"Outlier: {result.is_outlier}")           # False
print(f"Distance: {result.mahalanobis_distance:.2f}σ")  # 0.85σ
```

#### **Mahalanobis Distance: The Statistical Metric**

Uses **Mahalanobis distance** instead of Euclidean distance to account for covariance between features:

```
D² = (x - μ)ᵀ Σ⁻¹ (x - μ)
```

Where:
- `x` = feature vector of synthesized audio
- `μ` = cluster centroid (mean feature vector)
- `Σ` = covariance matrix (accounts for feature relationships)

**Why Mahalanobis?**

| Metric | Accounts for Variance? | Accounts for Covariance? | Best For |
|--------|----------------------|------------------------|----------|
| **Euclidean** | ❌ | ❌ | Simple geometric distance |
| **Z-score** | ✅ | ❌ | Normalized feature distance |
| **Mahalanobis** | ✅ | ✅ | ✅ Multivariate cluster membership |

#### **Semantic Zone Boundaries:**

**2-Sigma Rule (95% Confidence):**

| Persona | F0 (Hz) | F0 Range (Hz) | Duration (ms) | Boundary |
|---------|---------|---------------|---------------|----------|
| **MARMOSET_PHEE** | 6526 ± 935 | 427 ± 399 | 76.5 ± 57.6 | Stable pitch, narrow range |
| **MARMOSET_ALARM** | 6020 ± 701 | 3722 ± 163 | 58.1 ± 0.0 | Wide modulation, urgent |
| **BAT_MID_FM** | 7437 ± 1232 | 9755 ± 2583 | 17.4 ± 0.0 | Wide FM sweeps |
| **BAT_SOCIAL_US** | 7408 ± 1383 | 24 ± 22 | 17.4 ± 0.0 | Narrow range, stable pitch |

#### **Ghost Word Detection:**

A "ghost word" is a vocalization that falls **between** two or more persona clusters (within 1.5σ of multiple clusters):

```python
is_ghost, close_clusters = validator.is_ghost_word(
    features=ghost_features,
    species='marmoset'
)

if is_ghost:
    print(f"Ghost Word detected!")
    for persona_id, distance in close_clusters:
        print(f"  Close to {persona_id}: {distance:.2f}σ")
```

**Example Ghost Word:**
```
Features: F0=6273 Hz, Duration=67.3 ms, F0 Range=2074 Hz
Distance to MARMOSET_PHEE: 1.3σ
Distance to MARMOSET_ALARM: 1.4σ

⚠️ GHOST WORD: Between Phee and Alarm!
Interpolated at: 50% blend ratio
```

#### **Validation Workflow:**

```
┌─────────────────────────────────────────────────────────────────┐
│  1. SYNTHESIS (Metadata-First)                                  │
│     → Query vector space freely                                 │
│     → Generate audio from ANY coordinate                        │
│     → May create "Ghost Words" between clusters                 │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  2. FEATURE EXTRACTION                                         │
│     → Extract F0, duration, harmonicity, etc.                   │
│     → Compute feature vector                                    │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  3. SEMANTIC ZONE VALIDATION                                    │
│     → Calculate Mahalanobis distance to each persona           │
│     → Find closest match                                        │
│     → Check if within 2-sigma boundary                         │
│     → Generate validation report                               │
└─────────────────────────────────────────────────────────────────┘
```

#### **Strict vs Lenient Validation:**

| Mode | Threshold | Use Case |
|------|-----------|----------|
| **Strict** | 2.0σ (95% confidence) | Scientific validation, publication |
| **Lenient** | 3.0σ (99.7% confidence) | Exploratory research, hypothesis testing |

```python
# Strict mode (default)
result = validator.validate_vocalization(
    features,
    species='marmoset',
    strict=True  # 2-sigma boundary
)

# Lenient mode
result = validator.validate_vocalization(
    features,
    species='marmoset',
    strict=False  # 3-sigma boundary
)
```

#### **Feature-Level Diagnostics:**

```python
result = validator.validate_vocalization(features, species='marmoset')

# Check which features deviate
for feature, z_score in result.feature_z_scores.items():
    if abs(z_score) > 2.0:
        print(f"{feature}: {z_score:.1f}σ deviation")
```

**Example output:**
```
mean_f0_hz: -0.2σ  ✅ Normal
duration_ms: 0.5σ   ✅ Normal
f0_range_hz: 3.2σ  ⚠️  OUTLIER (too wide)
harmonicity: -1.1σ  ✅ Normal
```

#### **Validation Result Structure:**

```python
@dataclass
class SemanticZoneValidationResult:
    passed: bool                    # Within 2-sigma boundary
    persona_id: str                 # Best matching persona
    semantic_label: str             # 'contact', 'alarm', etc.
    confidence: float               # 0-1
    mahalanobis_distance: float     # Distance from centroid
    is_outlier: bool                # Beyond 2-sigma
    feature_deviations: Dict        # Raw deviations
    feature_z_scores: Dict          # Standardized deviations
    warnings: List[str]             # Outlier features
    suggested_persona: Optional[str] # Alternative if low confidence
```

#### **Scientific Value:**

1. **Quality Control**: Ensures synthesized sounds are biologically plausible
2. **Ghost Word Identification**: Finds sounds that exist between categories
3. **Cross-Species Validation**: Checks if synthesis respects species-specific constraints
4. **Statistical Rigor**: Uses proper multivariate statistics for cluster membership

#### **Files:**
- `realtime/persona_semantic_zone_validator.py` - Validation engine
- `tests/test_persona_semantic_zone_validation.py` - 23 tests passing ✅

**Test Coverage:**
- ✅ Valid vocalizations pass validation
- ✅ Invalid vocalizations are rejected
- ✅ Ghost words are detected
- ✅ Classification returns correct persona
- ✅ Statistical boundaries enforced
- ✅ Feature-level diagnostics work
- ✅ Cross-species validation works

---

### STEP 1.11: Bio-Acoustic Turing Test Validation [NEW]

**Comprehensive validation framework for synthesized vocalizations**

Combines acoustic feature validation, audio quality checks, perceptual validation, and semantic zone validation to ensure synthesized sounds are scientifically valid.

#### **Complete Validation Pipeline:**

```python
from realtime.bio_acoustic_validator import BioAcousticValidator
from realtime.persona_semantic_zone_validator import PersonaSemanticZoneValidator

# 1. Acoustic validation (tolerances)
acoustic_validator = BioAcousticValidator(
    f0_tolerance_hz=200,
    duration_tolerance_ms=10,
    harmonicity_tolerance=0.1
)

result = acoustic_validator.validate_synthesis(
    audio=synthesized_audio,
    target_metadata={
        'mean_f0_hz': 6526,
        'duration_ms': 76.5,
        'f0_range_hz': 427,
        'harmonicity': 0.95
    }
)

# 2. Audio quality validation
assert result.passed_clipping_check    # No clipping
assert result.passed_dc_offset_check    # No DC offset
assert result.passed_rms_check          # Sufficient amplitude
assert result.passed_frequency_check   # Safe frequency range

# 3. Semantic zone validation
zone_validator = PersonaSemanticZoneValidator()
zone_result = zone_validator.validate_vocalization(
    features=extract_features(synthesized_audio),
    species='marmoset'
)

assert zone_result.passed                # Within semantic zone
assert zone_result.confidence > 0.95     # High confidence
assert not zone_result.is_outlier        # Not an outlier
```

#### **Validation Layers:**

| Layer | What It Checks | Method | Threshold |
|-------|---------------|--------|-----------|
| **Acoustic Features** | F0, duration, harmonicity | Tolerance comparison | ±200 Hz, ±10 ms |
| **Audio Quality** | Clipping, DC offset, RMS | Peak/amplitude analysis | < -1 dBFS, < 0.01 |
| **Perceptual** | Cluster membership | Mahalanobis distance | < 2.0σ |
| **Semantic Zone** | Persona validity | Cluster boundaries | 95% confidence |
| **Statistical** | Effect size, confidence intervals | Cohen's d, t-tests | d > 0.8 |

#### **Test Status:**
- ✅ **32/32 tests passing** in `test_bio_acoustic_validation.py`
- ✅ **23/23 tests passing** in `test_persona_semantic_zone_validation.py`

**Test Categories:**
1. Acoustic Feature Validation (7 tests)
2. Audio Quality Validation (10 tests)
3. Perceptual Validation (6 tests)
4. Metadata-First Synthesis Validation (4 tests)
5. Bio-Acoustic Turing Test Scenarios (4 tests)
6. Statistical Validation (3 tests)

**Files:**
- `realtime/bio_acoustic_validator.py` - Comprehensive validator
- `realtime/persona_semantic_zone_validator.py` - Semantic zone validator
- `tests/test_bio_acoustic_validation.py` - 32 tests ✅
- `tests/test_persona_semantic_zone_validation.py` - 23 tests ✅

**Usage:**
```bash
# Run validation tests
python3 -m pytest tests/test_bio_acoustic_validation.py -v
python3 -m pytest tests/test_persona_semantic_zone_validation.py -v
```

---

### STEP 1.12: High-Dimensional Acoustic Algebra [NEW]

**30-Dimensional Vector Interpolation with Z-Score Normalization**

Enhanced Acoustic Algebra that uses all 30 micro-dynamics features for true timbral interpolation, not just pitch shifting. This enables **"Phonetic Constraints"** - discovering the physical limits of vocal production.

#### **The Problem with Simple Interpolation**

You cannot interpolate raw acoustic features:

| Feature | Range | Problem |
|---------|-------|---------|
| **F0** | 5000-8000 Hz | Large magnitude |
| **Attack Time** | 0.001-0.050 seconds | Small magnitude |
| **HNR** | -10 to +30 dB | Different units |

Without normalization, **Attack Time** and **HNR** dominate the interpolation math, while **F0** barely changes.

#### **The Solution: Z-Score Normalization**

Normalize all 30 features to standard deviations before interpolation:

```
Z = (X - μ) / σ  (Normalize)
V_target = Z_A * (1-α) + Z_B * α  (Interpolate in Z-space)
X_target = Z_target * σ + μ  (Denormalize)
```

#### **30-Dimensional Feature Vector**

```python
from analysis.rosetta_stone.high_dimensional_acoustic_algebra import (
    AcousticFeatureVector30,
    ZScoreNormalizer,
    AcousticAlgebraEngine30D
)

# Create 30-dim vectors
phee = AcousticFeatureVector30(
    # Fundamental (3)
    mean_f0_hz=6526,
    f0_range_hz=427,
    duration_ms=76.5,
    # Grit Factors (3)
    harmonic_to_noise_ratio=20.0,
    spectral_flatness=0.1,
    harmonicity=0.8,
    # Motion Factors (7)
    attack_time_ms=0.010,
    decay_time_ms=0.050,
    sustain_level=0.7,
    vibrato_rate_hz=8.0,
    vibrato_depth=0.03,
    jitter=0.02,
    shimmer=0.03,
    # Fingerprint Factors (13 MFCCs)
    mfcc_1=-500.0,
    mfcc_2=-100.0,
    mfcc_3=-50.0,
    mfcc_4=-20.0,
    mfcc_5=-0.5,
    mfcc_6=-0.3,
    mfcc_7=-0.2,
    mfcc_8=-0.1,
    mfcc_9=0.0,
    mfcc_10=0.1,
    mfcc_11=0.2,
    mfcc_12=0.3,
    mfcc_13=0.4,
    # Spectral Dynamics (1)
    spectral_flux=0.5,
    # Rhythm Factors (3)
    median_ici_ms=45.0,
    onset_rate_hz=12.0,
    ici_coefficient_of_variation=0.25
)

algebra = AcousticAlgebraEngine30D()

# Interpolate ALL features simultaneously
midpoint = algebra.interpolate(phee, alarm, alpha=0.5)

# Result: True timbral morph (not just pitch shift)
print(f"F0: {midpoint.mean_f0_hz:.0f} Hz")
print(f"Attack: {midpoint.attack_time_ms:.1f} ms")
print(f"HNR: {midpoint.harmonic_to_noise_ratio:.1f} dB")
print(f"Harmonicity: {midpoint.harmonicity:.2f}")
print(f"Flatness: {midpoint.spectral_flatness:.2f}")
```

#### **Phonetic Constraints**

By interpolating in 30-dimensional space (both island_hopping navigation and synthesis API), you can discover **"Physical Limits"** of vocal production:

```python
# Check for constraint violations
constraints = algebra.check_phonetic_constraints(midpoint)

if not constraints['valid']:
    for violation in constraints['violations']:
        print(f"⚠️  {violation}")

# Examples:
# - "HNR < 0 dB: -5.2 dB (Silence)"
# - "Attack < 0 ms: -2.1 ms (Impossible)"
# - "Duration <= 0 ms: 0.0 ms (Impossible)"
```

**Scientific Value:**
- **Phonetic Boundaries**: Discover where "Phee" turns into "Silence"
- **Timbral Continuums**: Morph between harmonic tonal and noisy sounds
- **Physical Limits**: Find the mathematical boundaries of vocal production

#### **Features (30 Dimensions)**

| Group | Features | Description |
|-------|----------|-------------|
| **Fundamental** | mean_f0_hz, duration_ms, f0_range_hz | Base frequency, duration, range |
| **Grit Factors** | harmonic_to_noise_ratio, spectral_flatness, harmonicity | Timbre texture |
| **Motion Factors** | attack_time_ms, decay_time_ms, sustain_level, vibrato_rate_hz, vibrato_depth, jitter, shimmer | Time envelope and modulation |
| **Fingerprint Factors** | mfcc_1-13, spectral_flux | Spectral envelope (14 dimensions) |
| **Rhythm Factors** | median_ici_ms, onset_rate_hz, ici_coefficient_of_variation | Temporal patterns |

**Files:**
- `analysis/rosetta_stone/high_dimensional_acoustic_algebra.py` - 30-dim algebra engine
- `tests/test_high_dimensional_algebra.py` - 21 tests passing ✅

---

### STEP 1.13: Grain-Based Grammar Discovery [NEW]

**Bottom-Up Discovery of "Atomic Phrases" and Grammar via Unsupervised Learning**

A powerful **"Bottom-Up"** discovery approach that shifts from **Label-Based Analysis** (what humans think) to **Physics-Based Analysis** (what the sound physically is).

#### **The Workflow: From Audio to Grammar**

```
┌─────────────────────────────────────────────────────────────────┐
│  1. GRAIN EXTRACTION                                            │
│     → Chop sentence into tiny grains (e.g., 10ms)              │
│     → Extract 30-dim feature vector per grain                    │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  2. DBSCAN CLUSTERING (Discovery)                              │
│     → Feed grain features into DBSCAN                           │
│     → Discover "Atomic Phrases" (words)                         │
│     → Assign label to each grain                                │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  3. SEQUENCE RECONSTRUCTION                                     │
│     → Reconstruct sentence as phrase sequence                  │
│     → E.g., [Phrase_A] → [Phrase_B] → [Phrase_A]               │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  4. TRANSITION ENTROPY ANALYSIS (Grammar)                       │
│     → Build transition matrix                                  │
│     → Calculate entropy: H(A) = -Σ p(x) log₂ p(x)              │
│     → Low entropy (0 bits): Predictable → Strong Grammar       │
│     → High entropy (2+ bits): Random → No Grammar               │
└─────────────────────────────────────────────────────────────────┘
```

#### **Step A: Grain Extraction**

Since you don't know where the "words" are yet, chop the sentence into tiny grains:

```python
# Input: sentence.wav (2 seconds long)
# Action: Slice into 200 segments of 10ms each
# Output: 200 feature vectors (30-dim each)

from analysis.rosetta_stone.grain_based_grammar_discovery import GrainExtractor

extractor = GrainExtractor(grain_duration_ms=10.0, hop_size_ms=5.0)
grains = extractor.extract_grains_from_features(features, sample_rate=48000)

print(f"Extracted {len(grains)} grains")
```

#### **Step B: DBSCAN Clustering**

Discover atomic phrases without human labels:

```python
from analysis.rosetta_stone.grain_based_grammar_discovery import AtomicPhraseDiscoverer

discoverer = AtomicPhraseDiscoverer(eps=0.5, min_samples=5)
phrases, labels = discoverer.discover_phrases(grains)

# DBSCAN assigns labels to each grain:
# - Grains 0-50: Label 0 (Phrase_A)
# - Grains 51-150: Label 1 (Phrase_B)
# - Grains 151-200: Label 0 (Phrase_A)

print(f"Discovered {len(phrases)} atomic phrases")
print(f"Sentence structure: [Phrase_A] → [Phrase_B] → [Phrase_A]")
```

#### **Step C: Transition Entropy Analysis**

The most appropriate technique for discovering **Structure** in unsupervised data:

```python
from analysis.rosetta_stone.grain_based_grammar_discovery import (
    SentenceReconstructor,
    TransitionEntropyAnalyzer
)

# Reconstruct sentence
reconstructor = SentenceReconstructor()
structure = reconstructor.reconstruct(grains)

# Analyze grammar via entropy
analyzer = TransitionEntropyAnalyzer()
grammar_stats = analyzer.analyze(structure)

print(f"Grammar rigidity: {grammar_stats.grammar_rigidity:.2f}")
print(f"Mean entropy: {grammar_stats.mean_entropy:.2f} bits")

# Interpretation:
# - H = 0 bits: "A always follows B" → Strong grammar
# - H = 1 bit: "A follows B with 50% probability" → Weak grammar
# - H = 2+ bits: "A can follow anything" → No grammar
```

#### **Variable-Length N-Grams**

Standard N-Grams (Bigrams/Trigrams) are good, but **Variable-Length N-Grams** are better for animal data:

```python
# Compresses sequences like "A-A-A-B" into "3A, 1B"
# Handles repetition (chirping/rattling) better than standard bigrams

n_grams = grammar_stats.n_grams
# Examples:
# - "((0, 5),)": 3 occurrences (Phrase A repeated 5 times)
# - "((0, 2), (1, 3))": 1 occurrence (Phrase A x2, then Phrase B x3)
```

#### **Complete Pipeline**

```python
from analysis.rosetta_stone.grain_based_grammar_discovery import GrammarDiscoveryPipeline

pipeline = GrammarDiscoveryPipeline(
    grain_duration_ms=10.0,
    dbscan_eps=0.5,
    dbscan_min_samples=5
)

# Run complete discovery
phrases, structure, grammar_stats = pipeline.discover_from_features(
    features,
    sample_rate=48000
)

# Results
print(f"Atomic phrases: {len(phrases)}")
print(f"Sentence: {structure.phrase_sequence}")
print(f"Grammar rigidity: {grammar_stats.grammar_rigidity:.2f}")
```

#### **Scientific Applications**

| Application | Technique | Value |
|------------|-----------|-------|
| **Word Discovery** | Grain → DBSCAN | Unsupervised discovery of sub-units |
| **Grammar Detection** | Transition Entropy | Measures "rigidity" of syntax |
| **Repetition Analysis** | Variable N-Grams | Handles chirping/rattling |
| **Structure Comparison** | Cross-species entropy | Compare syntax complexity |

**Key Insight:**
> **Label-Based Analysis**: "This is a 'phee' call" → Biased by human categories
> **Physics-Based Analysis**: "This has F0=6526, duration=76ms, HNR=20dB" → Unbiased discovery

**Files:**
- `analysis/rosetta_stone/grain_based_grammar_discovery.py` - Grammar discovery engine
- `tests/test_high_dimensional_algebra.py` - 21 tests passing ✅ (includes grammar tests)

**Test Coverage:**
- ✅ 30-dimensional feature vectors (13 tests) - island_hopping.rs, synthesis.rs (SourceMetadata API)
- ✅ Z-score normalization (3 tests)
- ✅ High-dimensional interpolation (4 tests)
- ✅ Phonetic constraints (3 tests)
- ✅ Grain extraction (3 tests)
- ✅ DBSCAN phrase discovery (3 tests)

---

### STEP 2: Data Import (`data_import/`)

Import the analyzed data into the unified database structure.

```bash
python3 src/data_import/import_vocalization_data.py
```

Creates `vocalization_database.json` with 2,882 phrases from 4 species.

---

### STEP 3: Query Interface (`query_interface/`)

High-performance querying with pre-built indexes.

```python
from src import get_query_interface

interface = get_query_interface()

# Search by F0 range
results = interface.search_phrases_by_f0_range(5000, 10000)

# Search by duration
results = interface.search_phrases_by_duration(30, 100)

# Find similar phrases
results = interface.find_similar_phrases(phrase, n=10)
```

---

### STEP 4: Cognitive Intelligence (`cognitive_intelligence/`, `semiotics/`)

Cognitive analysis for deception detection, innovation tracking, and cross-modal fusion.

```python
from src import SemioticEngine, SemioticContext

engine = SemioticEngine()
context = SemioticContext(species=Species.MARMOSET, ...)
result = engine.analyze_semiotics(phrase, context)
```

---

### STEP 5: Python Logic Layer (`realtime/`)

Cognitive decision making and phrase selection logic.

```python
from realtime.cognitive_layer import CognitiveLayer
from realtime.context_aware_synthesis import ContextAwareSynthesizer

# Make cognitive decisions
layer = CognitiveLayer()
decision = layer.decide(context, state)

# Select phrases based on context
synthesizer = ContextAwareSynthesizer()
selected = synthesizer.select_phrases(context, library)
```

---

## Audio Discovery and Synthesis Workflow [NEW]

### Overview

This framework implements a **hybrid Python/Rust architecture** for audio discovery and synthesis:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AUDIO DISCOVERY → SYNTHESIS WORKFLOW                    │
└─────────────────────────────────────────────────────────────────────────────┘

Python (Discovery)                Rust (Synthesis)
┌─────────────────────────┐       ┌─────────────────────────┐
│  1. Load Audio Files    │       │  1. Load .pkl Library   │
│  2. Extract Grains      │  ───► │  2. Parse Metadata      │
│  3. Discover Phrases    │       │  3. Synthesize Audio    │
│  4. Analyze Grammar     │       │     • Concatenative      │
│  5. Export .pkl Library │       │     • Superpositional    │
└─────────────────────────┘       │     • Combined           │
          │                         │     • Granular          │
          │                         └─────────────────────────┘
          ▼
    ┌─────────────────┐
    │  .pkl Library   │  ← Metadata bridge (3-5 KB)
    │                 │     • source_file paths
    │  • phrase_keys  │     • timestamps (start/end)
    │  • F0, duration │     • acoustic features
    │  • source_paths │     • quality scores
    └─────────────────┘
```

### Architecture Benefits

| Aspect | Python (Discovery) | Rust (Synthesis) |
|--------|-------------------|------------------|
| **Role** | Grain extraction, phrase discovery, grammar analysis | High-performance audio synthesis |
| **Strengths** | Rapid development, scientific computing (librosa, sklearn) | Zero-copy, memory safety, deterministic performance |
| **Library Size** | .pkl: ~3-5 KB (metadata only) | Audio segments loaded on-demand |
| **File Format** | PhraseAudioLibrary (.pkl) | PhraseSegment struct (Rust) |

### Phase 1: Discovery (Python)

**Module**: `realtime/audio_aware_grammar_discovery.py`

**Workflow**:
1. Load original audio files
2. Extract 10ms grains with audio samples
3. Run DBSCAN clustering to discover atomic phrases
4. Reconstruct sentence structure
5. Analyze grammar via transition entropy
6. Export PhraseAudioLibrary (.pkl) with source file paths

```python
from realtime.audio_aware_grammar_discovery import AudioAwareGrammarDiscovery

# Create pipeline
pipeline = AudioAwareGrammarDiscovery(
    grain_duration_ms=10.0,
    hop_size_ms=5.0,
    dbscan_eps=0.8,
    dbscan_min_samples=3,
    sample_rate=22050
)

# Load and process audio
pipeline.load_audio_file("recording.wav")
pipeline.extract_audio_grains(audio, sample_rate)
pipeline.discover_atomic_phrases()
pipeline.reconstruct_sentence()
pipeline.analyze_grammar()

# Export .pkl library (contains source file paths for extraction)
pipeline.build_phrase_library(
    species="marmoset",
    export_path="phrase_library.pkl"
)
```

**Output (.pkl contents)**:
```python
{
    'species': 'marmoset',
    'sr': 22050,
    'phrase_segments': {
        'F0_6400_DUR_10_RANGE_0': [
            {
                'phrase_key': 'F0_6400_DUR_10_RANGE_0',
                'source_file': '/path/to/recording.wav',  # Original file path
                'start_time_ms': 1234.5,
                'end_time_ms': 1244.5,
                'mean_f0_hz': 6400.0,
                'std_f0_hz': 50.0,
                'mean_duration_ms': 10.0,
                'mean_range_hz': 0,
                'encoding': 'horizontal',
                'quality_score': 0.85
            },
            # ... more occurrences
        ],
        # ... more phrase types
    },
    'total_segments': 42,
    'creation_time': '2026-01-06T04:34:49.057550'
}
```

### Phase 2: Extraction (Python - Optional)

**Module**: `realtime/phrase_library_segment_extractor.py`

**Purpose**: Extract actual audio segments from source files using .pkl metadata

```python
from realtime.phrase_library_segment_extractor import PhraseLibrarySegmentExtractor

# Load library and extract segments
extractor = PhraseLibrarySegmentExtractor(
    library_path="phrase_library.pkl",
    source_audio_dir="/path/to/audio/files"
)

# Extract all segments
output_files = extractor.extract_segments(
    output_dir="audio_segments",
    progress=True
)

# Result: audio_segments/F0_6400_DUR_10_RANGE_0_000.wav
#         audio_segments/F0_6400_DUR_10_RANGE_0_001.wav
#         ...
```

**Why this matters**:
- .pkl files are small (3-5 KB) - easy to share
- Source audio stays in one location
- Extract segments on-demand for synthesis
- Cross-session phrase recovery

### Phase 3: Synthesis (Rust)

**Module**: `technical_architecture/src/synthesis.rs`

**Modes**:
1. **Horizontal (Concatenative)**: Sequential phrase combination
2. **Vertical (Superpositional)**: Simultaneous phrase overlay
3. **Combined**: Mixed horizontal and vertical
4. **Granular**: Grain-level manipulation

```rust
use technical_architecture::synthesis::{
    EnhancedMicroharmonicSynthesizer,
    MicroharmonicConstraints,
    SynthesisMode
};

// Create synthesizer from .pkl or extracted segments
let synthesizer = EnhancedMicroharmonicSynthesizer::new(
    species.to_string(),
    phrase_segments,
    sample_rate
);

// Define constraints
let constraints = MicroharmonicConstraints {
    frequency_range: (4000.0, 8000.0),
    harmonic_tolerance: 0.5,
    phase_coherence: true,
    amplitude_balancing: true,
    temporal_alignment: "start".to_string(),
    crossfade_duration_ms: 5.0,
    max_phrases: 100,
    min_quality_score: 0.5,
};

// Horizontal (concatenative) synthesis
let result = synthesizer.synthesize_horizontal(
    vec!["F0_6400_DUR_10_RANGE_0".to_string()],
    &constraints
).await?;

// Vertical (superpositional) synthesis
let result = synthesizer.synthesize_vertical(
    vec![
        "F0_6400_DUR_10_RANGE_0".to_string(),
        "F0_6600_DUR_10_RANGE_0".to_string()
    ],
    &constraints
).await?;

// Combined synthesis
let synthesis_plan = vec![
    (SynthesisMode::Horizontal, vec!["F0_6400_DUR_10".to_string()]),
    (SynthesisMode::Vertical, vec!["F0_6400_DUR_10".to_string()]),
];
let result = synthesizer.synthesize_combined(synthesis_plan, &constraints).await?;
```

### Alternative: Python Synthesis (phrase_audio_library.py)

For development/testing without Rust:

```python
from realtime.phrase_audio_library import VocalizationSynthesizer, PhraseAudioLibrary

# Load library
library = PhraseAudioLibrary(species="marmoset", sr=22050)
library.load("phrase_library.pkl")

# Create synthesizer
synthesizer = VocalizationSynthesizer(library, crossfade_ms=5.0)

# Horizontal synthesis
audio, sr = synthesizer.synthesize_horizontal(
    phrase_sequence=['F0_6400_DUR_10_RANGE_0', 'F0_6600_DUR_10_RANGE_0'],
    gap_ms=0.0
)

# Vertical synthesis
audio, sr = synthesizer.synthesize_vertical(
    phrase_set=['F0_6400_DUR_10_RANGE_0', 'F0_6600_DUR_10_RANGE_0'],
    alignment="start"
)

# Combined synthesis
synthesis_plan = [
    ('horizontal', ['F0_6400_DUR_10_RANGE_0']),
    ('vertical', ['F0_6400_DUR_10_RANGE_0', 'F0_6600_DUR_10_RANGE_0'])
]
audio, sr = synthesizer.synthesize_combined(synthesis_plan)
```

### Complete Workflow Example

```bash
# Step 1: Discovery (Python)
python3 realtime/audio_aware_grammar_discovery.py recording.wav \
    --export-library phrase_library.pkl

# Step 2: Extract segments (optional - for direct file access)
python3 realtime/phrase_library_segment_extractor.py phrase_library.pkl \
    --source-audio-dir /path/to/audio \
    --output-dir audio_segments

# Step 3: Build Rust synthesis
cd technical_architecture
cargo build --release

# Step 4: Run synthesis (Rust)
./target/release/synthesize_from_pkl phrase_library.pkl \
    --mode horizontal \
    --output synthesized.wav
```

### Demo

```bash
# Run complete discovery to synthesis demo
python3 realtime/demo_discovery_to_synthesis.py
```

**This demo shows**:
1. Grain-based phrase discovery from synthetic audio
2. .pkl library creation with source file tracking
3. Audio segment extraction from source files
4. Rust synthesis integration (code examples)

### Key Files

| File | Purpose |
|------|---------|
| `realtime/audio_aware_grammar_discovery.py` | Grain-based discovery, .pkl export |
| `realtime/phrase_library_segment_extractor.py` | Extract audio from .pkl metadata |
| `realtime/phrase_audio_library.py` | Data structures, Python synthesis |
| `technical_architecture/src/synthesis.rs` | Rust synthesis (horizontal, vertical, granular) |
| `realtime/demo_discovery_to_synthesis.py` | Complete workflow demo |
| `realtime/demo_audio_aware_grammar_discovery.py` | Discovery demo |

### .pkl Format Compatibility

The .pkl files are designed to bridge Python discovery and Rust synthesis:

| Rust PhraseSegment Field | Python .pkl Source |
|------------------------|-------------------|
| `audio: Vec<f32>` | `segment.audio` (numpy array) |
| `duration_ms: f32` | `segment.mean_duration_ms` |
| `mean_f0_hz: f32` | `segment.mean_f0_hz` |
| `f0_range_hz: f32` | `segment.mean_range_hz` |
| `std_f0_hz: f32` | `segment.std_f0_hz` |
| `quality_score: f32` | `segment.quality_score` |
| `sample_rate: usize` | `segment.sr` |

### File Sizes

| Content | Size |
|---------|------|
| .pkl library (metadata) | ~3-5 KB |
| Source audio (original) | ~100+ MB |
| Extracted segments | ~10-50 MB |
| **Total (with .pkl)** | ~100 MB |
| **Without .pkl** | ~200 MB (audio embedded) |

**Storage savings**: ~50% by keeping audio separate and using .pkl metadata

---

## Field Deployment: Online Phrase Discovery [NEW]

### Overview

In the field, **offline batch processing** (DBSCAN every night) is not feasible. The system must switch to **online/stream processing** for real-time phrase discovery.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LAB vs FIELD: WORKFLOW COMPARISON                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LAB (OFFLINE)                    FIELD (REAL-TIME)                         │
│  ┌──────────────────────┐         ┌──────────────────────┐                 │
│  │ 1. Record all audio  │         │ 1. Stream to KNN     │                 │
│  │ 2. Store for batch   │         │ 2. Detect unknown    │                 │
│  │ 3. Run DBSCAN        │◄────vs──►│ 3. Cold store        │                 │
│  │ 4. Manual validate   │         │ 4. Repetition check  │                 │
│  │ 5. Update library    │         │ 5. Hot swap to Rust  │                 │
│  └──────────────────────┘         └──────────────────────┘                 │
│                                                                             │
│  Latency: Minutes/Hours            Latency: Milliseconds                     │
│  Validation: Manual                Validation: Statistical (repetition)      │
│  Discovery: DBSCAN (batch)         Discovery: KNN (streaming)               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Concepts

| Concept | Lab (DBSCAN) | Field (KNN) |
|---------|--------------|-------------|
| **Discovery** | DBSCAN re-processes all data | KNN/thresholding on each frame |
| **Validation** | Visual inspection by researcher | Repetition count (statistical) |
| **Storage** | Append to master `.pkl` | Cold store → Hot swap to RAM |
| **Latency** | Minutes/Hours | Milliseconds |
| **Deployment** | Offline analysis | Online/stream processing |

### Two-Stage Validation: "Cold Storage" → "Hot Swap"

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHRASE LIFECYCLE IN FIELD                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [T=0:00] New Sound Detected                                               │
│     │                                                                      │
│     ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 1: COLD STORAGE (Immediate)                                   │   │
│  │                                                                     │   │
│  │  • Save audio to disk immediately                                    │   │
│  │  • Label: UNKNOWN_001                                              │   │
│  │  • State: CANDIDATE                                                 │   │
│  │  • NOT available for synthesis (prevents babbling)                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│     │                                                                      │
│     ▼                                                                      │
│  [T=0:05] Same sound detected again                                        │
│     │                                                                      │
│     ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 2: REPETITION VALIDATION                                     │   │
│  │                                                                     │   │
│  │  • Increment count: UNKNOWN_001 → count=2                          │   │
│  │  • Wait for CONFIDENCE_THRESHOLD (e.g., 3 repetitions)            │   │
│  │  • Statistical validation (not manual)                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│     │                                                                      │
│     ▼                                                                      │
│  [T=0:10] Third occurrence - THRESHOLD REACHED                             │
│     │                                                                      │
│     ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 3: HOT SWAP (Promotion)                                       │   │
│  │                                                                     │   │
│  │  • Rename: UNKNOWN_001 → DISCOVERED_42                             │   │
│  │  • State: ACTIVE                                                    │   │
│  │  • Load into Rust engine (async, non-blocking)                     │   │
│  │  • Available for immediate synthesis                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│     │                                                                      │
│     ▼                                                                      │
│  [T=0:15] Agent responds using DISCOVERED_42 (learned mid-conversation)    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementation

**Module**: `realtime/online_phrase_discovery_agent.py`

```python
from realtime.online_phrase_discovery_agent import (
    OnlinePhraseDiscoveryAgent,
    DiscoveryConfig
)

# Create agent
config = DiscoveryConfig(
    known_phrase_threshold=2.0,        # Z-score < 2.0 = known
    confidence_threshold=3,             # Need 3 repetitions
    validation_window_sec=300.0,       # 5 min validation window
    enable_rust_bridge=True            # Async hot swapping
)

agent = OnlinePhraseDiscoveryAgent("phrase_library.pkl", config)

# Start background monitoring (checks for repetitions)
agent.start_background_monitor(interval_sec=5.0)

# Process live audio stream
for audio_buffer in audio_stream:
    result = agent.process_live_audio(audio_buffer, sample_rate)

    if result == "UNKNOWN_DETECTED":
        print("New phrase detected - cold stored for validation")

# Get synthesis candidates (with babble prevention)
candidates = agent.get_synthesis_candidates(intent="aggression")
```

### Smart Babble Prevention

A risk in field deployment is **false positives** (wind, clicks treated as phrases).

**Strategy**:
1. **Lower selection weight** for candidate phrases (0.2 vs 1.0)
2. **Maximum candidate ratio** (e.g., 30% candidates vs active)
3. **Repetition threshold** (minimum 3 occurrences)

```python
def get_synthesis_candidates(self, intent: str, max_candidates: int = 10):
    """
    Get candidates for synthesis with babble prevention.

    Prioritizes validated phrases over candidates.
    """
    all_phrases = []

    # Active phrases (weight = 1.0)
    for phrase in self.active_phrases.values():
        all_phrases.append((phrase, 1.0))

    # Candidate phrases (weight = 0.2) - lower priority
    if len(self.candidates) / len(self.active_phrases) < 0.3:
        for phrase in self.candidates.values():
            all_phrases.append((phrase, 0.2))

    # Weighted lottery selection
    selected = []
    for phrase, weight in all_phrases:
        if random.random() < weight:
            selected.append(phrase)

    return selected
```

### Async Rust Bridge for Hot Swapping

The Rust engine must support **async loading** to prevent conversation freezing.

```rust
// synthesis.rs

impl GranularConcatenativeSynthesizer {
    /// Called by Python thread asynchronously
    pub fn load_source_async(&self, phrase_id: String, path: PathBuf) {
        // 1. Load file in background thread
        let buffer = self.audio_io.load_wav_blocking(path);

        // 2. Lock and update
        let mut voices = self.voices.lock().unwrap();
        voices.insert(phrase_id, GrainVoice::new(buffer));

        // 3. Update registry (available immediately)
        self.registry.register_id(phrase_id);
    }
}
```

### Comparison Table

| Operation | Lab (DBSCAN) | Field (KNN) |
|-----------|--------------|-------------|
| **Discovery** | Re-processes all data | Incremental, per-frame |
| **Validation** | Visual inspection | Repetition count |
| **Storage** | Append to `.pkl` | Cold store + Hot swap |
| **Latency** | Minutes/Hours | Milliseconds |
| **False Positives** | Low (human check) | Medium (statistical) |
| **Mid-conversation learning** | No | Yes (hot swap) |

### Files

| File | Purpose |
|------|---------|
| `realtime/online_phrase_discovery_agent.py` | Online KNN discovery, cold storage, hot swap |
| `realtime/demo_online_phrase_discovery.py` | Field deployment demo |
| `technical_architecture/src/synthesis.rs` | Rust async loading |

### Demo

```bash
# Run field deployment demo
python3 realtime/demo_online_phrase_discovery.py
```

**Demo shows**:
1. KNN-based detection (not DBSCAN)
2. Cold storage buffering
3. Repetition-based validation
4. Async hot swap to Rust
5. Smart babble prevention
6. Lab vs Field workflow comparison

---

## Context-Aware Phrase Discovery: The Crucial Missing Link [NEW]

### Overview

Behavioral context is the **crucial missing link** for intelligent synthesis. Without context, the system cannot:
- Select appropriate phrases for synthesis (random babbling)
- Detect deception (signal vs context mismatch)
- Study behavioral function (which phrases in which contexts)
- Respond appropriately in field deployment

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THE MISSING LINK: CONTEXT                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  WITHOUT Context (Before):                    WITH Context (After):          │
│  ┌──────────────────────────────┐             ┌─────────────────────────────┐│
│  │ Discovers:                  │             │ Discovers:                 ││
│  │   F0_7000_DUR_250           │             │   F0_7000_DUR_250           ││
│  │   F0_6000_DUR_250           │  ────vs────►│   F0_6000_DUR_250           ││
│  │   F0_5000_DUR_250           │             │   F0_5000_DUR_250           ││
│  │                             │             │                             ││
│  │ Animal shows aggression:     │             │ Animal shows aggression:   ││
│  │   System: ??? (random)       │             │   System: aggression pool  ││
│  │   Result: ❌ Might use       │             │   Result: ✓ Uses F0_7000   ││
│  │           courtship phrase   │             │           (aggression)      ││
│  └──────────────────────────────┘             └─────────────────────────────┘│
│                                                                             │
│  ❌ Random babbling                ✓ Intelligent synthesis              │
│  ❌ Cannot detect deception          ✓ Deception detection             │
│  ❌ Unknown behavioral function      ✓ Behavioral analysis            │
│  ❌ Inappropriate field responses    ✓ Appropriate responses          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### What is Behavioral Context?

Behavioral context describes **why** a vocalization was produced:
- **Aggression**: Threat displays, territorial disputes
- **Courtship**: Mating calls, pair bonding
- **Food Discovery**: Food found, sharing
- **Alarm**: Predator alert, danger
- **Contact**: Group cohesion, location

### Annotation Formats Supported

The `annotation_loader.py` module supports multiple formats:

| Format | Extension | Description |
|--------|-----------|-------------|
| **ELAN** | .eaf | XML-based multi-tier annotation (common in primatology) |
| **Praat** | .TextGrid | Interval/point tiers (linguistics/phonetics) |
| **JSON** | .json | Custom structured annotations |
| **CSV** | .csv | Tabular time-stamped annotations |

### Implementation

**Module**: `realtime/annotation_loader.py`

```python
from realtime.annotation_loader import AnnotationLoader

# Load annotations (auto-detects format)
loader = AnnotationLoader()
annotations = loader.load("annotations.json")

# Access annotations
context = annotations.get_primary_context(time_ms=1500)
individual = annotations.get_context_at_time(time_ms=1500, track="individual")
```

**Integration with Discovery Pipeline**:

```python
from realtime.audio_aware_grammar_discovery import AudioAwareGrammarDiscovery

# Create pipeline
pipeline = AudioAwareGrammarDiscovery()

# Load audio AND annotations
pipeline.load_audio_file("recording.wav")
pipeline.load_annotations("annotations.json")  # ✓ NEW

# Discover phrases
pipeline.extract_audio_grains(audio, sr)
pipeline.discover_atomic_phrases()

# Build library WITH context association
library = pipeline.build_phrase_library(
    species="marmoset",
    export_path="library.pkl",
    associate_context=True  # ✓ Enable context
)
```

### Context-Aware Synthesis

Once the library has context, synthesis becomes intelligent:

```python
from realtime.phrase_audio_library import PhraseAudioLibrary

# Load library
library = PhraseAudioLibrary()
library.load("library.pkl")

# Get phrases for specific context
aggression_phrases = library.get_phrases_by_context("aggression")
food_phrases = library.get_phrases_by_context("food_discovery")

# Get segments with context filter
segments = library.get_segments_for_synthesis_by_context(
    phrase_keys=aggression_phrases,
    context="aggression",
    strategy="best"
)

# Synthesize appropriate response
synthesizer = VocalizationSynthesizer(library)
audio, sr = synthesizer.synthesize_horizontal(
    phrase_sequence=[s.phrase_key for s in segments]
)
```

### Deception Detection

Context enables deception detection by comparing **signal vs reality**:

```
Example: Food Call Deception

Observation:
  Time: 10:30 AM
  Individual: marmoset_A
  Call: "food_discovery" (F0_5000 phrase)
  Context: NO FOOD PRESENT

Analysis:
  Signal: "food found"
  Reality: "no food"
  Mismatch: DECEPTION or FALSE ALARM

With Context Tracking:
  1. Count mismatches per individual
  2. Compare to baseline (honest vs deceptive rate)
  3. Identify chronic deceivers
  4. Study social dynamics of deception
```

### Annotation File Examples

**JSON Format**:
```json
{
  "metadata": {
    "species": "marmoset",
    "recording_date": "2026-01-06"
  },
  "annotations": [
    {
      "start_time_ms": 0.0,
      "end_time_ms": 250.0,
      "context": "aggression",
      "individual_id": "marmoset_A",
      "notes": "Territorial dispute"
    },
    {
      "start_time_ms": 300.0,
      "end_time_ms": 550.0,
      "context": "courtship",
      "individual_id": "marmoset_B",
      "notes": "Mating call"
    }
  ]
}
```

**CSV Format**:
```csv
start_time_ms,end_time_ms,context,individual_id,notes
0.0,250.0,aggression,marmoset_A,Territorial dispute
300.0,550.0,courtship,marmoset_B,Mating call
600.0,850.0,food_discovery,marmoset_A,Food found
```

### Context Statistics

The library tracks context-phrase associations:

```python
stats = library.get_context_statistics()

# Sample output:
{
  'context_statistics': {
    'aggression': {
      'total_occurrences': 15,
      'phrases': [
        {
          'phrase_key': 'F0_7000_DUR_250_RANGE_0',
          'count': 12,
          'probability': 0.80,
          'enrichment': 4.5  # 4.5x more likely in aggression
        },
        ...
      ],
      'num_unique_phrases': 3
    },
    'courtship': {
      'total_occurrences': 20,
      ...
    }
  },
  'segments_with_context': 42,
  'segments_without_context': 5
}
```

### Files

| File | Purpose |
|------|---------|
| `realtime/annotation_loader.py` | Load ELAN/Praat/JSON/CSV annotations |
| `realtime/audio_aware_grammar_discovery.py` | Enhanced with `load_annotations()` |
| `realtime/demo_context_aware_discovery.py` | Complete context-aware workflow demo |

### Demo

```bash
# Run context-aware discovery demo
python3 realtime/demo_context_aware_discovery.py
```

**Demo shows**:
1. Synthetic audio with different behavioral contexts
2. Annotation file creation (JSON format)
3. Phrase discovery with context association
4. Context-aware synthesis
5. Deception detection simulation
6. Comparison: with vs without context

### Impact Summary

| Aspect | Without Context | With Context |
|--------|----------------|--------------|
| **Synthesis** | Random babbling | Intelligent selection |
| **Deception Detection** | Impossible | Signal vs context analysis |
| **Behavioral Research** | Unknown function | Context-function mapping |
| **Field Deployment** | Inappropriate responses | Contextually appropriate |
| **Scientific Value** | Limited | Comprehensive |

**This is the CRUCIAL MISSING LINK** that transforms random audio processing into intelligent behavioral analysis!

---

## PRODUCTION DEPLOYMENT PIPELINE

### End-to-End Workflow for Deployed Systems

This section provides the complete methodology for deploying animal vocalization analysis systems in the field, from raw audio to cognitive understanding with species-specific source separation.

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PRODUCTION DEPLOYMENT PIPELINE                          │
└─────────────────────────────────────────────────────────────────────────────┘

Raw Audio
   │
   ├─► [STEP 1] Species-Specific Source Separation
   │       └─► Separate target species from background noise
   │       └─► Uses Conv-TasNet models optimized for species F0 range
   │
   ├─► [STEP 2] Phrase, Sentence & Grammar Extraction
   │       ├─► Segment phrases (acoustic units)
   │       ├─► Detect sentences (phrase sequences)
   │       ├─► Discover grammar (transition patterns)
   │       └─► Context association (environmental, temporal)
   │
   ├─► [STEP 3] Synthesis Method Selection
   │       ├─► Concatenative (perfect fidelity, low flexibility)
   │       └─► Granular Concatenative (near-perfect fidelity, high flexibility)
   │
   ├─► [STEP 4] Cognitive Intelligence
   │       ├─► Phrase selection based on context
   │       ├─► Semiotic analysis (deception, innovation)
   │       └─► Cross-modal data fusion
   │
   └─► [STEP 5] Response Generation
       ├─► Rust execution layer (safety-critical)
       ├─► Environmental monitoring
       └─► Real-time synthesis
```

---

### STEP 1: Species-Specific Source Separation

**Objective**: Isolate target species vocalizations from environmental noise and other species.

#### Train Species-Specific Models

```bash
# Train models for your target species
cd cognitive_intelligence

# Option A: Train all species
python train_asteroid_multispecies.py --all

# Option B: Train specific species
python train_asteroid_multispecies.py --species marmoset egyptian_bat

# Option C: Train individual species
python train_asteroid_marmoset.py       # F0: 4000-8000 Hz
python train_asteroid_bat.py            # F0: 100-17000 Hz
python train_asteroid_dolphin.py        # F0: 500-16000 Hz
python train_asteroid_chimpanzee.py     # F0: 100-1900 Hz
```

#### Species-Specific Configuration

| Species | F0 Range | Filter Range | Sample Rate | Use Case |
|---------|----------|--------------|-------------|----------|
| **Marmoset** | 4000-8000 Hz | 2800-10400 Hz | 44.1kHz | Mid-frequency primate |
| **Egyptian Fruit Bat** | 100-17000 Hz | 100-22100 Hz | 96kHz* | Wide range, FM sweeps |
| **Dolphin** | 500-16000 Hz | 350-20800 Hz | 96kHz* | Whistles, clicks |
| **Chimpanzee** | 100-1900 Hz | 100-2470 Hz | 44.1kHz | Low-frequency primate |

*Use 96kHz for ultrasonic vocalizations (bats, dolphins)

#### Deploy Source Separation in Rust

```rust
// Update technical_architecture/src/source_separation.rs
use technical_architecture::SourceSeparator;

let config = SeparatorConfig {
    model_path: "models/checkpoints/marmoset/conv_tasnet_marmoset.onnx",
    sample_rate: 44100,
    num_sources: 2,  // Target + background
    chunk_size: 4096,
};

let separator = SourceSeparator::new(config)?;
let separated = separator.separate(audio_buffer)?;
```

**Output**: Clean audio with target species isolated from background.

---

### STEP 2: Complete Extraction Pipeline (One Script)

**New Script**: `analysis/rosetta_stone/complete_extraction_pipeline.py`

Extracts phrases, sentences, grammar, and segmented audio in one pass with context association.

```bash
# Run complete extraction pipeline
python analysis/rosetta_stone/complete_extraction_pipeline.py \
    --input audio/field_recording.wav \
    --species marmoset \
    --output results/marmoset_session_001 \
    --separate-model models/checkpoints/marmoset/conv_tasnet_marmoset.onnx
```

#### What It Extracts

**1. Phrase Segmentation**
```python
# Acoustic phrase extraction
phrases = [
    Phrase(
        phrase_key="F0_7400_DUR_50_RANGE_300",
        audio_segment=audio[start:end],
        f0_mean_hz=7400,
        duration_ms=50,
        f0_range_hz=300,
        modality="harmonic",
        timestamp_ms=1234.5
    ),
    # ... more phrases
]
```

**2. Sentence Detection**
```python
# Phrase sequences with timing
sentences = [
    Sentence(
        sentence_id=1,
        phrase_sequence=["F0_7400", "F0_7800", "F0_7500"],
        start_time_ms=1234.5,
        end_time_ms=1456.8,
        gap_pattern="medium"  # Inter-phrase gap pattern
    ),
    # ... more sentences
]
```

**3. Grammar Discovery**
```python
# Transition probability matrix
grammar = {
    "F0_7400": {"F0_7800": 0.6, "F0_7500": 0.3, "F0_7200": 0.1},
    "F0_7800": {"F0_7500": 0.7, "F0_7400": 0.2, "F0_8000": 0.1},
    # ... more transitions
}

# Syntax rules discovered from patterns
syntax_rules = [
    "PHEE_CALL → TRILL → TWITTER",  # Common sequence
    "PHEE_CALL → PHEE_CALL (0.3-1.0s gap)",  # Repetition pattern
]
```

**4. Context Association**
```python
# Environmental and temporal context
context = {
    "time_of_day": "dawn",  # From timestamp
    "weather": "clear",     # From environmental sensors
    "location": "feeding_site",  # From GPS/cage ID
    "social_context": "group_present",  # From multi-animal detection
    "previous_interactions": [
        {"type": "response_to", "speaker": "marmoset_002", "latency_s": 0.5}
    ]
}
```

**5. Segmented Audio Export**
```
results/marmoset_session_001/
├── phrases/
│   ├── F0_7400_DUR_50_RANGE_300_phrase_001.wav
│   ├── F0_7800_DUR_50_RANGE_200_phrase_002.wav
│   └── ...
├── sentences/
│   ├── sentence_001_phee_call_sequence.wav
│   ├── sentence_002_trill_sequence.wav
│   └── ...
├── grammar.json
├── phrases.json
├── sentences.json
└── context.json
```

#### Python API

```python
from analysis.rosetta_stone import CompleteExtractionPipeline
from technical_architecture import SourceSeparator

# Initialize pipeline
pipeline = CompleteExtractionPipeline(
    species="marmoset",
    sample_rate=44100,
    source_separator=SourceSeparator(species_model_path="...")
)

# Process audio file
results = pipeline.process(
    audio_path="audio/field_recording.wav",
    extract_audio_segments=True,
    discover_grammar=True,
    associate_context=True
)

# Access results
print(f"Extracted {len(results['phrases'])} phrases")
print(f"Detected {len(results['sentences'])} sentences")
print(f"Grammar: {results['grammar']}")
print(f"Context: {results['context']}")
```

---

### STEP 3: Synthesis Method Selection

**Decision Tree**: Choose the right synthesis method for your use case.

```
                    Start
                      │
          Do you need parameter variation?
                      │
           ┌──────────┴──────────┐
           │                     │
          YES                   NO
           │                     │
    Need specific pitch?    Have exact segment?
           │                     │
    ┌──────┴──────┐        ┌────┴────┐
   YES           NO       YES       NO
    │             │         │         │
Granular     Granular   Concatenative  [Error: No audio]
(1 voice)   (Morpher)   (perfect)
```

#### Concatenative Synthesis

**Use when**: You have exact audio segments and need perfect fidelity.

```python
# Load segmented phrases from extraction pipeline
from realtime.phrase_audio_library import PhraseAudioLibrary

library = PhraseAudioLibrary.load("results/marmoset_session_001/")

# Select phrases by acoustic features
phrases = library.get_phrases(
    f0_min=7000,
    f0_max=8000,
    duration_min=40,
    duration_max=60
)

# Concatenate (perfect fidelity, no manipulation)
output = library.concatenate(phrases)
```

**Characteristics**:
- ✅ Perfect fidelity (t-SNE distance: 4.2)
- ✅ Preserves all natural characteristics
- ❌ No parameter flexibility
- ❌ Limited to available phrases

#### Granular Concatenative Synthesis

**Use when**: You need systematic parameter variation while preserving formants.

```python
from technical_architecture import GranularConcatenativeSynthesizer

# Load source phrase
source_phrase = library.get_phrase("F0_7400_DUR_50_RANGE_300")

# Create synthesizer
synth = GranularConcatenativeSynthesizer(sample_rate=22050)
synth.load_source(source_phrase.audio)

# Systematic parameter variation
for pitch_shift in [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]:
    synth.set_pitch_shift(pitch_shift)
    output = synth.synthesize(duration_ms=50.0)

    # Generate pitch continuum (7400, 7030, 6660, 7400, 8140, 8880, 9620 Hz)
    # Even if these exact pitches don't exist in database!
```

**Characteristics**:
- ✅ Near-perfect fidelity (t-SNE distance: 6.452)
- ✅ Preserves formant structure
- ✅ Enables parameter variation
- ✅ 76.1% better than additive synthesis
- ⚠️ Requires real audio source

#### Comparison Table

| Feature | Concatenative | Granular | Additive |
|---------|---------------|----------|----------|
| **Fidelity (t-SNE)** | 4.208 | 6.452 | 27.052 |
| **Formant Preservation** | ✅ Perfect | ✅ Excellent | ❌ Poor |
| **Pitch Flexibility** | ❌ None | ✅ Excellent | ✅ Excellent |
| **Time Flexibility** | ❌ None | ✅ Yes | ✅ Yes |
| **Requires Real Audio** | ✅ Yes | ✅ Yes | ❌ No |
| **Use Case** | Natural playback | Systematic variation | Synthetic sounds |

---

### STEP 4: Species-Specific Source Separation (Production)

#### Model Selection Guide

**For Field Deployment**:

```python
# Deploy with species-specific models
DEPLOYMENT_CONFIG = {
    "location_jungle": {
        "primary_species": "marmoset",
        "separator_model": "models/checkpoints/marmoset/conv_tasnet_marmoset.onnx",
        "f0_range": (4000, 8000),
        "filter_range": (2800, 10400),
    },
    "location_cave": {
        "primary_species": "egyptian_bat",
        "separator_model": "models/checkpoints/egyptian_bat/conv_tasnet_egyptian_bat.onnx",
        "f0_range": (100, 17000),
        "filter_range": (100, 22100),
        "sample_rate": 96000,  # Higher for ultrasonic
    },
    "location_marine": {
        "primary_species": "dolphin",
        "separator_model": "models/checkpoints/dolphin/conv_tasnet_dolphin.onnx",
        "f0_range": (500, 16000),
        "filter_range": (350, 20800),
        "sample_rate": 96000,  # For ultrasonic clicks
    },
}
```

#### Multi-Species Environments

```python
# For environments with multiple species
from technical_architecture import SourceSeparator, MultiSpeciesSeparator

# Load multiple models
separator = MultiSpeciesSeparator()
separator.load_model("marmoset", "models/checkpoints/marmoset/...")
separator.load_model("egyptian_bat", "models/checkpoints/egyptian_bat/...")

# Auto-detect species by frequency content
audio_buffer = read_audio("field_recording.wav")
species = separator.detect_species(audio_buffer)  # Returns: "marmoset"

# Use appropriate model
separated_audio = separator.separate(audio_buffer, model=species)
```

#### Rust Integration

```rust
// technical_architecture/src/source_separation.rs

pub struct SpeciesSeparator {
    models: HashMap<String, TractModel>,
    active_model: Option<String>,
}

impl SpeciesSeparator {
    pub fn detect_species(&self, audio: &[f32]) -> Option<String> {
        // Analyze frequency content
        let spectrum = self.compute_spectrum(audio);

        // Check F0 ranges
        let dominant_f0 = self.find_dominant_f0(&spectrum);

        match dominant_f0 {
            100..=1900 => Some("chimpanzee"),
            2800..=10400 => Some("marmoset"),
            100..=22100 => Some("egyptian_bat"),
            350..=20800 => Some("dolphin"),
            _ => None,
        }
    }

    pub fn separate(&mut self, audio: &[f32]) -> Result<Vec<f32>> {
        // Auto-detect species
        let species = self.detect_species(audio)
            .ok_or_else(|| anyhow!("Cannot detect species"))?;

        // Load appropriate model
        self.load_model_if_needed(species)?;

        // Separate
        self.models.get_mut(species).unwrap().separate(audio)
    }
}
```

---

### STEP 5: Response Generation

#### Complete Response Pipeline

```python
# Full pipeline: extraction → analysis → synthesis
from analysis.rosetta_stone import CompleteExtractionPipeline
from realtime.cognitive_layer import CognitiveLayer
from technical_architecture import GranularConcatenativeSynthesizer

# Step 1: Extract (with species-specific source separation)
pipeline = CompleteExtractionPipeline(species="marmoset")
results = pipeline.process(audio_path, separate_species=True)

# Step 2: Analyze (cognitive intelligence)
cognitive = CognitiveLayer()
response_type = cognitive.decide(
    context=results['context'],
    grammar=results['grammar'],
    detected_phrases=results['phrases']
)

# Step 3: Synthesize (concatenative or granular)
if response_type.requires_parameter_variation:
    # Use granular for systematic variation
    synth = GranularConcatenativeSynthesizer(sample_rate=22050)
    synth.load_source(response_type.source_phrase.audio)
    synth.set_pitch_shift(response_type.pitch_shift)
    output = synth.synthesize(duration_ms=response_type.duration_ms)
else:
    # Use concatenative for perfect fidelity
    output = library.concatenate(response_type.target_phrases)

# Step 4: Safety checks and playback
from technical_architecture import SafetyMonitor
safety = SafetyMonitor()
if safety.check_audio_safety(output):
    play_audio(output)
```

---

## METHODOLOGY RECOMMENDATIONS

### Latest Scientific Findings

**1. Granular Synthesis Achieves Bio-Acoustic Fidelity**
- t-SNE distance: 6.452 (< 7.0 target) ✅
- 76.1% improvement over additive synthesis
- Preserves formant structure while enabling parameter variation
- **Implication**: Use granular synthesis for systematic experiments

**2. Additive Synthesis is Insufficient**
- t-SNE distance: 27.052 (failed)
- Cannot capture inharmonic partials and formant structure
- **Implication**: Avoid additive synthesis for bio-acoustic research

**3. Species-Specific Source Separation Improves Accuracy**
- General model: 70-80% separation accuracy
- Species-specific model: 85-95% separation accuracy
- **Implication**: Train species-specific models for deployment

### Decision Matrix

| Scenario | Extraction Method | Synthesis Method | Source Separation |
|----------|-------------------|------------------|-------------------|
| **Basic analysis** | Universal Rosetta Stone | N/A | Not needed |
| **Playback natural calls** | Segmented phrases | Concatenative | Not needed |
| **Pitch continuum testing** | Segmented phrases | Granular | Not needed |
| **Noisy field recordings** | + Source separation | Concatenative/Granular | Species-specific |
| **Multi-species environment** | + Species detection | Concatenative/Granular | Multi-model |
| **Real-time interaction** | Real-time extraction | Granular (fast) | Species-specific |

### Production Checklist

**Before Field Deployment**:
- [ ] Train species-specific source separation model
- [ ] Test extraction pipeline on field recordings
- [ ] Validate synthesis method (concatenative vs granular)
- [ ] Configure Rust safety limits
- [ ] Test environmental monitoring integration
- [ ] Verify power management configuration

**During Deployment**:
- [ ] Monitor separation quality (log SI-SDR metrics)
- [ ] Track synthesis performance (t-SNE validation)
- [ ] Record environmental context
- [ ] Check thermal and power status
- [ ] Validate safety limits

**Post-Deployment**:
- [ ] Analyze extracted phrases and sentences
- [ ] Update grammar rules
- [ ] Re-train models if performance degrades
- [ ] Archive audio segments for future research

---

### STEP 6: Rust Execution Layer (`technical_architecture/`)

Safety-critical audio processing, granular synthesis, and field deployment.

```rust
use technical_architecture::TechnicalArchitect;

let architect = TechnicalArchitect::new(config).await?;
let result = architect.process_audio_frame(audio, intent).await?;
```

**Granular Concatenative Synthesis** - High-fidelity vocalization synthesis with vector delta commands:

```python
from technical_architecture import GranularConcatenativeSynthesizer, SourceMetadata

# Create synthesizer
synth = GranularConcatenativeSynthesizer(sample_rate=22050)

# Load source audio with metadata (enables vector delta commands)
metadata = SourceMetadata(
    mean_f0_hz=6800.0,      # Source F0
    duration_ms=50.0,       # Source duration
    f0_range_hz=400.0       # Source F0 range
)
synth.load_source_with_metadata(audio_buffer, metadata)

# Apply vector delta (relative to source!)
# This integrates with Acoustic Algebra: delta = virtual - nearest_real
synth.shift_pitch_by_hz(200.0)      # 6800 + 200 = 7000Hz
synth.shift_duration_by_ms(-10.0)   # 50 - 10 = 40ms

# Or apply all shifts at once (complete vector delta)
synth.apply_vector_delta(
    delta_f0_hz=200.0,         # Pitch shift
    delta_duration_ms=-10.0,   # Duration shift
    delta_f0_range_hz=100.0    # F0 range shift
)

# Synthesize at target duration
output = synth.synthesize(duration_ms=40.0)
```

**Vector Delta Commands** [NEW]:
- **Why Delta Commands Matter**: Acoustic Algebra generates absolute targets (e.g., "F0=7000Hz"), but Rust synthesis needs relative shifts from the source buffer
  - **Bad Command (Absolute)**: "Set pitch to 7000Hz" - Ignores that we started at 6800Hz
  - **Good Command (Delta)**: "Shift pitch by +200Hz relative to source" - Automatically adjusts based on source
- **Use Case**: Critical integration point between Acoustic Algebra and Rust synthesis
- **Documentation**: See `analysis/rosetta_stone/VECTOR_DELTA_INTEGRATION.md` for complete guide

**30D Micro-Dynamics Metadata (Synthesis API)** [NEW]:
- **Full Feature Support**: All 30 micro-dynamics features now tracked and accessible
  - **Fundamental (3)**: `mean_f0_hz`, `duration_ms`, `f0_range_hz`
  - **Grit Factors (3)**: `harmonic_to_noise_ratio`, `spectral_flatness`, `harmonicity` (timbre texture)
  - **Motion Factors (7)**: `attack_time_ms`, `decay_time_ms`, `sustain_level`, `vibrato_rate_hz`, `vibrato_depth`, `jitter`, `shimmer` (envelope dynamics)
  - **Fingerprint Factors (14)**: `mfcc_1` through `mfcc_13`, `spectral_flux` (spectral shape)
  - **Rhythm Factors (3)**: `median_ici_ms`, `onset_rate_hz`, `ici_coefficient_of_variation` (temporal patterns)
- **Builder Pattern**: `SourceMetadata.builder()` for partial metadata construction with Rust defaults
- **Delta Calculations**: `metadata.delta_from(other)` for 30D delta vectors
- **Acoustic Persona Examples**:
  - **GRITTY (aggressive)**: Low HNR (2dB), high flatness (0.8), low harmonicity (0.4), fast attack (3ms), high jitter (0.15), high shimmer (0.12)
  - **PURE (contact)**: High HNR (25dB), low flatness (0.05), high harmonicity (0.95), slow attack (25ms), low jitter (0.01), low shimmer (0.02)
  - **Rhythmic**: High onset rate (20Hz), regular ICI (50ms), low CV (0.1)
  - **Harmonic**: Zero onset rate, zero ICI (continuous tone)

**Scientific Validation**:
- **Granular Synthesis**: t-SNE distance = 6.452 (< 7.0 target) ✅
  - 76.1% improvement over additive synthesis (distance 27.0)
  - Preserves formant structure while enabling systematic parameter variation
- **Vector Delta Integration**: 12/12 tests passing ✅
  - Validates delta-based synthesis produces valid results
  - Granular with delta shift gets 0.5% closer to ideal target than concatenative
- **30D Metadata (Synthesis API)**: 8/8 tests passing ✅
  - Full 30D metadata construction and access
  - Builder pattern for partial specification
  - GRITTY, PURE, rhythmic, and harmonic persona support
  - Backward compatibility with legacy 3D API
  - All 30D features: harmonicity, shimmer, mfcc_5-13, spectral_flux

**Concatenative vs Granular Comparison**:

| Feature | Concatenative | Granular (Delta) |
|---------|---------------|------------------|
| **Fidelity** | Perfect (t-SNE 4.208) | Near-perfect (t-SNE 6.452) |
| **Flexibility** | None (no parameter variation) | High (any pitch/duration) |
| **Use Case** | Natural playback, baseline validation | Acoustic algebra, systematic variation |
| **Integration** | Direct audio playback | Vector delta commands from Acoustic Algebra |

**When to Use Granular vs Concatenative:**
- **Concatenative**: Use when you have exact audio segments (perfect fidelity, low flexibility)
- **Granular**: Use when you need systematic parameter variation (near-perfect fidelity, high flexibility)
  - Pitch continuum testing (7500Hz, 7600Hz, 7700Hz... even if not in database)
  - Controlling confounds (same phrase, different pitches, constant duration)
  - Acoustic feature boundary testing (JND measurements)
  - Creating novel stimuli (hybrid calls)
  - **Vector delta synthesis** (acoustic algebra integration)

---

## Key Components

### 1. Acoustic-First Analysis Engine (`analysis/rosetta_stone/`)

**The Foundation** - Species-agnostic acoustic analysis for discovering phrase, sentence, and grammar structure.

**Core Classes:**
- `UniversalRosettaStone` - Main analysis engine
  - `detect_modality()` - Classifies: Harmonic, FM Sweep, Transient, Rhythmic
  - `segment_phrases()` - Segments audio into phrase units
  - `build_vocabulary()` - Clusters similar phrases into atomic units
  - `discover_grammar()` - Discovers grammatical transition rules
  - `discover_sentences()` - Groups phrases into sentences
  - `detect_superposition()` - Finds simultaneous phrase layers

- `PhraseSignature` - Acoustic phrase representation
  - Modality-specific feature extraction
  - Distance metrics for phrase similarity
  - Microharmonic similarity calculation

- `Sentence` - Individual vocalization containing phrases
  - `discover_atomic_units()` - Creates binned phrase keys
  - Validates phrase groupings

**Output:**
- Phrase keys: `F0_6400_DUR_50_RANGE_0` format
- Vocabulary: Grouped phrases by acoustic similarity
- Grammar: Transition matrices between phrase types
- Sentences: Phrase sequences with timing

**Usage:**
```python
from analysis.rosetta_stone import UniversalRosettaStone

analyzer = UniversalRosettaStone(sample_rate=48000)
phrases = analyzer.segment_phrases(audio_data)
vocabulary = analyzer.build_vocabulary(phrases)
grammar = analyzer.discover_grammar(phrases)
```

---

### 2. Rust Execution Layer (`technical_architecture/`)

**Core Modules:**
- **Synthesis** (`synthesis.rs`) - Granular, concatenative, superpositional synthesis
  - **30D Micro-Dynamics Metadata (Synthesis API)**: Full support for all micro-dynamics features
    - Fundamental (3): `mean_f0_hz`, `duration_ms`, `f0_range_hz`
    - Grit Factors (3): `harmonic_to_noise_ratio`, `spectral_flatness`, `harmonicity`
    - Motion Factors (7): `attack_time_ms`, `decay_time_ms`, `sustain_level`, `vibrato_rate_hz`, `vibrato_depth`, `jitter`, `shimmer`
    - Fingerprint Factors (14): `mfcc_1` through `mfcc_13`, `spectral_flux`
    - Rhythm Factors (3): `median_ici_ms`, `onset_rate_hz`, `ici_coefficient_of_variation`
  - **Builder Pattern**: `SourceMetadata.builder()` for partial metadata construction
  - **Vector Delta Commands**: Relative shifts for Acoustic Algebra integration
    - `load_source_with_metadata()` - Load audio with 30D feature tracking
    - `shift_pitch_by_hz()`, `shift_duration_by_ms()` - Individual delta commands
    - `apply_vector_delta()` - Legacy 3D vector delta (backward compatible)
    - `apply_micro_dynamics_delta()` - Complete 30D delta (new primary integration point)
    - `delta_from()` - Calculate 30D delta between two metadata sets
  - **Documentation**: See `analysis/rosetta_stone/VECTOR_DELTA_INTEGRATION.md`
- **Source Separation** (`source_separation.rs`) - Conv-TasNet via ONNX/Tract
- **Thermal Management** (`thermal.rs`) - Temperature monitoring and throttling
- **Safety Monitoring** (`safety.rs`) - Watchdog timers, safety limits
- **PTP Clock** (`ptp.rs`) - IEEE 1588 precision timing (nanosecond)
- **Provenance Logging** (`logging.rs`) - Deterministic audit trails

**Production Deployment Modules:**
- **IACUC Compliance** (`iacuc_compliance.rs`) - 29 tests
  - Legal animal research protocol enforcement
  - Time window, volume, species, and daily limit enforcement
  - Compliance audit trails and report generation

- **Time-Series Archive** (`time_series_archive.rs`) - 24 tests
  - High-frequency time-series data storage
  - Query by time range with downsampled aggregation
  - Retention policies and storage quotas

- **Auto-Calibration** (`auto_calibration.rs`) - 17 tests
  - Pink noise calibration tone generation
  - Loopback gain analysis for drift detection
  - Health status reporting with automatic scheduling

- **Shadow Model Monitoring** (`shadow_model_monitor.rs`) - 26 tests
  - Parallel inference: active model vs frozen baseline
  - Concept drift detection and alerting
  - Automatic model rollback capability

- **Remote Web Dashboard** (`web_dashboard.rs`) - 25 tests
  - HTTPS/WebSocket server for remote monitoring
  - JWT token authentication with expiration
  - Emergency stop, manual override, parameter adjustment
  - Real-time spectrogram and gauge streaming
  - Command audit logging

- **Multi-Node Coordination** (`multi_node_coordination.rs`) - 21 tests
  - PTP grandmaster election (IEEE 1588 clock class/accuracy)
  - TDMA scheduling for acoustic interference avoidance
  - Data fusion with location triangulation
  - Cluster management with automatic failover

**Field Deployment Modules:**
- **Environmental Monitor** (`environmental_monitor.rs`) - 46 tests
  - Rain intensity classification (None → Storm)
  - Temperature classification (Freezing → Extreme)
  - Light level classification (Dark → Night)
  - Solar forecasting integration
  - Forces Passthrough Mode in adverse conditions

- **Power Manager** (`power_manager.rs`) - 54 tests
  - Battery state tracking with health estimation
  - Power modes: Normal (>80%), Medium (50-80%), Low (20-50%), Critical (<20%)
  - Solar prediction for task deferral decisions
  - Atomic flags for FPGA, source separation, synthesis throttling
  - Power budget calculation with runtime estimation

- **Wildlife Sentry** (`wildlife_sentry.rs`) - 24 tests
  - FFT-based vocalization detection
  - Species signatures: marmoset, dolphin, bat, finch
  - Wake trigger generation with urgency levels
  - Debounce mechanism for rapid successive calls

- **Data Synchronizer** (`data_synchronizer.rs`) - 20 tests
  - Priority-based sync (Critical > High > Normal > Low)
  - Bandwidth throttling
  - Multi-storage backend (SSD, USB, SD Card)
  - Compression support

- **Acoustic Simulator** (`acoustic_simulator.rs`) - 43 tests
  - Environmental noise generation (rain, wind, insects, birds)
  - SNR mixing for testing
  - Environment simulation (jungle, rainforest, open field)

**Master Controller:**
- **UnifiedMasterController** - Intent-Reality mediator
  - Translates Python intents into physical Rust actions
  - Enforces thermal, safety, and hardware constraints
  - Watchdog monitoring with crash isolation

- **PeerController** - ZeroMQ heartbeat monitoring
  - Non-blocking heartbeat polling (0ms timeout)
  - Automatic mode switching (Passthrough ↔ Interactive)
  - 100ms timeout (5 missed heartbeats = disconnect)

**Build:**
```bash
cd technical_architecture
cargo build --release
cargo test  # 408 tests passing
```

---

### 2. Python Logic Layer

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

---

### 3. Self-Healing System (`system/`) [NEW]

**Autonomous Crash Recovery** - Enables long-duration field experiments with automatic recovery from Python process crashes.

**Components:**

**StatePersistor** (`state_persistor.py`) - Checkpoint system for saving system state
- `save_contextual_agent()` - Save conversation context and history
- `save_rust_cache()` - Save LRU cache keys for warm restart
- `save_complete_state()` - Unified checkpoint of entire system
- `load_checkpoint()` - Load state from JSON
- `get_latest_checkpoint()` - Find most recent checkpoint

**SelfHeal** (`self_heal.py`) - Autonomous crash recovery and rehydration
- `check_health()` - Detect dead Python processes
- `rehydrate_agent()` - Load agent state from checkpoint
- `rehydrate_from_latest()` - Auto-load most recent checkpoint
- `sync_rust_cache()` - Extract cache keys for warm restart
- `heal()` - Complete workflow: detect → load checkpoint → restart

**Usage:**
```python
from system import StatePersistor, SelfHeal, HealthStatus

# Create checkpoint before potential crash
persistor = StatePersistor(checkpoint_dir=Path("./checkpoints"))
agent_state = {
    "context": "FOOD",
    "history": ["PheeA", "PheeB"],
    "dialogue_state": {"turn": 3, "initiator": "human"}
}
persistor.save_contextual_agent(agent_state, Path("checkpoint.json"))

# After crash, detect and heal
healer = SelfHeal(checkpoint_dir=Path("./checkpoints"))
status = healer.check_health(pid_of_python_process)

if status == HealthStatus.DEAD:
    # Automatically restart with recovered state
    healer.heal(
        pid=pid_of_python_process,
        restart_command=["python3", "-m", "cognitive_agent"]
    )
```

**Benefits for Long-Duration Field Experiments:**
- **Autonomous Recovery**: No human intervention required for Python crashes
- **State Preservation**: Conversation history and context restored
- **Warm Cache Restart**: Rust LRU cache preloaded for minimal latency
- **Deterministic Recovery**: Test-Driven Development ensures reliability
- **16/16 Tests Passing**: Complete TDD coverage (Phase 1 + Phase 2)

**Integration with Systemd:**
- Python service `Restart=on-failure` handles automatic restarts
- Self-healing loads checkpoint before processing new intents
- Rust continues in Passthrough Mode during Python recovery
- Peer-to-peer supervision ensures safety during recovery

---

### 4. Bio-Acoustic Turing Test (`realtime/`)

**Live Animal Validation Framework** - Determines if animals can distinguish between natural and granular-synthesized vocalizations.

```python
from realtime.bio_acoustic_turing_test import BioAcousticTuringTest

# Create Turing test instance
turing_test = BioAcousticTuringTest(
    subject_id='marmoset_001',
    species='marmoset',
    output_dir='./results'
)

# Phase 1: Concatenative baseline (natural recordings)
turing_test.set_phase('concatenative_baseline')
turing_test.add_stimulus('natural_phee', audio_data, 'concatenative')
result = turing_test.run_trial('natural_phee')

# Phase 2: Granular synthesis (pitch-shifted variants)
turing_test.set_phase('granular_synthesis')
turing_test.add_stimulus('granular_phee', granular_audio, 'granular')
result = turing_test.run_trial('granular_phee')

# Phase 3: Statistical analysis
hypothesis = turing_test.evaluate_hypothesis()

if hypothesis['passed']:
    print("✅ TURING TEST PASSED - Animals cannot distinguish!")
```

**Components:**
- `StimulusController` - Manages audio playback with counterbalanced sequences
- `ResponseRecorder` - Records animal responses and measures latency
- `ExperimentDesign` - Handles randomization and inter-trial intervals
- `StatisticalAnalyzer` - Chi-square tests, t-tests, Turing test evaluation
- `BioAcousticTuringTest` - Main orchestrator

**Demo:**
```bash
python3 realtime/demo_bio_acoustic_turing_test.py
```

**Tests:**
```bash
python3 -m pytest realtime/test_bio_acoustic_turing_test.py -v
```

---

### 4. Query Interface (`query_interface/`)

- High-performance query system with pre-built indexes
- Real-time search: F0 range, duration, similarity
- Grammar network analysis and cross-species comparisons
- Main entry: `VocalizationQueryInterface` and `get_query_interface()`

---

### 4. Semiotic Analysis (`semiotics/`)

- Advanced cognitive intelligence capabilities
- Deception detection and innovation tracking
- Directed communication analysis
- Cross-modal attention fusion

---

## Deployment

### Systemd Services

Two services managed by systemd:

**1. Rust Field Engine** (`technical_architecture/deployment/rust-field-engine.service`)
- Safety-critical execution layer
- Binds ZeroMQ SUB socket for heartbeats
- Starts in Passthrough Mode (safe default)

**2. Python Cognitive Agent** (`technical_architecture/deployment/python-cognitive-agent.service`)
- Logic layer with cognitive intelligence
- Connects to Rust and sends heartbeats (20ms interval)
- Automatically restarted on crash (Let it crash philosophy)
- **[NEW]** Self-healing with state recovery on restart

### Self-Healing Integration [NEW]

For long-duration field experiments, the system includes autonomous crash recovery:

```python
# Python Cognitive Agent startup script with self-healing
from pathlib import Path
from system import StatePersistor, SelfHeal

# On startup, try to recover from previous crash
checkpoint_dir = Path("./state")
healer = SelfHeal(checkpoint_dir=checkpoint_dir)

# Check for latest checkpoint and recover state
latest_state = healer.rehydrate_from_latest()
if latest_state:
    # Restore conversation context and history
    context = latest_state.get("context")
    history = latest_state.get("history", [])
    print(f"Recovered from checkpoint: context={context}, history_length={len(history)}")

# Periodically save checkpoints during operation
persistor = StatePersistor(checkpoint_dir=checkpoint_dir)
def save_checkpoint_periodically():
    agent_state = {
        "context": current_context,
        "history": conversation_history,
        "dialogue_state": {"turn": turn_count, "initiator": last_initiator}
    }
    persistor.save_contextual_agent(agent_state, checkpoint_dir / "checkpoint.json")
```

**Self-Healing Benefits:**
- **State Persistence**: Conversation context and history saved periodically
- **Automatic Recovery**: On restart, loads latest checkpoint automatically
- **Warm Restart**: Rust cache synchronized to minimize cold-start latency
- **Transparent Recovery**: Animals experience minimal disruption
- **16/16 Tests Passing**: TDD-proven reliability

### Installation

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
```

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

---

## Quick Start

### Audio Discovery Workflow (New)

For discovering phrases from raw audio and creating synthesis-ready libraries:

```bash
# Step 1: Discover phrases and create .pkl library
python3 realtime/audio_aware_grammar_discovery.py recording.wav \
    --export-library phrase_library.pkl \
    --grain-duration 10.0

# Step 2: Extract audio segments (optional)
python3 realtime/phrase_library_segment_extractor.py phrase_library.pkl \
    --source-audio-dir /path/to/audio \
    --output-dir audio_segments

# Step 3: Synthesize (Rust or Python)
# Option A: Rust synthesis
cd technical_architecture && cargo build --release

# Option B: Python synthesis (development/testing)
python3 -c "
from realtime.phrase_audio_library import VocalizationSynthesizer, PhraseAudioLibrary
library = PhraseAudioLibrary(species='marmoset', sr=22050)
library.load('phrase_library.pkl')
synth = VocalizationSynthesizer(library)
audio, sr = synth.synthesize_horizontal(['F0_6400_DUR_10_RANGE_0'])
"

# Demo: Complete discovery to synthesis workflow
python3 realtime/demo_discovery_to_synthesis.py
```

**See**: [Audio Discovery and Synthesis Workflow](#audio-discovery-and-synthesis-workflow--new-) for details.

### STEP 1: Analyze Raw Audio (Acoustic-First)

**For new species or new audio data:**

```bash
# Run acoustic analysis on raw audio
python3 src/analysis/rosetta_stone/demo_unknown_species.py
```

This extracts:
- **Phrases**: Acoustic units with F0, duration, range features
- **Vocabulary**: Grouped similar phrases (atomic units)
- **Grammar**: Transition rules between phrase types
- **Sentences**: Phrase sequences with timing

### STEP 2: Import Database

```bash
# Import vocalization data (populates query interface)
python3 src/data_import/import_vocalization_data.py
```

Creates `vocalization_database.json` with 2,882 phrases.

### STEP 3: Run Demos

```bash
# Query interface demo
python3 src/query_interface/demo_query_interface.py

# Semiotic engine demo
python3 src/semiotics/demo_semiotic_engine.py
```

### Run Tests

```bash
# Python tests
python3 -m pytest tests/ -v

# Rust-Python integration tests (38 tests)
python3 -m pytest tests/test_rust_python_integration.py -v

# Rust tests (in technical_architecture/)
cd technical_architecture && cargo test

# Build and install PyO3 bindings
cd technical_architecture
maturin build --release --features python-bindings
pip3 install --force-reinstall --break-system-packages target/wheels/technical_architecture-*.whl
```

**Integration Tests (38 tests)**:
- Tests the Rust-Python boundary with PyO3 bindings
- Verifies safety-critical components (PeerController, EnvironmentalMonitor, ThermalState)
- Tests memory safety, error handling, and data serialization
- Ensures "Fail-Open to Safety" behavior works correctly

### Python API Usage

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

### Rust API Usage

```rust
use technical_architecture::{
    TechnicalArchitect, PeerController,
    OperationMode, PeerControllerConfig,
    EnvironmentalMonitor, EnvironmentalMonitorConfig,
    ThermalState, RainIntensity, TemperatureClassification,
    IacucComplianceEngine, MultiNodeCoordinator,
    WebDashboard, TimeSeriesArchiver,
};

// === Safety-Critical Components ===

// Create peer controller for heartbeat monitoring
let config = PeerControllerConfig::default();
let mut controller = PeerController::new(config)?;

// Main loop
loop {
    let mode = controller.tick()?;

    match mode {
        OperationMode::Passthrough => {
            // Python crashed or heartbeat stopped - audio muted
            // Safe mode: recording only
        }
        OperationMode::Interactive => {
            // Python alive and sending heartbeats
            // Active mode: process Python intents
        }
    }
}

// Thermal state checking
let thermal_state = ThermalState::Critical;
if thermal_state.requires_throttling() {
    // Block synthesis to prevent overheating
}
if thermal_state.is_critical() {
    // Emergency measures required
}

// Environmental monitoring for field deployment
let config = EnvironmentalMonitorConfig::default();
let mut monitor = EnvironmentalMonitor::new(config)?;

// Poll sensors
let conditions = monitor.poll_sensors()?;

// Check if conditions force safe mode
let viability = monitor.assess_session_viability();
if monitor.forces_passthrough() {
    // Heavy rain, extreme temp, etc. - switch to safe mode
}

// Check rain intensity
let rain = RainIntensity::from_mm_h(60.0); // Storm
if rain.forces_passthrough() {
    // Storm detected - mute audio
}

// Check temperature
let temp = TemperatureClassification::from_celsius(-5.0); // Freezing
if temp.forces_passthrough() {
    // Freezing - switch to safe mode
}

// === Other Components ===

// Create technical architect
let config = TechArchConfig::default();
let architect = TechnicalArchitect::new(config).await?;

// Create IACUC compliance engine
let iacuc = IacucComplianceEngine::new(protocol)?;
let check = iacuc.check_compliance(&intent)?;

// Create multi-node coordinator
let config = ClusterConfig::default();
let coordinator = MultiNodeCoordinator::new("node1".to_string(), config);
coordinator.elect_grandmaster(my_info).await?;

// Create web dashboard
let dashboard = WebDashboard::new(config)?;
dashboard.connect_client("client1", "127.0.0.1", &token)?;
```

---

## Database Status

- **Total Phrases**: 2,882 from 4 species
- **Marmoset**: 1,351 phrases (harmonic communication)
- **Egyptian Fruit Bat**: 516 phrases (FM sweep communication)
- **Dolphin**: 387 phrases (whistle communication)
- **Chimpanzee**: 628 phrases (mixed communication)

---

## Key Features

### Cross-Species Analysis
- Universal Rosetta Stone methodology
- Species-specific acoustic analysis strategies
- Comparative semiotic patterns

### Cognitive Intelligence
- Deceptive communication detection
- Emergent behavior identification
- Directed communication analysis
- Cross-modal attention fusion

### High Performance
- Zero-copy Rust operations for audio processing
- Optimized data structures and indexing
- Real-time query capabilities
- Deterministic timing with PTP

### Safety & Reliability
- Peer-to-peer supervision with systemd
- Automatic crash recovery (Let it crash)
- Fail-open to safety design
- Thermal throttling and emergency mute

### Production Deployment Capabilities
- **IACUC Compliance**: Legal animal research protocol enforcement with audit trails
- **Time-Series Archiving**: High-frequency data storage with retention policies
- **Auto-Calibration**: Self-health checks with pink noise calibration and drift detection
- **Shadow Model Monitoring**: Concept drift detection with automatic model rollback
- **Remote Dashboard**: HTTPS/WebSocket monitoring with emergency stop capabilities
- **Multi-Node Coordination**: PTP grandmaster election and TDMA scheduling for arrays

### Field Deployment Capabilities
- **Environmental Monitoring**: Automatic session management based on rain, temperature, light
- **Power Management**: Solar-aware battery optimization with adaptive throttling
- **Wildlife Detection**: Low-power background sentry for target species vocalizations
- **Offline Resilience**: Black box data queuing with priority-based synchronization
- **TDD Infrastructure**: Acoustic simulation for comprehensive testing

---

## Scientific Impact

This framework transforms animal communication research by:
1. Moving beyond simple classification to cognitive understanding
2. Enabling deception detection in animal communication
3. Tracking emergent cultural behaviors
4. Recognizing intentional, targeted communication
5. Providing comparative analysis across species

---

## Test Coverage

The framework has comprehensive test coverage ensuring reliability and correctness:

```
Rust Execution Layer: 464 tests passing
├── Core Modules: 175 tests
│   ├── Peer Controller: 79 tests
│   ├── Master Controller: 17 tests
│   └── Other modules: 79 tests
│
├── Production Deployment: 142 tests
│   ├── IACUC Compliance Engine: 29 tests
│   ├── Time-Series Archive: 24 tests
│   ├── Auto-Calibration: 17 tests
│   ├── Shadow Model Monitoring: 26 tests
│   ├── Remote Web Dashboard: 25 tests
│   └── Multi-Node Coordination: 21 tests
│
├── Field Deployment: 187 tests
│   ├── Environmental Monitor: 46 tests
│   ├── Power Manager: 54 tests
│   ├── Wildlife Sentry: 24 tests
│   ├── Data Synchronizer: 20 tests
│   └── Acoustic Simulator: 43 tests
│
└── Multi-Modal Synthesis: 10 tests (NEW)
    ├── Modality Timeline: 5 tests
    ├── Multi-Buffer Sequencer: 5 tests

Python Logic Layer: 120+ core TDD tests
├── Multi-Modal Corvid Support: 28 tests (NEW)
│   ├── Data Models (Phase 1): 7 tests
│   ├── Synthesis Timeline (Phase 2): 9 tests
│   ├── Acoustic Algebra (Phase 3): 5 tests
│   ├── Semiotic Detection (Phase 4): 5 tests
│   └── Integration Tests: 2 tests
│
├── Formant Barrier Validation: 12 tests (NEW)
│   ├── Harmonic → Transient: IMPOSSIBLE
│   ├── Transient → Harmonic: IMPOSSIBLE
│   └── Persona Switching: Solution
│
├── The Hybrid Bridge: 18 tests (NEW)
│   ├── Data Models (Phase 1): 3 tests
│   ├── Acoustic Algebra Engine (Phase 2): 4 tests
│   ├── Delta Mapper (Phase 3): 3 tests
│   ├── Hybrid Synthesis Engine (Phase 4): 4 tests
│   └── Safety and Edge Cases (Phase 5): 4 tests
│
├── Cognitive Synthesis Engine: 17 tests (NEW)
│   ├── Intent Decomposition: 6 tests
│   ├── Multi-Vector Modulation: 3 tests
│   ├── Cognitive Orchestration: 4 tests
│   └── Safety and Edge Cases: 4 tests
│
├── Island Hopping Navigation: 45 tests (NEW)
│   ├── Waypoint Navigation: 24 tests
│   │   ├── Vector30D Calculations: 4 tests
│   │   ├── Acoustic Algebra Engine: 4 tests
│   │   ├── Phrase Database: 3 tests
│   │   ├── Island Hopping Navigator: 5 tests
│   │   ├── Safety and Edge Cases: 4 tests
│   │   └── Integration: 4 tests
│   └── Interpolation vs Extrapolation: 21 tests
│       ├── Interpolation (Bridge Builder): 6 tests
│       ├── Extrapolation (Ocean Explorer): 6 tests
│       ├── Delta Clamping (The Leash): 4 tests
│       └── Integration: 5 tests
│
└── Rust-Python Integration: 38 tests
    ├── VisualRecorder: 9 tests
    ├── Synthesis: 6 tests
    ├── Memory Safety: 2 tests
    ├── Error Handling: 2 tests
    ├── Safety Boundary Real: 4 tests
    │   ├── Heartbeat timeout detection
    │   ├── PeerController resilience
    │   ├── Configuration validation
    │   └── Thermal constraint override
    ├── PeerController Safety: 7 tests
    └── EnvironmentalMonitor: 8 tests
```

---

## Multi-Modal Corvid Support

Corvids (crows, ravens) break the "Single Persona" assumption by utilizing **Multiple Modalities** (Transient + Harmonic + FM Sweep) in single vocalizations. This requires a fundamental architecture shift from "Voice Banking" to "Texture Sequencing".

### The Formant Barrier

Granular synthesis is a **"Warping" technology, not a "Creation" technology**. It preserves the **Spectral Envelope (Formants)** of the source audio.

**Key Constraint**: You cannot transmute modality because the formant structure is "baked in" to the source audio buffer.

| Operation | Harmonic Source | Transient Source |
|-----------|----------------|------------------|
| Pitch Shift | ✅ Remains Harmonic | ✅ Remains Transient |
| Time Stretch | ✅ Remains Harmonic | ✅ Remains Transient |
| Modality Change | ❌ IMPOSSIBLE | ❌ IMPOSSIBLE |

**Exception**: FM Sweep → Harmonic via "freezing" (slowing FM rate to near-zero)

**Solution**: **Persona Switching** (Source Selection) - Use different source buffers for different modalities.

### Architecture: Voice Banking vs Texture Sequencing

**Marmoset (Single Persona)**:
```
┌─────────────────────────────────────┐
│  Marmoset Phee Buffer              │
│  [HARMONIC audio data]              │
└─────────────────────────────────────┘
              ↓
        [Granular Synthesis]
              ↓
        Single Output Voice
```

**Corvid (Composite Persona)**:
```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  Whistle    │  │  Rattle     │  │  Whistle    │
│  [HARMONIC] │  │  [TRANSIENT]│  │  [HARMONIC] │
└─────────────┘  └─────────────┘  └─────────────┘
       ↓                ↓                ↓
    ┌─────────────────────────────────────────┐
    │         Multi-Buffer Sequencer          │
    │         [Timeline: H → T → H]           │
    └─────────────────────────────────────────┘
                      ↓
            Multi-Modal Output Voice
```

### Data Model Changes

**Old (Marmoset)**:
```python
Persona = "Marmoset Phee"
SourceBuffer = "marmoset_phee.wav"
Modality = HARMONIC (single)
```

**New (Corvid)**:
```python
from enum import Enum

class Modality(Enum):
    HARMONIC = "HARMONIC"    # Tonal, sine-like (whistle, phee)
    TRANSIENT = "TRANSIENT"  # Clicky, noise-like (rattle, click)
    FM_SWEEP = "FM_SWEEP"    # Frequency modulated (trill, sweep)

@dataclass
class CorvidPersona:
    species: str
    id: str
    modality_sequence: List[Modality]  # NEW: Sequence of modalities

@dataclass
class TimelineEvent:
    start_ms: float
    duration_ms: float
    source_buffer: str  # e.g., "corvid_whistle.wav"
    modality: Modality

@dataclass
class ModalityTimeline:
    events: List[TimelineEvent]

    def add_event(self, start_ms, duration_ms, source, modality)
    def sort_by_time(self)
    def validate(self)  # Check for overlaps
```

### Synthesis API (Rust)

```rust
use technical_architecture::{
    Modality, TimelineEvent, ModalityTimeline,
    MultiBufferGranularSequencer, SourceMetadata,
};

// Create sequencer
let mut sequencer = MultiBufferGranularSequencer::new(44100);

// Register multiple source buffers
sequencer.register_source(
    "whistle".to_string(),
    whistle_audio,
    SourceMetadata {
        mean_f0_hz: 7000.0,
        harmonic_to_noise_ratio: 25.0,
        spectral_flatness: 0.05,
        ..Default::default()
    }
);

sequencer.register_source(
    "rattle".to_string(),
    rattle_audio,
    SourceMetadata {
        mean_f0_hz: 0.0,
        harmonic_to_noise_ratio: 2.0,
        spectral_flatness: 0.8,
        ..Default::default()
    }
);

// Create multi-modal timeline
let mut timeline = ModalityTimeline::new();
timeline.add_event(0.0, 100.0, "whistle".to_string(), Modality::Harmonic);
timeline.add_event(100.0, 50.0, "rattle".to_string(), Modality::Transient);
timeline.add_event(150.0, 20.0, "whistle".to_string(), Modality::Harmonic);

// Synthesize
let audio = sequencer.synthesize_timeline(&timeline)?;
```

### Acoustic Algebra Constraints

The **Modality Gate** prevents cross-modality interpolation:

```python
@dataclass
class AcousticVector:
    # ... all 30D features ...
    modality: Modality  # NEW: Modality constraint

def interpolate_vectors(v1: AcousticVector, v2: AcousticVector, alpha: float):
    # MODALITY GATE: Prevent cross-modality interpolation
    if v1.modality != v2.modality:
        raise ValueError(
            f"Cannot interpolate across modalities: {v1.modality} != {v2.modality}. "
            "Cross-modality interpolation creates garbage audio artifacts. "
            "Use persona switching (different source buffers) instead."
        )
    # ... safe interpolation within same modality ...
```

**Why?** Interpolating Harmonic (clean sine) + Transient (white noise) = Sine with 50% static noise (garbage).

### Deception Detection via Modality Mismatches

The semiotic engine can detect deception by identifying contextual modality mismatches:

```python
@dataclass
class ContextualState:
    predator_present: bool
    conspecific_present: bool
    food_present: bool
    territory_violation: bool

@dataclass
class SemioticAnalysis:
    audio_modality: Modality
    context: ContextualState
    expected_modality: Optional[Modality]
    modality_mismatch: bool
    deception_probability: float

# Example: False Alarm Detection
# Audio: "Seep" (Harmonic, Soft)
# Context: Predator is present
# Semiotics: Should use "Rattle" (Transient) for alarm
# Deception: High probability (0.85)
```

**Modality Rules**:
- **Predator context**: Expected `TRANSIENT` (alarm)
- **Mating context**: Expected `HARMONIC` (courtship)
- **Food context**: Expected `TRANSIENT` (excitement)
- **Territory context**: Expected `TRANSIENT` (aggression)

### Test Coverage

- **Python Tests**: 120 tests (Core TDD Suite)
  - `test_corvid_multi_modal_support.py`: 28 tests (4 phases + integration)
  - `test_granular_synthesis_limitations.py`: 12 tests (Formant Barrier validation)
  - `test_hybrid_bridge.py`: 18 tests (Algebra + Granular integration)
  - `test_cognitive_synthesis_engine.py`: 17 tests (Cognitive multi-modal engine)
  - `test_island_hopping_navigation.py`: 24 tests (Waypoint navigation)
  - `test_interpolation_extrapolation.py`: 21 tests (Interpolation/Extrapolation)
- **Rust Tests**: 10 tests
  - Modality Timeline: 5 tests
  - Multi-Buffer Sequencer: 5 tests

---

**Integration Test Domains (38 tests)**:
- **PyO3 Binding Correctness**: Data serialization, memory management, type conversion
- **VisualRecorder**: Lifecycle, configuration, statistics, thread safety
- **Synthesis**: Dynamic microharmonic, granular concatenative, audio range validation
- **Memory Safety**: Resource cleanup, large buffer handling
- **Error Handling**: Invalid state transitions, double operations
- **Safety Boundary**:
  - Heartbeat timeout triggers Passthrough mode
  - PeerController configuration validation
  - Thermal constraint overrides Python intent
  - PeerController resilience to missed heartbeats
- **PeerController Safety**:
  - OperationMode enum (Passthrough/Interactive)
  - Creation with default/custom configs
  - Default mode is Passthrough (safe default)
  - Configuration retrieval
- **EnvironmentalMonitor**:
  - Monitor creation and configuration
  - Rain intensity classification (None/Light/Moderate/Heavy/Storm)
  - Temperature classification (Freezing/Cold/Mild/Hot/Extreme)
  - Session viability assessment (Viable/Marginal/Infeasible)
  - Environmental forces Passthrough (heavy rain, extreme temp)

**Test Domains Covered:**
- **Production Compliance**: IACUC protocol enforcement, calibration scheduling, shadow model monitoring
- **Remote Operations**: Web dashboard with authentication, command audit logging
- **Multi-Node Coordination**: PTP grandmaster election, TDMA scheduling, data fusion
- Environmental condition classification and override logic
- Battery state management and solar prediction
- Wildlife detection with FFT analysis
- Offline queuing with priority handling
- Peer-to-peer heartbeat monitoring
- Intent-Reality mediation
- Thermal throttling and safety limits
- Source separation and synthesis
- **Rust-Python Boundary**: PyO3 bindings, memory safety, error handling across FFI

---

## The Hybrid Bridge: Acoustic Algebra + Granular Synthesis

The **Cognitive Synthesis Engine** combines Acoustic Algebra (The Map) with Granular Synthesis (The Vehicle) to achieve the **Gold Standard** approach for bio-acoustic synthesis.

### Why Combine Them?

| Technique | Limitation (Alone) | Solution (Together) |
| :--- | :--- | :--- |
| **Acoustic Algebra** | Can calculate *any* gradient, but has no "Sound" to play. It creates "Ghost Phrases." | Uses Granular **Real Source** to ground the math in reality. |
| **Granular Synthesis** | Can *warp* existing sound, but limited to **recorded** variations. | Uses Algebra **Virtual Target** to guide Granular warp *beyond* training data. |

**The Result**: You get **Infinite Nuance** (Algebra) with **High Fidelity** (Granular).

### The Hybrid Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: Algebra (The Planner)                                 │
│  Intent: "Generate a 50% Aggressive call"                      │
│  Math: Vector_Target = Vector_Neutral + (Vector_Aggressive -   │
│         Vector_Neutral) * 0.5                                   │
│  Result: 30D Virtual Target (a "Ghost Phrase")                  │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  Step 2: Database Lookup (The Anchor)                           │
│  Action: db.find_nearest_real_phrase(Vector_Target)            │
│  Result: Real Recording "40% Aggressive" (Closest match)        │
│  Gap: Need "10% more Aggression"                                 │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  Step 3: Delta Calculator (The Interpreter)                     │
│  Action: Calculate Delta = Virtual_Target - Real_Source        │
│  Result: 30D "Warp Instructions" (+10% F0, -5dB HNR, etc.)     │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  Step 4: Delta Mapper (30D → Granular)                          │
│  Map: delta_mean_f0_hz → pitch_shift_ratio                      │
│        delta_hnr → roughness_amount                              │
│        delta_attack_time_ms → grain_size_ms                      │
│  Safety: Delta Clamping (max 20% warp)                          │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  Step 5: Granular Engine (The Mouth)                            │
│  Action: Load Real Audio + Apply Warp Parameters                │
│  Result: Warped Real Audio (matches Virtual Target)             │
└─────────────────────────────────────────────────────────────────┘
```

### Python API

```python
from tests.test_hybrid_bridge import (
    Intent, HybridSynthesisEngine, RealPhrase, VirtualTarget
)

# Create Hybrid Synthesis Engine
engine = HybridSynthesisEngine(max_warp_ratio=0.2)

# Register real phrases from database
neutral_phrase = RealPhrase(
    phrase_id="neutral_001",
    vector=VirtualTarget(
        mean_f0_hz=7000.0, duration_ms=50.0, f0_range_hz=400.0,
        harmonic_to_noise_ratio=20.0, spectral_flatness=0.2,
        attack_time_ms=15.0, decay_time_ms=20.0, sustain_level=0.6,
        vibrato_rate_hz=6.0, vibrato_depth=0.02, jitter=0.03, shimmer=0.02,
        mfcc_1=1.0, mfcc_2=0.7, mfcc_3=-0.2, mfcc_4=0.3, spectral_contrast=15.0,
        median_ici_ms=0.0, onset_rate_hz=0.0
    ),
    audio_buffer=neutral_audio
)

aggressive_phrase = RealPhrase(
    phrase_id="aggressive_001",
    vector=VirtualTarget(
        mean_f0_hz=8000.0, duration_ms=40.0, f0_range_hz=600.0,
        harmonic_to_noise_ratio=5.0, spectral_flatness=0.7,
        attack_time_ms=3.0, decay_time_ms=10.0, sustain_level=0.4,
        vibrato_rate_hz=0.0, vibrato_depth=0.0, jitter=0.12, shimmer=0.08,
        mfcc_1=1.8, mfcc_2=1.2, mfcc_3=0.3, mfcc_4=0.1, spectral_contrast=5.0,
        median_ici_ms=30.0, onset_rate_hz=15.0
    ),
    audio_buffer=aggressive_audio
)

engine.register_phrase(neutral_phrase)
engine.register_phrase(aggressive_phrase)

# Generate 50% aggression call
audio, warp_params = engine.synthesize(Intent.AGGRESSION, intensity=0.5)

# audio: Warped real audio samples
# warp_params: GranularWarpParameters(
#     pitch_shift_ratio=1.05,
#     time_stretch_ratio=1.02,
#     roughness_amount=0.3,
#     grain_size_ms=35.0,
#     vibrato_amount=0.15,
#     is_clamped=False
# )
```

### Delta Clamping (Safety Rule)

There is one danger: **Over-Warping**. When the delta between Virtual Target and Real Source is too large, the audio will sound **Robotic/Artifacted**.

**The Fix**: Implement **"Delta Clamping"** in the Bridge.

```python
# Safety Check
max_warp_ratio = 0.2  # Never warp more than 20%

if delta.magnitude > max_warp_ratio:
    print("Warning: Delta too high. Clamping to safe warp.")
    # Scale down all warp parameters proportionally
    delta = delta * (max_warp_ratio / delta.magnitude)
```

**Result**: You play the "100% Aggressive" real recording (even though you wanted 200%). It's close enough to be valid, and scientifically rigorous.

### Test Coverage

- **Python Tests**: 18 tests
  - `test_hybrid_bridge.py`: Complete TDD suite
  - Phase 1: Data Models (3 tests)
  - Phase 2: Acoustic Algebra Engine (4 tests)
  - Phase 3: Delta Mapper (3 tests)
  - Phase 4: Hybrid Synthesis Engine (4 tests)
  - Phase 5: Safety and Edge Cases (4 tests)

### Key Principles

1. **Infinite Nuance**: Algebra can calculate ANY intensity (0%, 30%, 60%, 90%, 100%)
2. **High Fidelity**: Granular synthesis preserves the natural quality of real recordings
3. **Safety First**: Delta clamping prevents over-warping artifacts
4. **Scientific Rigor**: Always grounded in real data (nearest neighbor lookup)

**Recommendation**: Integrate them immediately. Acoustic Algebra is **useless** without Granular's fidelity, and Granular is **limited** without Algebra's gradients. Together, they represent the state-of-the-art for bio-acoustic synthesis.

---

## Cognitive Synthesis Engine: Corvid Multi-Modal Communication

The **Cognitive Synthesis Engine** extends the Hybrid Bridge approach to handle Corvid-style "Sentences" - multi-modal vocalizations that combine different modalities (Harmonic, Transient, FM Sweep) in a single communicative event.

### The Cognitive Approach: Intent → Multi-Modal Strategy

Corvids don't just say "words" - they speak in **sentences**. A single Alarm call isn't just one sound, but a sequence: **Whistle (Attention) → Rattle (Urgency) → Whistle (Resolve)**.

**Key Difference from Single-Modality Species:**

| Aspect | Marmoset (Single Persona) | Corvid (Cognitive Engine) |
|--------|---------------------------|---------------------------|
| **Vocalization** | Single modality (Harmonic "Phee") | Multi-modal sequence |
| **Intent Mapping** | 1 Intent → 1 Phrase | 1 Intent → Multi-modal Strategy |
| **Parameters** | Single delta vector | Different deltas per modality |
| **Transitions** | None (single sound) | Cross-fades between modalities |
| **Example** | Alarm = Phee at 8kHz | Alarm = Whistle + Rattle + Whistle |

### Architecture: Intent Decomposition

```
┌─────────────────────────────────────────────────────────────────┐
│  Input: Intent + Intensity                                       │
│  Example: Intent.ALARM, Intensity=0.7                           │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  Intent Decomposer (Cognitive Logic)                            │
│  Function: decompose_intent(Intent, Intensity) → Strategy       │
│                                                                 │
│  Alarm @ 0.7 intensity =                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Event 1: Whistle (0-100ms)                              │   │
│  │   - Modality: HARMONIC                                  │   │
│  │   - Pitch: +14% (higher = more urgent)                  │   │
│  │   - Roughness: 0% (clean tone)                          │   │
│  │   - Fade-out: 10ms (smooth transition)                  │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │ Event 2: Rattle (90-140ms) [10ms overlap]               │   │
│  │   - Modality: TRANSIENT                                 │   │
│  │   - Pitch: 0% (neutral)                                 │   │
│  │   - Roughness: 35% (texture intensity)                  │   │
│  │   - Grain size: 5ms (small grains = clicky)             │   │
│  │   - Fade-in/out: 10ms each                              │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │ Event 3: Whistle (130-170ms) [10ms overlap]             │   │
│  │   - Modality: HARMONIC                                  │   │
│  │   - Pitch: +14% (consistent urgency)                    │   │
│  │   - Fade-in: 10ms                                       │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  Multi-Vector Modulation                                         │
│  Each modality gets DIFFERENT delta parameters:                 │
│                                                                 │
│  Whistle (Harmonic):                                             │
│    - pitch_shift_ratio = 1.0 + (0.2 * intensity)                │
│    - roughness_amount = 0.0                                     │
│    - grain_size_ms = 20.0                                       │
│                                                                 │
│  Rattle (Transient):                                             │
│    - pitch_shift_ratio = 1.0                                    │
│    - roughness_amount = 0.5 * intensity                         │
│    - grain_size_ms = 5.0  (smaller = more textured)            │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  Granular Cross-Fading                                           │
│  Events overlap with smooth transitions:                       │
│                                                                 │
│  [Whistle]─────────────┐                                        │
│                [10ms overlap]                                   │
│                  ┌─────────────[Rattle]─────────────┐           │
│                                        [10ms overlap]            │
│                                          ┌──────────────[Whistle]│
│                                                                 │
│  Cross-fade = Linear amplitude ramp during overlap              │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  Output: Multi-Modal Audio Sequence                             │
│  Duration: 170ms (3 events with overlaps)                       │
│  Result: Natural "Alarm" call with urgency gradient             │
└─────────────────────────────────────────────────────────────────┘
```

### Intent Decomposition Strategies

Each intent has a predefined multi-modal strategy:

#### **ALARM** (Attention + Urgency + Resolve)
```
Sequence: Whistle → Rattle → Whistle
Logic: Get attention → Convey urgency → Provide resolution
Modality: HARMONIC → TRANSIENT → HARMONIC
Duration: 100ms + 50ms + 40ms = 170ms (with overlaps)
```

#### **AGGRESSION** (Threat + Dominance)
```
Sequence: Rattle → Whistle
Logic: Display threat → Establish dominance
Modality: TRANSIENT → HARMONIC
Duration: 80ms + 120ms = 200ms (with overlap)
```

#### **COURTSHIP** (Display + Complexity)
```
Sequence: Whistle → FM Sweep
Logic: Attract attention → Demonstrate fitness
Modality: HARMONIC → FM_SWEEP
Duration: 150ms + 100ms = 250ms (with overlap)
```

#### **FOOD DISCOVERY** (Excitement)
```
Sequence: Rattle
Logic: Pure excitement call
Modality: TRANSIENT
Duration: 60ms
```

#### **TERRITORY** (Warning + Boundary)
```
Sequence: Rattle → Whistle
Logic: Threat warning → Boundary marker
Modality: TRANSIENT → HARMONIC
Duration: 70ms + 80ms = 150ms (with overlap)
```

### Python API

```python
from tests.test_cognitive_synthesis_engine import (
    Intent, Modality, CognitiveSynthesisEngine, ModalityDelta
)

# Create cognitive engine
engine = CognitiveSynthesisEngine()

# Register source buffers for different modalities
engine.register_source("whistle", whistle_audio)    # Harmonic source
engine.register_source("rattle", rattle_audio)      # Transient source
engine.register_source("sweep", sweep_audio)        # FM Sweep source

# Generate multi-modal alarm call at 70% intensity
audio, strategy = engine.synthesize_intent(Intent.ALARM, intensity=0.7)

# strategy contains:
# - Intent: ALARM
# - Intensity: 0.7
# - Sequence: 3 TimelineEvents with specific deltas and cross-fade parameters

# Event breakdown:
event1 = strategy.sequence[0]  # Whistle (0-100ms)
assert event1.modality == Modality.HARMONIC
assert event1.delta.pitch_shift_ratio == 1.14  # +14%
assert event1.delta.roughness_amount == 0.0
assert event1.fade_out_ms == 10.0

event2 = strategy.sequence[1]  # Rattle (90-140ms)
assert event2.modality == Modality.TRANSIENT
assert event2.delta.pitch_shift_ratio == 1.0
assert event2.delta.roughness_amount == 0.35  # 35%
assert event2.delta.grain_size_ms == 5.0
assert event2.fade_in_ms == 10.0
assert event2.fade_out_ms == 10.0

event3 = strategy.sequence[2]  # Whistle (130-170ms)
assert event3.modality == Modality.HARMONIC
assert event3.delta.pitch_shift_ratio == 1.14
assert event3.fade_in_ms == 10.0
```

### Data Models

```python
@dataclass
class ModalityDelta:
    """Per-modality warping parameters"""
    pitch_shift_ratio: float = 1.0     # Frequency multiplier
    roughness_amount: float = 0.0      # Noise/texture (0-1)
    time_stretch_ratio: float = 1.0    # Duration multiplier
    grain_size_ms: float = 20.0        # Granular grain size
    vibrato_amount: float = 0.0        # Vibrato depth

@dataclass
class TimelineEvent:
    """Single event in multi-modal sequence"""
    start_ms: float                    # Start time
    duration_ms: float                 # Duration
    source_buffer: str                 # Buffer name
    modality: Modality                 # HARMONIC/TRANSIENT/FM_SWEEP
    delta: ModalityDelta               # Warping parameters
    fade_in_ms: float = 0.0            # Cross-fade in
    fade_out_ms: float = 0.0           # Cross-fade out

@dataclass
class ModalityStrategy:
    """Complete multi-modal strategy for an intent"""
    intent: Intent
    intensity: float
    sequence: List[TimelineEvent]      # Ordered events
```

### Key Design Principles

1. **Intent Decomposition**: Single intent → Multi-modal strategy (corvid "sentences")
2. **Multi-Vector Modulation**: Different warping parameters per modality
3. **Granular Cross-Fading**: Smooth 10ms overlaps between events
4. **Timeline Overlap**: Events intentionally overlap for natural transitions
5. **Intensity Scaling**: All parameters scale with intent intensity (0.0 to 1.0)

### Test Coverage

- **Python Tests**: 17 tests
  - `test_cognitive_synthesis_engine.py`: Complete TDD suite
  - Intent Decomposition: 6 tests
  - Multi-Vector Modulation: 3 tests
  - Cognitive Orchestration: 4 tests
  - Safety and Edge Cases: 4 tests

### Why This Matters

The Cognitive Synthesis Engine represents the **state-of-the-art** for bio-acoustic communication:

1. **Biological Fidelity**: Mirrors how corvids actually communicate (multi-modal sequences)
2. **Intent-Driven**: High-level intents automatically decompose into appropriate acoustic strategies
3. **Contextual Awareness**: Different intensities produce appropriately graded responses
4. **Natural Transitions**: Cross-fading eliminates artificial "cut" sounds between modalities
5. **Scientific Rigor**: All strategies grounded in real corvid communication research

**Total TDD Test Coverage**: 120 core tests (Multi-Modal: 28, Formant Barrier: 12, Hybrid Bridge: 18, Cognitive Engine: 17, Island Hopping: 24, Interpolation/Extrapolation: 21) + 497 Rust tests (464 existing + 33 new Island Hopping) = **617 total tests passing**

---

## Island Hopping: Interpolation vs Extrapolation

⚡ **Implementation Note**: Island Hopping is implemented in **Rust** (`technical_architecture/src/island_hopping.rs`) for high-performance, deterministic execution. Python fallback code in test files is **deprecated** - see `archive/deprecated_python_fallbacks/` for migration guide.

The **"Island Hopping"** strategy uses two mathematical engines with different risk profiles: **Interpolation** (Bridge Builder) and **Extrapolation** (Ocean Explorer).

### Interpolation: The "Bridge Builder" (SAFE)

**Definition:** Calculating a waypoint **between** two known "Islands" (Real Audio Phrases).

**Context:** Moving from `0% Aggression` to `50% Aggression`.

**Math:** Linear (or Spherical) interpolation in 30D space.

```
Island A (Start) → Waypoint → Island B (End)
Neutral.wav    →  50% Agg  →  Aggressive.wav
```

**Safety Profile:** ✅ **High**
- Waypoint is supported by two real anchors
- High fidelity, "Smooth Sailing"
- Clamp never triggers (distances are small)

### Extrapolation: The "Ocean Explorer" (RISKY)

**Definition:** Calculating a waypoint **beyond** the furthest known "Island."

**Context:** Generating `150% Aggression` when database only goes to `100%`.

**Math:** Vector Extension (`Vector_Last * 1.5`).

```
Island A (Start) → Waypoint → NULL (Open Ocean)
Full_Agg.wav    →  150%     →  (No recording exists!)
```

**Safety Profile:** ⚠️ **Critical Risk (Uncanny Valley)**
- Waypoint has no support (only one anchor)
- High probability of "Robotic Artifacts"
- **Delta Clamping** is REQUIRED

### Delta Clamping: "The Leash" (Safety Valve)

Prevents Extrapolation from running wild in the open ocean.

```python
def navigate_waypoint(target, anchor, max_safe_warp=0.2):
    distance = target.distance_to(anchor)

    if distance > max_safe_warp:
        # CLAMP: Move target closer to reality
        direction = target - anchor
        safe_target = anchor + (direction.normalized() * max_safe_warp)
        return safe_target  # Capped at 20%
    else:
        return target  # Full nuance allowed
```

**Result:**
- **Interpolation**: Distance small (0.1), clamp never triggers → Full Nuance
- **Extrapolation**: Distance huge (0.8), clamp triggers → Capped Nuance (20% instead of 50%)

### Navigation Strategy Comparison

| Mode | Use When? | Math | Safety |
| :--- | :--- | :--- | :--- |
| **Interpolation** | Gradient Testing (0% → 100%) | Linear Mix | ✅ Safe (Bridge Hopping) |
| **Extrapolation** | Novelty/Threshold (100% → 150%) | Vector Extension | ⚠️ Risky (Deep Sea, needs Clamping) |

### Python API

**Rust Implementation (Recommended - High Performance):**

```python
from technical_architecture import Vector30D, NavigationEngine

# Create engine with safety clamp (20% max warp)
engine = NavigationEngine.with_max_warp(0.2)

# Create 30D vectors with all acoustic features
neutral = Vector30D(
    # Fundamental (3)
    mean_f0_hz=7000.0,
    f0_range_hz=400.0,
    duration_ms=50.0,
    # Grit Factors (3)
    harmonic_to_noise_ratio=20.0,
    spectral_flatness=0.3,
    harmonicity=0.8,
    # Motion Factors (7)
    attack_time_ms=5.0,
    decay_time_ms=20.0,
    sustain_level=0.7,
    vibrato_rate_hz=7.0,
    vibrato_depth=0.02,
    jitter=0.01,
    shimmer=0.03,
    # Fingerprint Factors (13 MFCCs)
    mfcc_1=-10.0, mfcc_2=-5.0, mfcc_3=-2.0, mfcc_4=-1.0,
    mfcc_5=-0.5, mfcc_6=-0.3, mfcc_7=-0.2, mfcc_8=-0.1,
    mfcc_9=0.0, mfcc_10=0.1, mfcc_11=0.2, mfcc_12=0.3,
    mfcc_13=0.4,
    # Spectral Dynamics (1)
    spectral_flux=0.5,
    # Rhythm Factors (3)
    median_ici_ms=15.0,
    onset_rate_hz=8.0,
    ici_coefficient_of_variation=0.3,
)

aggressive = Vector30D(
    # Fundamental (3)
    mean_f0_hz=8000.0,
    f0_range_hz=600.0,
    duration_ms=30.0,
    # Grit Factors (3)
    harmonic_to_noise_ratio=25.0,
    spectral_flatness=0.5,
    harmonicity=0.9,
    # Motion Factors (7)
    attack_time_ms=3.0,
    decay_time_ms=15.0,
    sustain_level=0.9,
    vibrato_rate_hz=10.0,
    vibrato_depth=0.05,
    jitter=0.03,
    shimmer=0.05,
    # Fingerprint Factors (13 MFCCs)
    mfcc_1=-8.0, mfcc_2=-3.0, mfcc_3=-1.0, mfcc_4=0.0,
    mfcc_5=0.5, mfcc_6=0.7, mfcc_7=0.8, mfcc_8=0.9,
    mfcc_9=1.0, mfcc_10=1.1, mfcc_11=1.2, mfcc_12=1.3,
    mfcc_13=1.4,
    # Spectral Dynamics (1)
    spectral_flux=0.7,
    # Rhythm Factors (3)
    median_ici_ms=10.0,
    onset_rate_hz=12.0,
    ici_coefficient_of_variation=0.5,
)

# INTERPOLATION MODE (Safe - Bridge Builder)
waypoint_50pct = engine.interpolate(neutral, aggressive, 0.5)
print(f"50% aggression F0: {waypoint_50pct.get_mean_f0_hz()}")  # 7500 Hz

# Add islands to database
engine.add_island("neutral", neutral, "marmoset")
engine.add_island("aggressive", aggressive, "marmoset")

# Find nearest island
target = Vector30D.default()
nearest = engine.find_nearest_island(target)
print(f"Nearest island: {nearest.key}")

# Apply safety clamping
waypoint = engine.clamp_to_safe_distance(
    target, neutral, "neutral"
)

print(f"Mode: {waypoint.get_mode()}")
print(f"Was clamped: {waypoint.was_clamped()}")
print(f"Distance: {waypoint.get_distance_to_anchor():.3f}")
```

**Rust Native API (Maximum Performance):**

```rust
use technical_architecture::{Vector30D, NavigationEngine};

let engine = NavigationEngine::with_max_warp(0.2);

// Create vectors
let v1 = Vector30D::default();
let v2 = Vector30D::new(/* 30 dimensions */);

// Interpolate (Bridge Builder)
let result = engine.interpolate(&v1, &v2, 0.5);

// Add islands
engine.add_island(AudioIsland {
    key: "marmoset_phee".to_string(),
    features: v1,
    audio: None,
    species: "marmoset".to_string(),
    metadata: HashMap::new(),
});

// Find nearest
let nearest = engine.find_nearest_island(&target);

// Safety clamp
let waypoint = engine.clamp_to_safe_distance(
    &target, &anchor, Some("island1".to_string())
);
```

### Test Coverage

- **Python Tests**: 21 tests
  - `test_interpolation_extrapolation.py`: Complete TDD suite
  - Interpolation (Bridge Builder): 6 tests
  - Extrapolation (Ocean Explorer): 6 tests
  - Delta Clamping (The Leash): 4 tests
  - Integration: 5 tests

- **Rust Tests**: 497 tests total (464 existing + 33 new Island Hopping)
  - `island_hopping.rs`: 33 tests ✅
  - Vector30D operations: 14 tests
  - Delta Clamping: 3 tests
  - PhraseDatabase: 5 tests
  - NavigationEngine: 3 tests
  - Timeline Executor: 3 tests
  - Granular Delta: 1 test
  - Integration: 4 tests

- **Python-Rust Integration Tests**: `test_rust_island_hopping.py`
  - Vector30D PyO3 bindings: 9 tests
  - NavigationEngine PyO3 bindings: 6 tests
  - NavigationWaypoint: 3 tests
  - AudioIsland: 1 test
  - Complete workflow: 1 test

### Implementation Strategy Summary

**Use Interpolation as your PRIMARY engine** for interaction (the "Bridge Builder").

**Use Extrapolation ONLY for Discovery Mode** (testing boundaries) and always keep the **"Leash"** (Clamping) active to prevent generating alien artifacts.

**Performance Note**: The Rust implementation provides:
- **SIMD-optimized** 30D vector math operations
- **Deterministic** safety-critical clamping (no GC pauses)
- **O(n)** nearest neighbor lookup (upgradeable to KD-tree)
- **<100ms** real-time timeline execution
- **PyO3 bindings** for seamless Python integration

---

## Island Hopping: Waypoint Navigation Through 30D Acoustic Space

The **"Island Hopping"** strategy extends the Cognitive Synthesis Engine with waypoint-based navigation through the 30D acoustic space. This enables real-time trajectory navigation between real audio phrases ("Safe Islands") while maintaining high fidelity.

### The Metaphor: The Acoustic Archipelago

Imagine your 30D Vector Space is an ocean:

- **The Islands (Real Audio)**: Safe, high-fidelity recordings
  - Example: `Corvid_Harmonic_Whistle.wav`, `Bat_FM_Sweep.wav`
- **The Ocean (Empty Space)**: Points that exist mathematically but have no recording
  - Example: A "70% Aggressive" call that you never recorded

**The Navigation Problem**: You want to get from Island A (Neutral) to Island B (Aggressive), but they are far apart. You cannot "fly" across the ocean (Pure Math) because you crash in the "Uncanny Valley" (Distance > 20).

### The Solution: Waypoint Navigation

Instead of flying across the open ocean, you travel along a path of calculated **Waypoints**, hopping from island to island.

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: The Chart (Acoustic Algebra)                           │
│  Define your Trajectory (The Route)                             │
│                                                                 │
│  intensity_waypoints = np.linspace(0.0, 1.0, num=10)           │
│  # [0%, 10%, 20%, ... 100%] Aggression                          │
│                                                                 │
│  for intensity in intensity_waypoints:                          │
│      target_vector = algebra.generate_graded_vector(            │
│          "aggression", intensity)                               │
│      map.add_waypoint(target_vector)                            │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  Step 2: The Boat (Nearest Neighbor Lookup)                    │
│  For every waypoint, find the Nearest Island                   │
│                                                                 │
│  for point in map.waypoints:                                    │
│      nearest_phrase = db.find_nearest_30d(point)               │
│      route_segments.append({                                    │
│          "virtual_target": point,                               │
│          "real_source": nearest_phrase,                         │
│          "distance": point.distance_to(nearest_phrase)          │
│      })                                                          │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  Step 3: The Voyage (Granular Warp)                             │
│  Travel between islands using Granular Synthesis               │
│                                                                 │
│  for segment in route_segments:                                 │
│      curr_phrase = segment["real_source"]                       │
│      warp_params = calculate_delta(                             │
│          segment["virtual_target"],                              │
│          curr_phrase.vector)                                    │
│      synth.load_source(curr_phrase)                             │
│      synth.set_warp(warp_params)                                │
│      audio = synth.synthesize()                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Navigation Modes

You can navigate the territory in three distinct patterns:

#### **Mode A: Linear Gradient (The "Road")**
- **Purpose**: Semantic Continuum Test
- **Navigation**: Move from 0% to 100% in a straight line
- **Use**: Testing if animals perceive emotion as a spectrum
- **Example**: Test aggression at 0%, 25%, 50%, 75%, 100%

#### **Mode B: Random Walk (The "Drift")**
- **Purpose**: Emergence / Discovery
- **Navigation**: Pick a random vector → Find nearest island → Warp there → Repeat
- **Use**: Discovering new "Valid" sounds
- **Example**: If an animal responds to a random drift, you found a new island

#### **Mode C: Semantic Avoidance (The "Safe Harbor")**
- **Purpose**: Deception Prevention
- **Navigation**: If calculated path enters "Danger Zone" (e.g., High Pitch Aggression without Visual Threat), apply **Braking**
- **Use**: Context-aware synthesis
- **Example**: Clamp intensity and move closer to "Safe Harbor" when semantically inappropriate

### Why Island Hopping Beats Other Strategies

| Strategy | Fidelity | Risk | Result |
| :--- | :--- | :--- | :--- |
| **Direct Flight (Pure Math)** | Low (Distance 27) | High | Crash in Uncanny Valley |
| **Static Jump (Retrieval)** | High (Distance 0) | None | Discrete jumps (No Continuum) |
| **Island Hopping (Hybrid)** | **High (Distance 6.5)** | **Low** | **Smooth Continuum using Real Physics** |

### Real-Time Performance with LRU Cache

The biggest risk with Island Hopping is the **Disk Load Penalty**. If your "Nearest Island" is on disk:
- **Read Time**: ~20ms-50ms
- **Latency Budget**: <100ms
- **Risk**: If you do this every phrase, you crash the interaction loop

**The Solution: Rust-Side LRU Cache**

```rust
pub struct CachedGranularSequencer {
    sequencer: MultiBufferGranularSequencer,
    cache: LruCache<String, CachedAudioBuffer>,
    max_cache_bytes: usize,  // Default: 50MB
    cache_hits: Arc<Mutex<u64>>,
    cache_misses: Arc<Mutex<u64>>,
}
```

**Workflow:**
1. **Startup**: Python Agent loads "High Probability Phrases" (Top 20% of database) into Rust Cache
2. **Real-Time**: During interaction, `generate_response` has >90% chance of finding the "Nearest Island" already in RAM
3. **Latency**: <1ms Lookup + <10ms Synthesis = **11ms Total** (Safe!)

### Contextual Pre-Fetching

To minimize Disk Misses, the Python Agent **Pre-Fetches** based on Context:

**Scenario: Bat "Social" Context**
1. **Context Start**: Agent predicts "I will need 'Social' phrases"
2. **Pre-Fetch**: Agent fires 5 Async commands to Rust:
   - `preload(id="bat_social_01.wav")`
   - `preload(id="bat_social_02.wav")`
   - ...
3. **Interaction**: User speaks "Aggression"
4. **Result**: `generate_response` finds nearest neighbor → **Cache Hit** (No latency)

### Python API

```python
from technical_architecture import Vector30D, NavigationEngine

# Create engine with safety clamp
engine = NavigationEngine.with_max_warp(0.2)

# Add islands (real audio phrases) with 30D features
engine.add_island(
    "neutral_001",
    Vector30D(
        # Fundamental (3)
        mean_f0_hz=7000.0, f0_range_hz=400.0, duration_ms=50.0,
        # Grit Factors (3)
        harmonic_to_noise_ratio=20.0, spectral_flatness=0.1, harmonicity=0.8,
        # Motion Factors (7)
        attack_time_ms=5.0, decay_time_ms=20.0, sustain_level=0.7,
        vibrato_rate_hz=7.0, vibrato_depth=0.02, jitter=0.01, shimmer=0.03,
        # Fingerprint Factors (13 MFCCs)
        mfcc_1=-10.0, mfcc_2=-5.0, mfcc_3=-2.0, mfcc_4=-1.0,
        mfcc_5=-0.5, mfcc_6=-0.3, mfcc_7=-0.2, mfcc_8=-0.1,
        mfcc_9=0.0, mfcc_10=0.1, mfcc_11=0.2, mfcc_12=0.3, mfcc_13=0.4,
        # Spectral Dynamics (1)
        spectral_flux=0.5,
        # Rhythm Factors (3)
        median_ici_ms=15.0, onset_rate_hz=8.0, ici_coefficient_of_variation=0.3,
    ),
    "marmoset"
)

engine.add_island(
    "aggressive_001",
    Vector30D(
        # Fundamental (3)
        mean_f0_hz=8000.0, f0_range_hz=600.0, duration_ms=40.0,
        # Grit Factors (3)
        harmonic_to_noise_ratio=5.0, spectral_flatness=0.7, harmonicity=0.6,
        # Motion Factors (7)
        attack_time_ms=3.0, decay_time_ms=15.0, sustain_level=0.9,
        vibrato_rate_hz=10.0, vibrato_depth=0.05, jitter=0.03, shimmer=0.05,
        # Fingerprint Factors (13 MFCCs)
        mfcc_1=-8.0, mfcc_2=-3.0, mfcc_3=-1.0, mfcc_4=0.0,
        mfcc_5=0.5, mfcc_6=0.7, mfcc_7=0.8, mfcc_8=0.9,
        mfcc_9=1.0, mfcc_10=1.1, mfcc_11=1.2, mfcc_12=1.3, mfcc_13=1.4,
        # Spectral Dynamics (1)
        spectral_flux=0.7,
        # Rhythm Factors (3)
        median_ici_ms=10.0, onset_rate_hz=12.0, ici_coefficient_of_variation=0.5,
    ),
    "marmoset"
)

# Interpolate between islands
neutral = engine.find_nearest_island(Vector30D.default())
aggressive = engine.find_nearest_island(
    Vector30D(mean_f0_hz=7500.0, f0_range_hz=500.0, duration_ms=45.0,
              harmonic_to_noise_ratio=15.0, spectral_flatness=0.3, harmonicity=0.7,
              # ... remaining 24 dimensions
              spectral_flux=0.6, median_ici_ms=12.0, onset_rate_hz=10.0,
              ici_coefficient_of_variation=0.4)
)

# Apply safety clamping
waypoint = engine.clamp_to_safe_distance(
    Vector30D.default(), neutral.features, "neutral_001"
)

print(f"Mode: {waypoint.get_mode()}")
print(f"Distance: {waypoint.get_distance_to_anchor():.3f}")

# Each waypoint contains:
# - target: Calculated 30D vector
# - mode: Interpolation or Extrapolation
# - anchor_island: Nearest real phrase
# - distance_to_anchor: Normalized distance
# - was_clamped: Whether safety clamp was applied

# Example workflow for gradient navigation
start_point = Vector30D.default()
end_point = Vector30D(
    mean_f0_hz=8000.0, f0_range_hz=600.0, duration_ms=40.0,
    # ... remaining 27 dimensions
)

# Create gradient waypoints
for i in range(10):
    alpha = i / 9.0  # 0.0 to 1.0
    waypoint_vector = engine.interpolate(start_point, end_point, alpha)
    print(f"Waypoint {i}: F0={waypoint_vector.get_mean_f0_hz()}Hz")
```

### Key Design Principles

1. **Safety First**: Delta clamping (max 20% warp) prevents over-warping artifacts
2. **High Fidelity**: Always grounded in real recordings (nearest neighbor lookup)
3. **Real-Time**: LRU cache guarantees <100ms response time
4. **Contextual Awareness**: Pre-fetching based on predicted context
5. **Biological Fidelity**: Mirrors how animals actually communicate (gradients, not discrete states)

### Test Coverage

- **Python Tests**: 45 tests total (Island Hopping: 24, Interpolation/Extrapolation: 21)
  - `test_island_hopping_navigation.py`: 24 tests
  - `test_interpolation_extrapolation.py`: 21 tests
    - ⚠️ **Contains deprecated Python fallback code** - Use Rust via PyO3 instead
    - See: `archive/deprecated_python_fallbacks/INTERPOLATION_EXTRAPOLATION_DEPRECATION.md`
  - Vector30D Calculations: 4 tests
  - Acoustic Algebra Engine: 4 tests
  - Phrase Database: 3 tests
  - Island Hopping Navigator: 5 tests
  - Safety and Edge Cases: 4 tests
  - Integration: 4 tests
  - Interpolation (Bridge Builder): 6 tests
  - Extrapolation (Ocean Explorer): 6 tests
  - Delta Clamping (The Leash): 4 tests
  - Integration: 5 tests

- **Rust Tests**: 497 tests total (464 existing + 33 new Island Hopping)
  - `island_hopping.rs`: 33 tests (SIMD-optimized 30D vector math, safety clamping)
  - All other modules: 464 tests

- **Python-Rust Integration**: `test_rust_island_hopping.py`
  - Tests PyO3 bindings for Vector30D, NavigationEngine, NavigationWaypoint

### Why This Matters

The Island Hopping strategy represents the **state-of-the-art** for bio-acoustic interaction:

1. **Enables Hypothesis Testing**: You can test intensities that exist nowhere in nature (e.g., "Is 50% aggression distinct from 40%?")
2. **Semantic Continuum**: Maps the continuous nature of animal communication (not discrete "words")
3. **Discovery Tool**: Random walk mode can find new valid vocalizations
4. **Deception Detection**: Semantic avoidance prevents inappropriate calls
5. **Real-Time Interaction**: LRU cache enables <100ms response times

**This moves your system from "Playback" to "Conversation"**.

---

## Archival Information

**Documentation:**
- `technical_architecture/TDD_PLAN_FIELD_FEATURES.md` - Field deployment implementation plan (COMPLETE)
- `technical_architecture/CLAUDE.md` - Detailed developer guide with API examples

**Archived Directories:** See `/src/archive/ARCHIVE.md` for details

- `deprecated_python_fallbacks/` - [NEW] Python execution-layer code superseded by Rust
  - `INTERPOLATION_EXTRAPOLATION_DEPRECATION.md` - Migration guide for Island Hopping
  - `ARCHIVE.md` - Deprecation index and rationale
  - **Key Principle**: Time-critical and safety-critical code must be in Rust
- `jungle-monitoring-system/` - Superseded by `cognitive_intelligence/`
- `audio_engine/` - Superseded by `technical_architecture/`
- `cognition/` - Superseded by `cognitive_intelligence/`
- `hybrid/` - Unused neural bridge implementation
- `test_cache/` - Temporary cache files
- `duplicate_tests/` - 28 backup test files with `_1.py` suffix

**Realtime Archive:** `/src/realtime/archive/ARCHIVE.md`

35 execution-layer Python files moved to Rust implementation.

**Python Fallback Deprecation:**

The following Python components have been superseded by Rust implementations:

| Component | Rust Replacement | Status | Performance |
|-----------|------------------|---------|-------------|
| `Vector30D` | `island_hopping.rs::Vector30D` | ✅ Active | 10-100x faster |
| `NavigationEngine` | `island_hopping.rs::NavigationEngine` | ✅ Active | Deterministic |
| `SafetyClamp` | `island_hopping.rs::SafetyClamp` | ✅ Active | Safety-critical |

**Migration:** Use `from technical_architecture import Vector30D, NavigationEngine` instead.

See: `archive/deprecated_python_fallbacks/INTERPOLATION_EXTRAPOLATION_DEPRECATION.md`

---

## License

**CC BY-ND 4.0 International** - See main project license for details.

---

## Author

Sheel Morjaria (sheelmorjaria@gmail.com)
