# Enhanced Field Deployment System Guide

## Overview

The Enhanced Field Deployment System provides a production-ready, jungle-safe audio processing and wildlife monitoring solution. It implements a hybrid Python/Rust architecture with comprehensive safety features, power management, and environmental monitoring for long-term autonomous field deployment.

**Architecture:** Peer-to-peer supervision with Rust (Execution Layer) and Python (Logic Layer)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Systemd Supervisor                        │
│  ┌──────────────────────────┐  ┌──────────────────────────┐     │
│  │  rust-field-engine       │  │  python-cognitive-agent  │     │
│  │  (Technical Architect)   │  │  (Logic Layer)           │     │
│  │                          │  │                          │     │
│  │  • Safety Critical       │  │  • Decision Making       │     │
│  │  • Audio Processing      │◄─┤  • Phrase Selection      │     │
│  │  • Hardware Control      │  │  • Learning              │     │
│  │  • Heartbeat Monitor     │  │  • Intent Generation     │     │
│  │                          │  │                          │     │
│  │  ZeroMQ SUB (Heartbeat)  │◄─┤  ZeroMQ PUB (Heartbeat)  │     │
│  └──────────────────────────┘  └──────────────────────────┘     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Key Principle:** Fail open to safety. If Python crashes, Rust immediately mutes audio and continues in Passthrough Mode.

---

## Field Deployment Modules

### 1. Environmental Monitor (`environmental_monitor.rs`)

**Tests:** 46 tests passing

Monitors environmental conditions and forces safe mode in adverse conditions.

**Capabilities:**
```rust
// Rain intensity classification
enum RainIntensity {
    None = 0,       // < 0.1 mm/hr
    Light = 1,      // 0.1 - 2.5 mm/hr
    Moderate = 2,   // 2.5 - 10 mm/hr
    Heavy = 3,      // 10 - 50 mm/hr
    Violent = 4,    // > 50 mm/hr
}

// Temperature classification
enum TemperatureClass {
    Freezing = 0,   // < 0°C
    Cold = 1,       // 0 - 10°C
    Cool = 2,       // 10 - 20°C
    Moderate = 3,   // 20 - 30°C
    Warm = 4,       // 30 - 35°C
    Hot = 5,        // 35 - 40°C
    Extreme = 6,    // > 40°C
}

// Light level classification
enum LightLevel {
    Dark = 0,       // < 10 lux
    Dim = 1,        // 10 - 100 lux
    Normal = 2,     // 100 - 1000 lux
    Bright = 3,     // 1000 - 10000 lux
    VeryBright = 4, // 10000 - 50000 lux
    Night = 5,      // > 50000 lux (moonlight)
}
```

**Session Management:**
- Forces Passthrough Mode during rain (Heavy+), extreme temperatures, or dark conditions
- Solar forecasting integration for predictive session planning
- Circadian rhythm awareness (night detection via light sensor)

---

### 2. Power Manager (`power_manager.rs`)

**Tests:** 54 tests passing

Intelligent power management with solar prediction and adaptive throttling.

**Power Modes:**
```rust
enum PowerMode {
    Normal,     // > 80% battery - full operations
    Medium,     // 50 - 80% - throttle non-critical
    Low,        // 20 - 50% - minimal processing
    Critical,   // < 20% - emergency conservation
}
```

**Throttling Control:**
```rust
struct AtomicThrottleFlags {
    fpga_throttle: Arc<AtomicBool>,      // FPGA acceleration
    source_sep_throttle: Arc<AtomicBool>, // Source separation
    synthesis_throttle: Arc<AtomicBool>,  // Synthesis quality
}
```

**Solar Prediction:**
- Task deferral based on predicted solar generation
- Battery health estimation with cycle counting
- Power budget calculation with runtime estimation

**Example Usage:**
```rust
let power_manager = PowerManager::new(config)?;

// Get current power status
let status = power_manager.get_power_status().await?;
println!("Battery: {}%, Solar: {} W", status.battery_percentage, status.solar_generation_w);

// Calculate power budget for task
let budget = power_manager.calculate_power_budget(&task).await?;
if budget.can_execute {
    power_manager.execute_task(task).await?;
}
```

---

### 3. Wildlife Sentry (`wildlife_sentry.rs`)

**Tests:** 24 tests passing

Low-power background monitoring for target species vocalizations.

**Species Signatures:**
```rust
enum WildlifeSpecies {
    Marmoset,      // 5-12 kHz harmonic calls
    Dolphin,       // 2-24 kHz whistles
    Bat,           // 20-90 kHz FM sweeps
    Finch,         // 2-8 kHz song patterns
}
```

**Detection:**
- FFT-based vocalization detection (configurable FFT size: 512-4096)
- Band-limited energy thresholding
- Wake trigger generation with urgency levels (Low, Medium, High, Critical)

**Wake Triggers:**
```rust
struct WakeTrigger {
    species: WildlifeSpecies,
    urgency: UrgencyLevel,
    confidence: f32,
    timestamp: DateTime<Utc>,
    frequency_band: (f32, f32),
}
```

**Debounce:** 5-second debounce to prevent rapid successive wake-ups

---

### 4. Data Synchronizer (`data_synchronizer.rs`)

**Tests:** 20 tests passing

Priority-based offline data queuing for intermittent connectivity.

**Priority Levels:**
```rust
enum SyncPriority {
    Critical,    // Safety events, compliance data
    High,        // Wildlife detections, research data
    Normal,      // Telemetry, metrics
    Low,         // Logs, diagnostics
}
```

**Storage Backends:**
- **SSD** - Primary, fast storage
- **USB** - Removable backup
- **SD Card** - Tertiary, archival

**Bandwidth Throttling:**
```rust
config.max_bandwidth_bps = 1_000_000;  // 1 Mbps
config.compression_enabled = true;
config.compression_level = 6;
```

**Sync Operations:**
```rust
// Queue critical data
synchronizer.queue_sync(
    data,
    SyncPriority::Critical,
    StorageBackend::Ssd
).await?;

// Process queue when connection available
synchronizer.process_queue().await?;
```

---

### 5. Acoustic Simulator (`acoustic_simulator.rs`)

**Tests:** 43 tests passing

TDD test fixture for comprehensive environmental simulation.

**Environmental Noise Generation:**
- Rain (colored noise with intensity variation)
- Wind (low-frequency rumble with gusts)
- Insects (high-frequency buzzing)
- Birds (harmonic chirps)

**SNR Mixing:**
```rust
// Mix target vocalization with noise at specified SNR
let mixed = simulator.mix_with_snr(
    target_audio,
    noise_type,
    signal_to_noise_db
)?;
```

**Environment Presets:**
```rust
enum EnvironmentType {
    Jungle,       // High rain, wind, insects
    Rainforest,   // Moderate rain, birds
    OpenField,    // Wind, minimal rain
    Laboratory,   // Minimal noise
}
```

---

## Production Features

### IACUC Compliance Engine (`iacuc_compliance.rs`)

**Tests:** 29 tests passing

Legal animal research protocol enforcement with audit trails.

**Compliance Checks:**
```python
# Time window enforcement
check = iacuc.check_compliance(intent)
# Returns: {compliant: bool, violations: [...], audit_log: [...]}

# Daily limits
- Max volume: < 85 dB SPL at 1m
- Max session duration: 4 hours
- Species-specific limits
- Veterinarian approval required for exceptions
```

---

### Time-Series Archive (`time_series_archive.rs`)

**Tests:** 24 tests passing

High-frequency time-series data storage with downsampled aggregation.

**Query API:**
```rust
// Query by time range
let data = archive.query_by_timerange(
    start,
    end,
    downsampling_factor
)?;

// Retention policies
config.retention_days_raw = 7;      // Keep raw data 7 days
config.retention_days_downsampled = 365;  // Keep downsampled 1 year
```

---

### Auto-Calibration (`auto_calibration.rs`)

**Tests:** 17 tests passing

Self-health checks with pink noise calibration and drift detection.

**Calibration Cycle:**
```rust
// Generate pink noise calibration tone
let calibration_tone = calibrator.generate_calibration_tone()?;

// Loopback analysis
let health = calibrator.analyze_loopback(recorded_tone)?;

// Schedule next calibration
calibrator.schedule_next_calibration(Duration::hours(24))?;
```

---

### Shadow Model Monitoring (`shadow_model_monitor.rs`)

**Tests:** 26 tests passing

Concept drift detection with automatic model rollback.

**Parallel Inference:**
```rust
// Run both active and baseline models
let active_result = active_model.predict(&features)?;
let baseline_result = baseline_model.predict(&features)?;

// Detect drift
let drift_detected = monitor.compare_results(&active_result, &baseline_result)?;

if drift_detected {
    monitor.rollback_to_baseline()?;
}
```

---

### Remote Web Dashboard (`web_dashboard.rs`)

**Tests:** 25 tests passing

HTTPS/WebSocket monitoring with emergency stop capabilities.

**WebSocket Events:**
```javascript
// Real-time spectrogram streaming
ws.on('spectrogram', (data) => {
    displaySpectrogram(data);
});

// Gauge updates
ws.on('gauge', (data) => {
    updateGauge('battery', data.battery_percentage);
    updateGauge('temperature', data.cpu_temp);
});

// Emergency stop
ws.emit('emergency_stop', { reason: 'User initiated' });
```

**JWT Authentication:**
```rust
let token = dashboard.generate_token(client_id, expiration)?;
dashboard.connect_client(client_id, ip_address, &token)?;
```

---

### Multi-Node Coordination (`multi_node_coordination.rs`)

**Tests:** 21 tests passing

PTP grandmaster election and TDMA scheduling for arrays.

**Grandmaster Election:**
```rust
// IEEE 1588 clock class comparison
let my_info = PtpClockInfo {
    clock_class: 248,      // Slave clock class
    clock_accuracy: 0x25,  // 1 μs accuracy
    priority2: 255,
};

let is_grandmaster = coordinator.elect_grandmaster(my_info).await?;
```

**TDMA Scheduling:**
```rust
// Schedule time slot for acoustic transmission
let slot = coordinator.schedule_transmission_slot(
    duration_ms,
    priority
).await?;
```

---

## Deployment Guide

### 1. Build Rust Components

```bash
cd technical_architecture
cargo build --release
```

### 2. Install Systemd Services

```bash
# Copy service files
sudo cp deployment/*.service /etc/systemd/system/
sudo systemctl daemon-reload

# Enable services
sudo systemctl enable rust-field-engine.service
sudo systemctl enable python-cognitive-agent.service
```

### 3. Configure Hardware

**Required Hardware:**
- Microphone (ultrasonic for bats/dolphins)
- Speaker (flat response 20 Hz - 20 kHz+)
- Environmental sensors (rain, temperature, light)
- Solar panel + battery system (for field deployment)
- GPS (for location tagging, optional)

**I2C Sensor Setup:**
```bash
# Rain sensor (I2C address 0x40)
# Temperature sensor (I2C address 0x48)
# Light sensor (I2C address 0x4A)
```

### 4. Start Services

```bash
sudo systemctl start rust-field-engine.service
sudo systemctl start python-cognitive-agent.service

# Check status
sudo systemctl status rust-field-engine.service
sudo systemctl status python-cognitive-agent.service
```

---

## Operation Modes

### Passthrough Mode (Safe Default)
- Python disconnected or heartbeats stopped
- Audio muted
- Raw audio recording continues
- Passive monitoring
- **Always safe - never outputs audio**

### Interactive Mode (Active)
- Python connected and sending heartbeats
- Processing intents from Python
- Synthesizing responses
- Full cognitive interaction

**Mode Switching:**
```rust
// Automatic based on heartbeat (100ms timeout)
let mode = peer_controller.tick()?;

match mode {
    OperationMode::Passthrough => {
        // Safe mode - recording only
    }
    OperationMode::Interactive => {
        // Active mode - process Python intents
    }
}
```

---

## Configuration Files

### Rust Field Engine Config

```toml
# technical_architecture/deployment/config.toml
[audio]
sample_rate = 48000
channels = 1
buffer_size = 1024

[safety]
max_rms_level = 0.8
min_duration_ms = 10
max_duration_ms = 5000

[peer_monitor]
heartbeat_timeout_ms = 100
heartbeat_port = 5555
connection_string = "tcp://127.0.0.1:5555"

[environmental]
rain_sensor_enabled = true
temperature_sensor_enabled = true
light_sensor_enabled = true

[power]
battery_capacity_wh = 500.0
solar_panel_area_m2 = 2.0
low_power_threshold = 20.0
```

### Python Cognitive Agent Config

```python
# realtime/config.py
AUDIO_CONFIG = {
    'sample_rate': 48000,
    'chunk_size': 1024,
}

HEARTBEAT_CONFIG = {
    'interval_ms': 20,
    'port': 5555,
}

COGNITIVE_CONFIG = {
    'model_path': 'models/',
    'confidence_threshold': 0.7,
}
```

---

## Monitoring & Logging

### View Logs

```bash
# Rust field engine logs
sudo journalctl -u rust-field-engine.service -f

# Python cognitive agent logs
sudo journalctl -u python-cognitive-agent.service -f
```

### Health Status

```bash
# Check service status
sudo systemctl status rust-field-engine.service
sudo systemctl status python-cognitive-agent.service

# Check peer-to-peer connection
sudo journalctl -u rust-field-engine.service | grep "Heartbeat"
```

---

## Troubleshooting

### Rust Engine Not Starting
```bash
# Check dependencies
ldd technical_architecture/target/release/rust_field_engine

# Check logs
sudo journalctl -u rust-field-engine.service -n 50
```

### Python Agent Crashes
```bash
# "Let it crash" philosophy - systemd auto-restarts
sudo systemctl restart python-cognitive-agent.service

# View crash logs
sudo journalctl -u python-cognitive-agent.service -n 100
```

### High CPU Usage
```bash
# Check thermal throttling
cat /sys/class/thermal/thermal_zone*/temp

# Adjust processing parameters
# Edit config.toml to reduce buffer_size or FFT size
```

### Battery Draining
```bash
# Check power status logs
sudo journalctl -u rust-field-engine.service | grep -i "power"

# Enter low power mode manually
# Publish low_power intent to ZeroMQ
```

---

## Testing

### Run All Tests

```bash
cd technical_architecture
cargo test  # 415 tests passing
```

### Run Field Deployment Tests

```bash
# Environmental monitor
cargo test environmental_monitor

# Power manager
cargo test power_manager

# Wildlife sentry
cargo test wildlife_sentry

# Data synchronizer
cargo test data_synchronizer

# Acoustic simulator
cargo test acoustic_simulator
```

### Run Python Tests

```bash
# From src/ directory
python3 -m pytest tests/ -v
```

---

## Performance Benchmarks

**Rust Performance (Release Build):**
- Audio processing: < 1ms latency per 1024-sample frame
- Heartbeat monitoring: < 1 μs timeout detection
- Mode switching: Immediate flag update
- FFT computation (1024): ~50 μs
- Wildlife detection: ~200 μs per frame

**Power Consumption:**
- Idle: ~2W (Rust) + ~5W (Python) = 7W total
- Processing: ~5W (Rust) + ~15W (Python) = 20W total
- Low power mode: ~1W total (Rust only, Python paused)

---

## Safety Compliance

### IACUC Protocol Enforcement
- Time window checking (active hours only)
- Volume limits (85 dB SPL max at 1m)
- Daily limits (4 hours max interaction time)
- Species-specific restrictions
- Audit trail for all interactions

### Environmental Safety
- Automatic muting during adverse weather
- Temperature-based operation limits
- Emergency stop capability
- Watchdog timer (100ms heartbeat timeout)

### Data Safety
- Provenance logging (deterministic audit trails)
- Black box data queuing (offline resilience)
- Automatic backup with retention policies
- Time-series archival with compression

---

## License

**CC BY-ND 4.0 International** - See main project license for details.

---

## Author

Sheel Morjaria (sheelmorjaria@gmail.com)

**Animal Vocalization Analysis Framework**
*Production-ready field deployment for cross-species communication research*
