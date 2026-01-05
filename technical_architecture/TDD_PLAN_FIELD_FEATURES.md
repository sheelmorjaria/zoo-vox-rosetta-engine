# TDD Plan: Field Deployment Features

## Implementation Status: ✅ COMPLETE

All 5 field deployment features have been successfully implemented using TDD methodology:

| Feature | Status | Tests | File |
|---------|--------|-------|------|
| Feature 1: Environmental Monitor | ✅ Complete | 46 passing | `src/environmental_monitor.rs` |
| Feature 2: Power Manager | ✅ Complete | 54 passing | `src/power_manager.rs` |
| Feature 3: Wildlife Sentry | ✅ Complete | 24 passing | `src/wildlife_sentry.rs` |
| Feature 4: Data Synchronizer | ✅ Complete | 20 passing | `src/data_synchronizer.rs` |
| Feature 5: Acoustic Simulator | ✅ Complete | 43 passing | `src/acoustic_simulator.rs` |
| **Total** | ✅ **Complete** | **187 passing** | **~4,200 LOC** |

### Test Results
```
running 266 tests
test result: ok. 266 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

### Module Exports (src/lib.rs)
All features are exported and accessible via the public API:
```rust
// Environmental monitor exports
pub use environmental_monitor::{
    EnvironmentalMonitor, EnvironmentalMonitorConfig, EnvironmentalConditions,
    RainIntensity, TemperatureClassification, LightLevel, SessionViability,
    SolarForecast, SensorReading,
};

// Power manager exports
pub use power_manager::{
    PowerManager, PowerManagerConfig, BatteryState, PowerMode,
    PowerBudget, SolarPrediction, ThrottleState,
};

// Wildlife sentry exports
pub use wildlife_sentry::{
    WildlifeSentry, WildlifeSentryConfig, SpeciesSignature,
    DetectionEvent, WakeTrigger, TriggerUrgency,
};

// Data synchronizer exports
pub use data_synchronizer::{
    DataSynchronizer, SyncConfig, LogEntry, QueuedEntry, SyncPriority,
    SyncStatus, StorageBackend, StorageType,
};

// Acoustic simulator exports
pub use acoustic_simulator::{
    AcousticSimulator, NoiseProfile, SpectralColor, TemporalCharacteristics,
    AcousticEnvironment, EnvironmentType, NoiseMixture,
};
```

---

## Overview

This document outlines the Test-Driven Development plan for implementing 5 critical field deployment features in the Rust Execution Layer. These features give the peer-to-peer system the "survival" layer needed for field deployment.

## Architecture Integration

The new `TechnicalArchitect` will be extended as follows:

```rust
pub struct TechnicalArchitect {
    // === Core Interaction (Existing) ===
    pub synthesizer: Arc<RwLock<GranularSynthesizer>>,
    pub safety_monitor: Arc<SafetyMonitor>,
    pub thermal_governor: Arc<ThermalGovernor>,
    pub ptp_clock: Arc<PtpClock>,
    pub logger: Arc<ProvenanceLogger>,

    // === Environmental Layer (NEW) ===
    pub power_manager: Arc<PowerManager>,           // Feature 2: Solar/Battery
    pub env_monitor: Arc<EnvironmentalMonitor>,     // Feature 1: Rain/Temp/Light
    pub wildlife_sentry: Arc<WildlifeSentry>,       // Feature 3: Wake-up trigger

    // === Data Layer (NEW) ===
    pub sync_manager: Arc<DataSynchronizer>,        // Feature 4: Offline queue

    // === Test Support (NEW) ===
    #[cfg(test)]
    pub acoustic_simulator: Option<AcousticSimulator>, // Feature 5: Test fixture
}
```

---

## Feature 1: Environmental Sentry (Environmental Monitor)

### Purpose
Monitor environmental conditions and override Python requests when conditions are unsuitable for vocalization interaction.

### Test Domains

#### 1.1 Sensor Polling Tests
- **test_env_sensor_polling**: Verify temperature, humidity, light sensors are polled
- **test_env_sensor_validation**: Reject invalid sensor readings
- **test_env_sensor_timeout**: Handle sensor unavailability gracefully

#### 1.2 Condition Classification Tests
- **test_rain_detection**: Classify rain intensity (None, Light, Moderate, Heavy, Storm)
- **test_temperature_classification**: Classify temperature ranges (Freezing, Cold, Mild, Hot, Extreme)
- **test_light_classification**: Classify ambient light (Dark, Dawn, Day, Dusk, Night)

#### 1.3 Session Override Tests
- **test_heavy_rain_forces_passthrough**: Heavy rain → Passthrough Mode
- **test_extreme_temperature_forces_passthrough**: Extreme temp → Passthrough Mode
- **test_normal_conditions_allow_interaction**: Normal conditions → Allow Interactive
- **test_env_override_python_request**: Python requests Interactive but rain forces Passthrough

#### 1.4 Forecast Integration Tests
- **test_solar_forecast_retrieval**: Retrieve solar forecast data
- **test_optimal_window_calculation**: Calculate optimal interaction windows
- **test_forecast_informs_python**: Send forecast data to Python agent

### Data Structures

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentalConditions {
    pub timestamp: PtpTimestamp,
    pub temperature_celsius: f32,
    pub humidity_percent: f32,
    pub light_lux: f32,
    pub rain_intensity_mm_h: f32,
    pub wind_speed_m_s: f32,
    pub atmospheric_pressure_hpa: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RainIntensity {
    None,
    Light,      // < 2.5 mm/h
    Moderate,   // 2.5 - 10 mm/h
    Heavy,      // 10 - 50 mm/h
    Storm,      // > 50 mm/h
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SessionViability {
    Viability,     // Conditions suitable for interaction
    Marginal,      // Borderline, use caution
    Infeasible,    // Conditions unsuitable, force Passthrough
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolarForecast {
    pub date: chrono::NaiveDate,
    pub sunrise_hour: u8,
    pub sunset_hour: u8,
    pub peak_solar_hours: Vec<(u8, u8)>, // (start_hour, end_hour)
    pub expected_cloud_cover_percent: f32,
}
```

---

## Feature 2: Intelligent Power Management (Solar Optimization)

### Purpose
Monitor battery/solar state and throttle system power consumption to extend field deployment time.

### Test Domains

#### 2.1 Battery State Tests
- **test_battery_percentage_read**: Read current battery percentage
- **test_battery_voltage_read**: Read battery voltage
- **test_battery_cycle_count**: Track battery charge cycles
- **test_battery_health_estimation**: Estimate battery health degradation

#### 2.2 Power Mode Tests
- **test_normal_power_mode**: > 80% battery → All features enabled
- **test_medium_power_mode**: 50-80% battery → Disable FPGA
- **test_low_power_mode**: 20-50% battery → Disable Conv-TasNet
- **test_critical_power_mode**: < 20% battery → Detection only

#### 2.3 Solar Prediction Tests
- **test_solar_gain_prediction**: Predict solar gain for next hour
- **test_power_budget_calculation**: Calculate available power budget
- **test_defer_heavy_tasks**: Defer synthesis when solar gain low

#### 2.4 Throttle Integration Tests
- **test_power_throttle_synthesis**: Throttle synthesis in low power
- **test_power_throttle_source_separation**: Disable separation in critical mode
- **test_power_notify_python**: Send power state to Python for scheduling

### Data Structures

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatteryState {
    pub percentage: f32,           // 0.0 - 100.0
    pub voltage_v: f32,            // Typically 11.0V - 14.4V for LiFePO4
    pub current_a: f32,            // Positive = charging, Negative = discharging
    pub cycle_count: u32,
    pub health_percent: f32,
    pub temperature_celsius: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PowerMode {
    Normal,      // > 80%: All features
    Medium,      // 50-80%: Disable FPGA
    Low,         // 20-50%: Disable Conv-TasNet, basic synthesis
    Critical,    // < 20%: Detection only, minimal processing
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerBudget {
    pub available_wh: f32,         // Watt-hours available
    pub predicted_solar_wh: f32,   // Predicted solar gain
    pub base_consumption_w: f32,   // Base system consumption
    pub synthesis_consumption_w: f32,
    pub estimated_runtime_hours: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolarPrediction {
    pub timestamp: PtpTimestamp,
    pub next_hour_gain_wh: f32,
    pub next_day_gain_wh: f32,
    pub confidence: f32,           // 0.0 - 1.0
}
```

---

## Feature 3: Broad-Spectrum Wildlife Sentry

### Purpose
Run a low-power background detector that wakes the Python agent when target species vocalizations are detected.

### Test Domains

#### 3.1 Sentry Initialization Tests
- **test_sentry_initialization**: Initialize sentry with species database
- **test_sentry_low_power_mode**: Verify sentry runs in low power mode

#### 3.2 Detection Tests
- **test_detect_marmoset_call**: Detect marmoset call in audio buffer
- **test_detect_dolphin_whistle**: Detect dolphin whistle
- **test_detect_bat_fm_sweep**: Detect bat FM sweep
- **test_reject_noise**: Reject non-vocalization audio

#### 3.3 Wake Trigger Tests
- **test_trigger_wake_python**: Send wake signal to Python on detection
- **test_no_trigger_on_noise**: Don't wake on non-target audio
- **test_debounce_detections**: Debounce rapid successive calls

#### 3.4 Multi-Species Tests
- **test_multi_species_detection**: Detect multiple species simultaneously
- **test_species_priority**: Prioritize certain species over others

### Data Structures

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WildlifeSentryConfig {
    pub target_species: Vec<String>,
    pub detection_threshold: f32,
    pub debounce_ms: u64,
    pub sample_rate: usize,
    pub fft_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionEvent {
    pub species: String,
    pub confidence: f32,
    pub timestamp: PtpTimestamp,
    pub start_sample: usize,
    pub duration_samples: usize,
    pub dominant_frequency_hz: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WakeTrigger {
    pub detections: Vec<DetectionEvent>,
    pub urgency: TriggerUrgency,
    pub suggested_response_duration_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TriggerUrgency {
    Low,      // Single distant call
    Medium,   // Multiple calls
    High,     // Close proximity or agitated calls
    Critical, // Alarm calls or distress
}

/// Species database with acoustic signatures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeciesSignature {
    pub name: String,
    pub frequency_range_hz: (f32, f32),
    pub call_duration_ms: (f32, f32),
    pub spectral_pattern: Vec<f32>, // MFCC-like template
    pub typical_snr_db: f32,
}
```

---

## Feature 4: Resilient Data Synchronization (Black Box)

### Purpose
Queue and synchronize log data reliably over intermittent/unreliable network connections.

### Test Domains

#### 4.1 Queue Management Tests
- **test_queue_entry**: Add log entry to queue
- **test_queue_persistence**: Queue survives process restart
- **test_queue_size_limit**: Enforce maximum queue size

#### 4.2 Redundancy Tests
- **test_dual_storage**: Store to both local SSD and USB
- **test_usb_hot_swap**: Handle USB removal/reinsertion
- **test_storage_fallback**: Fall back to single storage if one fails

#### 4.3 Compression Tests
- **test_compress_log_entry**: Compress log entries
- **test_compression_ratio**: Verify compression achieves target ratio
- **test_decompression**: Decompress and verify data integrity

#### 4.4 Sync Strategy Tests
- **test_bandwidth_throttling**: Limit sync bandwidth
- **test_prioritize_critical**: Prioritize critical entries
- **test_offline_queue**: Queue entries when offline
- **test_resume_sync**: Resume sync when connection available

### Data Structures

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncConfig {
    pub max_queue_size: usize,
    pub compression_enabled: bool,
    pub compression_level: u32,  // 0-9
    pub max_bandwidth_kbps: f32,
    pub storage_paths: Vec<String>, // Primary, secondary (USB)
    pub sync_endpoints: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueuedEntry {
    pub id: String,
    pub entry: LogEntry,
    pub compressed_data: Option<Vec<u8>>,
    pub priority: SyncPriority,
    pub created_at: PtpTimestamp,
    pub retry_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SyncPriority {
    Critical,  // Safety events, errors
    High,      // Session data, detections
    Normal,    // Regular logs
    Low,       // Telemetry, metrics
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncStatus {
    pub queue_size: usize,
    pub pending_upload: usize,
    pub last_sync: Option<PtpTimestamp>,
    pub bandwidth_usage_kbps: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageBackend {
    pub backend_type: StorageType,
    pub path: String,
    pub available_bytes: u64,
    pub is_mounted: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum StorageType {
    LocalSSD,
    USBDrive,
    SDCard,
    NetworkMount,
}
```

---

## Feature 5: Acoustic Simulation for TDD

### Purpose
Generate realistic environmental noise mixtures for testing audio processing algorithms.

### Test Domains

#### 5.1 Noise Generation Tests
- **test_generate_rain_noise**: Generate rain noise at specified intensity
- **test_generate_thunder**: Generate thunder effects
- **test_generate_wind_noise**: Generate wind noise
- **test_generate_insect_chorus**: Generate insect background
- **test_generate_bird_chorus**: Generate bird chorus

#### 5.2 Signal Mixing Tests
- **test_mix_clean_signal**: Mix clean vocalization with noise
- **test_set_snr**: Set specific SNR for mixture
- **test_add_reverb**: Add environmental reverb

#### 5.3 Environment Simulation Tests
- **test_simulate_dense_jungle**: Simulate dense jungle acoustic environment
- **test_simulate_rainforest**: Simulate rainforest with rain
- **test_simulate_open_field**: Simulate open field conditions

#### 5.4 Integration Tests
- **test_conv_tasnet_separation_test**: Test source separation with simulated noise
- **test_detection_in_noise**: Test wildlife detection in noisy conditions
- **test_synthesis_output_in_noise**: Verify synthesis quality in noise

### Data Structures

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseProfile {
    pub name: String,
    pub spectral_color: SpectralColor,
    pub amplitude_variation: f32,
    pub temporal_characteristics: TemporalCharacteristics,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SpectralColor {
    White,     // Flat spectrum
    Pink,      // 1/f
    Brown,     // 1/f²
    Blue,      // f
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalCharacteristics {
    pub attack_ms: f32,
    pub sustain_ms: f32,
    pub decay_ms: f32,
    pub modulation_rate_hz: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcousticEnvironment {
    pub environment_type: EnvironmentType,
    pub temperature_celsius: f32,
    pub humidity_percent: f32,
    pub wind_speed_m_s: f32,
    pub rain_intensity_mm_h: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseMixture {
    pub target_signal: Vec<f32>,
    pub noise_layers: Vec<(NoiseProfile, Vec<f32>, f32)>, // (profile, audio, level)
    pub snr_db: f32,
    pub sample_rate: u32,
}
```

---

## Implementation Order

### Phase 1: Foundation (Feature 5 - Acoustic Simulator)
**Why first?** Provides test infrastructure for other features.

1. Implement noise generation primitives
2. Implement signal mixing
3. Create realistic environment profiles
4. Write tests using simulator

### Phase 2: Environmental Layer (Features 1 & 2)
**Why second?** Foundation for survival logic.

1. Implement EnvironmentalMonitor (Feature 1)
2. Implement PowerManager (Feature 2)
3. Integrate with TechnicalArchitect
4. Write integration tests

### Phase 3: Detection Layer (Feature 3)
**Why third?** Depends on environmental context.

1. Implement WildlifeSentry
2. Integrate with environmental monitoring
3. Implement wake trigger logic
4. Write tests using acoustic simulator

### Phase 4: Data Layer (Feature 4)
**Why last?** Independent, can be parallel.

1. Implement DataSynchronizer
2. Integrate with ProvenanceLogger
3. Test with simulated network conditions

---

## Success Criteria

Each feature must have:
1. ✅ **100% test coverage** of public API
2. ✅ **Integration tests** with TechnicalArchitect
3. ✅ **Performance benchmarks** meeting targets
4. ✅ **Documentation** with examples
5. ✅ **Error handling** for all failure modes

---

## Testing Strategy

### Unit Tests
- Test individual functions in isolation
- Use mock data for deterministic results
- Cover all error branches

### Integration Tests
- Test interaction between components
- Use realistic (but reproducible) scenarios
- Verify state transitions

### Property-Based Tests
- Use proptest for invariants
- Test with random inputs
- Verify round-trip serialization

### Performance Tests
- Benchmark critical paths
- Verify timing constraints
- Test under load

---

## Files to Create

```
technical_architecture/src/
├── environmental_monitor.rs    # Feature 1
├── power_manager.rs             # Feature 2
├── wildlife_sentry.rs           # Feature 3
├── data_synchronizer.rs         # Feature 4
├── acoustic_simulator.rs        # Feature 5
└── lib.rs                       # Update with new modules

technical_architecture/tests/
├── integration_environmental_power.rs
├── integration_wildlife_detection.rs
├── integration_data_sync.rs
└── integration_full_system.rs
```

---

## Timeline Estimate

- **Phase 1** (Acoustic Simulator): 2-3 hours
- **Phase 2** (Environmental + Power): 4-5 hours
- **Phase 3** (Wildlife Sentry): 3-4 hours
- **Phase 4** (Data Sync): 3-4 hours
- **Integration & Documentation**: 2-3 hours

**Total**: ~14-19 hours of focused development

---

**Author:** Sheel Morjaria (sheelmorjaria@gmail.com)
**License:** CC BY-ND 4.0 International
