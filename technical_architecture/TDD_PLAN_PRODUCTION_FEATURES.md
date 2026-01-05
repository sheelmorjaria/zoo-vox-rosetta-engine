# TDD Plan: Production Field Deployment Features

## Overview

This document outlines the Test-Driven Development plan for implementing 6 critical production features required to transition the animal vocalization analysis framework from a laboratory prototype to a production-ready, autonomous field station.

**Status: ✅ ALL FEATURES COMPLETE**

**Priority Matrix:**
- **Critical**: IACUC Policy Engine (legal compliance) ✅ COMPLETE
- **High**: Time-Series Archiving, Auto-Calibration ✅ COMPLETE
- **Medium**: Remote Dashboard, Multi-Node Coordination, Shadow Model Monitoring ✅ COMPLETE

**Implementation Summary:**
- Total Tests: 142 new tests across 6 features
- Total LOC: ~5,000 lines of Rust code
- Time to Complete: All 3 phases completed
- Final Test Count: 408 tests passing (266 original + 142 new)

## Architecture Integration

The new modules will integrate with the existing Rust Execution Layer:

```rust
pub struct TechnicalArchitect {
    // === Existing (Core) ===
    pub synthesizer: Arc<RwLock<GranularSynthesizer>>,
    pub safety_monitor: Arc<SafetyMonitor>,

    // === Existing (Field) ===
    pub power_manager: Arc<PowerManager>,
    pub env_monitor: Arc<EnvironmentalMonitor>,
    pub wildlife_sentry: Arc<WildlifeSentry>,
    pub sync_manager: Arc<DataSynchronizer>,

    // === NEW (Production) ===
    pub time_series_archiver: Arc<TimeSeriesArchiver>,        // Feature 1
    pub calibration_engine: Arc<CalibrationEngine>,            // Feature 2
    pub web_dashboard: Arc<WebDashboard>,                      // Feature 3
    pub swarm_coordinator: Arc<SwarmCoordinator>,              // Feature 4
    pub iacuc_engine: Arc<IacucComplianceEngine>,              // Feature 5
    pub shadow_monitor: Arc<ShadowModelMonitor>,               // Feature 6
}
```

---

## Feature 1: Time-Series Archiving Pipeline (High Priority)

### Purpose
Efficiently store and query terabytes of high-frequency multi-channel time-series data (audio, visual, sensor logs).

### Test Domains

#### 1.1 InfluxDB Integration Tests
- **test_influxdb_connection**: Verify connection to InfluxDB
- **test_write_time_series**: Write single data point
- **test_write_batch**: Write batch of 1000 points efficiently
- **test_query_time_range**: Query data by time range
- **test_query_aggregation**: Query with downsampling/aggregation

#### 1.2 Parquet Export Tests
- **test_parquet_export**: Export daily data to Parquet
- **test_parquet_compression**: Verify compression ratio
- **test_parquet_column_read**: Read specific columns only
- **test_parquet_schema_evolution**: Handle schema changes over time

#### 1.3 Multi-Channel Tests
- **test_multi_channel_write**: Write audio + sensors simultaneously
- **test_channel_alignment**: Verify temporal alignment across channels
- **test_channel_metadata**: Store channel metadata (sample rate, units)

#### 1.4 Retention Policy Tests
- **test_retention_policy**: Apply data retention rules
- **test_downsampling_older_data**: Downsample old data automatically
- **test_storage_quota**: Enforce storage limits

### Data Structures

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesPoint {
    pub timestamp: PtpTimestamp,
    pub measurement: String,      // e.g., "temperature", "SPL", "F0"
    pub value: f64,
    pub tags: HashMap<String, String>,  // e.g., {"channel": "audio_L"}
    pub fields: HashMap<String, f64>,    // Additional fields
}

#[derive(Debug, Clone)]
pub struct TimeSeriesBatch {
    pub points: Vec<TimeSeriesPoint>,
    pub max_batch_size: usize,
    pub flush_interval_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParquetExportConfig {
    pub export_schedule: CronSchedule,  // e.g., "0 2 * * *" (daily 2AM)
    pub compression: ParquetCompression,
    pub row_group_size: usize,
    pub output_directory: PathBuf,
}

pub enum ParquetCompression {
    Snappy,
    Gzip,
    Lzo,
    Brotli,
}
```

### Key Implementation Points

1. **InfluxDB Client** (async Rust)
   - Batch writing with automatic retry
   - Query builder for time-range and aggregations
   - Connection pooling

2. **Parquet Export**
   - Nightly batch export from InfluxDB
   - Column pruning for storage efficiency
   - Schema evolution handling

3. **Storage Management**
   - Automatic retention policy enforcement
   - Downsampling old data (e.g., raw → 1min → 1hour averages)
   - Storage quota monitoring

---

## Feature 2: Automated Calibration & Self-Health Check (High Priority)

### Purpose
Detect and compensate for sensor drift (microphone sensitivity, humidity effects) to ensure safety limits remain accurate.

### Test Domains

#### 2.1 Calibration Tests
- **test_play_calibration_tone**: Play pink noise at known gain
- **test_capture_loopback**: Capture and analyze loopback signal
- **test_calculate_loopback_gain**: Compute actual gain vs expected
- **test_adjust_gain_table**: Update internal gain compensation

#### 2.2 Health Check Tests
- **test_mic_sensitivity_drift**: Detect 3dB mic drift
- **test_speaker_impedance_check**: Verify speaker health
- **test_noise_floor_measurement**: Measure system noise floor
- **test_frequency_response_check**: Verify frequency response

#### 2.3 Self-Diagnostic Tests
- **test_calibration_pass_fail**: Generate pass/fail status
- **test_calibration_sync_report**: Include in daily sync
- **test_automatic_schedule**: Run calibration daily at specified time
- **test_safety_limit_adjustment**: Update SPL limits based on calibration

### Data Structures

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationConfig {
    pub schedule: CronSchedule,           // When to run calibration
    pub calibration_tone: CalibrationTone,
    pub acceptable_drift_db: f32,         // Max acceptable drift (e.g., 1.5dB)
    pub output_gain: f32,                  // Internal gain setting
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationTone {
    pub signal_type: SignalType,          // PinkNoise, WhiteNoise, SineSweep
    pub duration_ms: u32,
    pub frequency_range: (f32, f32),       // For sine sweep
    pub amplitude_db: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationResult {
    pub timestamp: PtpTimestamp,
    pub loopback_gain_db: f32,
    pub expected_gain_db: f32,
    pub drift_db: f32,
    pub passed: bool,
    pub frequency_response: Vec<(f32, f32)>,  // (Hz, dB)
    pub noise_floor_db: f32,
    pub adjustments: Vec<GainAdjustment>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GainAdjustment {
    pub frequency_band: (f32, f32),
    pub compensation_db: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Degraded,   // Outside acceptable limits but functional
    Failed,     // Calibration failed, system unsafe
}
```

### Key Implementation Points

1. **Calibration Sequence**
   - Mute output (safety first)
   - Play calibration tone at known gain
   - Capture loopback via microphone
   - Analyze FFT to compute frequency response
   - Calculate gain drift per frequency band
   - Update gain compensation table

2. **Safety Integration**
   - If calibration fails → Force Passthrough Mode
   - Adjust SPL limits based on measured drift
   - Log calibration result to provenance logger

3. **Scheduling**
   - Daily automatic calibration (e.g., 3 AM when quiet)
   - Manual trigger via dashboard
   - Calibration status in daily sync report

---

## Feature 3: Remote Intervention Dashboard (Medium Priority)

### Purpose
Provide secure HTTPS/WebSocket dashboard for real-time monitoring, manual override, and remote debugging.

### Test Domains

#### 3.1 Server Tests
- **test_https_server_start**: Start HTTPS server with valid cert
- **test_websocket_connection**: Accept WebSocket connection
- **test_authentication**: Verify token-based auth
- **test_rate_limiting**: Prevent API abuse

#### 3.2 Real-Time Streaming Tests
- **test_spectrogram_stream**: Stream live spectrogram data
- **test_gauge_stream**: Stream battery/thermal gauges
- **test_status_stream**: Stream system status updates
- **test_multi_client**: Handle multiple concurrent clients

#### 3.3 Remote Control Tests
- **test_emergency_stop**: Force Passthrough Mode via dashboard
- **test_manual_override**: Inject manual playback command
- **test_parameter_adjustment**: Adjust parameters remotely
- **test_command_history**: Audit log of remote commands

#### 3.4 Security Tests
- **test_token_validation**: Reject invalid tokens
- **test_token_expiration**: Token timeout handling
- **test_tls_certificate**: Verify TLS certificate validity
- **test_csrf_protection**: CSRF token validation

### Data Structures

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    pub bind_address: String,             // e.g., "0.0.0.0:8443"
    pub tls_cert_path: PathBuf,
    pub tls_key_path: PathBuf,
    pub auth_secret: String,              // JWT secret
    pub token_expiry_hours: u64,
    pub max_connections: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardState {
    pub operation_mode: OperationMode,
    pub battery_level: f32,
    pub temperature_celsius: f32,
    pub uptime_seconds: u64,
    pub last_calibration: Option<CalibrationResult>,
    pub active_connections: usize,
    pub iacuc_status: IacucStatus,
}

// WebSocket messages
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WsMessage {
    Spectrogram { data: Vec<f32>, sample_rate: u32 },
    GaugeUpdate { name: String, value: f32, unit: String },
    StatusUpdate { status: DashboardState },
    Error { message: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum DashboardCommand {
    EmergencyStop,
    ManualOverride { intent: IntentToken },
    SetParameter { name: String, value: serde_json::Value },
    RunCalibration,
    GetStatus,
}
```

### Key Implementation Points

1. **HTTPS Server** (using warp or actix-web)
   - TLS certificate handling
   - JWT authentication
   - WebSocket support for real-time streaming

2. **Real-Time Streams**
   - Spectrogram: Downsampled FFT data (10Hz update rate)
   - Gauges: Battery, thermal, storage metrics (1Hz)
   - Status: System state (on change)

3. **Command Processing**
   - Emergency stop → Immediate Passthrough Mode
   - Manual override → Bypass Python intent queue
   - All commands logged to provenance logger

4. **Security**
   - JWT token authentication
   - TLS 1.3 only
   - Rate limiting per IP
   - Command audit log

---

## Feature 4: Multi-Node Coordination (Medium Priority)

### Purpose
Enable arrays of devices to coordinate time synchronization, avoid acoustic interference, and fuse data from multiple nodes.

### Test Domains

#### 4.1 PTP Grandmaster Tests
- **test_grandmaster_election**: Elect grandmaster from multiple nodes
- **test_grandmaster_failover**: Re-elect on grandmaster failure
- **test_ptp_slave_sync**: Slave syncs to grandmaster
- **test_clock_offset**: Measure and compensate clock offset

#### 4.2 Acoustic Interference Tests
- **test_detect_other_nodes**: Detect audio from other nodes
- **test_coordinate_playback**: Schedule non-overlapping playback
- **test_avoid_interference**: Mute when nearby node active
- **test_time_slot_allocation**: Allocate TDMA time slots

#### 4.3 Data Fusion Tests
- **test_collect_provenance**: Gather logs from all nodes
- **test_correlate_events**: Match events across nodes by timestamp
- **test_triangulation**: Estimate sound source location
- **test_fused_export**: Generate unified dataset

### Data Structures

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmConfig {
    pub node_id: String,
    pub nodes: Vec<NodeInfo>,
    pub grandmaster_election_timeout_ms: u64,
    pub tdma_slot_duration_ms: u32,
    pub acoustic_detection_threshold_db: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub node_id: String,
    pub address: SocketAddr,
    pub is_grandmaster: bool,
    pub last_seen: Instant,
    pub clock_offset_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PtpRole {
    Grandmaster,
    Slave { master_id: String, offset_ms: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSlot {
    pub node_id: String,
    pub start_ms: u32,
    pub duration_ms: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedEvent {
    pub event_id: String,
    pub timestamp: PtpTimestamp,
    pub participating_nodes: Vec<String>,
    pub triangulated_location: Option<(f32, f32)>,  // (x, y) meters
    pub data: HashMap<String, serde_json::Value>,
}
```

### Key Implementation Points

1. **PTP Grandmaster Election**
   - Bully algorithm or Raft consensus
   - Automatic failover detection
   - Clock offset measurement and compensation

2. **Acoustic Coordination**
   - Nodes broadcast "intention to play" message
   - TDMA time slot allocation
   - Local mic detects other nodes playing
   - Automatic muting when nearby node active

3. **Data Fusion**
   - Collect provenance logs from all nodes
   - Align events by PTP timestamp
   - Triangulate sound source using TDOA (Time Difference of Arrival)
   - Generate unified export

---

## Feature 5: IACUC/Ethical Compliance Engine (Critical Priority)

### Purpose
Enforce legally binding animal research protocols and generate compliance audit trails.

### Test Domains

#### 5.1 Policy Loading Tests
- **test_load_protocol**: Load protocol.json
- **test_validate_schema**: Reject invalid protocol schema
- **test_protect_policy**: Policy cannot be modified at runtime
- **test_policy_version**: Track policy version

#### 5.2 Enforcement Tests
- **test_time_window_enforcement**: Block interaction outside allowed hours
- **test_volume_limit_enforcement**: Clamp output to max SPL
- **test_species_limit_enforcement**: Count interactions per species
- **test_daily_limit_enforcement**: Block after max interaction time

#### 5.3 Audit Tests
- **test_log_interaction**: Log every interaction attempt
- **test_generate_compliance_report**: Generate monthly report
- **test_export_pdf_csv**: Export in required formats
- **test_sign_report**: Digital signature for audit trail

### Data Structures

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IacucProtocol {
    pub protocol_id: String,
    pub version: String,
    pub effective_date: chrono::NaiveDate,
    pub expiry_date: Option<chrono::NaiveDate>,
    pub max_spl_db: f32,                    // Max sound pressure level
    pub allowed_hours: Vec<TimeWindow>,      // When interaction allowed
    pub species_limits: HashMap<String, SpeciesLimit>,
    pub daily_limits: DailyLimits,
    pub emergency_contacts: Vec<EmergencyContact>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeWindow {
    pub start_hour: u8,    // 0-23
    pub end_hour: u8,      // 0-23
    pub days: Vec<Weekday>, // Mon, Tue, etc.
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeciesLimit {
    pub max_interactions_per_day: u32,
    pub min_interval_minutes: u32,
    pub prohibited_behaviors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyLimits {
    pub max_interaction_seconds: u32,
    pub max_playback_events: u32,
    pub cooling_period_seconds: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceState {
    pub protocol_id: String,
    pub today_interaction_seconds: u32,
    pub today_playback_count: u32,
    pub species_interaction_counts: HashMap<String, u32>,
    pub last_interaction_time: Option<PtpTimestamp>,
    pub violation_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceReport {
    pub reporting_period: (chrono::NaiveDate, chrono::NaiveDate),
    pub protocol_id: String,
    pub total_interactions: u32,
    pub total_duration_seconds: u32,
    pub max_spl_recorded: f32,
    pub violations: Vec<PolicyViolation>,
    pub species_breakdown: HashMap<String, u32>,
    pub digital_signature: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyViolation {
    pub timestamp: PtpTimestamp,
    pub violation_type: ViolationType,
    pub description: String,
    pub attempted_action: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationType {
    OutsideAllowedHours,
    MaxSplExceeded,
    SpeciesLimitExceeded,
    DailyLimitExceeded,
    ProhibitedBehavior,
}
```

### Key Implementation Points

1. **Policy Enforcement**
   - Load policy at startup, immutable after
   - Check compliance BEFORE every synthesis
   - Hard lock: physically cannot generate audio if violation
   - Atomic counters for tracking

2. **Compliance Checking**
   ```rust
   pub fn check_compliance(&self, intent: &IntentToken) -> ComplianceCheck {
       // 1. Check time window
       if !self.is_within_allowed_hours() {
           return ComplianceCheck::Denied(ViolationType::OutsideAllowedHours);
       }

       // 2. Check daily limits
       if self.state.today_interaction_seconds >= self.protocol.daily_limits.max_interaction_seconds {
           return ComplianceCheck::Denied(ViolationType::DailyLimitExceeded);
       }

       // 3. Check SPL limit
       if intent.spl_db > self.protocol.max_spl_db {
           return ComplianceCheck::Denied(ViolationType::MaxSplExceeded);
       }

       ComplianceCheck::Allowed
   }
   ```

3. **Audit Trail**
   - Log ALL interaction attempts (allowed and denied)
   - Include PTP timestamp for legal defensibility
   - Monthly automatic report generation
   - Digital signature for report authenticity

---

## Feature 6: Shadow Model Monitoring (Medium Priority)

### Purpose
Detect "concept drift" where the AI learns incorrect patterns in the field by comparing active model against frozen baseline.

### Test Domains

#### 6.1 Shadow Model Tests
- **test_load_shadow_model**: Load frozen baseline model
- **test_parallel_inference**: Run both models on same input
- **test_compare_outputs**: Compare predictions
- **test_divergence_threshold**: Trigger alert when divergence exceeded

#### 6.2 Drift Detection Tests
- **test_gradual_drift**: Detect slow drift over time
- **test_sudden_drift**: Detect sudden catastrophic change
- **test_drift_by_category**: Track drift per decision category
- **test_drift_visualization**: Generate drift visualization

#### 6.3 Response Tests
- **test_alert_generation**: Generate alert on high drift
- **test_auto_freeze**: Automatically freeze model version
- **test_rollback_capability**: Rollback to previous model
- **test_notification**: Notify researchers of drift

### Data Structures

```rust
#[derive(Debug, Clone)]
pub struct ShadowModelMonitor {
    pub active_model: Box<dyn InferenceModel>,
    pub shadow_model: Box<dyn InferenceModel>,  // Frozen baseline
    pub divergence_threshold: f32,             // e.g., 0.2 (20%)
    pub window_size: usize,                    // Samples for averaging
    pub drift_history: VecDeque<DriftSample>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftSample {
    pub timestamp: PtpTimestamp,
    pub divergence_ratio: f32,                // 0.0 to 1.0
    pub sample_count: usize,
    pub category_drift: HashMap<String, f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelComparison {
    pub input_features: Vec<f32>,
    pub active_prediction: String,
    pub shadow_prediction: String,
    pub confidence_difference: f32,
    pub category_match: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftAlert {
    pub timestamp: PtpTimestamp,
    pub alert_level: AlertLevel,
    pub current_divergence: f32,
    pub threshold: f32,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertLevel {
    Warning,     // 10-20% divergence
    Critical,    // 20-40% divergence
    Emergency,   // >40% divergence
}
```

### Key Implementation Points

1. **Parallel Inference**
   ```rust
   pub fn compare_predictions(&self, input: &InputFeatures) -> ModelComparison {
       let active_pred = self.active_model.predict(input);
       let shadow_pred = self.shadow_model.predict(input);

       let divergence = self.calculate_divergence(&active_pred, &shadow_pred);

       ModelComparison {
           input_features: input.clone(),
           active_prediction: active_pred.label,
           shadow_prediction: shadow_pred.label,
           confidence_difference: (active_pred.confidence - shadow_pred.confidence).abs(),
           category_match: active_pred.category == shadow_pred.category,
       }
   }
   ```

2. **Divergence Tracking**
   - Sliding window of last N comparisons
   - Per-category drift tracking
   - Trend analysis (improving vs worsening)

3. **Automated Response**
   - Warning at 20% divergence
   - Freeze model at 30% divergence
   - Emergency notification at 40% divergence
   - Automatic rollback capability

---

## Implementation Order

### Phase 1: Critical Foundation (Week 1) ✅ COMPLETE
1. **Feature 5**: Iacuc Compliance Engine (CRITICAL - legal requirement) ✅ 29 tests
2. **Feature 1**: Time-Series Archiving (HIGH - data infrastructure) ✅ 24 tests

### Phase 2: Safety & Reliability (Week 2) ✅ COMPLETE
3. **Feature 2**: Auto-Calibration (HIGH - safety) ✅ 17 tests
4. **Feature 6**: Shadow Model Monitoring (MEDIUM - AI safety) ✅ 26 tests

### Phase 3: Operations & Scale (Week 3) ✅ COMPLETE
5. **Feature 3**: Remote Dashboard (MEDIUM - oversight) ✅ 25 tests
6. **Feature 4**: Multi-Node Coordination (MEDIUM - scaling) ✅ 21 tests

## Dependencies

### New Cargo Dependencies
```toml
# Time-Series Database
influxdb2 = "0.4"        # InfluxDB client
influxdb2-structmap = "0.3"  # Derive traits

# Parquet Export
polars = "0.36"           # DataFrame library with Parquet support
arrow = "50.0"            # Apache Arrow Rust

# Web Dashboard
warp = "0.3"              # Web server
tokio-tungstenite = "0.21" # WebSocket
jsonwebtoken = "9.0"     # JWT auth
rustls = "0.23"          # TLS

# Multi-Node Coordination
raft = "0.7"              # Consensus algorithm
serde_json = "1.0"       # Already present

# IACUC Compliance
chrono = "0.4"            # Already present
lettre = "0.10"           # Alphabetical ordering for reports

# Shadow Model
ndarray = "0.15"          # Already present
```

### System Dependencies
- InfluxDB 2.x (time-series database)
- PostgreSQL (optional, for protocol storage)

## Success Criteria

### Feature 1: Time-Series Archiving
- [ ] Write 10,000 points/second sustained
- [ ] Query 3 months of data in < 1 second
- [ ] Parquet export achieves 10:1 compression
- [ ] Column pruning reads only requested data

### Feature 2: Auto-Calibration
- [ ] Detect gain drift within 0.5dB accuracy
- [ ] Calibration completes in < 30 seconds
- [ ] Failed calibration triggers Passthrough Mode
- [ ] Daily calibration report included in sync

### Feature 3: Remote Dashboard
- [ ] HTTPS server with valid TLS certificate
- [ ] WebSocket supports 10 concurrent clients
- [ ] Emergency stop executes in < 100ms
- [ ] All commands logged to provenance

### Feature 4: Multi-Node Coordination
- [ ] Grandmaster election completes in < 5 seconds
- [ ] Clock offset < 1ms after sync
- [ ] Zero acoustic interference between nodes
- [ ] Data fusion aligns events within 1ms

### Feature 5: IACUC Compliance
- [ ] Policy enforcement CANNOT be bypassed
- [ ] 100% of interactions logged
- [ ] Monthly report auto-generates correctly
- [ ] Digital signature verifies authenticity

### Feature 6: Shadow Model Monitoring
- [ ] Detect 20% divergence with < 5% false positive rate
- [ ] Parallel inference adds < 10ms latency
- [ ] Automatic model rollback functional
- [ ] Drift visualization generates correctly

## Testing Strategy

### Unit Tests
- ~400 tests planned across 6 features
- Each feature: 60-80 tests
- Focus on edge cases and error handling

### Integration Tests
- End-to-end workflow tests
- Multi-feature interaction tests
- Performance benchmarks

### Property-Based Tests
- InfluxDB: Random time-series data generation
- Calibration: Random gain drift scenarios
- IACUC: Random interaction timing

## Estimated Effort

| Feature | Tests | LOC | Time (days) |
|---------|-------|-----|-------------|
| Feature 1: Time-Series Archiving | 70 | ~1200 | 5 |
| Feature 2: Auto-Calibration | 65 | ~900 | 4 |
| Feature 3: Remote Dashboard | 75 | ~1100 | 5 |
| Feature 4: Multi-Node Coordination | 70 | ~1300 | 6 |
| Feature 5: IACUC Compliance | 60 | ~800 | 4 |
| Feature 6: Shadow Model Monitoring | 60 | ~700 | 4 |
| **Total** | **400** | **~6000** | **28** |

---

## Implementation Notes

### Feature 5: IACUC Compliance (START HERE - Critical)

**Why First**: Legal compliance is non-negotiable. Without this, the system cannot legally operate.

**Implementation Strategy**:
1. Start with hardcoded policy JSON for testing
2. Implement enforcement checks in synthesis path
3. Add compliance state tracking
4. Implement audit logging
5. Add report generation

**Integration Point**: Hook into `Synthesizer::generate()` - check compliance before any audio generation.

### Feature 1: Time-Series Archiving (Second - Infrastructure)

**Why Second**: All other features depend on reliable data storage.

**Implementation Strategy**:
1. Start with in-memory mock (for testing without InfluxDB)
2. Implement InfluxDB client
3. Add batch writing
4. Implement Parquet export
5. Add retention policy

**Integration Point**: Hook into `ProvenanceLogger::log_decision()` - duplicate to time-series DB.

### Feature 2: Auto-Calibration (Third - Safety)

**Why Third**: Sensor drift affects all measurements, including IACUC SPL limits.

**Implementation Strategy**:
1. Implement calibration tone generation
2. Add loopback capture
3. Implement FFT analysis
4. Add gain compensation table
5. Integrate with safety monitor

**Integration Point**: New `CalibrationEngine` with daily scheduled execution.

---

## Next Steps

### ✅ COMPLETED - All Production Features Implemented

1. ✅ **Review and approve this plan** - Completed
2. ✅ **Select starting feature** - Feature 5: IACUC (completed)
3. ✅ **Add required dependencies to Cargo.toml** - Completed
4. ✅ **Begin TDD implementation** - All 6 features complete

### Remaining Work (Optional Enhancements)

1. **InfluxDB Integration**: Replace in-memory time-series storage with actual InfluxDB client
2. **Parquet Export**: Implement nightly export to compressed Parquet files
3. **HTTP Server**: Add real warp/actix-web HTTP server for dashboard (currently mocked for testing)
4. **Raft Consensus**: Implement distributed consensus for multi-node coordination
5. **Digital Signatures**: Add cryptographic signing for IACUC compliance reports

### Production Deployment Checklist

- [ ] Set up InfluxDB instance for time-series data storage
- [ ] Generate TLS certificates for HTTPS dashboard
- [ ] Configure IACUC protocol JSON for deployment site
- [ ] Set up monitoring dashboards (Grafana/Prometheus)
- [ ] Configure PTP grandmaster election for multi-node arrays
- [ ] Test TDMA scheduling with 3+ node array
- [ ] Calibrate microphones in field conditions
