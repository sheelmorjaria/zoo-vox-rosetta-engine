/**
 * Technical Architecture - Rust Execution Layer
 * ===============================================
 *
 * This crate provides the Rust execution layer for the animal vocalization
 * analysis system. It handles all time-critical operations including:
 *
 * - Source separation using Conv-TasNet (via ONNX/Tract)
 * - Real-time audio synthesis with granular engines
 * - Thermal management and power governance
 * - Safety monitoring with watchdog timers
 * - IEEE 1588 PTP for precision timing
 * - Deterministic provenance logging
 *
 * Architecture Strategy:
 * ----------------------
 * This crate follows the "Execution vs. Logic" split:
 *
 * - **Execution Layer (Rust)**: Signal processing, hardware access, safety
 * - **Logic Layer (Python)**: Cognitive intelligence, decision making, learning
 *
 * The crate exposes a clean PyO3 interface for Python integration.
 *
 * Author: Sheel Morjaria (sheelmorjaria@gmail.com)
 * License: CC BY-ND 4.0 International
 */

use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::RwLock;
use parking_lot::Mutex;
use anyhow::{Result, Context};
use log::{info, warn, error};
use serde::{Deserialize, Serialize};

// Re-export public types
pub use source_separation::{ConvTasNetSeparator, SeparatorConfig};
pub use thermal::{ThermalGovernor, ThermalState, TemperatureReading, ThermalStats};
pub use safety::{SafetyMonitor, SafetyConfig, SafetyViolation, WatchdogTimer, SafetyStats};
pub use synthesis::{
    GranularSynthesizer, SynthesisConfig, AudioSegment, AudioFeatures,
    SynthesisMode, PhraseSegment, MicroharmonicConstraints, SynthesisResult,
    ValidationResult, SafetyCheck, SpeciesParameters,
    MicroharmonicValidator, RealTimeSafetyMonitor, CrossSpeciesAdapter,
    ConcatenativeSynthesizer, SuperpositionalSynthesizer, CombinedSynthesizer,
    EnhancedMicroharmonicSynthesizer, SynthesisPerformanceStats,
    // Dynamic Microharmonic (NEW)
    DynamicMicroharmonicParams, DynamicMicroharmonicSynthesizer,
    generate_dynamic_microharmonic_sample,
};
pub use ptp::{PtpClock, PtpTimestamp};
pub use logging::ProvenanceLogger;
pub use master_controller::{
    IntentToken, ExecutionReceipt, Action, HealthStatus,
    IntentPriority, SynthesisComplexity, RejectionReason,
    SessionProfile, CognitiveProcessor, WatchdogConfig,
    SharedMemoryConfig, SharedMemoryRingBuffer, AtomicParameters,
    detect_fpga,
};

#[cfg(feature = "python-bindings")]
pub use master_controller::PyCognitiveProcessor;

// Peer controller exports
pub use peer_controller::{
    PeerController, PeerControllerConfig, OperationMode, AudioMuteState,
    HeartbeatMessage,
};

// Acoustic simulator exports (for TDD testing)
pub use acoustic_simulator::{
    AcousticSimulator, NoiseProfile, SpectralColor, TemporalCharacteristics,
    AcousticEnvironment, EnvironmentType, NoiseMixture,
};

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

// IACUC compliance exports
pub use iacuc_compliance::{
    IacucComplianceEngine, IacucProtocol, ComplianceState,
    ComplianceCheck, IacucIntent, IacucIntentType, TimeWindow, Weekday,
    SpeciesLimit, DailyLimits, EmergencyContact, PolicyViolation,
    ViolationType,
};

// Time-series archive exports
pub use time_series_archive::{
    TimeSeriesArchiver, TimeSeriesConfig, TimeSeriesPoint, TimeSeriesBatch,
    ParquetExportConfig, ParquetCompression, RetentionPolicy, StorageQuota,
    StorageStats,
};

// Auto-calibration exports
pub use auto_calibration::{
    CalibrationEngine, CalibrationConfig, CalibrationTone, CalibrationResult,
    GainAdjustment, CalibrationHealthStatus, SignalType, SpeakerImpedance,
    FrequencyResponsePoint,
};

// Shadow model monitor exports
pub use shadow_model_monitor::{
    ShadowModelMonitor, ShadowModelConfig, InputFeatures, ModelPrediction,
    DriftSample, ModelComparison, DriftAlert, AlertLevel,
    InferenceModel, MockActiveModel, MockShadowModel,
};

// Web dashboard exports
pub use web_dashboard::{
    WebDashboard, DashboardConfig, DashboardState, DashboardOperationMode,
    IacucStatus, CalibrationDashboardStatus, WsMessage, DashboardCommand,
    AuthToken, CommandAuditLog, CommandResult, GaugeValue,
};

// Multi-node coordination exports
pub use multi_node_coordination::{
    MultiNodeCoordinator, ClusterConfig, NodeInfo, NodeCapabilities,
    TdmaSlot, TdmaSchedule, FusedDetectionData, LocationEstimate,
    ElectionResult, NodeId, ClusterId, ClockClass, ClockAccuracy,
};

// Performance testing exports
pub use peer_controller_performance::{
    PerformanceMetrics, PeerControllerSimulator,
    benchmark_serialization_throughput, benchmark_message_processing,
    benchmark_timeout_detection, benchmark_mode_switching,
    benchmark_concurrent_processing, benchmark_memory_allocation,
    run_all_benchmarks, format_metrics,
};

// Import modules
mod source_separation;
mod thermal;
mod safety;
mod synthesis;
mod ptp;
mod logging;
mod master_controller;
mod peer_controller;
mod acoustic_simulator;
mod environmental_monitor;
mod power_manager;
mod wildlife_sentry;
mod data_synchronizer;
mod iacuc_compliance;
mod time_series_archive;
mod auto_calibration;
mod shadow_model_monitor;
mod web_dashboard;
mod multi_node_coordination;
pub mod peer_controller_performance;

/// Configuration for the Technical Architect
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechArchConfig {
    /// Source separation configuration
    pub separator: SeparatorConfig,
    /// Thermal configuration
    pub thermal: thermal::ThermalConfig,
    /// Safety configuration
    pub safety: SafetyConfig,
    /// Synthesis configuration
    pub synthesis: SynthesisConfig,
    /// PTP configuration
    pub ptp: ptp::PtpConfig,
    /// Logging configuration
    pub logging: logging::LoggingConfig,
    /// Target latency budget in milliseconds
    pub target_latency_ms: f64,
}

impl Default for TechArchConfig {
    fn default() -> Self {
        Self {
            separator: SeparatorConfig::default(),
            thermal: thermal::ThermalConfig::default(),
            safety: SafetyConfig::default(),
            synthesis: SynthesisConfig::default(),
            ptp: ptp::PtpConfig::default(),
            logging: logging::LoggingConfig::default(),
            target_latency_ms: 100.0, // 100ms budget
        }
    }
}

/// Technical Architect - Main entry point for the Rust execution layer
///
/// This struct coordinates all time-critical operations and provides
/// a clean API for both Rust and Python consumers.
pub struct TechnicalArchitect {
    /// Configuration
    config: TechArchConfig,
    /// Source separator
    separator: Arc<RwLock<ConvTasNetSeparator>>,
    /// Thermal governor
    thermal: Arc<ThermalGovernor>,
    /// Safety monitor
    safety: Arc<SafetyMonitor>,
    /// Synthesis engine
    synthesizer: Arc<RwLock<GranularSynthesizer>>,
    /// PTP clock
    ptp_clock: Arc<PtpClock>,
    /// Provenance logger
    logger: Arc<ProvenanceLogger>,
    /// Performance statistics
    stats: Arc<Mutex<PerformanceStats>>,
    /// System state
    state: Arc<RwLock<SystemState>>,
}

/// System state
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SystemState {
    /// Whether the system is operational
    pub is_operational: bool,
    /// Current thermal state
    pub thermal_state: ThermalState,
    /// Number of safety violations since start
    pub safety_violations: u64,
    /// Last heartbeat timestamp
    pub last_heartbeat: chrono::DateTime<chrono::Utc>,
    /// Current latency in milliseconds
    pub current_latency_ms: f64,
}

/// Performance statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceStats {
    /// Total audio frames processed
    pub frames_processed: u64,
    /// Total source separations performed
    pub separations: u64,
    /// Average processing time per frame (ms)
    pub avg_frame_time_ms: f64,
    /// Maximum processing time (ms)
    pub max_frame_time_ms: f64,
    /// Number of thermal throttling events
    pub thermal_throttle_count: u64,
    /// Number of safety interventions
    pub safety_interventions: u64,
    /// System uptime in seconds
    pub uptime_seconds: u64,
}

impl TechnicalArchitect {
    /// Create a new Technical Architect
    pub async fn new(config: TechArchConfig) -> Result<Self> {
        info!("Initializing Technical Architect with config: {:?}", config);

        // Initialize separator
        let separator = Arc::new(RwLock::new(
            ConvTasNetSeparator::new(config.separator.clone()).await?
        ));

        // Initialize thermal governor
        let thermal = Arc::new(
            ThermalGovernor::new(config.thermal.clone()).await?
        );

        // Initialize safety monitor
        let safety = Arc::new(
            SafetyMonitor::new(config.safety.clone()).await?
        );

        // Initialize synthesizer
        let synthesizer = Arc::new(RwLock::new(
            GranularSynthesizer::new(config.synthesis.clone()).await?
        ));

        // Initialize PTP clock
        let ptp_clock = Arc::new(
            PtpClock::new(config.ptp.clone()).await?
        );

        // Initialize logger
        let logger = Arc::new(
            ProvenanceLogger::new(config.logging.clone()).await?
        );

        let start_time = chrono::Utc::now();

        let architect = Self {
            config,
            separator,
            thermal,
            safety,
            synthesizer,
            ptp_clock,
            logger,
            stats: Arc::new(Mutex::new(PerformanceStats::default())),
            state: Arc::new(RwLock::new(SystemState {
                is_operational: true,
                thermal_state: ThermalState::Normal,
                safety_violations: 0,
                last_heartbeat: start_time,
                current_latency_ms: 0.0,
            })),
        };

        // Start background tasks
        architect.start_background_tasks().await?;

        info!("Technical Architect initialized successfully");
        Ok(architect)
    }

    /// Start background monitoring tasks
    async fn start_background_tasks(&self) -> Result<()> {
        let thermal = self.thermal.clone();
        let safety = self.safety.clone();
        let state = self.state.clone();
        let stats = self.stats.clone();

        // Thermal monitoring task
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(1));
            loop {
                interval.tick().await;
                if let Err(e) = thermal.monitor().await {
                    error!("Thermal monitoring error: {}", e);
                }
            }
        });

        // Safety monitoring task
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(100));
            loop {
                interval.tick().await;
                if let Err(e) = safety.monitor().await {
                    error!("Safety monitoring error: {}", e);
                }
            }
        });

        // Heartbeat task
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(1));
            loop {
                interval.tick().await;
                let mut state = state.write().await;
                state.last_heartbeat = chrono::Utc::now();
            }
        });

        Ok(())
    }

    /// Process an audio frame (main entry point)
    ///
    /// This method processes a noisy audio frame and returns the cleaned audio.
    /// It checks safety and thermal conditions before processing.
    pub async fn process_audio_frame(&self, audio: Vec<f32>) -> Result<Vec<f32>> {
        let start = std::time::Instant::now();

        // Update heartbeat
        {
            let mut state = self.state.write().await;
            state.last_heartbeat = chrono::Utc::now();
        }

        // Check safety
        let safety_check = self.safety.check_safety().await?;
        if !safety_check.is_safe {
            let violation = SafetyViolation {
                violation_type: "SAFETY_CHECK_FAILED".to_string(),
                severity: "CRITICAL".to_string(),
                timestamp: chrono::Utc::now(),
            };
            self.safety.trigger_shutdown(violation).await?;
            return Err(anyhow::anyhow!("Safety check failed"));
        }

        // Check thermal state
        let thermal_state = self.thermal.get_state().await;
        let mut state = self.state.write().await;
        state.thermal_state = thermal_state.clone();

        // If throttling, return simplified processing
        if matches!(thermal_state, ThermalState::Critical | ThermalState::Throttling) {
            warn!("Thermal throttling active, simplifying processing");
            self.stats.lock().thermal_throttle_count += 1;
            return Ok(audio); // Return raw audio
        }

        // Log provenance
        let timestamp = self.ptp_clock.get_timestamp().await?;
        self.logger.log_decision("process_audio_frame", timestamp).await;

        // Run source separation
        let clean_audio = {
            let separator = self.separator.read().await;
            separator.separate(&audio).await?
        };

        // Update statistics
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        {
            let mut stats = self.stats.lock();
            stats.frames_processed += 1;
            stats.separations += 1;
            stats.avg_frame_time_ms = elapsed;
            stats.max_frame_time_ms = stats.max_frame_time_ms.max(elapsed);
        }

        // Update state
        {
            let mut state = self.state.write().await;
            state.current_latency_ms = elapsed;
        }

        // Check latency budget
        if elapsed > self.config.target_latency_ms {
            warn!("Latency budget exceeded: {:.2}ms > {:.2}ms",
                elapsed, self.config.target_latency_ms);
        }

        Ok(clean_audio)
    }

    /// Get current performance statistics
    pub async fn get_stats(&self) -> PerformanceStats {
        self.stats.lock().clone()
    }

    /// Get current system state
    pub async fn get_state(&self) -> SystemState {
        self.state.read().await.clone()
    }

    /// Get thermal state
    pub async fn get_thermal_state(&self) -> ThermalState {
        self.thermal.get_state().await
    }

    /// Get thermal statistics
    pub async fn get_thermal_stats(&self) -> thermal::ThermalStats {
        self.thermal.get_stats().await
    }

    /// Get safety statistics
    pub async fn get_safety_stats(&self) -> SafetyStats {
        self.safety.get_stats().await
    }

    /// Get PTP timestamp
    pub async fn get_ptp_timestamp(&self) -> Result<PtpTimestamp> {
        self.ptp_clock.get_timestamp().await
    }

    /// Get reference to thermal governor (for master controller)
    pub fn get_thermal_governor(&self) -> &Arc<ThermalGovernor> {
        &self.thermal
    }

    /// Get reference to safety monitor (for master controller)
    pub fn get_safety_monitor(&self) -> &Arc<SafetyMonitor> {
        &self.safety
    }

    /// Get reference to PTP clock (for master controller)
    pub fn get_ptp_clock(&self) -> &Arc<PtpClock> {
        &self.ptp_clock
    }

    /// Shutdown the system gracefully
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down Technical Architect");

        {
            let mut state = self.state.write().await;
            state.is_operational = false;
        }

        self.logger.flush().await?;
        self.synthesizer.write().await.shutdown().await?;
        self.ptp_clock.shutdown().await?;

        info!("Technical Architect shutdown complete");
        Ok(())
    }

    /// Emergency mute - immediately silence all audio output
    ///
    /// This is a safety-critical function that:
    /// 1. Immediately sets output gain to zero
    /// 2. Stops any ongoing synthesis
    /// 3. Logs the event with PTP timestamp
    ///
    /// This function must complete in < 1ms to be effective for safety.
    pub async fn emergency_mute(&self) -> Result<()> {
        error!("EMERGENCY MUTE activated");

        // Get PTP timestamp for logging
        let timestamp = self.ptp_clock.get_timestamp().await?;

        // Immediately stop synthesis
        {
            let mut synthesizer = self.synthesizer.write().await;
            synthesizer.emergency_stop()?;
        }

        // Update system state to reflect muted status
        {
            let mut state = self.state.write().await;
            state.current_latency_ms = 0.0; // Reset latency
        }

        // Log the emergency mute event with provenance
        self.logger.log_emergency_event("emergency_mute", timestamp).await?;

        error!("Emergency mute completed at PTP timestamp: {:?}", timestamp);
        Ok(())
    }

    /// Generate a response audio segment
    pub async fn generate_response(&self, features: &synthesis::AudioFeatures) -> Result<Vec<f32>> {
        let synthesizer = self.synthesizer.read().await;
        synthesizer.generate(features).await
    }

    // ========================================================================
    // Enhanced Synthesis Methods
    // ========================================================================

    /// Create an enhanced microharmonic synthesizer for the given species
    pub fn create_microharmonic_synthesizer(
        &self,
        species: String,
        phrase_segments: HashMap<String, synthesis::PhraseSegment>,
    ) -> EnhancedMicroharmonicSynthesizer {
        EnhancedMicroharmonicSynthesizer::new(
            species,
            phrase_segments,
            self.config.synthesis.sample_rate,
        )
    }

    /// Synthesize in horizontal mode (sequential concatenation)
    pub async fn synthesize_horizontal(
        &self,
        synthesizer: &EnhancedMicroharmonicSynthesizer,
        phrase_keys: Vec<String>,
        constraints: Option<&MicroharmonicConstraints>,
    ) -> Result<SynthesisResult> {
        let default_constraints = MicroharmonicConstraints::default();
        let constraints = constraints.unwrap_or(&default_constraints);
        synthesizer.synthesize_horizontal(&phrase_keys, constraints).await
    }

    /// Synthesize in vertical mode (simultaneous layering)
    pub async fn synthesize_vertical(
        &self,
        synthesizer: &EnhancedMicroharmonicSynthesizer,
        phrase_keys: Vec<String>,
        constraints: Option<&MicroharmonicConstraints>,
    ) -> Result<SynthesisResult> {
        let default_constraints = MicroharmonicConstraints::default();
        let constraints = constraints.unwrap_or(&default_constraints);
        synthesizer.synthesize_vertical(&phrase_keys, constraints).await
    }

    /// Synthesize in combined mode (mixed encoding)
    pub async fn synthesize_combined(
        &self,
        synthesizer: &EnhancedMicroharmonicSynthesizer,
        synthesis_plan: Vec<(SynthesisMode, Vec<String>)>,
        constraints: Option<&MicroharmonicConstraints>,
    ) -> Result<SynthesisResult> {
        let default_constraints = MicroharmonicConstraints::default();
        let constraints = constraints.unwrap_or(&default_constraints);
        synthesizer.synthesize_combined(&synthesis_plan, constraints).await
    }
}

// PyO3 Python bindings (when feature is enabled)
#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;

/// Python wrapper for TechnicalArchitect
#[cfg(feature = "python-bindings")]
#[pyclass(name = "TechnicalArchitect")]
pub struct PyTechnicalArchitect {
    inner: Arc<TechnicalArchitect>,
}

/// Python wrapper for Dynamic Microharmonic Synthesizer
#[cfg(feature = "python-bindings")]
#[pyclass(name = "DynamicMicroharmonicSynthesizer")]
pub struct PyDynamicMicroharmonicSynthesizer {
    inner: synthesis::DynamicMicroharmonicSynthesizer,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyDynamicMicroharmonicSynthesizer {
    /// Create a new Dynamic Microharmonic Synthesizer
    #[new]
    fn new(sample_rate: usize) -> Self {
        Self {
            inner: synthesis::DynamicMicroharmonicSynthesizer::new(sample_rate),
        }
    }

    /// Synthesize a single phrase with given parameters
    ///
    /// Parameters:
    /// - f0_base: Fundamental frequency in Hz
    /// - duration_ms: Duration in milliseconds
    /// - attack_ms: Attack time in milliseconds
    /// - decay_ms: Decay time in milliseconds
    /// - sustain_level: Sustain amplitude (0.0 to 1.0)
    /// - vibrato_rate_hz: Vibrato rate in Hz
    /// - vibrato_depth_cents: Vibrato depth in cents
    /// - jitter_amount: Jitter amount (0.0 to 0.1)
    /// - shimmer_amount: Shimmer amount (0.0 to 0.1)
    /// - spectral_tilt: Spectral tilt in dB/octave (negative values)
    /// - hnr_db: Harmonic-to-noise ratio in dB
    ///
    /// Returns: List of audio samples
    fn synthesize_phrase(
        &self,
        f0_base: f32,
        duration_ms: f32,
        attack_ms: f32,
        decay_ms: f32,
        sustain_level: f32,
        vibrato_rate_hz: f32,
        vibrato_depth_cents: f32,
        jitter_amount: f32,
        shimmer_amount: f32,
        spectral_tilt: f32,
        hnr_db: f32,
    ) -> Vec<f32> {
        let params = synthesis::DynamicMicroharmonicParams {
            f0_base,
            duration_ms,
            attack_ms,
            decay_ms,
            sustain_level,
            vibrato_rate_hz,
            vibrato_depth_cents,
            jitter_amount,
            shimmer_amount,
            spectral_tilt,
            hnr_db,
        };

        self.inner.synthesize_phrase(&params)
    }

    /// Synthesize a sequence of phrases (sentence)
    ///
    /// Parameters:
    /// - phrase_params_json: JSON string of list of phrase parameter dicts
    /// - crossfade_ms: Crossfade duration between phrases
    ///
    /// Returns: List of audio samples for the entire sequence
    fn synthesize_sequence(
        &self,
        phrase_params_json: String,
        crossfade_ms: f32,
    ) -> PyResult<Vec<f32>> {
        let phrase_params: Vec<synthesis::DynamicMicroharmonicParams> = serde_json::from_str(&phrase_params_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid parameters JSON: {}", e)))?;

        Ok(self.inner.synthesize_sequence(&phrase_params, crossfade_ms))
    }

    /// Generate random micro-dynamics parameters for exploration
    ///
    /// Parameters:
    /// - f0_base: Target fundamental frequency
    /// - duration_ms: Target duration
    /// - variability: Randomness amount (0.0 to 1.0)
    ///
    /// Returns: JSON string of parameters
    fn generate_random_params(
        &self,
        f0_base: f32,
        duration_ms: f32,
        variability: f32,
    ) -> PyResult<String> {
        let params = self.inner.generate_random_params(f0_base, duration_ms, variability);

        serde_json::to_string(&params)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Serialization failed: {}", e)))
    }

    /// Get default parameters for marmoset vocalizations
    ///
    /// Parameters:
    /// - f0_base: Fundamental frequency in Hz
    /// - duration_ms: Duration in milliseconds
    ///
    /// Returns: JSON string of default marmoset parameters
    fn marmoset_default(&self, f0_base: f32, duration_ms: f32) -> PyResult<String> {
        let params = synthesis::DynamicMicroharmonicParams::marmoset_default(f0_base, duration_ms);

        serde_json::to_string(&params)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Serialization failed: {}", e)))
    }

    /// Get default parameters for bat vocalizations
    ///
    /// Parameters:
    /// - f0_base: Fundamental frequency in Hz
    /// - duration_ms: Duration in milliseconds
    ///
    /// Returns: JSON string of default bat parameters
    fn bat_default(&self, f0_base: f32, duration_ms: f32) -> PyResult<String> {
        let params = synthesis::DynamicMicroharmonicParams::bat_default(f0_base, duration_ms);

        serde_json::to_string(&params)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Serialization failed: {}", e)))
    }
}

/// PyO3 bindings for Granular Concatenative Synthesizer
///
/// High-fidelity synthesizer that preserves formant structure
/// by manipulating real audio samples.
#[cfg(feature = "python-bindings")]
#[pyclass(name = "GranularConcatenativeSynthesizer")]
pub struct PyGranularConcatenativeSynthesizer {
    inner: synthesis::GranularConcatenativeSynthesizer,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyGranularConcatenativeSynthesizer {
    /// Create a new Granular Concatenative Synthesizer
    ///
    /// Parameters:
    /// - sample_rate: Audio sample rate (e.g., 22050)
    #[new]
    fn new(sample_rate: usize) -> Self {
        Self {
            inner: synthesis::GranularConcatenativeSynthesizer::new(sample_rate),
        }
    }

    /// Load source audio buffer (real recording)
    ///
    /// Parameters:
    /// - source: List of audio samples (f32 values)
    fn load_source(&mut self, source: Vec<f32>) {
        self.inner.load_source(source);
    }

    /// Set pitch shift ratio
    ///
    /// Parameters:
    /// - ratio: Pitch shift ratio (0.5 = octave down, 1.0 = natural, 2.0 = octave up)
    fn set_pitch_shift(&mut self, ratio: f32) {
        self.inner.set_pitch_shift(ratio);
    }

    /// Set grain size in milliseconds
    ///
    /// Parameters:
    /// - size_ms: Grain window size (typically 10-50ms)
    fn set_grain_size_ms(&mut self, size_ms: f32) {
        self.inner.set_grain_size_ms(size_ms);
    }

    /// Synthesize audio with specified duration
    ///
    /// This manipulates the loaded source audio using granular synthesis,
    /// preserving formant structure while allowing pitch/time flexibility.
    ///
    /// Parameters:
    /// - duration_ms: Output duration in milliseconds
    ///
    /// Returns: List of synthesized audio samples
    fn synthesize(&mut self, duration_ms: f32) -> Vec<f32> {
        self.inner.synthesize(duration_ms)
    }

    /// Convenience method: Synthesize from file path
    ///
    /// Loads audio from file and synthesizes with given parameters.
    ///
    /// Parameters:
    /// - file_path: Path to audio file (WAV)
    /// - duration_ms: Output duration in milliseconds
    /// - pitch_shift: Pitch shift ratio (default 1.0)
    /// - grain_size_ms: Grain size in milliseconds (default 20.0)
    ///
    /// Returns: List of synthesized audio samples
    fn synthesize_from_file(
        &mut self,
        file_path: String,
        duration_ms: f32,
        pitch_shift: Option<f32>,
        grain_size_ms: Option<f32>,
    ) -> PyResult<Vec<f32>> {
        // Read audio file using soundfile
        use std::path::Path;
        let path = Path::new(&file_path);

        if !path.exists() {
            return Err(pyo3::exceptions::PyFileNotFoundError::new_err(
                format!("Audio file not found: {}", file_path)
            ));
        }

        // For now, return error - we'll need to add proper audio file loading
        // This is a placeholder for the actual implementation
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "synthesize_from_file not yet implemented - use load_source() with pre-loaded audio"
        ))
    }
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyTechnicalArchitect {
    /// Create a new Technical Architect from Python
    #[new]
    fn new(config_json: String) -> PyResult<Self> {
        let config: TechArchConfig = serde_json::from_str(&config_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid config: {}", e)))?;

        // Use tokio runtime
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;

        let inner = rt.block_on(async {
            TechnicalArchitect::new(config).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to initialize: {}", e)))?;

        Ok(Self { inner: Arc::new(inner) })
    }

    /// Process an audio frame from Python
    fn process_audio_frame(&self, audio: Vec<f32>) -> PyResult<Vec<f32>> {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;

        rt.block_on(async {
            self.inner.process_audio_frame(audio).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Processing failed: {}", e)))
    }

    /// Get thermal state as string
    fn get_thermal_state(&self) -> PyResult<String> {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;

        let state = rt.block_on(async {
            self.inner.get_thermal_state().await
        });

        Ok(format!("{:?}", state))
    }

    /// Get statistics as JSON string
    fn get_stats(&self) -> PyResult<String> {
        let stats = self.inner.stats.lock();
        serde_json::to_string(&*stats)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to serialize: {}", e)))
    }
}

#[cfg(feature = "python-bindings")]
#[pymodule]
fn technical_architecture(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTechnicalArchitect>()?;
    m.add_class::<PyDynamicMicroharmonicSynthesizer>()?;
    m.add_class::<PyGranularConcatenativeSynthesizer>()?;
    Ok(())
}

// Re-export for use in other Rust modules
pub use TechArchConfig as Config;

impl TechArchConfig {
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json)
            .map_err(|e| anyhow::anyhow!("Failed to parse TechArchConfig from JSON: {}", e))
    }
}
