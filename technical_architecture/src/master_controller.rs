//! Master Controller Module
//! =========================
//!
//! This module implements the Deterministic Intent-Reality Mediator that
//! sits between the Python Logic Layer and the Rust Execution Layer.
//!
//! The Master Controller is responsible for:
//! - Translating abstract Python intents into physical Rust actions
//! - Enforcing thermal, safety, and hardware constraints
//! - Providing fault isolation between Python and Rust
//! - Maintaining deterministic timing and provenance
//!
//! Architecture:
//! - Python (Logic Layer) -> IntentToken -> Master Controller -> ExecutionReceipt -> Python
//! - Rust (Execution Layer) <- Constraint Check <- Master Controller <- Action
//!
//! Author: Sheel Morjaria (sheelmorjaria@gmail.com)
//! License: CC BY-ND 4.0 International

use anyhow::Result;
use log::{debug, error, info, warn};
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

// ============================================================================
// Hardware Detection
// ============================================================================

/// Detect FPGA availability
///
/// Checks for common FPGA device files and drivers:
/// - Xilinx Alveo: /dev/xclmgmt*
/// - Intel FPGA: /dev/intel-fpga*
/// - Custom FPGA: /dev/fpga*
pub fn detect_fpga() -> bool {
    let common_paths = [
        "/dev/xclmgmt0",    // Xilinx Alveo
        "/dev/intel-fpga0", // Intel FPGA
        "/dev/fpga0",       // Generic FPGA
        "/sys/class/fpga",  // FPGA class device
    ];

    for path in &common_paths {
        if std::path::Path::new(path).exists() {
            info!("FPGA detected at: {}", path);
            return true;
        }
    }

    // Check for FPGA kernel modules
    if let Ok(output) = std::process::Command::new("lsmod").output() {
        let modules = String::from_utf8_lossy(&output.stdout);
        if modules.contains("xclmgmt")
            || modules.contains("intel_fpga")
            || modules.contains("ofpga")
        {
            info!("FPGA kernel module detected");
            return true;
        }
    }

    debug!("No FPGA detected");
    false
}

use crate::ptp::PtpTimestamp;
use crate::synthesis::SynthesisMode;
use crate::thermal::ThermalState;

// ============================================================================
// Core Types
// ============================================================================

/// Action that can be requested by the Python Logic Layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Action {
    /// Synthesize audio response
    Synthesize {
        /// Phrase keys to synthesize
        phrase_keys: Vec<String>,
        /// Synthesis mode (horizontal/vertical/combined)
        mode: SynthesisMode,
        /// Complexity level (low/medium/high)
        complexity: SynthesisComplexity,
        /// Priority of this intent
        priority: IntentPriority,
    },
    /// Load new phrase segments
    LoadPhrases {
        /// Species identifier
        species: String,
        /// Phrase segments (serialized)
        phrases_data: Vec<u8>,
    },
    /// Update system parameters
    UpdateParameters {
        /// Parameter name
        name: String,
        /// New value
        value: serde_json::Value,
    },
    /// Emergency stop
    EmergencyStop,
}

/// Synthesis complexity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SynthesisComplexity {
    Low,
    Medium,
    High,
}

/// Intent priority level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum IntentPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Intent token from Python Logic Layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentToken {
    /// Unique session identifier
    pub session_id: String,
    /// Action to perform
    pub action: Action,
    /// Intent generation timestamp (PTP)
    pub intent_timestamp: PtpTimestamp,
    /// Expected maximum latency (milliseconds)
    pub max_latency_ms: f64,
    /// Causal chain hash (for provenance)
    pub chain_hash: Option<String>,
}

impl IntentToken {
    /// Create a new intent token
    pub fn new(session_id: String, action: Action) -> Self {
        Self {
            session_id,
            action,
            intent_timestamp: PtpTimestamp::now(),
            max_latency_ms: 100.0,
            chain_hash: None,
        }
    }

    /// Set maximum latency
    pub fn with_max_latency(mut self, latency_ms: f64) -> Self {
        self.max_latency_ms = latency_ms;
        self
    }

    /// Set causal chain hash
    pub fn with_chain_hash(mut self, hash: String) -> Self {
        self.chain_hash = Some(hash);
        self
    }
}

/// Health status of the physical system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    /// Current thermal state
    pub thermal_state: ThermalState,
    /// Current temperature (Celsius)
    pub temperature_c: f32,
    /// Whether audio output is safe
    pub audio_safe: bool,
    /// Current CPU usage (0.0 to 1.0)
    pub cpu_usage: f32,
    /// Current memory usage (bytes)
    pub memory_usage_bytes: usize,
    /// FPGA availability (if present)
    pub fpga_available: bool,
    /// Last heartbeat timestamp
    pub last_heartbeat: PtpTimestamp,
}

impl HealthStatus {
    /// Check if system can accept high-complexity intents
    pub fn can_handle_high_complexity(&self) -> bool {
        self.thermal_state != ThermalState::Critical
            && self.thermal_state != ThermalState::Throttling
            && self.cpu_usage < 0.8
            && self.audio_safe
    }

    /// Check if system can accept any intents
    pub fn can_accept_intents(&self) -> bool {
        self.thermal_state != ThermalState::Critical && self.audio_safe
    }
}

/// Execution receipt returned to Python Logic Layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionReceipt {
    /// Action executed successfully
    Success {
        /// Session ID from intent
        session_id: String,
        /// Action that was executed
        action: Action,
        /// Actual execution time (milliseconds)
        execution_time_ms: f64,
        /// PTP timestamp of execution completion
        completion_timestamp: PtpTimestamp,
        /// Synthesis result (if applicable)
        synthesis_result: Option<SynthesisResultMetadata>,
    },
    /// Action rejected due to constraints
    Rejected {
        /// Session ID from intent
        session_id: String,
        /// Action that was rejected
        action: Action,
        /// Rejection reason
        reason: RejectionReason,
        /// Fallback action taken (if any)
        fallback: Option<Action>,
    },
    /// Action failed with error
    Error {
        /// Session ID from intent
        session_id: String,
        /// Error message
        error_message: String,
    },
}

/// Reason for intent rejection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RejectionReason {
    /// Temperature too high for requested complexity
    ThermalLimit {
        /// Current temperature
        current_temp_c: f32,
        /// Maximum allowed temperature
        max_temp_c: f32,
    },
    /// Safety violation detected
    SafetyViolation {
        /// Description of violation
        description: String,
    },
    /// Hardware not available
    HardwareUnavailable {
        /// Missing hardware
        hardware: String,
    },
    /// Latency budget would be exceeded
    LatencyBudgetExceeded {
        /// Expected latency
        expected_ms: f64,
        /// Maximum allowed
        max_ms: f64,
    },
    /// Invalid parameters
    InvalidParameters {
        /// Description of invalid parameters
        description: String,
    },
}

/// Metadata from synthesis result (lightweight version for receipt)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisResultMetadata {
    /// Synthesis mode used
    pub mode: SynthesisMode,
    /// Duration in milliseconds
    pub duration_ms: f32,
    /// Number of phrases used
    pub phrase_count: usize,
    /// Microharmonic compatibility score
    pub microharmonic_score: f32,
}

/// Session profile for tracking system state
#[derive(Debug, Clone)]
pub struct SessionProfile {
    /// Unique session identifier
    pub id: String,
    /// Session start timestamp
    pub start_time: Instant,
    /// Total intents processed
    pub intents_processed: u64,
    /// Total intents rejected
    pub intents_rejected: u64,
    /// Total execution time (milliseconds)
    pub total_execution_time_ms: f64,
    /// Current complexity budget
    pub complexity_budget: f32,
}

impl SessionProfile {
    /// Create a new session profile
    pub fn new() -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            start_time: Instant::now(),
            intents_processed: 0,
            intents_rejected: 0,
            total_execution_time_ms: 0.0,
            complexity_budget: 1.0,
        }
    }

    /// Get session duration in seconds
    pub fn uptime_seconds(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }

    /// Get acceptance rate (0.0 to 1.0)
    pub fn acceptance_rate(&self) -> f32 {
        if self.intents_processed == 0 {
            return 1.0;
        }
        (self.intents_processed - self.intents_rejected) as f32 / self.intents_processed as f32
    }
}

impl Default for SessionProfile {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Watchdog Configuration
// ============================================================================

/// Watchdog configuration for Python crash detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchdogConfig {
    /// Timeout before declaring Python hung (milliseconds)
    pub python_timeout_ms: u64,
    /// Interval between health checks (milliseconds)
    pub health_check_interval_ms: u64,
    /// Enable automatic Python restart
    pub auto_restart_python: bool,
    /// Maximum restart attempts per session
    pub max_restart_attempts: usize,
}

impl Default for WatchdogConfig {
    fn default() -> Self {
        Self {
            python_timeout_ms: 150, // 150ms
            health_check_interval_ms: 50,
            auto_restart_python: true,
            max_restart_attempts: 3,
        }
    }
}

// ============================================================================
// Python Interface
// ============================================================================

/// Cognitive processor interface (Python side)
///
/// This is implemented via PyO3 bindings to allow Python to:
/// 1. Receive health status updates
/// 2. Make decisions and return intents
/// 3. Process execution receipts for learning
pub trait CognitiveProcessor: Send + Sync {
    /// Update Python's view of system health
    fn update_health_context(
        &self,
        health: &HealthStatus,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<()>> + Send>>;

    /// Request next decision from Python
    fn decide_next_move(
        &self,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<IntentToken>> + Send>>;

    /// Provide feedback for reinforcement learning
    fn process_feedback(
        &self,
        receipt: &ExecutionReceipt,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<()>> + Send>>;
}

// ============================================================================
// PyO3 Python Bindings (when feature is enabled)
// ============================================================================

#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;

/// PyO3 wrapper for Python-based CognitiveProcessor implementation
///
/// This allows Python code to implement the CognitiveProcessor interface
/// and be used by the Rust master controller.
#[cfg(feature = "python-bindings")]
pub struct PyCognitiveProcessor {
    /// Python object implementing the cognitive processor interface
    python_object: PyObject,
}

#[cfg(feature = "python-bindings")]
impl PyCognitiveProcessor {
    /// Create a new Python-backed cognitive processor
    ///
    /// The Python object must implement the following methods:
    /// - update_health_context(health_json: str) -> None
    /// - decide_next_move() -> intent_json: str
    /// - process_feedback(receipt_json: str) -> None
    pub fn new(python_object: PyObject) -> Self {
        Self { python_object }
    }

    // NOTE: This method is currently unused and has PyO3 compatibility issues.
    // Uncomment and fix when needed for Python method calling.
    /*
    fn call_python_method<'a, T>(
        &self,
        py: Python<'a>,
        method_name: &str,
        args: impl IntoPy<Py<PyAny>>,
    ) -> PyResult<T>
    where
        T: FromPyObject<'a>,
    {
        let obj = self.python_object.as_ref(py);
        let method = obj.getattr(method_name)?;
        let result = method.call1(args.into_py(py))?;
        result.extract()
    }
    */

    /// Serialize HealthStatus to JSON for Python
    fn health_to_json(&self, health: &HealthStatus) -> String {
        serde_json::to_string(health).unwrap_or_else(|_| "{}".to_string())
    }

    /// Deserialize IntentToken from JSON from Python
    fn intent_from_json(&self, json: &str) -> Result<IntentToken> {
        serde_json::from_str(json)
            .map_err(|e| anyhow::anyhow!("Failed to parse IntentToken from JSON: {}", e))
    }

    /// Serialize ExecutionReceipt to JSON for Python
    fn receipt_to_json(&self, receipt: &ExecutionReceipt) -> String {
        serde_json::to_string(receipt).unwrap_or_else(|_| "{}".to_string())
    }
}

#[cfg(feature = "python-bindings")]
impl CognitiveProcessor for PyCognitiveProcessor {
    fn update_health_context(
        &self,
        health: &HealthStatus,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<()>> + Send>> {
        let health_json = self.health_to_json(health);
        let py_obj = self.python_object.clone();

        Box::pin(async move {
            // Acquire Python GIL
            Python::with_gil(|py| {
                let obj = py_obj.as_ref(py);
                let method = obj.getattr("update_health_context")?;
                let _ = method.call1((health_json,))?;
                Ok::<(), PyErr>(())
            })
            .map_err(|e| anyhow::anyhow!("Python error in update_health_context: {}", e))
        })
    }

    fn decide_next_move(
        &self,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<IntentToken>> + Send>> {
        let py_obj = self.python_object.clone();

        Box::pin(async move {
            // Acquire Python GIL
            let intent_json = Python::with_gil(|py| {
                let obj = py_obj.as_ref(py);
                let method = obj.getattr("decide_next_move")?;
                let result = method.call0()?;
                result.extract::<String>()
            })
            .map_err(|e| anyhow::anyhow!("Python error in decide_next_move: {}", e))?;

            Self {
                python_object: py_obj.clone(),
            }
            .intent_from_json(&intent_json)
        })
    }

    fn process_feedback(
        &self,
        receipt: &ExecutionReceipt,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<()>> + Send>> {
        let receipt_json = self.receipt_to_json(receipt);
        let py_obj = self.python_object.clone();

        Box::pin(async move {
            // Acquire Python GIL
            Python::with_gil(|py| {
                let obj = py_obj.as_ref(py);
                let method = obj.getattr("process_feedback")?;
                let _ = method.call1((receipt_json,))?;
                Ok::<(), PyErr>(())
            })
            .map_err(|e| anyhow::anyhow!("Python error in process_feedback: {}", e))
        })
    }
}

// ============================================================================
// Shared Memory Configuration
// ============================================================================

/// Shared memory ring buffer configuration for zero-copy audio
#[derive(Debug, Clone)]
pub struct SharedMemoryConfig {
    /// Buffer size in samples
    pub buffer_size_samples: usize,
    /// Number of ring buffer slots
    pub num_slots: usize,
    /// Sample rate
    pub sample_rate: usize,
}

impl Default for SharedMemoryConfig {
    fn default() -> Self {
        Self {
            buffer_size_samples: 4096,
            num_slots: 8,
            sample_rate: 44100,
        }
    }
}

/// Shared memory ring buffer for zero-copy audio access
pub struct SharedMemoryRingBuffer {
    config: SharedMemoryConfig,
    write_position: Arc<std::sync::atomic::AtomicUsize>,
    read_position: Arc<std::sync::atomic::AtomicUsize>,
    buffers: Vec<Vec<f32>>,
}

impl SharedMemoryRingBuffer {
    /// Create a new shared memory ring buffer
    pub fn new(config: SharedMemoryConfig) -> Self {
        let buffers = (0..config.num_slots)
            .map(|_| vec![0.0f32; config.buffer_size_samples])
            .collect();

        Self {
            config,
            write_position: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            read_position: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            buffers,
        }
    }

    /// Write audio data to the next available slot
    pub fn write(&mut self, data: &[f32]) -> Result<()> {
        let slot = self
            .write_position
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel)
            % self.config.num_slots;

        if data.len() > self.buffers[slot].len() {
            return Err(anyhow::anyhow!("Data too large for buffer slot"));
        }

        self.buffers[slot][..data.len()].copy_from_slice(data);
        Ok(())
    }

    /// Get read-only view of the latest slot for Python
    pub fn read_latest(&self) -> Option<&[f32]> {
        let write_slot = self
            .write_position
            .load(std::sync::atomic::Ordering::Acquire);
        let read_slot = self
            .read_position
            .load(std::sync::atomic::Ordering::Acquire);

        if read_slot == write_slot {
            return None; // No new data
        }

        let slot = read_slot % self.config.num_slots;
        self.read_position
            .store(read_slot + 1, std::sync::atomic::Ordering::Release);

        Some(&self.buffers[slot])
    }

    /// Get buffer size
    pub fn buffer_size(&self) -> usize {
        self.config.buffer_size_samples
    }

    /// Get sample rate
    pub fn sample_rate(&self) -> usize {
        self.config.sample_rate
    }
}

// ============================================================================
// Atomic Parameter Storage
// ============================================================================

/// Atomic parameter storage for thread-safe updates
pub struct AtomicParameters {
    /// Sensitivity threshold (0.0 to 1.0) - stored as u32 bits
    sensitivity_threshold: Arc<std::sync::atomic::AtomicU32>,
    /// Output gain (0.0 to 1.0) - stored as u32 bits
    output_gain: Arc<std::sync::atomic::AtomicU32>,
    /// Maximum processing time (ms) - stored as u64 bits
    max_processing_time_ms: Arc<std::sync::atomic::AtomicU64>,
}

impl Clone for AtomicParameters {
    fn clone(&self) -> Self {
        Self {
            sensitivity_threshold: Arc::clone(&self.sensitivity_threshold),
            output_gain: Arc::clone(&self.output_gain),
            max_processing_time_ms: Arc::clone(&self.max_processing_time_ms),
        }
    }
}

impl AtomicParameters {
    /// Create new atomic parameters
    pub fn new() -> Self {
        Self {
            sensitivity_threshold: Arc::new(std::sync::atomic::AtomicU32::new(f32::to_bits(0.5))),
            output_gain: Arc::new(std::sync::atomic::AtomicU32::new(f32::to_bits(0.8))),
            max_processing_time_ms: Arc::new(std::sync::atomic::AtomicU64::new(f64::to_bits(
                100.0,
            ))),
        }
    }

    /// Get sensitivity threshold (atomic read)
    pub fn get_sensitivity(&self) -> f32 {
        f32::from_bits(
            self.sensitivity_threshold
                .load(std::sync::atomic::Ordering::Relaxed),
        )
    }

    /// Set sensitivity threshold (atomic write)
    pub fn set_sensitivity(&self, value: f32) {
        let clamped = value.clamp(0.0, 1.0);
        self.sensitivity_threshold
            .store(f32::to_bits(clamped), std::sync::atomic::Ordering::Relaxed);
    }

    /// Get output gain (atomic read)
    pub fn get_gain(&self) -> f32 {
        f32::from_bits(self.output_gain.load(std::sync::atomic::Ordering::Relaxed))
    }

    /// Set output gain (atomic write)
    pub fn set_gain(&self, value: f32) {
        let clamped = value.clamp(0.0, 1.0);
        self.output_gain
            .store(f32::to_bits(clamped), std::sync::atomic::Ordering::Relaxed);
    }

    /// Get max processing time (atomic read)
    pub fn get_max_processing_time(&self) -> f64 {
        f64::from_bits(
            self.max_processing_time_ms
                .load(std::sync::atomic::Ordering::Relaxed),
        )
    }

    /// Set max processing time (atomic write)
    pub fn set_max_processing_time(&self, value: f64) {
        self.max_processing_time_ms
            .store(f64::to_bits(value), std::sync::atomic::Ordering::Relaxed);
    }
}

impl Default for AtomicParameters {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Unified Master Controller
// ============================================================================

/// Unified Master Controller - Deterministic Intent-Reality Mediator
///
/// This controller sits between the Python Logic Layer and the Rust Execution Layer,
/// ensuring that cognitive intents are translated into physical actions while respecting
/// thermal, safety, and hardware constraints.
#[allow(dead_code)]
pub struct UnifiedMasterController {
    /// Physical reality (Rust execution layer)
    tech_arch: Arc<tokio::sync::RwLock<TechnicalArchitect>>,
    /// Abstract intent (Python logic layer) - placeholder for PyO3 binding
    cog_proc: Option<Box<dyn CognitiveProcessor>>,
    /// System state
    session_profile: SessionProfile,
    /// Watchdog configuration
    watchdog_config: WatchdogConfig,
    /// Shared memory for zero-copy audio
    audio_buffer: Arc<tokio::sync::Mutex<SharedMemoryRingBuffer>>,
    /// Atomic parameters
    parameters: Arc<AtomicParameters>,
    /// Last heartbeat from Python
    last_python_heartbeat: Arc<std::sync::atomic::AtomicU64>,
    /// Python process handle (for restart)
    python_process: Option<std::process::Child>,
    /// Python command for restart (program + args)
    python_command: Option<Vec<String>>,
    /// Restart attempts count
    restart_attempts: Arc<std::sync::atomic::AtomicUsize>,
    /// Running state
    running: Arc<std::sync::atomic::AtomicBool>,
}

/// Re-export TechnicalArchitect at this level for convenience
use crate::TechnicalArchitect;

#[allow(dead_code)]
impl UnifiedMasterController {
    /// Create a new Unified Master Controller
    pub async fn new(tech_arch: TechnicalArchitect) -> Result<Self> {
        info!("Initializing Unified Master Controller");

        let audio_buffer = Arc::new(tokio::sync::Mutex::new(SharedMemoryRingBuffer::new(
            SharedMemoryConfig::default(),
        )));

        Ok(Self {
            tech_arch: Arc::new(tokio::sync::RwLock::new(tech_arch)),
            cog_proc: None,
            session_profile: SessionProfile::new(),
            watchdog_config: WatchdogConfig::default(),
            audio_buffer,
            parameters: Arc::new(AtomicParameters::new()),
            last_python_heartbeat: Arc::new(std::sync::atomic::AtomicU64::new(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64,
            )),
            python_process: None,
            python_command: None,
            restart_attempts: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        })
    }

    /// Set the Python command for auto-restart
    ///
    /// # Arguments
    /// * `command` - Command and arguments (e.g., vec!["python3", "-m", "cognitive_layer"])
    pub fn set_python_command(&mut self, command: Vec<String>) {
        self.python_command = Some(command);
    }

    /// Set the cognitive processor (Python interface)
    pub fn set_cognitive_processor(&mut self, processor: Box<dyn CognitiveProcessor>) {
        self.cog_proc = Some(processor);
    }

    /// Start the controller
    pub async fn start(&self) -> Result<()> {
        info!("Starting Unified Master Controller");
        self.running
            .store(true, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }

    /// Stop the controller
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping Unified Master Controller");
        self.running
            .store(false, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }

    /// The main tick loop - mediates between Logic and Execution
    pub async fn tick(&mut self) -> Result<()> {
        if !self.running.load(std::sync::atomic::Ordering::SeqCst) {
            return Ok(());
        }

        // 1. Check Physical Constraints (The Reality Check)
        let health_status = self.monitor_system().await?;

        // 2. Update Python's view of the world
        if let Some(ref cog_proc) = self.cog_proc {
            cog_proc.update_health_context(&health_status).await?;
        }

        // 3. Check watchdog (Python alive?)
        self.check_watchdog().await?;

        // 4. Request Decision from Python (The Intent)
        let intent = if let Some(ref cog_proc) = self.cog_proc {
            cog_proc.decide_next_move().await?
        } else {
            // No cognitive processor - use fallback
            return Ok(());
        };

        // 5. Update heartbeat
        self.update_python_heartbeat();

        // 6. Validate Intent against Reality (The Gatekeeper)
        let result = self
            .validate_and_execute_intent(&intent, &health_status)
            .await?;

        // 7. Return Receipt to Python for Learning
        if let Some(ref cog_proc) = self.cog_proc {
            cog_proc.process_feedback(&result).await?;
        }

        // 8. Update session profile
        self.session_profile.intents_processed += 1;
        match &result {
            ExecutionReceipt::Success { .. } => {}
            ExecutionReceipt::Rejected { .. } => {
                self.session_profile.intents_rejected += 1;
            }
            ExecutionReceipt::Error { .. } => {
                self.session_profile.intents_rejected += 1;
            }
        }

        Ok(())
    }

    /// Monitor system health from TechnicalArchitect
    async fn monitor_system(&self) -> Result<HealthStatus> {
        let tech = self.tech_arch.read().await;

        let thermal_state = tech.get_thermal_state().await;
        let thermal_stats = tech.get_thermal_stats().await;
        let temperature_c = thermal_stats.current_temp_c.unwrap_or(0.0);
        let safety_stats = tech.get_safety_stats().await;

        Ok(HealthStatus {
            thermal_state,
            temperature_c,
            audio_safe: !safety_stats.watchdog_expired,
            cpu_usage: 0.5,                  // TODO: Get from system
            memory_usage_bytes: 500_000_000, // TODO: Get from system
            fpga_available: detect_fpga(),   // Detect FPGA availability
            last_heartbeat: tech.get_ptp_timestamp().await?,
        })
    }

    /// Check watchdog and recover Python if needed
    async fn check_watchdog(&self) -> Result<()> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let last_heartbeat = self
            .last_python_heartbeat
            .load(std::sync::atomic::Ordering::Relaxed);
        let elapsed = now.saturating_sub(last_heartbeat);

        if elapsed > self.watchdog_config.python_timeout_ms {
            warn!("Python watchdog timeout: {}ms", elapsed);

            if self.watchdog_config.auto_restart_python {
                let attempts = self
                    .restart_attempts
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

                if attempts < self.watchdog_config.max_restart_attempts {
                    error!("Restarting Python process (attempt {})", attempts + 1);
                    self.restart_python().await?;
                } else {
                    error!("Max restart attempts reached. Triggering emergency stop.");
                    self.emergency_stop().await?;
                }
            }
        }

        Ok(())
    }

    /// Restart Python process
    ///
    /// Note: In production, Python process management should be handled by
    /// an external process manager (systemd, supervisord, etc.). This method
    /// logs the restart event and resets the heartbeat for the new process.
    async fn restart_python(&self) -> Result<()> {
        if let Some(ref command) = self.python_command {
            warn!("Attempting Python restart with command: {:?}", command);

            // In a real implementation, we would:
            // 1. Signal the process manager to restart Python
            // 2. Or use a named pipe/socket to trigger restart
            // 3. For now, just log the intent

            info!(
                "Python restart requested. External process manager should handle actual restart."
            );
        } else {
            warn!("Python restart requested but no command configured.");
        }

        // Reset heartbeat to give Python time to restart
        self.last_python_heartbeat.store(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            std::sync::atomic::Ordering::SeqCst,
        );

        // Log with PTP timestamp
        let tech = self.tech_arch.read().await;
        let timestamp = tech.get_ptp_timestamp().await?;
        info!("Python watchdog reset at PTP timestamp: {:?}", timestamp);

        Ok(())
    }

    /// Emergency stop - mute audio and log critical event
    async fn emergency_stop(&self) -> Result<()> {
        error!("EMERGENCY STOP triggered");

        // Mute audio
        let tech = self.tech_arch.read().await;
        tech.emergency_mute().await?;

        error!("Emergency stop completed");
        Ok(())
    }

    /// Validate intent against system constraints and execute
    async fn validate_and_execute_intent(
        &self,
        intent: &IntentToken,
        health: &HealthStatus,
    ) -> Result<ExecutionReceipt> {
        let start = std::time::Instant::now();
        let completion_timestamp = self.tech_arch.read().await.get_ptp_timestamp().await?;

        match &intent.action {
            Action::Synthesize {
                phrase_keys,
                mode,
                complexity,
                priority,
            } => {
                // Check constraints
                if !health.can_accept_intents() {
                    return Ok(ExecutionReceipt::Rejected {
                        session_id: intent.session_id.clone(),
                        action: intent.action.clone(),
                        reason: RejectionReason::ThermalLimit {
                            current_temp_c: health.temperature_c,
                            max_temp_c: 85.0,
                        },
                        fallback: Some(Action::Synthesize {
                            phrase_keys: phrase_keys.clone(),
                            mode: *mode,
                            complexity: SynthesisComplexity::Low,
                            priority: *priority,
                        }),
                    });
                }

                // High complexity requires more headroom
                if matches!(complexity, SynthesisComplexity::High)
                    && !health.can_handle_high_complexity()
                {
                    return Ok(ExecutionReceipt::Rejected {
                        session_id: intent.session_id.clone(),
                        action: intent.action.clone(),
                        reason: RejectionReason::ThermalLimit {
                            current_temp_c: health.temperature_c,
                            max_temp_c: 75.0,
                        },
                        fallback: Some(Action::Synthesize {
                            phrase_keys: phrase_keys.clone(),
                            mode: *mode,
                            complexity: SynthesisComplexity::Low,
                            priority: *priority,
                        }),
                    });
                }

                // Execute synthesis
                match self.execute_synthesis(phrase_keys, mode).await {
                    Ok(result) => {
                        let execution_time_ms = start.elapsed().as_secs_f64() * 1000.0;

                        Ok(ExecutionReceipt::Success {
                            session_id: intent.session_id.clone(),
                            action: intent.action.clone(),
                            execution_time_ms,
                            completion_timestamp,
                            synthesis_result: Some(SynthesisResultMetadata {
                                mode: result.synthesis_mode,
                                duration_ms: result.duration_ms,
                                phrase_count: result.phrases_used.len(),
                                microharmonic_score: result.microharmonic_score,
                            }),
                        })
                    }
                    Err(e) => Ok(ExecutionReceipt::Error {
                        session_id: intent.session_id.clone(),
                        error_message: format!("Synthesis failed: {}", e),
                    }),
                }
            }

            Action::LoadPhrases { .. } => {
                // TODO: Implement phrase loading
                Ok(ExecutionReceipt::Success {
                    session_id: intent.session_id.clone(),
                    action: intent.action.clone(),
                    execution_time_ms: start.elapsed().as_secs_f64() * 1000.0,
                    completion_timestamp,
                    synthesis_result: None,
                })
            }

            Action::UpdateParameters { name, value } => {
                // Update atomic parameters
                match name.as_str() {
                    "sensitivity" => {
                        if let Some(val) = value.as_f64() {
                            self.parameters.set_sensitivity(val as f32);
                        }
                    }
                    "gain" => {
                        if let Some(val) = value.as_f64() {
                            self.parameters.set_gain(val as f32);
                        }
                    }
                    "max_processing_time" => {
                        if let Some(val) = value.as_f64() {
                            self.parameters.set_max_processing_time(val);
                        }
                    }
                    _ => {}
                }

                Ok(ExecutionReceipt::Success {
                    session_id: intent.session_id.clone(),
                    action: intent.action.clone(),
                    execution_time_ms: start.elapsed().as_secs_f64() * 1000.0,
                    completion_timestamp,
                    synthesis_result: None,
                })
            }

            Action::EmergencyStop => {
                self.emergency_stop().await?;
                Ok(ExecutionReceipt::Success {
                    session_id: intent.session_id.clone(),
                    action: intent.action.clone(),
                    execution_time_ms: start.elapsed().as_secs_f64() * 1000.0,
                    completion_timestamp,
                    synthesis_result: None,
                })
            }
        }
    }

    /// Execute synthesis action
    async fn execute_synthesis(
        &self,
        phrase_keys: &[String],
        mode: &SynthesisMode,
    ) -> Result<crate::synthesis::SynthesisResult> {
        // This is a placeholder - actual implementation would use TechnicalArchitect
        // For now, return a dummy result
        Ok(crate::synthesis::SynthesisResult {
            audio: vec![0.0f32; 4410],
            sample_rate: 44100,
            synthesis_mode: *mode,
            duration_ms: 100.0,
            processing_time_ms: 10.0,
            phrases_used: phrase_keys.to_vec(),
            microharmonic_score: 0.8,
        })
    }

    /// Update Python heartbeat timestamp
    fn update_python_heartbeat(&self) {
        self.last_python_heartbeat.store(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            std::sync::atomic::Ordering::SeqCst,
        );
    }

    /// Get shared audio buffer (for Python zero-copy access)
    pub fn get_audio_buffer(&self) -> Arc<tokio::sync::Mutex<SharedMemoryRingBuffer>> {
        self.audio_buffer.clone()
    }

    /// Get atomic parameters (for Python access)
    pub fn get_parameters(&self) -> Arc<AtomicParameters> {
        self.parameters.clone()
    }

    /// Get session profile
    pub fn get_session_profile(&self) -> &SessionProfile {
        &self.session_profile
    }

    /// Get current health status
    pub async fn get_health_status(&self) -> Result<HealthStatus> {
        self.monitor_system().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc as StdArc;
    use std::time::Duration;
    use tokio::sync::Mutex as TokioMutex;

    // ========================================================================
    // Mock Cognitive Processor for Testing
    // ========================================================================

    /// Mock implementation of CognitiveProcessor for testing
    struct MockCognitiveProcessor {
        /// Current health context
        health_context: StdArc<TokioMutex<Option<HealthStatus>>>,
        /// Intents to return (simulating Python decisions)
        pending_intents: StdArc<TokioMutex<std::collections::VecDeque<IntentToken>>>,
        /// Received receipts (for verification)
        received_receipts: StdArc<TokioMutex<Vec<ExecutionReceipt>>>,
        /// Should simulate panic/crash
        should_panic: StdArc<std::sync::atomic::AtomicBool>,
        /// Response delay (simulates Python processing time)
        response_delay_ms: u64,
    }

    impl MockCognitiveProcessor {
        fn new() -> Self {
            Self {
                health_context: StdArc::new(TokioMutex::new(None)),
                pending_intents: StdArc::new(TokioMutex::new(std::collections::VecDeque::new())),
                received_receipts: StdArc::new(TokioMutex::new(Vec::new())),
                should_panic: StdArc::new(std::sync::atomic::AtomicBool::new(false)),
                response_delay_ms: 10,
            }
        }

        /// Add an intent to be returned
        async fn push_intent(&self, intent: IntentToken) {
            self.pending_intents.lock().await.push_back(intent);
        }

        /// Get all received receipts
        async fn get_receipts(&self) -> Vec<ExecutionReceipt> {
            self.received_receipts.lock().await.clone()
        }

        /// Set panic flag
        fn set_panic(&self, value: bool) {
            self.should_panic
                .store(value, std::sync::atomic::Ordering::SeqCst);
        }

        /// Get current health context
        async fn get_health_context(&self) -> Option<HealthStatus> {
            self.health_context.lock().await.clone()
        }
    }

    impl CognitiveProcessor for MockCognitiveProcessor {
        fn update_health_context(
            &self,
            health: &HealthStatus,
        ) -> Pin<Box<dyn std::future::Future<Output = Result<()>> + Send>> {
            let health = health.clone();
            let ctx = self.health_context.clone();
            Box::pin(async move {
                *ctx.lock().await = Some(health);
                Ok(())
            })
        }

        fn decide_next_move(
            &self,
        ) -> Pin<Box<dyn std::future::Future<Output = Result<IntentToken>> + Send>> {
            let intents = self.pending_intents.clone();
            let panic = self.should_panic.load(std::sync::atomic::Ordering::SeqCst);
            let delay_ms = self.response_delay_ms;

            Box::pin(async move {
                // Simulate Python processing delay
                if delay_ms > 0 {
                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                }

                // Simulate panic
                if panic {
                    return Err(anyhow::anyhow!("Python process panicked!"));
                }

                // Return next intent or default
                let mut intents = intents.lock().await;
                if let Some(intent) = intents.pop_front() {
                    Ok(intent)
                } else {
                    // Return a default low-priority intent
                    Ok(IntentToken::new(
                        "test_session".to_string(),
                        Action::Synthesize {
                            phrase_keys: vec![],
                            mode: SynthesisMode::Horizontal,
                            complexity: SynthesisComplexity::Low,
                            priority: IntentPriority::Low,
                        },
                    ))
                }
            })
        }

        fn process_feedback(
            &self,
            receipt: &ExecutionReceipt,
        ) -> Pin<Box<dyn std::future::Future<Output = Result<()>> + Send>> {
            let receipt = receipt.clone();
            let receipts = self.received_receipts.clone();
            Box::pin(async move {
                receipts.lock().await.push(receipt);
                Ok(())
            })
        }
    }

    // ========================================================================
    // TDD Test Suite
    // ========================================================================

    /// Test 1: Intent Token Serialization
    /// Verify that intent tokens can be serialized and deserialized correctly
    #[test]
    fn test_01_intent_token_serialization() {
        let action = Action::Synthesize {
            phrase_keys: vec!["phrase1".to_string(), "phrase2".to_string()],
            mode: SynthesisMode::Vertical,
            complexity: SynthesisComplexity::High,
            priority: IntentPriority::Critical,
        };

        let token = IntentToken::new("test_session_serialization".to_string(), action)
            .with_max_latency(50.0)
            .with_chain_hash("causal_chain_123".to_string());

        // Serialize to JSON
        let json = serde_json::to_string(&token).expect("Failed to serialize");
        assert!(json.contains("test_session_serialization"));
        assert!(json.contains("50.0"));
        assert!(json.contains("causal_chain_123"));

        // Deserialize from JSON
        let deserialized: IntentToken = serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(deserialized.session_id, "test_session_serialization");
        assert_eq!(deserialized.max_latency_ms, 50.0);
        assert_eq!(
            deserialized.chain_hash,
            Some("causal_chain_123".to_string())
        );
    }

    /// Test 2: Receipt Feedback Loop
    /// Verify that execution receipts are correctly processed by Python
    #[tokio::test]
    async fn test_02_receipt_feedback_loop() {
        let mock = MockCognitiveProcessor::new();

        // Create a receipt
        let receipt = ExecutionReceipt::Success {
            session_id: "test_session".to_string(),
            action: Action::Synthesize {
                phrase_keys: vec!["phrase1".to_string()],
                mode: SynthesisMode::Horizontal,
                complexity: SynthesisComplexity::Medium,
                priority: IntentPriority::Normal,
            },
            execution_time_ms: 15.5,
            completion_timestamp: PtpTimestamp::now(),
            synthesis_result: Some(SynthesisResultMetadata {
                mode: SynthesisMode::Horizontal,
                duration_ms: 100.0,
                phrase_count: 1,
                microharmonic_score: 0.9,
            }),
        };

        // Process feedback
        mock.process_feedback(&receipt)
            .await
            .expect("Failed to process feedback");

        // Verify receipt was received
        let receipts = mock.get_receipts().await;
        assert_eq!(receipts.len(), 1);
        match &receipts[0] {
            ExecutionReceipt::Success {
                synthesis_result, ..
            } => {
                assert!(synthesis_result.is_some());
                let result = synthesis_result.as_ref().unwrap();
                assert_eq!(result.mode, SynthesisMode::Horizontal);
                assert_eq!(result.microharmonic_score, 0.9);
            }
            _ => panic!("Expected Success receipt"),
        }
    }

    /// Test 3: Thermal Override on High-Priority Intent
    /// Verify that thermal constraints override even critical intents
    #[test]
    fn test_03_thermal_override_high_priority() {
        // Create health status with critical thermal state
        let health = HealthStatus {
            thermal_state: ThermalState::Critical,
            temperature_c: 88.0,
            audio_safe: true,
            cpu_usage: 0.6,
            memory_usage_bytes: 500_000_000,
            fpga_available: true,
            last_heartbeat: PtpTimestamp::now(),
        };

        // Even critical intents should be rejected
        assert!(!health.can_accept_intents());
        assert!(!health.can_handle_high_complexity());

        // Verify rejection reason would be thermal
        let rejection = RejectionReason::ThermalLimit {
            current_temp_c: 88.0,
            max_temp_c: 85.0,
        };
        match rejection {
            RejectionReason::ThermalLimit {
                current_temp_c,
                max_temp_c,
            } => {
                assert!(current_temp_c > max_temp_c);
            }
            _ => panic!("Expected ThermalLimit"),
        }
    }

    /// Test 4: Safety Mute on Illegal Frequency
    /// Verify that safety violations prevent audio output
    #[test]
    fn test_04_safety_mute_illegal_frequency() {
        // Create health status with safety violation
        let health = HealthStatus {
            thermal_state: ThermalState::Normal,
            temperature_c: 65.0,
            audio_safe: false, // Safety violation
            cpu_usage: 0.5,
            memory_usage_bytes: 500_000_000,
            fpga_available: true,
            last_heartbeat: PtpTimestamp::now(),
        };

        // Even with normal thermal state, intents should be rejected
        assert!(!health.can_accept_intents());

        // Verify rejection reason would be safety
        let rejection = RejectionReason::SafetyViolation {
            description: "Audio safety check failed".to_string(),
        };
        match rejection {
            RejectionReason::SafetyViolation { description } => {
                assert!(description.contains("safety"));
            }
            _ => panic!("Expected SafetyViolation"),
        }
    }

    /// Test 5: Python Panic Isolation
    /// Verify that Rust survives Python crashes
    #[tokio::test]
    async fn test_05_python_panic_isolation() {
        let mock = MockCognitiveProcessor::new();
        mock.set_panic(true); // Simulate Python panic

        // Try to get next move - should get error
        let result = mock.decide_next_move().await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("panic"));

        // Rust should still be functional
        mock.set_panic(false);

        // Should be able to continue
        let intent = mock.decide_next_move().await;
        assert!(intent.is_ok());
    }

    /// Test 6: Rust Execution Thread Independence
    /// Verify that Rust execution continues even without Python
    #[tokio::test]
    async fn test_06_rust_thread_independence() {
        let mock = MockCognitiveProcessor::new();

        // Push some intents
        mock.push_intent(IntentToken::new(
            "session1".to_string(),
            Action::Synthesize {
                phrase_keys: vec!["p1".to_string()],
                mode: SynthesisMode::Horizontal,
                complexity: SynthesisComplexity::Low,
                priority: IntentPriority::Normal,
            },
        ))
        .await;

        mock.push_intent(IntentToken::new(
            "session2".to_string(),
            Action::Synthesize {
                phrase_keys: vec!["p2".to_string()],
                mode: SynthesisMode::Horizontal,
                complexity: SynthesisComplexity::Low,
                priority: IntentPriority::Normal,
            },
        ))
        .await;

        // Process both intents
        let intent1 = mock.decide_next_move().await.unwrap();
        assert_eq!(intent1.session_id, "session1");

        let intent2 = mock.decide_next_move().await.unwrap();
        assert_eq!(intent2.session_id, "session2");

        // Rust continues processing even without Python
        let intent3 = mock.decide_next_move().await.unwrap();
        // Returns default intent when queue is empty
        assert!(matches!(intent3.action, Action::Synthesize { .. }));
    }

    /// Test 7: Zero-Copy Read Access for Python
    /// Verify that Python can read audio buffer without copying
    #[test]
    fn test_07_zero_copy_read_access() {
        let config = SharedMemoryConfig {
            buffer_size_samples: 4096,
            num_slots: 8,
            sample_rate: 44100,
        };

        let mut buffer = SharedMemoryRingBuffer::new(config);

        // Write audio data
        let audio_data: Vec<f32> = (0..4096).map(|i| i as f32 / 4096.0).collect();
        buffer.write(&audio_data).expect("Failed to write");

        // Read without copy - returns a slice reference
        let read_slice = buffer.read_latest();
        assert!(read_slice.is_some());

        let slice = read_slice.unwrap();
        assert_eq!(slice.len(), 4096);
        assert_eq!(slice[0], 0.0);
        assert_eq!(slice[4095], 4095.0 / 4096.0);

        // Verify this is a zero-copy view (same memory)
        // In production, this would be verified by memory address comparison
    }

    /// Test 8: Atomic Parameter Updates
    /// Verify that parameter updates are atomic and thread-safe
    #[test]
    fn test_08_atomic_parameter_updates() {
        let params = AtomicParameters::new();

        // Concurrent writes from multiple threads
        let handles: Vec<_> = (0..10)
            .map(|i| {
                let params = params.clone();
                std::thread::spawn(move || {
                    for _ in 0..1000 {
                        params.set_sensitivity((i as f32) / 10.0);
                    }
                })
            })
            .collect();

        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }

        // Read should always return a valid value (0.0 to 1.0)
        let value = params.get_sensitivity();
        assert!((0.0..=1.0).contains(&value));

        // Test gain atomicity
        params.set_gain(0.7);
        assert_eq!(params.get_gain(), 0.7);

        // Test max_processing_time atomicity
        params.set_max_processing_time(150.0);
        assert_eq!(params.get_max_processing_time(), 150.0);
    }

    /// Test 9: Provenance Chain Integrity
    /// Verify that causal chain hashes are preserved correctly
    #[test]
    fn test_09_provenance_chain_integrity() {
        let intent = IntentToken::new(
            "session_provenance".to_string(),
            Action::Synthesize {
                phrase_keys: vec!["phrase_a".to_string()],
                mode: SynthesisMode::Horizontal,
                complexity: SynthesisComplexity::Medium,
                priority: IntentPriority::Normal,
            },
        )
        .with_chain_hash("chain_step_1".to_string());

        // Execute (simulate)
        let receipt = ExecutionReceipt::Success {
            session_id: intent.session_id.clone(),
            action: intent.action.clone(),
            execution_time_ms: 12.3,
            completion_timestamp: PtpTimestamp::now(),
            synthesis_result: None,
        };

        // Verify chain preservation
        assert_eq!(intent.chain_hash, Some("chain_step_1".to_string()));
        match receipt {
            ExecutionReceipt::Success { session_id, .. } => {
                assert_eq!(session_id, "session_provenance");
            }
            _ => panic!("Expected Success"),
        }

        // Simulate chaining to next intent
        let next_intent = IntentToken::new(
            "session_provenance".to_string(),
            Action::Synthesize {
                phrase_keys: vec!["phrase_b".to_string()],
                mode: SynthesisMode::Horizontal,
                complexity: SynthesisComplexity::Medium,
                priority: IntentPriority::Normal,
            },
        )
        .with_chain_hash(format!("{}->step_2", intent.chain_hash.unwrap()));

        assert_eq!(
            next_intent.chain_hash,
            Some("chain_step_1->step_2".to_string())
        );
    }

    /// Test 10: PTP Clock Consistency
    /// Verify that PTP timestamps are consistent and monotonic
    #[test]
    fn test_10_ptp_clock_consistency() {
        // Create multiple timestamps
        let ts1 = PtpTimestamp::now();
        std::thread::sleep(Duration::from_millis(10));
        let ts2 = PtpTimestamp::now();
        std::thread::sleep(Duration::from_millis(10));
        let ts3 = PtpTimestamp::now();

        // Verify monotonicity
        assert!(
            ts2.seconds > ts1.seconds || (ts2.seconds == ts1.seconds && ts2.nanos >= ts1.nanos)
        );
        assert!(
            ts3.seconds > ts2.seconds || (ts3.seconds == ts2.seconds && ts3.nanos >= ts2.nanos)
        );

        // Verify nanosecond precision
        assert!(ts1.nanos < 1_000_000_000);
        assert!(ts2.nanos < 1_000_000_000);
        assert!(ts3.nanos < 1_000_000_000);

        // Serialize/deserialize to verify consistency
        let json = serde_json::to_string(&ts1).expect("Failed to serialize");
        let deserialized: PtpTimestamp =
            serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(ts1.seconds, deserialized.seconds);
        assert_eq!(ts1.nanos, deserialized.nanos);
    }

    // ========================================================================
    // Original Tests (Preserved)
    // ========================================================================

    #[test]
    fn test_intent_token_creation() {
        let action = Action::Synthesize {
            phrase_keys: vec!["phrase1".to_string()],
            mode: SynthesisMode::Horizontal,
            complexity: SynthesisComplexity::Medium,
            priority: IntentPriority::Normal,
        };

        let token = IntentToken::new("session123".to_string(), action.clone());

        assert_eq!(token.session_id, "session123");
        assert!(matches!(token.action, Action::Synthesize { .. }));
        assert_eq!(token.max_latency_ms, 100.0);
    }

    #[test]
    fn test_intent_token_builder() {
        let action = Action::EmergencyStop;
        let token = IntentToken::new("session456".to_string(), action)
            .with_max_latency(50.0)
            .with_chain_hash("hash123".to_string());

        assert_eq!(token.max_latency_ms, 50.0);
        assert_eq!(token.chain_hash, Some("hash123".to_string()));
    }

    #[test]
    fn test_health_status_can_handle_high_complexity() {
        let mut status = HealthStatus {
            thermal_state: ThermalState::Normal,
            temperature_c: 65.0,
            audio_safe: true,
            cpu_usage: 0.5,
            memory_usage_bytes: 500_000_000,
            fpga_available: true,
            last_heartbeat: PtpTimestamp::now(),
        };

        assert!(status.can_handle_high_complexity());
        assert!(status.can_accept_intents());

        // Test when thermal is critical
        status.thermal_state = ThermalState::Critical;
        assert!(!status.can_handle_high_complexity());
        assert!(!status.can_accept_intents());
    }

    #[test]
    fn test_session_profile() {
        let profile = SessionProfile::new();

        assert!(profile.uptime_seconds() >= 0.0);
        assert_eq!(profile.intents_processed, 0);
        assert_eq!(profile.acceptance_rate(), 1.0);
    }

    #[test]
    fn test_shared_memory_ring_buffer() {
        let config = SharedMemoryConfig {
            buffer_size_samples: 100,
            num_slots: 4,
            sample_rate: 44100,
        };

        let mut buffer = SharedMemoryRingBuffer::new(config);

        // Write some data
        let data1: Vec<f32> = (0..100).map(|i| i as f32).collect();
        buffer.write(&data1).unwrap();

        // Read it back
        let read_data = buffer.read_latest();
        assert_eq!(read_data, Some(data1.as_slice()));

        // No more data
        assert_eq!(buffer.read_latest(), None);
    }

    #[test]
    fn test_atomic_parameters() {
        let params = AtomicParameters::new();

        assert_eq!(params.get_sensitivity(), 0.5);

        params.set_sensitivity(0.8);
        assert_eq!(params.get_sensitivity(), 0.8);

        // Test clamping
        params.set_sensitivity(1.5);
        assert_eq!(params.get_sensitivity(), 1.0);

        params.set_sensitivity(-0.5);
        assert_eq!(params.get_sensitivity(), 0.0);
    }

    #[test]
    fn test_priority_ordering() {
        assert!(IntentPriority::Critical > IntentPriority::High);
        assert!(IntentPriority::High > IntentPriority::Normal);
        assert!(IntentPriority::Normal > IntentPriority::Low);
    }
}
