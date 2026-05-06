//! Peer-to-Peer Controller Module
//! ===============================
//!
//! This module implements the "Supervisor Tree" architecture where:
//! - Rust (Field System) and Python (Cognitive Agent) are independent processes
//! - Systemd/supervisord manages process lifecycle
//! - ZeroMQ handles heartbeat monitoring and control messages
//! - Rust fails open to safety (Passthrough Mode) when Python is unavailable
//!
//! Architecture:
//! - Rust binds to ZeroMQ SUB socket for heartbeats
//! - Python connects and sends heartbeats
//! - Rust detects presence/absence and switches modes accordingly
//!
//! Author: Sheel Morjaria (sheelmorjaria@gmail.com)
//! License: CC BY-ND 4.0 International

use anyhow::{Context, Result};
use log::{error, info, warn};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

// ============================================================================
// Operation Modes
// ============================================================================

/// Operation mode of the field system
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationMode {
    /// Passthrough Mode - Safe default when Python is unavailable
    /// - Recording raw audio
    /// - Passive monitoring
    /// - No synthesis output
    Passthrough,

    /// Interactive Mode - Active when Python is connected and healthy
    /// - Processing intents from Python
    /// - Synthesizing responses
    /// - Full cognitive interaction
    Interactive,
}

/// Audio mute state for safety
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioMuteState {
    /// Audio output is active (but may be at zero gain)
    Active,
    /// Audio output is muted (safety-critical)
    Muted,
}

// ============================================================================
// Peer Controller Configuration
// ============================================================================

/// Configuration for the peer controller
#[derive(Debug, Clone)]
pub struct PeerControllerConfig {
    /// ZeroMQ endpoint for heartbeat socket
    pub heartbeat_endpoint: String,

    /// Heartbeat timeout in milliseconds
    pub heartbeat_timeout_ms: u64,

    /// How often to check for heartbeats
    pub poll_interval_ms: u64,

    /// Enable detailed logging
    pub verbose_logging: bool,
}

impl Default for PeerControllerConfig {
    fn default() -> Self {
        Self {
            heartbeat_endpoint: "ipc:///tmp/cognitive_heartbeat.ipc".to_string(),
            heartbeat_timeout_ms: 100, // 100ms timeout
            poll_interval_ms: 10,      // 10ms poll interval
            verbose_logging: false,
        }
    }
}

// ============================================================================
// Heartbeat Message
// ============================================================================

/// Heartbeat message from Python to Rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatMessage {
    /// Timestamp when heartbeat was sent (Python time)
    pub timestamp: u64,

    /// Sequence number for detecting missed heartbeats
    pub sequence: u64,

    /// Python process ID
    pub pid: u32,

    /// Current state of Python agent
    pub state: String,
}

impl HeartbeatMessage {
    /// Create a new heartbeat message
    ///
    /// # Errors
    /// Returns an error if the system clock is set before Unix epoch
    pub fn new(sequence: u64, pid: u32) -> Result<Self> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .context("system clock is before Unix epoch")?
            .as_millis() as u64;

        Ok(Self {
            timestamp,
            sequence,
            pid,
            state: "active".to_string(),
        })
    }

    /// Serialize to JSON bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(|e| anyhow::anyhow!("Failed to serialize heartbeat: {}", e))
    }

    /// Deserialize from JSON bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        serde_json::from_slice(bytes).map_err(|e| anyhow::anyhow!("Failed to deserialize heartbeat: {}", e))
    }
}

// ============================================================================
// Peer Controller
// ============================================================================

/// Peer-to-Peer Controller
///
/// Manages the connection to the Python Cognitive Agent and switches
/// between Passthrough and Interactive modes based on heartbeat status.
pub struct PeerController {
    /// ZeroMQ context
    #[allow(dead_code)]
    ctx: zmq::Context,

    /// Heartbeat subscriber socket
    heartbeat_sock: zmq::Socket,

    /// Current operation mode
    mode: OperationMode,

    /// Audio mute state
    audio_mute: AudioMuteState,

    /// Whether Python agent is connected
    python_alive: bool,

    /// Last heartbeat timestamp
    last_heartbeat: Option<Instant>,

    /// Last heartbeat sequence number
    last_sequence: u64,

    /// Configuration
    config: PeerControllerConfig,
}

impl PeerController {
    /// Create a new peer controller
    ///
    /// # Arguments
    /// * `config` - Configuration for the controller
    ///
    /// # Returns
    /// * `Result<Self>` - The controller or an error
    ///
    /// # Example
    /// ```no_run
    /// use technical_architecture::{PeerController, PeerControllerConfig};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let config = PeerControllerConfig::default();
    /// let controller = PeerController::new(config)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(config: PeerControllerConfig) -> Result<Self> {
        info!(
            "Initializing Peer Controller with endpoint: {}",
            config.heartbeat_endpoint
        );

        let ctx = zmq::Context::new();
        let sock = ctx.socket(zmq::SUB)?;

        // Bind to heartbeat endpoint (Python will connect)
        sock.bind(&config.heartbeat_endpoint)?;

        // Subscribe to all messages
        sock.set_subscribe(b"")?;

        info!("Peer Controller bound to: {}", config.heartbeat_endpoint);
        info!("Starting in Passthrough Mode (Safe Default)");

        Ok(Self {
            ctx,
            heartbeat_sock: sock,
            mode: OperationMode::Passthrough,
            audio_mute: AudioMuteState::Muted, // Start muted for safety
            python_alive: false,
            last_heartbeat: None,
            last_sequence: 0,
            config,
        })
    }

    /// Get the configuration
    ///
    /// # Returns
    /// * `&PeerControllerConfig` - Reference to the configuration
    pub fn get_config(&self) -> &PeerControllerConfig {
        &self.config
    }

    /// Main tick loop - call this periodically
    ///
    /// This method:
    /// 1. Polls for heartbeat (non-blocking)
    /// 2. Updates state based on heartbeat presence
    /// 3. Returns current operation mode
    ///
    /// # Returns
    /// * `Result<OperationMode>` - Current operation mode
    pub fn tick(&mut self) -> Result<OperationMode> {
        // 1. Poll for heartbeat (non-blocking)
        let _has_heartbeat = self.poll_heartbeat()?;

        // 2. Check for timeout
        self.check_timeout();

        // 3. Update mode based on state
        self.update_mode();

        Ok(self.mode)
    }

    /// Poll for heartbeat (non-blocking)
    fn poll_heartbeat(&mut self) -> Result<bool> {
        // Try to receive with timeout of 0 (non-blocking)
        match self.heartbeat_sock.recv_bytes(zmq::DONTWAIT) {
            Ok(bytes) => match HeartbeatMessage::from_bytes(&bytes) {
                Ok(heartbeat) => {
                    self.handle_heartbeat(heartbeat);
                    Ok(true)
                }
                Err(e) => {
                    warn!("Received invalid heartbeat: {}", e);
                    Ok(false)
                }
            },
            Err(zmq::Error::EAGAIN) => {
                // No message available (expected for non-blocking)
                Ok(false)
            }
            Err(e) => {
                // Socket error (likely disconnected)
                warn!("Heartbeat socket error: {}", e);
                self.handle_disconnect();
                Ok(false)
            }
        }
    }

    /// Handle received heartbeat
    fn handle_heartbeat(&mut self, heartbeat: HeartbeatMessage) {
        let now = Instant::now();

        // Check for sequence jump (detect missed heartbeats)
        if self.last_sequence > 0 && heartbeat.sequence > self.last_sequence + 1 {
            warn!("Missed {} heartbeats", heartbeat.sequence - self.last_sequence - 1);
        }

        self.last_sequence = heartbeat.sequence;
        self.last_heartbeat = Some(now);

        if !self.python_alive {
            info!("⚡ Cognitive Agent (Python) RECONNECTED - PID: {}", heartbeat.pid);
            self.python_alive = true;
            self.audio_mute = AudioMuteState::Active;
            info!("Switching to Interactive Mode");
        }

        if self.config.verbose_logging {
            info!("Heartbeat received: seq={}, pid={}", heartbeat.sequence, heartbeat.pid);
        }
    }

    /// Check for heartbeat timeout
    fn check_timeout(&mut self) {
        if let Some(last) = self.last_heartbeat {
            if last.elapsed() > Duration::from_millis(self.config.heartbeat_timeout_ms) {
                self.handle_timeout();
            }
        }
    }

    /// Handle heartbeat timeout
    fn handle_timeout(&mut self) {
        if self.python_alive {
            warn!("Heartbeat timeout - Python agent appears frozen");
            self.handle_disconnect();
        }
    }

    /// Handle disconnection
    fn handle_disconnect(&mut self) {
        if self.python_alive {
            error!("❌ Cognitive Agent (Python) LOST - Muting Audio");
            self.python_alive = false;
            self.audio_mute = AudioMuteState::Muted;
            self.last_heartbeat = None;
            self.last_sequence = 0;
            info!("Switching to Passthrough Mode");
        }
    }

    /// Update operation mode based on state
    fn update_mode(&mut self) {
        let new_mode = if self.python_alive {
            OperationMode::Interactive
        } else {
            OperationMode::Passthrough
        };

        if new_mode != self.mode {
            info!("Mode change: {:?} -> {:?}", self.mode, new_mode);
            self.mode = new_mode;
        }
    }

    /// Get current operation mode
    pub fn mode(&self) -> OperationMode {
        self.mode
    }

    /// Get audio mute state
    pub fn audio_mute(&self) -> AudioMuteState {
        self.audio_mute
    }

    /// Check if Python agent is alive
    pub fn is_python_alive(&self) -> bool {
        self.python_alive
    }

    /// Get time since last heartbeat
    pub fn time_since_last_heartbeat(&self) -> Option<Duration> {
        self.last_heartbeat.map(|t| t.elapsed())
    }

    /// Get heartbeat endpoint string
    pub fn heartbeat_endpoint(&self) -> &str {
        &self.config.heartbeat_endpoint
    }
}

// ============================================================================
// Feature Event Types (for Rust → Python streaming)
// ============================================================================

/// Feature extraction event sent to Python Logic Layer
///
/// This struct carries the output of Stage 1 (NBD) and Stage 2 (112D Feature Extraction)
/// to the Python Cognitive Agent for decision-making.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureEvent {
    /// Event type identifier (always "feature_extraction")
    pub event_type: String,

    /// Cluster ID from corpus analysis (k=1020)
    pub cluster_id: u32,

    /// 112D feature vector from RosettaFeatures
    pub features_112d: Vec<f32>,

    /// Unix timestamp in seconds
    pub timestamp: f64,

    /// Sequence number for ordering and gap detection
    pub sequence: u64,

    /// Emitter identity from vocalization source separation
    /// None when identity is unknown or not yet resolved
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub emitter_id: Option<i32>,

    /// Confidence score from Student inference (0-1)
    /// None when using original cluster_id (not BGMM-distilled)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}

impl FeatureEvent {
    /// Create a new feature event
    ///
    /// # Arguments
    /// * `cluster_id` - Cluster ID from corpus analysis
    /// * `features_112d` - 112D feature vector (must have 112 elements)
    /// * `sequence` - Sequence number for ordering
    ///
    /// # Returns
    /// * `Result<Self>` - The event or an error if features dimension is wrong
    pub fn new(cluster_id: u32, features_112d: Vec<f32>, sequence: u64) -> Result<Self> {
        if features_112d.len() != 112 {
            anyhow::bail!("Feature vector must have 112 elements, got {}", features_112d.len());
        }

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);

        Ok(Self {
            event_type: "feature_extraction".to_string(),
            cluster_id,
            features_112d,
            timestamp,
            sequence,
            emitter_id: None,
            confidence: None,
        })
    }

    /// Create a feature event with emitter identity
    pub fn with_emitter(mut self, emitter_id: i32) -> Self {
        self.emitter_id = Some(emitter_id);
        self
    }

    /// Create a feature event with confidence score
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence);
        self
    }

    /// Create a feature event from an existing array
    ///
    /// # Arguments
    /// * `cluster_id` - Cluster ID from corpus analysis
    /// * `features_112d` - 112D feature array
    /// * `sequence` - Sequence number for ordering
    pub fn from_array(cluster_id: u32, features_112d: [f32; 112], sequence: u64) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);

        Self {
            event_type: "feature_extraction".to_string(),
            cluster_id,
            features_112d: features_112d.to_vec(),
            timestamp,
            sequence,
            emitter_id: None,
            confidence: None,
        }
    }

    /// Create a test event with zero features
    #[cfg(test)]
    pub fn test_event(cluster_id: u32, sequence: u64) -> Self {
        Self::from_array(cluster_id, [0.0f32; 112], sequence)
    }

    /// Serialize to JSON bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(|e| anyhow::anyhow!("Failed to serialize feature event: {}", e))
    }

    /// Deserialize from JSON bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        serde_json::from_slice(bytes).map_err(|e| anyhow::anyhow!("Failed to deserialize feature event: {}", e))
    }
}

impl std::fmt::Display for FeatureEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "FeatureEvent(cluster={}, seq={}, time={:.3})",
            self.cluster_id, self.sequence, self.timestamp
        )
    }
}

// ============================================================================
// Feature Event Publisher Configuration
// ============================================================================

/// Configuration for feature event publisher
#[derive(Debug, Clone)]
pub struct EventPublisherConfig {
    /// ZeroMQ endpoint for feature events
    pub event_endpoint: String,

    /// High water mark for outbound messages
    pub send_high_water_mark: i32,

    /// Enable verbose logging
    pub verbose_logging: bool,
}

impl Default for EventPublisherConfig {
    fn default() -> Self {
        Self {
            event_endpoint: "ipc:///tmp/cognitive_features.ipc".to_string(),
            send_high_water_mark: 100,
            verbose_logging: false,
        }
    }
}

// ============================================================================
// Feature Event Publisher
// ============================================================================

/// Publisher for feature events to Python Logic Layer
///
/// Uses ZeroMQ PUB socket to stream feature extraction events
/// to the Python Cognitive Agent.
pub struct FeatureEventPublisher {
    /// ZeroMQ context
    #[allow(dead_code)]
    ctx: zmq::Context,

    /// Publisher socket
    sock: zmq::Socket,

    /// Configuration
    config: EventPublisherConfig,

    /// Sequence counter
    sequence: u64,

    /// Events published counter
    events_published: u64,
}

impl FeatureEventPublisher {
    /// Create a new feature event publisher
    ///
    /// # Arguments
    /// * `config` - Publisher configuration
    ///
    /// # Returns
    /// * `Result<Self>` - The publisher or an error
    pub fn new(config: EventPublisherConfig) -> Result<Self> {
        info!("Initializing Feature Event Publisher on: {}", config.event_endpoint);

        let ctx = zmq::Context::new();
        let sock = ctx.socket(zmq::PUB)?;

        // Set high water mark
        sock.set_sndhwm(config.send_high_water_mark)?;

        // Bind to endpoint (Python will connect)
        sock.bind(&config.event_endpoint)?;

        info!("Feature Event Publisher bound to: {}", config.event_endpoint);

        Ok(Self {
            ctx,
            sock,
            config,
            sequence: 0,
            events_published: 0,
        })
    }

    /// Publish a feature event
    ///
    /// # Arguments
    /// * `event` - The feature event to publish
    ///
    /// # Returns
    /// * `Result<()>` - Success or error
    pub fn publish(&mut self, event: &FeatureEvent) -> Result<()> {
        let bytes = event.to_bytes()?;
        self.sock.send(&bytes, zmq::DONTWAIT)?;

        self.events_published += 1;

        if self.config.verbose_logging && self.events_published.is_multiple_of(100) {
            info!("Published {} feature events", self.events_published);
        }

        Ok(())
    }

    /// Create and publish a feature event in one step
    ///
    /// # Arguments
    /// * `cluster_id` - Cluster ID from corpus analysis
    /// * `features_112d` - 112D feature vector
    ///
    /// # Returns
    /// * `Result<()>` - Success or error
    pub fn publish_features(&mut self, cluster_id: u32, features_112d: Vec<f32>) -> Result<()> {
        self.sequence += 1;
        let event = FeatureEvent::new(cluster_id, features_112d, self.sequence)?;
        self.publish(&event)
    }

    /// Create and publish a feature event with emitter identity
    ///
    /// # Arguments
    /// * `cluster_id` - Cluster ID from corpus analysis
    /// * `features_112d` - 112D feature vector
    /// * `emitter_id` - Emitter identity from source separation
    ///
    /// # Returns
    /// * `Result<()>` - Success or error
    pub fn publish_features_with_emitter(
        &mut self,
        cluster_id: u32,
        features_112d: Vec<f32>,
        emitter_id: i32,
    ) -> Result<()> {
        self.sequence += 1;
        let event = FeatureEvent::new(cluster_id, features_112d, self.sequence)?.with_emitter(emitter_id);
        self.publish(&event)
    }

    /// Publish features with Student (BGMM-distilled) cluster assignment
    ///
    /// This is the closed-loop integration: the Student overrides the raw cluster_id
    /// with the BGMM-distilled cluster_id and applies OOD filtering.
    ///
    /// # Arguments
    /// * `features_112d` - 112D feature vector
    /// * `emitter_id` - Optional emitter identity
    /// * `exemplar_manager` - The ExemplarManager with loaded centroids from BGMM Teacher
    ///
    /// # Returns
    /// * `Result<Option<u64>>` - Some(sequence) if published, None if OOD-rejected
    ///
    /// # Behavior
    /// - Finds nearest centroid using Student inference
    /// - Rejects if distance exceeds OOD threshold (returns None)
    /// - Publishes with corrected cluster_id and confidence score
    pub fn publish_with_student(
        &mut self,
        features_112d: Vec<f32>,
        emitter_id: Option<i32>,
        exemplar_manager: &crate::semantic_reconstruction::ExemplarManager,
    ) -> Result<Option<u64>> {
        // Convert Vec to array for Student lookup
        let features_array: [f32; 112] = features_112d
            .as_slice()
            .try_into()
            .map_err(|_| anyhow::anyhow!("Feature vector must have exactly 112 elements"))?;

        // Student inference: find nearest centroid with OOD check
        match exemplar_manager.find_nearest_centroid_with_ood_check(&features_array) {
            Some((cluster_id, distance)) => {
                // Accepted: within BGMM-defined acoustic space
                self.sequence += 1;

                // Calculate confidence from normalized distance (0-1)
                // distance=0 → confidence=1.0, distance=threshold → confidence=0.0
                let threshold = exemplar_manager.ood_threshold();
                let confidence = if threshold > 0.0 {
                    1.0 - (distance / threshold)
                } else {
                    1.0
                }.max(0.0).min(1.0);

                let mut event = FeatureEvent::new(cluster_id, features_112d, self.sequence)?
                    .with_confidence(confidence);

                if let Some(emitter) = emitter_id {
                    event = event.with_emitter(emitter);
                }

                self.publish(&event)?;
                Ok(Some(self.sequence))
            }
            None => {
                // Rejected: OOD - feature doesn't belong to any BGMM cluster
                // This is the Safety-Critical Perception Filter in action!
                log::debug!("Student rejected OOD feature - dropped event");
                Ok(None)
            }
        }
    }

    /// Check if publisher is ready
    pub fn is_ready(&self) -> bool {
        true // Socket is bound
    }

    /// Get the endpoint
    pub fn endpoint(&self) -> &str {
        &self.config.event_endpoint
    }

    /// Get statistics
    pub fn stats(&self) -> EventPublisherStats {
        EventPublisherStats {
            events_published: self.events_published,
            current_sequence: self.sequence,
            endpoint: self.config.event_endpoint.clone(),
        }
    }
}

/// Statistics for event publisher
#[derive(Debug, Clone)]
pub struct EventPublisherStats {
    /// Total events published
    pub events_published: u64,

    /// Current sequence number
    pub current_sequence: u64,

    /// Endpoint string
    pub endpoint: String,
}

// ============================================================================
// Config Server (REQ/REP for Python to load profile data from Rust)
// ============================================================================

/// Default endpoint for the config REQ/REP channel
pub const CONFIG_ENDPOINT: &str = "ipc:///tmp/cognitive_config.ipc";

/// Request from Python Logic Layer for configuration data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigRequest {
    /// Type of config data requested (e.g., "acoustic_profile")
    pub request_type: String,
    /// Species name for species-specific data
    pub species: String,
    /// Unique request ID for correlation
    pub request_id: String,
}

/// Response from Rust with configuration data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigResponse {
    /// Request ID being responded to
    pub request_id: String,
    /// Whether the request was successful
    pub success: bool,
    /// JSON-encoded data payload
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
    /// Error message if unsuccessful
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// REQ/REP server that provides configuration data to Python at startup
///
/// Python connects, requests the acoustic profile for a species, and
/// receives all grammar data (bigrams, openers, closers, idioms) from
/// the authoritative Rust source. This eliminates data drift between layers.
pub struct ConfigServer {
    sock: zmq::Socket,
    endpoint: String,
}

impl ConfigServer {
    /// Create a new config server bound to the given endpoint
    pub fn new(endpoint: &str) -> Result<Self> {
        let ctx = zmq::Context::new();
        let sock = ctx.socket(zmq::REP)?;
        sock.bind(endpoint)?;

        info!("ConfigServer bound to: {}", endpoint);

        Ok(Self {
            sock,
            endpoint: endpoint.to_string(),
        })
    }

    /// Create with default endpoint
    pub fn with_default_endpoint() -> Result<Self> {
        Self::new(CONFIG_ENDPOINT)
    }

    /// Receive a config request (blocking)
    pub fn recv_request(&self) -> Result<ConfigRequest> {
        let msg = self
            .sock
            .recv_string(0)
            .map_err(|e| anyhow::anyhow!("Failed to recv: {:?}", e))?
            .map_err(|e| anyhow::anyhow!("Failed to decode message: {:?}", e))?;
        let request: ConfigRequest = serde_json::from_str(&msg)?;
        Ok(request)
    }

    /// Send a config response
    pub fn send_response(&self, response: &ConfigResponse) -> Result<()> {
        let json = serde_json::to_vec(response)?;
        self.sock.send(&json, 0)?;
        Ok(())
    }

    /// Handle a single request/response cycle
    pub fn handle_request(&self, request: ConfigRequest) -> Result<()> {
        let response = match request.request_type.as_str() {
            "acoustic_profile" => {
                use crate::acoustic_profile::AcousticProfileFactory;
                let profile = AcousticProfileFactory::create(&request.species);
                let export = profile.to_export();
                ConfigResponse {
                    request_id: request.request_id,
                    success: true,
                    data: Some(serde_json::to_value(&export)?),
                    error: None,
                }
            }
            _ => ConfigResponse {
                request_id: request.request_id,
                success: false,
                data: None,
                error: Some(format!("Unknown request type: {}", request.request_type)),
            },
        };

        self.send_response(&response)
    }

    /// Get the endpoint string
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }
}

// ============================================================================
// Synthesis Action Types (for Python → Rust communication)
// ============================================================================
// These types are serialized/deserialized via Serde for ZeroMQ IPC protocol.
// The compiler sees them as "dead" because they're constructed on the Python
// side and consumed via deserialization, not direct Rust construction.
#[allow(dead_code)]
/// Priority levels for synthesis actions
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ActionPriority {
    /// Low priority - can be delayed
    Low,

    /// Normal priority - default
    #[default]
    Normal,

    /// High priority - should be processed quickly
    High,

    /// Critical priority - must be processed immediately
    Critical,
}

/// Single event in a synthesis timeline
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineEvent {
    /// Cluster ID for synthesis
    pub cluster_id: u32,

    /// Start time in milliseconds from timeline start
    pub start_time_ms: f64,

    /// Duration in milliseconds
    pub duration_ms: f64,

    /// Amplitude (0.0 to 1.0)
    #[serde(default = "default_amplitude")]
    pub amplitude: f32,
}

#[allow(dead_code)]
fn default_amplitude() -> f32 {
    1.0
}

#[allow(dead_code)]
impl TimelineEvent {
    /// Create a new timeline event
    pub fn new(cluster_id: u32, start_time_ms: f64, duration_ms: f64) -> Self {
        Self {
            cluster_id,
            start_time_ms,
            duration_ms,
            amplitude: 1.0,
        }
    }

    /// Create a timeline event with custom amplitude
    pub fn with_amplitude(mut self, amplitude: f32) -> Self {
        self.amplitude = amplitude;
        self
    }

    /// Create a test event
    #[cfg(test)]
    pub fn test_event(cluster_id: u32) -> Self {
        Self::new(cluster_id, 0.0, 150.0)
    }
}

/// Micro-dynamics delta for synthesis modification
#[allow(dead_code)]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MicroDynamicsDelta {
    /// Change to mean F0 (Hz)
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub delta_mean_f0_hz: f32,

    /// Change to duration (ms)
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub delta_duration_ms: f32,

    /// Change to F0 range (Hz)
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub delta_f0_range_hz: f32,

    /// Change to harmonic-to-noise ratio
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub delta_harmonic_to_noise_ratio: f32,

    /// Change to attack time (ms)
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub delta_attack_time_ms: f32,

    /// Change to sustain level
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub delta_sustain_level: f32,

    /// Change to RMS energy
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub delta_rms_energy: f32,
}

#[allow(dead_code)]
impl MicroDynamicsDelta {
    /// Create with only F0 shift
    pub fn with_f0_shift(delta_mean_f0_hz: f32) -> Self {
        Self {
            delta_mean_f0_hz,
            ..Default::default()
        }
    }

    /// Create with only duration shift
    pub fn with_duration_shift(delta_duration_ms: f32) -> Self {
        Self {
            delta_duration_ms,
            ..Default::default()
        }
    }

    /// Check if all deltas are zero (no-op)
    pub fn is_empty(&self) -> bool {
        self.delta_mean_f0_hz == 0.0
            && self.delta_duration_ms == 0.0
            && self.delta_f0_range_hz == 0.0
            && self.delta_harmonic_to_noise_ratio == 0.0
            && self.delta_attack_time_ms == 0.0
            && self.delta_sustain_level == 0.0
            && self.delta_rms_energy == 0.0
    }
}

#[allow(dead_code)]
fn is_zero_f32(v: &f32) -> bool {
    *v == 0.0
}

/// Synthesis action command from Python
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisAction {
    /// Action type (e.g., "synthesize_timeline")
    pub action_type: String,

    /// Timeline of events to synthesize
    pub timeline: Vec<TimelineEvent>,

    /// Optional micro-dynamics deltas
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deltas: Option<MicroDynamicsDelta>,

    /// Action priority
    #[serde(default)]
    pub priority: ActionPriority,
}

#[allow(dead_code)]
impl SynthesisAction {
    /// Create a new synthesis action
    pub fn new(timeline: Vec<TimelineEvent>) -> Self {
        Self {
            action_type: "synthesize_timeline".to_string(),
            timeline,
            deltas: None,
            priority: ActionPriority::Normal,
        }
    }

    /// Create an action with deltas
    pub fn with_deltas(mut self, deltas: MicroDynamicsDelta) -> Self {
        self.deltas = Some(deltas);
        self
    }

    /// Create an action with priority
    pub fn with_priority(mut self, priority: ActionPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Create a single-event action
    pub fn single_event(cluster_id: u32, duration_ms: f64) -> Self {
        Self::new(vec![TimelineEvent::new(cluster_id, 0.0, duration_ms)])
    }

    /// Deserialize from JSON bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        serde_json::from_slice(bytes).map_err(|e| anyhow::anyhow!("Failed to deserialize synthesis action: {}", e))
    }

    /// Serialize to JSON bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(|e| anyhow::anyhow!("Failed to serialize synthesis action: {}", e))
    }

    /// Create a test action
    #[cfg(test)]
    pub fn test_action() -> Self {
        Self::single_event(42, 150.0)
    }
}

// ============================================================================
// Action Subscriber Configuration
// ============================================================================

/// Configuration for action subscriber
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ActionSubscriberConfig {
    /// ZeroMQ endpoint for action commands
    pub action_endpoint: String,

    /// Receive timeout in milliseconds
    pub receive_timeout_ms: u64,

    /// High water mark for receiving
    pub receive_high_water_mark: i32,
}

impl Default for ActionSubscriberConfig {
    fn default() -> Self {
        Self {
            action_endpoint: "ipc:///tmp/cognitive_actions.ipc".to_string(),
            receive_timeout_ms: 100,
            receive_high_water_mark: 10,
        }
    }
}

// ============================================================================
// Action Subscriber
// ============================================================================

/// Subscriber for synthesis actions from Python
///
/// Uses ZeroMQ SUB socket to receive synthesis timeline commands
/// from the Python Logic Layer.
#[allow(dead_code)]
pub struct ActionSubscriber {
    /// ZeroMQ context
    #[allow(dead_code)]
    ctx: zmq::Context,

    /// Subscriber socket
    sock: zmq::Socket,

    /// Configuration
    config: ActionSubscriberConfig,

    /// Actions received counter
    actions_received: u64,
}

#[allow(dead_code)]
impl ActionSubscriber {
    /// Create a new action subscriber
    pub fn new(config: ActionSubscriberConfig) -> Result<Self> {
        info!("Initializing Action Subscriber on: {}", config.action_endpoint);

        let ctx = zmq::Context::new();
        let sock = ctx.socket(zmq::SUB)?;

        // Set socket options
        sock.set_rcvtimeo(config.receive_timeout_ms as i32)?;
        sock.set_rcvhwm(config.receive_high_water_mark)?;

        // Bind to endpoint (Python will connect)
        sock.bind(&config.action_endpoint)?;

        // Subscribe to all messages
        sock.set_subscribe(b"")?;

        info!("Action Subscriber bound to: {}", config.action_endpoint);

        Ok(Self {
            ctx,
            sock,
            config,
            actions_received: 0,
        })
    }

    /// Try to receive an action (non-blocking)
    ///
    /// # Returns
    /// * `Result<Option<SynthesisAction>>` - Received action or None if no message
    pub fn try_recv(&mut self) -> Result<Option<SynthesisAction>> {
        match self.sock.recv_bytes(zmq::DONTWAIT) {
            Ok(bytes) => {
                let action = SynthesisAction::from_bytes(&bytes)?;
                self.actions_received += 1;

                if self.actions_received.is_multiple_of(100) {
                    info!("Received {} actions", self.actions_received);
                }

                Ok(Some(action))
            }
            Err(zmq::Error::EAGAIN) => Ok(None),
            Err(e) => Err(anyhow::anyhow!("Failed to receive action: {}", e)),
        }
    }

    /// Receive an action with timeout
    ///
    /// # Returns
    /// * `Result<Option<SynthesisAction>>` - Received action or None on timeout
    pub fn recv_timeout(&mut self) -> Result<Option<SynthesisAction>> {
        match self.sock.recv_bytes(0) {
            Ok(bytes) => {
                let action = SynthesisAction::from_bytes(&bytes)?;
                self.actions_received += 1;
                Ok(Some(action))
            }
            Err(zmq::Error::EAGAIN) => Ok(None),
            Err(e) => Err(anyhow::anyhow!("Failed to receive action: {}", e)),
        }
    }

    /// Check if subscriber is ready
    pub fn is_ready(&self) -> bool {
        true // Socket is bound
    }

    /// Get the endpoint
    pub fn endpoint(&self) -> &str {
        &self.config.action_endpoint
    }

    /// Get statistics
    pub fn stats(&self) -> ActionSubscriberStats {
        ActionSubscriberStats {
            actions_received: self.actions_received,
            endpoint: self.config.action_endpoint.clone(),
        }
    }
}

/// Statistics for action subscriber
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ActionSubscriberStats {
    /// Total actions received
    pub actions_received: u64,

    /// Endpoint string
    pub endpoint: String,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_heartbeat_message_serialization() {
        let msg = HeartbeatMessage::new(1, 12345).unwrap();
        let bytes = msg.to_bytes().unwrap();
        let decoded = HeartbeatMessage::from_bytes(&bytes).unwrap();

        assert_eq!(decoded.sequence, 1);
        assert_eq!(decoded.pid, 12345);
        assert_eq!(decoded.state, "active");
    }

    #[test]
    fn test_peer_controller_creation() {
        let config = PeerControllerConfig {
            heartbeat_endpoint: "ipc:///tmp/test_heartbeat.ipc".to_string(),
            ..Default::default()
        };

        let controller = PeerController::new(config).unwrap();
        assert_eq!(controller.mode(), OperationMode::Passthrough);
        assert_eq!(controller.audio_mute(), AudioMuteState::Muted);
        assert!(!controller.is_python_alive());
    }

    #[test]
    fn test_peer_controller_no_heartbeat_starts_in_passthrough() {
        let config = PeerControllerConfig {
            heartbeat_endpoint: "ipc:///tmp/test_passthrough.ipc".to_string(),
            ..Default::default()
        };

        let mut controller = PeerController::new(config).unwrap();

        // Tick multiple times without heartbeat
        for _ in 0..10 {
            let mode = controller.tick().unwrap();
            assert_eq!(mode, OperationMode::Passthrough);
            assert!(!controller.is_python_alive());
        }
    }

    #[test]
    fn test_peer_controller_timeout_detection() {
        let config = PeerControllerConfig {
            heartbeat_endpoint: "ipc:///tmp/test_timeout.ipc".to_string(),
            heartbeat_timeout_ms: 50, // Short timeout for testing
            ..Default::default()
        };

        let mut controller = PeerController::new(config).unwrap();

        // Simulate heartbeat received
        controller.last_heartbeat = Some(Instant::now());
        controller.python_alive = true;
        controller.mode = OperationMode::Interactive;

        // Wait past timeout
        thread::sleep(Duration::from_millis(60));

        // Check timeout
        controller.check_timeout();
        controller.update_mode();

        // Should have disconnected
        assert!(!controller.is_python_alive());
        assert_eq!(controller.mode(), OperationMode::Passthrough);
    }

    #[test]
    fn test_heartbeat_sequence_detection() {
        let config = PeerControllerConfig {
            heartbeat_endpoint: "ipc:///tmp/test_sequence.ipc".to_string(),
            ..Default::default()
        };

        let mut controller = PeerController::new(config).unwrap();

        // Send heartbeats with gap
        let msg1 = HeartbeatMessage::new(1, 12345).unwrap();
        controller.handle_heartbeat(msg1);

        let msg2 = HeartbeatMessage::new(5, 12345).unwrap(); // Skipped 2, 3, 4
        controller.handle_heartbeat(msg2);

        // Should have detected gap (logged, but controller continues)
        assert_eq!(controller.last_sequence, 5);
        assert!(controller.is_python_alive());
    }

    #[test]
    fn test_mode_switch_on_heartbeat_received() {
        let config = PeerControllerConfig {
            heartbeat_endpoint: "ipc:///tmp/test_mode_switch.ipc".to_string(),
            ..Default::default()
        };

        let mut controller = PeerController::new(config).unwrap();

        // Start in passthrough
        assert_eq!(controller.mode(), OperationMode::Passthrough);
        assert!(!controller.is_python_alive());

        // Receive heartbeat
        let msg = HeartbeatMessage::new(1, 12345).unwrap();
        controller.handle_heartbeat(msg);

        // Should switch to interactive
        controller.update_mode();
        assert_eq!(controller.mode(), OperationMode::Interactive);
        assert!(controller.is_python_alive());
    }

    #[test]
    fn test_audio_mute_state_transitions() {
        let config = PeerControllerConfig {
            heartbeat_endpoint: "ipc:///tmp/test_mute.ipc".to_string(),
            ..Default::default()
        };

        let mut controller = PeerController::new(config).unwrap();

        // Start muted
        assert_eq!(controller.audio_mute(), AudioMuteState::Muted);

        // Receive heartbeat - audio should become active
        let msg = HeartbeatMessage::new(1, 12345).unwrap();
        controller.handle_heartbeat(msg);
        assert_eq!(controller.audio_mute(), AudioMuteState::Active);

        // Disconnect - audio should mute
        controller.handle_disconnect();
        assert_eq!(controller.audio_mute(), AudioMuteState::Muted);
    }

    #[test]
    fn test_disconnect_handling() {
        let config = PeerControllerConfig {
            heartbeat_endpoint: "ipc:///tmp/test_disconnect.ipc".to_string(),
            ..Default::default()
        };

        let mut controller = PeerController::new(config).unwrap();

        // Simulate connected state
        let msg = HeartbeatMessage::new(1, 12345).unwrap();
        controller.handle_heartbeat(msg);
        assert!(controller.is_python_alive());

        // Disconnect
        controller.handle_disconnect();
        assert!(!controller.is_python_alive());
        assert_eq!(controller.mode(), OperationMode::Passthrough);
        assert!(controller.last_heartbeat.is_none());
        assert_eq!(controller.last_sequence, 0);
    }

    #[test]
    fn test_time_since_last_heartbeat() {
        let config = PeerControllerConfig {
            heartbeat_endpoint: "ipc:///tmp/test_time_since.ipc".to_string(),
            ..Default::default()
        };

        let mut controller = PeerController::new(config).unwrap();

        // No heartbeat yet
        assert!(controller.time_since_last_heartbeat().is_none());

        // Receive heartbeat
        let msg = HeartbeatMessage::new(1, 12345).unwrap();
        controller.handle_heartbeat(msg);

        // Should have a time
        let elapsed = controller.time_since_last_heartbeat();
        assert!(elapsed.is_some());
        assert!(elapsed.unwrap() < Duration::from_secs(1));
    }

    #[test]
    fn test_config_access() {
        let config = PeerControllerConfig {
            heartbeat_endpoint: "ipc:///tmp/test_config.ipc".to_string(),
            heartbeat_timeout_ms: 200,
            poll_interval_ms: 20,
            verbose_logging: true,
        };

        let controller = PeerController::new(config.clone()).unwrap();
        let retrieved_config = controller.get_config();

        assert_eq!(retrieved_config.heartbeat_endpoint, "ipc:///tmp/test_config.ipc");
        assert_eq!(retrieved_config.heartbeat_timeout_ms, 200);
        assert_eq!(retrieved_config.poll_interval_ms, 20);
        assert!(retrieved_config.verbose_logging);
    }

    #[test]
    fn test_heartbeat_endpoint_string() {
        let config = PeerControllerConfig {
            heartbeat_endpoint: "ipc:///tmp/my_endpoint.ipc".to_string(),
            ..Default::default()
        };

        let controller = PeerController::new(config).unwrap();
        assert_eq!(controller.heartbeat_endpoint(), "ipc:///tmp/my_endpoint.ipc");
    }

    #[test]
    fn test_reconnection_scenario() {
        let config = PeerControllerConfig {
            heartbeat_endpoint: "ipc:///tmp/test_reconnect.ipc".to_string(),
            heartbeat_timeout_ms: 50,
            ..Default::default()
        };

        let mut controller = PeerController::new(config).unwrap();

        // Initial state: disconnected
        assert!(!controller.is_python_alive());
        assert_eq!(controller.mode(), OperationMode::Passthrough);

        // Connect
        let msg1 = HeartbeatMessage::new(1, 12345).unwrap();
        controller.handle_heartbeat(msg1);
        assert!(controller.is_python_alive());
        // handle_heartbeat doesn't call update_mode, so mode is still Passthrough
        controller.update_mode();
        assert_eq!(controller.mode(), OperationMode::Interactive);

        // Disconnect - handle_disconnect doesn't call update_mode
        controller.handle_disconnect();
        assert!(!controller.is_python_alive());
        controller.update_mode();
        assert_eq!(controller.mode(), OperationMode::Passthrough);

        // Reconnect with new PID
        let msg2 = HeartbeatMessage::new(10, 67890).unwrap();
        controller.handle_heartbeat(msg2);
        assert!(controller.is_python_alive());
        controller.update_mode();
        assert_eq!(controller.mode(), OperationMode::Interactive);
    }
}

// ============================================================================
// Feature Event Tests
// ============================================================================

#[cfg(test)]
mod feature_event_tests {
    use super::*;

    #[test]
    fn test_feature_event_serialization() {
        let event = FeatureEvent::from_array(42, [0.0f32; 112], 1);
        let json = serde_json::to_string(&event).unwrap();

        assert!(json.contains("\"event_type\":\"feature_extraction\""));
        assert!(json.contains("\"cluster_id\":42"));
        assert!(json.contains("\"sequence\":1"));
    }

    #[test]
    fn test_feature_event_deserialization() {
        let json = r#"{
            "event_type": "feature_extraction",
            "cluster_id": 42,
            "features_112d": [0.0, 1.0, 2.0],
            "timestamp": 1699345823.456,
            "sequence": 12345
        }"#;

        let event: FeatureEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event.cluster_id, 42);
        assert_eq!(event.event_type, "feature_extraction");
        assert_eq!(event.features_112d.len(), 3);
        assert_eq!(event.timestamp, 1699345823.456);
        assert_eq!(event.sequence, 12345);
    }

    #[test]
    fn test_feature_event_new_with_valid_features() {
        let features: Vec<f32> = (0..112).map(|i| i as f32).collect();
        let event = FeatureEvent::new(42, features.clone(), 1).unwrap();

        assert_eq!(event.cluster_id, 42);
        assert_eq!(event.features_112d.len(), 112);
        assert_eq!(event.sequence, 1);
        assert_eq!(event.event_type, "feature_extraction");
        assert!(event.timestamp > 0.0);
    }

    #[test]
    fn test_feature_event_new_with_wrong_dimension() {
        let features: Vec<f32> = vec![0.0; 100]; // Wrong size
        let result = FeatureEvent::new(42, features, 1);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("must have 112 elements"));
    }

    #[test]
    fn test_feature_event_from_array() {
        let mut features = [0.0f32; 112];
        features[0] = 1.0;
        features[111] = 112.0;

        let event = FeatureEvent::from_array(42, features, 1);

        assert_eq!(event.cluster_id, 42);
        assert_eq!(event.features_112d[0], 1.0);
        assert_eq!(event.features_112d[111], 112.0);
        assert_eq!(event.sequence, 1);
    }

    #[test]
    fn test_feature_event_to_bytes() {
        let event = FeatureEvent::from_array(42, [0.0f32; 112], 1);
        let bytes = event.to_bytes().unwrap();

        assert!(!bytes.is_empty());

        // Verify it's valid JSON
        let decoded: FeatureEvent = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(decoded.cluster_id, 42);
    }

    #[test]
    fn test_feature_event_from_bytes() {
        let original = FeatureEvent::from_array(42, [1.0f32; 112], 1);
        let bytes = original.to_bytes().unwrap();
        let decoded = FeatureEvent::from_bytes(&bytes).unwrap();

        assert_eq!(decoded.cluster_id, original.cluster_id);
        assert_eq!(decoded.features_112d, original.features_112d);
        assert_eq!(decoded.sequence, original.sequence);
    }

    #[test]
    fn test_feature_event_test_event() {
        let event = FeatureEvent::test_event(42, 1);

        assert_eq!(event.cluster_id, 42);
        assert_eq!(event.features_112d.len(), 112);
        assert!(event.features_112d.iter().all(|&f| f == 0.0));
        assert_eq!(event.sequence, 1);
    }
}

#[cfg(test)]
mod feature_publisher_tests {
    use super::*;

    #[test]
    fn test_event_publisher_config_default() {
        let config = EventPublisherConfig::default();

        assert_eq!(config.event_endpoint, "ipc:///tmp/cognitive_features.ipc");
        assert_eq!(config.send_high_water_mark, 100);
        assert!(!config.verbose_logging);
    }

    #[test]
    fn test_feature_event_publisher_creation() {
        let config = EventPublisherConfig {
            event_endpoint: "ipc:///tmp/test_features.ipc".to_string(),
            ..Default::default()
        };

        let publisher = FeatureEventPublisher::new(config).unwrap();
        assert!(publisher.is_ready());
        assert_eq!(publisher.endpoint(), "ipc:///tmp/test_features.ipc");
    }

    #[test]
    fn test_feature_event_publisher_stats() {
        let config = EventPublisherConfig {
            event_endpoint: "ipc:///tmp/test_stats.ipc".to_string(),
            ..Default::default()
        };

        let publisher = FeatureEventPublisher::new(config).unwrap();
        let stats = publisher.stats();

        assert_eq!(stats.events_published, 0);
        assert_eq!(stats.current_sequence, 0);
        assert_eq!(stats.endpoint, "ipc:///tmp/test_stats.ipc");
    }

    #[test]
    fn test_feature_event_publisher_publish() {
        let config = EventPublisherConfig {
            event_endpoint: "ipc:///tmp/test_publish.ipc".to_string(),
            ..Default::default()
        };

        let mut publisher = FeatureEventPublisher::new(config).unwrap();

        let event = FeatureEvent::test_event(42, 1);
        let result = publisher.publish(&event);

        assert!(result.is_ok());

        let stats = publisher.stats();
        assert_eq!(stats.events_published, 1);
    }

    #[test]
    fn test_feature_event_publisher_publish_features() {
        let config = EventPublisherConfig {
            event_endpoint: "ipc:///tmp/test_publish_features.ipc".to_string(),
            ..Default::default()
        };

        let mut publisher = FeatureEventPublisher::new(config).unwrap();

        // Publish features directly
        let features: Vec<f32> = (0..112).map(|i| i as f32).collect();
        let result = publisher.publish_features(42, features);

        assert!(result.is_ok());

        let stats = publisher.stats();
        assert_eq!(stats.events_published, 1);
        assert_eq!(stats.current_sequence, 1);
    }

    #[test]
    fn test_feature_event_publisher_sequence_incrementing() {
        let config = EventPublisherConfig {
            event_endpoint: "ipc:///tmp/test_sequence_inc.ipc".to_string(),
            ..Default::default()
        };

        let mut publisher = FeatureEventPublisher::new(config).unwrap();

        // Publish multiple events
        for i in 1..=5 {
            let features: Vec<f32> = vec![i as f32; 112];
            publisher.publish_features(i as u32, features).unwrap();
        }

        let stats = publisher.stats();
        assert_eq!(stats.events_published, 5);
        assert_eq!(stats.current_sequence, 5);
    }

    #[test]
    fn test_feature_event_publisher_wrong_dimension_fails() {
        let config = EventPublisherConfig {
            event_endpoint: "ipc:///tmp/test_wrong_dim.ipc".to_string(),
            ..Default::default()
        };

        let mut publisher = FeatureEventPublisher::new(config).unwrap();

        // Try to publish with wrong dimensions
        let features: Vec<f32> = vec![0.0; 100]; // Wrong size
        let result = publisher.publish_features(42, features);

        assert!(result.is_err());
    }
}

// ============================================================================
// Synthesis Action Tests
// ============================================================================

#[cfg(test)]
mod synthesis_action_tests {
    use super::*;

    #[test]
    fn test_timeline_event_serialization() {
        let event = TimelineEvent::new(42, 0.0, 150.0);
        let json = serde_json::to_string(&event).unwrap();

        assert!(json.contains("\"cluster_id\":42"));
        assert!(json.contains("\"start_time_ms\":0.0"));
        assert!(json.contains("\"duration_ms\":150.0"));
    }

    #[test]
    fn test_timeline_event_deserialization() {
        let json = r#"{"cluster_id":42,"start_time_ms":0.0,"duration_ms":150.0,"amplitude":0.8}"#;
        let event: TimelineEvent = serde_json::from_str(json).unwrap();

        assert_eq!(event.cluster_id, 42);
        assert_eq!(event.start_time_ms, 0.0);
        assert_eq!(event.duration_ms, 150.0);
        assert!((event.amplitude - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_timeline_event_default_amplitude() {
        let json = r#"{"cluster_id":42,"start_time_ms":0.0,"duration_ms":150.0}"#;
        let event: TimelineEvent = serde_json::from_str(json).unwrap();

        assert!((event.amplitude - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_micro_dynamics_delta_default() {
        let delta = MicroDynamicsDelta::default();

        assert_eq!(delta.delta_mean_f0_hz, 0.0);
        assert_eq!(delta.delta_duration_ms, 0.0);
        assert!(delta.is_empty());
    }

    #[test]
    fn test_micro_dynamics_delta_with_f0_shift() {
        let delta = MicroDynamicsDelta::with_f0_shift(100.0);

        assert_eq!(delta.delta_mean_f0_hz, 100.0);
        assert!(!delta.is_empty());
    }

    #[test]
    fn test_micro_dynamics_delta_serialization() {
        let delta = MicroDynamicsDelta::with_f0_shift(100.0);
        let json = serde_json::to_string(&delta).unwrap();

        assert!(json.contains("\"delta_mean_f0_hz\":100.0"));
    }

    #[test]
    fn test_synthesis_action_creation() {
        let timeline = vec![TimelineEvent::new(42, 0.0, 150.0)];
        let action = SynthesisAction::new(timeline.clone());

        assert_eq!(action.action_type, "synthesize_timeline");
        assert_eq!(action.timeline.len(), 1);
        assert_eq!(action.timeline[0].cluster_id, 42);
        assert!(action.deltas.is_none());
        assert_eq!(action.priority, ActionPriority::Normal);
    }

    #[test]
    fn test_synthesis_action_single_event() {
        let action = SynthesisAction::single_event(42, 150.0);

        assert_eq!(action.timeline.len(), 1);
        assert_eq!(action.timeline[0].cluster_id, 42);
        assert_eq!(action.timeline[0].duration_ms, 150.0);
    }

    #[test]
    fn test_synthesis_action_with_deltas() {
        let delta = MicroDynamicsDelta::with_f0_shift(100.0);
        let action = SynthesisAction::single_event(42, 150.0).with_deltas(delta);

        assert!(action.deltas.is_some());
        assert_eq!(action.deltas.unwrap().delta_mean_f0_hz, 100.0);
    }

    #[test]
    fn test_synthesis_action_with_priority() {
        let action = SynthesisAction::single_event(42, 150.0).with_priority(ActionPriority::Critical);

        assert_eq!(action.priority, ActionPriority::Critical);
    }

    #[test]
    fn test_synthesis_action_serialization() {
        let action = SynthesisAction::single_event(42, 150.0).with_deltas(MicroDynamicsDelta::with_f0_shift(100.0));

        let json = action.to_bytes().unwrap();
        let json_str = String::from_utf8(json).unwrap();

        assert!(json_str.contains("\"action_type\":\"synthesize_timeline\""));
        assert!(json_str.contains("\"cluster_id\":42"));
    }

    #[test]
    fn test_synthesis_action_deserialization() {
        let json = r#"{
            "action_type": "synthesize_timeline",
            "timeline": [{"cluster_id":42,"start_time_ms":0.0,"duration_ms":150.0,"amplitude":1.0}],
            "deltas": {"delta_mean_f0_hz": 100.0},
            "priority": "high"
        }"#;

        let action: SynthesisAction = serde_json::from_str(json).unwrap();

        assert_eq!(action.action_type, "synthesize_timeline");
        assert_eq!(action.timeline.len(), 1);
        assert_eq!(action.timeline[0].cluster_id, 42);
        assert!(action.deltas.is_some());
        assert_eq!(action.priority, ActionPriority::High);
    }

    #[test]
    fn test_synthesis_action_roundtrip() {
        let original = SynthesisAction::single_event(42, 150.0)
            .with_deltas(MicroDynamicsDelta::with_f0_shift(100.0))
            .with_priority(ActionPriority::Critical);

        let bytes = original.to_bytes().unwrap();
        let decoded = SynthesisAction::from_bytes(&bytes).unwrap();

        assert_eq!(decoded.action_type, original.action_type);
        assert_eq!(decoded.timeline.len(), original.timeline.len());
        assert_eq!(decoded.timeline[0].cluster_id, original.timeline[0].cluster_id);
        assert_eq!(decoded.priority, original.priority);
    }

    #[test]
    fn test_action_priority_serde() {
        assert_eq!(ActionPriority::Normal, ActionPriority::default());

        let json = "\"normal\"";
        let priority: ActionPriority = serde_json::from_str(json).unwrap();
        assert_eq!(priority, ActionPriority::Normal);

        let json = "\"critical\"";
        let priority: ActionPriority = serde_json::from_str(json).unwrap();
        assert_eq!(priority, ActionPriority::Critical);
    }
}

#[cfg(test)]
mod action_subscriber_tests {
    use super::*;

    #[test]
    fn test_action_subscriber_config_default() {
        let config = ActionSubscriberConfig::default();

        assert_eq!(config.action_endpoint, "ipc:///tmp/cognitive_actions.ipc");
        assert_eq!(config.receive_timeout_ms, 100);
        assert_eq!(config.receive_high_water_mark, 10);
    }

    #[test]
    fn test_action_subscriber_creation() {
        let config = ActionSubscriberConfig {
            action_endpoint: "ipc:///tmp/test_actions.ipc".to_string(),
            ..Default::default()
        };

        let subscriber = ActionSubscriber::new(config).unwrap();
        assert!(subscriber.is_ready());
        assert_eq!(subscriber.endpoint(), "ipc:///tmp/test_actions.ipc");
    }

    #[test]
    fn test_action_subscriber_stats() {
        let config = ActionSubscriberConfig {
            action_endpoint: "ipc:///tmp/test_action_stats.ipc".to_string(),
            ..Default::default()
        };

        let subscriber = ActionSubscriber::new(config).unwrap();
        let stats = subscriber.stats();

        assert_eq!(stats.actions_received, 0);
        assert_eq!(stats.endpoint, "ipc:///tmp/test_action_stats.ipc");
    }

    #[test]
    fn test_action_subscriber_try_recv_empty() {
        let config = ActionSubscriberConfig {
            action_endpoint: "ipc:///tmp/test_try_recv.ipc".to_string(),
            ..Default::default()
        };

        let mut subscriber = ActionSubscriber::new(config).unwrap();

        // Should return None when no message available
        let result = subscriber.try_recv().unwrap();
        assert!(result.is_none());
    }
}
