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

use anyhow::Result;
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
    pub fn new(sequence: u64, pid: u32) -> Self {
        Self {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            sequence,
            pid,
            state: "active".to_string(),
        }
    }

    /// Serialize to JSON bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self)
            .map_err(|e| anyhow::anyhow!("Failed to serialize heartbeat: {}", e))
    }

    /// Deserialize from JSON bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        serde_json::from_slice(bytes)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize heartbeat: {}", e))
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
            warn!(
                "Missed {} heartbeats",
                heartbeat.sequence - self.last_sequence - 1
            );
        }

        self.last_sequence = heartbeat.sequence;
        self.last_heartbeat = Some(now);

        if !self.python_alive {
            info!(
                "⚡ Cognitive Agent (Python) RECONNECTED - PID: {}",
                heartbeat.pid
            );
            self.python_alive = true;
            self.audio_mute = AudioMuteState::Active;
            info!("Switching to Interactive Mode");
        }

        if self.config.verbose_logging {
            info!(
                "Heartbeat received: seq={}, pid={}",
                heartbeat.sequence, heartbeat.pid
            );
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
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_heartbeat_message_serialization() {
        let msg = HeartbeatMessage::new(1, 12345);
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
        let msg1 = HeartbeatMessage::new(1, 12345);
        controller.handle_heartbeat(msg1);

        let msg2 = HeartbeatMessage::new(5, 12345); // Skipped 2, 3, 4
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
        let msg = HeartbeatMessage::new(1, 12345);
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
        let msg = HeartbeatMessage::new(1, 12345);
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
        let msg = HeartbeatMessage::new(1, 12345);
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
        let msg = HeartbeatMessage::new(1, 12345);
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

        assert_eq!(
            retrieved_config.heartbeat_endpoint,
            "ipc:///tmp/test_config.ipc"
        );
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
        assert_eq!(
            controller.heartbeat_endpoint(),
            "ipc:///tmp/my_endpoint.ipc"
        );
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
        let msg1 = HeartbeatMessage::new(1, 12345);
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
        let msg2 = HeartbeatMessage::new(10, 67890);
        controller.handle_heartbeat(msg2);
        assert!(controller.is_python_alive());
        controller.update_mode();
        assert_eq!(controller.mode(), OperationMode::Interactive);
    }
}
