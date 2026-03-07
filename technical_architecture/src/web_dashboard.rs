// Remote Intervention Dashboard
//
// Provides secure HTTPS/WebSocket dashboard for real-time monitoring,
// manual override, and remote debugging.

use crate::ptp::PtpTimestamp;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ============================================================================
// Data Structures
// ============================================================================

/// Dashboard configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    pub bind_address: String, // e.g., "0.0.0.0:8443"
    pub tls_cert_path: PathBuf,
    pub tls_key_path: PathBuf,
    pub auth_secret: String, // JWT secret
    pub token_expiry_hours: u64,
    pub max_connections: usize,
    pub enable_tls: bool, // Can disable for testing
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            bind_address: "127.0.0.1:8443".to_string(),
            tls_cert_path: PathBuf::from("/etc/dashboard/cert.pem"),
            tls_key_path: PathBuf::from("/etc/dashboard/key.pem"),
            auth_secret: "secret_change_in_production".to_string(),
            token_expiry_hours: 24,
            max_connections: 10,
            enable_tls: false, // Disabled by default for testing
        }
    }
}

/// Dashboard state (sent to clients)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardState {
    pub operation_mode: DashboardOperationMode,
    pub battery_level: f32, // 0-100%
    pub temperature_celsius: f32,
    pub uptime_seconds: u64,
    pub active_connections: usize,
    pub iacuc_status: IacucStatus,
    pub calibration_status: CalibrationDashboardStatus,
    pub last_updated: PtpTimestamp,
}

/// Operation mode for dashboard
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DashboardOperationMode {
    Passthrough,
    Interactive,
    Maintenance,
    Emergency,
}

/// IACUC compliance status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IacucStatus {
    Compliant,
    Warning,
    Violation,
    Unknown,
}

/// Calibration status for dashboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationDashboardStatus {
    pub last_calibration: Option<PtpTimestamp>,
    pub health_status: String,
    pub drift_db: Option<f32>,
    pub last_check: PtpTimestamp,
}

impl Default for CalibrationDashboardStatus {
    fn default() -> Self {
        Self {
            last_calibration: None,
            health_status: "Unknown".to_string(),
            drift_db: None,
            last_check: PtpTimestamp::from(chrono::Utc::now()),
        }
    }
}

/// WebSocket message types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WsMessage {
    Spectrogram { data: Vec<f32>, sample_rate: u32 },
    GaugeUpdate { name: String, value: f32, unit: String },
    StatusUpdate { status: DashboardState },
    Error { message: String },
    Info { message: String },
}

/// Dashboard command from client
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum DashboardCommand {
    EmergencyStop,
    ManualOverride { intent: String },
    SetParameter { name: String, value: serde_json::Value },
    RunCalibration,
    GetStatus,
    SubscribeSpectrogram,
    UnsubscribeSpectrogram,
}

/// Auth token
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthToken {
    pub token: String,
    pub expires_at: PtpTimestamp,
    pub user: String,
}

/// Command audit log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandAuditLog {
    pub timestamp: PtpTimestamp,
    pub command: DashboardCommand,
    pub user: String,
    pub ip_address: String,
    pub result: CommandResult,
}

/// Command result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommandResult {
    Success { message: String },
    Failed { error: String },
    Rejected { reason: String },
}

/// Gauge value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaugeValue {
    pub name: String,
    pub value: f32,
    pub unit: String,
    pub timestamp: PtpTimestamp,
}

// ============================================================================
// Web Dashboard
// ============================================================================

/// Remote intervention dashboard
pub struct WebDashboard {
    config: DashboardConfig,
    state: Arc<Mutex<DashboardState>>,
    connections: Arc<Mutex<HashMap<String, ClientConnection>>>,
    command_log: Arc<Mutex<Vec<CommandAuditLog>>>,
    active_tokens: Arc<Mutex<HashMap<String, AuthToken>>>,
    start_time: Instant,
    is_running: Arc<Mutex<bool>>,
}

/// Client connection info
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ClientConnection {
    #[allow(dead_code)]
    id: String,
    #[allow(dead_code)]
    connected_at: Instant,
    #[allow(dead_code)]
    ip_address: String,
    #[allow(dead_code)]
    subscriptions: Vec<String>,
    last_heartbeat: Instant,
}

impl WebDashboard {
    pub fn new(config: DashboardConfig) -> Self {
        let start_time = Instant::now();

        let state = DashboardState {
            operation_mode: DashboardOperationMode::Passthrough,
            battery_level: 100.0,
            temperature_celsius: 25.0,
            uptime_seconds: 0,
            active_connections: 0,
            iacuc_status: IacucStatus::Unknown,
            calibration_status: CalibrationDashboardStatus::default(),
            last_updated: PtpTimestamp::from(chrono::Utc::now()),
        };

        Self {
            config,
            state: Arc::new(Mutex::new(state)),
            connections: Arc::new(Mutex::new(HashMap::new())),
            command_log: Arc::new(Mutex::new(Vec::new())),
            active_tokens: Arc::new(Mutex::new(HashMap::new())),
            start_time,
            is_running: Arc::new(Mutex::new(false)),
        }
    }

    pub fn with_default_config() -> Self {
        Self::new(DashboardConfig::default())
    }

    /// Start the dashboard server
    pub async fn start(&self) -> Result<()> {
        *self.is_running.lock().unwrap() = true;
        log::info!("Web dashboard starting on {}", self.config.bind_address);

        // In production, this would start the actual HTTP/WebSocket server
        // For now, we simulate the server
        Ok(())
    }

    /// Stop the dashboard server
    pub async fn stop(&self) -> Result<()> {
        *self.is_running.lock().unwrap() = false;
        log::info!("Web dashboard stopped");

        // Disconnect all clients
        self.connections.lock().unwrap().clear();

        Ok(())
    }

    /// Check if server is running
    pub fn is_running(&self) -> bool {
        *self.is_running.lock().unwrap()
    }

    /// Authenticate a client and return token
    pub fn authenticate(&self, username: &str, password: &str) -> Result<AuthToken> {
        // In production, this would validate against a real user database
        if username == "admin" && password == "admin" {
            let token = self.generate_token(username);
            Ok(token)
        } else {
            Err(anyhow::anyhow!("Invalid credentials"))
        }
    }

    /// Generate auth token
    fn generate_token(&self, user: &str) -> AuthToken {
        use chrono::Duration;
        let expires_at =
            PtpTimestamp::from(chrono::Utc::now() + Duration::hours(self.config.token_expiry_hours as i64));

        // Simple token generation (in production, use proper JWT)
        let token = format!("{}_{}", user, expires_at.as_nanos());

        AuthToken {
            token,
            expires_at,
            user: user.to_string(),
        }
    }

    /// Validate auth token
    pub fn validate_token(&self, token: &str) -> Result<bool> {
        let tokens = self.active_tokens.lock().unwrap();
        let now = PtpTimestamp::from(chrono::Utc::now());

        if let Some(auth_token) = tokens.get(token) {
            Ok(auth_token.expires_at.as_nanos() > now.as_nanos())
        } else {
            Ok(false)
        }
    }

    /// Add active token
    pub fn add_token(&self, token: &AuthToken) {
        let mut tokens = self.active_tokens.lock().unwrap();
        tokens.insert(token.token.clone(), token.clone());

        // Clean up expired tokens
        let now = PtpTimestamp::from(chrono::Utc::now());
        tokens.retain(|_, t| t.expires_at.as_nanos() > now.as_nanos());
    }

    /// Connect a client
    pub fn connect_client(&self, client_id: &str, ip_address: &str, token: &str) -> Result<()> {
        // Check capacity
        if self.is_at_capacity() {
            return Err(anyhow::anyhow!("Dashboard at maximum capacity"));
        }

        // Validate token
        let valid = self.validate_token(token)?;
        if !valid {
            return Err(anyhow::anyhow!("Invalid or expired token"));
        }

        let connection = ClientConnection {
            id: client_id.to_string(),
            connected_at: Instant::now(),
            ip_address: ip_address.to_string(),
            subscriptions: Vec::new(),
            last_heartbeat: Instant::now(),
        };

        self.connections
            .lock()
            .unwrap()
            .insert(client_id.to_string(), connection);

        // Update connection count
        let mut state = self.state.lock().unwrap();
        state.active_connections = self.connections.lock().unwrap().len();
        state.last_updated = PtpTimestamp::from(chrono::Utc::now());

        log::info!("Client {} connected from {}", client_id, ip_address);
        Ok(())
    }

    /// Disconnect a client
    pub fn disconnect_client(&self, client_id: &str) {
        self.connections.lock().unwrap().remove(client_id);

        // Update connection count
        let mut state = self.state.lock().unwrap();
        state.active_connections = self.connections.lock().unwrap().len();
        state.last_updated = PtpTimestamp::from(chrono::Utc::now());

        log::info!("Client {} disconnected", client_id);
    }

    /// Process a dashboard command
    pub fn process_command(&self, command: DashboardCommand, user: &str, ip_address: &str) -> CommandResult {
        log::info!("Processing command from {}: {:?}", user, command);

        let result = match &command {
            DashboardCommand::EmergencyStop => {
                self.handle_emergency_stop();
                CommandResult::Success {
                    message: "Emergency stop executed".to_string(),
                }
            }
            DashboardCommand::ManualOverride { intent } => {
                self.handle_manual_override(intent);
                CommandResult::Success {
                    message: format!("Manual override executed with intent: {}", intent),
                }
            }
            DashboardCommand::SetParameter { name, value } => {
                self.handle_set_parameter(name, value);
                CommandResult::Success {
                    message: format!("Parameter {} set to {:?}", name, value),
                }
            }
            DashboardCommand::RunCalibration => {
                // In production, this would trigger actual calibration
                CommandResult::Success {
                    message: "Calibration scheduled".to_string(),
                }
            }
            DashboardCommand::GetStatus => {
                // Status is returned separately
                CommandResult::Success {
                    message: "Status retrieved".to_string(),
                }
            }
            DashboardCommand::SubscribeSpectrogram => CommandResult::Success {
                message: "Subscribed to spectrogram stream".to_string(),
            },
            DashboardCommand::UnsubscribeSpectrogram => CommandResult::Success {
                message: "Unsubscribed from spectrogram stream".to_string(),
            },
        };

        // Log the command
        let log_entry = CommandAuditLog {
            timestamp: PtpTimestamp::from(chrono::Utc::now()),
            command: command.clone(),
            user: user.to_string(),
            ip_address: ip_address.to_string(),
            result: result.clone(),
        };

        self.command_log.lock().unwrap().push(log_entry);

        result
    }

    /// Handle emergency stop command
    fn handle_emergency_stop(&self) {
        let mut state = self.state.lock().unwrap();
        state.operation_mode = DashboardOperationMode::Emergency;
        state.last_updated = PtpTimestamp::from(chrono::Utc::now());

        log::warn!("EMERGENCY STOP activated via dashboard");
    }

    /// Handle manual override command
    fn handle_manual_override(&self, _intent: &str) {
        let mut state = self.state.lock().unwrap();
        state.operation_mode = DashboardOperationMode::Interactive;
        state.last_updated = PtpTimestamp::from(chrono::Utc::now());

        log::info!("Manual override activated via dashboard");
    }

    /// Handle set parameter command
    fn handle_set_parameter(&self, name: &str, value: &serde_json::Value) {
        log::info!("Parameter {} set to {:?}", name, value);
        // In production, this would update actual system parameters
    }

    /// Get current dashboard state
    pub fn get_state(&self) -> DashboardState {
        let mut state = self.state.lock().unwrap();

        // Update uptime
        state.uptime_seconds = self.start_time.elapsed().as_secs();
        state.last_updated = PtpTimestamp::from(chrono::Utc::now());

        state.clone()
    }

    /// Update a gauge value
    pub fn update_gauge(&self, name: &str, value: f32, unit: &str) {
        log::debug!("Gauge update: {} = {} {}", name, value, unit);

        // In production, this would broadcast to subscribed clients
        // via WebSocket
    }

    /// Broadcast spectrogram data
    pub fn broadcast_spectrogram(&self, data: Vec<f32>, sample_rate: u32) {
        log::debug!("Broadcasting spectrogram: {} samples @ {}Hz", data.len(), sample_rate);

        // In production, this would broadcast to subscribed clients
        // via WebSocket
    }

    /// Update battery level
    pub fn update_battery(&self, level: f32) {
        let mut state = self.state.lock().unwrap();
        state.battery_level = level.clamp(0.0, 100.0);
        state.last_updated = PtpTimestamp::from(chrono::Utc::now());

        // Broadcast gauge update
        self.update_gauge("battery", level, "%");
    }

    /// Update temperature
    pub fn update_temperature(&self, temp: f32) {
        let mut state = self.state.lock().unwrap();
        state.temperature_celsius = temp;
        state.last_updated = PtpTimestamp::from(chrono::Utc::now());

        // Broadcast gauge update
        self.update_gauge("temperature", temp, "°C");
    }

    /// Update operation mode
    pub fn update_operation_mode(&self, mode: DashboardOperationMode) {
        let mut state = self.state.lock().unwrap();
        state.operation_mode = mode;
        state.last_updated = PtpTimestamp::from(chrono::Utc::now());

        log::info!("Operation mode changed to {:?}", mode);
    }

    /// Update IACUC status
    pub fn update_iacuc_status(&self, status: IacucStatus) {
        let mut state = self.state.lock().unwrap();
        state.iacuc_status = status;
        state.last_updated = PtpTimestamp::from(chrono::Utc::now());

        log::info!("IACUC status changed to {:?}", status);
    }

    /// Update calibration status
    pub fn update_calibration_status(&self, status: CalibrationDashboardStatus) {
        let mut state = self.state.lock().unwrap();
        state.calibration_status = status;
        state.last_updated = PtpTimestamp::from(chrono::Utc::now());
    }

    /// Get command log
    pub fn get_command_log(&self) -> Vec<CommandAuditLog> {
        self.command_log.lock().unwrap().clone()
    }

    /// Get connected clients count
    pub fn connected_clients_count(&self) -> usize {
        self.connections.lock().unwrap().len()
    }

    /// Check if server is at capacity
    pub fn is_at_capacity(&self) -> bool {
        self.connected_clients_count() >= self.config.max_connections
    }

    /// Get configuration
    pub fn config(&self) -> &DashboardConfig {
        &self.config
    }

    /// Heartbeat check (called periodically to clean up stale connections)
    pub fn heartbeat_check(&self) {
        let timeout = Duration::from_secs(60);
        let now = Instant::now();

        let mut connections = self.connections.lock().unwrap();
        let mut stale_clients = Vec::new();

        for (id, conn) in connections.iter() {
            if now.duration_since(conn.last_heartbeat) > timeout {
                stale_clients.push(id.clone());
            }
        }

        for id in stale_clients {
            connections.remove(&id);
            log::info!("Removed stale client {}", id);
        }

        // Update connection count
        let mut state = self.state.lock().unwrap();
        state.active_connections = connections.len();
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_dashboard() -> WebDashboard {
        let config = DashboardConfig {
            max_connections: 5,
            ..Default::default()
        };
        WebDashboard::new(config)
    }

    // ============================================================================
    // DashboardConfig Tests
    // ============================================================================

    #[test]
    fn test_config_default() {
        let config = DashboardConfig::default();
        assert_eq!(config.bind_address, "127.0.0.1:8443");
        assert_eq!(config.token_expiry_hours, 24);
        assert_eq!(config.max_connections, 10);
        assert!(!config.enable_tls);
    }

    // ============================================================================
    // DashboardOperationMode Tests
    // ============================================================================

    #[test]
    fn test_operation_mode() {
        assert_eq!(DashboardOperationMode::Passthrough, DashboardOperationMode::Passthrough);
        assert_ne!(DashboardOperationMode::Interactive, DashboardOperationMode::Passthrough);
    }

    // ============================================================================
    // DashboardState Tests
    // ============================================================================

    #[test]
    fn test_dashboard_state_creation() {
        let state = DashboardState {
            operation_mode: DashboardOperationMode::Interactive,
            battery_level: 85.0,
            temperature_celsius: 30.0,
            uptime_seconds: 3600,
            active_connections: 2,
            iacuc_status: IacucStatus::Compliant,
            calibration_status: CalibrationDashboardStatus::default(),
            last_updated: PtpTimestamp::from(chrono::Utc::now()),
        };

        assert_eq!(state.operation_mode, DashboardOperationMode::Interactive);
        assert_eq!(state.battery_level, 85.0);
        assert_eq!(state.active_connections, 2);
    }

    // ============================================================================
    // WebDashboard Tests
    // ============================================================================

    #[test]
    fn test_dashboard_creation() {
        let dashboard = create_test_dashboard();
        assert_eq!(dashboard.config().max_connections, 5);
        assert!(!dashboard.is_running());
    }

    #[test]
    fn test_dashboard_start_stop() {
        let dashboard = create_test_dashboard();

        // Mock starting
        dashboard.is_running.lock().unwrap().clone_from(&true);
        assert!(dashboard.is_running());

        // Mock stopping
        dashboard.is_running.lock().unwrap().clone_from(&false);
        assert!(!dashboard.is_running());
    }

    #[test]
    fn test_authenticate_success() {
        let dashboard = create_test_dashboard();
        let result = dashboard.authenticate("admin", "admin");

        assert!(result.is_ok());
        let token = result.unwrap();
        assert_eq!(token.user, "admin");
    }

    #[test]
    fn test_authenticate_failure() {
        let dashboard = create_test_dashboard();
        let result = dashboard.authenticate("admin", "wrong_password");

        assert!(result.is_err());
    }

    #[test]
    fn test_token_validation() {
        let dashboard = create_test_dashboard();

        // Generate token
        let token = dashboard.generate_token("test_user");

        // Add token
        dashboard.add_token(&token);

        // Validate token
        let valid = dashboard.validate_token(&token.token).unwrap();
        assert!(valid);
    }

    #[test]
    fn test_token_validation_invalid() {
        let dashboard = create_test_dashboard();
        let valid = dashboard.validate_token("invalid_token").unwrap();

        assert!(!valid);
    }

    #[test]
    fn test_connect_client() {
        let dashboard = create_test_dashboard();

        // Add token first
        let token = dashboard.generate_token("test_user");
        dashboard.add_token(&token);

        // Connect client
        let result = dashboard.connect_client("client1", "127.0.0.1", &token.token);
        assert!(result.is_ok());

        assert_eq!(dashboard.connected_clients_count(), 1);
    }

    #[test]
    fn test_connect_client_invalid_token() {
        let dashboard = create_test_dashboard();

        let result = dashboard.connect_client("client1", "127.0.0.1", "invalid_token");
        assert!(result.is_err());
    }

    #[test]
    fn test_disconnect_client() {
        let dashboard = create_test_dashboard();

        // Add token and connect
        let token = dashboard.generate_token("test_user");
        dashboard.add_token(&token);
        dashboard.connect_client("client1", "127.0.0.1", &token.token).unwrap();

        assert_eq!(dashboard.connected_clients_count(), 1);

        // Disconnect
        dashboard.disconnect_client("client1");
        assert_eq!(dashboard.connected_clients_count(), 0);
    }

    #[test]
    fn test_is_at_capacity() {
        let dashboard = create_test_dashboard();
        assert!(!dashboard.is_at_capacity());

        // Fill to capacity (max_connections = 5)
        let token = dashboard.generate_token("test_user");
        dashboard.add_token(&token);

        // First 5 connections should succeed
        for i in 0..5 {
            let result = dashboard.connect_client(&format!("client{}", i), "127.0.0.1", &token.token);
            assert!(result.is_ok(), "Connection {} should succeed", i);
        }

        // 6th connection should fail
        let result = dashboard.connect_client("client5", "127.0.0.1", &token.token);
        assert!(result.is_err(), "6th connection should fail");

        assert!(dashboard.is_at_capacity());
    }

    #[test]
    fn test_process_command_emergency_stop() {
        let dashboard = create_test_dashboard();
        let command = DashboardCommand::EmergencyStop;

        let result = dashboard.process_command(command, "admin", "127.0.0.1");

        match result {
            CommandResult::Success { .. } => {
                let state = dashboard.get_state();
                assert_eq!(state.operation_mode, DashboardOperationMode::Emergency);
            }
            _ => panic!("Expected Success result"),
        }
    }

    #[test]
    fn test_process_command_set_parameter() {
        let dashboard = create_test_dashboard();
        let command = DashboardCommand::SetParameter {
            name: "gain".to_string(),
            value: serde_json::json!(0.8),
        };

        let result = dashboard.process_command(command, "admin", "127.0.0.1");

        match result {
            CommandResult::Success { .. } => {}
            _ => panic!("Expected Success result"),
        }
    }

    #[test]
    fn test_get_state() {
        let dashboard = create_test_dashboard();
        let state = dashboard.get_state();

        assert_eq!(state.operation_mode, DashboardOperationMode::Passthrough);
        assert_eq!(state.active_connections, 0);
    }

    #[test]
    fn test_update_battery() {
        let dashboard = create_test_dashboard();
        dashboard.update_battery(75.0);

        let state = dashboard.get_state();
        assert_eq!(state.battery_level, 75.0);
    }

    #[test]
    fn test_update_battery_clamps() {
        let dashboard = create_test_dashboard();
        dashboard.update_battery(150.0); // Above max

        let state = dashboard.get_state();
        assert_eq!(state.battery_level, 100.0); // Clamped to max
    }

    #[test]
    fn test_update_temperature() {
        let dashboard = create_test_dashboard();
        dashboard.update_temperature(35.0);

        let state = dashboard.get_state();
        assert_eq!(state.temperature_celsius, 35.0);
    }

    #[test]
    fn test_update_operation_mode() {
        let dashboard = create_test_dashboard();
        dashboard.update_operation_mode(DashboardOperationMode::Interactive);

        let state = dashboard.get_state();
        assert_eq!(state.operation_mode, DashboardOperationMode::Interactive);
    }

    #[test]
    fn test_update_iacuc_status() {
        let dashboard = create_test_dashboard();
        dashboard.update_iacuc_status(IacucStatus::Warning);

        let state = dashboard.get_state();
        assert_eq!(state.iacuc_status, IacucStatus::Warning);
    }

    #[test]
    fn test_command_logging() {
        let dashboard = create_test_dashboard();
        let command = DashboardCommand::GetStatus;

        dashboard.process_command(command, "test_user", "10.0.0.1");

        let log = dashboard.get_command_log();
        assert_eq!(log.len(), 1);
        assert_eq!(log[0].user, "test_user");
        assert_eq!(log[0].ip_address, "10.0.0.1");
    }

    #[test]
    fn test_heartbeat_check() {
        let dashboard = create_test_dashboard();

        // Add some connections
        let token = dashboard.generate_token("test_user");
        dashboard.add_token(&token);
        dashboard.connect_client("client1", "127.0.0.1", &token.token).unwrap();
        dashboard.connect_client("client2", "127.0.0.1", &token.token).unwrap();

        assert_eq!(dashboard.connected_clients_count(), 2);

        // Heartbeat check should not remove recent connections
        dashboard.heartbeat_check();
        assert_eq!(dashboard.connected_clients_count(), 2);
    }

    #[test]
    fn test_command_result_serialization() {
        let result = CommandResult::Success {
            message: "Test success".to_string(),
        };

        let serialized = serde_json::to_string(&result).unwrap();
        assert!(serialized.contains("Success"));
    }

    #[test]
    fn test_ws_message_serialization() {
        let msg = WsMessage::GaugeUpdate {
            name: "temperature".to_string(),
            value: 25.5,
            unit: "°C".to_string(),
        };

        let serialized = serde_json::to_string(&msg).unwrap();
        assert!(serialized.contains("GaugeUpdate"));
    }
}
