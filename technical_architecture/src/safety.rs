//! Safety Monitor Module
//! ====================
//!
//! This module implements safety monitoring and protection for the
//! field deployment system. It includes:
//!
//! - Watchdog timer for hang detection
//! - Performance budget monitoring
//! - Emergency shutdown triggers
//! - Safety violation logging and reporting
//!
//! Author: Sheel Morjaria (sheelmorjaria@gmail.com)
//! License: CC BY-ND 4.0 International

use std::time::{Duration, Instant};
use anyhow::Result;
use log::{info, debug, warn, error};
use serde::{Deserialize, Serialize};

/// Safety configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyConfig {
    /// Watchdog timeout in milliseconds
    pub watchdog_timeout_ms: u64,

    /// Maximum processing time per frame (ms)
    pub max_frame_time_ms: f64,

    /// Maximum consecutive errors before shutdown
    pub max_consecutive_errors: usize,

    /// Memory usage threshold (bytes)
    pub memory_threshold_bytes: usize,

    /// Enable automatic emergency shutdown
    pub auto_emergency_shutdown: bool,

    /// Safety check interval (milliseconds)
    pub check_interval_ms: u64,
}

impl Default for SafetyConfig {
    fn default() -> Self {
        Self {
            watchdog_timeout_ms: 5000,  // 5 seconds
            max_frame_time_ms: 100.0,   // 100ms budget
            max_consecutive_errors: 10,
            memory_threshold_bytes: 2_000_000_000,  // 2GB
            auto_emergency_shutdown: true,
            check_interval_ms: 100,
        }
    }
}

/// Safety violation record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyViolation {
    /// Type of violation
    pub violation_type: String,
    /// Severity level
    pub severity: String,
    /// Timestamp of violation
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl SafetyViolation {
    /// Create a new safety violation
    pub fn new(violation_type: &str, severity: &str) -> Self {
        Self {
            violation_type: violation_type.to_string(),
            severity: severity.to_string(),
            timestamp: chrono::Utc::now(),
        }
    }

    /// Create a critical violation
    pub fn critical(violation_type: &str) -> Self {
        Self::new(violation_type, "CRITICAL")
    }

    /// Create a warning violation
    pub fn warning(violation_type: &str) -> Self {
        Self::new(violation_type, "WARNING")
    }
}

/// Safety check result
#[derive(Debug, Clone)]
pub struct SafetyCheck {
    /// Whether all checks passed
    pub is_safe: bool,
    /// Any violations found
    pub violations: Vec<SafetyViolation>,
    /// Current system metrics
    pub metrics: SafetyMetrics,
}

/// Safety metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyMetrics {
    /// Current memory usage (bytes)
    pub memory_usage_bytes: usize,
    /// Last frame processing time (ms)
    pub last_frame_time_ms: f64,
    /// Consecutive error count
    pub consecutive_errors: usize,
    /// Watchdog last fed time
    pub watchdog_last_fed: Option<chrono::DateTime<chrono::Utc>>,
    /// System uptime (seconds)
    pub uptime_seconds: u64,
}

/// Watchdog timer for hang detection
pub struct WatchdogTimer {
    /// Last time the watchdog was fed
    last_fed: std::sync::Mutex<Instant>,
    /// Watchdog timeout
    timeout: Duration,
    /// Whether watchdog is enabled
    enabled: std::sync::atomic::AtomicBool,
}

impl WatchdogTimer {
    /// Create a new watchdog timer
    pub fn new(timeout_ms: u64) -> Self {
        Self {
            last_fed: std::sync::Mutex::new(Instant::now()),
            timeout: Duration::from_millis(timeout_ms),
            enabled: std::sync::atomic::AtomicBool::new(true),
        }
    }

    /// Feed the watchdog (reset timer)
    pub fn feed(&self) {
        *self.last_fed.lock().unwrap() = Instant::now();
    }

    /// Check if watchdog has expired
    pub fn is_expired(&self) -> bool {
        if !self.enabled.load(std::sync::atomic::Ordering::SeqCst) {
            return false;
        }
        let last = *self.last_fed.lock().unwrap();
        last.elapsed() > self.timeout
    }

    /// Enable the watchdog
    pub fn enable(&self) {
        self.enabled.store(true, std::sync::atomic::Ordering::SeqCst);
        self.feed();
    }

    /// Disable the watchdog
    pub fn disable(&self) {
        self.enabled.store(false, std::sync::atomic::Ordering::SeqCst);
    }

    /// Get time since last feed
    pub fn time_since_last_feed(&self) -> Duration {
        let last = *self.last_fed.lock().unwrap();
        last.elapsed()
    }
}

/// Safety monitor for the system
pub struct SafetyMonitor {
    /// Configuration
    config: SafetyConfig,
    /// Watchdog timer
    watchdog: WatchdogTimer,
    /// Start time for uptime tracking
    start_time: Instant,
    /// Last frame processing time
    last_frame_time: tokio::sync::RwLock<Option<f64>>,
    /// Consecutive error count
    consecutive_errors: tokio::sync::RwLock<usize>,
    /// Safety violation history
    violations: tokio::sync::RwLock<Vec<SafetyViolation>>,
    /// Maximum violation history
    max_violations: usize,
    /// Whether monitoring is active
    monitoring_active: std::sync::atomic::AtomicBool,
}

impl SafetyMonitor {
    /// Create a new safety monitor
    pub async fn new(config: SafetyConfig) -> Result<Self> {
        info!("Initializing Safety Monitor");

        let watchdog = WatchdogTimer::new(config.watchdog_timeout_ms);

        Ok(Self {
            config,
            watchdog,
            start_time: Instant::now(),
            last_frame_time: tokio::sync::RwLock::new(None),
            consecutive_errors: tokio::sync::RwLock::new(0),
            violations: tokio::sync::RwLock::new(Vec::new()),
            max_violations: 1000,
            monitoring_active: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// Start safety monitoring
    pub async fn start_monitoring(&self) -> Result<()> {
        info!("Starting safety monitoring");
        self.monitoring_active.store(true, std::sync::atomic::Ordering::SeqCst);
        self.watchdog.enable();
        Ok(())
    }

    /// Stop safety monitoring
    pub async fn stop_monitoring(&self) -> Result<()> {
        info!("Stopping safety monitoring");
        self.monitoring_active.store(false, std::sync::atomic::Ordering::SeqCst);
        self.watchdog.disable();
        Ok(())
    }

    /// Run safety check
    pub async fn check_safety(&self) -> Result<SafetyCheck> {
        let mut violations = Vec::new();

        // Check watchdog
        if self.watchdog.is_expired() {
            let violation = SafetyViolation::critical("WATCHDOG_EXPIRED");
            warn!("Watchdog expired: {:?}", self.watchdog.time_since_last_feed());
            violations.push(violation);
        }

        // Check frame time budget
        if let Some(frame_time) = *self.last_frame_time.read().await {
            if frame_time > self.config.max_frame_time_ms {
                violations.push(SafetyViolation::warning("FRAME_TIME_EXCEEDED"));
            }
        }

        // Check consecutive errors
        let error_count = *self.consecutive_errors.read().await;
        if error_count >= self.config.max_consecutive_errors {
            violations.push(SafetyViolation::critical("MAX_CONSECUTIVE_ERRORS"));
        }

        // Check memory usage (mock)
        let memory_usage = self.get_memory_usage().await?;
        if memory_usage > self.config.memory_threshold_bytes {
            violations.push(SafetyViolation::warning("MEMORY_THRESHOLD_EXCEEDED"));
        }

        // Build metrics
        let frame_time = self.last_frame_time.read().await;
        let frame_time = frame_time.as_ref().copied().unwrap_or(0.0);
        let metrics = SafetyMetrics {
            memory_usage_bytes: memory_usage,
            last_frame_time_ms: frame_time,
            consecutive_errors: error_count,
            watchdog_last_fed: Some(chrono::Utc::now()),
            uptime_seconds: self.start_time.elapsed().as_secs(),
        };

        let is_safe = violations.is_empty();

        Ok(SafetyCheck {
            is_safe,
            violations,
            metrics,
        })
    }

    /// Monitor safety state (called periodically)
    pub async fn monitor(&self) -> Result<()> {
        if !self.monitoring_active.load(std::sync::atomic::Ordering::SeqCst) {
            return Ok(());
        }

        let check = self.check_safety().await?;

        if !check.is_safe {
            // Log violations
            for violation in &check.violations {
                self.log_violation(violation.clone()).await;

                // Trigger emergency shutdown for critical violations
                if violation.severity == "CRITICAL" && self.config.auto_emergency_shutdown {
                    error!("Critical safety violation: {:?}", violation.violation_type);
                    self.trigger_shutdown(violation.clone()).await?;
                }
            }
        }

        Ok(())
    }

    /// Log a safety violation
    async fn log_violation(&self, violation: SafetyViolation) {
        let mut violations = self.violations.write().await;
        violations.push(violation.clone());
        if violations.len() > self.max_violations {
            violations.remove(0);
        }
        warn!("Safety violation logged: {:?}", violation);
    }

    /// Trigger emergency shutdown
    pub async fn trigger_shutdown(&self, violation: SafetyViolation) -> Result<()> {
        error!("EMERGENCY SHUTDOWN triggered by: {:?}", violation.violation_type);

        // In a real implementation, this would:
        // 1. Stop all processing
        // 2. Save critical data
        // 3. Signal shutdown to main system
        // 4. Possibly power down hardware

        self.monitoring_active.store(false, std::sync::atomic::Ordering::SeqCst);

        Ok(())
    }

    /// Feed the watchdog
    pub async fn feed_watchdog(&self) {
        self.watchdog.feed();
        debug!("Watchdog fed");
    }

    /// Report frame processing time
    pub async fn report_frame_time(&self, time_ms: f64) {
        *self.last_frame_time.write().await = Some(time_ms);

        // Reset consecutive errors on successful frame
        if time_ms <= self.config.max_frame_time_ms {
            *self.consecutive_errors.write().await = 0;
        }

        // Feed watchdog as part of normal operation
        self.watchdog.feed();
    }

    /// Report an error
    pub async fn report_error(&self) {
        *self.consecutive_errors.write().await += 1;
        warn!("Error reported, consecutive errors: {}", *self.consecutive_errors.read().await);
    }

    /// Get current memory usage (mock implementation)
    async fn get_memory_usage(&self) -> Result<usize> {
        // In a real implementation, this would query the OS
        // For now, return a mock value
        Ok(500_000_000) // 500MB
    }

    /// Get safety statistics
    pub async fn get_stats(&self) -> SafetyStats {
        let uptime = self.start_time.elapsed().as_secs();
        let violations = self.violations.read().await.clone();
        let consecutive_errors = *self.consecutive_errors.read().await;
        let last_frame_time = *self.last_frame_time.read().await;
        let monitoring_active = self.monitoring_active.load(std::sync::atomic::Ordering::SeqCst);

        SafetyStats {
            uptime_seconds: uptime,
            total_violations: violations.len(),
            recent_violations: violations.into_iter().rev().take(10).collect(),
            consecutive_errors,
            last_frame_time_ms: last_frame_time,
            watchdog_expired: self.watchdog.is_expired(),
            monitoring_active,
        }
    }

    /// Get violation history
    pub async fn get_violations(&self) -> Vec<SafetyViolation> {
        self.violations.read().await.clone()
    }

    /// Clear violation history
    pub async fn clear_violations(&self) {
        self.violations.write().await.clear();
    }
}

/// Safety statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyStats {
    pub uptime_seconds: u64,
    pub total_violations: usize,
    pub recent_violations: Vec<SafetyViolation>,
    pub consecutive_errors: usize,
    pub last_frame_time_ms: Option<f64>,
    pub watchdog_expired: bool,
    pub monitoring_active: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_safety_config_default() {
        let config = SafetyConfig::default();
        assert_eq!(config.watchdog_timeout_ms, 5000);
        assert_eq!(config.max_frame_time_ms, 100.0);
    }

    #[tokio::test]
    async fn test_watchdog_timer() {
        let watchdog = WatchdogTimer::new(100); // 100ms timeout

        assert!(!watchdog.is_expired());
        watchdog.feed();
        assert!(!watchdog.is_expired());

        // Simulate timeout
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert!(watchdog.is_expired());

        watchdog.feed();
        assert!(!watchdog.is_expired());
    }

    #[tokio::test]
    async fn test_safety_monitor_creation() {
        let config = SafetyConfig::default();
        let monitor = SafetyMonitor::new(config).await.unwrap();
        let check = monitor.check_safety().await.unwrap();
        assert!(check.is_safe);
    }

    #[tokio::test]
    async fn test_frame_time_reporting() {
        let config = SafetyConfig::default();
        let monitor = SafetyMonitor::new(config).await.unwrap();

        monitor.report_frame_time(50.0).await;
        let stats = monitor.get_stats().await;
        assert_eq!(stats.last_frame_time_ms, Some(50.0));
        assert_eq!(stats.consecutive_errors, 0);
    }

    #[tokio::test]
    async fn test_frame_time_violation() {
        let config = SafetyConfig {
            max_frame_time_ms: 100.0,
            ..Default::default()
        };
        let monitor = SafetyMonitor::new(config).await.unwrap();

        monitor.report_frame_time(150.0).await;
        let check = monitor.check_safety().await.unwrap();

        assert!(!check.is_safe);
        assert!(check.violations.iter().any(|v| v.violation_type == "FRAME_TIME_EXCEEDED"));
    }

    #[tokio::test]
    async fn test_error_reporting() {
        let config = SafetyConfig::default();
        let monitor = SafetyMonitor::new(config).await.unwrap();

        monitor.report_error().await;
        monitor.report_error().await;

        let stats = monitor.get_stats().await;
        assert_eq!(stats.consecutive_errors, 2);
    }

    #[tokio::test]
    async fn test_violation_logging() {
        let config = SafetyConfig::default();
        let monitor = SafetyMonitor::new(config).await.unwrap();

        let violation = SafetyViolation::warning("TEST_VIOLATION");
        monitor.log_violation(violation.clone()).await;

        let violations = monitor.get_violations().await;
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].violation_type, "TEST_VIOLATION");
    }

    #[tokio::test]
    async fn test_watchdog_expiration_detection() {
        let config = SafetyConfig {
            watchdog_timeout_ms: 100,
            ..Default::default()
        };
        let monitor = SafetyMonitor::new(config).await.unwrap();
        monitor.start_monitoring().await.unwrap();

        // Wait for watchdog to expire
        tokio::time::sleep(Duration::from_millis(150)).await;

        let check = monitor.check_safety().await.unwrap();
        assert!(!check.is_safe);
        assert!(check.violations.iter().any(|v| v.violation_type == "WATCHDOG_EXPIRED"));
    }
}
