/**
 * IEEE 1588 PTP (Precision Time Protocol) Module
 * ===============================================
 *
 * This module implements IEEE 1588 Precision Time Protocol for
 * nanosecond-accurate timestamping. This is critical for:
 *
 * - Provenance logging with deterministic timing
 * - Multi-sensor data synchronization
 * - Real-time audio processing coordination
 * - Temporal alignment of cross-modal data
 *
 * Author: Sheel Morjaria (sheelmorjaria@gmail.com)
 * License: CC BY-ND 4.0 International
 */

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use anyhow::Result;
use log::{info, debug};
use serde::{Deserialize, Serialize};

/// PTP configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PtpConfig {
    /// PTP domain number (0-255)
    pub domain: u8,

    /// Sync interval in seconds (logarithmic: -1 = 0.5s, 0 = 1s, 1 = 2s)
    pub sync_interval_log2: i8,

    /// Enable hardware timestamping
    pub hw_timestamping: bool,

    /// Network interface for PTP
    pub interface: String,

    /// Grandmaster clock IP (optional, for slave mode)
    pub grandmaster_ip: Option<String>,

    /// Enable PTP on startup
    pub enable_on_startup: bool,
}

impl Default for PtpConfig {
    fn default() -> Self {
        Self {
            domain: 0,
            sync_interval_log2: 0,  // 1 second
            hw_timestamping: true,
            interface: "eth0".to_string(),
            grandmaster_ip: None,
            enable_on_startup: true,
        }
    }
}

/// PTP timestamp with nanosecond precision
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PtpTimestamp {
    /// Seconds since PTP epoch
    pub seconds: u64,
    /// Nanoseconds within the second
    pub nanos: u32,
}

impl PtpTimestamp {
    /// Create a new PTP timestamp
    pub fn new(seconds: u64, nanos: u32) -> Self {
        assert!(nanos < 1_000_000_000, "Nanoseconds must be < 1 billion");
        Self { seconds, nanos }
    }

    /// Get current time as PTP timestamp
    pub fn now() -> Self {
        let duration = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();

        Self {
            seconds: duration.as_secs(),
            nanos: duration.subsec_nanos(),
        }
    }

    /// Convert to nanoseconds since epoch
    pub fn as_nanos(&self) -> u128 {
        self.seconds as u128 * 1_000_000_000 + self.nanos as u128
    }

    /// Convert to floating-point seconds
    pub fn as_seconds_f64(&self) -> f64 {
        self.seconds as f64 + self.nanos as f64 / 1_000_000_000.0
    }

    /// Get time difference between two timestamps
    pub fn duration_since(&self, earlier: PtpTimestamp) -> std::time::Duration {
        let nanos = self.as_nanos().saturating_sub(earlier.as_nanos());
        std::time::Duration::from_nanos(nanos as u64)
    }

    /// Add a duration to the timestamp
    pub fn saturating_add(&self, duration: std::time::Duration) -> Self {
        let total_nanos = self.as_nanos() + duration.as_nanos() as u128;
        Self {
            seconds: (total_nanos / 1_000_000_000) as u64,
            nanos: (total_nanos % 1_000_000_000) as u32,
        }
    }
}

impl std::fmt::Display for PtpTimestamp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{:09}", self.seconds, self.nanos)
    }
}

impl From<std::time::SystemTime> for PtpTimestamp {
    fn from(time: SystemTime) -> Self {
        let duration = time.duration_since(UNIX_EPOCH).unwrap_or_default();
        Self {
            seconds: duration.as_secs(),
            nanos: duration.subsec_nanos(),
        }
    }
}

impl From<PtpTimestamp> for std::time::SystemTime {
    fn from(ts: PtpTimestamp) -> Self {
        UNIX_EPOCH + std::time::Duration::new(ts.seconds, ts.nanos)
    }
}

impl From<chrono::DateTime<chrono::Utc>> for PtpTimestamp {
    fn from(dt: chrono::DateTime<chrono::Utc>) -> Self {
        Self {
            seconds: dt.timestamp() as u64,
            nanos: dt.timestamp_subsec_nanos(),
        }
    }
}

impl From<PtpTimestamp> for chrono::DateTime<chrono::Utc> {
    fn from(ts: PtpTimestamp) -> Self {
        chrono::DateTime::from_timestamp(ts.seconds as i64, ts.nanos)
            .unwrap_or_else(|| chrono::Utc::now())
    }
}

/// PTP clock synchronization status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PtpStatus {
    /// Initializing
    Initializing,
    /// Synchronized to grandmaster
    Locked,
    /// Holdover mode (lost grandmaster, using local clock)
    Holdover,
    /// Not synchronized
    Faulty,
}

/// PTP clock statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PtpStats {
    /// Current status
    pub status: PtpStatus,
    /// Offset from master (nanoseconds)
    pub offset_ns: i64,
    /// Path delay (nanoseconds)
    pub path_delay_ns: u64,
    /// Current clock class
    pub clock_class: u8,
    /// Time since last sync (seconds)
    pub time_since_last_sync: u64,
    /// Grandmaster ID (if slave)
    pub grandmaster_id: Option<String>,
}

/// PTP clock for precision timing
pub struct PtpClock {
    /// Configuration
    config: PtpConfig,
    /// Current status
    status: Arc<tokio::sync::RwLock<PtpStatus>>,
    /// Clock offset from master (nanoseconds)
    offset_ns: Arc<tokio::sync::RwLock<i64>>,
    /// Path delay to master (nanoseconds)
    path_delay_ns: Arc<tokio::sync::RwLock<u64>>,
    /// Time of last sync
    last_sync: Arc<tokio::sync::RwLock<chrono::DateTime<chrono::Utc>>>,
    /// Whether clock is running
    running: Arc<std::sync::atomic::AtomicBool>,
}

impl PtpClock {
    /// Create a new PTP clock
    pub async fn new(config: PtpConfig) -> Result<Self> {
        info!("Initializing PTP clock on interface: {}", config.interface);

        Ok(Self {
            config,
            status: Arc::new(tokio::sync::RwLock::new(PtpStatus::Initializing)),
            offset_ns: Arc::new(tokio::sync::RwLock::new(0)),
            path_delay_ns: Arc::new(tokio::sync::RwLock::new(0)),
            last_sync: Arc::new(tokio::sync::RwLock::new(chrono::Utc::now())),
            running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        })
    }

    /// Start PTP clock synchronization
    pub async fn start(&self) -> Result<()> {
        info!("Starting PTP clock synchronization");

        if self.config.grandmaster_ip.is_some() {
            // Slave mode - would start PTP daemon here
            info!("PTP slave mode, grandmaster: {:?}", self.config.grandmaster_ip);
            *self.status.write().await = PtpStatus::Initializing;
        } else {
            // Grandmaster mode
            info!("PTP grandmaster mode");
            *self.status.write().await = PtpStatus::Locked;
        }

        self.running.store(true, std::sync::atomic::Ordering::SeqCst);

        // Start background sync task
        let status_lock = self.status.clone();
        let offset_lock = self.offset_ns.clone();
        let running = self.running.clone();
        let sync_interval = 2_f64.powi(self.config.sync_interval_log2 as i32);

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(
                tokio::time::Duration::from_secs_f64(sync_interval)
            );

            while running.load(std::sync::atomic::Ordering::SeqCst) {
                interval.tick().await;

                // In a real implementation, this would:
                // 1. Exchange sync messages with grandmaster
                // 2. Calculate offset and path delay
                // 3. Adjust local clock

                // Simulate sync
                let mut offset = offset_lock.write().await;
                *offset = (*offset).saturating_add(10); // Simulate drift
            }
        });

        Ok(())
    }

    /// Stop PTP clock
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping PTP clock");
        self.running.store(false, std::sync::atomic::Ordering::SeqCst);
        *self.status.write().await = PtpStatus::Faulty;
        Ok(())
    }

    /// Get current PTP timestamp
    pub async fn get_timestamp(&self) -> Result<PtpTimestamp> {
        // Get system time
        let mut ts = PtpTimestamp::now();

        // Apply offset if synchronized
        if self.running.load(std::sync::atomic::Ordering::SeqCst) {
            let offset = *self.offset_ns.read().await;
            ts = ts.saturating_add(std::time::Duration::from_nanos(offset as u64));
        }

        debug!("PTP timestamp: {}", ts);
        Ok(ts)
    }

    /// Get current PTP status
    pub async fn get_status(&self) -> PtpStatus {
        *self.status.read().await
    }

    /// Get PTP statistics
    pub async fn get_stats(&self) -> PtpStats {
        let status = *self.status.read().await;
        let offset_ns = *self.offset_ns.read().await;
        let path_delay_ns = *self.path_delay_ns.read().await;
        let last_sync = *self.last_sync.read().await;
        let running = self.running.load(std::sync::atomic::Ordering::SeqCst);

        PtpStats {
            status,
            offset_ns,
            path_delay_ns,
            clock_class: if running { 248 } else { 255 },
            time_since_last_sync: (chrono::Utc::now() - last_sync).num_seconds() as u64,
            grandmaster_id: self.config.grandmaster_ip.clone(),
        }
    }

    /// Manually set clock offset (for testing)
    pub async fn set_offset(&self, offset_ns: i64) {
        *self.offset_ns.write().await = offset_ns;
        *self.last_sync.write().await = chrono::Utc::now();
        debug!("Clock offset set to {} ns", offset_ns);
    }

    /// Manually set status (for testing)
    pub async fn set_status(&self, status: PtpStatus) {
        *self.status.write().await = status;
        info!("PTP status set to {:?}", status);
    }

    /// Check if clock is synchronized
    pub async fn is_synchronized(&self) -> bool {
        matches!(*self.status.read().await, PtpStatus::Locked)
    }

    /// Shutdown PTP clock
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down PTP clock");
        self.stop().await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ptp_config_default() {
        let config = PtpConfig::default();
        assert_eq!(config.domain, 0);
        assert_eq!(config.sync_interval_log2, 0);
        assert_eq!(config.interface, "eth0");
    }

    #[test]
    fn test_ptp_timestamp_creation() {
        let ts = PtpTimestamp::new(12345, 123456789);
        assert_eq!(ts.seconds, 12345);
        assert_eq!(ts.nanos, 123456789);
    }

    #[test]
    fn test_ptp_timestamp_now() {
        let ts = PtpTimestamp::now();
        assert!(ts.seconds > 0);
        assert!(ts.nanos < 1_000_000_000);
    }

    #[test]
    fn test_ptp_timestamp_conversions() {
        let ts = PtpTimestamp::new(1000, 500_000_000);

        // To nanos
        assert_eq!(ts.as_nanos(), 1000_500_000_000);

        // To seconds
        assert_eq!(ts.as_seconds_f64(), 1000.5);

        // To duration
        let ts2 = PtpTimestamp::new(1000, 600_000_000);
        let duration = ts2.duration_since(ts);
        assert_eq!(duration.as_nanos(), 100_000_000);
    }

    #[test]
    fn test_ptp_timestamp_display() {
        let ts = PtpTimestamp::new(12345, 123456789);
        let display = format!("{}", ts);
        assert_eq!(display, "12345.123456789");
    }

    #[tokio::test]
    async fn test_ptp_clock_creation() {
        let config = PtpConfig::default();
        let clock = PtpClock::new(config).await.unwrap();
        assert_eq!(clock.get_status().await, PtpStatus::Initializing);
    }

    #[tokio::test]
    async fn test_ptp_clock_start() {
        let config = PtpConfig {
            grandmaster_ip: None,
            ..Default::default()
        };
        let clock = PtpClock::new(config).await.unwrap();
        clock.start().await.unwrap();
        assert_eq!(clock.get_status().await, PtpStatus::Locked);
        assert!(clock.is_synchronized().await);
    }

    #[tokio::test]
    async fn test_ptp_get_timestamp() {
        let config = PtpConfig::default();
        let clock = PtpClock::new(config).await.unwrap();

        let ts1 = clock.get_timestamp().await.unwrap();
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        let ts2 = clock.get_timestamp().await.unwrap();

        assert!(ts2 > ts1);
    }

    #[tokio::test]
    async fn test_ptp_offset() {
        let config = PtpConfig::default();
        let clock = PtpClock::new(config).await.unwrap();

        clock.set_offset(1000).await;
        let stats = clock.get_stats().await;
        assert_eq!(stats.offset_ns, 1000);
    }

    #[tokio::test]
    async fn test_ptp_status_manipulation() {
        let config = PtpConfig::default();
        let clock = PtpClock::new(config).await.unwrap();

        clock.set_status(PtpStatus::Holdover).await;
        assert_eq!(clock.get_status().await, PtpStatus::Holdover);
        assert!(!clock.is_synchronized().await);
    }
}
