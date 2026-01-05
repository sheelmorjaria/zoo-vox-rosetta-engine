/**
 * Provenance Logging Module
 * ==========================
 *
 * This module implements deterministic provenance logging for the
 * field deployment system. It tracks:
 *
 * - All audio processing decisions
 * - Model inference results
 * - Safety interventions
 * - System state changes
 * - Data lineage and audit trail
 *
 * Author: Sheel Morjaria (sheelmorjaria@gmail.com)
 * License: CC BY-ND 4.0 International
 */

use std::path::PathBuf;
use std::sync::Arc;
use anyhow::{Result, Context};
use log::{info, debug};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex as TokioMutex;
use chrono::{DateTime, Utc};

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log file path
    pub log_file_path: PathBuf,

    /// Maximum log file size (bytes)
    pub max_file_size: usize,

    /// Number of backup files to keep
    pub num_backups: usize,

    /// Enable async logging
    pub async_logging: bool,

    /// Log level filter
    pub log_level: String,

    /// Include PTP timestamps in logs
    pub include_ptp_timestamps: bool,

    /// Enable compression for backup logs
    pub compress_backups: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            log_file_path: PathBuf::from("logs/provenance.log"),
            max_file_size: 100_000_000,  // 100MB
            num_backups: 5,
            async_logging: true,
            log_level: "info".to_string(),
            include_ptp_timestamps: true,
            compress_backups: true,
        }
    }
}

/// Provenance log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    /// Unique entry ID
    pub id: String,
    /// Timestamp (UTC)
    pub timestamp: DateTime<Utc>,
    /// PTP timestamp (optional)
    pub ptp_timestamp: Option<u128>,
    /// Entry type/category
    pub entry_type: String,
    /// Severity level
    pub severity: String,
    /// Component/module that generated the entry
    pub component: String,
    /// Log message
    pub message: String,
    /// Additional structured data
    pub data: Option<serde_json::Value>,
    /// Chain of causality (parent entry IDs)
    pub causality: Vec<String>,
}

impl LogEntry {
    /// Create a new log entry
    pub fn new(entry_type: &str, component: &str, message: &str) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            ptp_timestamp: None,
            entry_type: entry_type.to_string(),
            severity: "info".to_string(),
            component: component.to_string(),
            message: message.to_string(),
            data: None,
            causality: Vec::new(),
        }
    }

    /// Set severity level
    pub fn with_severity(mut self, severity: &str) -> Self {
        self.severity = severity.to_string();
        self
    }

    /// Set PTP timestamp
    pub fn with_ptp_timestamp(mut self, ts: u128) -> Self {
        self.ptp_timestamp = Some(ts);
        self
    }

    /// Set structured data
    pub fn with_data(mut self, data: serde_json::Value) -> Self {
        self.data = Some(data);
        self
    }

    /// Add causality parent
    pub fn with_parent(mut self, parent_id: String) -> Self {
        self.causality.push(parent_id);
        self
    }

    /// Convert to JSON string
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .context("Failed to serialize log entry to JSON")
    }
}

/// Provenance logger
pub struct ProvenanceLogger {
    /// Configuration
    config: LoggingConfig,
    /// Log entries buffer
    entries: Arc<TokioMutex<Vec<LogEntry>>>,
    /// Current log file size
    file_size: Arc<TokioMutex<usize>>,
    /// Whether logging is active
    active: Arc<std::sync::atomic::AtomicBool>,
}

impl ProvenanceLogger {
    /// Create a new provenance logger
    pub async fn new(config: LoggingConfig) -> Result<Self> {
        info!("Initializing Provenance Logger: {:?}", config.log_file_path);

        // Create log directory if it doesn't exist
        if let Some(parent) = config.log_file_path.parent() {
            tokio::fs::create_dir_all(parent).await
                .context("Failed to create log directory")?;
        }

        Ok(Self {
            config,
            entries: Arc::new(TokioMutex::new(Vec::new())),
            file_size: Arc::new(TokioMutex::new(0)),
            active: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        })
    }

    /// Start logging
    pub async fn start(&self) -> Result<()> {
        info!("Starting Provenance Logger");
        self.active.store(true, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }

    /// Stop logging
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping Provenance Logger");
        self.active.store(false, std::sync::atomic::Ordering::SeqCst);
        self.flush().await?;
        Ok(())
    }

    /// Log a decision with PTP timestamp
    pub async fn log_decision(&self, decision: &str, ptp_ts: crate::ptp::PtpTimestamp) -> Result<String> {
        let entry = LogEntry::new("decision", "technical_arch", decision)
            .with_ptp_timestamp(ptp_ts.as_nanos())
            .with_data(serde_json::json!({
                "decision_type": decision,
                "ptp_seconds": ptp_ts.seconds,
                "ptp_nanos": ptp_ts.nanos,
            }));

        self.log_entry(entry).await
    }

    /// Log a generic event
    pub async fn log_event(&self, component: &str, event_type: &str, message: &str) -> Result<String> {
        let entry = LogEntry::new(event_type, component, message);
        self.log_entry(entry).await
    }

    /// Log a safety event
    pub async fn log_safety(&self, event: &str, severity: &str) -> Result<String> {
        let entry = LogEntry::new("safety", "safety_monitor", event)
            .with_severity(severity);
        self.log_entry(entry).await
    }

    /// Log an emergency event with PTP timestamp
    ///
    /// This is for safety-critical events that require immediate attention
    /// and must be logged for provenance and audit purposes.
    pub async fn log_emergency_event(&self, event: &str, ptp_ts: crate::ptp::PtpTimestamp) -> Result<String> {
        let entry = LogEntry::new("emergency", "technical_arch", event)
            .with_severity("CRITICAL")
            .with_ptp_timestamp(ptp_ts.as_nanos())
            .with_data(serde_json::json!({
                "emergency_type": event,
                "ptp_seconds": ptp_ts.seconds,
                "ptp_nanos": ptp_ts.nanos,
                "system_state": "emergency",
            }));

        // Ensure emergency logs are immediately flushed
        let entry_id = self.log_entry(entry).await?;
        self.flush().await?;

        Ok(entry_id)
    }

    /// Log a processing event
    pub async fn log_processing(&self, operation: &str, duration_ms: f64) -> Result<String> {
        let entry = LogEntry::new("processing", "technical_arch", operation)
            .with_data(serde_json::json!({
                "operation": operation,
                "duration_ms": duration_ms,
            }));
        self.log_entry(entry).await
    }

    /// Log a thermal event
    pub async fn log_thermal(&self, state: &str, temp_c: f32) -> Result<String> {
        let entry = LogEntry::new("thermal", "thermal_governor", state)
            .with_data(serde_json::json!({
                "state": state,
                "temp_c": temp_c,
            }));
        self.log_entry(entry).await
    }

    /// Log a custom entry
    pub async fn log_entry(&self, entry: LogEntry) -> Result<String> {
        if !self.active.load(std::sync::atomic::Ordering::SeqCst) {
            return Ok(entry.id.clone());
        }

        let entry_id = entry.id.clone();

        // Add to buffer
        {
            let mut entries = self.entries.lock().await;
            entries.push(entry.clone());

            // Trim buffer if needed
            if entries.len() > 10000 {
                entries.remove(0);
            }
        }

        // Write to file if async logging is disabled
        if !self.config.async_logging {
            self.write_entry(&entry).await?;
        }

        debug!("Logged entry: {} - {}", entry_id, entry.message);
        Ok(entry_id)
    }

    /// Write entry to file
    async fn write_entry(&self, entry: &LogEntry) -> Result<()> {
        let json = entry.to_json()?;
        let line = format!("{}\n", json);
        let line_len = line.len();

        // Check file size and rotate if needed
        let current_size = *self.file_size.lock().await;
        if current_size + line_len > self.config.max_file_size {
            self.rotate_log().await?;
        }

        // Append to file
        tokio::fs::write(&self.config.log_file_path, line)
            .await
            .context("Failed to write log entry")?;

        // Update file size
        *self.file_size.lock().await += line_len;

        Ok(())
    }

    /// Rotate log file
    async fn rotate_log(&self) -> Result<()> {
        info!("Rotating log file");

        // Remove oldest backup if we have too many
        for i in (1..=self.config.num_backups).rev() {
            let backup_path = self.config.log_file_path.with_extension(format!("log.{}", i));
            if i == self.config.num_backups {
                tokio::fs::remove_file(backup_path).await.ok();
            } else {
                // Rename backups (move up one number)
                let next_backup = self.config.log_file_path.with_extension(format!("log.{}", i + 1));
                tokio::fs::rename(&backup_path, &next_backup).await.ok();
            }
        }

        // Rename current log to .1
        let backup_1 = self.config.log_file_path.with_extension("log.1");
        tokio::fs::rename(&self.config.log_file_path, &backup_1).await.ok();

        // Reset file size
        *self.file_size.lock().await = 0;

        Ok(())
    }

    /// Flush all buffered entries to disk
    pub async fn flush(&self) -> Result<()> {
        info!("Flushing log entries to disk");

        let entries = {
            let mut buffer = self.entries.lock().await;
            let entries_to_flush = buffer.clone();
            buffer.clear();
            entries_to_flush
        };

        let entry_count = entries.len();
        for entry in entries {
            self.write_entry(&entry).await?;
        }

        info!("Flushed {} log entries", entry_count);
        Ok(())
    }

    /// Get all log entries
    pub async fn get_entries(&self) -> Vec<LogEntry> {
        self.entries.lock().await.clone()
    }

    /// Get entries by type
    pub async fn get_entries_by_type(&self, entry_type: &str) -> Vec<LogEntry> {
        let entries = self.entries.lock().await;
        entries.iter()
            .filter(|e| e.entry_type == entry_type)
            .cloned()
            .collect()
    }

    /// Get entries by component
    pub async fn get_entries_by_component(&self, component: &str) -> Vec<LogEntry> {
        let entries = self.entries.lock().await;
        entries.iter()
            .filter(|e| e.component == component)
            .cloned()
            .collect()
    }

    /// Get entries by severity
    pub async fn get_entries_by_severity(&self, severity: &str) -> Vec<LogEntry> {
        let entries = self.entries.lock().await;
        entries.iter()
            .filter(|e| e.severity == severity)
            .cloned()
            .collect()
    }

    /// Query entries by causality
    pub async fn get_causal_chain(&self, entry_id: &str) -> Vec<LogEntry> {
        let entries = self.entries.lock().await;
        let mut chain = Vec::new();

        // Find the starting entry
        if let Some(start) = entries.iter().find(|e| e.id == entry_id) {
            chain.push(start.clone());

            // Trace parents
            let mut current_id = entry_id.to_string();
            while let Some(parent) = entries.iter().find(|e| e.id == current_id) {
                for parent_id in &parent.causality {
                    if let Some(parent_entry) = entries.iter().find(|e| e.id == *parent_id) {
                        chain.push(parent_entry.clone());
                        current_id = parent_id.clone();
                    }
                }
            }
        }

        chain.reverse(); // Put in chronological order
        chain
    }

    /// Get logger statistics
    pub async fn get_stats(&self) -> LoggingStats {
        let entries = self.entries.lock().await;
        let file_size = *self.file_size.lock().await;
        let active = self.active.load(std::sync::atomic::Ordering::SeqCst);

        LoggingStats {
            total_entries: entries.len(),
            file_size_bytes: file_size,
            active,
            log_file_path: self.config.log_file_path.clone(),
        }
    }

    /// Clear all log entries
    pub async fn clear(&self) -> Result<()> {
        self.entries.lock().await.clear();
        Ok(())
    }

    /// Export logs to JSON file
    pub async fn export(&self, path: &PathBuf) -> Result<()> {
        let entries = self.entries.lock().await;
        let json = serde_json::to_string_pretty(&*entries)
            .context("Failed to serialize log entries")?;

        tokio::fs::write(path, json).await
            .context("Failed to write export file")?;

        info!("Exported {} log entries to {:?}", entries.len(), path);
        Ok(())
    }
}

/// Logging statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingStats {
    pub total_entries: usize,
    pub file_size_bytes: usize,
    pub active: bool,
    pub log_file_path: PathBuf,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_logging_config_default() {
        let config = LoggingConfig::default();
        assert_eq!(config.max_file_size, 100_000_000);
        assert_eq!(config.num_backups, 5);
        assert!(config.async_logging);
    }

    #[tokio::test]
    async fn test_log_entry_creation() {
        let entry = LogEntry::new("test", "test_component", "test message");

        assert_eq!(entry.entry_type, "test");
        assert_eq!(entry.component, "test_component");
        assert_eq!(entry.message, "test message");
        assert_eq!(entry.severity, "info");
        assert!(entry.causality.is_empty());
    }

    #[tokio::test]
    async fn test_log_entry_builder() {
        let entry = LogEntry::new("test", "test_component", "test message")
            .with_severity("warning")
            .with_ptp_timestamp(1234567890);

        assert_eq!(entry.severity, "warning");
        assert_eq!(entry.ptp_timestamp, Some(1234567890));
    }

    #[tokio::test]
    async fn test_provenance_logger_creation() {
        let config = LoggingConfig {
            log_file_path: PathBuf::from("/tmp/test_provenance.log"),
            ..Default::default()
        };

        let logger = ProvenanceLogger::new(config).await.unwrap();
        let stats = logger.get_stats().await;

        assert_eq!(stats.total_entries, 0);
    }

    #[tokio::test]
    async fn test_log_event() {
        let config = LoggingConfig {
            log_file_path: PathBuf::from("/tmp/test_provenance.log"),
            async_logging: false,
            ..Default::default()
        };

        let logger = ProvenanceLogger::new(config).await.unwrap();
        logger.start().await.unwrap();

        let entry_id = logger.log_event("test_component", "test_event", "test message").await.unwrap();
        assert!(!entry_id.is_empty());

        let entries = logger.get_entries().await;
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].component, "test_component");
    }

    #[tokio::test]
    async fn test_filter_by_type() {
        let config = LoggingConfig::default();
        let logger = ProvenanceLogger::new(config).await.unwrap();
        logger.start().await.unwrap();

        logger.log_event("comp1", "type1", "msg1").await.unwrap();
        logger.log_event("comp2", "type2", "msg2").await.unwrap();
        logger.log_event("comp3", "type1", "msg3").await.unwrap();

        let type1_entries = logger.get_entries_by_type("type1").await;
        assert_eq!(type1_entries.len(), 2);
    }

    #[tokio::test]
    async fn test_log_decision() {
        let config = LoggingConfig::default();
        let logger = ProvenanceLogger::new(config).await.unwrap();
        logger.start().await.unwrap();

        let ptp_ts = crate::ptp::PtpTimestamp::new(1000, 500_000_000);
        let entry_id = logger.log_decision("test_decision", ptp_ts).await.unwrap();

        let entries = logger.get_entries().await;
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].entry_type, "decision");
        assert_eq!(entries[0].ptp_timestamp, Some(ptp_ts.as_nanos()));
    }
}
