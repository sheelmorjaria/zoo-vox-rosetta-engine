// Time-Series Archiving Pipeline
//
// Efficiently stores and queries terabytes of high-frequency multi-channel
// time-series data (audio, visual, sensor logs) using InfluxDB and Parquet.

use crate::ptp::PtpTimestamp;
use anyhow::Result;
use chrono::{NaiveDateTime, TimeZone, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

// ============================================================================
// Data Structures
// ============================================================================

/// A single time-series data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesPoint {
    pub timestamp: PtpTimestamp,
    pub measurement: String,  // e.g., "temperature", "SPL", "F0"
    pub value: f64,
    pub tags: HashMap<String, String>,  // e.g., {"channel": "audio_L"}
    pub fields: HashMap<String, f64>,   // Additional fields
}

impl TimeSeriesPoint {
    pub fn new(measurement: impl Into<String>, value: f64) -> Self {
        Self {
            timestamp: PtpTimestamp::from(Utc::now()),
            measurement: measurement.into(),
            value,
            tags: HashMap::new(),
            fields: HashMap::new(),
        }
    }

    pub fn with_timestamp(mut self, timestamp: PtpTimestamp) -> Self {
        self.timestamp = timestamp;
        self
    }

    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.insert(key.into(), value.into());
        self
    }

    pub fn with_field(mut self, key: impl Into<String>, value: f64) -> Self {
        self.fields.insert(key.into(), value);
        self
    }
}

/// Batch of time-series points for efficient writing
#[derive(Debug, Clone)]
pub struct TimeSeriesBatch {
    pub points: Vec<TimeSeriesPoint>,
    pub max_batch_size: usize,
    pub flush_interval_ms: u64,
}

impl TimeSeriesBatch {
    pub fn new(max_batch_size: usize, flush_interval_ms: u64) -> Self {
        Self {
            points: Vec::with_capacity(max_batch_size),
            max_batch_size,
            flush_interval_ms,
        }
    }

    pub fn add_point(&mut self, point: TimeSeriesPoint) -> bool {
        self.points.push(point);
        self.points.len() >= self.max_batch_size
    }

    pub fn is_full(&self) -> bool {
        self.points.len() >= self.max_batch_size
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    pub fn clear(&mut self) {
        self.points.clear();
    }
}

/// Parquet compression type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParquetCompression {
    Snappy,
    Gzip,
    Lzo,
    Brotli,
}

/// Parquet export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParquetExportConfig {
    pub export_schedule_cron: String,  // e.g., "0 2 * * *" (daily 2AM)
    pub compression: ParquetCompression,
    pub row_group_size: usize,
    pub output_directory: PathBuf,
}

impl Default for ParquetExportConfig {
    fn default() -> Self {
        Self {
            export_schedule_cron: "0 2 * * *".to_string(),
            compression: ParquetCompression::Snappy,
            row_group_size: 10000,
            output_directory: PathBuf::from("/data/parquet_exports"),
        }
    }
}

/// Retention policy for time-series data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicy {
    pub name: String,
    pub duration_days: u32,
    pub shard_duration_days: u32,
    pub default_resolution: Duration,
    pub replica_count: u32,
}

impl RetentionPolicy {
    pub fn raw_data() -> Self {
        Self {
            name: "raw".to_string(),
            duration_days: 7,  // Keep 7 days of raw data
            shard_duration_days: 1,
            default_resolution: Duration::from_secs(1),
            replica_count: 1,
        }
    }

    pub fn downsampled_1min() -> Self {
        Self {
            name: "downsampled_1m".to_string(),
            duration_days: 90,  // Keep 90 days of 1min data
            shard_duration_days: 7,
            default_resolution: Duration::from_secs(60),
            replica_count: 1,
        }
    }

    pub fn downsampled_1hour() -> Self {
        Self {
            name: "downsampled_1h".to_string(),
            duration_days: 365,  // Keep 1 year of 1hour data
            shard_duration_days: 30,
            default_resolution: Duration::from_secs(3600),
            replica_count: 1,
        }
    }
}

/// Storage quota configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageQuota {
    pub max_bytes: u64,
    pub warning_threshold_percent: u8,  // Warn at X% full
    pub enforce_hard_limit: bool,
}

impl StorageQuota {
    pub fn new(max_bytes: u64) -> Self {
        Self {
            max_bytes,
            warning_threshold_percent: 80,
            enforce_hard_limit: true,
        }
    }

    pub fn is_over_quota(&self, used_bytes: u64) -> bool {
        self.enforce_hard_limit && used_bytes >= self.max_bytes
    }

    pub fn is_warning_level(&self, used_bytes: u64) -> bool {
        let percent = (used_bytes as f64 / self.max_bytes as f64 * 100.0) as u8;
        percent >= self.warning_threshold_percent
    }
}

/// Time-series archiver configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesConfig {
    pub influxdb_url: String,
    pub database_name: String,
    pub batch_size: usize,
    pub flush_interval_ms: u64,
    pub retention_policies: Vec<RetentionPolicy>,
    pub storage_quota: StorageQuota,
    pub parquet_export: ParquetExportConfig,
}

impl Default for TimeSeriesConfig {
    fn default() -> Self {
        Self {
            influxdb_url: "http://localhost:8086".to_string(),
            database_name: "animal_vocalization".to_string(),
            batch_size: 1000,
            flush_interval_ms: 5000,
            retention_policies: vec![
                RetentionPolicy::raw_data(),
                RetentionPolicy::downsampled_1min(),
                RetentionPolicy::downsampled_1hour(),
            ],
            storage_quota: StorageQuota::new(1_000_000_000_000), // 1TB
            parquet_export: ParquetExportConfig::default(),
        }
    }
}

// ============================================================================
// Time-Series Archiver
// ============================================================================

/// Time-series archiver for InfluxDB and Parquet export
pub struct TimeSeriesArchiver {
    config: TimeSeriesConfig,
    batch: Arc<Mutex<TimeSeriesBatch>>,
    retention_policies: Vec<RetentionPolicy>,
    storage_quota: StorageQuota,
    parquet_export_enabled: bool,
}

impl TimeSeriesArchiver {
    pub fn new(config: TimeSeriesConfig) -> Self {
        let batch = TimeSeriesBatch::new(config.batch_size, config.flush_interval_ms);

        Self {
            retention_policies: config.retention_policies.clone(),
            storage_quota: config.storage_quota.clone(),
            parquet_export_enabled: true,
            config,
            batch: Arc::new(Mutex::new(batch)),
        }
    }

    /// Create a new archiver with default configuration
    pub fn with_default_url(url: impl Into<String>) -> Self {
        let mut config = TimeSeriesConfig::default();
        config.influxdb_url = url.into();
        Self::new(config)
    }

    /// Write a single time-series point
    pub fn write_point(&self, point: TimeSeriesPoint) -> Result<()> {
        let mut batch = self.batch.lock().unwrap();
        if batch.add_point(point) {
            // Batch is full, flush it
            drop(batch);
            self.flush_batch()?;
        }
        Ok(())
    }

    /// Write multiple time-series points
    pub fn write_points(&self, points: Vec<TimeSeriesPoint>) -> Result<()> {
        let mut batch = self.batch.lock().unwrap();
        for point in points {
            if batch.add_point(point) {
                // Batch is full, flush and continue
                drop(batch);
                self.flush_batch()?;
                batch = self.batch.lock().unwrap();
            }
        }
        Ok(())
    }

    /// Flush current batch to storage
    pub fn flush(&self) -> Result<()> {
        let batch = self.batch.lock().unwrap();
        if !batch.is_empty() {
            drop(batch);
            self.flush_batch()?;
        }
        Ok(())
    }

    /// Internal method to flush batch to storage
    fn flush_batch(&self) -> Result<()> {
        let mut batch = self.batch.lock().unwrap();
        if batch.is_empty() {
            return Ok(());
        }

        // In production, this would write to InfluxDB
        // For now, we simulate the write
        let count = batch.points.len();
        batch.clear();

        log::debug!("Flushed {} time-series points to InfluxDB", count);
        Ok(())
    }

    /// Query time-series data by time range
    pub fn query_time_range(
        &self,
        measurement: &str,
        start: NaiveDateTime,
        end: NaiveDateTime,
    ) -> Result<Vec<TimeSeriesPoint>> {
        // In production, this would query InfluxDB
        // For now, return empty vec
        log::debug!(
            "Querying measurement '{}' from {:?} to {:?}",
            measurement,
            start,
            end
        );
        Ok(Vec::new())
    }

    /// Export data to Parquet format
    pub fn export_to_parquet(
        &self,
        measurement: &str,
        start: NaiveDateTime,
        end: NaiveDateTime,
        output_path: &Path,
    ) -> Result<usize> {
        // In production, this would:
        // 1. Query data from InfluxDB
        // 2. Convert to Parquet format
        // 3. Compress using specified compression
        // 4. Write to file

        log::debug!(
            "Exporting measurement '{}' to Parquet: {:?}",
            measurement,
            output_path
        );

        Ok(0)
    }

    /// Apply retention policy (delete old data)
    pub fn apply_retention_policy(&self, policy: &RetentionPolicy) -> Result<u64> {
        // In production, this would delete old data from InfluxDB
        log::debug!(
            "Applying retention policy '{}' ({} days)",
            policy.name,
            policy.duration_days
        );
        Ok(0)
    }

    /// Downsample old data to lower resolution
    pub fn downsample_data(
        &self,
        source_policy: &RetentionPolicy,
        target_policy: &RetentionPolicy,
    ) -> Result<u64> {
        // In production, this would:
        // 1. Query data at source resolution
        // 2. Aggregate to target resolution
        // 3. Write to target retention policy
        log::debug!(
            "Downsampling from '{}' to '{}'",
            source_policy.name,
            target_policy.name
        );
        Ok(0)
    }

    /// Get storage usage statistics
    pub fn get_storage_stats(&self) -> StorageStats {
        // In production, this would query InfluxDB for actual stats
        StorageStats {
            total_bytes: 0,
            used_bytes: 0,
            available_bytes: self.storage_quota.max_bytes,
            series_count: 0,
            measurement_count: 0,
        }
    }

    /// Check if storage is over quota
    pub fn is_over_quota(&self) -> bool {
        let stats = self.get_storage_stats();
        self.storage_quota.is_over_quota(stats.used_bytes)
    }

    /// Get current configuration
    pub fn config(&self) -> &TimeSeriesConfig {
        &self.config
    }
}

/// Storage usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    pub total_bytes: u64,
    pub used_bytes: u64,
    pub available_bytes: u64,
    pub series_count: u64,
    pub measurement_count: u64,
}

impl StorageStats {
    pub fn usage_percent(&self) -> f64 {
        if self.total_bytes == 0 {
            0.0
        } else {
            (self.used_bytes as f64 / self.total_bytes as f64) * 100.0
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_point() -> TimeSeriesPoint {
        TimeSeriesPoint::new("temperature", 25.5)
            .with_tag("location", "sensor_1")
            .with_field("humidity", 60.0)
    }

    // ============================================================================
    // TimeSeriesPoint Tests
    // ============================================================================

    #[test]
    fn test_point_creation() {
        let point = TimeSeriesPoint::new("test_metric", 42.0);
        assert_eq!(point.measurement, "test_metric");
        assert_eq!(point.value, 42.0);
        assert!(point.tags.is_empty());
        assert!(point.fields.is_empty());
    }

    #[test]
    fn test_point_with_tags() {
        let point = TimeSeriesPoint::new("test", 1.0)
            .with_tag("device", "sensor_1")
            .with_tag("location", "field");

        assert_eq!(point.tags.len(), 2);
        assert_eq!(point.tags.get("device"), Some(&"sensor_1".to_string()));
        assert_eq!(point.tags.get("location"), Some(&"field".to_string()));
    }

    #[test]
    fn test_point_with_fields() {
        let point = TimeSeriesPoint::new("test", 1.0)
            .with_field("extra", 99.0)
            .with_field("quality", 0.95);

        assert_eq!(point.fields.len(), 2);
        assert_eq!(point.fields.get("extra"), Some(&99.0));
        assert_eq!(point.fields.get("quality"), Some(&0.95));
    }

    // ============================================================================
    // TimeSeriesBatch Tests
    // ============================================================================

    #[test]
    fn test_batch_creation() {
        let batch = TimeSeriesBatch::new(100, 5000);
        assert_eq!(batch.max_batch_size, 100);
        assert_eq!(batch.flush_interval_ms, 5000);
        assert!(batch.is_empty());
        assert!(!batch.is_full());
    }

    #[test]
    fn test_batch_add_points() {
        let mut batch = TimeSeriesBatch::new(10, 5000);

        assert!(!batch.add_point(create_test_point()));
        assert_eq!(batch.len(), 1);

        for _ in 0..9 {
            batch.add_point(create_test_point());
        }
        assert_eq!(batch.len(), 10);
        assert!(batch.is_full());
    }

    #[test]
    fn test_batch_add_returns_flush_trigger() {
        let mut batch = TimeSeriesBatch::new(5, 5000);

        // First 4 adds should not trigger flush
        assert!(!batch.add_point(create_test_point()));
        assert!(!batch.add_point(create_test_point()));
        assert!(!batch.add_point(create_test_point()));
        assert!(!batch.add_point(create_test_point()));

        // 5th add should trigger flush
        assert!(batch.add_point(create_test_point()));
    }

    #[test]
    fn test_batch_clear() {
        let mut batch = TimeSeriesBatch::new(10, 5000);

        for _ in 0..5 {
            batch.add_point(create_test_point());
        }

        assert_eq!(batch.len(), 5);
        batch.clear();
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
    }

    // ============================================================================
    // RetentionPolicy Tests
    // ============================================================================

    #[test]
    fn test_retention_policies() {
        let raw = RetentionPolicy::raw_data();
        assert_eq!(raw.name, "raw");
        assert_eq!(raw.duration_days, 7);

        let downsampled_1m = RetentionPolicy::downsampled_1min();
        assert_eq!(downsampled_1m.name, "downsampled_1m");
        assert_eq!(downsampled_1m.duration_days, 90);

        let downsampled_1h = RetentionPolicy::downsampled_1hour();
        assert_eq!(downsampled_1h.name, "downsampled_1h");
        assert_eq!(downsampled_1h.duration_days, 365);
    }

    // ============================================================================
    // StorageQuota Tests
    // ============================================================================

    #[test]
    fn test_storage_quota() {
        let quota = StorageQuota::new(1000);

        assert!(!quota.is_over_quota(500));
        assert!(!quota.is_over_quota(999));
        assert!(quota.is_over_quota(1000));
        assert!(quota.is_over_quota(1500));
    }

    #[test]
    fn test_storage_quota_warning() {
        let quota = StorageQuota {
            max_bytes: 1000,
            warning_threshold_percent: 80,
            enforce_hard_limit: true,
        };

        assert!(!quota.is_warning_level(500));  // 50%
        assert!(!quota.is_warning_level(799));  // 79.9%
        assert!(quota.is_warning_level(800));   // 80%
        assert!(quota.is_warning_level(900));   // 90%
    }

    // ============================================================================
    // TimeSeriesArchiver Tests
    // ============================================================================

    #[test]
    fn test_archiver_creation() {
        let archiver = TimeSeriesArchiver::with_default_url("http://localhost:8086");
        assert_eq!(archiver.config().influxdb_url, "http://localhost:8086");
        assert_eq!(archiver.config().batch_size, 1000);
    }

    #[test]
    fn test_archiver_write_single_point() {
        let archiver = TimeSeriesArchiver::with_default_url("http://localhost:8086");
        let point = create_test_point();

        assert!(archiver.write_point(point).is_ok());
    }

    #[test]
    fn test_archiver_write_multiple_points() {
        let archiver = TimeSeriesArchiver::with_default_url("http://localhost:8086");

        let points = vec![
            create_test_point(),
            create_test_point(),
            create_test_point(),
        ];

        assert!(archiver.write_points(points).is_ok());
    }

    #[test]
    fn test_archiver_flush() {
        let archiver = TimeSeriesArchiver::with_default_url("http://localhost:8086");
        assert!(archiver.write_point(create_test_point()).is_ok());
        assert!(archiver.flush().is_ok());
    }

    #[test]
    fn test_archiver_query_time_range() {
        let archiver = TimeSeriesArchiver::with_default_url("http://localhost:8086");

        let start = Utc.timestamp_opt(0, 0).unwrap().naive_utc();
        let end = Utc::now().naive_utc();

        let result = archiver.query_time_range("temperature", start, end);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty()); // Empty in test mode
    }

    #[test]
    fn test_archiver_export_to_parquet() {
        let archiver = TimeSeriesArchiver::with_default_url("http://localhost:8086");

        let start = Utc.timestamp_opt(0, 0).unwrap().naive_utc();
        let end = Utc::now().naive_utc();
        let output_path = Path::new("/tmp/test_export.parquet");

        let result = archiver.export_to_parquet("temperature", start, end, output_path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_archiver_apply_retention_policy() {
        let archiver = TimeSeriesArchiver::with_default_url("http://localhost:8086");
        let policy = RetentionPolicy::raw_data();

        let result = archiver.apply_retention_policy(&policy);
        assert!(result.is_ok());
    }

    #[test]
    fn test_archiver_downsample_data() {
        let archiver = TimeSeriesArchiver::with_default_url("http://localhost:8086");

        let source = RetentionPolicy::raw_data();
        let target = RetentionPolicy::downsampled_1min();

        let result = archiver.downsample_data(&source, &target);
        assert!(result.is_ok());
    }

    #[test]
    fn test_archiver_storage_stats() {
        let archiver = TimeSeriesArchiver::with_default_url("http://localhost:8086");
        let stats = archiver.get_storage_stats();

        assert_eq!(stats.used_bytes, 0);
        assert_eq!(stats.series_count, 0);
        assert_eq!(stats.usage_percent(), 0.0);
    }

    #[test]
    fn test_archiver_is_over_quota() {
        let archiver = TimeSeriesArchiver::with_default_url("http://localhost:8086");
        assert!(!archiver.is_over_quota());
    }

    // ============================================================================
    // StorageStats Tests
    // ============================================================================

    #[test]
    fn test_storage_stats_usage_percent() {
        let stats = StorageStats {
            total_bytes: 1000,
            used_bytes: 500,
            available_bytes: 500,
            series_count: 10,
            measurement_count: 5,
        };

        assert_eq!(stats.usage_percent(), 50.0);
    }

    #[test]
    fn test_storage_stats_usage_percent_zero_total() {
        let stats = StorageStats {
            total_bytes: 0,
            used_bytes: 0,
            available_bytes: 0,
            series_count: 0,
            measurement_count: 0,
        };

        assert_eq!(stats.usage_percent(), 0.0);
    }

    // ============================================================================
    // Default Configuration Tests
    // ============================================================================

    #[test]
    fn test_default_config() {
        let config = TimeSeriesConfig::default();
        assert_eq!(config.influxdb_url, "http://localhost:8086");
        assert_eq!(config.database_name, "animal_vocalization");
        assert_eq!(config.batch_size, 1000);
        assert_eq!(config.flush_interval_ms, 5000);
    }

    #[test]
    fn test_default_parquet_config() {
        let config = ParquetExportConfig::default();
        assert_eq!(config.export_schedule_cron, "0 2 * * *");
        assert_eq!(config.compression, ParquetCompression::Snappy);
        assert_eq!(config.row_group_size, 10000);
    }
}
