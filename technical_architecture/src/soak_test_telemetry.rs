//! Soak Test Telemetry
//! ===================
//!
//! System metrics collection for 24-hour soak testing. Monitors RAM/VRAM,
//! thermal state, ZMQ queue depth, and RTL latency for stability validation.
//!
//! Author: Sheel Morjaria (sheelmorjaria@gmail.com)
//! License: CC BY-ND 4.0 International

use crate::ptp::PtpTimestamp;
use anyhow::Result;
use chrono::{DateTime, Utc};
use log::{info, warn};
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Thermal zone for Jetson devices
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemperatureZone {
    CPU,
    GPU,
    Thermal,
    PMIC,
    Ambient,
}

/// Soak test metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoakTestMetrics {
    /// Timestamp of measurement
    pub timestamp: PtpTimestamp,

    /// System time
    pub system_time: DateTime<Utc>,

    /// RAM usage in MB
    pub ram_usage_mb: f32,

    /// VRAM usage in MB
    pub vram_usage_mb: f32,

    /// CPU temperature in Celsius
    pub cpu_temperature_c: Option<f32>,

    /// GPU temperature in Celsius
    pub gpu_temperature_c: Option<f32>,

    /// Jetson thermal zone readings
    pub thermal_zones: Vec<(TemperatureZone, f32)>,

    /// ZMQ queue depth
    pub zmq_queue_depth: usize,

    /// RTL P99 latency in ms
    pub rtl_p99_ms: f64,

    /// RTL P50 latency in ms
    pub rtl_p50_ms: f64,

    /// Active connections
    pub active_connections: u32,
}

/// Memory leak detection result
#[derive(Debug, Clone)]
pub enum MemoryLeakStatus {
    NoLeak,
    PotentialLeak { growth_percent: f32 },
    ConfirmedLeak { growth_percent: f32 },
}

/// Configuration for soak test telemetry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoakTestTelemetryConfig {
    /// Log interval in seconds
    pub log_interval_secs: u64,

    /// Memory leak threshold percent
    pub memory_leak_threshold_percent: f32,

    /// Memory leak window in hours
    pub memory_leak_window_hours: f32,

    /// Output log file path
    pub log_file_path: Option<String>,

    /// Enable thermal monitoring
    pub enable_thermal: bool,

    /// Enable ZMQ monitoring
    pub enable_zmq: bool,
}

impl Default for SoakTestTelemetryConfig {
    fn default() -> Self {
        Self {
            log_interval_secs: 60,
            memory_leak_threshold_percent: 5.0,
            memory_leak_window_hours: 1.0,
            log_file_path: None,
            enable_thermal: true,
            enable_zmq: true,
        }
    }
}

/// Soak test telemetry collector
pub struct SoakTestTelemetry {
    config: SoakTestTelemetryConfig,
    metrics_history: Arc<Mutex<Vec<SoakTestMetrics>>>,
    log_file: Arc<Mutex<Option<File>>>,
    start_time: DateTime<Utc>,
}

impl SoakTestTelemetry {
    /// Create a new soak test telemetry collector
    pub fn new(config: SoakTestTelemetryConfig) -> Result<Self> {
        let log_file = if let Some(path) = &config.log_file_path {
            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)?;

            Some(file)
        } else {
            None
        };

        Ok(Self {
            config,
            metrics_history: Arc::new(Mutex::new(vec![])),
            log_file: Arc::new(Mutex::new(log_file)),
            start_time: Utc::now(),
        })
    }

    /// Collect current system metrics
    pub fn collect_metrics(&self) -> Result<SoakTestMetrics> {
        let timestamp = PtpTimestamp::now();
        let system_time = Utc::now();

        // Collect RAM usage
        let ram_usage_mb = self.get_ram_usage()?;

        // Collect VRAM usage (if available)
        let vram_usage_mb = self.get_vram_usage().unwrap_or(0.0);

        // Collect thermal data
        let (cpu_temp, gpu_temp, thermal_zones) = if self.config.enable_thermal {
            self.collect_thermal_data()?
        } else {
            (None, None, vec![])
        };

        // ZMQ queue depth (placeholder - requires integration)
        let zmq_queue_depth = 0;

        // RTL metrics (placeholder - requires integration)
        let rtl_p99_ms = 0.0;
        let rtl_p50_ms = 0.0;

        let active_connections = 0;

        Ok(SoakTestMetrics {
            timestamp,
            system_time,
            ram_usage_mb,
            vram_usage_mb,
            cpu_temperature_c: cpu_temp,
            gpu_temperature_c: gpu_temp,
            thermal_zones,
            zmq_queue_depth,
            rtl_p99_ms,
            rtl_p50_ms,
            active_connections,
        })
    }

    /// Collect and log metrics
    pub fn collect_and_log(&self) -> Result<()> {
        let metrics = self.collect_metrics()?;

        // Add to history
        {
            let mut history = self.metrics_history.lock().unwrap();
            history.push(metrics.clone());
        }

        // Write to log file
        if let Some(ref mut file) = *self.log_file.lock().unwrap() {
            writeln!(file, "{}", serde_json::to_string(&metrics)?)?;
            file.flush()?;
        }

        info!(
            "Telemetry: RAM={:.1}MB, VRAM={:.1}MB, CPU_T={:.1}C, GPU_T={:.1}C, RTL_P99={:.2}ms",
            metrics.ram_usage_mb,
            metrics.vram_usage_mb,
            metrics.cpu_temperature_c.unwrap_or(0.0),
            metrics.gpu_temperature_c.unwrap_or(0.0),
            metrics.rtl_p99_ms
        );

        // Check for memory leaks
        let leak_status = self.check_memory_leak()?;
        if let MemoryLeakStatus::PotentialLeak { growth_percent } = leak_status {
            warn!(
                "Potential memory leak detected: {:.1}% growth over {:.1} hours",
                growth_percent, self.config.memory_leak_window_hours
            );
        } else if let MemoryLeakStatus::ConfirmedLeak { growth_percent } = leak_status {
            warn!(
                "Memory leak confirmed: {:.1}% growth over {:.1} hours",
                growth_percent, self.config.memory_leak_window_hours
            );
        }

        Ok(())
    }

    /// Check for memory leaks
    pub fn check_memory_leak(&self) -> Result<MemoryLeakStatus> {
        let history = self.metrics_history.lock().unwrap();

        if history.len() < 2 {
            return Ok(MemoryLeakStatus::NoLeak);
        }

        let window_samples = (self.config.memory_leak_window_hours * 3600.0
            / self.config.log_interval_secs as f32) as usize;

        if history.len() < window_samples {
            return Ok(MemoryLeakStatus::NoLeak);
        }

        // Get oldest and newest within window
        let oldest = &history[history.len() - window_samples];
        let newest = &history[history.len() - 1];

        // Calculate RAM growth
        let ram_growth = ((newest.ram_usage_mb - oldest.ram_usage_mb)
            / oldest.ram_usage_mb.max(0.1)) * 100.0;

        // Calculate VRAM growth
        let vram_growth = ((newest.vram_usage_mb - oldest.vram_usage_mb)
            / oldest.vram_usage_mb.max(0.1)) * 100.0;

        // Use maximum growth
        let max_growth = ram_growth.max(vram_growth);

        if max_growth > self.config.memory_leak_threshold_percent * 2.0 {
            Ok(MemoryLeakStatus::ConfirmedLeak {
                growth_percent: max_growth,
            })
        } else if max_growth > self.config.memory_leak_threshold_percent {
            Ok(MemoryLeakStatus::PotentialLeak {
                growth_percent: max_growth,
            })
        } else {
            Ok(MemoryLeakStatus::NoLeak)
        }
    }

    /// Get RAM usage in MB
    fn get_ram_usage(&self) -> Result<f32> {
        // Platform-specific implementation
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            let meminfo = fs::read_to_string("/proc/meminfo")?;

            let mut total = 0;
            let mut available = 0;

            for line in meminfo.lines() {
                if line.starts_with("MemTotal:") {
                    total = line.split_whitespace()
                        .nth(1)
                        .and_then(|s| s.parse::<u64>().ok())
                        .unwrap_or(0);
                } else if line.starts_with("MemAvailable:") {
                    available = line.split_whitespace()
                        .nth(1)
                        .and_then(|s| s.parse::<u64>().ok())
                        .unwrap_or(0);
                }
            }

            let used = total.saturating_sub(available);
            Ok((used as f32) / 1024.0)  // Convert KB to MB
        }

        #[cfg(not(target_os = "linux"))]
        {
            Ok(0.0)  // Placeholder for non-Linux
        }
    }

    /// Get VRAM usage in MB
    fn get_vram_usage(&self) -> Result<f32> {
        // Requires GPU-specific implementation (NVIDIA, etc.)
        // Return placeholder for now
        Ok(0.0)
    }

    /// Collect thermal data
    fn collect_thermal_data(&self) -> Result<(Option<f32>, Option<f32>, Vec<(TemperatureZone, f32)>)> {
        #[cfg(target_os = "linux")]
        {
            use std::fs;

            let mut zones = vec![];

            // Try to read from sysfs thermal zones
            if let Ok(entries) = fs::read_dir("/sys/class/thermal") {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.join("temp").exists() {
                        if let Ok(temp_str) = fs::read_to_string(path.join("temp")) {
                            if let Ok(temp_millidec) = temp_str.trim().parse::<i32>() {
                                let temp_c = temp_millidec as f32 / 1000.0;

                                // Try to determine zone type from path
                                let zone_type = if path.to_string_lossy().contains("cpu_thermal") {
                                    TemperatureZone::CPU
                                } else if path.to_string_lossy().contains("gpu_thermal") {
                                    TemperatureZone::GPU
                                } else {
                                    TemperatureZone::Thermal
                                };

                                zones.push((zone_type, temp_c));
                            }
                        }
                    }
                }
            }

            // Extract CPU and GPU temps
            let cpu_temp = zones.iter()
                .find(|(z, _)| *z == TemperatureZone::CPU)
                .map(|(_, t)| *t);

            let gpu_temp = zones.iter()
                .find(|(z, _)| *z == TemperatureZone::GPU)
                .map(|(_, t)| *t);

            Ok((cpu_temp, gpu_temp, zones))
        }

        #[cfg(not(target_os = "linux"))]
        {
            Ok((None, None, vec![]))
        }
    }

    /// Get metrics history
    pub fn get_history(&self) -> Vec<SoakTestMetrics> {
        self.metrics_history.lock().unwrap().clone()
    }

    /// Get uptime in hours
    pub fn uptime_hours(&self) -> f64 {
        (Utc::now() - self.start_time).num_seconds() as f64 / 3600.0
    }

    /// Reset history
    pub fn reset(&self) {
        let mut history = self.metrics_history.lock().unwrap();
        history.clear();
    }

    /// Generate summary report
    pub fn generate_summary(&self) -> String {
        let history = self.get_history();
        let uptime = self.uptime_hours();

        if history.is_empty() {
            return format!("No metrics collected (uptime: {:.2} hours)", uptime);
        }

        let first = &history[0];
        let last = &history[history.len() - 1];

        let ram_growth = ((last.ram_usage_mb - first.ram_usage_mb)
            / first.ram_usage_mb.max(0.1)) * 100.0;

        let vram_growth = ((last.vram_usage_mb - first.vram_usage_mb)
            / first.vram_usage_mb.max(0.1)) * 100.0;

        let max_cpu = history.iter()
            .filter_map(|m| m.cpu_temperature_c)
            .fold(0.0_f32, |a, b| a.max(b));

        let max_gpu = history.iter()
            .filter_map(|m| m.gpu_temperature_c)
            .fold(0.0_f32, |a, b| a.max(b));

        format!(
            "Soak Test Summary ({:.2} hours)\n\
             -----------------------------\n\
             Samples: {}\n\
             RAM: {:.1} MB -> {:.1} MB ({:.1}% growth)\n\
             VRAM: {:.1} MB -> {:.1} MB ({:.1}% growth)\n\
             Max CPU Temp: {:.1}°C\n\
             Max GPU Temp: {:.1}°C\n\
             RTL P99: {:.2} ms",
            uptime,
            history.len(),
            first.ram_usage_mb,
            last.ram_usage_mb,
            ram_growth,
            first.vram_usage_mb,
            last.vram_usage_mb,
            vram_growth,
            max_cpu,
            max_gpu,
            last.rtl_p99_ms
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_creation() {
        let config = SoakTestTelemetryConfig::default();
        let telemetry = SoakTestTelemetry::new(config).unwrap();

        assert_eq!(telemetry.get_history().len(), 0);
    }

    #[test]
    fn test_collect_metrics() {
        let config = SoakTestTelemetryConfig::default();
        let telemetry = SoakTestTelemetry::new(config).unwrap();

        let metrics = telemetry.collect_metrics().unwrap();

        assert!(metrics.ram_usage_mb >= 0.0);
    }

    #[test]
    fn test_check_memory_leak_no_data() {
        let config = SoakTestTelemetryConfig::default();
        let telemetry = SoakTestTelemetry::new(config).unwrap();

        let result = telemetry.check_memory_leak().unwrap();
        assert!(matches!(result, MemoryLeakStatus::NoLeak));
    }

    #[test]
    fn test_memory_leak_status() {
        let no_leak = MemoryLeakStatus::NoLeak;
        let potential = MemoryLeakStatus::PotentialLeak { growth_percent: 3.0 };
        let confirmed = MemoryLeakStatus::ConfirmedLeak { growth_percent: 10.0 };

        match no_leak {
            MemoryLeakStatus::NoLeak => {}
            _ => panic!("Expected NoLeak"),
        }

        match potential {
            MemoryLeakStatus::PotentialLeak { growth_percent } => {
                assert_eq!(growth_percent, 3.0);
            }
            _ => panic!("Expected PotentialLeak"),
        }

        match confirmed {
            MemoryLeakStatus::ConfirmedLeak { growth_percent } => {
                assert_eq!(growth_percent, 10.0);
            }
            _ => panic!("Expected ConfirmedLeak"),
        }
    }

    #[test]
    fn test_generate_summary_empty() {
        let config = SoakTestTelemetryConfig::default();
        let telemetry = SoakTestTelemetry::new(config).unwrap();

        let summary = telemetry.generate_summary();
        assert!(summary.contains("No metrics collected"));
    }
}
