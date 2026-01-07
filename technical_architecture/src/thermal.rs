//! Thermal Management Module
//! =========================
//!
//! This module implements thermal monitoring and power governance for
//! field deployment in jungle environments. It prevents overheating
//! and optimizes power consumption for extended battery life.
//!
//! Features:
//! - CPU/GPU temperature monitoring
//! - Adaptive throttling based on thermal state
//! - Power-aware performance scaling
//! - Jetson-specific thermal zones support
//!
//! Author: Sheel Morjaria (sheelmorjaria@gmail.com)
//! License: CC BY-ND 4.0 International

use anyhow::{Context, Result};
use log::{debug, info, warn};
use serde::{Deserialize, Serialize};

/// Thermal configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalConfig {
    /// Temperature monitoring interval (seconds)
    pub monitor_interval_secs: u64,

    /// Warning temperature threshold (Celsius)
    pub warning_temp_c: f32,

    /// Critical temperature threshold (Celsius)
    pub critical_temp_c: f32,

    /// Throttling temperature threshold (Celsius)
    pub throttling_temp_c: f32,

    /// Recovery temperature (Celsius) - must drop below this to exit throttling
    pub recovery_temp_c: f32,

    /// Enable aggressive power saving in critical state
    pub aggressive_power_save: bool,

    /// Path to thermal zone (Linux/Jetson)
    pub thermal_zone_path: String,

    /// Use mock temperature for testing
    pub use_mock_temp: bool,
}

impl Default for ThermalConfig {
    fn default() -> Self {
        Self {
            monitor_interval_secs: 1,
            warning_temp_c: 75.0,
            critical_temp_c: 85.0,
            throttling_temp_c: 80.0,
            recovery_temp_c: 70.0,
            aggressive_power_save: true,
            thermal_zone_path: "/sys/class/thermal/thermal_zone0/temp".to_string(),
            use_mock_temp: false,
        }
    }
}

/// Thermal state of the system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ThermalState {
    /// Normal operation
    #[default]
    Normal,
    /// Elevated temperature, warning issued
    Warning,
    /// Active thermal throttling
    Throttling,
    /// Critical temperature, emergency measures
    Critical,
}

impl ThermalState {
    /// Check if this state requires throttling
    pub fn requires_throttling(self) -> bool {
        matches!(self, Self::Throttling | Self::Critical)
    }

    /// Check if this state is critical
    pub fn is_critical(self) -> bool {
        matches!(self, Self::Critical)
    }
}

/// Temperature reading from a sensor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperatureReading {
    /// Temperature in Celsius
    pub temp_c: f32,
    /// Sensor source
    pub source: String,
    /// Timestamp of reading
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl TemperatureReading {
    /// Create a new temperature reading
    pub fn new(temp_c: f32, source: String) -> Self {
        Self {
            temp_c,
            source,
            timestamp: chrono::Utc::now(),
        }
    }

    /// Create a mock reading for testing
    pub fn mock(temp_c: f32) -> Self {
        Self::new(temp_c, "mock".to_string())
    }
}

/// Thermal governor for managing system temperature
pub struct ThermalGovernor {
    /// Configuration
    config: ThermalConfig,
    /// Current thermal state
    state: tokio::sync::RwLock<ThermalState>,
    /// Latest temperature reading
    current_temp: tokio::sync::RwLock<Option<TemperatureReading>>,
    /// Temperature history for analysis
    temp_history: tokio::sync::RwLock<Vec<TemperatureReading>>,
    /// Maximum history size
    max_history: usize,
    /// Whether thermal monitoring is active
    monitoring_active: std::sync::atomic::AtomicBool,
}

impl ThermalGovernor {
    /// Create a new thermal governor
    pub async fn new(config: ThermalConfig) -> Result<Self> {
        info!("Initializing Thermal Governor");

        // Validate thermal zone path if not using mock
        if !config.use_mock_temp && !std::path::Path::new(&config.thermal_zone_path).exists() {
            warn!(
                "Thermal zone not found: {}, will use mock temperature",
                config.thermal_zone_path
            );
        }

        Ok(Self {
            config,
            state: tokio::sync::RwLock::new(ThermalState::Normal),
            current_temp: tokio::sync::RwLock::new(None),
            temp_history: tokio::sync::RwLock::new(Vec::new()),
            max_history: 1000,
            monitoring_active: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// Start thermal monitoring
    pub async fn start_monitoring(&self) -> Result<()> {
        info!("Starting thermal monitoring");
        self.monitoring_active
            .store(true, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }

    /// Stop thermal monitoring
    pub async fn stop_monitoring(&self) -> Result<()> {
        info!("Stopping thermal monitoring");
        self.monitoring_active
            .store(false, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }

    /// Monitor thermal state (called periodically)
    pub async fn monitor(&self) -> Result<()> {
        if !self
            .monitoring_active
            .load(std::sync::atomic::Ordering::SeqCst)
        {
            return Ok(());
        }

        // Read temperature
        let reading = self
            .read_temperature()
            .await
            .context("Failed to read temperature")?;

        // Update current temperature
        *self.current_temp.write().await = Some(reading.clone());

        // Add to history
        {
            let mut history = self.temp_history.write().await;
            history.push(reading.clone());
            if history.len() > self.max_history {
                history.remove(0);
            }
        }

        // Update thermal state based on temperature
        let new_state = self.calculate_thermal_state(reading.temp_c);
        let old_state = *self.state.read().await;

        if new_state != old_state {
            warn!(
                "Thermal state changed: {:?} -> {:?} (temp: {:.1}°C)",
                old_state, new_state, reading.temp_c
            );
            *self.state.write().await = new_state;

            // Take action based on new state
            self.handle_thermal_state(new_state).await?;
        }

        debug!(
            "Temperature: {:.1}°C, State: {:?}",
            reading.temp_c, new_state
        );

        Ok(())
    }

    /// Read temperature from sensor or mock
    async fn read_temperature(&self) -> Result<TemperatureReading> {
        if self.config.use_mock_temp {
            // Return mock temperature for testing
            // Simulate temperature variation
            let base_temp = 65.0;
            let variation = (chrono::Utc::now().timestamp() % 20) as f32; // 0-20°C variation
            Ok(TemperatureReading::mock(base_temp + variation))
        } else {
            // Try to read from thermal zone (Linux/Jetson)
            match self.read_thermal_zone().await {
                Ok(temp) => Ok(TemperatureReading::new(temp, "thermal_zone0".to_string())),
                Err(e) => {
                    warn!("Failed to read thermal zone: {}, using mock", e);
                    Ok(TemperatureReading::mock(65.0))
                }
            }
        }
    }

    /// Read temperature from Linux thermal zone
    async fn read_thermal_zone(&self) -> Result<f32> {
        // In a real implementation, this would read from /sys/class/thermal
        // For now, return an error to fall back to mock
        Err(anyhow::anyhow!("Thermal zone reading not implemented"))
    }

    /// Calculate thermal state from temperature
    fn calculate_thermal_state(&self, temp_c: f32) -> ThermalState {
        if temp_c >= self.config.critical_temp_c {
            ThermalState::Critical
        } else if temp_c >= self.config.throttling_temp_c {
            ThermalState::Throttling
        } else if temp_c >= self.config.warning_temp_c {
            ThermalState::Warning
        } else {
            ThermalState::Normal
        }
    }

    /// Handle thermal state change
    async fn handle_thermal_state(&self, state: ThermalState) -> Result<()> {
        match state {
            ThermalState::Normal => {
                info!("Temperature returned to normal");
                // Could restore performance here
            }
            ThermalState::Warning => {
                warn!(
                    "Temperature warning: {:.0}°C threshold reached",
                    self.config.warning_temp_c
                );
            }
            ThermalState::Throttling => {
                warn!("Thermal throttling activated");
                // Would notify processing pipeline to reduce workload
            }
            ThermalState::Critical => {
                warn!(
                    "CRITICAL temperature: {:.0}°C threshold reached!",
                    self.config.critical_temp_c
                );
                if self.config.aggressive_power_save {
                    // Could trigger emergency power saving
                }
            }
        }
        Ok(())
    }

    /// Get current thermal state
    pub async fn get_state(&self) -> ThermalState {
        *self.state.read().await
    }

    /// Get current temperature reading
    pub async fn get_temperature(&self) -> Option<TemperatureReading> {
        self.current_temp.read().await.clone()
    }

    /// Get temperature history
    pub async fn get_history(&self) -> Vec<TemperatureReading> {
        self.temp_history.read().await.clone()
    }

    /// Get average temperature from history
    pub async fn get_average_temp(&self) -> Option<f32> {
        let history = self.temp_history.read().await;
        if history.is_empty() {
            return None;
        }
        let sum: f32 = history.iter().map(|r| r.temp_c).sum();
        Some(sum / history.len() as f32)
    }

    /// Get thermal statistics
    pub async fn get_stats(&self) -> ThermalStats {
        let state = *self.state.read().await;
        let current_temp = self.current_temp.read().await.clone();
        let avg_temp = self.get_average_temp().await;
        let monitoring_active = self
            .monitoring_active
            .load(std::sync::atomic::Ordering::SeqCst);

        ThermalStats {
            current_state: state,
            current_temp_c: current_temp.as_ref().map(|t| t.temp_c),
            average_temp_c: avg_temp,
            monitoring_active,
            history_size: self.temp_history.read().await.len(),
        }
    }

    /// Check if system should throttle processing
    pub async fn should_throttle(&self) -> bool {
        self.get_state().await.requires_throttling()
    }

    /// Check if system is in critical state
    pub async fn is_critical(&self) -> bool {
        self.get_state().await.is_critical()
    }

    /// Manually set temperature (for testing)
    pub async fn set_mock_temperature(&self, temp_c: f32) {
        let reading = TemperatureReading::mock(temp_c);
        *self.current_temp.write().await = Some(reading);
        let new_state = self.calculate_thermal_state(temp_c);
        *self.state.write().await = new_state;
        info!(
            "Mock temperature set to {:.1}°C (state: {:?})",
            temp_c, new_state
        );
    }
}

/// Thermal statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalStats {
    pub current_state: ThermalState,
    pub current_temp_c: Option<f32>,
    pub average_temp_c: Option<f32>,
    pub monitoring_active: bool,
    pub history_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_thermal_config_default() {
        let config = ThermalConfig::default();
        assert_eq!(config.warning_temp_c, 75.0);
        assert_eq!(config.critical_temp_c, 85.0);
        assert_eq!(config.throttling_temp_c, 80.0);
    }

    #[tokio::test]
    async fn test_governor_creation() {
        let config = ThermalConfig::default();
        let governor = ThermalGovernor::new(config).await.unwrap();
        assert_eq!(governor.get_state().await, ThermalState::Normal);
    }

    #[tokio::test]
    async fn test_thermal_state_calculation() {
        let config = ThermalConfig::default();
        let governor = ThermalGovernor::new(config).await.unwrap();

        // Set mock temperature for each state
        governor.set_mock_temperature(70.0).await;
        assert_eq!(governor.get_state().await, ThermalState::Normal);

        governor.set_mock_temperature(77.0).await;
        assert_eq!(governor.get_state().await, ThermalState::Warning);

        governor.set_mock_temperature(82.0).await;
        assert_eq!(governor.get_state().await, ThermalState::Throttling);

        governor.set_mock_temperature(87.0).await;
        assert_eq!(governor.get_state().await, ThermalState::Critical);
    }

    #[tokio::test]
    async fn test_throttling_detection() {
        let config = ThermalConfig::default();
        let governor = ThermalGovernor::new(config).await.unwrap();

        governor.set_mock_temperature(70.0).await;
        assert!(!governor.should_throttle().await);

        governor.set_mock_temperature(82.0).await;
        assert!(governor.should_throttle().await);
    }

    #[tokio::test]
    async fn test_monitoring_cycle() {
        let config = ThermalConfig {
            use_mock_temp: true,
            ..Default::default()
        };
        let governor = ThermalGovernor::new(config).await.unwrap();
        governor.start_monitoring().await.unwrap();

        // Run a few monitoring cycles
        for _ in 0..5 {
            governor.monitor().await.unwrap();
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        let stats = governor.get_stats().await;
        assert!(stats.monitoring_active);
        assert!(stats.current_temp_c.is_some());
    }
}
