/*!
Intelligent Power Management (Solar Optimization)
==============================================

Monitors battery/solar state and throttles system power consumption
to extend field deployment time.

Features:
- Battery state tracking (voltage, current, cycle count, health)
- Solar power prediction
- Power mode management (Normal, Medium, Low, Critical)
- Power budget calculation
- Throttle integration with synthesis and source separation
*/

use crate::ptp::PtpTimestamp;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Battery state information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatteryState {
    pub percentage: f32, // 0.0 - 100.0
    pub voltage_v: f32,  // Typically 11.0V - 14.4V for LiFePO4
    pub current_a: f32,  // Positive = charging, Negative = discharging
    pub cycle_count: u32,
    pub health_percent: f32, // Estimated health (0-100)
    pub temperature_celsius: f32,
    pub timestamp: PtpTimestamp,
}

impl BatteryState {
    /// Check if battery is charging
    pub fn is_charging(&self) -> bool {
        self.current_a > 0.0
    }

    /// Check if battery is discharging
    pub fn is_discharging(&self) -> bool {
        self.current_a < 0.0
    }

    /// Check if battery is critically low
    pub fn is_critical(&self) -> bool {
        self.percentage < 20.0
    }

    /// Check if battery is low
    pub fn is_low(&self) -> bool {
        self.percentage < 50.0
    }

    /// Calculate estimated capacity based on health
    pub fn effective_capacity_percent(&self) -> f32 {
        self.percentage * (self.health_percent / 100.0)
    }
}

impl Default for BatteryState {
    fn default() -> Self {
        Self {
            percentage: 100.0,
            voltage_v: 13.2,
            current_a: 0.0,
            cycle_count: 0,
            health_percent: 100.0,
            temperature_celsius: 25.0,
            timestamp: PtpTimestamp::new(0, 0),
        }
    }
}

/// Power mode for system operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PowerMode {
    /// Normal mode: > 80% battery, all features enabled
    Normal,
    /// Medium mode: 50-80% battery, disable FPGA
    Medium,
    /// Low mode: 20-50% battery, disable Conv-TasNet, basic synthesis
    Low,
    /// Critical mode: < 20% battery, detection only, minimal processing
    Critical,
}

impl PowerMode {
    /// Determine power mode from battery percentage
    pub fn from_battery_percentage(percentage: f32) -> Self {
        if percentage > 80.0 {
            Self::Normal
        } else if percentage > 50.0 {
            Self::Medium
        } else if percentage > 20.0 {
            Self::Low
        } else {
            Self::Critical
        }
    }

    /// Check if FPGA should be enabled in this mode
    pub fn fpga_enabled(&self) -> bool {
        matches!(self, Self::Normal)
    }

    /// Check if source separation should be enabled in this mode
    pub fn source_separation_enabled(&self) -> bool {
        matches!(self, Self::Normal | Self::Medium)
    }

    /// Check if full synthesis should be enabled in this mode
    /// Low mode has "basic synthesis" enabled
    pub fn full_synthesis_enabled(&self) -> bool {
        matches!(self, Self::Normal | Self::Medium | Self::Low)
    }
}

/// Power budget information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerBudget {
    pub available_wh: f32,       // Watt-hours available
    pub predicted_solar_wh: f32, // Predicted solar gain (next hour)
    pub base_consumption_w: f32, // Base system consumption
    pub synthesis_consumption_w: f32,
    pub fpga_consumption_w: f32,       // FPGA consumption
    pub separation_consumption_w: f32, // Source separation consumption
    pub estimated_runtime_hours: f32,
}

impl PowerBudget {
    /// Calculate total consumption
    pub fn total_consumption_w(&self) -> f32 {
        self.base_consumption_w + self.synthesis_consumption_w + self.fpga_consumption_w + self.separation_consumption_w
    }

    /// Check if power budget is sufficient
    pub fn is_sufficient(&self, min_runtime_hours: f32) -> bool {
        self.estimated_runtime_hours >= min_runtime_hours
    }
}

/// Solar power prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolarPrediction {
    pub timestamp: PtpTimestamp,
    pub next_hour_gain_wh: f32,
    pub next_day_gain_wh: f32,
    pub confidence: f32, // 0.0 - 1.0
    pub cloud_cover_percent: f32,
}

impl Default for SolarPrediction {
    fn default() -> Self {
        Self {
            timestamp: PtpTimestamp::new(0, 0),
            next_hour_gain_wh: 50.0,
            next_day_gain_wh: 400.0,
            confidence: 0.7,
            cloud_cover_percent: 30.0,
        }
    }
}

impl SolarPrediction {
    /// Create default prediction
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if solar conditions are good
    pub fn is_good(&self) -> bool {
        self.confidence > 0.5 && self.cloud_cover_percent < 50.0
    }
}

/// Power throttle state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThrottleState {
    None,
    Throttled,
    Disabled,
}

/// Power Manager configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerManagerConfig {
    pub poll_interval_ms: u64,
    pub battery_capacity_wh: f32,      // Total battery capacity in Wh
    pub base_consumption_w: f32,       // Base system consumption
    pub synthesis_consumption_w: f32,  // Synthesis module consumption
    pub fpga_consumption_w: f32,       // FPGA module consumption
    pub separation_consumption_w: f32, // Source separation consumption
    pub mock_mode: bool,
}

impl Default for PowerManagerConfig {
    fn default() -> Self {
        Self {
            poll_interval_ms: 10000,        // Poll every 10 seconds
            battery_capacity_wh: 200.0,     // 200Wh battery
            base_consumption_w: 5.0,        // 5W base consumption
            synthesis_consumption_w: 15.0,  // 15W for synthesis
            fpga_consumption_w: 10.0,       // 10W for FPGA
            separation_consumption_w: 20.0, // 20W for source separation
            mock_mode: false,
        }
    }
}

/// Power Manager
pub struct PowerManager {
    config: PowerManagerConfig,
    battery_state: BatteryState,
    power_mode: PowerMode,
    solar_prediction: Option<SolarPrediction>,
    last_poll: Option<Instant>,
    fpga_enabled: Arc<AtomicBool>,
    source_separation_enabled: Arc<AtomicBool>,
    synthesis_enabled: Arc<AtomicBool>,
}

impl PowerManager {
    /// Create a new power manager
    pub fn new(config: PowerManagerConfig) -> Self {
        let initial_mode = PowerMode::from_battery_percentage(100.0);

        Self {
            config: config.clone(),
            battery_state: BatteryState::default(),
            power_mode: initial_mode,
            solar_prediction: None,
            last_poll: None,
            fpga_enabled: Arc::new(AtomicBool::new(initial_mode.fpga_enabled())),
            source_separation_enabled: Arc::new(AtomicBool::new(initial_mode.source_separation_enabled())),
            synthesis_enabled: Arc::new(AtomicBool::new(initial_mode.full_synthesis_enabled())),
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(PowerManagerConfig::default())
    }

    /// Create for testing (mock mode)
    #[allow(clippy::field_reassign_with_default)]
    pub fn for_testing() -> Self {
        let mut config = PowerManagerConfig::default();
        config.mock_mode = true;
        Self::new(config)
    }

    /// Poll battery state and update power mode
    pub fn poll_battery(&mut self) -> Result<BatteryState> {
        if self.config.mock_mode {
            // In mock mode, return current state with updated timestamp
            let mut state = self.battery_state.clone();
            state.timestamp = PtpTimestamp::from(chrono::Utc::now());
            self.update_battery_state(state.clone());
            self.last_poll = Some(Instant::now());
            return Ok(state);
        }

        // TODO: Implement actual battery polling via hardware interface
        // For now, return default state
        let state = BatteryState::default();
        self.update_battery_state(state.clone());
        self.last_poll = Some(Instant::now());
        Ok(state)
    }

    /// Update battery state and recalculate power mode
    fn update_battery_state(&mut self, state: BatteryState) {
        self.battery_state = state.clone();
        let new_mode = PowerMode::from_battery_percentage(state.percentage);

        if new_mode != self.power_mode {
            self.power_mode = new_mode;
            self.update_throttle_state();
        }
    }

    /// Update throttle state based on power mode
    fn update_throttle_state(&self) {
        self.fpga_enabled
            .store(self.power_mode.fpga_enabled(), Ordering::Relaxed);
        self.source_separation_enabled
            .store(self.power_mode.source_separation_enabled(), Ordering::Relaxed);
        self.synthesis_enabled
            .store(self.power_mode.full_synthesis_enabled(), Ordering::Relaxed);
    }

    /// Get current battery state
    pub fn battery_state(&self) -> &BatteryState {
        &self.battery_state
    }

    /// Get current power mode
    pub fn power_mode(&self) -> PowerMode {
        self.power_mode
    }

    /// Calculate power budget
    pub fn calculate_power_budget(&self) -> PowerBudget {
        let available_wh = self.battery_state.effective_capacity_percent() / 100.0 * self.config.battery_capacity_wh;

        let solar_gain = self
            .solar_prediction
            .as_ref()
            .map(|p| p.next_hour_gain_wh)
            .unwrap_or(0.0);

        let total_consumption = match self.power_mode {
            PowerMode::Normal => {
                self.config.base_consumption_w
                    + self.config.synthesis_consumption_w
                    + self.config.fpga_consumption_w
                    + self.config.separation_consumption_w
            }
            PowerMode::Medium => {
                self.config.base_consumption_w
                    + self.config.synthesis_consumption_w
                    + self.config.separation_consumption_w
            }
            PowerMode::Low => self.config.base_consumption_w + self.config.synthesis_consumption_w,
            PowerMode::Critical => self.config.base_consumption_w,
        };

        let synthesis_consumption = if self.power_mode.full_synthesis_enabled() {
            self.config.synthesis_consumption_w
        } else {
            0.0
        };

        let fpga_consumption = if self.power_mode.fpga_enabled() {
            self.config.fpga_consumption_w
        } else {
            0.0
        };

        let separation_consumption = if self.power_mode.source_separation_enabled() {
            self.config.separation_consumption_w
        } else {
            0.0
        };

        let estimated_runtime = (available_wh + solar_gain) / total_consumption.max(0.1);

        PowerBudget {
            available_wh,
            predicted_solar_wh: solar_gain,
            base_consumption_w: self.config.base_consumption_w,
            synthesis_consumption_w: synthesis_consumption,
            fpga_consumption_w: fpga_consumption,
            separation_consumption_w: separation_consumption,
            estimated_runtime_hours: estimated_runtime,
        }
    }

    /// Update solar prediction
    pub fn update_solar_prediction(&mut self, prediction: SolarPrediction) {
        self.solar_prediction = Some(prediction);
    }

    /// Get solar prediction
    pub fn solar_prediction(&self) -> Option<&SolarPrediction> {
        self.solar_prediction.as_ref()
    }

    /// Check if FPGA is enabled
    pub fn is_fpga_enabled(&self) -> bool {
        self.fpga_enabled.load(Ordering::Relaxed)
    }

    /// Check if source separation is enabled
    pub fn is_source_separation_enabled(&self) -> bool {
        self.source_separation_enabled.load(Ordering::Relaxed)
    }

    /// Check if full synthesis is enabled
    pub fn is_synthesis_enabled(&self) -> bool {
        self.synthesis_enabled.load(Ordering::Relaxed)
    }

    /// Get FPGA enable flag (for sharing with other modules)
    pub fn fpga_enabled_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.fpga_enabled)
    }

    /// Get source separation enable flag
    pub fn source_separation_enabled_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.source_separation_enabled)
    }

    /// Get synthesis enable flag
    pub fn synthesis_enabled_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.synthesis_enabled)
    }

    /// Check if should poll battery
    pub fn should_poll(&self) -> bool {
        match self.last_poll {
            None => true,
            Some(last) => last.elapsed() >= Duration::from_millis(self.config.poll_interval_ms),
        }
    }

    /// Check if should defer heavy tasks
    pub fn should_defer_heavy_tasks(&self) -> bool {
        // Critical mode ALWAYS defers (safety-first)
        if matches!(self.power_mode, PowerMode::Critical) {
            return true;
        }

        // Low mode defers unless good solar gain expected
        if matches!(self.power_mode, PowerMode::Low) {
            let solar_good = self.solar_prediction.as_ref().map(|p| p.is_good()).unwrap_or(false);

            return !solar_good;
        }

        // Normal and Medium modes don't defer
        false
    }

    /// Set battery state (for testing)
    #[cfg(test)]
    pub fn set_battery_state(&mut self, state: BatteryState) {
        self.update_battery_state(state);
    }

    /// Force power mode (for testing)
    #[cfg(test)]
    pub fn set_power_mode(&mut self, mode: PowerMode) {
        self.power_mode = mode;
        self.update_throttle_state();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_battery_state(percentage: f32) -> BatteryState {
        BatteryState {
            percentage,
            voltage_v: 13.2,
            current_a: -1.0,
            cycle_count: 100,
            health_percent: 95.0,
            temperature_celsius: 25.0,
            timestamp: PtpTimestamp::new(0, 0),
        }
    }

    // Battery State Tests
    #[test]
    fn test_battery_default_state() {
        let state = BatteryState::default();
        assert_eq!(state.percentage, 100.0);
        assert!(!state.is_charging());
        assert!(!state.is_discharging());
        assert!(!state.is_critical());
        assert!(!state.is_low());
    }

    #[test]
    fn test_battery_charging() {
        let state = BatteryState {
            current_a: 2.0,
            ..Default::default()
        };
        assert!(state.is_charging());
        assert!(!state.is_discharging());
    }

    #[test]
    fn test_battery_discharging() {
        let state = BatteryState {
            current_a: -2.0,
            ..Default::default()
        };
        assert!(state.is_discharging());
        assert!(!state.is_charging());
    }

    #[test]
    fn test_battery_critical() {
        let state = BatteryState {
            percentage: 15.0,
            ..Default::default()
        };
        assert!(state.is_critical());
    }

    #[test]
    fn test_battery_low() {
        let state = BatteryState {
            percentage: 30.0,
            ..Default::default()
        };
        assert!(state.is_low());
    }

    #[test]
    fn test_effective_capacity_with_degraded_health() {
        let state = BatteryState {
            percentage: 80.0,
            health_percent: 50.0,
            ..Default::default()
        };
        assert!((state.effective_capacity_percent() - 40.0).abs() < 0.01);
    }

    // Power Mode Tests
    #[test]
    fn test_power_mode_normal() {
        let mode = PowerMode::from_battery_percentage(90.0);
        assert_eq!(mode, PowerMode::Normal);
        assert!(mode.fpga_enabled());
        assert!(mode.source_separation_enabled());
        assert!(mode.full_synthesis_enabled());
    }

    #[test]
    fn test_power_mode_medium() {
        let mode = PowerMode::from_battery_percentage(60.0);
        assert_eq!(mode, PowerMode::Medium);
        assert!(!mode.fpga_enabled());
        assert!(mode.source_separation_enabled());
        assert!(mode.full_synthesis_enabled());
    }

    #[test]
    fn test_power_mode_low() {
        let mode = PowerMode::from_battery_percentage(30.0);
        assert_eq!(mode, PowerMode::Low);
        assert!(!mode.fpga_enabled());
        assert!(!mode.source_separation_enabled());
        assert!(mode.full_synthesis_enabled());
    }

    #[test]
    fn test_power_mode_critical() {
        let mode = PowerMode::from_battery_percentage(10.0);
        assert_eq!(mode, PowerMode::Critical);
        assert!(!mode.fpga_enabled());
        assert!(!mode.source_separation_enabled());
        assert!(!mode.full_synthesis_enabled());
    }

    #[test]
    fn test_power_mode_boundary_80() {
        let mode = PowerMode::from_battery_percentage(80.0);
        assert_eq!(mode, PowerMode::Medium);
    }

    #[test]
    fn test_power_mode_boundary_50() {
        let mode = PowerMode::from_battery_percentage(50.0);
        assert_eq!(mode, PowerMode::Low);
    }

    #[test]
    fn test_power_mode_boundary_20() {
        let mode = PowerMode::from_battery_percentage(20.0);
        assert_eq!(mode, PowerMode::Critical);
    }

    // Power Budget Tests
    #[test]
    fn test_power_budget_total_consumption() {
        let budget = PowerBudget {
            available_wh: 100.0,
            predicted_solar_wh: 50.0,
            base_consumption_w: 5.0,
            synthesis_consumption_w: 15.0,
            fpga_consumption_w: 10.0,
            separation_consumption_w: 20.0,
            estimated_runtime_hours: 10.0,
        };
        assert_eq!(budget.total_consumption_w(), 50.0);
    }

    #[test]
    fn test_power_budget_sufficient() {
        let budget = PowerBudget {
            available_wh: 100.0,
            predicted_solar_wh: 0.0,
            base_consumption_w: 10.0,
            synthesis_consumption_w: 0.0,
            fpga_consumption_w: 0.0,
            separation_consumption_w: 0.0,
            estimated_runtime_hours: 10.0,
        };
        assert!(budget.is_sufficient(8.0));
        assert!(!budget.is_sufficient(12.0));
    }

    // Solar Prediction Tests
    #[test]
    fn test_solar_prediction_default() {
        let prediction = SolarPrediction::default();
        assert_eq!(prediction.next_hour_gain_wh, 50.0);
        assert!(prediction.confidence > 0.5);
    }

    #[test]
    fn test_solar_prediction_good() {
        let prediction = SolarPrediction {
            confidence: 0.8,
            cloud_cover_percent: 20.0,
            ..Default::default()
        };
        assert!(prediction.is_good());
    }

    #[test]
    fn test_solar_prediction_bad() {
        let prediction = SolarPrediction {
            confidence: 0.3,
            cloud_cover_percent: 70.0,
            ..Default::default()
        };
        assert!(!prediction.is_good());
    }

    // Power Manager Tests
    #[test]
    fn test_manager_creation() {
        let manager = PowerManager::with_defaults();
        assert_eq!(manager.power_mode(), PowerMode::Normal);
        assert!(manager.is_fpga_enabled());
        assert!(manager.is_source_separation_enabled());
        assert!(manager.is_synthesis_enabled());
    }

    #[test]
    fn test_manager_for_testing() {
        let manager = PowerManager::for_testing();
        assert!(manager.config.mock_mode);
    }

    #[test]
    fn test_manager_poll_battery() {
        let mut manager = PowerManager::for_testing();
        let state = manager.poll_battery().unwrap();
        assert_eq!(state.percentage, 100.0);
    }

    #[test]
    fn test_manager_battery_state_update() {
        let mut manager = PowerManager::for_testing();
        let state = create_test_battery_state(40.0); // Changed from 60% to 40%
        manager.set_battery_state(state);

        assert_eq!(manager.power_mode(), PowerMode::Low);
        assert!(!manager.is_fpga_enabled());
        assert!(!manager.is_source_separation_enabled());
        assert!(manager.is_synthesis_enabled());
    }

    #[test]
    fn test_manager_power_mode_normal() {
        let mut manager = PowerManager::for_testing();
        manager.set_battery_state(create_test_battery_state(90.0));

        assert_eq!(manager.power_mode(), PowerMode::Normal);
        assert!(manager.is_fpga_enabled());
    }

    #[test]
    fn test_manager_power_mode_medium() {
        let mut manager = PowerManager::for_testing();
        manager.set_battery_state(create_test_battery_state(70.0));

        assert_eq!(manager.power_mode(), PowerMode::Medium);
        assert!(!manager.is_fpga_enabled());
        assert!(manager.is_source_separation_enabled());
    }

    #[test]
    fn test_manager_power_mode_low() {
        let mut manager = PowerManager::for_testing();
        manager.set_battery_state(create_test_battery_state(30.0));

        assert_eq!(manager.power_mode(), PowerMode::Low);
        assert!(!manager.is_fpga_enabled());
        assert!(!manager.is_source_separation_enabled());
    }

    #[test]
    fn test_manager_power_mode_critical() {
        let mut manager = PowerManager::for_testing();
        manager.set_battery_state(create_test_battery_state(15.0));

        assert_eq!(manager.power_mode(), PowerMode::Critical);
        assert!(!manager.is_fpga_enabled());
        assert!(!manager.is_source_separation_enabled());
        assert!(!manager.is_synthesis_enabled());
    }

    #[test]
    fn test_manager_calculate_power_budget() {
        let manager = PowerManager::with_defaults();
        let budget = manager.calculate_power_budget();

        assert!(budget.available_wh > 0.0);
        assert!(budget.estimated_runtime_hours > 0.0);
    }

    #[test]
    fn test_manager_calculate_power_budget_low_battery() {
        let mut manager = PowerManager::for_testing();
        manager.set_battery_state(create_test_battery_state(30.0));

        let budget = manager.calculate_power_budget();
        // Lower battery = less available Wh
        assert!(budget.available_wh < manager.config.battery_capacity_wh);
    }

    #[test]
    fn test_manager_solar_prediction() {
        let mut manager = PowerManager::with_defaults();
        let prediction = SolarPrediction::default();
        manager.update_solar_prediction(prediction);

        assert!(manager.solar_prediction().is_some());
    }

    #[test]
    fn test_manager_solar_prediction_affects_budget() {
        let mut manager = PowerManager::for_testing();
        manager.set_battery_state(create_test_battery_state(30.0));

        let budget_without_solar = manager.calculate_power_budget();

        manager.update_solar_prediction(SolarPrediction {
            next_hour_gain_wh: 50.0,
            ..Default::default()
        });

        let budget_with_solar = manager.calculate_power_budget();

        assert!(budget_with_solar.predicted_solar_wh > budget_without_solar.predicted_solar_wh);
    }

    #[test]
    fn test_manager_should_poll() {
        let manager = PowerManager::for_testing();
        assert!(manager.should_poll());

        let mut manager = PowerManager::for_testing();
        manager.poll_battery().unwrap();
        // Immediately after poll, should not poll again
        assert!(!manager.should_poll());
    }

    #[test]
    fn test_manager_should_defer_heavy_tasks_normal_mode() {
        let manager = PowerManager::with_defaults();
        assert!(!manager.should_defer_heavy_tasks());
    }

    #[test]
    fn test_manager_should_defer_heavy_tasks_low_mode_no_solar() {
        let mut manager = PowerManager::for_testing();
        manager.set_battery_state(create_test_battery_state(30.0));

        // Low mode without solar = defer
        assert!(manager.should_defer_heavy_tasks());
    }

    #[test]
    fn test_manager_should_defer_heavy_tasks_low_mode_with_solar() {
        let mut manager = PowerManager::for_testing();
        manager.set_battery_state(create_test_battery_state(30.0));

        manager.update_solar_prediction(SolarPrediction {
            confidence: 0.8,
            cloud_cover_percent: 20.0,
            ..Default::default()
        });

        // Low mode with good solar = don't defer
        assert!(!manager.should_defer_heavy_tasks());
    }

    #[test]
    fn test_manager_critical_mode_defers() {
        let mut manager = PowerManager::for_testing();
        manager.set_battery_state(create_test_battery_state(10.0));

        assert!(manager.should_defer_heavy_tasks());
    }

    #[test]
    fn test_manager_enable_flags() {
        let mut manager = PowerManager::for_testing();
        manager.set_battery_state(create_test_battery_state(40.0)); // Changed to Low mode

        let fpga_flag = manager.fpga_enabled_flag();
        let sep_flag = manager.source_separation_enabled_flag();
        let synth_flag = manager.synthesis_enabled_flag();

        assert!(!fpga_flag.load(Ordering::Relaxed));
        assert!(!sep_flag.load(Ordering::Relaxed));
        assert!(synth_flag.load(Ordering::Relaxed));
    }

    #[test]
    fn test_manager_flags_update_with_mode_change() {
        let mut manager = PowerManager::for_testing();

        // Start in Normal mode
        assert!(manager.is_fpga_enabled());

        // Switch to Low mode
        manager.set_battery_state(create_test_battery_state(30.0));
        assert!(!manager.is_fpga_enabled());
    }

    #[test]
    fn test_power_budget_synthesis_consumption() {
        let mut manager = PowerManager::for_testing();
        manager.set_battery_state(create_test_battery_state(90.0));

        let budget = manager.calculate_power_budget();
        // Normal mode should have synthesis consumption
        assert!(budget.synthesis_consumption_w > 0.0);
    }

    #[test]
    fn test_power_budget_no_synthesis_in_critical() {
        let mut manager = PowerManager::for_testing();
        manager.set_battery_state(create_test_battery_state(10.0));

        let budget = manager.calculate_power_budget();
        // Critical mode should have no synthesis consumption
        assert_eq!(budget.synthesis_consumption_w, 0.0);
    }

    #[test]
    fn test_effective_capacity_affects_runtime() {
        let mut manager = PowerManager::for_testing();
        manager.set_battery_state(create_test_battery_state(50.0));

        let budget1 = manager.calculate_power_budget();

        // Degrade battery health
        let mut degraded_state = create_test_battery_state(50.0);
        degraded_state.health_percent = 50.0;
        manager.set_battery_state(degraded_state);

        let budget2 = manager.calculate_power_budget();

        // Degraded battery should have less runtime
        assert!(budget2.estimated_runtime_hours < budget1.estimated_runtime_hours);
    }

    #[test]
    fn test_cycle_count_tracking() {
        let state = BatteryState {
            cycle_count: 500,
            ..Default::default()
        };
        assert_eq!(state.cycle_count, 500);
    }

    #[test]
    fn test_battery_temperature() {
        let state = BatteryState {
            temperature_celsius: 30.0,
            ..Default::default()
        };
        assert_eq!(state.temperature_celsius, 30.0);
    }

    #[test]
    fn test_voltage_range() {
        let state = BatteryState {
            voltage_v: 12.0,
            ..Default::default()
        };
        assert_eq!(state.voltage_v, 12.0);
    }

    #[test]
    fn test_power_mode_equality() {
        assert_eq!(PowerMode::Normal, PowerMode::Normal);
        assert_ne!(PowerMode::Normal, PowerMode::Critical);
    }

    #[test]
    fn test_throttle_states() {
        assert_eq!(ThrottleState::None, ThrottleState::None);
        assert_ne!(ThrottleState::None, ThrottleState::Throttled);
    }

    #[test]
    fn test_config_defaults() {
        let config = PowerManagerConfig::default();
        assert_eq!(config.poll_interval_ms, 10000);
        assert_eq!(config.battery_capacity_wh, 200.0);
    }

    #[test]
    fn test_manager_mode_transition() {
        let mut manager = PowerManager::for_testing();

        // Normal -> Medium
        manager.set_battery_state(create_test_battery_state(75.0));
        assert_eq!(manager.power_mode(), PowerMode::Medium);

        // Medium -> Low
        manager.set_battery_state(create_test_battery_state(40.0));
        assert_eq!(manager.power_mode(), PowerMode::Low);

        // Low -> Critical
        manager.set_battery_state(create_test_battery_state(15.0));
        assert_eq!(manager.power_mode(), PowerMode::Critical);

        // Critical -> Normal (recovery)
        manager.set_battery_state(create_test_battery_state(95.0));
        assert_eq!(manager.power_mode(), PowerMode::Normal);
    }

    #[test]
    fn test_solar_prediction_confidence_threshold() {
        let prediction = SolarPrediction {
            confidence: 0.6,
            cloud_cover_percent: 40.0,
            ..Default::default()
        };
        // High confidence but moderate cloud cover
        assert!(prediction.is_good());
    }

    #[test]
    fn test_solar_prediction_low_confidence() {
        let prediction = SolarPrediction {
            confidence: 0.4,
            cloud_cover_percent: 10.0,
            ..Default::default()
        };
        // Low confidence despite clear skies
        assert!(!prediction.is_good());
    }

    #[test]
    fn test_solar_prediction_high_cloud_cover() {
        let prediction = SolarPrediction {
            confidence: 0.9,
            cloud_cover_percent: 60.0,
            ..Default::default()
        };
        // High confidence but heavy cloud cover
        assert!(!prediction.is_good());
    }

    #[test]
    fn test_battery_state_cloning() {
        let state1 = BatteryState::default();
        let state2 = state1.clone();
        assert_eq!(state1.percentage, state2.percentage);
    }

    #[test]
    fn test_power_budget_calculation_with_solar() {
        let mut manager = PowerManager::for_testing();
        manager.set_battery_state(create_test_battery_state(50.0));

        manager.update_solar_prediction(SolarPrediction {
            next_hour_gain_wh: 100.0,
            ..Default::default()
        });

        let budget = manager.calculate_power_budget();
        assert_eq!(budget.predicted_solar_wh, 100.0);
    }

    #[test]
    fn test_runtime_calculation() {
        let manager = PowerManager::with_defaults();

        let budget = manager.calculate_power_budget();
        let expected_runtime = manager.config.battery_capacity_wh / budget.total_consumption_w();

        assert!((budget.estimated_runtime_hours - expected_runtime).abs() < 0.1);
    }

    #[test]
    fn test_should_defer_respects_solar() {
        let mut manager = PowerManager::for_testing();

        // Critical mode without solar
        manager.set_battery_state(create_test_battery_state(15.0));
        assert!(manager.should_defer_heavy_tasks());

        // Add good solar prediction
        manager.update_solar_prediction(SolarPrediction {
            confidence: 0.9,
            cloud_cover_percent: 10.0,
            ..Default::default()
        });

        // Still in critical mode, but with good solar
        // Current implementation: critical mode still defers
        assert!(manager.should_defer_heavy_tasks());
    }
}
