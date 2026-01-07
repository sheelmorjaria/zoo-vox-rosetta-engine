/*!
Environmental Sentry (Environmental Monitor)
==========================================

Monitors environmental conditions and determines session viability.
Provides survival logic for field deployment by overriding Python requests
when conditions are unsuitable for vocalization interaction.

Features:
- Temperature, humidity, light sensor polling
- Rain intensity detection and classification
- Session viability assessment
- Solar forecasting integration
- Environmental override logic
*/

use crate::ptp::PtpTimestamp;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Rain intensity classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RainIntensity {
    None,     // 0 mm/h
    Light,    // < 2.5 mm/h
    Moderate, // 2.5 - 10 mm/h
    Heavy,    // 10 - 50 mm/h
    Storm,    // > 50 mm/h
}

impl RainIntensity {
    /// Classify rain from intensity in mm/h
    pub fn from_mm_h(mm_h: f32) -> Self {
        if mm_h <= 0.0 {
            Self::None
        } else if mm_h < 2.5 {
            Self::Light
        } else if mm_h < 10.0 {
            Self::Moderate
        } else if mm_h < 50.0 {
            Self::Heavy
        } else {
            Self::Storm
        }
    }

    /// Check if rain forces passthrough mode
    pub fn forces_passthrough(&self) -> bool {
        matches!(self, Self::Heavy | Self::Storm)
    }
}

/// Temperature classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemperatureClassification {
    Freezing, // < 0°C
    Cold,     // 0 - 10°C
    Mild,     // 10 - 25°C
    Hot,      // 25 - 35°C
    Extreme,  // > 35°C
}

impl TemperatureClassification {
    /// Classify temperature in Celsius
    pub fn from_celsius(celsius: f32) -> Self {
        if celsius < 0.0 {
            Self::Freezing
        } else if celsius < 10.0 {
            Self::Cold
        } else if celsius < 25.0 {
            Self::Mild
        } else if celsius < 35.0 {
            Self::Hot
        } else {
            Self::Extreme
        }
    }

    /// Check if temperature forces passthrough mode
    pub fn forces_passthrough(&self) -> bool {
        matches!(self, Self::Freezing | Self::Extreme)
    }
}

/// Light level classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LightLevel {
    Dark,  // < 1 lux
    Dawn,  // 1 - 10 lux
    Day,   // 10 - 10000 lux
    Dusk,  // 10000 - 100000 lux
    Night, // > 100000 lux (actually moonlight, but low)
}

impl LightLevel {
    /// Classify light level from lux
    pub fn from_lux(lux: f32) -> Self {
        if lux < 1.0 {
            Self::Dark
        } else if lux < 10.0 {
            Self::Dawn
        } else if lux < 10000.0 {
            Self::Day
        } else if lux < 100000.0 {
            Self::Dusk
        } else {
            Self::Night
        }
    }
}

/// Session viability assessment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionViability {
    Viable,     // Conditions suitable for interaction
    Marginal,   // Borderline, use caution
    Infeasible, // Conditions unsuitable, force Passthrough
}

/// Current environmental conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentalConditions {
    pub timestamp: PtpTimestamp,
    pub temperature_celsius: f32,
    pub humidity_percent: f32,
    pub light_lux: f32,
    pub rain_intensity_mm_h: f32,
    pub wind_speed_m_s: f32,
    pub atmospheric_pressure_hpa: f32,
    pub battery_temperature_celsius: f32,
}

impl Default for EnvironmentalConditions {
    fn default() -> Self {
        Self {
            timestamp: PtpTimestamp::new(0, 0),
            temperature_celsius: 22.0, // Clearly in "Mild" range (10-25°C)
            humidity_percent: 60.0,
            light_lux: 500.0,
            rain_intensity_mm_h: 0.0,
            wind_speed_m_s: 2.0,
            atmospheric_pressure_hpa: 1013.25,
            battery_temperature_celsius: 25.0,
        }
    }
}

impl EnvironmentalConditions {
    /// Get rain intensity classification
    pub fn rain_intensity(&self) -> RainIntensity {
        RainIntensity::from_mm_h(self.rain_intensity_mm_h)
    }

    /// Get temperature classification
    pub fn temperature_classification(&self) -> TemperatureClassification {
        TemperatureClassification::from_celsius(self.temperature_celsius)
    }

    /// Get light level
    pub fn light_level(&self) -> LightLevel {
        LightLevel::from_lux(self.light_lux)
    }

    /// Assess session viability
    pub fn assess_viability(&self) -> SessionViability {
        let rain = self.rain_intensity();
        let temp = self.temperature_classification();

        // Check for conditions that force passthrough
        if rain.forces_passthrough() || temp.forces_passthrough() {
            return SessionViability::Infeasible;
        }

        // Check for marginal conditions
        if matches!(rain, RainIntensity::Moderate)
            || matches!(
                temp,
                TemperatureClassification::Cold | TemperatureClassification::Hot
            )
        {
            return SessionViability::Marginal;
        }

        SessionViability::Viable
    }
}

/// Solar forecast for optimal interaction windows
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolarForecast {
    pub date: String, // ISO 8601 date
    pub sunrise_hour: u8,
    pub sunset_hour: u8,
    pub peak_solar_hours: Vec<(u8, u8)>, // (start_hour, end_hour)
    pub expected_cloud_cover_percent: f32,
    pub predicted_temperature_range: (f32, f32), // (min, max) Celsius
}

impl SolarForecast {
    /// Calculate optimal interaction windows
    pub fn optimal_windows(&self, min_duration_hours: f32) -> Vec<(u8, u8)> {
        let mut windows = Vec::new();

        // Peak solar hours are optimal
        for (start, end) in &self.peak_solar_hours {
            if (*end as f32 - *start as f32) >= min_duration_hours {
                windows.push((*start, *end));
            }
        }

        // If no peak windows, use daytime
        if windows.is_empty() {
            let day_start = self.sunrise_hour.max(8); // After 8 AM
            let day_end = self.sunset_hour.min(17); // Before 5 PM

            if day_end > day_start && (day_end - day_start) as f32 >= min_duration_hours {
                windows.push((day_start, day_end));
            }
        }

        windows
    }

    /// Create default forecast (when actual data unavailable)
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Self {
        Self {
            date: chrono::Utc::now().format("%Y-%m-%d").to_string(),
            sunrise_hour: 6,
            sunset_hour: 18,
            peak_solar_hours: vec![(10, 14)],
            expected_cloud_cover_percent: 30.0,
            predicted_temperature_range: (20.0, 30.0),
        }
    }
}

/// Environmental sensor reading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorReading {
    pub sensor_type: String,
    pub value: f32,
    pub unit: String,
    pub timestamp: PtpTimestamp,
    pub valid: bool,
}

/// Environmental Monitor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentalMonitorConfig {
    pub poll_interval_ms: u64,
    pub sensor_timeout_ms: u64,
    pub enable_rain_detection: bool,
    pub enable_solar_forecast: bool,
    pub mock_mode: bool, // For testing without real sensors
}

impl Default for EnvironmentalMonitorConfig {
    fn default() -> Self {
        Self {
            poll_interval_ms: 5000, // Poll every 5 seconds
            sensor_timeout_ms: 1000,
            enable_rain_detection: true,
            enable_solar_forecast: true,
            mock_mode: false,
        }
    }
}

/// Environmental Monitor
pub struct EnvironmentalMonitor {
    config: EnvironmentalMonitorConfig,
    current_conditions: EnvironmentalConditions,
    last_poll: Option<Instant>,
    solar_forecast: Option<SolarForecast>,
}

impl EnvironmentalMonitor {
    /// Create a new environmental monitor
    pub fn new(config: EnvironmentalMonitorConfig) -> Self {
        Self {
            config: config.clone(),
            current_conditions: EnvironmentalConditions::default(),
            last_poll: None,
            solar_forecast: if config.enable_solar_forecast {
                Some(SolarForecast::default())
            } else {
                None
            },
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(EnvironmentalMonitorConfig::default())
    }

    /// Create for testing (mock mode)
    #[allow(clippy::field_reassign_with_default)]
    pub fn for_testing() -> Self {
        let mut config = EnvironmentalMonitorConfig::default();
        config.mock_mode = true;
        Self::new(config)
    }

    /// Poll sensors and update conditions
    #[allow(clippy::field_reassign_with_default)]
    pub fn poll_sensors(&mut self) -> Result<EnvironmentalConditions> {
        if self.config.mock_mode {
            // In mock mode, return default conditions with current timestamp
            let mut conditions = EnvironmentalConditions::default();
            conditions.timestamp = PtpTimestamp::from(chrono::Utc::now());
            self.current_conditions = conditions.clone();
            self.last_poll = Some(Instant::now());
            return Ok(conditions);
        }

        // TODO: Implement actual sensor polling
        // For now, return default conditions
        let conditions = EnvironmentalConditions::default();
        self.current_conditions = conditions.clone();
        self.last_poll = Some(Instant::now());
        Ok(conditions)
    }

    /// Get current conditions
    pub fn current_conditions(&self) -> &EnvironmentalConditions {
        &self.current_conditions
    }

    /// Assess session viability
    pub fn assess_session_viability(&self) -> SessionViability {
        self.current_conditions.assess_viability()
    }

    /// Check if conditions force passthrough mode
    pub fn forces_passthrough(&self) -> bool {
        self.assess_session_viability() == SessionViability::Infeasible
    }

    /// Update solar forecast
    pub fn update_solar_forecast(&mut self, forecast: SolarForecast) {
        self.solar_forecast = Some(forecast);
    }

    /// Get solar forecast
    pub fn solar_forecast(&self) -> Option<&SolarForecast> {
        self.solar_forecast.as_ref()
    }

    /// Get optimal interaction windows
    pub fn optimal_interaction_windows(&self, min_duration_hours: f32) -> Vec<(u8, u8)> {
        match &self.solar_forecast {
            Some(forecast) => forecast.optimal_windows(min_duration_hours),
            None => vec![(10, 14)], // Default: 10 AM to 2 PM
        }
    }

    /// Check if should poll sensors
    pub fn should_poll(&self) -> bool {
        match self.last_poll {
            None => true,
            Some(last) => last.elapsed() >= Duration::from_millis(self.config.poll_interval_ms),
        }
    }

    /// Set conditions (for testing and Python bindings)
    pub fn set_conditions(&mut self, conditions: EnvironmentalConditions) {
        self.current_conditions = conditions;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_conditions() -> EnvironmentalConditions {
        EnvironmentalConditions {
            timestamp: PtpTimestamp::new(0, 0),
            temperature_celsius: 22.0, // Clearly in "Mild" range (10-25°C)
            humidity_percent: 60.0,
            light_lux: 500.0,
            rain_intensity_mm_h: 0.0,
            wind_speed_m_s: 2.0,
            atmospheric_pressure_hpa: 1013.25,
            battery_temperature_celsius: 25.0,
        }
    }

    // Rain Intensity Tests
    #[test]
    fn test_rain_intensity_none() {
        let intensity = RainIntensity::from_mm_h(0.0);
        assert_eq!(intensity, RainIntensity::None);
        assert!(!intensity.forces_passthrough());
    }

    #[test]
    fn test_rain_intensity_light() {
        let intensity = RainIntensity::from_mm_h(2.0);
        assert_eq!(intensity, RainIntensity::Light);
        assert!(!intensity.forces_passthrough());
    }

    #[test]
    fn test_rain_intensity_moderate() {
        let intensity = RainIntensity::from_mm_h(5.0);
        assert_eq!(intensity, RainIntensity::Moderate);
        assert!(!intensity.forces_passthrough());
    }

    #[test]
    fn test_rain_intensity_heavy() {
        let intensity = RainIntensity::from_mm_h(25.0);
        assert_eq!(intensity, RainIntensity::Heavy);
        assert!(intensity.forces_passthrough());
    }

    #[test]
    fn test_rain_intensity_storm() {
        let intensity = RainIntensity::from_mm_h(60.0);
        assert_eq!(intensity, RainIntensity::Storm);
        assert!(intensity.forces_passthrough());
    }

    // Temperature Classification Tests
    #[test]
    fn test_temperature_freezing() {
        let temp = TemperatureClassification::from_celsius(-5.0);
        assert_eq!(temp, TemperatureClassification::Freezing);
        assert!(temp.forces_passthrough());
    }

    #[test]
    fn test_temperature_cold() {
        let temp = TemperatureClassification::from_celsius(5.0);
        assert_eq!(temp, TemperatureClassification::Cold);
        assert!(!temp.forces_passthrough());
    }

    #[test]
    fn test_temperature_mild() {
        let temp = TemperatureClassification::from_celsius(20.0);
        assert_eq!(temp, TemperatureClassification::Mild);
        assert!(!temp.forces_passthrough());
    }

    #[test]
    fn test_temperature_hot() {
        let temp = TemperatureClassification::from_celsius(30.0);
        assert_eq!(temp, TemperatureClassification::Hot);
        assert!(!temp.forces_passthrough());
    }

    #[test]
    fn test_temperature_extreme() {
        let temp = TemperatureClassification::from_celsius(40.0);
        assert_eq!(temp, TemperatureClassification::Extreme);
        assert!(temp.forces_passthrough());
    }

    // Light Level Tests
    #[test]
    fn test_light_dark() {
        let level = LightLevel::from_lux(0.5);
        assert_eq!(level, LightLevel::Dark);
    }

    #[test]
    fn test_light_dawn() {
        let level = LightLevel::from_lux(5.0);
        assert_eq!(level, LightLevel::Dawn);
    }

    #[test]
    fn test_light_day() {
        let level = LightLevel::from_lux(1000.0);
        assert_eq!(level, LightLevel::Day);
    }

    #[test]
    fn test_light_dusk() {
        let level = LightLevel::from_lux(50000.0);
        assert_eq!(level, LightLevel::Dusk);
    }

    // Session Viability Tests
    #[test]
    fn test_viability_normal_conditions() {
        let conditions = create_test_conditions();
        let viability = conditions.assess_viability();
        assert_eq!(viability, SessionViability::Viable);
    }

    #[test]
    fn test_viability_heavy_rain_forces_passthrough() {
        let mut conditions = create_test_conditions();
        conditions.rain_intensity_mm_h = 25.0;
        let viability = conditions.assess_viability();
        assert_eq!(viability, SessionViability::Infeasible);
    }

    #[test]
    fn test_viability_storm_forces_passthrough() {
        let mut conditions = create_test_conditions();
        conditions.rain_intensity_mm_h = 60.0;
        let viability = conditions.assess_viability();
        assert_eq!(viability, SessionViability::Infeasible);
    }

    #[test]
    fn test_viability_freezing_forces_passthrough() {
        let mut conditions = create_test_conditions();
        conditions.temperature_celsius = -5.0;
        let viability = conditions.assess_viability();
        assert_eq!(viability, SessionViability::Infeasible);
    }

    #[test]
    fn test_viability_extreme_heat_forces_passthrough() {
        let mut conditions = create_test_conditions();
        conditions.temperature_celsius = 40.0;
        let viability = conditions.assess_viability();
        assert_eq!(viability, SessionViability::Infeasible);
    }

    #[test]
    fn test_viability_moderate_rain_is_marginal() {
        let mut conditions = create_test_conditions();
        conditions.rain_intensity_mm_h = 5.0;
        let viability = conditions.assess_viability();
        assert_eq!(viability, SessionViability::Marginal);
    }

    #[test]
    fn test_viability_cold_is_marginal() {
        let mut conditions = create_test_conditions();
        conditions.temperature_celsius = 5.0;
        let viability = conditions.assess_viability();
        assert_eq!(viability, SessionViability::Marginal);
    }

    #[test]
    fn test_viability_hot_is_marginal() {
        let mut conditions = create_test_conditions();
        conditions.temperature_celsius = 30.0;
        let viability = conditions.assess_viability();
        assert_eq!(viability, SessionViability::Marginal);
    }

    // Environmental Monitor Tests
    #[test]
    fn test_monitor_creation() {
        let monitor = EnvironmentalMonitor::with_defaults();
        assert!(!monitor.forces_passthrough());
    }

    #[test]
    fn test_monitor_for_testing() {
        let mut monitor = EnvironmentalMonitor::for_testing();
        let conditions = monitor.poll_sensors().unwrap();
        assert_eq!(conditions.temperature_celsius, 22.0);
    }

    #[test]
    fn test_monitor_poll_updates_conditions() {
        let mut monitor = EnvironmentalMonitor::for_testing();
        monitor.poll_sensors().unwrap();
        assert!(monitor.last_poll.is_some());
    }

    #[test]
    fn test_monitor_should_poll() {
        let monitor = EnvironmentalMonitor::for_testing();
        assert!(monitor.should_poll());
    }

    #[test]
    fn test_monitor_should_not_poll_too_soon() {
        let mut monitor = EnvironmentalMonitor::for_testing();
        monitor.poll_sensors().unwrap();
        // Immediately after poll, should not poll again
        assert!(!monitor.should_poll());
    }

    #[test]
    fn test_monitor_assess_viability() {
        let monitor = EnvironmentalMonitor::for_testing();
        let viability = monitor.assess_session_viability();
        assert_eq!(viability, SessionViability::Viable);
    }

    #[test]
    fn test_monitor_forces_passthrough_with_heavy_rain() {
        let mut monitor = EnvironmentalMonitor::for_testing();
        let mut conditions = create_test_conditions();
        conditions.rain_intensity_mm_h = 25.0;
        monitor.set_conditions(conditions);
        assert!(monitor.forces_passthrough());
    }

    #[test]
    fn test_monitor_forces_passthrough_with_freezing() {
        let mut monitor = EnvironmentalMonitor::for_testing();
        let mut conditions = create_test_conditions();
        conditions.temperature_celsius = -5.0;
        monitor.set_conditions(conditions);
        assert!(monitor.forces_passthrough());
    }

    #[test]
    fn test_monitor_does_not_force_passthrough_in_normal_conditions() {
        let monitor = EnvironmentalMonitor::for_testing();
        assert!(!monitor.forces_passthrough());
    }

    // Solar Forecast Tests
    #[test]
    fn test_solar_forecast_default() {
        let forecast = SolarForecast::default();
        assert_eq!(forecast.sunrise_hour, 6);
        assert_eq!(forecast.sunset_hour, 18);
        assert!(!forecast.peak_solar_hours.is_empty());
    }

    #[test]
    fn test_solar_forecast_optimal_windows() {
        let forecast = SolarForecast::default();
        let windows = forecast.optimal_windows(2.0);
        assert!(!windows.is_empty());
    }

    #[test]
    fn test_solar_forecast_optimal_windows_minimum_duration() {
        let forecast = SolarForecast::default();
        // Request 5 hour window, but peak is only 4 hours
        let windows = forecast.optimal_windows(5.0);
        // Should return daytime window instead
        assert!(!windows.is_empty());
    }

    #[test]
    fn test_monitor_solar_forecast() {
        let monitor = EnvironmentalMonitor::with_defaults();
        let forecast = monitor.solar_forecast();
        assert!(forecast.is_some());
    }

    #[test]
    fn test_monitor_update_solar_forecast() {
        let mut monitor = EnvironmentalMonitor::with_defaults();
        let new_forecast = SolarForecast {
            date: "2024-01-01".to_string(),
            sunrise_hour: 7,
            sunset_hour: 17,
            peak_solar_hours: vec![(11, 13)],
            expected_cloud_cover_percent: 50.0,
            predicted_temperature_range: (15.0, 25.0),
        };
        monitor.update_solar_forecast(new_forecast.clone());

        let retrieved = monitor.solar_forecast().unwrap();
        assert_eq!(retrieved.sunrise_hour, 7);
    }

    #[test]
    fn test_monitor_optimal_windows() {
        let monitor = EnvironmentalMonitor::with_defaults();
        let windows = monitor.optimal_interaction_windows(2.0);
        assert!(!windows.is_empty());
    }

    #[test]
    fn test_monitor_optimal_windows_no_forecast() {
        let config = EnvironmentalMonitorConfig {
            enable_solar_forecast: false,
            ..Default::default()
        };
        let monitor = EnvironmentalMonitor::new(config);
        let windows = monitor.optimal_interaction_windows(2.0);
        // Should return default window
        assert_eq!(windows, vec![(10, 14)]);
    }

    // Test Conditions Accessor
    #[test]
    fn test_conditions_rain_intensity() {
        let conditions = create_test_conditions();
        assert_eq!(conditions.rain_intensity(), RainIntensity::None);
    }

    #[test]
    fn test_conditions_temperature_classification() {
        let conditions = create_test_conditions();
        assert_eq!(
            conditions.temperature_classification(),
            TemperatureClassification::Mild
        );
    }

    #[test]
    fn test_conditions_light_level() {
        let conditions = create_test_conditions();
        assert_eq!(conditions.light_level(), LightLevel::Day);
    }

    #[test]
    fn test_conditions_heavy_rain_classification() {
        let mut conditions = create_test_conditions();
        conditions.rain_intensity_mm_h = 25.0;
        assert_eq!(conditions.rain_intensity(), RainIntensity::Heavy);
    }

    #[test]
    fn test_conditions_hot_classification() {
        let mut conditions = create_test_conditions();
        conditions.temperature_celsius = 30.0;
        assert_eq!(
            conditions.temperature_classification(),
            TemperatureClassification::Hot
        );
    }

    #[test]
    fn test_conditions_dark_classification() {
        let mut conditions = create_test_conditions();
        conditions.light_lux = 0.5;
        assert_eq!(conditions.light_level(), LightLevel::Dark);
    }

    // Combined Conditions Tests
    #[test]
    fn test_multiple_adverse_conditions() {
        let mut conditions = create_test_conditions();
        conditions.rain_intensity_mm_h = 5.0; // Moderate
        conditions.temperature_celsius = 30.0; // Hot
                                               // Should be marginal, not infeasible
        assert_eq!(conditions.assess_viability(), SessionViability::Marginal);
    }

    #[test]
    fn test_perfect_conditions() {
        let conditions = EnvironmentalConditions {
            temperature_celsius: 22.0,
            humidity_percent: 50.0,
            light_lux: 1000.0,
            rain_intensity_mm_h: 0.0,
            ..Default::default()
        };
        assert_eq!(conditions.assess_viability(), SessionViability::Viable);
    }
}
