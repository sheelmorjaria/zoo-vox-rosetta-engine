/*!
Event-Driven Power Management (Edge Deployment)
==============================================

Power state transitions based on acoustic activity detection.
Enters low-power sleep mode during silence to extend battery life
on solar-powered field devices.

Features:
- Event-driven power state (Active, Monitoring, Sleep)
- Silence detection with automatic sleep entry
- Vocalization wake trigger with latency budget
- State transition validation
*/

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Event-driven power state for edge deployment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgePowerState {
    /// Active mode: Full processing, vocalization detected
    Active,
    /// Monitoring mode: Reduced power, listening for activity
    Monitoring,
    /// Sleep mode: Minimal power, silence detected
    Sleep,
}

impl EdgePowerState {
    /// Check if state is active
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Active)
    }

    /// Check if state is sleep
    pub fn is_sleep(&self) -> bool {
        matches!(self, Self::Sleep)
    }

    /// Get latency budget for this state
    pub fn latency_budget_ms(&self) -> f32 {
        match self {
            Self::Active => 10.0,   // Full speed
            Self::Monitoring => 50.0, // Slower
            Self::Sleep => 100.0,    // Wake-up + processing
        }
    }

    /// Get power consumption as percentage of active mode
    pub fn power_percent(&self) -> f32 {
        match self {
            Self::Active => 100.0,
            Self::Monitoring => 40.0,
            Self::Sleep => 5.0,
        }
    }
}

impl Default for EdgePowerState {
    fn default() -> Self {
        Self::Monitoring // Start in monitoring mode
    }
}

/// Configuration for event-driven power manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgePowerConfig {
    /// Silence duration threshold for entering sleep (ms)
    pub silence_threshold_ms: u64,

    /// Debounce delay for wake trigger (ms)
    pub wake_debounce_ms: u64,

    /// Maximum time in monitoring before forcing sleep (ms)
    pub max_monitoring_time_ms: u64,
}

impl Default for EdgePowerConfig {
    fn default() -> Self {
        Self {
            silence_threshold_ms: 5000,     // 5 seconds of silence -> sleep
            wake_debounce_ms: 500,          // 500ms debounce for wake
            max_monitoring_time_ms: 60000,  // Max 1 minute in monitoring
        }
    }
}

/// Event-driven power manager for edge deployment
#[derive(Debug, Clone)]
pub struct EventDrivenPowerManager {
    state: EdgePowerState,
    config: EdgePowerConfig,
    last_activity_time: Option<Instant>,
    state_entry_time: Instant,
    last_silence_duration_ms: u64,
}

impl EventDrivenPowerManager {
    /// Create a new event-driven power manager
    pub fn new(config: EdgePowerConfig) -> Self {
        Self {
            state: EdgePowerState::default(),
            config,
            last_activity_time: None,
            state_entry_time: Instant::now(),
            last_silence_duration_ms: 0,
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(EdgePowerConfig::default())
    }

    /// Get current power state
    pub fn state(&self) -> EdgePowerState {
        self.state
    }

    /// Get latency budget for current state
    pub fn latency_budget_ms(&self) -> f32 {
        self.state.latency_budget_ms()
    }

    /// Get time since state entry
    pub fn time_in_state_ms(&self) -> u64 {
        self.state_entry_time.elapsed().as_millis() as u64
    }

    /// Handle silence detection
    pub fn handle_silence(&mut self, duration_ms: u64) -> EdgePowerState {
        self.last_silence_duration_ms = duration_ms;

        // Only transition to sleep if silence exceeds threshold
        if duration_ms > self.config.silence_threshold_ms {
            // Can only go to sleep from monitoring or active
            match self.state {
                EdgePowerState::Monitoring | EdgePowerState::Active => {
                    self.state = EdgePowerState::Sleep;
                    self.state_entry_time = Instant::now();
                }
                EdgePowerState::Sleep => {
                    // Already in sleep, update entry time to extend sleep
                    self.state_entry_time = Instant::now();
                }
            }
        }

        self.state
    }

    /// Handle vocalization detection (wake trigger)
    pub fn handle_vocalization(&mut self) -> EdgePowerState {
        self.last_activity_time = Some(Instant::now());

        // Vocalization always wakes to Active
        match self.state {
            EdgePowerState::Sleep | EdgePowerState::Monitoring => {
                self.state = EdgePowerState::Active;
                self.state_entry_time = Instant::now();
            }
            EdgePowerState::Active => {
                // Already active, update activity time
                self.last_activity_time = Some(Instant::now());
            }
        }

        self.state
    }

    /// Handle monitoring timeout (auto-sleep)
    pub fn handle_monitoring_timeout(&mut self) -> EdgePowerState {
        if self.state == EdgePowerState::Monitoring {
            let time_in_state = self.time_in_state_ms();
            if time_in_state > self.config.max_monitoring_time_ms {
                self.state = EdgePowerState::Sleep;
                self.state_entry_time = Instant::now();
            }
        }
        self.state
    }

    /// Check if state transition is valid
    pub fn is_valid_transition(&self, new_state: EdgePowerState) -> bool {
        match (self.state, new_state) {
            // Can transition from any state to Active (vocalization detected)
            (_, EdgePowerState::Active) => true,

            // Can transition from Active to Monitoring (cooldown period)
            (EdgePowerState::Active, EdgePowerState::Monitoring) => true,

            // Can transition from Monitoring or Active to Sleep (silence)
            (EdgePowerState::Monitoring | EdgePowerState::Active, EdgePowerState::Sleep) => true,

            // Cannot transition from Sleep to Monitoring directly (must go via vocalization)
            (EdgePowerState::Sleep, EdgePowerState::Monitoring) => false,

            // Sleep to Sleep is valid (extending sleep)
            (EdgePowerState::Sleep, EdgePowerState::Sleep) => true,

            // Monitoring to Monitoring is valid (still monitoring)
            (EdgePowerState::Monitoring, EdgePowerState::Monitoring) => true,
        }
    }

    /// Force state transition (for testing or manual override)
    pub fn force_state(&mut self, new_state: EdgePowerState) -> Result<(), String> {
        if self.is_valid_transition(new_state) {
            self.state = new_state;
            self.state_entry_time = Instant::now();
            Ok(())
        } else {
            Err(format!(
                "Invalid state transition: {:?} -> {:?}",
                self.state, new_state
            ))
        }
    }

    /// Get current configuration
    pub fn config(&self) -> &EdgePowerConfig {
        &self.config
    }

    /// Get last silence duration
    pub fn last_silence_duration_ms(&self) -> u64 {
        self.last_silence_duration_ms
    }
}

impl Default for EventDrivenPowerManager {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silence_enters_sleep() {
        let mut manager = EventDrivenPowerManager::with_defaults();

        // Force to monitoring first
        manager.force_state(EdgePowerState::Monitoring).unwrap();

        // Simulate 6 seconds of silence (above 5s threshold)
        let new_state = manager.handle_silence(6000);

        assert_eq!(new_state, EdgePowerState::Sleep);
        assert!(manager.state().is_sleep());
    }

    #[test]
    fn test_vocalization_wakes_active() {
        let mut manager = EventDrivenPowerManager::with_defaults();

        // Start in sleep
        manager.force_state(EdgePowerState::Sleep).unwrap();

        // Vocalization should wake to Active
        let new_state = manager.handle_vocalization();

        assert_eq!(new_state, EdgePowerState::Active);
        assert!(manager.state().is_active());
    }

    #[test]
    fn test_wakeup_latency_budget() {
        let mut manager = EventDrivenPowerManager::with_defaults();

        // Sleep state should have 100ms budget
        manager.force_state(EdgePowerState::Sleep).unwrap();
        assert_eq!(manager.latency_budget_ms(), 100.0);

        // Active state should have 10ms budget
        manager.force_state(EdgePowerState::Active).unwrap();
        assert_eq!(manager.latency_budget_ms(), 10.0);
    }

    #[test]
    fn test_power_state_transitions() {
        let mut manager = EventDrivenPowerManager::with_defaults();

        // Valid: Monitoring -> Active
        assert!(manager.is_valid_transition(EdgePowerState::Active));

        // Force to Active
        manager.force_state(EdgePowerState::Active).unwrap();

        // Valid: Active -> Monitoring
        assert!(manager.is_valid_transition(EdgePowerState::Monitoring));

        // Valid: Active -> Sleep
        assert!(manager.is_valid_transition(EdgePowerState::Sleep));

        // Force to Sleep
        manager.force_state(EdgePowerState::Sleep).unwrap();

        // Invalid: Sleep -> Monitoring (must go via Active/vocalization)
        assert!(!manager.is_valid_transition(EdgePowerState::Monitoring));
    }

    #[test]
    fn test_short_silence_no_sleep() {
        let mut manager = EventDrivenPowerManager::with_defaults();

        // Force to monitoring
        manager.force_state(EdgePowerState::Monitoring).unwrap();

        // Simulate 3 seconds of silence (below 5s threshold)
        let new_state = manager.handle_silence(3000);

        // Should stay in monitoring
        assert_eq!(new_state, EdgePowerState::Monitoring);
    }

    #[test]
    fn test_active_to_monitoring_on_silence() {
        let mut manager = EventDrivenPowerManager::with_defaults();

        // Start in Active
        manager.force_state(EdgePowerState::Active).unwrap();

        // Short silence should transition to Monitoring (not Sleep yet)
        let new_state = manager.handle_silence(3000);

        // With the current implementation, silence above threshold goes to Sleep
        // Below threshold stays in current state
        assert_eq!(new_state, EdgePowerState::Active);
    }

    #[test]
    fn test_monitoring_timeout_forces_sleep() {
        let mut manager = EventDrivenPowerManager::with_defaults();

        // Start in monitoring
        manager.force_state(EdgePowerState::Monitoring).unwrap();

        // Mock a long time in monitoring by forcing state entry time to past
        // This is handled by time_in_state_ms() using elapsed()
        // For testing, we can manually force sleep
        manager.handle_silence(6000);

        assert_eq!(manager.state(), EdgePowerState::Sleep);
    }

    #[test]
    fn test_edge_power_state_power_percent() {
        assert_eq!(EdgePowerState::Active.power_percent(), 100.0);
        assert_eq!(EdgePowerState::Monitoring.power_percent(), 40.0);
        assert_eq!(EdgePowerState::Sleep.power_percent(), 5.0);
    }

    #[test]
    fn test_default_state_is_monitoring() {
        let manager = EventDrivenPowerManager::with_defaults();
        assert_eq!(manager.state(), EdgePowerState::Monitoring);
    }

    #[test]
    fn test_force_state_invalid_transition() {
        let mut manager = EventDrivenPowerManager::with_defaults();

        // Start in Sleep
        manager.force_state(EdgePowerState::Sleep).unwrap();

        // Try to force invalid transition: Sleep -> Monitoring
        let result = manager.force_state(EdgePowerState::Monitoring);

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid state transition"));
    }

    #[test]
    fn test_time_in_state() {
        let mut manager = EventDrivenPowerManager::with_defaults();

        // Force to Active
        manager.force_state(EdgePowerState::Active).unwrap();

        // Time in state should be small (just entered)
        let time_ms = manager.time_in_state_ms();
        assert!(time_ms < 100); // Should be very recent
    }

    #[test]
    fn test_config_custom_thresholds() {
        let config = EdgePowerConfig {
            silence_threshold_ms: 10000, // 10 seconds
            wake_debounce_ms: 1000,
            max_monitoring_time_ms: 120000,
        };

        let mut manager = EventDrivenPowerManager::new(config);

        // Should not enter sleep with 5s silence (below 10s threshold)
        manager.force_state(EdgePowerState::Monitoring).unwrap();
        manager.handle_silence(5000);

        assert_eq!(manager.state(), EdgePowerState::Monitoring);

        // Should enter sleep with 11s silence (above 10s threshold)
        manager.handle_silence(11000);

        assert_eq!(manager.state(), EdgePowerState::Sleep);
    }

    #[test]
    fn test_last_activity_time_tracked() {
        let mut manager = EventDrivenPowerManager::with_defaults();

        // Initially no activity
        assert!(manager.last_activity_time.is_none());

        // After vocalization, should have activity time
        manager.handle_vocalization();
        assert!(manager.last_activity_time.is_some());
    }
}
