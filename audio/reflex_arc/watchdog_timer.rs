use std::time::{Duration, Instant};

pub struct WatchdogTimer {
    timeout: Duration,
    last_update: Instant,
    triggered: bool,
}

impl WatchdogTimer {
    pub fn new(timeout: Duration) -> Self {
        Self {
            timeout,
            last_update: Instant::now(),
            triggered: false,
        }
    }

    pub fn update(&mut self) {
        self.last_update = Instant::now();
        self.triggered = false;
    }

    pub fn should_trigger(&mut self) -> bool {
        let elapsed = self.last_update.elapsed();

        if elapsed > self.timeout && !self.triggered {
            self.triggered = true;
            true
        } else {
            false
        }
    }

    pub fn get_elapsed_time(&self) -> Duration {
        self.last_update.elapsed()
    }

    pub fn is_healthy(&self) -> bool {
        self.last_update.elapsed() < self.timeout
    }

    pub fn reset(&mut self) {
        self.last_update = Instant::now();
        self.triggered = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_watchdog_initialization() {
        let mut watchdog = WatchdogTimer::new(Duration::from_millis(100));
        assert!(watchdog.is_healthy());
        assert!(!watchdog.should_trigger());
    }

    #[test]
    fn test_watchdog_update() {
        let mut watchdog = WatchdogTimer::new(Duration::from_millis(100));

        // Update immediately
        watchdog.update();
        assert!(watchdog.is_healthy());
        assert!(!watchdog.should_trigger());

        // Wait a little
        std::thread::sleep(Duration::from_millis(50));
        watchdog.update();
        assert!(watchdog.is_healthy());
        assert!(!watchdog.should_trigger());
    }

    #[test]
    fn test_watchdog_trigger() {
        let mut watchdog = WatchdogTimer::new(Duration::from_millis(50));

        // Let it timeout
        std::thread::sleep(Duration::from_millis(60));

        assert!(!watchdog.is_healthy());
        assert!(watchdog.should_trigger());

        // Should not trigger again
        assert!(!watchdog.should_trigger());
    }

    #[test]
    fn test_watchdog_reset() {
        let mut watchdog = WatchdogTimer::new(Duration::from_millis(50));

        // Let it timeout
        std::thread::sleep(Duration::from_millis(60));
        watchdog.should_trigger(); // Trigger once

        // Reset
        watchdog.reset();
        assert!(watchdog.is_healthy());
        assert!(!watchdog.should_trigger());
    }
}