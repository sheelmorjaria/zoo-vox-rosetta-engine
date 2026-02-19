//! Async Semiotic State Sharing for Rust-Python Decoupling
//!
//! Implements the Async Decoupling Architecture:
//! - Python Slow Path (10-20 Hz): Updates SemioticState periodically
//! - Rust Fast Path (audio rate): Reads latest state without blocking
//!
//! This enables <50ms response time by ensuring Rust never waits for Python.
//! If Python takes 600ms to update, Rust continues using the previous strategy.
//!
//! Architecture:
//! ```
//! ┌─────────────────────────────────────────────────────────────────┐
//! │  Python Slow Path (100ms)         Rust Fast Path (20ms blocks)  │
//! │  ─────────────────────────        ────────────────────────────  │
//! │                                                                   │
//! │  Audio ──► Rosetta ──┐                       ┌──► Synthesis ──► Out
//! │                       │                       │
//! │                       ▼                       ▲
//! │              ┌─────────────────────┐
//! │              │  SharedSemioticState │
//! │              │  (Arc<RwLock<T>>)    │
//! │              └─────────────────────┘
//! │                       ▲                       │
//! │                       │                       │
//! │  Semiotic ────────────┘                       │
//! │  Analysis                                      │
//! │  (updates state)                               │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

// =============================================================================
// Semiotic State Types (mirrors Python)
// =============================================================================

/// Response modification based on semiotic analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseModification {
    /// Standard response
    Normal,
    /// Acknowledge but don't echo deception
    DeceptionAcknowledge,
    /// Ignore deceptive signal
    DeceptionIgnore,
    /// Log novel phrase for review
    EmergenceLog,
    /// Echo novel behavior for observation
    EmergenceEcho,
    /// Reply to specific target
    DirectedReply,
    /// Boost response intensity
    UrgencyBoost,
    /// Reduce response intensity (calming)
    UrgencyReduce,
}

impl Default for ResponseModification {
    fn default() -> Self {
        Self::Normal
    }
}

/// Context states from probabilistic context machine
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextState {
    Silence,
    Contact,
    Alarm,
    Food,
    Neutral,
    Uncertain,
}

impl Default for ContextState {
    fn default() -> Self {
        Self::Neutral
    }
}

/// Semiotic scores from analysis
#[derive(Debug, Clone, Copy)]
pub struct SemioticScores {
    /// 0.0-1.0: Is this call deceptive?
    pub deception: f32,
    /// 0.0-1.0: Is this a novel behavior?
    pub emergence: f32,
    /// 0.0-1.0: Is this intentionally directed?
    pub directed: f32,
    /// Confidence in the analysis
    pub confidence: f32,
}

impl Default for SemioticScores {
    fn default() -> Self {
        Self {
            deception: 0.0,
            emergence: 0.0,
            directed: 0.0,
            confidence: 0.5,
        }
    }
}

/// The shared semiotic state updated by Python, read by Rust
#[derive(Debug, Clone)]
pub struct SemioticState {
    /// Current semiotic scores
    pub scores: SemioticScores,

    /// Response modification recommendation
    pub response_modification: ResponseModification,

    /// Current context from probabilistic machine
    pub context: ContextState,

    /// Context confidence (0.0-1.0)
    pub context_confidence: f32,

    /// Predicted next context
    pub predicted_context: ContextState,

    /// Communication target (if directed)
    pub communication_target: Option<String>,

    /// Whether deception was detected
    pub deception_detected: bool,

    /// Whether emergence was detected
    pub emergence_detected: bool,

    /// Effectiveness score of last response (0.0-1.0)
    pub last_effectiveness: f32,

    /// Timestamp of last update (for staleness detection)
    pub last_update: Instant,

    /// Number of updates received
    pub update_count: u64,
}

impl Default for SemioticState {
    fn default() -> Self {
        Self {
            scores: SemioticScores::default(),
            response_modification: ResponseModification::Normal,
            context: ContextState::Neutral,
            context_confidence: 0.5,
            predicted_context: ContextState::Neutral,
            communication_target: None,
            deception_detected: false,
            emergence_detected: false,
            last_effectiveness: 0.5,
            last_update: Instant::now(),
            update_count: 0,
        }
    }
}

impl SemioticState {
    /// Check if the state is stale (hasn't been updated recently)
    pub fn is_stale(&self, max_age: Duration) -> bool {
        self.last_update.elapsed() > max_age
    }

    /// Check if we should respond based on semiotic analysis
    pub fn should_respond(&self) -> bool {
        // Don't respond to high-confidence deception
        if self.scores.deception > 0.85 {
            return false;
        }
        // Don't respond in silence context
        if self.context == ContextState::Silence {
            return false;
        }
        true
    }

    /// Check if we should echo the input or use a different response
    pub fn should_echo(&self) -> bool {
        // Don't echo deceptive signals
        if self.deception_detected {
            return false;
        }
        // Echo novel behaviors for observation
        if self.emergence_detected {
            return true;
        }
        true
    }

    /// Get recommended response label modification
    pub fn get_response_label(&self, input_label: &str, default_response: &str) -> String {
        match self.response_modification {
            ResponseModification::DeceptionAcknowledge | ResponseModification::DeceptionIgnore => {
                // Use calming response for deceptive signals
                if input_label == "Tsik" {
                    "Phee".to_string()
                } else {
                    default_response.to_string()
                }
            }
            ResponseModification::EmergenceEcho => {
                // Echo the input for observation
                input_label.to_string()
            }
            _ => default_response.to_string(),
        }
    }
}

// =============================================================================
// Thread-Safe Shared State
// =============================================================================

/// Thread-safe wrapper for sharing SemioticState between Python and Rust
///
/// Uses RwLock for:
/// - Multiple readers (Rust fast path) don't block each other
/// - Single writer (Python slow path) has exclusive access during update
#[derive(Clone)]
pub struct SharedSemioticState {
    inner: Arc<RwLock<SemioticState>>,
}

impl Default for SharedSemioticState {
    fn default() -> Self {
        Self::new()
    }
}

impl SharedSemioticState {
    /// Create a new shared state
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(SemioticState::default())),
        }
    }

    /// Read the current state (non-blocking for Rust fast path)
    ///
    /// Returns a cloned copy of the state to avoid holding the lock.
    /// If the lock is poisoned or unavailable, returns the default state.
    pub fn read(&self) -> SemioticState {
        match self.inner.read() {
            Ok(state) => state.clone(),
            Err(_) => SemioticState::default(),
        }
    }

    /// Try to read with a timeout (for cases where we want to wait briefly)
    pub fn try_read_timeout(&self, timeout: Duration) -> Option<SemioticState> {
        let start = Instant::now();
        loop {
            match self.inner.try_read() {
                Ok(state) => return Some(state.clone()),
                Err(std::sync::TryLockError::Poisoned(_)) => {
                    return Some(SemioticState::default());
                }
                Err(std::sync::TryLockError::WouldBlock) => {
                    if start.elapsed() > timeout {
                        return None;
                    }
                    std::thread::yield_now();
                }
            }
        }
    }

    /// Update the state (called by Python slow path)
    ///
    /// This should be called every 50-100ms by the Python cognitive layer.
    pub fn update(&self, new_state: SemioticState) {
        match self.inner.write() {
            Ok(mut state) => {
                *state = new_state;
                state.last_update = Instant::now();
                state.update_count += 1;
            }
            Err(_) => {
                // Lock is poisoned, reset to default
                if let Ok(mut state) = self.inner.write() {
                    *state = SemioticState::default();
                }
            }
        }
    }

    /// Update only the semiotic scores (partial update)
    pub fn update_scores(&self, scores: SemioticScores) {
        if let Ok(mut state) = self.inner.write() {
            state.scores = scores;
            state.deception_detected = scores.deception > 0.7;
            state.emergence_detected = scores.emergence > 0.6;
            state.last_update = Instant::now();
            state.update_count += 1;
        }
    }

    /// Update only the context (partial update)
    pub fn update_context(&self, context: ContextState, confidence: f32, predicted: ContextState) {
        if let Ok(mut state) = self.inner.write() {
            state.context = context;
            state.context_confidence = confidence;
            state.predicted_context = predicted;
            state.last_update = Instant::now();
            state.update_count += 1;
        }
    }

    /// Update effectiveness score
    pub fn update_effectiveness(&self, effectiveness: f32) {
        if let Ok(mut state) = self.inner.write() {
            state.last_effectiveness = effectiveness;
            state.last_update = Instant::now();
        }
    }

    /// Get update statistics
    pub fn stats(&self) -> (u64, bool) {
        match self.inner.read() {
            Ok(state) => (state.update_count, state.is_stale(Duration::from_millis(500))),
            Err(_) => (0, true),
        }
    }
}

// =============================================================================
// Python FFI Helpers (for PyO3 integration)
// =============================================================================

/// Builder for creating SemioticState from Python
#[derive(Debug, Clone, Default)]
pub struct SemioticStateBuilder {
    state: SemioticState,
}

impl SemioticStateBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn deception(mut self, score: f32) -> Self {
        self.state.scores.deception = score;
        self.state.deception_detected = score > 0.7;
        self
    }

    pub fn emergence(mut self, score: f32) -> Self {
        self.state.scores.emergence = score;
        self.state.emergence_detected = score > 0.6;
        self
    }

    pub fn directed(mut self, score: f32) -> Self {
        self.state.scores.directed = score;
        self
    }

    pub fn confidence(mut self, confidence: f32) -> Self {
        self.state.scores.confidence = confidence;
        self
    }

    pub fn context(mut self, context: ContextState, confidence: f32) -> Self {
        self.state.context = context;
        self.state.context_confidence = confidence;
        self
    }

    pub fn predicted_context(mut self, predicted: ContextState) -> Self {
        self.state.predicted_context = predicted;
        self
    }

    pub fn response_modification(mut self, modification: ResponseModification) -> Self {
        self.state.response_modification = modification;
        self
    }

    pub fn communication_target(mut self, target: Option<String>) -> Self {
        self.state.communication_target = target;
        self
    }

    pub fn effectiveness(mut self, effectiveness: f32) -> Self {
        self.state.last_effectiveness = effectiveness;
        self
    }

    pub fn build(self) -> SemioticState {
        self.state
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shared_state_read_write() {
        let state = SharedSemioticState::new();

        // Read default state
        let read = state.read();
        assert_eq!(read.context, ContextState::Neutral);
        assert!(!read.deception_detected);

        // Update state
        let new_state = SemioticStateBuilder::new()
            .deception(0.8)
            .emergence(0.3)
            .context(ContextState::Alarm, 0.9)
            .build();
        state.update(new_state);

        // Read updated state
        let read = state.read();
        assert!(read.deception_detected);
        assert_eq!(read.context, ContextState::Alarm);
        assert_eq!(read.update_count, 1);
    }

    #[test]
    fn test_should_respond() {
        let mut state = SemioticState::default();
        assert!(state.should_respond());

        // High deception = don't respond
        state.scores.deception = 0.9;
        state.deception_detected = true;
        assert!(!state.should_respond());

        // Silence context = don't respond
        state.scores.deception = 0.0;
        state.context = ContextState::Silence;
        assert!(!state.should_respond());
    }

    #[test]
    fn test_should_echo() {
        let mut state = SemioticState::default();
        assert!(state.should_echo());

        // Deception detected = don't echo
        state.deception_detected = true;
        assert!(!state.should_echo());

        // Emergence = echo for observation
        state.deception_detected = false;
        state.emergence_detected = true;
        assert!(state.should_echo());
    }

    #[test]
    fn test_response_label_modification() {
        let state = SemioticStateBuilder::new()
            .response_modification(ResponseModification::DeceptionAcknowledge)
            .build();

        // Tsik alarm -> Phee calming
        assert_eq!(state.get_response_label("Tsik", "Tsik"), "Phee");

        // Other calls unchanged
        assert_eq!(state.get_response_label("Phee", "Phee"), "Phee");
    }

    #[test]
    fn test_staleness_detection() {
        let state = SemioticState::default();
        assert!(!state.is_stale(Duration::from_millis(100)));

        // Sleep a bit and check staleness
        std::thread::sleep(Duration::from_millis(50));
        assert!(state.is_stale(Duration::from_millis(10)));
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::thread;

        let state = SharedSemioticState::new();
        let read_count = Arc::new(AtomicU32::new(0));
        let write_count = Arc::new(AtomicU32::new(0));

        let state_clone = state.clone();
        let read_count_clone = read_count.clone();
        let reader = thread::spawn(move || {
            for _ in 0..1000 {
                let _ = state_clone.read();
                read_count_clone.fetch_add(1, Ordering::SeqCst);
            }
        });

        let state_clone = state.clone();
        let write_count_clone = write_count.clone();
        let writer = thread::spawn(move || {
            for i in 0..100 {
                let new_state = SemioticStateBuilder::new()
                    .deception(i as f32 / 100.0)
                    .build();
                state_clone.update(new_state);
                write_count_clone.fetch_add(1, Ordering::SeqCst);
                thread::sleep(Duration::from_micros(100));
            }
        });

        reader.join().unwrap();
        writer.join().unwrap();

        assert_eq!(read_count.load(Ordering::SeqCst), 1000);
        assert_eq!(write_count.load(Ordering::SeqCst), 100);
    }

    #[test]
    fn test_partial_updates() {
        let state = SharedSemioticState::new();

        // Update only scores
        state.update_scores(SemioticScores {
            deception: 0.8,
            emergence: 0.2,
            directed: 0.5,
            confidence: 0.9,
        });

        let read = state.read();
        assert_eq!(read.scores.deception, 0.8);
        assert!(read.deception_detected);
        assert_eq!(read.update_count, 1);

        // Update only context
        state.update_context(ContextState::Alarm, 0.85, ContextState::Contact);

        let read = state.read();
        assert_eq!(read.context, ContextState::Alarm);
        assert_eq!(read.context_confidence, 0.85);
        assert_eq!(read.update_count, 2);

        // Scores should be preserved
        assert_eq!(read.scores.deception, 0.8);
    }
}
