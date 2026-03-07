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

use log::{info, warn};

// =============================================================================
// Semiotic State Types (mirrors Python)
// =============================================================================

/// Response modification based on semiotic analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResponseModification {
    /// Standard response
    #[default]
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

/// Context states from probabilistic context machine
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ContextState {
    Silence,
    Contact,
    Alarm,
    Food,
    #[default]
    Neutral,
    Uncertain,
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

/// Visual attention target for multi-modal fusion
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GazeTarget {
    /// Looking at the speaker/device
    Speaker,
    /// Looking at another animal of same species
    Conspecific,
    /// Looking at food source
    Food,
    /// Looking away or unknown
    #[default]
    Unknown,
}

/// Multi-modal visual attention data
#[derive(Debug, Clone)]
pub struct VisualAttention {
    /// Visual attention level (0.0-1.0)
    /// 0.0 = no attention, 1.0 = high attention
    pub level: f32,

    /// What the animal is looking at
    pub gaze_target: GazeTarget,

    /// Confidence in gaze detection (0.0-1.0)
    pub gaze_confidence: f32,

    /// Movement intensity (0.0-1.0)
    pub movement_intensity: f32,

    /// Whether face is detected
    pub face_detected: bool,

    /// Timestamp of visual data
    pub timestamp: Instant,
}

impl Default for VisualAttention {
    fn default() -> Self {
        Self {
            level: 0.0,
            gaze_target: GazeTarget::Unknown,
            gaze_confidence: 0.0,
            movement_intensity: 0.0,
            face_detected: false,
            timestamp: Instant::now(),
        }
    }
}

impl VisualAttention {
    /// Check if animal is looking at the speaker with high attention
    pub fn is_looking_at_speaker(&self) -> bool {
        self.gaze_target == GazeTarget::Speaker && self.gaze_confidence > 0.5
    }

    /// Get attention boost factor for response loudness
    /// Returns 0.0-0.2 boost based on attention level
    pub fn get_loudness_boost(&self) -> f32 {
        if self.is_looking_at_speaker() {
            // 20% boost when looking at speaker with high attention
            0.2 * self.level
        } else {
            0.0
        }
    }

    /// Get priority boost for directed communication
    /// Higher priority when animal is looking at speaker
    pub fn get_priority_boost(&self) -> f32 {
        if self.gaze_target == GazeTarget::Speaker {
            0.3 * self.gaze_confidence
        } else {
            0.0
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

    /// Multi-modal visual attention data
    pub visual_attention: VisualAttention,

    /// Combined directed score (semiotic + visual)
    /// Higher when animal is looking at speaker during contact call
    pub combined_directed_score: f32,

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
            visual_attention: VisualAttention::default(),
            combined_directed_score: 0.0,
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

    /// Compute combined directed score from semiotic + visual attention
    ///
    /// This is the key multi-modal fusion method:
    /// - If animal is looking at speaker during contact call: boosted directed score
    /// - If animal is looking elsewhere: use semiotic score only
    pub fn compute_combined_directed_score(&mut self) {
        let semiotic_directed = self.scores.directed;
        let visual_boost = self.visual_attention.get_priority_boost();

        // Combine with weighted average
        // Visual attention adds 30% weight when looking at speaker
        if self.visual_attention.is_looking_at_speaker() {
            self.combined_directed_score = (semiotic_directed * 0.7) + (visual_boost + 0.7).min(1.0) * 0.3;
        } else {
            self.combined_directed_score = semiotic_directed;
        }
    }

    /// Get loudness delta for multi-modal response
    ///
    /// Returns additional loudness boost when:
    /// - Animal is looking at speaker (ensure it's heard)
    /// - Contact call context with directed communication
    pub fn get_multimodal_loudness_boost(&self) -> f32 {
        let mut boost = 0.0;

        // Visual attention boost (up to 20%)
        boost += self.visual_attention.get_loudness_boost();

        // Directed communication boost (up to 10%)
        if self.combined_directed_score > 0.7 && self.context == ContextState::Contact {
            boost += 0.1;
        }

        boost
    }

    /// Check if this is a high-priority directed communication
    ///
    /// True when:
    /// - Animal is looking at speaker
    /// - Combined directed score is high
    /// - Context is contact call
    pub fn is_high_priority_directed(&self) -> bool {
        self.visual_attention.is_looking_at_speaker()
            && self.combined_directed_score > 0.7
            && self.context == ContextState::Contact
    }

    /// Get response priority level (0-3)
    ///
    /// 0 = Normal
    /// 1 = Elevated (directed communication)
    /// 2 = High (looking at speaker + contact call)
    /// 3 = Critical (high priority directed + high attention)
    pub fn get_response_priority(&self) -> u8 {
        if self.is_high_priority_directed() && self.visual_attention.level > 0.8 {
            return 3;
        }
        if self.is_high_priority_directed() {
            return 2;
        }
        if self.combined_directed_score > 0.5 {
            return 1;
        }
        0
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

    /// Update visual attention (partial update for multi-modal fusion)
    ///
    /// Called by Python when new visual data is available.
    /// Also recomputes combined directed score.
    pub fn update_visual_attention(&self, visual: VisualAttention) {
        if let Ok(mut state) = self.inner.write() {
            state.visual_attention = visual;
            state.compute_combined_directed_score();
            state.last_update = Instant::now();
            state.update_count += 1;
        }
    }

    /// Reset state to default (for crash recovery)
    ///
    /// Called by Python when it restarts after a crash to clear stale state
    /// and establish a clean baseline for re-synchronization.
    pub fn reset(&self) {
        if let Ok(mut state) = self.inner.write() {
            *state = SemioticState::default();
            info!("SemioticState reset for crash recovery");
        }
    }

    /// Sync state from Python (for warm restart)
    ///
    /// After Python restarts, it can push its recovered state to Rust
    /// without requiring a full system reboot.
    pub fn sync_from_python(&self, new_state: SemioticState) {
        if let Ok(mut state) = self.inner.write() {
            *state = new_state;
            state.last_update = Instant::now();
            state.update_count += 1;
            info!("SemioticState synced from Python (update #{})", state.update_count);
        }
    }

    /// Mark state as stale (for graceful degradation)
    ///
    /// Called when Python heartbeat is lost, indicating the cognitive
    /// layer may be unresponsive. Rust can continue operating with
    /// degraded capabilities.
    pub fn mark_stale(&self) {
        if let Ok(mut state) = self.inner.write() {
            // Reduce confidence in all scores
            state.scores.confidence *= 0.5;
            state.context_confidence *= 0.5;
            state.visual_attention.gaze_confidence *= 0.5;
            warn!("SemioticState marked stale - Python heartbeat lost");
        }
    }

    /// Check if state is healthy (not stale, reasonable confidence)
    pub fn is_healthy(&self) -> bool {
        match self.inner.read() {
            Ok(state) => {
                !state.is_stale(Duration::from_secs(5)) && state.scores.confidence > 0.3 && state.update_count > 0
            }
            Err(_) => false,
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

    /// Set visual attention level (0.0-1.0)
    pub fn visual_attention_level(mut self, level: f32) -> Self {
        self.state.visual_attention.level = level;
        self
    }

    /// Set gaze target
    pub fn gaze_target(mut self, target: GazeTarget) -> Self {
        self.state.visual_attention.gaze_target = target;
        self
    }

    /// Set gaze confidence (0.0-1.0)
    pub fn gaze_confidence(mut self, confidence: f32) -> Self {
        self.state.visual_attention.gaze_confidence = confidence;
        self
    }

    /// Set full visual attention
    pub fn visual_attention(mut self, visual: VisualAttention) -> Self {
        self.state.visual_attention = visual;
        self
    }

    pub fn build(self) -> SemioticState {
        // Compute combined directed score before returning
        let mut state = self.state;
        state.compute_combined_directed_score();
        state
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
                let new_state = SemioticStateBuilder::new().deception(i as f32 / 100.0).build();
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

    #[test]
    fn test_visual_attention_looking_at_speaker() {
        let visual = VisualAttention {
            level: 0.8,
            gaze_target: GazeTarget::Speaker,
            gaze_confidence: 0.9,
            ..Default::default()
        };

        assert!(visual.is_looking_at_speaker());
        assert!((visual.get_loudness_boost() - 0.16).abs() < 0.01); // 0.2 * 0.8
        assert!((visual.get_priority_boost() - 0.27).abs() < 0.01); // 0.3 * 0.9
    }

    #[test]
    fn test_visual_attention_not_looking_at_speaker() {
        let visual = VisualAttention {
            level: 0.8,
            gaze_target: GazeTarget::Conspecific,
            gaze_confidence: 0.9,
            ..Default::default()
        };

        assert!(!visual.is_looking_at_speaker());
        assert_eq!(visual.get_loudness_boost(), 0.0);
        assert_eq!(visual.get_priority_boost(), 0.0);
    }

    #[test]
    fn test_multimodal_directed_score_boost() {
        // Without visual attention
        let state = SemioticStateBuilder::new()
            .directed(0.7)
            .context(ContextState::Contact, 0.8)
            .build();
        let baseline_score = state.combined_directed_score;

        // With visual attention (looking at speaker)
        let state = SemioticStateBuilder::new()
            .directed(0.7)
            .context(ContextState::Contact, 0.8)
            .visual_attention_level(0.9)
            .gaze_target(GazeTarget::Speaker)
            .gaze_confidence(0.9)
            .build();

        // Combined score should be higher with visual attention
        assert!(state.combined_directed_score > baseline_score);
        assert!(state.is_high_priority_directed());
    }

    #[test]
    fn test_multimodal_loudness_boost() {
        // Contact call with high visual attention
        let state = SemioticStateBuilder::new()
            .directed(0.8)
            .context(ContextState::Contact, 0.9)
            .visual_attention_level(0.9)
            .gaze_target(GazeTarget::Speaker)
            .gaze_confidence(0.9)
            .build();

        // Should get loudness boost
        let boost = state.get_multimodal_loudness_boost();
        assert!(boost > 0.15); // At least 15% boost
    }

    #[test]
    fn test_response_priority_levels() {
        // Level 0: Normal
        let state = SemioticState::default();
        assert_eq!(state.get_response_priority(), 0);

        // Level 1: Directed communication
        let state = SemioticStateBuilder::new().directed(0.7).build();
        assert_eq!(state.get_response_priority(), 1);

        // Level 2: Looking at speaker + contact call
        let state = SemioticStateBuilder::new()
            .directed(0.8)
            .context(ContextState::Contact, 0.9)
            .visual_attention_level(0.8)
            .gaze_target(GazeTarget::Speaker)
            .gaze_confidence(0.9)
            .build();
        assert_eq!(state.get_response_priority(), 2);

        // Level 3: Critical - high priority + very high attention
        let state = SemioticStateBuilder::new()
            .directed(0.9)
            .context(ContextState::Contact, 0.95)
            .visual_attention_level(0.95)
            .gaze_target(GazeTarget::Speaker)
            .gaze_confidence(0.95)
            .build();
        assert_eq!(state.get_response_priority(), 3);
    }

    #[test]
    fn test_visual_attention_update() {
        let state = SharedSemioticState::new();

        // Update visual attention
        let visual = VisualAttention {
            level: 0.9,
            gaze_target: GazeTarget::Speaker,
            gaze_confidence: 0.85,
            ..Default::default()
        };
        state.update_visual_attention(visual);

        let read = state.read();
        assert_eq!(read.visual_attention.level, 0.9);
        assert_eq!(read.visual_attention.gaze_target, GazeTarget::Speaker);
        assert!(read.visual_attention.is_looking_at_speaker());
        assert!(read.combined_directed_score > 0.0); // Should have computed combined score
    }

    #[test]
    fn test_crash_recovery_reset() {
        let state = SharedSemioticState::new();

        // Set up complex state
        let complex_state = SemioticStateBuilder::new()
            .deception(0.8)
            .emergence(0.3)
            .directed(0.9)
            .context(ContextState::Alarm, 0.95)
            .visual_attention_level(0.8)
            .gaze_target(GazeTarget::Speaker)
            .build();
        state.update(complex_state);

        // Verify state is set
        let before = state.read();
        assert_eq!(before.context, ContextState::Alarm);
        assert_eq!(before.update_count, 1);

        // Simulate crash - reset state
        state.reset();

        // Verify state is now default
        let after = state.read();
        assert_eq!(after.context, ContextState::Neutral);
        assert_eq!(after.scores.deception, 0.0);
        assert_eq!(after.visual_attention.level, 0.0);
        assert_eq!(after.update_count, 0); // Reset clears update count
    }

    #[test]
    fn test_sync_from_python_after_restart() {
        let state = SharedSemioticState::new();

        // Initial state
        let initial = state.read();
        assert_eq!(initial.update_count, 0);

        // Python restarts and syncs recovered state
        let recovered_state = SemioticStateBuilder::new()
            .deception(0.5)
            .directed(0.8)
            .context(ContextState::Contact, 0.9)
            .effectiveness(0.75)
            .build();

        state.sync_from_python(recovered_state);

        // Verify sync worked
        let synced = state.read();
        assert_eq!(synced.context, ContextState::Contact);
        assert_eq!(synced.scores.deception, 0.5);
        assert_eq!(synced.scores.directed, 0.8);
        assert_eq!(synced.last_effectiveness, 0.75);
        assert_eq!(synced.update_count, 1);
    }

    #[test]
    fn test_mark_stale_graceful_degradation() {
        let state = SharedSemioticState::new();

        // Set up high confidence state
        let high_confidence = SemioticStateBuilder::new()
            .directed(0.9)
            .confidence(0.9)
            .context(ContextState::Contact, 0.9)
            .gaze_confidence(0.9)
            .build();
        state.update(high_confidence);

        // Verify high confidence
        let before = state.read();
        assert!((before.scores.confidence - 0.9).abs() < 0.01);
        assert!((before.context_confidence - 0.9).abs() < 0.01);

        // Python heartbeat lost - mark stale
        state.mark_stale();

        // Verify confidence reduced (graceful degradation)
        let after = state.read();
        assert!((after.scores.confidence - 0.45).abs() < 0.01); // 0.9 * 0.5
        assert!((after.context_confidence - 0.45).abs() < 0.01); // 0.9 * 0.5
        assert!((after.visual_attention.gaze_confidence - 0.45).abs() < 0.01);
    }

    #[test]
    fn test_is_healthy_check() {
        let state = SharedSemioticState::new();

        // Fresh state with no updates is not healthy
        assert!(!state.is_healthy());

        // Update with good state
        let good_state = SemioticStateBuilder::new().confidence(0.8).build();
        state.update(good_state);

        // Now should be healthy
        assert!(state.is_healthy());

        // Mark stale multiple times to drop confidence below 0.3
        state.mark_stale(); // 0.8 * 0.5 = 0.4
        state.mark_stale(); // 0.4 * 0.5 = 0.2

        // Low confidence = not healthy
        assert!(!state.is_healthy());
    }
}
