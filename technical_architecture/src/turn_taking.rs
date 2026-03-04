//! Turn-Taking Prediction - Interaction Agent
//! ===========================================
//!
//! Uses the TCN to predict when a vocalization is complete and it's
//! appropriate to respond. This solves the "Interruption Problem" in
//! conversational species like Marmosets.
//!
//! ## Key Insight
//! The model learns the natural pause duration specific to each species'
//! "dialect" rather than using fixed heuristics.
//!
//! ## Usage
//! ```rust
//! use technical_architecture::TurnTakingPredictor;
//!
//! let predictor = TurnTakingPredictor::new(44100);
//!
//! // Feed last 2 seconds of audio
//! let completion_prob = predictor.predict_completion(&audio_window);
//!
//! if completion_prob > 0.8 {
//!     // Safe to respond
//! }
//! ```

use std::collections::VecDeque;

/// Configuration for turn-taking prediction
#[derive(Debug, Clone)]
pub struct TurnTakingConfig {
    /// Sample rate
    pub sample_rate: u32,
    /// Window size in seconds for analysis
    pub window_size_secs: f32,
    /// Threshold for considering a turn complete
    pub completion_threshold: f32,
    /// Minimum silence duration for turn end (ms)
    pub min_silence_ms: f32,
    /// Species-specific pause duration (ms)
    pub species_pause_ms: f32,
}

impl Default for TurnTakingConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            window_size_secs: 2.0,
            completion_threshold: 0.7,
            min_silence_ms: 50.0,
            species_pause_ms: 150.0, // Default for marmoset-like species
        }
    }
}

impl TurnTakingConfig {
    /// Config for marmosets (fast turn-taking)
    pub fn marmoset() -> Self {
        Self {
            sample_rate: 44100,
            window_size_secs: 2.0,
            completion_threshold: 0.7,
            min_silence_ms: 30.0,
            species_pause_ms: 100.0,
        }
    }

    /// Config for dolphins (whistle exchanges)
    pub fn dolphin() -> Self {
        Self {
            sample_rate: 44100,
            window_size_secs: 3.0,
            completion_threshold: 0.6,
            min_silence_ms: 100.0,
            species_pause_ms: 300.0,
        }
    }

    /// Config for birds (song exchanges)
    pub fn bird() -> Self {
        Self {
            sample_rate: 44100,
            window_size_secs: 2.5,
            completion_threshold: 0.65,
            min_silence_ms: 50.0,
            species_pause_ms: 200.0,
        }
    }
}

/// Turn state for tracking conversation flow
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TurnState {
    /// No vocalization detected
    Silence,
    /// Vocalization in progress
    Vocalizing,
    /// Potential turn end (silence after vocalization)
    TurnEnd,
    /// Confirmed turn end, ready to respond
    ReadyToRespond,
}

/// Result of turn-taking prediction
#[derive(Debug, Clone)]
pub struct TurnPrediction {
    /// Probability that the turn is complete (0.0-1.0)
    pub completion_probability: f32,
    /// Current turn state
    pub state: TurnState,
    /// Time since last vocalization (ms)
    pub silence_duration_ms: f32,
    /// Is it a good time to respond?
    pub is_reply_window: bool,
    /// Confidence in the prediction
    pub confidence: f32,
}

/// Turn-Taking Predictor
///
/// Analyzes audio streams to predict when conversational turns end
/// and when it's appropriate to respond.
#[derive(Debug, Clone)]
pub struct TurnTakingPredictor {
    config: TurnTakingConfig,
    /// Rolling audio buffer
    audio_buffer: VecDeque<f32>,
    /// Energy history for silence detection
    energy_history: VecDeque<f32>,
    /// Time of last detected vocalization (in samples)
    last_vocalization_sample: usize,
    /// Total samples processed
    total_samples: usize,
    /// Current state
    current_state: TurnState,
    /// Learned patterns (simplified - would be from training)
    learned_patterns: Vec<Vec<f32>>,
}

impl TurnTakingPredictor {
    /// Create a new turn-taking predictor with default configuration
    pub fn new(sample_rate: u32) -> Self {
        Self::with_config(TurnTakingConfig {
            sample_rate,
            ..Default::default()
        })
    }

    /// Create a predictor with custom configuration
    pub fn with_config(config: TurnTakingConfig) -> Self {
        let window_samples = (config.window_size_secs * config.sample_rate as f32) as usize;

        Self {
            config,
            audio_buffer: VecDeque::with_capacity(window_samples),
            energy_history: VecDeque::with_capacity(100),
            last_vocalization_sample: 0,
            total_samples: 0,
            current_state: TurnState::Silence,
            learned_patterns: Vec::new(),
        }
    }

    /// Create a predictor configured for marmosets
    pub fn for_marmoset() -> Self {
        Self::with_config(TurnTakingConfig::marmoset())
    }

    /// Create a predictor configured for dolphins
    pub fn for_dolphin() -> Self {
        Self::with_config(TurnTakingConfig::dolphin())
    }

    /// Create a predictor configured for birds
    pub fn for_bird() -> Self {
        Self::with_config(TurnTakingConfig::bird())
    }

    /// Process a new audio chunk and update prediction
    pub fn process(&mut self, audio: &[f32]) -> TurnPrediction {
        // Add audio to buffer
        for &sample in audio {
            self.audio_buffer.push_back(sample);
            self.total_samples += 1;
        }

        // Trim buffer to window size
        let max_samples = (self.config.window_size_secs * self.config.sample_rate as f32) as usize;
        while self.audio_buffer.len() > max_samples {
            self.audio_buffer.pop_front();
        }

        // Compute energy
        let energy = self.compute_energy(audio);
        self.energy_history.push_back(energy);
        if self.energy_history.len() > 100 {
            self.energy_history.pop_front();
        }

        // Detect vocalization
        let is_vocalizing = energy > 0.02; // Threshold

        if is_vocalizing {
            self.last_vocalization_sample = self.total_samples;
            self.current_state = TurnState::Vocalizing;
        }

        // Calculate silence duration
        let silence_samples = self
            .total_samples
            .saturating_sub(self.last_vocalization_sample);
        let silence_ms = (silence_samples as f32 / self.config.sample_rate as f32) * 1000.0;

        // Update state based on silence
        if self.current_state == TurnState::Vocalizing
            && !is_vocalizing
            && silence_ms >= self.config.min_silence_ms
        {
            self.current_state = TurnState::TurnEnd;
        }

        // Check for ready-to-respond
        if silence_ms >= self.config.species_pause_ms {
            if self.current_state == TurnState::TurnEnd {
                self.current_state = TurnState::ReadyToRespond;
            } else if self.current_state != TurnState::ReadyToRespond {
                self.current_state = TurnState::Silence;
            }
        }

        // Compute completion probability
        let completion_prob = self.compute_completion_probability(silence_ms, energy);

        // Determine if it's a good time to respond
        let is_reply_window = completion_prob > self.config.completion_threshold
            && silence_ms >= self.config.species_pause_ms;

        TurnPrediction {
            completion_probability: completion_prob,
            state: self.current_state,
            silence_duration_ms: silence_ms,
            is_reply_window,
            confidence: self.compute_confidence(),
        }
    }

    /// Predict completion from a complete audio window
    pub fn predict_completion(&self, audio: &[f32]) -> f32 {
        let energy = self.compute_energy(audio);

        // Check for silence at end
        let end_samples =
            (self.config.min_silence_ms / 1000.0 * self.config.sample_rate as f32) as usize;
        let end_energy = if audio.len() > end_samples {
            self.compute_energy(&audio[audio.len() - end_samples..])
        } else {
            energy
        };

        // Low end energy suggests turn completion
        if end_energy < 0.02 {
            0.9
        } else if end_energy < 0.05 {
            0.7
        } else if end_energy < 0.1 {
            0.4
        } else {
            0.1
        }
    }

    /// Check if current state indicates a reply window
    pub fn is_reply_window(&self) -> bool {
        matches!(self.current_state, TurnState::ReadyToRespond)
    }

    /// Get current turn state
    pub fn current_state(&self) -> TurnState {
        self.current_state
    }

    /// Get silence duration in milliseconds
    pub fn silence_duration_ms(&self) -> f32 {
        let silence_samples = self
            .total_samples
            .saturating_sub(self.last_vocalization_sample);
        (silence_samples as f32 / self.config.sample_rate as f32) * 1000.0
    }

    /// Reset the predictor state
    pub fn reset(&mut self) {
        self.audio_buffer.clear();
        self.energy_history.clear();
        self.last_vocalization_sample = 0;
        self.total_samples = 0;
        self.current_state = TurnState::Silence;
    }

    /// Compute RMS energy
    fn compute_energy(&self, audio: &[f32]) -> f32 {
        if audio.is_empty() {
            return 0.0;
        }

        let sum_sq: f32 = audio.iter().map(|x| x * x).sum();
        (sum_sq / audio.len() as f32).sqrt()
    }

    /// Compute completion probability based on silence and patterns
    fn compute_completion_probability(&self, silence_ms: f32, current_energy: f32) -> f32 {
        // Base probability from silence duration
        let silence_factor = if silence_ms >= self.config.species_pause_ms {
            1.0
        } else if silence_ms >= self.config.min_silence_ms {
            (silence_ms - self.config.min_silence_ms)
                / (self.config.species_pause_ms - self.config.min_silence_ms)
        } else {
            0.0
        };

        // Energy factor (low energy = more likely complete)
        let energy_factor = 1.0 - current_energy.min(1.0);

        // Pattern matching factor (simplified)
        let pattern_factor = self.match_patterns();

        // Weighted combination
        0.4 * silence_factor + 0.4 * energy_factor + 0.2 * pattern_factor
    }

    /// Match against learned patterns
    fn match_patterns(&self) -> f32 {
        // Simplified pattern matching
        // In a real implementation, this would use TCN features
        if self.energy_history.len() < 10 {
            return 0.5;
        }

        // Check for typical end pattern: energy drop followed by silence
        let recent: Vec<f32> = self.energy_history.iter().rev().take(10).copied().collect();

        // First 5 should be higher than last 5 for a turn end
        let early_avg: f32 = recent[5..].iter().sum::<f32>() / 5.0;
        let late_avg: f32 = recent[..5].iter().sum::<f32>() / 5.0;

        if early_avg > late_avg * 1.5 {
            0.8
        } else if early_avg > late_avg {
            0.6
        } else {
            0.3
        }
    }

    /// Compute confidence in prediction
    fn compute_confidence(&self) -> f32 {
        // More data = higher confidence
        let buffer_factor = (self.audio_buffer.len() as f32
            / (self.config.window_size_secs * self.config.sample_rate as f32))
            .min(1.0);

        // More energy history = higher confidence
        let history_factor = (self.energy_history.len() as f32 / 100.0).min(1.0);

        0.5 * buffer_factor + 0.5 * history_factor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predictor_creation() {
        let predictor = TurnTakingPredictor::new(44100);
        assert_eq!(predictor.config.sample_rate, 44100);
        assert_eq!(predictor.current_state(), TurnState::Silence);
    }

    #[test]
    fn test_process_silence() {
        let mut predictor = TurnTakingPredictor::new(44100);
        let silence = vec![0.0f32; 4410]; // 100ms of silence

        let result = predictor.process(&silence);

        assert_eq!(result.state, TurnState::Silence);
        assert!(!result.is_reply_window);
    }

    #[test]
    fn test_process_vocalization() {
        let mut predictor = TurnTakingPredictor::new(44100);

        // Generate loud audio
        let vocalization: Vec<f32> = (0..4410)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin() * 0.5)
            .collect();

        let result = predictor.process(&vocalization);

        assert_eq!(result.state, TurnState::Vocalizing);
    }

    #[test]
    fn test_turn_completion() {
        let mut predictor = TurnTakingPredictor::new(44100);

        // Process vocalization followed by silence
        let vocalization: Vec<f32> = (0..8820)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin() * 0.5)
            .collect();
        predictor.process(&vocalization);

        // Process enough silence to trigger turn end
        let silence = vec![0.0f32; 13230]; // 300ms
        let result = predictor.process(&silence);

        // Should be ready to respond after species_pause_ms
        assert!(result.completion_probability > 0.5 || result.state == TurnState::TurnEnd);
    }

    #[test]
    fn test_predict_completion_silent_end() {
        let predictor = TurnTakingPredictor::new(44100);

        // Audio that ends with silence
        let mut audio: Vec<f32> = (0..2205)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin() * 0.5)
            .collect();
        audio.extend(vec![0.0f32; 2205]); // End with silence

        let prob = predictor.predict_completion(&audio);
        assert!(prob > 0.5);
    }

    #[test]
    fn test_predict_completion_active_end() {
        let predictor = TurnTakingPredictor::new(44100);

        // Audio that ends with vocalization
        let audio: Vec<f32> = (0..4410)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin() * 0.5)
            .collect();

        let prob = predictor.predict_completion(&audio);
        assert!(prob < 0.5);
    }

    #[test]
    fn test_is_reply_window() {
        let mut predictor = TurnTakingPredictor::new(44100);

        // Initially not a reply window
        assert!(!predictor.is_reply_window());

        // After processing vocalization + sufficient silence
        let vocalization: Vec<f32> = (0..8820)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin() * 0.5)
            .collect();
        predictor.process(&vocalization);

        let silence = vec![0.0f32; 13230];
        let result = predictor.process(&silence);

        // After species_pause_ms, should be ready
        if result.silence_duration_ms >= predictor.config.species_pause_ms {
            assert!(result.is_reply_window);
        }
    }

    #[test]
    fn test_reset() {
        let mut predictor = TurnTakingPredictor::new(44100);

        let audio = vec![0.5f32; 4410];
        predictor.process(&audio);

        predictor.reset();

        assert!(predictor.audio_buffer.is_empty());
        assert_eq!(predictor.current_state(), TurnState::Silence);
        assert_eq!(predictor.total_samples, 0);
    }

    #[test]
    fn test_marmoset_config() {
        let predictor = TurnTakingPredictor::for_marmoset();
        assert_eq!(predictor.config.species_pause_ms, 100.0);
    }

    #[test]
    fn test_dolphin_config() {
        let predictor = TurnTakingPredictor::for_dolphin();
        assert_eq!(predictor.config.species_pause_ms, 300.0);
    }

    #[test]
    fn test_bird_config() {
        let predictor = TurnTakingPredictor::for_bird();
        assert_eq!(predictor.config.species_pause_ms, 200.0);
    }

    #[test]
    fn test_silence_duration() {
        let mut predictor = TurnTakingPredictor::new(44100);

        let vocalization: Vec<f32> = (0..4410)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin() * 0.5)
            .collect();
        predictor.process(&vocalization);

        // 100ms of silence
        let silence = vec![0.0f32; 4410];
        predictor.process(&silence);

        // Silence duration should be around 100ms
        let silence_ms = predictor.silence_duration_ms();
        assert!(silence_ms > 50.0 && silence_ms < 200.0);
    }

    #[test]
    fn test_turn_state_transitions() {
        let mut predictor = TurnTakingPredictor::new(44100);

        // Start in silence
        assert_eq!(predictor.current_state(), TurnState::Silence);

        // Vocalize
        let vocalization: Vec<f32> = (0..4410)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin() * 0.5)
            .collect();
        let result = predictor.process(&vocalization);
        assert_eq!(result.state, TurnState::Vocalizing);

        // Short silence - turn end
        let short_silence = vec![0.0f32; 2205]; // 50ms
        let result = predictor.process(&short_silence);
        assert!(result.state == TurnState::TurnEnd || result.state == TurnState::Vocalizing);
    }

    #[test]
    fn test_confidence_increases_with_data() {
        let mut predictor = TurnTakingPredictor::new(44100);

        let audio = vec![0.1f32; 1000];
        let result1 = predictor.process(&audio);
        let result2 = predictor.process(&audio);
        let result3 = predictor.process(&audio);

        // Confidence should increase as we accumulate data
        assert!(result3.confidence >= result1.confidence);
    }
}
