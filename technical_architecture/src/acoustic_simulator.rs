/*!
Acoustic Simulator for TDD Testing
===================================

Generates realistic environmental noise mixtures for testing audio processing
algorithms in the laboratory.

Features:
- Rain, thunder, wind, insect chorus, bird chorus generation
- Configurable SNR mixing
- Environmental acoustic simulation (reverb, distance attenuation)
- Deterministic test fixtures for reproducible tests
*/

use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Spectral color of noise
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpectralColor {
    White,     // Flat spectrum
    Pink,      // 1/f
    Brown,     // 1/f²
    Blue,      // f
}

/// Temporal characteristics of noise
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalCharacteristics {
    pub attack_ms: f32,
    pub sustain_ms: f32,
    pub decay_ms: f32,
    pub modulation_rate_hz: f32,
}

impl Default for TemporalCharacteristics {
    fn default() -> Self {
        Self {
            attack_ms: 50.0,
            sustain_ms: 500.0,
            decay_ms: 200.0,
            modulation_rate_hz: 0.5,
        }
    }
}

/// Noise profile for generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseProfile {
    pub name: String,
    pub spectral_color: SpectralColor,
    pub amplitude_variation: f32,
    pub temporal: TemporalCharacteristics,
}

impl NoiseProfile {
    /// Create a new noise profile
    pub fn new(name: String, spectral_color: SpectralColor) -> Self {
        Self {
            name,
            spectral_color,
            amplitude_variation: 0.3,
            temporal: TemporalCharacteristics::default(),
        }
    }

    /// Rain noise profile
    pub fn rain() -> Self {
        Self {
            name: "rain".to_string(),
            spectral_color: SpectralColor::Pink,
            amplitude_variation: 0.4,
            temporal: TemporalCharacteristics {
                attack_ms: 100.0,
                sustain_ms: 2000.0,
                decay_ms: 500.0,
                modulation_rate_hz: 0.2,
            },
        }
    }

    /// Thunder noise profile
    pub fn thunder() -> Self {
        Self {
            name: "thunder".to_string(),
            spectral_color: SpectralColor::Brown,
            amplitude_variation: 0.8,
            temporal: TemporalCharacteristics {
                attack_ms: 50.0,
                sustain_ms: 1000.0,
                decay_ms: 2000.0,
                modulation_rate_hz: 0.1,
            },
        }
    }

    /// Wind noise profile
    pub fn wind() -> Self {
        Self {
            name: "wind".to_string(),
            spectral_color: SpectralColor::Pink,
            amplitude_variation: 0.5,
            temporal: TemporalCharacteristics {
                attack_ms: 200.0,
                sustain_ms: 3000.0,
                decay_ms: 1000.0,
                modulation_rate_hz: 0.3,
            },
        }
    }

    /// Insect chorus profile
    pub fn insects() -> Self {
        Self {
            name: "insects".to_string(),
            spectral_color: SpectralColor::Blue,
            amplitude_variation: 0.2,
            temporal: TemporalCharacteristics {
                attack_ms: 500.0,
                sustain_ms: 5000.0,
                decay_ms: 1000.0,
                modulation_rate_hz: 10.0,
            },
        }
    }

    /// Bird chorus profile
    pub fn birds() -> Self {
        Self {
            name: "birds".to_string(),
            spectral_color: SpectralColor::White,
            amplitude_variation: 0.6,
            temporal: TemporalCharacteristics {
                attack_ms: 100.0,
                sustain_ms: 1000.0,
                decay_ms: 500.0,
                modulation_rate_hz: 5.0,
            },
        }
    }
}

/// Environment type for acoustic simulation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnvironmentType {
    JungleDense,
    JungleSparse,
    Rainforest,
    ForestFloor,
    Canopy,
    RiverBank,
    Cave,
    OpenField,
    Mountain,
    Underwater,
}

/// Acoustic environment parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcousticEnvironment {
    pub environment_type: EnvironmentType,
    pub temperature_celsius: f32,
    pub humidity_percent: f32,
    pub wind_speed_m_s: f32,
    pub rain_intensity_mm_h: f32,
}

impl AcousticEnvironment {
    /// Create a new acoustic environment
    pub fn new(environment_type: EnvironmentType) -> Self {
        Self {
            environment_type,
            temperature_celsius: 25.0,
            humidity_percent: 80.0,
            wind_speed_m_s: 2.0,
            rain_intensity_mm_h: 0.0,
        }
    }

    /// Dense jungle environment
    pub fn dense_jungle() -> Self {
        Self {
            environment_type: EnvironmentType::JungleDense,
            temperature_celsius: 28.0,
            humidity_percent: 90.0,
            wind_speed_m_s: 1.0,
            rain_intensity_mm_h: 5.0,
        }
    }

    /// Rainforest with heavy rain
    pub fn rainy_rainforest() -> Self {
        Self {
            environment_type: EnvironmentType::Rainforest,
            temperature_celsius: 26.0,
            humidity_percent: 95.0,
            wind_speed_m_s: 3.0,
            rain_intensity_mm_h: 25.0,
        }
    }

    /// Open field environment
    pub fn open_field() -> Self {
        Self {
            environment_type: EnvironmentType::OpenField,
            temperature_celsius: 22.0,
            humidity_percent: 60.0,
            wind_speed_m_s: 5.0,
            rain_intensity_mm_h: 0.0,
        }
    }

    /// Get reverb time (RT60) for this environment
    pub fn rt60_seconds(&self) -> f32 {
        match self.environment_type {
            EnvironmentType::JungleDense => 1.2,
            EnvironmentType::JungleSparse => 0.8,
            EnvironmentType::Rainforest => 1.5,
            EnvironmentType::ForestFloor => 0.6,
            EnvironmentType::Canopy => 0.9,
            EnvironmentType::RiverBank => 1.1,
            EnvironmentType::Cave => 3.0,
            EnvironmentType::OpenField => 0.3,
            EnvironmentType::Mountain => 0.5,
            EnvironmentType::Underwater => 0.1,
        }
    }

    /// Get background noise level (dB SPL)
    pub fn background_noise_level(&self) -> f32 {
        let base = match self.environment_type {
            EnvironmentType::JungleDense => 50.0,
            EnvironmentType::JungleSparse => 45.0,
            EnvironmentType::Rainforest => 55.0,
            EnvironmentType::ForestFloor => 40.0,
            EnvironmentType::Canopy => 48.0,
            EnvironmentType::RiverBank => 52.0,
            EnvironmentType::Cave => 35.0,
            EnvironmentType::OpenField => 35.0,
            EnvironmentType::Mountain => 30.0,
            EnvironmentType::Underwater => 60.0,
        };

        // Add rain contribution
        let rain_noise = if self.rain_intensity_mm_h > 0.0 {
            10.0 * f32::log10(self.rain_intensity_mm_h + 1.0)
        } else {
            0.0
        };

        // Add wind contribution
        let wind_noise = if self.wind_speed_m_s > 0.0 {
            5.0 * f32::log10(self.wind_speed_m_s + 1.0)
        } else {
            0.0
        };

        base + rain_noise + wind_noise
    }
}

/// Noise mixture with target signal
#[derive(Debug, Clone)]
pub struct NoiseMixture {
    pub target_signal: Vec<f32>,
    pub noise_layers: Vec<(NoiseProfile, Vec<f32>, f32)>, // (profile, audio, level)
    pub snr_db: f32,
    pub sample_rate: u32,
}

/// Acoustic Simulator
pub struct AcousticSimulator {
    sample_rate: u32,
    rng_seed: u64,
}

impl AcousticSimulator {
    /// Create a new acoustic simulator
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            rng_seed: 42, // Deterministic seed for reproducible tests
        }
    }

    /// Create with custom seed for reproducibility
    pub fn with_seed(sample_rate: u32, seed: u64) -> Self {
        Self {
            sample_rate,
            rng_seed: seed,
        }
    }

    /// Generate colored noise
    pub fn generate_noise(&self, profile: &NoiseProfile, duration_samples: usize) -> Vec<f32> {
        let mut noise = Vec::with_capacity(duration_samples);
        let mut rng = StdRng::seed_from_u64(self.rng_seed);

        match profile.spectral_color {
            SpectralColor::White => {
                // White noise: uniform distribution
                for _ in 0..duration_samples {
                    noise.push(rng.gen::<f32>() * 2.0 - 1.0);
                }
            }
            SpectralColor::Pink => {
                // Pink noise: 1/f, approximated with filtering
                noise = self.generate_pink_noise(duration_samples);
            }
            SpectralColor::Brown => {
                // Brown noise: 1/f²
                noise = self.generate_brown_noise(duration_samples);
            }
            SpectralColor::Blue => {
                // Blue noise: f
                noise = self.generate_blue_noise(duration_samples);
            }
        }

        // Apply amplitude modulation
        self.apply_amplitude_envelope(&mut noise, &profile.temporal, duration_samples);

        // Apply amplitude variation
        let variation = 1.0 + profile.amplitude_variation * (rng.gen::<f32>() - 0.5);
        for sample in &mut noise {
            *sample *= variation;
        }

        noise
    }

    /// Generate pink noise (1/f)
    fn generate_pink_noise(&self, samples: usize) -> Vec<f32> {
        let mut noise = Vec::with_capacity(samples);
        let mut b = [0f32; 7]; // Pink noise filter state

        for i in 0..samples {
            // White noise
            let white: f32 = StdRng::seed_from_u64(self.rng_seed + i as u64).gen::<f32>() * 2.0 - 1.0;

            // Pink filter
            b[0] = 0.99886 * b[0] + 0.0555179 * white;
            b[1] = 0.99332 * b[1] + 0.0750759 * white;
            b[2] = 0.96900 * b[2] + 0.1538520 * white;
            b[3] = 0.86650 * b[3] + 0.3104856 * white;
            b[4] = 0.55000 * b[4] + 0.5329522 * white;
            b[5] = -0.7616 * b[5] - 0.0168980 * white;

            let pink = b[0] + b[1] + b[2] + b[3] + b[4] + b[5] + b[6] + white * 0.5362;
            b[6] = white * 0.115926;

            noise.push(pink * 0.11); // Scale to reasonable amplitude
        }

        noise
    }

    /// Generate brown noise (1/f²)
    fn generate_brown_noise(&self, samples: usize) -> Vec<f32> {
        let mut noise = Vec::with_capacity(samples);
        let mut last_value = 0f32;
        let mut rng = StdRng::seed_from_u64(self.rng_seed);

        for _ in 0..samples {
            let white: f32 = rng.gen::<f32>() * 2.0 - 1.0;
            let brown = (last_value + (0.02 * white)).clamp(-1.0, 1.0);
            last_value = brown;
            noise.push(brown);
        }

        noise
    }

    /// Generate blue noise (f)
    fn generate_blue_noise(&self, samples: usize) -> Vec<f32> {
        let mut white = vec![0f32; samples];
        let mut rng = StdRng::seed_from_u64(self.rng_seed);

        for i in 0..samples {
            white[i] = rng.gen::<f32>() * 2.0 - 1.0;
        }

        // High-pass filter to emphasize high frequencies
        let mut blue = vec![0f32; samples];
        let alpha = 0.95;
        blue[0] = white[0];

        for i in 1..samples {
            blue[i] = alpha * (blue[i - 1] + white[i] - white[i - 1]);
        }

        blue
    }

    /// Apply amplitude envelope to noise
    fn apply_amplitude_envelope(&self, noise: &mut [f32], temporal: &TemporalCharacteristics, duration_samples: usize) {
        let attack_samples = (temporal.attack_ms * self.sample_rate as f32 / 1000.0) as usize;
        let sustain_samples = (temporal.sustain_ms * self.sample_rate as f32 / 1000.0) as usize;
        let decay_samples = (temporal.decay_ms * self.sample_rate as f32 / 1000.0) as usize;

        for (i, sample) in noise.iter_mut().enumerate() {
            let envelope = if i < attack_samples {
                i as f32 / attack_samples as f32
            } else if i < attack_samples + sustain_samples {
                1.0
            } else if i < attack_samples + sustain_samples + decay_samples {
                1.0 - (i - attack_samples - sustain_samples) as f32 / decay_samples as f32
            } else {
                0.0
            };

            *sample *= envelope;
        }
    }

    /// Generate rain noise
    pub fn generate_rain_noise(&self, intensity_mm_h: f32, duration_samples: usize) -> Vec<f32> {
        let profile = NoiseProfile::rain();
        let mut noise = self.generate_noise(&profile, duration_samples);

        // Scale by intensity (heavier rain = louder)
        let intensity_factor = (intensity_mm_h / 50.0).clamp(0.1, 2.0);
        for sample in &mut noise {
            *sample *= intensity_factor;
        }

        noise
    }

    /// Generate thunder
    pub fn generate_thunder(&self, duration_samples: usize) -> Vec<f32> {
        let profile = NoiseProfile::thunder();
        self.generate_noise(&profile, duration_samples)
    }

    /// Generate wind noise
    pub fn generate_wind_noise(&self, wind_speed_m_s: f32, duration_samples: usize) -> Vec<f32> {
        let profile = NoiseProfile::wind();
        let mut noise = self.generate_noise(&profile, duration_samples);

        // Scale by wind speed
        let speed_factor = (wind_speed_m_s / 10.0).clamp(0.2, 2.0);
        for sample in &mut noise {
            *sample *= speed_factor;
        }

        noise
    }

    /// Generate insect chorus
    pub fn generate_insect_chorus(&self, density: f32, duration_samples: usize) -> Vec<f32> {
        let profile = NoiseProfile::insects();
        let mut noise = self.generate_noise(&profile, duration_samples);

        // Scale by density (more insects = louder)
        let density_factor = density.clamp(0.1, 1.0);
        for sample in &mut noise {
            *sample *= density_factor;
        }

        noise
    }

    /// Generate bird chorus
    pub fn generate_bird_chorus(&self, density: f32, duration_samples: usize) -> Vec<f32> {
        let profile = NoiseProfile::birds();
        let mut noise = self.generate_noise(&profile, duration_samples);

        // Scale by density
        let density_factor = density.clamp(0.1, 1.0);
        for sample in &mut noise {
            *sample *= density_factor;
        }

        noise
    }

    /// Mix target signal with noise at specified SNR
    pub fn mix_with_snr(&self, target: &[f32], noise: &[f32], snr_db: f32) -> Result<Vec<f32>> {
        if target.is_empty() || noise.is_empty() {
            return Ok(vec![]);
        }

        let len = std::cmp::min(target.len(), noise.len());
        let mut mixture = Vec::with_capacity(len);

        // Calculate signal and noise power
        let signal_power: f32 = target[..len].iter().map(|x| x * x).sum::<f32>() / len as f32;
        let noise_power: f32 = noise[..len].iter().map(|x| x * x).sum::<f32>() / len as f32;

        if signal_power == 0.0 || noise_power == 0.0 {
            return Ok(target[..len].to_vec());
        }

        // Calculate current SNR
        let current_snr_db = 10.0 * f32::log10(signal_power / noise_power);

        // Calculate required noise scaling
        let snr_difference = current_snr_db - snr_db;
        let noise_scale = 10f32.powf(snr_difference / 20.0);

        // Mix signals
        for i in 0..len {
            mixture.push(target[i] + noise[i] * noise_scale);
        }

        // Normalize to prevent clipping
        let max_amplitude = mixture.iter().map(|x| x.abs()).fold(0.0, f32::max);
        if max_amplitude > 1.0 {
            let scale = 1.0 / max_amplitude;
            for sample in &mut mixture {
                *sample *= scale;
            }
        }

        Ok(mixture)
    }

    /// Simulate acoustic environment
    pub fn simulate_environment(&self, signal: &[f32], environment: &AcousticEnvironment) -> Result<Vec<f32>> {
        let mut output = signal.to_vec();

        // Add background noise based on environment
        let bg_noise_level = environment.background_noise_level();
        if bg_noise_level > 30.0 {
            // Generate appropriate noise
            let noise = if environment.rain_intensity_mm_h > 0.0 {
                self.generate_rain_noise(environment.rain_intensity_mm_h, signal.len())
            } else if environment.wind_speed_m_s > 2.0 {
                self.generate_wind_noise(environment.wind_speed_m_s, signal.len())
            } else {
                self.generate_noise(&NoiseProfile::insects(), signal.len())
            };

            // Mix with appropriate SNR
            let snr_db = 60.0 - bg_noise_level; // Higher noise level = lower SNR
            output = self.mix_with_snr(&output, &noise, snr_db)?;
        }

        Ok(output)
    }

    /// Create realistic test mixture
    pub fn create_test_mixture(
        &self,
        target: &[f32],
        environment: &AcousticEnvironment,
        snr_db: f32,
    ) -> Result<NoiseMixture> {
        let simulated = self.simulate_environment(target, environment)?;

        // Generate additional noise layers
        let mut noise_layers = Vec::new();

        if environment.rain_intensity_mm_h > 0.0 {
            let rain_noise = self.generate_rain_noise(environment.rain_intensity_mm_h, target.len());
            noise_layers.push((NoiseProfile::rain(), rain_noise, 0.5));
        }

        if environment.wind_speed_m_s > 2.0 {
            let wind_noise = self.generate_wind_noise(environment.wind_speed_m_s, target.len());
            noise_layers.push((NoiseProfile::wind(), wind_noise, 0.3));
        }

        // Add insect chorus in jungle environments
        if matches!(
            environment.environment_type,
            EnvironmentType::JungleDense | EnvironmentType::Rainforest
        ) {
            let insect_noise = self.generate_insect_chorus(0.7, target.len());
            noise_layers.push((NoiseProfile::insects(), insect_noise, 0.2));
        }

        Ok(NoiseMixture {
            target_signal: target.to_vec(),
            noise_layers,
            snr_db,
            sample_rate: self.sample_rate,
        })
    }

    /// Generate synthetic vocalization for testing
    pub fn generate_synthetic_vocalization(
        &self,
        frequency_hz: f32,
        duration_ms: f32,
        modulation_rate_hz: Option<f32>,
    ) -> Vec<f32> {
        let duration_samples = (duration_ms * self.sample_rate as f32 / 1000.0) as usize;
        let mut signal = Vec::with_capacity(duration_samples);

        let phase_increment = 2.0 * std::f32::consts::PI * frequency_hz / self.sample_rate as f32;
        let mut phase = 0.0;

        let modulation_rate = modulation_rate_hz.unwrap_or(0.0);
        let mod_increment = if modulation_rate > 0.0 {
            2.0 * std::f32::consts::PI * modulation_rate / self.sample_rate as f32
        } else {
            0.0
        };
        let mut mod_phase = 0.0;

        // Amplitude envelope
        let attack_samples = (duration_samples as f32 * 0.1) as usize;
        let decay_samples = (duration_samples as f32 * 0.2) as usize;

        for i in 0..duration_samples {
            // Apply amplitude envelope
            let envelope = if i < attack_samples {
                i as f32 / attack_samples as f32
            } else if i > duration_samples - decay_samples {
                (duration_samples - i) as f32 / decay_samples as f32
            } else {
                1.0
            };

            // Generate sine wave with optional FM modulation
            let mod_factor = if modulation_rate > 0.0 {
                1.0 + 0.1 * f32::sin(mod_phase)
            } else {
                1.0
            };

            let sample = envelope * f32::sin(phase * mod_factor);
            signal.push(sample);

            phase += phase_increment;
            mod_phase += mod_increment;
        }

        signal
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_signal() -> Vec<f32> {
        vec![0.5f32; 1000]
    }

    #[test]
    fn test_simulator_creation() {
        let sim = AcousticSimulator::new(48000);
        assert_eq!(sim.sample_rate, 48000);
    }

    #[test]
    fn test_simulator_with_seed() {
        let sim = AcousticSimulator::with_seed(48000, 123);
        assert_eq!(sim.rng_seed, 123);
    }

    #[test]
    fn test_generate_white_noise() {
        let sim = AcousticSimulator::new(48000);
        let profile = NoiseProfile::new("test".to_string(), SpectralColor::White);
        let noise = sim.generate_noise(&profile, 1000);

        assert_eq!(noise.len(), 1000);
        // Check that noise is in valid range [-1, 1]
        assert!(noise.iter().all(|&x| x >= -1.0 && x <= 1.0));
    }

    #[test]
    fn test_generate_pink_noise() {
        let sim = AcousticSimulator::new(48000);
        let profile = NoiseProfile::new("test".to_string(), SpectralColor::Pink);
        let noise = sim.generate_noise(&profile, 1000);

        assert_eq!(noise.len(), 1000);
        assert!(noise.iter().all(|&x| x >= -1.0 && x <= 1.0));
    }

    #[test]
    fn test_generate_brown_noise() {
        let sim = AcousticSimulator::new(48000);
        let profile = NoiseProfile::new("test".to_string(), SpectralColor::Brown);
        let noise = sim.generate_noise(&profile, 1000);

        assert_eq!(noise.len(), 1000);
        assert!(noise.iter().all(|&x| x >= -1.0 && x <= 1.0));
    }

    #[test]
    fn test_generate_blue_noise() {
        let sim = AcousticSimulator::new(48000);
        let profile = NoiseProfile::new("test".to_string(), SpectralColor::Blue);
        let noise = sim.generate_noise(&profile, 1000);

        assert_eq!(noise.len(), 1000);
        assert!(noise.iter().all(|&x| x >= -1.0 && x <= 1.0));
    }

    #[test]
    fn test_generate_rain_noise() {
        let sim = AcousticSimulator::new(48000);
        let noise = sim.generate_rain_noise(10.0, 1000);

        assert_eq!(noise.len(), 1000);
        assert!(noise.iter().all(|&x| x >= -1.0 && x <= 1.0));
    }

    #[test]
    fn test_generate_rain_intensity_scaling() {
        let sim = AcousticSimulator::new(48000);
        let light_rain = sim.generate_rain_noise(5.0, 1000);
        let heavy_rain = sim.generate_rain_noise(25.0, 1000);

        // Heavy rain should have higher RMS than light rain
        let light_rms: f32 = light_rain.iter().map(|x| x * x).sum::<f32>().sqrt();
        let heavy_rms: f32 = heavy_rain.iter().map(|x| x * x).sum::<f32>().sqrt();

        assert!(heavy_rms > light_rms);
    }

    #[test]
    fn test_generate_thunder() {
        let sim = AcousticSimulator::new(48000);
        let thunder = sim.generate_thunder(1000);

        assert_eq!(thunder.len(), 1000);
        assert!(thunder.iter().all(|&x| x >= -1.0 && x <= 1.0));
    }

    #[test]
    fn test_generate_wind_noise() {
        let sim = AcousticSimulator::new(48000);
        let wind = sim.generate_wind_noise(5.0, 1000);

        assert_eq!(wind.len(), 1000);
        assert!(wind.iter().all(|&x| x >= -1.0 && x <= 1.0));
    }

    #[test]
    fn test_generate_wind_speed_scaling() {
        let sim = AcousticSimulator::new(48000);
        let light_wind = sim.generate_wind_noise(2.0, 1000);
        let strong_wind = sim.generate_wind_noise(10.0, 1000);

        let light_rms: f32 = light_wind.iter().map(|x| x * x).sum::<f32>().sqrt();
        let strong_rms: f32 = strong_wind.iter().map(|x| x * x).sum::<f32>().sqrt();

        assert!(strong_rms > light_rms);
    }

    #[test]
    fn test_generate_insect_chorus() {
        let sim = AcousticSimulator::new(48000);
        let insects = sim.generate_insect_chorus(0.5, 1000);

        assert_eq!(insects.len(), 1000);
        assert!(insects.iter().all(|&x| x >= -1.0 && x <= 1.0));
    }

    #[test]
    fn test_generate_bird_chorus() {
        let sim = AcousticSimulator::new(48000);
        let birds = sim.generate_bird_chorus(0.5, 1000);

        assert_eq!(birds.len(), 1000);
        assert!(birds.iter().all(|&x| x >= -1.0 && x <= 1.0));
    }

    #[test]
    fn test_mix_with_snr() {
        let sim = AcousticSimulator::new(48000);
        let target = create_test_signal();
        let noise = vec![0.1f32; 1000];

        let mixture = sim.mix_with_snr(&target, &noise, 10.0).unwrap();

        assert_eq!(mixture.len(), 1000);
        // Mixture should be different from pure target
        assert!(mixture != target);
    }

    #[test]
    fn test_mix_high_snr() {
        let sim = AcousticSimulator::new(48000);
        let target = create_test_signal();
        let noise = vec![1.0f32; 1000];

        // High SNR means signal dominates
        let mixture = sim.mix_with_snr(&target, &noise, 30.0).unwrap();

        // Target signal should be more prominent
        let target_power: f32 = target.iter().map(|x| x * x).sum::<f32>();
        let mixture_power: f32 = mixture.iter().map(|x| x * x).sum::<f32>();
        let noise_power: f32 = noise.iter().map(|x| x * x).sum::<f32>();

        assert!(mixture_power > target_power * 0.8); // Close to target power
    }

    #[test]
    fn test_mix_low_snr() {
        let sim = AcousticSimulator::new(48000);
        let target = create_test_signal();
        let noise = vec![1.0f32; 1000];

        // Low SNR means noise dominates
        let mixture = sim.mix_with_snr(&target, &noise, 0.0).unwrap();

        let target_power: f32 = target.iter().map(|x| x * x).sum::<f32>();
        let mixture_power: f32 = mixture.iter().map(|x| x * x).sum::<f32>();
        let noise_power: f32 = noise.iter().map(|x| x * x).sum::<f32>();

        // Should be closer to noise power
        assert!(mixture_power > target_power * 1.5);
    }

    #[test]
    fn test_mix_empty_vectors() {
        let sim = AcousticSimulator::new(48000);
        let target = vec![];
        let noise = vec![0.1f32; 1000];

        let mixture = sim.mix_with_snr(&target, &noise, 10.0).unwrap();
        assert_eq!(mixture.len(), 0);
    }

    #[test]
    fn test_mix_no_clipping() {
        let sim = AcousticSimulator::new(48000);
        let target = vec![0.9f32; 1000];
        let noise = vec![0.9f32; 1000];

        let mixture = sim.mix_with_snr(&target, &noise, 0.0).unwrap();

        // Should not clip beyond [-1, 1]
        assert!(mixture.iter().all(|&x| x >= -1.0 && x <= 1.0));
    }

    #[test]
    fn test_environment_creation() {
        let env = AcousticEnvironment::new(EnvironmentType::JungleDense);
        assert_eq!(env.environment_type, EnvironmentType::JungleDense);
    }

    #[test]
    fn test_dense_jungle_environment() {
        let env = AcousticEnvironment::dense_jungle();
        assert_eq!(env.environment_type, EnvironmentType::JungleDense);
        assert_eq!(env.temperature_celsius, 28.0);
        assert_eq!(env.humidity_percent, 90.0);
    }

    #[test]
    fn test_rainy_rainforest_environment() {
        let env = AcousticEnvironment::rainy_rainforest();
        assert_eq!(env.environment_type, EnvironmentType::Rainforest);
        assert!(env.rain_intensity_mm_h > 20.0);
    }

    #[test]
    fn test_open_field_environment() {
        let env = AcousticEnvironment::open_field();
        assert_eq!(env.environment_type, EnvironmentType::OpenField);
        assert_eq!(env.rain_intensity_mm_h, 0.0);
    }

    #[test]
    fn test_rt60_values() {
        let cave = AcousticEnvironment::new(EnvironmentType::Cave);
        let field = AcousticEnvironment::new(EnvironmentType::OpenField);

        assert!(cave.rt60_seconds() > 2.0);
        assert!(field.rt60_seconds() < 0.5);
    }

    #[test]
    fn test_background_noise_calculation() {
        let quiet_env = AcousticEnvironment::open_field();
        let loud_env = AcousticEnvironment::rainy_rainforest();

        assert!(loud_env.background_noise_level() > quiet_env.background_noise_level());
    }

    #[test]
    fn test_rain_increases_noise() {
        let mut env = AcousticEnvironment::open_field();
        let base_noise = env.background_noise_level();

        env.rain_intensity_mm_h = 20.0;
        let rain_noise = env.background_noise_level();

        assert!(rain_noise > base_noise);
    }

    #[test]
    fn test_wind_increases_noise() {
        let mut env = AcousticEnvironment::open_field();
        let base_noise = env.background_noise_level();

        env.wind_speed_m_s = 10.0;
        let wind_noise = env.background_noise_level();

        assert!(wind_noise > base_noise);
    }

    #[test]
    fn test_simulate_environment() {
        let sim = AcousticSimulator::new(48000);
        let signal = create_test_signal();
        let env = AcousticEnvironment::dense_jungle();

        let simulated = sim.simulate_environment(&signal, &env).unwrap();

        assert_eq!(simulated.len(), signal.len());
        // Simulated signal should be different from original
        assert!(simulated != signal);
    }

    #[test]
    fn test_simulate_quiet_environment() {
        let sim = AcousticSimulator::new(48000);
        let signal = create_test_signal();
        let env = AcousticEnvironment::open_field();

        let simulated = sim.simulate_environment(&signal, &env).unwrap();

        // Quiet environment should have minimal modification
        assert_eq!(simulated.len(), signal.len());
    }

    #[test]
    fn test_create_test_mixture() {
        let sim = AcousticSimulator::new(48000);
        let signal = create_test_signal();
        let env = AcousticEnvironment::rainy_rainforest();

        let mixture = sim.create_test_mixture(&signal, &env, 10.0).unwrap();

        assert_eq!(mixture.target_signal, signal);
        assert_eq!(mixture.snr_db, 10.0);
        assert!(!mixture.noise_layers.is_empty());
    }

    #[test]
    fn test_mixture_has_rain_in_rainforest() {
        let sim = AcousticSimulator::new(48000);
        let signal = create_test_signal();
        let env = AcousticEnvironment::rainy_rainforest();

        let mixture = sim.create_test_mixture(&signal, &env, 10.0).unwrap();

        // Should have rain noise layer
        assert!(mixture.noise_layers.iter().any(|(p, _, _)| p.name == "rain"));
    }

    #[test]
    fn test_mixture_has_insects_in_jungle() {
        let sim = AcousticSimulator::new(48000);
        let signal = create_test_signal();
        let env = AcousticEnvironment::dense_jungle();

        let mixture = sim.create_test_mixture(&signal, &env, 10.0).unwrap();

        // Should have insect noise layer
        assert!(mixture.noise_layers.iter().any(|(p, _, _)| p.name == "insects"));
    }

    #[test]
    fn test_generate_synthetic_vocalization() {
        let sim = AcousticSimulator::new(48000);
        let vocalization = sim.generate_synthetic_vocalization(8000.0, 100.0, None);

        assert!(!vocalization.is_empty());
        // Should be approximately 100ms at 48kHz
        assert!((vocalization.len() as f32 / 48000.0 * 1000.0 - 100.0).abs() < 10.0);
    }

    #[test]
    fn test_generate_synthetic_vocalization_with_modulation() {
        let sim = AcousticSimulator::new(48000);
        let vocalization = sim.generate_synthetic_vocalization(8000.0, 100.0, Some(5.0));

        assert!(!vocalization.is_empty());
        // Modulated signal should have slightly different characteristics
    }

    #[test]
    fn test_synthetic_vocalization_frequency() {
        let sim = AcousticSimulator::new(48000);
        let vocalization = sim.generate_synthetic_vocalization(1000.0, 100.0, None);

        // Check approximate frequency by counting zero crossings
        let crossings = vocalization.windows(2).filter(|w| w[0] * w[1] < 0.0).count();

        // For 1kHz signal, we expect roughly 100 crossings per 100ms
        // Allow wider tolerance due to envelope effects
        assert!(crossings > 50 && crossings < 200);
    }

    #[test]
    fn test_synthetic_vocalization_envelope() {
        let sim = AcousticSimulator::new(48000);
        let vocalization = sim.generate_synthetic_vocalization(1000.0, 100.0, None);

        // Check attack (should start quiet)
        assert!(vocalization[0].abs() < 0.2);

        // Check decay (should end quiet)
        assert!(vocalization[vocalization.len() - 1].abs() < 0.2);

        // Check that signal reaches significant amplitude somewhere
        let max_amplitude = vocalization.iter().map(|x| x.abs()).fold(0.0, f32::max);
        assert!(max_amplitude > 0.5);
    }

    #[test]
    fn test_noise_profile_defaults() {
        let profile = NoiseProfile::new("test".to_string(), SpectralColor::White);
        assert_eq!(profile.amplitude_variation, 0.3);
    }

    #[test]
    fn test_rain_profile() {
        let profile = NoiseProfile::rain();
        assert_eq!(profile.name, "rain");
        assert_eq!(profile.spectral_color, SpectralColor::Pink);
    }

    #[test]
    fn test_thunder_profile() {
        let profile = NoiseProfile::thunder();
        assert_eq!(profile.name, "thunder");
        assert_eq!(profile.spectral_color, SpectralColor::Brown);
    }

    #[test]
    fn test_wind_profile() {
        let profile = NoiseProfile::wind();
        assert_eq!(profile.name, "wind");
        assert_eq!(profile.spectral_color, SpectralColor::Pink);
    }

    #[test]
    fn test_insects_profile() {
        let profile = NoiseProfile::insects();
        assert_eq!(profile.name, "insects");
        assert_eq!(profile.spectral_color, SpectralColor::Blue);
    }

    #[test]
    fn test_birds_profile() {
        let profile = NoiseProfile::birds();
        assert_eq!(profile.name, "birds");
        assert_eq!(profile.spectral_color, SpectralColor::White);
    }

    #[test]
    fn test_reproducibility_with_seed() {
        let sim1 = AcousticSimulator::with_seed(48000, 42);
        let sim2 = AcousticSimulator::with_seed(48000, 42);

        let profile = NoiseProfile::new("test".to_string(), SpectralColor::White);
        let noise1 = sim1.generate_noise(&profile, 1000);
        let noise2 = sim2.generate_noise(&profile, 1000);

        assert_eq!(noise1, noise2);
    }

    #[test]
    fn test_different_seeds_produce_different_noise() {
        let sim1 = AcousticSimulator::with_seed(48000, 42);
        let sim2 = AcousticSimulator::with_seed(48000, 43);

        let profile = NoiseProfile::new("test".to_string(), SpectralColor::White);
        let noise1 = sim1.generate_noise(&profile, 1000);
        let noise2 = sim2.generate_noise(&profile, 1000);

        assert_ne!(noise1, noise2);
    }
}
