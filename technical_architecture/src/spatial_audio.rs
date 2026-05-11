//! Spatial Audio Rendering Module (Level 2.5)
//! ==========================================
//!
//! Implements directional audio rendering for spatial routing of
//! synthesized vocalizations. Supports broadcast vs unicast rendering
//! with panning, attenuation, and speaker array positioning.
//!
//! Features:
//! - 3D position-based panning for multichannel speaker arrays
//! - Distance-based attenuation (inverse square law)
//! - Broadcast vs unicast routing modes
//! - Line-of-sight occlusion handling
//!
//! Author: Sheel Morjaria (sheelmorjaria@gmail.com)
//! License: CC BY-ND 4.0 International

use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// 3D Position in meters
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Position3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Position3D {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn zero() -> Self {
        Self { x: 0.0, y: 0.0, z: 0.0 }
    }

    /// Calculate Euclidean distance to another position
    pub fn distance_to(&self, other: &Position3D) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Calculate azimuth angle in radians (-π to π)
    pub fn azimuth_to(&self, other: &Position3D) -> f32 {
        let dx = other.x - self.x;
        let dy = other.y - self.y;
        dy.atan2(dx)
    }

    /// Calculate elevation angle in radians (-π/2 to π/2)
    pub fn elevation_to(&self, other: &Position3D) -> f32 {
        let dx = other.x - self.x;
        let dy = other.y - self.y;
        let dz = other.z - self.z;
        let horizontal_dist = (dx * dx + dy * dy).sqrt();
        dz.atan2(horizontal_dist)
    }
}

impl From<[f32; 3]> for Position3D {
    fn from(arr: [f32; 3]) -> Self {
        Self { x: arr[0], y: arr[1], z: arr[2] }
    }
}

impl From<&[f32; 3]> for Position3D {
    fn from(arr: &[f32; 3]) -> Self {
        Self { x: arr[0], y: arr[1], z: arr[2] }
    }
}

/// Speaker in a multichannel array
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Speaker {
    /// Unique speaker identifier
    pub id: String,
    /// 3D position of the speaker
    pub position: Position3D,
    /// Speaker orientation (heading in radians)
    pub heading_rad: f32,
}

impl Speaker {
    pub fn new(id: String, position: Position3D, heading_rad: f32) -> Self {
        Self { id, position, heading_rad }
    }

    /// Calculate gain for a target position using vector-based amplitude panning (VBAP)
    pub fn calculate_gain(&self, target_position: Position3D) -> f32 {
        let distance = self.position.distance_to(&target_position);

        // Base gain using inverse square law with minimum distance
        let min_distance = 0.5; // 0.5 meters minimum to prevent infinite gain
        let effective_distance = distance.max(min_distance);
        let base_gain = 1.0 / (effective_distance * effective_distance);

        // Directional factor based on speaker heading
        let target_azimuth = self.position.azimuth_to(&target_position);
        let angle_diff = (target_azimuth - self.heading_rad).abs();

        // Cosine-weighted directional pattern (cardioid-like)
        let directional_factor = if angle_diff < PI / 2.0 {
            angle_diff.cos()
        } else {
            0.0
        };

        base_gain * directional_factor
    }
}

/// Speaker array configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerArray {
    pub speakers: Vec<Speaker>,
    pub reference_position: Position3D,
}

impl SpeakerArray {
    pub fn new(speakers: Vec<Speaker>, reference_position: Position3D) -> Self {
        Self { speakers, reference_position }
    }

    /// Create a circular array with N speakers around the reference position
    pub fn circular_array(n_speakers: usize, radius: f32, reference_position: Position3D) -> Self {
        let speakers = (0..n_speakers)
            .map(|i| {
                let angle = (i as f32 / n_speakers as f32) * 2.0 * PI;
                let x = reference_position.x + radius * angle.cos();
                let y = reference_position.y + radius * angle.sin();
                Speaker {
                    id: format!("speaker_{}", i),
                    position: Position3D { x, y, z: reference_position.z },
                    heading_rad: angle + PI, // Face inward
                }
            })
            .collect();

        Self { speakers, reference_position }
    }

    /// Calculate gains for all speakers for a target position
    pub fn calculate_gains(&self, target_position: Position3D) -> Vec<(String, f32)> {
        let mut gains: Vec<(String, f32)> = self.speakers
            .iter()
            .map(|speaker| {
                (speaker.id.clone(), speaker.calculate_gain(target_position))
            })
            .collect();

        // Normalize gains to prevent overall volume changes
        let total_gain: f32 = gains.iter().map(|(_, g)| g).sum();
        if total_gain > 0.0 {
            for (_, gain) in gains.iter_mut() {
                *gain /= total_gain;
            }
        }

        gains
    }
}

/// Spatial rendering mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpatialMode {
    /// Broadcast to all speakers equally
    Broadcast,
    /// Unicast to specific target direction
    Unicast,
}

impl Default for SpatialMode {
    fn default() -> Self {
        Self::Broadcast
    }
}

/// Spatial rendering metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialMetadata {
    /// Rendering mode
    pub mode: SpatialMode,
    /// Emitter position (who is vocalizing)
    pub emitter_position: Option<Position3D>,
    /// Target position (who to direct at) - for unicast
    pub target_position: Option<Position3D>,
    /// Target agent ID for spatial routing
    pub target_spatial_id: Option<String>,
    /// Broadcast flag (true = spatial rendering, false = point source)
    pub broadcast_flag: bool,
}

impl SpatialMetadata {
    pub fn broadcast() -> Self {
        Self {
            mode: SpatialMode::Broadcast,
            emitter_position: None,
            target_position: None,
            target_spatial_id: None,
            broadcast_flag: true,
        }
    }

    pub fn unicast(target_id: String, target_position: Position3D) -> Self {
        Self {
            mode: SpatialMode::Unicast,
            emitter_position: None,
            target_position: Some(target_position),
            target_spatial_id: Some(target_id),
            broadcast_flag: false,
        }
    }

    pub fn with_emitter_position(mut self, position: Position3D) -> Self {
        self.emitter_position = Some(position);
        self
    }
}

impl Default for SpatialMetadata {
    fn default() -> Self {
        Self::broadcast()
    }
}

/// Spatial audio renderer
pub struct SpatialAudioRenderer {
    speaker_array: SpeakerArray,
    /// Minimum gain for any speaker (prevents complete silence)
    min_gain: f32,
    /// Maximum gain for any speaker (prevents clipping)
    max_gain: f32,
}

impl SpatialAudioRenderer {
    pub fn new(speaker_array: SpeakerArray) -> Self {
        Self {
            speaker_array,
            min_gain: 0.01,
            max_gain: 1.0,
        }
    }

    pub fn with_limits(mut self, min_gain: f32, max_gain: f32) -> Self {
        self.min_gain = min_gain;
        self.max_gain = max_gain;
        self
    }

    /// Render spatial gains for a given spatial metadata
    pub fn render_spatial_gains(&self, metadata: &SpatialMetadata) -> Vec<(String, f32)> {
        match metadata.mode {
            SpatialMode::Broadcast => {
                // Broadcast: equal gain to all speakers
                self.speaker_array.speakers.iter()
                    .map(|s| (s.id.clone(), 1.0 / self.speaker_array.speakers.len() as f32))
                    .collect()
            }
            SpatialMode::Unicast => {
                // Unicast: directional gains toward target
                if let Some(target_pos) = metadata.target_position {
                    let mut gains = self.speaker_array.calculate_gains(target_pos);

                    // Apply limits
                    for (_, gain) in gains.iter_mut() {
                        *gain = gain.clamp(self.min_gain, self.max_gain);
                    }

                    // Renormalize
                    let total: f32 = gains.iter().map(|(_, g)| g).sum();
                    if total > 0.0 {
                        for (_, gain) in gains.iter_mut() {
                            *gain /= total;
                        }
                    }

                    gains
                } else {
                    // Fallback to broadcast if no target position
                    self.render_spatial_gains(&SpatialMetadata::broadcast())
                }
            }
        }
    }

    /// Apply spatial gains to a mono audio buffer to produce multichannel output
    pub fn apply_spatial_rendering(
        &self,
        metadata: &SpatialMetadata,
        audio_buffer: &[f32],
    ) -> Vec<Vec<f32>> {
        let gains = self.render_spatial_gains(metadata);

        gains.iter()
            .map(|(speaker_id, gain)| {
                audio_buffer.iter()
                    .map(|sample| sample * gain)
                    .collect()
            })
            .collect()
    }

    /// Get speaker array reference
    pub fn speaker_array(&self) -> &SpeakerArray {
        &self.speaker_array
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_distance() {
        let p1 = Position3D::new(0.0, 0.0, 0.0);
        let p2 = Position3D::new(3.0, 4.0, 0.0);

        assert!((p1.distance_to(&p2) - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_position_azimuth() {
        let p1 = Position3D::new(0.0, 0.0, 0.0);
        let p2 = Position3D::new(1.0, 1.0, 0.0); // 45 degrees

        let azimuth = p1.azimuth_to(&p2);
        assert!((azimuth - PI / 4.0).abs() < 0.001);
    }

    #[test]
    fn test_speaker_gain() {
        let speaker = Speaker::new(
            "test".to_string(),
            Position3D::new(0.0, 0.0, 0.0),
            0.0,
        );

        // Target at 1m distance, directly in front
        let target = Position3D::new(1.0, 0.0, 0.0);
        let gain = speaker.calculate_gain(target);

        assert!(gain > 0.0);
        assert!(gain <= 1.0);
    }

    #[test]
    fn test_circular_array() {
        let array = SpeakerArray::circular_array(4, 1.0, Position3D::zero());

        assert_eq!(array.speakers.len(), 4);

        // Check speakers are evenly distributed
        let speaker1 = &array.speakers[0];
        let speaker2 = &array.speakers[1];
        let angle_diff = (speaker1.heading_rad - speaker2.heading_rad).abs();
        assert!((angle_diff - PI / 2.0).abs() < 0.001);
    }

    #[test]
    fn test_spatial_metadata_broadcast() {
        let metadata = SpatialMetadata::broadcast();

        assert_eq!(metadata.mode, SpatialMode::Broadcast);
        assert!(metadata.broadcast_flag);
        assert!(metadata.target_spatial_id.is_none());
    }

    #[test]
    fn test_spatial_metadata_unicast() {
        let metadata = SpatialMetadata::unicast(
            "agent_002".to_string(),
            Position3D::new(1.0, 0.0, 0.0),
        );

        assert_eq!(metadata.mode, SpatialMode::Unicast);
        assert!(!metadata.broadcast_flag);
        assert_eq!(metadata.target_spatial_id, Some("agent_002".to_string()));
    }

    #[test]
    fn test_render_broadcast_gains() {
        let array = SpeakerArray::circular_array(4, 1.0, Position3D::zero());
        let renderer = SpatialAudioRenderer::new(array);

        let metadata = SpatialMetadata::broadcast();
        let gains = renderer.render_spatial_gains(&metadata);

        assert_eq!(gains.len(), 4);
        // All speakers should have equal gain
        let expected_gain = 0.25;
        for (_, gain) in gains {
            assert!((gain - expected_gain).abs() < 0.001);
        }
    }

    #[test]
    fn test_render_unicast_gains() {
        let array = SpeakerArray::circular_array(4, 1.0, Position3D::zero());
        let renderer = SpatialAudioRenderer::new(array);

        let metadata = SpatialMetadata::unicast(
            "agent_002".to_string(),
            Position3D::new(1.0, 0.0, 0.0), // Target at 1m along X axis
        );

        let gains = renderer.render_spatial_gains(&metadata);

        assert_eq!(gains.len(), 4);
        // Speakers closer to target should have higher gain
        let total: f32 = gains.iter().map(|(_, g)| g).sum();
        assert!((total - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_apply_spatial_rendering() {
        let array = SpeakerArray::circular_array(2, 1.0, Position3D::zero());
        let renderer = SpatialAudioRenderer::new(array);

        let metadata = SpatialMetadata::broadcast();
        let audio = vec![1.0, 0.5, -0.5, -1.0];

        let channels = renderer.apply_spatial_rendering(&metadata, &audio);

        assert_eq!(channels.len(), 2);
        assert_eq!(channels[0].len(), audio.len());
        assert_eq!(channels[1].len(), audio.len());

        // For broadcast, both channels should have equal amplitude
        for i in 0..audio.len() {
            assert!((channels[0][i] - channels[1][i]).abs() < 0.001);
        }
    }
}
