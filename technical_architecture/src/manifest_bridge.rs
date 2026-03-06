//! Manifest Bridge - Rust/Python Pipeline Communication
//! =====================================================
//!
//! This module provides JSON manifest file communication between:
//! - Rust Stage 1-2 (NBD + Feature Extraction) → segments_manifest.json
//! - Python Stage 3 (Clustering + Exemplar Selection) → clusters.json
//! - Rust Stage 4 (Granular Synthesis) ← synthesis_manifest.json
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                     PIPELINE CONTROLLER                          │
//! │                  (Rust: technical_architecture)                  │
//! ├──────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │  [1. NBD]     Load Raw Audio --> Segment --> Save to Cache      │
//! │       │         (Rust: neural_boundary.rs)                       │
//! │       ▼                                                          │
//! │  [2. 112D]    Load Segments --> Extract 112D --> Save .json     │
//! │       │         (Rust: micro_dynamics_extractor.rs)              │
//! │       ▼                                                          │
//! │  ╔═══════════════════════════════════════════════════════════╗  │
//! │  ║ [3. CORPUS ANALYSIS] (Python Bridge)                      ║  │
//! │  ║  - Load segments_manifest.json                            ║  │
//! │  ║  - Run Clustering (k=1020)                                ║  │
//! │  ║  - Output: clusters.json {id: [112D_mean, best_wav]}      ║  │
//! │  ╚═══════════════════════════════════════════════════════════╝  │
//! │       │                                                          │
//! │       ▼                                                          │
//! │  [4. SYNTHESIS] Load best_wav + 112D_mean --> Granular Synth    │
//! │       │         (Rust: synthesis.rs)                             │
//! │       ▼                                                          │
//! │  [5. PLAYBACK] Output Audio                                      │
//! │                                                                  │
//! └──────────────────────────────────────────────────────────────────┘
//! ```
//!
//! Author: Sheel Morjaria (sheelmorjaria@gmail.com)
//! License: CC BY-ND 4.0 International

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use crate::micro_dynamics_extractor::RosettaFeatures;
use crate::neural_boundary::PhraseBoundary;

// =============================================================================
// SEGMENTS MANIFEST (Stage 1-2 Output)
// =============================================================================

/// Information about a segmented audio file for the manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentEntry {
    /// Path to the audio segment file (WAV)
    pub file_path: String,
    /// 112D feature vector (RosettaFeatures as array)
    pub features_112d: Vec<f32>,
    /// Duration of the segment in milliseconds
    pub duration_ms: f32,
    /// Mean fundamental frequency in Hz
    pub mean_f0_hz: f32,
    /// Optional: Start time in original audio (ms)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_time_ms: Option<f64>,
    /// Optional: End time in original audio (ms)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_time_ms: Option<f64>,
}

/// Manifest containing all segmented audio features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentsManifest {
    /// Version of the manifest format
    pub version: String,
    /// Sample rate of the audio
    pub sample_rate: u32,
    /// Source audio file (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_file: Option<String>,
    /// All segment entries
    pub segments: Vec<SegmentEntry>,
}

impl SegmentsManifest {
    /// Create a new empty manifest
    pub fn new(sample_rate: u32) -> Self {
        Self {
            version: "1.0".to_string(),
            sample_rate,
            source_file: None,
            segments: Vec::new(),
        }
    }

    /// Create a manifest from source file
    pub fn from_source(source_file: &str, sample_rate: u32) -> Self {
        Self {
            version: "1.0".to_string(),
            sample_rate,
            source_file: Some(source_file.to_string()),
            segments: Vec::new(),
        }
    }

    /// Add a segment to the manifest
    pub fn add_segment(
        &mut self,
        file_path: &str,
        features: &RosettaFeatures,
        start_time_ms: Option<f64>,
        end_time_ms: Option<f64>,
    ) {
        let entry = SegmentEntry {
            file_path: file_path.to_string(),
            features_112d: features.to_array().to_vec(), // Convert [f32; 112] to Vec<f32>
            duration_ms: features.duration_ms,
            mean_f0_hz: features.mean_f0_hz,
            start_time_ms,
            end_time_ms,
        };
        self.segments.push(entry);
    }

    /// Save the manifest to a JSON file
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }

    /// Load a manifest from a JSON file
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let manifest = serde_json::from_reader(reader)?;
        Ok(manifest)
    }

    /// Get the number of segments
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    /// Check if the manifest is empty
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }
}

// =============================================================================
// CLUSTERS MANIFEST (Stage 3 Output / Stage 4 Input)
// =============================================================================

/// Exemplar metadata for synthesis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExemplarMetadata {
    pub mean_f0_hz: f32,
    pub duration_ms: f32,
    pub f0_range_hz: f32,
    pub rms_energy: f32,
    pub harmonic_to_noise_ratio: f32,
    pub attack_time_ms: f32,
    pub decay_time_ms: f32,
}

/// A single exemplar entry in the synthesis manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExemplarEntry {
    /// Cluster ID (0 to k-1)
    pub cluster_id: u32,
    /// Path to the exemplar audio file
    pub audio_path: String,
    /// Synthesis metadata extracted from features
    pub metadata: ExemplarMetadata,
}

/// Synthesis manifest containing exemplars for all clusters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisManifest {
    /// Number of clusters (vocabulary size)
    pub vocabulary_size: usize,
    /// All exemplar entries
    pub exemplars: Vec<ExemplarEntry>,
}

impl SynthesisManifest {
    /// Create a new empty synthesis manifest
    pub fn new() -> Self {
        Self {
            vocabulary_size: 0,
            exemplars: Vec::new(),
        }
    }

    /// Load a synthesis manifest from a JSON file
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let manifest = serde_json::from_reader(reader)?;
        Ok(manifest)
    }

    /// Get an exemplar by cluster ID
    pub fn get_exemplar(&self, cluster_id: u32) -> Option<&ExemplarEntry> {
        self.exemplars
            .iter()
            .find(|e| e.cluster_id == cluster_id)
    }

    /// Get the number of exemplars
    pub fn len(&self) -> usize {
        self.exemplars.len()
    }

    /// Check if the manifest is empty
    pub fn is_empty(&self) -> bool {
        self.exemplars.is_empty()
    }
}

impl Default for SynthesisManifest {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// CLUSTER INFO (Full cluster information from Python)
// =============================================================================

/// Full cluster information from Python corpus analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterInfo {
    /// Cluster ID
    pub cluster_id: u32,
    /// Cluster centroid in 112D feature space
    pub centroid_112d: Vec<f32>,
    /// Path to the best exemplar audio
    pub exemplar_audio: String,
    /// 112D features of the exemplar
    pub exemplar_features_112d: Vec<f32>,
    /// Number of segments assigned to this cluster
    pub num_segments: usize,
    /// Mean distance of segments to centroid
    pub mean_distance_to_centroid: f64,
}

/// Full clusters manifest from Python Stage 3
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClustersManifest {
    /// Vocabulary size (k)
    pub vocabulary_size: usize,
    /// Number of clusters
    pub num_clusters: usize,
    /// All cluster information
    pub clusters: HashMap<String, ClusterInfo>,
}

impl ClustersManifest {
    /// Load a clusters manifest from a JSON file
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let manifest = serde_json::from_reader(reader)?;
        Ok(manifest)
    }

    /// Get cluster info by cluster ID
    pub fn get_cluster(&self, cluster_id: u32) -> Option<&ClusterInfo> {
        self.clusters.get(&cluster_id.to_string())
    }

    /// Convert to a synthesis manifest for use in Stage 4
    pub fn to_synthesis_manifest(&self) -> SynthesisManifest {
        let exemplars: Vec<ExemplarEntry> = self
            .clusters
            .values()
            .map(|cluster| {
                // Extract key metadata from features
                let features = &cluster.exemplar_features_112d;
                ExemplarEntry {
                    cluster_id: cluster.cluster_id,
                    audio_path: cluster.exemplar_audio.clone(),
                    metadata: ExemplarMetadata {
                        mean_f0_hz: features.get(0).copied().unwrap_or(5000.0),
                        duration_ms: features.get(1).copied().unwrap_or(100.0),
                        f0_range_hz: features.get(2).copied().unwrap_or(500.0),
                        rms_energy: features.get(3).copied().unwrap_or(0.5),
                        harmonic_to_noise_ratio: features.get(6).copied().unwrap_or(15.0),
                        attack_time_ms: features.get(9).copied().unwrap_or(10.0),
                        decay_time_ms: features.get(10).copied().unwrap_or(50.0),
                    },
                }
            })
            .collect();

        SynthesisManifest {
            vocabulary_size: self.vocabulary_size,
            exemplars,
        }
    }
}

// =============================================================================
// PIPELINE CONTROLLER
// =============================================================================

/// Pipeline controller that orchestrates the 5-stage synthesis pipeline
pub struct PipelineController {
    /// Sample rate for audio processing
    sample_rate: u32,
    /// Current segments manifest (Stage 1-2 output)
    segments_manifest: Option<SegmentsManifest>,
    /// Current clusters manifest (Stage 3 output)
    clusters_manifest: Option<ClustersManifest>,
    /// Current synthesis manifest (Stage 3-4 bridge)
    synthesis_manifest: Option<SynthesisManifest>,
}

impl PipelineController {
    /// Create a new pipeline controller
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            segments_manifest: None,
            clusters_manifest: None,
            synthesis_manifest: None,
        }
    }

    /// Stage 1-2: Create segments manifest from extracted features
    pub fn create_segments_manifest(
        &mut self,
        source_file: Option<&str>,
    ) -> &mut SegmentsManifest {
        let manifest = if let Some(src) = source_file {
            SegmentsManifest::from_source(src, self.sample_rate)
        } else {
            SegmentsManifest::new(self.sample_rate)
        };
        self.segments_manifest = Some(manifest);
        self.segments_manifest.as_mut().unwrap()
    }

    /// Get the current segments manifest
    pub fn segments_manifest(&self) -> Option<&SegmentsManifest> {
        self.segments_manifest.as_ref()
    }

    /// Get mutable segments manifest
    pub fn segments_manifest_mut(&mut self) -> Option<&mut SegmentsManifest> {
        self.segments_manifest.as_mut()
    }

    /// Save segments manifest to file (for Python Stage 3)
    pub fn save_segments_manifest(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(ref manifest) = self.segments_manifest {
            manifest.save(path)?;
            Ok(())
        } else {
            Err(anyhow::anyhow!("No segments manifest to save"))
        }
    }

    /// Stage 3-4: Load clusters manifest from Python
    pub fn load_clusters_manifest(&mut self, path: &Path) -> anyhow::Result<()> {
        self.clusters_manifest = Some(ClustersManifest::load(path)?);

        // Also create synthesis manifest for Stage 4
        if let Some(ref clusters) = self.clusters_manifest {
            self.synthesis_manifest = Some(clusters.to_synthesis_manifest());
        }

        Ok(())
    }

    /// Load synthesis manifest directly
    pub fn load_synthesis_manifest(&mut self, path: &Path) -> anyhow::Result<()> {
        self.synthesis_manifest = Some(SynthesisManifest::load(path)?);
        Ok(())
    }

    /// Get the synthesis manifest for Stage 4
    pub fn synthesis_manifest(&self) -> Option<&SynthesisManifest> {
        self.synthesis_manifest.as_ref()
    }

    /// Get an exemplar for synthesis by cluster ID
    pub fn get_exemplar(&self, cluster_id: u32) -> Option<&ExemplarEntry> {
        self.synthesis_manifest
            .as_ref()?
            .get_exemplar(cluster_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_segments_manifest_creation() {
        let manifest = SegmentsManifest::new(44100);
        assert_eq!(manifest.sample_rate, 44100);
        assert!(manifest.is_empty());
    }

    #[test]
    fn test_segments_manifest_add_segment() {
        let mut manifest = SegmentsManifest::new(44100);
        let features = RosettaFeatures::default();

        manifest.add_segment("test.wav", &features, Some(0.0), Some(100.0));

        assert_eq!(manifest.len(), 1);
        assert_eq!(manifest.segments[0].file_path, "test.wav");
        assert_eq!(manifest.segments[0].features_112d.len(), 112);
    }

    #[test]
    fn test_segments_manifest_save_load() {
        let mut manifest = SegmentsManifest::new(44100);
        let features = RosettaFeatures::default();

        manifest.add_segment("seg_001.wav", &features, Some(0.0), Some(100.0));
        manifest.add_segment("seg_002.wav", &features, Some(100.0), Some(200.0));

        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        // Save
        manifest.save(path).unwrap();

        // Load
        let loaded = SegmentsManifest::load(path).unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.sample_rate, 44100);
        assert_eq!(loaded.segments[0].file_path, "seg_001.wav");
    }

    #[test]
    fn test_synthesis_manifest_creation() {
        let manifest = SynthesisManifest::new();
        assert!(manifest.is_empty());
    }

    #[test]
    fn test_pipeline_controller() {
        let mut controller = PipelineController::new(44100);

        // Create segments manifest
        let segments = controller.create_segments_manifest(Some("test_audio.wav"));
        assert!(segments.is_empty());

        // Add a segment
        let features = RosettaFeatures::default();
        controller
            .segments_manifest_mut()
            .unwrap()
            .add_segment("seg_001.wav", &features, Some(0.0), Some(100.0));

        assert_eq!(controller.segments_manifest().unwrap().len(), 1);
    }

    #[test]
    fn test_cluster_to_synthesis_conversion() {
        let mut clusters = ClustersManifest {
            vocabulary_size: 2,
            num_clusters: 2,
            clusters: HashMap::new(),
        };

        // Add a cluster
        let cluster = ClusterInfo {
            cluster_id: 0,
            centroid_112d: vec![0.0; 112],
            exemplar_audio: "seg_001.wav".to_string(),
            exemplar_features_112d: vec![8000.0, 100.0, 500.0, 0.5, 0.1, 0.8, 15.0],
            num_segments: 10,
            mean_distance_to_centroid: 0.85,
        };
        clusters.clusters.insert("0".to_string(), cluster);

        // Convert to synthesis manifest
        let synth_manifest = clusters.to_synthesis_manifest();

        assert_eq!(synth_manifest.vocabulary_size, 2);
        assert_eq!(synth_manifest.len(), 1);

        let exemplar = synth_manifest.get_exemplar(0).unwrap();
        assert_eq!(exemplar.audio_path, "seg_001.wav");
        assert_eq!(exemplar.metadata.mean_f0_hz, 8000.0);
    }
}
