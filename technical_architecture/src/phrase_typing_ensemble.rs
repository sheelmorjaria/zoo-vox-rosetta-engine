//! Phrase Typing Ensemble - Combines Closed-Set Matching with Open-Set Discovery
//!
//! Solves the "Stability vs. Discovery" trade-off in intra-call linguistics:
//! - k-NN Matching (Closed-Set): Stable IDs, consistent across days
//! - HDBSCAN Clustering (Open-Set): Discovers new phrases dynamically
//!
//! # Usage
//!
//! ```rust
//! use technical_architecture::phrase_typing_ensemble::{PhraseTypingEnsemble, PhraseLabel};
//!
//! let ensemble = PhraseTypingEnsemble::new("phrase_library.bin", 0.85);
//!
//! // Classify a segment
//! let label = ensemble.classify(&segment_features);
//!
//! match label {
//!     PhraseLabel::Known(id) => println!("Known phrase: {}", id),
//!     PhraseLabel::Discovered(id) => println!("New phrase discovered: {}", id),
//!     PhraseLabel::Noise => println!("Rejected as noise"),
//! }
//! ```

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Threshold for template matching confidence
const DEFAULT_MATCH_THRESHOLD: f32 = 0.85;

/// Minimum cluster size for HDBSCAN discovery
const MIN_CLUSTER_SIZE: usize = 3;

/// Phrase label types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PhraseLabel {
    /// Known phrase from library (high confidence match)
    Known(String),
    /// Newly discovered phrase (HDBSCAN cluster)
    Discovered(String),
    /// Rejected as noise (purity gate)
    Noise,
    /// Uncertain (needs review)
    Uncertain { best_match: String, score: f32 },
}

/// Phrase library entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhraseTemplate {
    pub id: String,
    pub features: Vec<f32>,
    pub sample_count: usize,
    pub first_seen: String,
    pub last_seen: String,
    pub aliases: Vec<String>,
}

/// Phrase library for template matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhraseLibrary {
    pub phrases: HashMap<String, PhraseTemplate>,
    pub version: String,
    pub created: String,
}

impl PhraseLibrary {
    /// Load library from file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())
            .with_context(|| "Failed to read phrase library")?;
        let library: PhraseLibrary =
            serde_json::from_str(&content).with_context(|| "Failed to parse phrase library")?;
        Ok(library)
    }

    /// Save library to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = serde_json::to_string_pretty(self)
            .with_context(|| "Failed to serialize phrase library")?;
        std::fs::write(path.as_ref(), content).with_context(|| "Failed to write phrase library")?;
        Ok(())
    }

    /// Create empty library
    pub fn new() -> Self {
        Self {
            phrases: HashMap::new(),
            version: "1.0.0".to_string(),
            created: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Find best matching phrase
    pub fn find_best_match(&self, features: &[f32]) -> Option<(String, f32)> {
        let mut best_id = None;
        let mut best_score = 0.0f32;

        for (id, template) in &self.phrases {
            let score = compute_similarity(features, &template.features);
            if score > best_score {
                best_score = score;
                best_id = Some(id.clone());
            }
        }

        best_id.map(|id| (id, best_score))
    }

    /// Add new phrase to library
    pub fn add_phrase(&mut self, id: String, features: Vec<f32>) {
        let now = chrono::Utc::now().to_rfc3339();
        let template = PhraseTemplate {
            id: id.clone(),
            features,
            sample_count: 1,
            first_seen: now.clone(),
            last_seen: now,
            aliases: vec![],
        };
        self.phrases.insert(id, template);
    }

    /// Update existing phrase with new sample
    pub fn update_phrase(&mut self, id: &str, features: &[f32]) {
        if let Some(template) = self.phrases.get_mut(id) {
            // Update with exponential moving average
            let alpha = 0.1;
            for i in 0..template.features.len().min(features.len()) {
                template.features[i] = template.features[i] * (1.0 - alpha) + features[i] * alpha;
            }
            template.sample_count += 1;
            template.last_seen = chrono::Utc::now().to_rfc3339();
        }
    }

    /// Get next available phrase ID
    pub fn next_phrase_id(&self, prefix: &str) -> String {
        let mut max_num = 0;
        for id in self.phrases.keys() {
            if id.starts_with(prefix) {
                if let Some(num) = id.strip_prefix(prefix) {
                    if let Ok(n) = num.parse::<u32>() {
                        max_num = max_num.max(n);
                    }
                }
            }
        }
        format!("{}{}", prefix, max_num + 1)
    }
}

impl Default for PhraseLibrary {
    fn default() -> Self {
        Self::new()
    }
}

/// HDBSCAN clusterer for phrase discovery
pub struct PhraseClusterer {
    min_cluster_size: usize,
    min_samples: usize,
}

impl PhraseClusterer {
    pub fn new(min_cluster_size: usize) -> Self {
        Self {
            min_cluster_size,
            min_samples: 2,
        }
    }

    /// Cluster segments into phrase types
    pub fn cluster(&self, segments: &[Vec<f32>]) -> Vec<Option<usize>> {
        if segments.is_empty() {
            return vec![];
        }

        // Compute distance matrix
        let n = segments.len();
        let mut distances = vec![0.0f32; n * n];

        for i in 0..n {
            for j in (i + 1)..n {
                let dist = 1.0 - compute_similarity(&segments[i], &segments[j]);
                distances[i * n + j] = dist;
                distances[j * n + i] = dist;
            }
        }

        // Simple clustering using distance threshold
        // (Full HDBSCAN would be more sophisticated)
        let threshold = 0.3;
        let mut labels = vec![None; n];
        let mut cluster_id = 0;

        for i in 0..n {
            if labels[i].is_some() {
                continue;
            }

            // Find neighbors
            let mut neighbors: Vec<usize> = (0..n)
                .filter(|&j| j != i && distances[i * n + j] < threshold)
                .collect();

            if neighbors.len() + 1 >= self.min_cluster_size {
                neighbors.push(i);
                for &j in &neighbors {
                    labels[j] = Some(cluster_id);
                }
                cluster_id += 1;
            }
        }

        labels
    }
}

/// Phrase Typing Ensemble - Combines Template Matching + Clustering
pub struct PhraseTypingEnsemble {
    /// Phrase library for template matching
    library: PhraseLibrary,
    /// Clusterer for discovery
    clusterer: PhraseClusterer,
    /// Match confidence threshold
    match_threshold: f32,
    /// Pending segments for batch clustering
    pending_segments: Vec<(Vec<f32>, String)>,
    /// Discovery counter
    discovery_counter: usize,
}

impl PhraseTypingEnsemble {
    /// Create new ensemble with library
    pub fn new(library_path: &str, match_threshold: f32) -> Result<Self> {
        let library = if Path::new(library_path).exists() {
            PhraseLibrary::load(library_path)?
        } else {
            PhraseLibrary::new()
        };

        Ok(Self {
            library,
            clusterer: PhraseClusterer::new(MIN_CLUSTER_SIZE),
            match_threshold,
            pending_segments: vec![],
            discovery_counter: 0,
        })
    }

    /// Classify a single segment
    pub fn classify(&mut self, features: &[f32], segment_id: &str) -> PhraseLabel {
        // Stage A: Template Matching (Stability)
        if let Some((match_id, score)) = self.library.find_best_match(features) {
            if score > self.match_threshold {
                // High confidence - known phrase
                self.library.update_phrase(&match_id, features);
                return PhraseLabel::Known(match_id);
            } else if score > self.match_threshold * 0.8 {
                // Medium confidence - uncertain
                return PhraseLabel::Uncertain {
                    best_match: match_id,
                    score,
                };
            }
        }

        // Stage B: Add to pending for batch clustering
        self.pending_segments
            .push((features.to_vec(), segment_id.to_string()));

        // If we have enough pending, run clustering
        if self.pending_segments.len() >= 10 {
            self.flush_pending()
        } else {
            PhraseLabel::Noise // Tentatively noise
        }
    }

    /// Process pending segments with clustering
    fn flush_pending(&mut self) -> PhraseLabel {
        if self.pending_segments.is_empty() {
            return PhraseLabel::Noise;
        }

        let segments: Vec<Vec<f32>> = self
            .pending_segments
            .iter()
            .map(|(f, _)| f.clone())
            .collect();

        let labels = self.clusterer.cluster(&segments);

        // Find the label for the most recent segment
        let last_idx = labels.len() - 1;
        let result = if let Some(cluster_id) = labels[last_idx] {
            // Create new discovered phrase
            self.discovery_counter += 1;
            let phrase_id = format!("discovered_{}", self.discovery_counter);

            // Add to library
            self.library
                .add_phrase(phrase_id.clone(), self.pending_segments[last_idx].0.clone());

            PhraseLabel::Discovered(phrase_id)
        } else {
            PhraseLabel::Noise
        };

        // Clear pending
        self.pending_segments.clear();

        result
    }

    /// Batch classify multiple segments
    pub fn classify_batch(&mut self, segments: &[(Vec<f32>, String)]) -> Vec<PhraseLabel> {
        let mut results = Vec::with_capacity(segments.len());

        for (features, id) in segments {
            let label = self.classify(features, id);
            results.push(label);
        }

        // Flush any remaining
        if !self.pending_segments.is_empty() {
            // Run clustering on all pending
            let segs: Vec<Vec<f32>> = self
                .pending_segments
                .iter()
                .map(|(f, _)| f.clone())
                .collect();
            let labels = self.clusterer.cluster(&segs);

            // Assign discovered labels
            for (i, label) in labels.iter().enumerate() {
                if let Some(cluster_id) = label {
                    let phrase_id = format!("discovered_{}", cluster_id);
                    self.library
                        .add_phrase(phrase_id.clone(), self.pending_segments[i].0.clone());
                }
            }
            self.pending_segments.clear();
        }

        results
    }

    /// Save updated library
    pub fn save_library<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        self.library.save(path)
    }

    /// Get library statistics
    pub fn stats(&self) -> EnsembleStats {
        EnsembleStats {
            known_phrases: self.library.phrases.len(),
            pending_segments: self.pending_segments.len(),
            discovery_counter: self.discovery_counter,
        }
    }
}

/// Ensemble statistics
#[derive(Debug, Clone)]
pub struct EnsembleStats {
    pub known_phrases: usize,
    pub pending_segments: usize,
    pub discovery_counter: usize,
}

/// Compute cosine similarity between two feature vectors
fn compute_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom > 0.0 {
        dot / denom
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phrase_library() {
        let mut library = PhraseLibrary::new();

        // Add phrase
        let features = vec![1.0, 0.5, 0.3];
        library.add_phrase("phrase_1".to_string(), features.clone());

        assert_eq!(library.phrases.len(), 1);

        // Find match
        let (id, score) = library.find_best_match(&features).unwrap();
        assert_eq!(id, "phrase_1");
        assert!(score > 0.99);
    }

    #[test]
    fn test_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((compute_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!(compute_similarity(&a, &c) < 0.01);
    }

    #[test]
    fn test_clusterer() {
        let clusterer = PhraseClusterer::new(2);

        // Two distinct clusters
        let seg1 = vec![1.0, 0.0, 0.0];
        let seg2 = vec![0.95, 0.05, 0.0];
        let seg3 = vec![0.0, 1.0, 0.0];
        let seg4 = vec![0.05, 0.95, 0.0];

        let segments = vec![seg1, seg2, seg3, seg4];
        let labels = clusterer.cluster(&segments);

        // Should have two clusters
        let clusters: std::collections::HashSet<Option<usize>> = labels.iter().copied().collect();
        assert!(clusters.len() >= 2);
    }
}
