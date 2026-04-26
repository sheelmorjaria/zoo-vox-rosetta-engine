//! Phrase Discovery Pipeline: ASE → Distance Matrix → HDBSCAN
//!
//! Strategic Integration (from manuscript review):
//!
//! The Problem with standard HDBSCAN:
//!   - Uses Euclidean distance in 112D space
//!   - Suffers from "Curse of Dimensionality"
//!   - Scale-sensitive: Duration (0-3000ms) dominates HNR (0-40dB)
//!
//! The Solution: "Weighted Distance Matrix" Approach
//!   1. ASE calculates distance with Physics/Texture weighting
//!   2. HDBSCAN clusters the pre-computed matrix
//!   3. Result: Clusters respect "Physics-Texture" hierarchy
//!
//! Pipeline:
//!   INPUT: Raw Audio
//!         ↓
//!   [1] Smart Segmenter (CPD vs NBD)
//!         ↓
//!   [2] Purity Gate (HNR/Flatness Filter)
//!         ↓
//!   [3] 112D Feature Extraction
//!         ↓
//!   [4] Acoustic Similarity Engine (Distance Matrix)
//!         ↓
//!   [5] HDBSCAN Clustering (Phrase Discovery)
//!         ↓
//!   [6] Linguistic Analysis (Zipf/Perplexity/Duration CV)
//!         ↓
//!   [7] Ensemble Classifier (Species ID)

use std::collections::HashMap;

use crate::acoustic_algebra_105d::Vector112D;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the phrase discovery pipeline
#[derive(Debug, Clone)]
pub struct PhraseDiscoveryConfig {
    /// Purity Gate settings
    pub purity_min_hnr_db: f32,
    pub purity_max_flatness: f32,
    pub purity_min_duration_ms: f32,

    /// HDBSCAN settings
    pub min_cluster_size: usize,
    pub min_samples: usize,

    /// ASE weighting
    pub physics_weight: f32, // For broad taxonomy
    pub texture_weight: f32, // For fine species ID
}

impl Default for PhraseDiscoveryConfig {
    fn default() -> Self {
        Self {
            purity_min_hnr_db: 3.0,
            purity_max_flatness: 0.6,
            purity_min_duration_ms: 40.0,
            min_cluster_size: 5,
            min_samples: 3,
            physics_weight: 0.6, // 60% physics for broad groups
            texture_weight: 0.4, // 40% texture for fine ID
        }
    }
}

// ============================================================================
// Phrase Discovery Pipeline
// ============================================================================

pub struct PhraseDiscoveryPipeline {
    config: PhraseDiscoveryConfig,
    stats: PipelineStats,
}

#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    pub total_segments: usize,
    pub purity_rejected: usize,
    pub noise_clusters: usize,
    pub phrase_clusters: usize,
}

/// A discovered phrase with its cluster assignment
#[derive(Debug, Clone)]
pub struct DiscoveredPhrase {
    /// Segment ID
    pub segment_id: usize,
    /// Start time in ms
    pub start_ms: f32,
    /// End time in ms
    pub end_ms: f32,
    /// 112D features (46D physics + 66D texture)
    pub features: Vec<f32>,
    /// Cluster assignment (-1 = noise)
    pub cluster_id: i32,
    /// Phrase type label (assigned post-clustering)
    pub phrase_label: Option<String>,
}

/// Result of the discovery pipeline
#[derive(Debug, Clone)]
pub struct DiscoveryResult {
    /// All discovered phrases
    pub phrases: Vec<DiscoveredPhrase>,
    /// Cluster centroids (cluster_id -> centroid features)
    pub centroids: HashMap<i32, Vec<f32>>,
    /// Cluster sizes
    pub cluster_sizes: HashMap<i32, usize>,
    /// Linguistic metrics
    pub metrics: LinguisticMetrics,
}

/// Linguistic metrics for discovered phrases
#[derive(Debug, Clone, Default)]
pub struct LinguisticMetrics {
    /// Zipf's law R²
    pub zipf_r2: f64,
    /// Perplexity
    pub perplexity: f64,
    /// Duration CV (Coefficient of Variation)
    pub duration_cv: f64,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Total phrases
    pub total_phrases: usize,
}

impl PhraseDiscoveryPipeline {
    pub fn new(config: PhraseDiscoveryConfig) -> Self {
        Self {
            config,
            stats: PipelineStats::default(),
        }
    }

    /// Run the full discovery pipeline
    ///
    /// # Arguments
    /// * `segments` - Pre-segmented audio with 112D features
    ///
    /// # Returns
    /// * DiscoveryResult with cluster assignments and linguistic metrics
    pub fn discover(&mut self, segments: &[SegmentFeatures]) -> DiscoveryResult {
        self.stats.total_segments = segments.len();

        // Step 1: Apply Purity Gate
        let pure_segments: Vec<(usize, &[f32])> = segments
            .iter()
            .enumerate()
            .filter(|(_, seg)| self.passes_purity_gate(seg))
            .map(|(i, seg)| (i, &seg.features[..]))
            .collect();

        self.stats.purity_rejected = segments.len() - pure_segments.len();

        // Step 2: Compute ASE Distance Matrix
        let distance_matrix = self.compute_ase_distance_matrix(&pure_segments);

        // Step 3: Run HDBSCAN on the distance matrix
        let cluster_labels = self.run_hdbscan(&distance_matrix);

        // Step 4: Build result
        let mut phrases = Vec::new();
        let mut cluster_features: HashMap<i32, Vec<Vec<f32>>> = HashMap::new();

        for (idx, (seg_idx, _)) in pure_segments.iter().enumerate() {
            let cluster_id = cluster_labels.get(idx).copied().unwrap_or(-1);
            let seg = &segments[*seg_idx];

            phrases.push(DiscoveredPhrase {
                segment_id: *seg_idx,
                start_ms: seg.start_ms,
                end_ms: seg.end_ms,
                features: seg.features.clone(),
                cluster_id,
                phrase_label: None,
            });

            if cluster_id >= 0 {
                cluster_features
                    .entry(cluster_id)
                    .or_default()
                    .push(seg.features.clone());
            }
        }

        // Step 5: Compute centroids
        let centroids: HashMap<i32, Vec<f32>> = cluster_features
            .iter()
            .map(|(&cluster_id, feats)| {
                let centroid = compute_centroid(feats);
                (cluster_id, centroid)
            })
            .collect();

        // Step 6: Count cluster sizes
        let cluster_sizes: HashMap<i32, usize> =
            cluster_labels
                .iter()
                .filter(|&&c| c >= 0)
                .fold(HashMap::new(), |mut acc, &c| {
                    *acc.entry(c).or_insert(0) += 1;
                    acc
                });

        self.stats.noise_clusters = cluster_labels.iter().filter(|&&c| c < 0).count();
        self.stats.phrase_clusters = cluster_sizes.len();

        // Step 7: Compute linguistic metrics
        let metrics = self.compute_linguistic_metrics(&phrases, &cluster_sizes);

        DiscoveryResult {
            phrases,
            centroids,
            cluster_sizes,
            metrics,
        }
    }

    /// Purity Gate: Reject non-biological sounds
    fn passes_purity_gate(&self, seg: &SegmentFeatures) -> bool {
        // Extract purity features from 112D
        let hnr_db = seg.features.get(23).copied().unwrap_or(0.0);
        let flatness = seg.features.get(16).copied().unwrap_or(0.5);
        let duration_ms = seg.end_ms - seg.start_ms;

        // Apply filters
        duration_ms >= self.config.purity_min_duration_ms
            && hnr_db >= self.config.purity_min_hnr_db
            && flatness <= self.config.purity_max_flatness
    }

    /// Compute ASE-weighted distance matrix using Vector112D algebra
    fn compute_ase_distance_matrix(&self, segments: &[(usize, &[f32])]) -> Vec<Vec<f64>> {
        let n = segments.len();
        let mut matrix = vec![vec![0.0; n]; n];

        // Pre-convert to Vector112D
        let vectors: Vec<Vector112D> = segments
            .iter()
            .map(|(_, feat)| Vector112D::from_array(feat))
            .collect();

        for i in 0..n {
            for j in (i + 1)..n {
                let dist = vectors[i].distance_to(&vectors[j]) as f64;
                matrix[i][j] = dist;
                matrix[j][i] = dist;
            }
        }

        matrix
    }

    /// Run HDBSCAN on pre-computed distance matrix
    fn run_hdbscan(&self, distance_matrix: &[Vec<f64>]) -> Vec<i32> {
        let n = distance_matrix.len();
        if n == 0 {
            return Vec::new();
        }

        // Simplified HDBSCAN implementation
        // In production, use the `hdbscan` crate with pre-computed distances

        // For now, implement a simple density-based clustering
        let min_pts = self.config.min_samples;
        let mut labels = vec![-1i32; n];
        let mut cluster_id = 0i32;

        // Compute local density (k-distance)
        let densities: Vec<f64> = (0..n)
            .map(|i| {
                let mut distances: Vec<f64> = distance_matrix[i].to_vec();
                distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
                distances.iter().take(min_pts).sum::<f64>() / min_pts as f64
            })
            .collect();

        // Find core points (low k-distance = high density)
        let avg_density = densities.iter().sum::<f64>() / n as f64;

        for i in 0..n {
            if labels[i] >= 0 {
                continue;
            }

            // Check if this is a core point
            if densities[i] < avg_density {
                // Start new cluster
                labels[i] = cluster_id;

                // Expand cluster
                let mut queue = vec![i];
                while let Some(current) = queue.pop() {
                    for j in 0..n {
                        if labels[j] < 0 && distance_matrix[current][j] < avg_density {
                            labels[j] = cluster_id;
                            queue.push(j);
                        }
                    }
                }

                cluster_id += 1;
            }
        }

        labels
    }

    /// Compute linguistic metrics
    fn compute_linguistic_metrics(
        &self,
        phrases: &[DiscoveredPhrase],
        cluster_sizes: &HashMap<i32, usize>,
    ) -> LinguisticMetrics {
        let vocab_size = cluster_sizes.len();
        let total_phrases = phrases.iter().filter(|p| p.cluster_id >= 0).count();

        // Compute Duration CV
        let durations: Vec<f64> = phrases
            .iter()
            .filter(|p| p.cluster_id >= 0)
            .map(|p| (p.end_ms - p.start_ms) as f64)
            .collect();

        let duration_cv = if !durations.is_empty() {
            let mean = durations.iter().sum::<f64>() / durations.len() as f64;
            let variance =
                durations.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / durations.len() as f64;
            let std = variance.sqrt();
            if mean > 0.0 {
                std / mean
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Compute Zipf's Law R²
        let zipf_r2 = self.compute_zipf_r2(cluster_sizes);

        // Compute perplexity
        let perplexity = self.compute_perplexity(cluster_sizes, total_phrases);

        LinguisticMetrics {
            zipf_r2,
            perplexity,
            duration_cv,
            vocab_size,
            total_phrases,
        }
    }

    /// Compute Zipf's Law R²
    fn compute_zipf_r2(&self, cluster_sizes: &HashMap<i32, usize>) -> f64 {
        let mut sizes: Vec<usize> = cluster_sizes.values().copied().collect();
        if sizes.len() < 3 {
            return 0.0;
        }

        sizes.sort_by(|a, b| b.cmp(a));

        // Log-log regression
        let n = sizes.len() as f64;
        let log_ranks: Vec<f64> = (1..=sizes.len()).map(|i| (i as f64).ln()).collect();
        let log_freqs: Vec<f64> = sizes.iter().map(|&f| (f as f64).ln()).collect();

        let sum_x: f64 = log_ranks.iter().sum();
        let sum_y: f64 = log_freqs.iter().sum();
        let sum_xy: f64 = log_ranks
            .iter()
            .zip(log_freqs.iter())
            .map(|(x, y)| x * y)
            .sum();
        let sum_xx: f64 = log_ranks.iter().map(|x| x * x).sum();

        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-10 {
            return 0.0;
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denom;
        let intercept = (sum_y - slope * sum_x) / n;

        // R² calculation
        let y_mean = sum_y / n;
        let ss_tot: f64 = log_freqs.iter().map(|y| (y - y_mean).powi(2)).sum();
        let ss_res: f64 = log_ranks
            .iter()
            .zip(log_freqs.iter())
            .map(|(x, y)| {
                let pred = slope * x + intercept;
                (y - pred).powi(2)
            })
            .sum();

        if ss_tot > 1e-10 {
            1.0 - ss_res / ss_tot
        } else {
            0.0
        }
    }

    /// Compute perplexity
    fn compute_perplexity(&self, cluster_sizes: &HashMap<i32, usize>, total: usize) -> f64 {
        if total == 0 {
            return 0.0;
        }

        let entropy: f64 = cluster_sizes
            .values()
            .map(|&size| {
                let p = size as f64 / total as f64;
                -p * p.log2()
            })
            .sum();

        2f64.powf(entropy)
    }

    /// Get pipeline statistics
    pub fn stats(&self) -> &PipelineStats {
        &self.stats
    }
}

// ============================================================================
// Supporting Types
// ============================================================================

/// Segment with extracted features
#[derive(Debug, Clone)]
pub struct SegmentFeatures {
    pub start_ms: f32,
    pub end_ms: f32,
    pub features: Vec<f32>,
}

/// Compute centroid of feature vectors
fn compute_centroid(features: &[Vec<f32>]) -> Vec<f32> {
    if features.is_empty() {
        return Vec::new();
    }

    let dim = features[0].len();
    let mut centroid = vec![0.0f32; dim];

    for feat in features {
        for (i, &v) in feat.iter().enumerate() {
            if i < dim {
                centroid[i] += v;
            }
        }
    }

    let n = features.len() as f32;
    for c in &mut centroid {
        *c /= n;
    }

    centroid
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_purity_gate() {
        let config = PhraseDiscoveryConfig::default();
        let mut pipeline = PhraseDiscoveryPipeline::new(config);

        // Should pass: biological sound
        let mut features = vec![0.0f32; 112];
        features[23] = 10.0; // HNR = 10dB
        features[16] = 0.3; // Flatness = 0.3

        let seg = SegmentFeatures {
            start_ms: 0.0,
            end_ms: 100.0, // Duration = 100ms
            features,
        };

        assert!(pipeline.passes_purity_gate(&seg));

        // Should fail: noise (low HNR)
        let mut noise_features = vec![0.0f32; 112];
        noise_features[23] = 1.0; // HNR = 1dB (too low)
        noise_features[16] = 0.3;

        let noise_seg = SegmentFeatures {
            start_ms: 0.0,
            end_ms: 100.0,
            features: noise_features,
        };

        assert!(!pipeline.passes_purity_gate(&noise_seg));
    }

    #[test]
    fn test_ase_distance() {
        // Identical features = 0 distance
        let a = Vector112D::from_array(&vec![1.0f32; 112]);
        let b = Vector112D::from_array(&vec![1.0f32; 112]);
        let dist = a.distance_to(&b);
        assert!(dist < 0.001);

        // Different features > 0 distance
        let c = Vector112D::from_array(&vec![0.0f32; 112]);
        let d = Vector112D::from_array(&vec![2.0f32; 112]);
        let dist2 = c.distance_to(&d);
        assert!(dist2 > 0.0);
    }

    #[test]
    fn test_zipf_r2() {
        let config = PhraseDiscoveryConfig::default();
        let pipeline = PhraseDiscoveryPipeline::new(config);

        // Perfect Zipf distribution
        let mut sizes = HashMap::new();
        sizes.insert(0, 100);
        sizes.insert(1, 50);
        sizes.insert(2, 33);
        sizes.insert(3, 25);
        sizes.insert(4, 20);

        let r2 = pipeline.compute_zipf_r2(&sizes);
        assert!(
            r2 > 0.9,
            "Zipf R² should be > 0.9 for perfect Zipf, got {}",
            r2
        );
    }

    #[test]
    fn test_discovery_pipeline() {
        let config = PhraseDiscoveryConfig::default();
        let mut pipeline = PhraseDiscoveryPipeline::new(config);

        // Create test segments
        let segments: Vec<SegmentFeatures> = (0..10)
            .map(|i| {
                let mut features = vec![0.0f32; 112];
                features[23] = 10.0; // HNR
                features[16] = 0.3; // Flatness
                features[0] = i as f32; // Unique ID
                SegmentFeatures {
                    start_ms: i as f32 * 100.0,
                    end_ms: i as f32 * 100.0 + 50.0,
                    features,
                }
            })
            .collect();

        let result = pipeline.discover(&segments);

        assert!(result.phrases.len() > 0);
        assert!(result.metrics.vocab_size > 0 || result.phrases.is_empty());
    }
}
