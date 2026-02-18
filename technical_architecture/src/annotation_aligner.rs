//! Annotation Aligner: Human-Guided Context Discovery
//!
//! This module bridges the gap between "Fuzzy Human Timestamps" and "Precise
//! Dynamic Boundaries" discovered by the segmentation engine.
//!
//! Strategy: "Anchor and Propagate"
//! ─────────────────────────────────
//! 1. Input: Audio + Human Annotation File (Raven, Audacity, CSV)
//! 2. Discovery: Run Dynamic Segmentation to find precise boundaries
//! 3. Alignment: Match discovered phrases to human annotations (Time Overlap)
//! 4. Labeling: Assign the human label (Context) to the Phrase Type
//! 5. Propagation: Use Acoustic Similarity to find "Anchored Types" in unlabeled data
//!
//! This solves the "Cold Start Problem" - transitioning from outputting
//! "Type_1" (Acoustic ID) to "Alarm_Call" (Semantic ID).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Human Annotation Types
// =============================================================================

/// Represents a human annotation from tools like Raven, Audacity, or CSV
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanAnnotation {
    /// Start time in milliseconds
    pub start_ms: f32,
    /// End time in milliseconds
    pub end_ms: f32,
    /// Primary label (e.g., "Alarm", "Contact", "Song")
    pub label: String,
    /// Optional context (e.g., "Predator Present", "Feeding")
    #[serde(default)]
    pub context: String,
    /// Optional confidence from annotator (0.0-1.0)
    #[serde(default = "default_confidence")]
    pub confidence: f32,
    /// Optional annotator ID for tracking
    #[serde(default)]
    pub annotator_id: String,
}

fn default_confidence() -> f32 {
    1.0
}

/// Format of annotation file
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnnotationFormat {
    /// Raven Pro selection table
    Raven,
    /// Audacity labels (tab-separated)
    Audacity,
    /// CSV with columns: start, end, label
    Csv,
    /// JSON array
    Json,
}

// =============================================================================
// Labeled Phrase Candidate
// =============================================================================

/// A phrase candidate with human-assigned semantic label
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabeledPhraseCandidate {
    /// Index into original phrase candidates
    pub candidate_idx: usize,
    /// Start time in milliseconds
    pub start_ms: f32,
    /// End time in milliseconds
    pub end_ms: f32,
    /// Duration in milliseconds
    pub duration_ms: f32,
    /// Human-assigned label (or "Unknown")
    pub label: String,
    /// Human-assigned context
    pub context: String,
    /// Overlap confidence (IoU with annotation)
    pub overlap_confidence: f32,
    /// Original annotation confidence
    pub annotation_confidence: f32,
    /// Combined confidence score
    pub combined_confidence: f32,
    /// Which annotation this was matched to (-1 if none)
    pub annotation_idx: isize,
}

// =============================================================================
// Annotation Aligner
// =============================================================================

/// Configuration for annotation alignment
#[derive(Debug, Clone)]
pub struct AnnotationAlignerConfig {
    /// Minimum overlap threshold (IoU) to assign a label
    pub overlap_threshold: f32,
    /// Minimum overlap in ms (for short annotations)
    pub min_overlap_ms: f32,
    /// Whether to use IoU or simple overlap ratio
    pub use_iou: bool,
    /// Weight for annotation confidence in combined score
    pub annotation_weight: f32,
}

impl Default for AnnotationAlignerConfig {
    fn default() -> Self {
        Self {
            overlap_threshold: 0.3,  // 30% overlap required
            min_overlap_ms: 10.0,    // At least 10ms overlap
            use_iou: true,           // Use Intersection over Union
            annotation_weight: 0.3,  // 30% weight to annotation confidence
        }
    }
}

/// Aligns dynamic phrases to human annotations using time overlap
pub struct AnnotationAligner {
    config: AnnotationAlignerConfig,
}

impl AnnotationAligner {
    /// Create new aligner with default configuration
    pub fn new() -> Self {
        Self {
            config: AnnotationAlignerConfig::default(),
        }
    }

    /// Create aligner with custom configuration
    pub fn with_config(config: AnnotationAlignerConfig) -> Self {
        Self { config }
    }

    /// Align phrase candidates to human annotations
    ///
    /// For each candidate, finds the best matching annotation based on
    /// temporal overlap. Returns labeled candidates with confidence scores.
    pub fn align(
        &self,
        candidates: &[PhraseCandidateForAlignment],
        annotations: &[HumanAnnotation],
    ) -> Vec<LabeledPhraseCandidate> {
        let mut labeled_candidates = Vec::with_capacity(candidates.len());

        for (idx, candidate) in candidates.iter().enumerate() {
            let mut best_match: Option<(&HumanAnnotation, f32, usize)> = None;

            for (ann_idx, annotation) in annotations.iter().enumerate() {
                // Calculate overlap (IoU or simple ratio)
                let overlap = if self.config.use_iou {
                    Self::calculate_iou(
                        candidate.start_ms, candidate.end_ms,
                        annotation.start_ms, annotation.end_ms,
                    )
                } else {
                    Self::calculate_overlap_ratio(
                        candidate.start_ms, candidate.end_ms,
                        annotation.start_ms, annotation.end_ms,
                    )
                };

                // Check if overlap meets thresholds
                let overlap_ms = (candidate.end_ms.min(annotation.end_ms)
                    - candidate.start_ms.max(annotation.start_ms)).max(0.0);

                if overlap >= self.config.overlap_threshold
                    && overlap_ms >= self.config.min_overlap_ms
                {
                    // Keep the best matching annotation
                    if best_match.is_none() || overlap > best_match.unwrap().1 {
                        best_match = Some((annotation, overlap, ann_idx));
                    }
                }
            }

            // Construct the labeled candidate
            let labeled = if let Some((annotation, overlap, ann_idx)) = best_match {
                let combined = (1.0 - self.config.annotation_weight) * overlap
                    + self.config.annotation_weight * annotation.confidence;

                LabeledPhraseCandidate {
                    candidate_idx: idx,
                    start_ms: candidate.start_ms,
                    end_ms: candidate.end_ms,
                    duration_ms: candidate.end_ms - candidate.start_ms,
                    label: annotation.label.clone(),
                    context: annotation.context.clone(),
                    overlap_confidence: overlap,
                    annotation_confidence: annotation.confidence,
                    combined_confidence: combined,
                    annotation_idx: ann_idx as isize,
                }
            } else {
                LabeledPhraseCandidate {
                    candidate_idx: idx,
                    start_ms: candidate.start_ms,
                    end_ms: candidate.end_ms,
                    duration_ms: candidate.end_ms - candidate.start_ms,
                    label: "Unknown".to_string(),
                    context: String::new(),
                    overlap_confidence: 0.0,
                    annotation_confidence: 0.0,
                    combined_confidence: 0.0,
                    annotation_idx: -1,
                }
            };

            labeled_candidates.push(labeled);
        }

        labeled_candidates
    }

    /// Calculate Intersection over Union (IoU)
    fn calculate_iou(s1: f32, e1: f32, s2: f32, e2: f32) -> f32 {
        let intersection = (e1.min(e2) - s1.max(s2)).max(0.0);
        let union = (e1 - s1) + (e2 - s2) - intersection;
        if union <= 0.0 { 0.0 } else { intersection / union }
    }

    /// Calculate simple overlap ratio (intersection / candidate duration)
    fn calculate_overlap_ratio(s1: f32, e1: f32, s2: f32, e2: f32) -> f32 {
        let intersection = (e1.min(e2) - s1.max(s2)).max(0.0);
        let candidate_duration = e1 - s1;
        if candidate_duration <= 0.0 { 0.0 } else { intersection / candidate_duration }
    }

    /// Parse annotations from various file formats
    pub fn parse_annotations(
        &self,
        content: &str,
        format: AnnotationFormat,
    ) -> Result<Vec<HumanAnnotation>, AnnotationParseError> {
        match format {
            AnnotationFormat::Raven => self.parse_raven(content),
            AnnotationFormat::Audacity => self.parse_audacity(content),
            AnnotationFormat::Csv => self.parse_csv(content),
            AnnotationFormat::Json => self.parse_json(content),
        }
    }

    /// Parse Raven Pro selection table format
    fn parse_raven(&self, content: &str) -> Result<Vec<HumanAnnotation>, AnnotationParseError> {
        let mut annotations = Vec::new();
        let mut lines = content.lines();

        // Skip header line
        let _header = lines.next().ok_or(AnnotationParseError::EmptyFile)?;

        for line in lines {
            let fields: Vec<&str> = line.split('\t').collect();
            if fields.len() >= 5 {
                let start_s: f32 = fields[3].parse()
                    .map_err(|_| AnnotationParseError::ParseError("Invalid start time".into()))?;
                let end_s: f32 = fields[4].parse()
                    .map_err(|_| AnnotationParseError::ParseError("Invalid end time".into()))?;

                annotations.push(HumanAnnotation {
                    start_ms: start_s * 1000.0,
                    end_ms: end_s * 1000.0,
                    label: fields.get(7).unwrap_or(&"").to_string(),
                    context: String::new(),
                    confidence: 1.0,
                    annotator_id: String::new(),
                });
            }
        }

        Ok(annotations)
    }

    /// Parse Audacity label format (tab-separated: start, end, label)
    fn parse_audacity(&self, content: &str) -> Result<Vec<HumanAnnotation>, AnnotationParseError> {
        let mut annotations = Vec::new();

        for line in content.lines() {
            let fields: Vec<&str> = line.split('\t').collect();
            if fields.len() >= 3 {
                let start_s: f32 = fields[0].parse()
                    .map_err(|_| AnnotationParseError::ParseError("Invalid start time".into()))?;
                let end_s: f32 = fields[1].parse()
                    .map_err(|_| AnnotationParseError::ParseError("Invalid end time".into()))?;

                // Parse label and optional context (format: "label|context")
                let (label, context) = if fields[2].contains('|') {
                    let parts: Vec<&str> = fields[2].splitn(2, '|').collect();
                    (parts[0].to_string(), parts[1].to_string())
                } else {
                    (fields[2].to_string(), String::new())
                };

                annotations.push(HumanAnnotation {
                    start_ms: start_s * 1000.0,
                    end_ms: end_s * 1000.0,
                    label,
                    context,
                    confidence: 1.0,
                    annotator_id: String::new(),
                });
            }
        }

        Ok(annotations)
    }

    /// Parse CSV format (start_ms, end_ms, label, [context])
    fn parse_csv(&self, content: &str) -> Result<Vec<HumanAnnotation>, AnnotationParseError> {
        let mut annotations = Vec::new();
        let mut lines = content.lines();

        // Skip header if present
        let first_line = lines.next().ok_or(AnnotationParseError::EmptyFile)?;
        let first_fields: Vec<&str> = first_line.split(',').collect();

        // Check if first line is header
        let start_idx = if first_fields[0].parse::<f32>().is_err() { 1 } else { 0 };

        // Process first line if it wasn't a header
        if start_idx == 0 {
            if let Ok(ann) = Self::parse_csv_line(&first_fields) {
                annotations.push(ann);
            }
        }

        for line in lines {
            let fields: Vec<&str> = line.split(',').collect();
            if let Ok(ann) = Self::parse_csv_line(&fields) {
                annotations.push(ann);
            }
        }

        Ok(annotations)
    }

    fn parse_csv_line(fields: &[&str]) -> Result<HumanAnnotation, AnnotationParseError> {
        if fields.len() < 3 {
            return Err(AnnotationParseError::ParseError("Not enough fields".into()));
        }

        let start: f32 = fields[0].parse()
            .map_err(|_| AnnotationParseError::ParseError("Invalid start time".into()))?;
        let end: f32 = fields[1].parse()
            .map_err(|_| AnnotationParseError::ParseError("Invalid end time".into()))?;

        Ok(HumanAnnotation {
            start_ms: start,
            end_ms: end,
            label: fields[2].trim().to_string(),
            context: fields.get(3).map(|s| s.trim().to_string()).unwrap_or_default(),
            confidence: fields.get(4)
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(1.0),
            annotator_id: fields.get(5)
                .map(|s| s.trim().to_string())
                .unwrap_or_default(),
        })
    }

    /// Parse JSON format
    fn parse_json(&self, content: &str) -> Result<Vec<HumanAnnotation>, AnnotationParseError> {
        serde_json::from_str(content)
            .map_err(|e| AnnotationParseError::ParseError(format!("JSON parse error: {}", e)))
    }
}

impl Default for AnnotationAligner {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Phrase Candidate for Alignment
// =============================================================================

/// Simplified phrase candidate for alignment (without full feature vector)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhraseCandidateForAlignment {
    pub start_ms: f32,
    pub end_ms: f32,
    pub sample_rate: u32,
    pub start_sample: usize,
    pub end_sample: usize,
}

impl PhraseCandidateForAlignment {
    pub fn from_bounds(start_ms: f32, end_ms: f32, sample_rate: u32) -> Self {
        Self {
            start_ms,
            end_ms,
            sample_rate,
            start_sample: ((start_ms / 1000.0) * sample_rate as f32) as usize,
            end_sample: ((end_ms / 1000.0) * sample_rate as f32) as usize,
        }
    }
}

// =============================================================================
// Parse Error
// =============================================================================

#[derive(Debug, Clone)]
pub enum AnnotationParseError {
    EmptyFile,
    ParseError(String),
}

impl std::fmt::Display for AnnotationParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyFile => write!(f, "Empty annotation file"),
            Self::ParseError(msg) => write!(f, "Parse error: {}", msg),
        }
    }
}

impl std::error::Error for AnnotationParseError {}

// =============================================================================
// Semantic Phrase Dictionary
// =============================================================================

/// Maps acoustic phrase types to semantic labels with confidence scores
///
/// This is the output of the "Anchor and Propagate" pipeline:
/// - Cluster discovered phrases into Types
/// - Count label occurrences per cluster
/// - Convert to probability distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticPhraseDictionary {
    /// Map: Phrase Type ID -> (Label -> Probability)
    /// e.g., "Type_1" -> {"Alarm": 0.90, "Contact": 0.10}
    pub type_to_labels: HashMap<String, HashMap<String, f32>>,

    /// Map: Phrase Type ID -> (Context -> Probability)
    pub type_to_contexts: HashMap<String, HashMap<String, f32>>,

    /// Map: Label -> [Phrase Type IDs]
    /// For reverse lookup: "What types are associated with 'Alarm'?"
    pub label_to_types: HashMap<String, Vec<String>>,

    /// Centroid features for each type (for similarity matching)
    pub type_centroids: HashMap<String, Vec<f32>>,

    /// Statistics
    pub total_phrases: usize,
    pub total_types: usize,
    pub total_labels: usize,
}

impl SemanticPhraseDictionary {
    /// Create empty dictionary
    pub fn new() -> Self {
        Self {
            type_to_labels: HashMap::new(),
            type_to_contexts: HashMap::new(),
            label_to_types: HashMap::new(),
            type_centroids: HashMap::new(),
            total_phrases: 0,
            total_types: 0,
            total_labels: 0,
        }
    }

    /// Build dictionary from aligned candidates and cluster assignments
    ///
    /// # Arguments
    /// * `labeled_candidates` - Phrase candidates with human labels
    /// * `cluster_assignments` - (candidate_idx, type_id) pairs
    /// * `features` - 45D feature vectors for centroid computation
    pub fn build(
        labeled_candidates: &[LabeledPhraseCandidate],
        cluster_assignments: &[(usize, String)],
        features: &[Vec<f32>],
    ) -> Self {
        let mut dict = Self::new();
        dict.total_phrases = labeled_candidates.len();

        // Count label occurrences per type
        let mut label_counts: HashMap<String, HashMap<String, usize>> = HashMap::new();
        let mut context_counts: HashMap<String, HashMap<String, usize>> = HashMap::new();
        let mut type_feature_sums: HashMap<String, Vec<f32>> = HashMap::new();
        let mut type_counts: HashMap<String, usize> = HashMap::new();

        for (candidate_idx, type_id) in cluster_assignments {
            if *candidate_idx >= labeled_candidates.len() {
                continue;
            }

            let labeled = &labeled_candidates[*candidate_idx];

            // Count labels (excluding "Unknown")
            if labeled.label != "Unknown" {
                *label_counts
                    .entry(type_id.clone())
                    .or_default()
                    .entry(labeled.label.clone())
                    .or_insert(0) += 1;
            }

            // Count contexts
            if !labeled.context.is_empty() {
                *context_counts
                    .entry(type_id.clone())
                    .or_default()
                    .entry(labeled.context.clone())
                    .or_insert(0) += 1;
            }

            // Accumulate features for centroid
            if *candidate_idx < features.len() {
                let sum = type_feature_sums.entry(type_id.clone()).or_insert_with(|| {
                    vec![0.0; features[*candidate_idx].len()]
                });
                for (i, &f) in features[*candidate_idx].iter().enumerate() {
                    sum[i] += f;
                }
                *type_counts.entry(type_id.clone()).or_insert(0) += 1;
            }
        }

        // Convert counts to probabilities
        let all_labels: std::collections::HashSet<_> = label_counts.values()
            .flat_map(|m| m.keys().cloned())
            .collect();

        for (type_id, counts) in &label_counts {
            let total: usize = counts.values().sum();
            if total > 0 {
                let probs: HashMap<String, f32> = counts.iter()
                    .map(|(label, &count)| (label.clone(), count as f32 / total as f32))
                    .collect();
                dict.type_to_labels.insert(type_id.clone(), probs);
            }
        }

        for (type_id, counts) in &context_counts {
            let total: usize = counts.values().sum();
            if total > 0 {
                let probs: HashMap<String, f32> = counts.iter()
                    .map(|(ctx, &count)| (ctx.clone(), count as f32 / total as f32))
                    .collect();
                dict.type_to_contexts.insert(type_id.clone(), probs);
            }
        }

        // Compute centroids
        for (type_id, sum) in type_feature_sums {
            if let Some(&count) = type_counts.get(&type_id) {
                if count > 0 {
                    let centroid: Vec<f32> = sum.iter()
                        .map(|&s| s / count as f32)
                        .collect();
                    dict.type_centroids.insert(type_id, centroid);
                }
            }
        }

        // Build reverse lookup
        for (type_id, labels) in &dict.type_to_labels {
            for label in labels.keys() {
                dict.label_to_types
                    .entry(label.clone())
                    .or_default()
                    .push(type_id.clone());
            }
        }

        dict.total_types = dict.type_centroids.len();
        dict.total_labels = all_labels.len();

        dict
    }

    /// Get the most likely label for a phrase type
    pub fn get_primary_label(&self, type_id: &str) -> Option<(&String, f32)> {
        self.type_to_labels
            .get(type_id)
            .and_then(|labels| {
                labels.iter()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(k, &v)| (k, v))
            })
    }

    /// Get all types associated with a semantic label
    pub fn get_types_for_label(&self, label: &str) -> Option<&Vec<String>> {
        self.label_to_types.get(label)
    }

    /// Get semantic description for a type (human-readable)
    pub fn describe_type(&self, type_id: &str) -> String {
        let label_part = self.get_primary_label(type_id)
            .map(|(l, p)| format!("{} ({:.0}%)", l, p * 100.0))
            .unwrap_or_else(|| "Unknown".to_string());

        let context_part = self.type_to_contexts.get(type_id)
            .and_then(|ctxs| {
                ctxs.iter()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(c, p)| format!(" [{}]", c))
            })
            .unwrap_or_default();

        format!("{}{}", label_part, context_part)
    }

    /// Serialize to JSON for storage
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

impl Default for SemanticPhraseDictionary {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iou_calculation() {
        // Perfect overlap
        let iou = AnnotationAligner::calculate_iou(0.0, 100.0, 0.0, 100.0);
        assert!((iou - 1.0).abs() < 0.001);

        // Partial overlap
        let iou = AnnotationAligner::calculate_iou(0.0, 100.0, 50.0, 150.0);
        assert!((iou - 0.333).abs() < 0.01);

        // No overlap
        let iou = AnnotationAligner::calculate_iou(0.0, 100.0, 200.0, 300.0);
        assert!((iou - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_alignment() {
        let aligner = AnnotationAligner::new();

        let candidates = vec![
            PhraseCandidateForAlignment::from_bounds(0.0, 100.0, 48000),
            PhraseCandidateForAlignment::from_bounds(150.0, 250.0, 48000),
            PhraseCandidateForAlignment::from_bounds(300.0, 400.0, 48000),
        ];

        let annotations = vec![
            HumanAnnotation {
                start_ms: 0.0,
                end_ms: 120.0,
                label: "Song".to_string(),
                context: "Territory".to_string(),
                confidence: 1.0,
                annotator_id: String::new(),
            },
            HumanAnnotation {
                start_ms: 280.0,
                end_ms: 420.0,
                label: "Alarm".to_string(),
                context: "Predator".to_string(),
                confidence: 0.9,
                annotator_id: String::new(),
            },
        ];

        let labeled = aligner.align(&candidates, &annotations);

        assert_eq!(labeled.len(), 3);
        assert_eq!(labeled[0].label, "Song");
        assert_eq!(labeled[0].context, "Territory");
        assert_eq!(labeled[1].label, "Unknown");
        assert_eq!(labeled[2].label, "Alarm");
        assert_eq!(labeled[2].context, "Predator");
    }

    #[test]
    fn test_parse_audacity() {
        let aligner = AnnotationAligner::new();
        let content = "0.0\t1.0\tSong|Territory\n1.5\t2.5\tAlarm|Predator\n";

        let annotations = aligner.parse_annotations(content, AnnotationFormat::Audacity).unwrap();

        assert_eq!(annotations.len(), 2);
        assert_eq!(annotations[0].label, "Song");
        assert_eq!(annotations[0].context, "Territory");
        assert_eq!(annotations[0].start_ms, 0.0);
        assert_eq!(annotations[0].end_ms, 1000.0);
    }

    #[test]
    fn test_semantic_dictionary() {
        let mut dict = SemanticPhraseDictionary::new();

        // Manually add some data
        let mut labels = HashMap::new();
        labels.insert("Alarm".to_string(), 0.9);
        labels.insert("Contact".to_string(), 0.1);
        dict.type_to_labels.insert("Type_1".to_string(), labels);

        let primary = dict.get_primary_label("Type_1");
        assert!(primary.is_some());
        let (label, prob) = primary.unwrap();
        assert_eq!(label, "Alarm");
        assert!((prob - 0.9).abs() < 0.001);

        assert_eq!(dict.describe_type("Type_1"), "Alarm (90%)");
    }

    #[test]
    fn test_dictionary_build() {
        let labeled = vec![
            LabeledPhraseCandidate {
                candidate_idx: 0,
                start_ms: 0.0,
                end_ms: 100.0,
                duration_ms: 100.0,
                label: "Song".to_string(),
                context: "Territory".to_string(),
                overlap_confidence: 0.9,
                annotation_confidence: 1.0,
                combined_confidence: 0.93,
                annotation_idx: 0,
            },
            LabeledPhraseCandidate {
                candidate_idx: 1,
                start_ms: 150.0,
                end_ms: 250.0,
                duration_ms: 100.0,
                label: "Song".to_string(),
                context: "Territory".to_string(),
                overlap_confidence: 0.85,
                annotation_confidence: 1.0,
                combined_confidence: 0.895,
                annotation_idx: 0,
            },
            LabeledPhraseCandidate {
                candidate_idx: 2,
                start_ms: 300.0,
                end_ms: 400.0,
                duration_ms: 100.0,
                label: "Alarm".to_string(),
                context: "Predator".to_string(),
                overlap_confidence: 0.95,
                annotation_confidence: 0.9,
                combined_confidence: 0.93,
                annotation_idx: 1,
            },
        ];

        let clusters = vec![
            (0, "Type_A".to_string()),
            (1, "Type_A".to_string()),
            (2, "Type_B".to_string()),
        ];

        let features = vec![
            vec![0.1, 0.2, 0.3],
            vec![0.15, 0.25, 0.35],
            vec![0.8, 0.9, 1.0],
        ];

        let dict = SemanticPhraseDictionary::build(&labeled, &clusters, &features);

        assert_eq!(dict.total_phrases, 3);
        assert_eq!(dict.total_types, 2);
        assert_eq!(dict.total_labels, 2);

        // Type_A should be 100% Song
        let primary_a = dict.get_primary_label("Type_A").unwrap();
        assert_eq!(primary_a.0, "Song");
        assert!((primary_a.1 - 1.0).abs() < 0.001);

        // Type_B should be 100% Alarm
        let primary_b = dict.get_primary_label("Type_B").unwrap();
        assert_eq!(primary_b.0, "Alarm");
    }
}
