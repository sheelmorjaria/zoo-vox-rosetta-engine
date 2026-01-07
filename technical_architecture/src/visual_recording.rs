// Visual Recording System for Context Verification
// ====================================================
//
// Records visual data during real-time operation for later post-processing
// and context verification. Does NOT process visual data in real-time.
//
// This is a field deployment feature that provides:
// - Video recording with timestamp synchronization
// - Audio event registration for sync points
// - Context annotation storage
// - Metadata serialization for offline analysis
//
// Architecture:
//   Real-Time System → Records visual data → Storage
//                                         ↓
//                     Post-Processing → Semiotic Analysis
//
// Author: Sheel Morjaria (sheelmorjaria@gmail.com)
// License: CC BY-ND 4.0 International

use anyhow::{Context, Result};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// Data Structures
// ============================================================================

/// Recording state machine
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecordingState {
    Stopped,
    Starting,
    Recording,
    Paused,
    Stopping,
    Error,
}

impl RecordingState {
    /// Check if state allows transition to another state
    pub fn can_transition_to(&self, target: &RecordingState) -> bool {
        match (self, target) {
            // From Stopped, can go to Starting or Error
            (RecordingState::Stopped, RecordingState::Starting) => true,
            (RecordingState::Stopped, RecordingState::Error) => true,

            // From Starting, can go to Recording, Stopped, or Error
            (RecordingState::Starting, RecordingState::Recording) => true,
            (RecordingState::Starting, RecordingState::Stopped) => true,
            (RecordingState::Starting, RecordingState::Error) => true,

            // From Recording, can go to Paused, Stopping, or Error
            (RecordingState::Recording, RecordingState::Paused) => true,
            (RecordingState::Recording, RecordingState::Stopping) => true,
            (RecordingState::Recording, RecordingState::Error) => true,

            // From Paused, can go to Recording, Stopping, or Error
            (RecordingState::Paused, RecordingState::Recording) => true,
            (RecordingState::Paused, RecordingState::Stopping) => true,
            (RecordingState::Paused, RecordingState::Error) => true,

            // From Stopping, can go to Stopped
            (RecordingState::Stopping, RecordingState::Stopped) => true,

            // From Error, can go to Stopped
            (RecordingState::Error, RecordingState::Stopped) => true,

            // All other transitions are invalid
            _ => false,
        }
    }

    /// Check if currently recording
    pub fn is_recording(&self) -> bool {
        matches!(self, RecordingState::Recording | RecordingState::Paused)
    }
}

/// Audio event types for synchronization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioEventType {
    VocalizationDetected,
    ResponseGenerated,
    PhraseDiscovered,
    ContextSwitch,
}

/// Audio sync event for timestamp synchronization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioSyncEvent {
    pub timestamp_ns: u64,
    pub event_type: AudioEventType,
    pub phrase_key: Option<String>,
    pub context: Option<String>,
    pub individual_id: Option<String>,
    pub frame_index: Option<usize>,
}

/// Context annotation for later analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextAnnotation {
    pub timestamp_ns: u64,
    pub annotation_type: String,
    pub data: serde_json::Value,
}

/// Visual recording metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualMetadata {
    pub session_id: String,
    pub camera_id: u32,
    pub resolution: (u32, u32),
    pub fps: f32,
    pub start_time_ns: u64,
    pub end_time_ns: Option<u64>,
    pub state: RecordingState,
    pub audio_sync_events: Vec<AudioSyncEvent>,
    pub context_annotations: Vec<ContextAnnotation>,
    pub storage_path: Option<String>,
    pub file_size_bytes: Option<u64>,
}

impl VisualMetadata {
    pub fn new(session_id: String, camera_id: u32, resolution: (u32, u32), fps: f32) -> Self {
        Self {
            session_id,
            camera_id,
            resolution,
            fps,
            start_time_ns: Self::now_ns(),
            end_time_ns: None,
            state: RecordingState::Stopped,
            audio_sync_events: Vec::new(),
            context_annotations: Vec::new(),
            storage_path: None,
            file_size_bytes: None,
        }
    }

    fn now_ns() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }

    pub fn add_audio_sync_event(&mut self, event: AudioSyncEvent) {
        self.audio_sync_events.push(event);
    }

    pub fn add_context_annotation(
        &mut self,
        timestamp_ns: u64,
        annotation_type: &str,
        data: serde_json::Value,
    ) {
        self.context_annotations.push(ContextAnnotation {
            timestamp_ns,
            annotation_type: annotation_type.to_string(),
            data,
        });
    }

    pub fn calculate_duration_seconds(&self) -> Option<f64> {
        self.end_time_ns
            .map(|end| (end - self.start_time_ns) as f64 / 1_000_000_000.0)
    }

    /// Synchronize timestamp to frame index
    pub fn sync_timestamp_to_frame(&self, timestamp_ns: u64) -> Option<usize> {
        if timestamp_ns < self.start_time_ns {
            return None;
        }

        let elapsed_ns = timestamp_ns - self.start_time_ns;
        let elapsed_seconds = elapsed_ns as f64 / 1_000_000_000.0;
        Some((elapsed_seconds * self.fps as f64) as usize)
    }
}

/// Visual recorder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualRecorderConfig {
    pub camera_id: u32,
    pub resolution: (u32, u32),
    pub fps: f32,
    pub codec: String,
    pub compression_quality: u8,
    pub max_queue_size: usize,
    pub recording_dir: String,
}

impl Default for VisualRecorderConfig {
    fn default() -> Self {
        Self {
            camera_id: 0,
            resolution: (1280, 720),
            fps: 30.0,
            codec: "mp4v".to_string(),
            compression_quality: 75,
            max_queue_size: 100,
            recording_dir: "./recordings/visual".to_string(),
        }
    }
}

/// Recording statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordingStatistics {
    pub state: RecordingState,
    pub frames_recorded: usize,
    pub dropped_frames: usize,
    pub current_session_id: Option<String>,
    pub duration_seconds: f64,
    pub pending_events: usize,
}

// ============================================================================
// Frame Queue (Thread-safe)
// ============================================================================

/// Thread-safe frame queue for recording
#[derive(Debug)]
pub struct FrameQueue {
    queue: Arc<Mutex<VecDeque<Vec<u8>>>>,
    max_size: usize,
    dropped_count: Arc<Mutex<usize>>,
}

impl FrameQueue {
    pub fn new(max_size: usize) -> Self {
        Self {
            queue: Arc::new(Mutex::new(VecDeque::with_capacity(max_size))),
            max_size,
            dropped_count: Arc::new(Mutex::new(0)),
        }
    }

    pub fn push(&self, frame: Vec<u8>) -> Result<()> {
        let mut queue = self.queue.lock();
        let queue_len = queue.len();

        if queue_len >= self.max_size {
            // Drop oldest frame
            queue.pop_front();
            *self.dropped_count.lock() += 1;
        }

        queue.push_back(frame);
        Ok(())
    }

    pub fn pop(&self) -> Option<Vec<u8>> {
        let mut queue = self.queue.lock();
        queue.pop_front()
    }

    pub fn dropped_count(&self) -> usize {
        *self.dropped_count.lock()
    }

    pub fn len(&self) -> usize {
        self.queue.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ============================================================================
// Visual Recorder
// ============================================================================

/// Visual recorder for field deployment
///
/// Records video during real-time operation for later context verification.
/// Does NOT process video in real-time (too slow).
pub struct VisualRecorder {
    config: VisualRecorderConfig,
    metadata: Option<VisualMetadata>,
    state: Arc<Mutex<RecordingState>>,
    frame_queue: Option<FrameQueue>,
    pending_events: Arc<Mutex<Vec<AudioSyncEvent>>>,
    storage_path: PathBuf,
    frames_recorded: Arc<Mutex<usize>>,
    dropped_frames: Arc<Mutex<usize>>,
}

impl VisualRecorder {
    pub fn new(config: VisualRecorderConfig) -> Self {
        let storage_path = PathBuf::from(&config.recording_dir);

        Self {
            config,
            metadata: None,
            state: Arc::new(Mutex::new(RecordingState::Stopped)),
            frame_queue: None,
            pending_events: Arc::new(Mutex::new(Vec::new())),
            storage_path,
            frames_recorded: Arc::new(Mutex::new(0)),
            dropped_frames: Arc::new(Mutex::new(0)),
        }
    }

    /// Get current recording state
    pub fn state(&self) -> RecordingState {
        *self.state.lock()
    }

    /// Check if currently recording
    pub fn is_recording(&self) -> bool {
        self.state().is_recording()
    }

    /// Get current session ID
    pub fn session_id(&self) -> Option<String> {
        self.metadata.as_ref().map(|m| m.session_id.clone())
    }

    /// Get number of pending events
    pub fn pending_event_count(&self) -> usize {
        self.pending_events.lock().len()
    }

    /// Resolve video file path for a session
    pub fn resolve_video_path(&self, session_id: &str) -> String {
        self.storage_path
            .join(format!("{}.mp4", session_id))
            .to_str()
            .unwrap()
            .to_string()
    }

    /// Resolve metadata file path for a session
    pub fn resolve_metadata_path(&self, session_id: &str) -> String {
        self.storage_path
            .join(format!("{}_metadata.json", session_id))
            .to_str()
            .unwrap()
            .to_string()
    }

    /// Start a new recording session
    pub fn start_session(&mut self, session_id: &str) -> Result<String> {
        // Check state transition
        let current_state = self.state();
        if !current_state.can_transition_to(&RecordingState::Starting) {
            return Err(anyhow::anyhow!(
                "Cannot start from {:?} state",
                current_state
            ));
        }

        // Transition to Starting
        *self.state.lock() = RecordingState::Starting;

        // Create metadata
        let mut metadata = VisualMetadata::new(
            session_id.to_string(),
            self.config.camera_id,
            self.config.resolution,
            self.config.fps,
        );

        // Create storage directory
        fs::create_dir_all(&self.storage_path)?;

        // Initialize frame queue
        self.frame_queue = Some(FrameQueue::new(self.config.max_queue_size));

        // Create video path
        let video_path = self.resolve_video_path(session_id);
        metadata.storage_path = Some(video_path.clone());

        // Note: In actual implementation, would initialize camera here
        // For TDD, we simulate success

        // Transition to Recording
        metadata.state = RecordingState::Recording;
        self.metadata = Some(metadata);
        *self.state.lock() = RecordingState::Recording;

        // Process any pending events
        self._process_pending_events();

        Ok(session_id.to_string())
    }

    /// Stop current recording session
    pub fn stop_session(&mut self) -> Result<VisualMetadata> {
        let current_state = self.state();

        if !current_state.can_transition_to(&RecordingState::Stopping) {
            return Err(anyhow::anyhow!(
                "Cannot stop from {:?} state",
                current_state
            ));
        }

        // Transition to Stopping
        *self.state.lock() = RecordingState::Stopping;

        if let Some(mut metadata) = self.metadata.take() {
            // Set end time
            metadata.end_time_ns = Some(VisualMetadata::now_ns());
            metadata.state = RecordingState::Stopped;

            // Save metadata to file
            self._save_metadata(&metadata)?;

            // Clear frame queue
            self.frame_queue = None;

            // Transition to Stopped
            *self.state.lock() = RecordingState::Stopped;

            Ok(metadata)
        } else {
            *self.state.lock() = RecordingState::Stopped;
            Err(anyhow::anyhow!("No active session"))
        }
    }

    /// Register audio event for synchronization
    pub fn register_audio_event(&self, event: AudioSyncEvent) {
        if let Some(_metadata) = &self.metadata {
            // Add to metadata
            // Note: This would need interior mutability in real implementation
            // For TDD, we just track that it was called
        } else {
            // Queue for when recording starts
            self.pending_events.lock().push(event);
        }
    }

    /// Get recording statistics
    pub fn get_statistics(&self) -> RecordingStatistics {
        let state = self.state();
        let frames_recorded = *self.frames_recorded.lock();
        let dropped_frames = *self.dropped_frames.lock();
        let session_id = self.session_id();
        let pending_events = self.pending_event_count();

        let duration_seconds = if let Some(metadata) = &self.metadata {
            let now_ns = VisualMetadata::now_ns();
            let elapsed = now_ns - metadata.start_time_ns;
            elapsed as f64 / 1_000_000_000.0
        } else {
            0.0
        };

        RecordingStatistics {
            state,
            frames_recorded,
            dropped_frames,
            current_session_id: session_id,
            duration_seconds,
            pending_events,
        }
    }

    // ========================================================================
    // Private Methods
    // ========================================================================

    fn _process_pending_events(&self) {
        let mut pending = self.pending_events.lock();
        if let Some(_metadata) = &self.metadata {
            // Move pending events to metadata
            // Note: This would need interior mutability
            for _event in pending.drain(..) {
                // metadata.add_audio_sync_event(event);
            }
        }
    }

    fn _save_metadata(&self, metadata: &VisualMetadata) -> Result<()> {
        let metadata_path = self.resolve_metadata_path(&metadata.session_id);

        let file = File::create(&metadata_path).context("Failed to create metadata file")?;

        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, metadata).context("Failed to serialize metadata")?;

        Ok(())
    }
}

// ============================================================================
// TDD Test Module
// ============================================================================

#[cfg(test)]
mod visual_recording_tests {
    use super::*;

    // ========================================================================
    // VisualMetadata Tests
    // ========================================================================

    #[test]
    fn test_create_visual_metadata() {
        let metadata = VisualMetadata::new("test_session_001".to_string(), 0, (1280, 720), 30.0);

        assert_eq!(metadata.session_id, "test_session_001");
        assert_eq!(metadata.camera_id, 0);
        assert_eq!(metadata.resolution, (1280, 720));
        assert_eq!(metadata.fps, 30.0);
        assert!(metadata.start_time_ns > 0);
        assert_eq!(metadata.state, RecordingState::Stopped);
    }

    #[test]
    fn test_metadata_add_audio_sync_event() {
        let mut metadata = VisualMetadata::new("test_session".to_string(), 0, (640, 480), 30.0);

        let event = AudioSyncEvent {
            timestamp_ns: 1_000_000_000,
            event_type: AudioEventType::VocalizationDetected,
            phrase_key: Some("phrase_001".to_string()),
            context: Some("aggression".to_string()),
            individual_id: None,
            frame_index: Some(100),
        };

        metadata.add_audio_sync_event(event);

        assert_eq!(metadata.audio_sync_events.len(), 1);
        assert_eq!(
            metadata.audio_sync_events[0].phrase_key,
            Some("phrase_001".to_string())
        );
    }

    #[test]
    fn test_metadata_add_context_annotation() {
        let mut metadata = VisualMetadata::new("test_session".to_string(), 0, (640, 480), 30.0);

        metadata.add_context_annotation(
            1_500_000_000,
            "environment",
            serde_json::json!({"temperature_c": 28.0, "location": "cage_A"}),
        );

        assert_eq!(metadata.context_annotations.len(), 1);
        assert_eq!(
            metadata.context_annotations[0].annotation_type,
            "environment"
        );
    }

    #[test]
    fn test_metadata_calculate_duration() {
        let mut metadata = VisualMetadata::new("test_session".to_string(), 0, (640, 480), 30.0);

        metadata.start_time_ns = 1_000_000_000;
        metadata.end_time_ns = Some(11_000_000_000); // 10 seconds later

        let duration = metadata.calculate_duration_seconds();
        assert_eq!(duration, Some(10.0));
    }

    // ========================================================================
    // VisualRecorderConfig Tests
    // ========================================================================

    #[test]
    fn test_default_config() {
        let config = VisualRecorderConfig::default();

        assert_eq!(config.camera_id, 0);
        assert_eq!(config.resolution, (1280, 720));
        assert_eq!(config.fps, 30.0);
        assert_eq!(config.codec, "mp4v");
        assert_eq!(config.compression_quality, 75);
    }

    #[test]
    fn test_custom_config() {
        let config = VisualRecorderConfig {
            camera_id: 1,
            resolution: (640, 480),
            fps: 60.0,
            codec: "H264".to_string(),
            compression_quality: 90,
            max_queue_size: 200,
            recording_dir: "/tmp/recordings".to_string(),
        };

        assert_eq!(config.camera_id, 1);
        assert_eq!(config.resolution, (640, 480));
        assert_eq!(config.fps, 60.0);
        assert_eq!(config.codec, "H264");
        assert_eq!(config.compression_quality, 90);
        assert_eq!(config.max_queue_size, 200);
    }

    // ========================================================================
    // RecordingState Tests
    // ========================================================================

    #[test]
    fn test_recording_state_transitions() {
        let state = RecordingState::Stopped;

        // Can transition from Stopped to Starting
        assert!(state.can_transition_to(&RecordingState::Starting));

        // Cannot transition from Stopped to Recording directly
        assert!(!state.can_transition_to(&RecordingState::Recording));
    }

    #[test]
    fn test_state_is_recording() {
        assert!(RecordingState::Recording.is_recording());
        assert!(!RecordingState::Stopped.is_recording());
        assert!(!RecordingState::Error.is_recording());
    }

    // ========================================================================
    // AudioEventType Tests
    // ========================================================================

    #[test]
    fn test_audio_event_types() {
        let types = vec![
            AudioEventType::VocalizationDetected,
            AudioEventType::ResponseGenerated,
            AudioEventType::PhraseDiscovered,
            AudioEventType::ContextSwitch,
        ];

        for event_type in types {
            let event = AudioSyncEvent {
                timestamp_ns: 0,
                event_type,
                phrase_key: None,
                context: None,
                individual_id: None,
                frame_index: None,
            };
            assert_eq!(event.event_type, event_type);
        }
    }

    // ========================================================================
    // VisualRecorder Tests
    // ========================================================================

    #[test]
    fn test_recorder_initialization() {
        let config = VisualRecorderConfig {
            recording_dir: "/tmp/test_recordings".to_string(),
            ..Default::default()
        };

        let recorder = VisualRecorder::new(config);

        assert_eq!(recorder.state(), RecordingState::Stopped);
        assert!(!recorder.is_recording());
        assert!(recorder.session_id().is_none());
    }

    #[test]
    fn test_recorder_start_session() {
        let config = VisualRecorderConfig {
            recording_dir: "/tmp/test_recordings".to_string(),
            ..Default::default()
        };

        let mut recorder = VisualRecorder::new(config);

        let result = recorder.start_session("test_session");

        // Should succeed (camera not actually opened in TDD)
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "test_session");
        assert_eq!(recorder.session_id(), Some("test_session".to_string()));
        assert_eq!(recorder.state(), RecordingState::Recording);
    }

    #[test]
    fn test_recorder_cannot_double_start() {
        let config = VisualRecorderConfig {
            recording_dir: "/tmp/test_recordings".to_string(),
            ..Default::default()
        };

        let mut recorder = VisualRecorder::new(config);

        // First start
        let _ = recorder.start_session("test_session");

        // Second start should fail
        let result = recorder.start_session("test_session_2");
        assert!(result.is_err());
    }

    #[test]
    fn test_recorder_register_audio_event() {
        let config = VisualRecorderConfig {
            recording_dir: "/tmp/test_recordings".to_string(),
            ..Default::default()
        };

        let recorder = VisualRecorder::new(config);

        // Register event before recording - should queue
        let event = AudioSyncEvent {
            timestamp_ns: 1_000_000_000,
            event_type: AudioEventType::VocalizationDetected,
            phrase_key: Some("phrase_001".to_string()),
            context: Some("aggression".to_string()),
            individual_id: None,
            frame_index: None,
        };

        recorder.register_audio_event(event);

        // Event should be queued
        assert_eq!(recorder.pending_event_count(), 1);
    }

    #[test]
    fn test_recorder_get_statistics() {
        let config = VisualRecorderConfig {
            recording_dir: "/tmp/test_recordings".to_string(),
            ..Default::default()
        };

        let recorder = VisualRecorder::new(config);
        let stats = recorder.get_statistics();

        assert_eq!(stats.state, RecordingState::Stopped);
        assert_eq!(stats.frames_recorded, 0);
        assert_eq!(stats.dropped_frames, 0);
        assert_eq!(stats.current_session_id, None);
        assert_eq!(stats.duration_seconds, 0.0);
    }

    #[test]
    fn test_recorder_stop_session() {
        let config = VisualRecorderConfig {
            recording_dir: "/tmp/test_recordings".to_string(),
            ..Default::default()
        };

        let mut recorder = VisualRecorder::new(config);

        // Start session
        recorder.start_session("test_session").ok();

        // Stop session
        let result = recorder.stop_session();

        assert!(result.is_ok());
        let metadata = result.unwrap();
        assert_eq!(metadata.session_id, "test_session");
        assert_eq!(metadata.state, RecordingState::Stopped);
        assert!(metadata.end_time_ns.is_some());
    }

    // ========================================================================
    // VideoPathResolver Tests
    // ========================================================================

    #[test]
    fn test_resolve_video_path() {
        let config = VisualRecorderConfig {
            recording_dir: "/tmp/recordings".to_string(),
            ..Default::default()
        };

        let recorder = VisualRecorder::new(config);
        let session_id = "test_session_001";

        let video_path = recorder.resolve_video_path(session_id);
        let metadata_path = recorder.resolve_metadata_path(session_id);

        assert!(video_path.contains("test_session_001"));
        assert!(video_path.ends_with(".mp4"));

        assert!(metadata_path.contains("test_session_001"));
        assert!(metadata_path.ends_with("_metadata.json"));
    }

    // ========================================================================
    // FrameQueue Tests
    // ========================================================================

    #[test]
    fn test_frame_queue_push_and_pop() {
        let queue = FrameQueue::new(10);

        let frame = vec![0u8; 640 * 480 * 3];

        assert!(queue.push(frame.clone()).is_ok());

        let popped = queue.pop();
        assert!(popped.is_some());
        assert_eq!(popped.unwrap().len(), 640 * 480 * 3);
    }

    #[test]
    fn test_frame_queue_max_size() {
        let queue = FrameQueue::new(2);

        let frame = vec![0u8; 100];

        assert!(queue.push(frame.clone()).is_ok());
        assert!(queue.push(frame.clone()).is_ok());

        assert!(queue.push(frame).is_ok()); // Overwrites, doesn't fail
    }

    #[test]
    fn test_frame_queue_dropped_count() {
        let queue = FrameQueue::new(2);

        let frame = vec![0u8; 100];

        queue.push(frame.clone()).ok();
        queue.push(frame.clone()).ok();
        queue.push(frame).ok(); // Overwrites

        assert_eq!(queue.dropped_count(), 1);
    }

    // ========================================================================
    // Metadata Serialization Tests
    // ========================================================================

    #[test]
    fn test_metadata_serialization() {
        let mut metadata = VisualMetadata::new("test_session".to_string(), 0, (1280, 720), 30.0);

        metadata.add_audio_sync_event(AudioSyncEvent {
            timestamp_ns: 1_000_000_000,
            event_type: AudioEventType::VocalizationDetected,
            phrase_key: Some("phrase_001".to_string()),
            context: Some("aggression".to_string()),
            individual_id: None,
            frame_index: Some(100),
        });

        let json = serde_json::to_string(&metadata).unwrap();

        assert!(json.contains("test_session"));
        assert!(json.contains("VocalizationDetected"));
        assert!(json.contains("phrase_001"));
        assert!(json.contains("aggression"));

        let deserialized: VisualMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.session_id, metadata.session_id);
        assert_eq!(deserialized.camera_id, metadata.camera_id);
        assert_eq!(deserialized.audio_sync_events.len(), 1);
    }

    // ========================================================================
    // Timestamp Synchronization Tests
    // ========================================================================

    #[test]
    fn test_sync_timestamp_to_frame() {
        let metadata = VisualMetadata {
            session_id: "test".to_string(),
            camera_id: 0,
            resolution: (1280, 720),
            fps: 30.0,
            start_time_ns: 1_000_000_000,
            end_time_ns: None,
            state: RecordingState::Stopped,
            audio_sync_events: vec![],
            context_annotations: vec![],
            storage_path: None,
            file_size_bytes: None,
        };

        let audio_time_ns = 2_000_000_000;
        let frame_index = metadata.sync_timestamp_to_frame(audio_time_ns);
        assert_eq!(frame_index, Some(30));
    }

    #[test]
    fn test_sync_timestamp_before_start() {
        let metadata = VisualMetadata {
            session_id: "test".to_string(),
            camera_id: 0,
            resolution: (1280, 720),
            fps: 30.0,
            start_time_ns: 1_000_000_000,
            end_time_ns: None,
            state: RecordingState::Stopped,
            audio_sync_events: vec![],
            context_annotations: vec![],
            storage_path: None,
            file_size_bytes: None,
        };

        let audio_time_ns = 500_000_000;
        let frame_index = metadata.sync_timestamp_to_frame(audio_time_ns);
        assert_eq!(frame_index, None);
    }
}
