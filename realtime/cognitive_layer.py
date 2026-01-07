"""
Cognitive Layer Intelligence
===========================

Advanced cognitive processing with online learning, multi-source separation,
and multi-modal fusion capabilities.

Classes:
- CognitiveLayer: Main cognitive processing engine
- OnlineLearner: Adaptive learning system
- SourceSeparator: Cocktail party problem solver
- MultiModalFuser: Audio-visual fusion engine
"""

import logging
import os
import pickle
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import istft, stft
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Try to import librosa for advanced audio processing
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


# Enums from cognitive_intelligence.py
class LearningMode(Enum):
    """Learning modes for adaptation"""
    NONE = "none"
    FEW_SHOT = "few_shot"
    REINFORCEMENT = "reinforcement"
    UNSUPERVISED = "unsupervised"

class VisualAttention(Enum):
    """Visual attention levels"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class ContextType(Enum):
    """Context types for learning"""
    CONTACT_CALL = "contact_call"
    ALARM_CALL = "alarm_call"
    FOOD_CALL = "food_call"
    SOCIAL_INTERACTION = "social_interaction"
    PLAY = "play"
    AGGRESSIVE = "aggressive"

# Configuration classes
@dataclass
class AdaptationParameters:
    """Parameters for adaptive behavior."""
    preferred_f0: float = 5000.0
    preferred_duration: float = 0.2
    preferred_amplitude: float = 0.5
    learning_rate: float = 0.01
    adaptation_threshold: int = 5
    adaptation_count: int = 0
    last_adaptation: float = 0.0

@dataclass
class LearningConfig:
    """Configuration for learning system"""
    learning_mode: LearningMode = LearningMode.FEW_SHOT
    adaptation_rate: float = 0.1
    memory_size: int = 1000
    confidence_threshold: float = 0.7
    learning_enabled: bool = True
    reinforcement_learning: bool = True
    memory_decay: float = 0.95

@dataclass
class VisualConfig:
    """Configuration for visual processing"""
    attention_model: str = "mediapipe"
    min_face_confidence: float = 0.5
    tracking_enabled: bool = True
    fusion_enabled: bool = True
    attention_boost: float = 0.2

@dataclass
class SourceSeparationConfig:
    """Configuration for source separation"""
    model_type: str = "conv_tasnet"
    model_path: Optional[str] = None
    denoising_enabled: bool = True
    enhancement_factor: float = 1.5

@dataclass
class MemoryEntry:
    """Memory entry for learning"""
    features: np.ndarray
    context: ContextType
    f0: float
    response_positive: bool
    timestamp: float
    weight: float = 1.0
    access_count: int = 0

@dataclass
class VisualState:
    """Visual processing state"""
    attention: VisualAttention = VisualAttention.NONE
    face_detected: bool = False
    face_confidence: float = 0.0
    gaze_direction: Optional[str] = None
    tracking_active: bool = False
    processing_time_ms: float = 0.0

@dataclass
class CognitiveMetrics:
    """Cognitive performance metrics"""
    learning_events: int = 0
    adaptation_rate: float = 0.0
    visual_confidence: float = 0.0
    source_separation_quality: float = 0.0
    response_accuracy: float = 0.0
    memory_usage: int = 0
    processing_time_ms: float = 0.0


class FewShotLearner:
    """Few-shot learning system for adaptation"""

    def __init__(self, config: LearningConfig):
        self.config = config
        self.memory = deque(maxlen=config.memory_size)
        self.feature_scaler = StandardScaler()
        self.context_preferences = {}
        self.feature_weights = {}
        self.adaptation_count = 0

        # Learning statistics
        self.success_history = deque(maxlen=100)
        self.learning_rate = config.adaptation_rate

    def extract_features(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract audio features for learning"""
        # Basic feature extraction (can be enhanced with deep learning)
        features = []

        # RMS energy
        rms = np.sqrt(np.mean(audio ** 2))
        features.append(rms)

        # Spectral centroid
        if len(audio) > 0:
            fft = np.fft.fft(audio)
            freqs = np.fft.fftfreq(len(audio), 1.0 / sample_rate)
            power_spectrum = np.abs(fft) ** 2
            spectral_centroid = np.sum(freqs * power_spectrum) / np.sum(power_spectrum)
            features.append(spectral_centroid)
        else:
            features.append(0.0)

        # Zero crossing rate
        zcr = np.mean(np.diff(np.sign(audio)) != 0)
        features.append(zcr)

        # F0 estimate (simplified)
        if len(audio) > 100:
            autocorr = np.correlate(audio, audio, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            peak_idx = np.argmax(autocorr[1:]) + 1
            f0 = sample_rate / peak_idx if peak_idx > 0 else 0.0
        else:
            f0 = 0.0
        features.append(f0)

        # Spectral rolloff
        if len(audio) > 0:
            fft = np.fft.fft(audio)
            power_spectrum = np.abs(fft) ** 2
            cumsum = np.cumsum(np.sort(power_spectrum)[::-1])
            rolloff_idx = np.where(cumsum > 0.85 * cumsum[-1])[0][0]
            rolloff_freq = rolloff_idx * sample_rate / len(audio)
            features.append(rolloff_freq)
        else:
            features.append(0.0)

        return np.array(features)

    def add_experience(self, audio: np.ndarray, context: ContextType,
                      f0: float, response_positive: bool, sample_rate: int = 44100):
        """Add learning experience to memory"""
        if not self.config.learning_enabled:
            return

        # Extract features
        features = self.extract_features(audio, sample_rate)

        # Create memory entry
        entry = MemoryEntry(
            features=features,
            context=context,
            f0=f0,
            response_positive=response_positive,
            timestamp=time.time(),
            weight=1.0
        )

        # Add to memory
        self.memory.append(entry)

        # Update context preferences
        self._update_context_preferences(entry)

        # Update adaptation statistics
        self.adaptation_count += 1
        self.success_history.append(response_positive)

        logger.info(f"Learning experience added: {context.value}, "
                   f"F0: {f0:.1f}Hz, Response: {'positive' if response_positive else 'negative'}")

    def _update_context_preferences(self, entry: MemoryEntry):
        """Update context-specific preferences based on experience"""
        context = entry.context.value

        if context not in self.context_preferences:
            self.context_preferences[context] = {
                'preferred_f0': entry.f0,
                'adaptation_count': 0,
                'success_rate': 0.0,
                'total_responses': 0,
                'successful_responses': 0
            }

        prefs = self.context_preferences[context]
        prefs['adaptation_count'] += 1

        # Update preferred F0 with adaptation
        if entry.response_positive:
            prefs['successful_responses'] += 1
            prefs['preferred_f0'] += (entry.f0 - prefs['preferred_f0']) * self.learning_rate

        prefs['total_responses'] += 1
        prefs['success_rate'] = prefs['successful_responses'] / prefs['total_responses']

    def adapt_to_success(self, audio: np.ndarray, context: ContextType,
                        sample_rate: int = 44100) -> Dict[str, Any]:
        """Adapt parameters based on successful interaction"""
        if not self.config.learning_enabled or len(self.memory) < 2:
            return {'adapted': False, 'reason': 'insufficient_memory'}

        # Extract features from successful audio
        features = self.extract_features(audio, sample_rate)

        # Find similar experiences in memory
        similar_experiences = self._find_similar_experiences(features, context)

        if not similar_experiences:
            return {'adapted': False, 'reason': 'no_similar_experiences'}

        # Adapt parameters based on similar successful experiences
        adaptation_result = self._generate_adaptation(similar_experiences)

        self.learning_events = self.adaptation_count
        logger.info(f"Adaptation successful: {adaptation_result}")

        return adaptation_result

    def _find_similar_experiences(self, features: np.ndarray,
                                 context: ContextType) -> List[MemoryEntry]:
        """Find similar experiences in memory"""
        context_entries = [e for e in self.memory if e.context == context]
        if not context_entries:
            return []

        # Calculate similarity
        similarities = []
        for entry in context_entries:
            # Use weighted similarity based on feature importance
            similarity = cosine_similarity(features.reshape(1, -1),
                                         entry.features.reshape(1, -1))[0][0]
            similarities.append((similarity, entry))

        # Sort by similarity and return top matches
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in similarities[:5]]  # Top 5 matches

    def _generate_adaptation(self, experiences: List[MemoryEntry]) -> Dict[str, Any]:
        """Generate adaptation based on similar experiences"""
        if not experiences:
            return {'adapted': False}

        # Calculate weighted average of successful experiences
        total_weight = sum(e.weight for e in experiences if e.response_positive)
        if total_weight == 0:
            return {'adapted': False}

        # Adapt F0 based on successful experiences
        weighted_f0 = sum(e.f0 * e.weight for e in experiences if e.response_positive)
        adapted_f0 = weighted_f0 / total_weight

        # Generate adaptation parameters
        adaptation = {
            'adapted': True,
            'target_f0': adapted_f0,
            'adaptation_strength': self.learning_rate,
            'num_similar_experiences': len([e for e in experiences if e.response_positive]),
            'confidence': min(len(experiences) / 5.0, 1.0),  # Normalize by expected 5
            'timestamp': time.time()
        }

        return adaptation

    def get_adaptation_status(self) -> Dict[str, Any]:
        """Get current adaptation status"""
        success_rate = np.mean(self.success_history) if self.success_history else 0.0

        return {
            'learning_events': self.adaptation_count,
            'adaptation_count': self.adaptation_count,
            'success_rate': success_rate,
            'memory_size': len(self.memory),
            'context_preferences': self.context_preferences,
            'learning_enabled': self.config.learning_enabled
        }

    def save_learning_state(self, filepath: str):
        """Save learning state to file"""
        state = {
            'context_preferences': self.context_preferences,
            'adaptation_count': self.adaptation_count
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"Learning state saved to {filepath}")

    def load_learning_state(self, filepath: str):
        """Load learning state from file"""
        if not os.path.exists(filepath):
            logger.warning(f"Learning state file not found: {filepath}")
            return

        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)

            # Restore state
            if 'context_preferences' in state:
                self.context_preferences = state['context_preferences']
            if 'adaptation_count' in state:
                self.adaptation_count = state['adaptation_count']

            logger.info(f"Learning state loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load learning state: {e}")


class OnlineLearner(FewShotLearner):
    """
    Enhanced online learner with both reinforcement and few-shot learning.

    Implements reinforcement learning to adapt to individual animal preferences
    and response patterns, extended with few-shot learning capabilities.
    """

    def __init__(self, learning_rate: float = 0.01, adaptation_threshold: int = 5,
                 config: LearningConfig = None):
        """
        Initialize online learner.

        Args:
            learning_rate: Learning rate for adaptation
            adaptation_threshold: Number of consistent responses before adaptation
            config: Learning configuration for few-shot capabilities
        """
        # Initialize with default config if none provided
        if config is None:
            config = LearningConfig()

        super().__init__(config)
        self.adaptation_threshold = adaptation_threshold
        self.context_preferences = defaultdict(AdaptationParameters)
        self.response_history = defaultdict(list)
        self.adaptation_lock = threading.Lock()

        # Convert context_preferences from few-shot format
        for context, prefs in self.context_preferences.items():
            context_name = ContextType(context)
            if context_name in [ContextType.CONTACT_CALL, ContextType.FOOD_CALL,
                               ContextType.ALARM_CALL, ContextType.SOCIAL_INTERACTION]:
                self.context_preferences[context_name] = AdaptationParameters(
                    preferred_f0=prefs.get('preferred_f0', 5000.0),
                    adaptation_count=prefs.get('adaptation_count', 0)
                )

    def process_animal_response(self, context: str, f0: float, response_positive: bool) -> Dict[str, Any]:
        """
        Process animal response and update preferences.

        Args:
            context: Context category (Food, Contact, Alarm, etc.)
            f0: Fundamental frequency of the call
            response_positive: Whether animal responded positively

        Returns:
            Adaptation information
        """
        with self.adaptation_lock:
            params = self.context_preferences[context]

            # Record response
            self.response_history[context].append({
                'timestamp': time.time(),
                'f0': f0,
                'response_positive': response_positive
            })

            # Keep only recent responses (last 100)
            if len(self.response_history[context]) > 100:
                self.response_history[context] = self.response_history[context][-100:]

            # Check if adaptation is needed
            recent_responses = self.response_history[context][-self.adaptation_threshold:]
            if len(recent_responses) < self.adaptation_threshold:
                return {'adapted': False, 'reason': 'Not enough data'}

            # Check if responses are consistently positive/negative
            consistent_responses = all(
                r['response_positive'] == recent_responses[0]['response_positive']
                for r in recent_responses
            )

            if not consistent_responses:
                return {'adapted': False, 'reason': 'Inconsistent responses'}

            # Adapt preferences
            if recent_responses[0]['response_positive']:
                # Positive response - reinforce current parameters
                params.preferred_f0 += (f0 - params.preferred_f0) * self.learning_rate
                params.preferred_duration += (0.2 - params.preferred_duration) * self.learning_rate
                params.preferred_amplitude += (0.5 - params.preferred_amplitude) * self.learning_rate
                params.adaptation_count += 1
                params.last_adaptation = time.time()

            return {
                'adapted': True,
                'context': context,
                'new_f0': params.preferred_f0,
                'adaptation_count': params.adaptation_count
            }

    def get_adapted_parameters(self, context: str) -> Dict[str, Any]:
        """
        Get adapted parameters for a context.

        Args:
            context: Context category

        Returns:
            Adapted parameters
        """
        params = self.context_preferences[context]
        return {
            'preferred_f0': params.preferred_f0,
            'preferred_duration': params.preferred_duration,
            'preferred_amplitude': params.preferred_amplitude,
            'adaptation_count': params.adaptation_count,
            'last_adaptation': params.last_adaptation
        }


class VisualFusion:
    """Visual attention fusion system"""

    def __init__(self, config: VisualConfig):
        self.config = config
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=config.min_face_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # State
        self.visual_state = VisualState()
        self.attention_history = deque(maxlen=100)
        self.face_tracking_active = False

    def process_visual_frame(self, frame: np.ndarray) -> VisualState:
        """Process visual frame for attention and tracking"""
        start_time = time.time()

        if not self.config.tracking_enabled:
            self.visual_state = VisualState(attention=VisualAttention.NONE)
            self.visual_state.processing_time_ms = (time.time() - start_time) * 1000
            return self.visual_state

        try:
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process face mesh
            results = self.mp_face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                # Face detected
                face_landmarks = results.multi_face_landmarks[0]
                self.visual_state.face_detected = True
                self.visual_state.face_confidence = 0.8  # Placeholder

                # Estimate attention based on face landmarks
                attention = self._estimate_attention(face_landmarks)
                self.visual_state.attention = attention

                # Estimate gaze direction
                gaze = self._estimate_gaze_direction(face_landmarks)
                self.visual_state.gaze_direction = gaze

                self.face_tracking_active = True
            else:
                # No face detected
                self.visual_state.face_detected = False
                self.visual_state.attention = VisualAttention.NONE
                self.visual_state.gaze_direction = None
                self.face_tracking_active = False

            # Store in history
            self.attention_history.append(self.visual_state.attention)

        except Exception as e:
            logger.error(f"Error processing visual frame: {e}")
            self.visual_state.attention = VisualAttention.NONE
            self.visual_state.face_detected = False

        # Update processing time
        self.visual_state.processing_time_ms = (time.time() - start_time) * 1000

        return self.visual_state

    def _estimate_attention(self, landmarks) -> VisualAttention:
        """Estimate attention level from face landmarks"""
        # Simple heuristic: attention based on eye landmarks
        # In practice, this would use more sophisticated analysis

        # Get eye landmarks (approximate indices)
        left_eye = landmarks[33]  # Left eye corner
        right_eye = landmarks[263]  # Right eye corner
        landmarks[1]  # Nose tip

        # Calculate eye openness (simplified)
        eye_distance = np.sqrt((left_eye.x - right_eye.x)**2 + (left_eye.y - right_eye.y)**2)

        # Simple attention estimation
        if eye_distance > 0.05:  # Threshold for "open" eyes
            if len(self.attention_history) > 10:
                # Check consistency
                recent_attention = list(self.attention_history)[-10:]
                if all(att == VisualAttention.HIGH for att in recent_attention[-5:]):
                    return VisualAttention.HIGH
                elif all(att in [VisualAttention.MEDIUM, VisualAttention.HIGH]
                        for att in recent_attention[-5:]):
                    return VisualAttention.MEDIUM
            return VisualAttention.MEDIUM
        else:
            return VisualAttention.LOW

    def _estimate_gaze_direction(self, landmarks) -> Optional[str]:
        """Estimate gaze direction from face landmarks"""
        # Simplified gaze estimation
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        nose = landmarks[1]

        # Calculate relative position
        nose_to_left = (left_eye.x - nose.x, left_eye.y - nose.y)
        nose_to_right = (right_eye.x - nose.x, right_eye.y - nose.y)

        # Simple heuristic
        if nose_to_left[0] > 0.1:
            return "right"
        elif nose_to_right[0] > 0.1:
            return "left"
        else:
            return "center"

    def get_attention_boost(self, audio_context: str) -> float:
        """Calculate attention boost for audio processing"""
        if not self.config.fusion_enabled:
            return 0.0

        # Map audio context to relevant visual contexts
        context_mapping = {
            'contact_call': [VisualAttention.HIGH, VisualAttention.MEDIUM],
            'alarm_call': [VisualAttention.MEDIUM],
            'food_call': [VisualAttention.HIGH, VisualAttention.MEDIUM]
        }

        relevant_attentions = context_mapping.get(audio_context, [])

        if self.visual_state.attention in relevant_attentions:
            return self.config.attention_boost

        return 0.0

    def get_visual_state(self) -> VisualState:
        """Get current visual state"""
        return self.visual_state

    def shutdown(self):
        """Shutdown visual processing"""
        if hasattr(self, 'mp_face_mesh'):
            self.mp_face_mesh.close()


class SourceSeparator:
    """
    Enhanced multi-source audio separator.

    Implements independent component analysis for solving the cocktail party problem
    and isolating target voices from background noise, with advanced source separation capabilities.
    """

    def __init__(self, config: SourceSeparationConfig = None, sample_rate: int = 48000):
        """
        Initialize source separator.

        Args:
            config: Source separation configuration
            sample_rate: Audio sample rate
        """
        self.config = config or SourceSeparationConfig()
        self.sample_rate = sample_rate
        self.model = None
        self.is_loaded = False

        # Performance tracking
        self.separation_times = deque(maxlen=100)
        self.quality_scores = deque(maxlen=100)

        # Initialize PCA for ICA separation
        self.pca = PCA(n_components=self.config.model_type == "conv_tasnet" and 3 or 3)
        self.is_trained = False
        self.training_data = deque(maxlen=1000)

        # Try to load model if provided
        if self.config.model_path:
            self._load_model()

    def _load_model(self):
        """Load source separation model"""
        if self.config.model_type == "conv_tasnet":
            try:
                # Placeholder for Conv-TasNet loading
                # In practice, this would load a pre-trained PyTorch/TensorFlow model
                logger.info("Conv-TasNet model placeholder loaded")
                self.is_loaded = True
            except Exception as e:
                logger.error(f"Failed to load source separation model: {e}")
        else:
            logger.warning(f"Unsupported model type: {self.config.model_type}")

    def separate_sources(self, mixed_audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Separate sources from mixed audio.

        Args:
            mixed_audio: Mixed audio signal

        Returns:
            Dictionary of separated sources
        """
        start_time = time.time()

        if not self.is_loaded:
            # Use enhanced frequency-based separation
            result = self._enhanced_frequency_based_separation(mixed_audio)
        else:
            # Use loaded model
            result = self._model_separation(mixed_audio, self.sample_rate)

        processing_time = (time.time() - start_time) * 1000

        # Track performance
        self.separation_times.append(processing_time)

        # Calculate quality score (simplified)
        if 'target' in result and 'noise' in result:
            quality = self._calculate_separation_quality(result['target'], result['noise'], mixed_audio)
            self.quality_scores.append(quality)

        return result

    def _enhanced_frequency_based_separation(self, mixed_audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Enhanced frequency-based source separation with librosa if available.

        Args:
            mixed_audio: Mixed audio signal

        Returns:
            Separated sources
        """
        if LIBROSA_AVAILABLE:
            # Use librosa for better processing
            return self._librosa_based_separation(mixed_audio)
        else:
            # Fall back to scipy-based separation
            return self._frequency_based_separation(mixed_audio)

    def _librosa_based_separation(self, mixed_audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Librosa-based source separation"""
        try:
            # Apply STFT with librosa
            D = librosa.stft(mixed_audio)
            magnitude, phase = librosa.magphase(D)

            # Spectral subtraction for denoising
            noise_frames = int(0.01 * len(mixed_audio))
            noise_profile = np.mean(np.abs(librosa.stft(mixed_audio[:noise_frames])), axis=1, keepdims=True)

            # Apply spectral subtraction
            alpha = 2.0
            beta = 0.01
            enhanced_magnitude = magnitude - alpha * noise_profile
            enhanced_magnitude = np.maximum(enhanced_magnitude, beta * magnitude)

            # Reconstruct
            enhanced_stft = enhanced_magnitude * phase
            target_audio = librosa.istft(enhanced_stft)

            # Apply enhancement if enabled
            if self.config.denoising_enabled:
                target_audio = target_audio * self.config.enhancement_factor
                # Clip to prevent distortion
                max_sample = np.max(np.abs(target_audio))
                if max_sample > 1.0:
                    target_audio = target_audio / max_sample * 0.95

            return {
                'target': target_audio,
                'noise': mixed_audio - target_audio,
                'interferer': np.zeros_like(mixed_audio)  # Placeholder
            }

        except Exception as e:
            logger.error(f"Librosa separation failed: {e}")
            # Fall back to scipy
            return self._frequency_based_separation(mixed_audio)

    def _frequency_based_separation(self, mixed_audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Simple frequency-based source separation.

        Args:
            mixed_audio: Mixed audio signal

        Returns:
            Separated sources
        """
        # Compute STFT
        f, t, Zxx = stft(mixed_audio, fs=self.sample_rate, nperseg=1024)

        # Simple frequency masking based on energy
        target_mask = np.zeros_like(Zxx, dtype=bool)
        noise_mask = np.zeros_like(Zxx, dtype=bool)
        interferer_mask = np.zeros_like(Zxx, dtype=bool)

        # Classify frequency bins
        energy = np.abs(Zxx)
        np.sum(energy, axis=1)

        # Target: 4-8 kHz (typical for many animals)
        target_freq_idx = np.where((f >= 4000) & (f <= 8000))[0]
        if len(target_freq_idx) > 0:
            target_mask[target_freq_idx, :] = True

        # Noise: random frequencies
        noise_freq_idx = np.where((f >= 0) & (f <= 2000))[0]
        if len(noise_freq_idx) > 0:
            noise_mask[noise_freq_idx, :] = True

        # Interferer: 2-4 kHz
        interferer_freq_idx = np.where((f >= 2000) & (f <= 4000))[0]
        if len(interferer_freq_idx) > 0:
            interferer_mask[interferer_freq_idx, :] = True

        # Apply masks
        target_component = np.zeros_like(Zxx)
        noise_component = np.zeros_like(Zxx)
        interferer_component = np.zeros_like(Zxx)

        target_component[target_mask] = Zxx[target_mask]
        noise_component[noise_mask] = Zxx[noise_mask]
        interferer_component[interferer_mask] = Zxx[interferer_mask]

        # Reconstruct time domain
        _, target_audio = istft(target_component, fs=self.sample_rate)
        _, noise_audio = istft(noise_component, fs=self.sample_rate)
        _, interferer_audio = istft(interferer_component, fs=self.sample_rate)

        return {
            'target': target_audio,
            'noise': noise_audio,
            'interferer': interferer_audio
        }

    def _model_separation(self, mixed_audio: np.ndarray, sample_rate: int) -> Dict[str, np.ndarray]:
        """Model-based source separation (placeholder)"""
        # In practice, this would use the loaded model
        # For now, use enhanced frequency-based separation
        return self._enhanced_frequency_based_separation(mixed_audio)

    def _calculate_separation_quality(self, target: np.ndarray, noise: np.ndarray,
                                    original: np.ndarray) -> float:
        """Calculate separation quality score"""
        # Signal-to-noise ratio improvement
        original_snr = 10 * np.log10(np.sum(target ** 2) / (np.sum(original ** 2) - np.sum(target ** 2) + 1e-10))
        separated_snr = 10 * np.log10(np.sum(target ** 2) / (np.sum(noise ** 2) + 1e-10))

        snr_improvement = separated_snr - original_snr
        quality = max(0.0, min(1.0, (snr_improvement + 20) / 40))  # Normalize to [0, 1]

        return quality

    def apply_enhancement(self, audio: np.ndarray) -> np.ndarray:
        """Apply enhancement to separated audio"""
        if not self.config.denoising_enabled:
            return audio

        target, _ = self.separate_sources(audio)
        enhanced = target * self.config.enhancement_factor

        # Clip to prevent distortion
        max_sample = np.max(np.abs(enhanced))
        if max_sample > 1.0:
            enhanced = enhanced / max_sample * 0.95

        return enhanced

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get source separation performance metrics"""
        return {
            'model_loaded': self.is_loaded,
            'model_type': self.config.model_type,
            'avg_processing_time_ms': np.mean(self.separation_times) if self.separation_times else 0.0,
            'avg_quality_score': np.mean(self.quality_scores) if self.quality_scores else 0.0,
            'total_separations': len(self.separation_times)
        }


    def separate_sources(self, mixed_audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Separate sources from mixed audio.

        Args:
            mixed_audio: Mixed audio signal

        Returns:
            Dictionary of separated sources
        """
        if not self.is_trained:
            # Use simple frequency-based separation if not trained
            return self._frequency_based_separation(mixed_audio)

        # Use trained model for separation
        return self._ica_separation(mixed_audio)

    def _frequency_based_separation(self, mixed_audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Simple frequency-based source separation.

        Args:
            mixed_audio: Mixed audio signal

        Returns:
            Separated sources
        """
        # Compute STFT
        f, t, Zxx = stft(mixed_audio, fs=self.sample_rate, nperseg=1024)

        # Simple frequency masking based on energy
        target_mask = np.zeros_like(Zxx, dtype=bool)
        noise_mask = np.zeros_like(Zxx, dtype=bool)
        interferer_mask = np.zeros_like(Zxx, dtype=bool)

        # Classify frequency bins
        energy = np.abs(Zxx)
        np.sum(energy, axis=1)

        # Target: 4-8 kHz (typical for many animals)
        target_freq_idx = np.where((f >= 4000) & (f <= 8000))[0]
        if len(target_freq_idx) > 0:
            target_mask[target_freq_idx, :] = True

        # Noise: random frequencies
        noise_freq_idx = np.where((f >= 0) & (f <= 2000))[0]
        if len(noise_freq_idx) > 0:
            noise_mask[noise_freq_idx, :] = True

        # Interferer: 2-4 kHz
        interferer_freq_idx = np.where((f >= 2000) & (f <= 4000))[0]
        if len(interferer_freq_idx) > 0:
            interferer_mask[interferer_freq_idx, :] = True

        # Apply masks
        target_component = np.zeros_like(Zxx)
        noise_component = np.zeros_like(Zxx)
        interferer_component = np.zeros_like(Zxx)

        target_component[target_mask] = Zxx[target_mask]
        noise_component[noise_mask] = Zxx[noise_mask]
        interferer_component[interferer_mask] = Zxx[interferer_mask]

        # Reconstruct time domain
        _, target_audio = istft(target_component, fs=self.sample_rate)
        _, noise_audio = istft(noise_component, fs=self.sample_rate)
        _, interferer_audio = istft(interferer_component, fs=self.sample_rate)

        return {
            'target': target_audio,
            'noise': noise_audio,
            'interferer': interferer_audio
        }

    def _ica_separation(self, mixed_audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        ICA-based source separation.

        Args:
            mixed_audio: Mixed audio signal

        Returns:
            Separated sources
        """
        # Reshape for PCA
        segments = self._segment_audio(mixed_audio)
        features = self._extract_features(segments)

        # Apply PCA
        features_pca = self.pca.fit_transform(features)

        # Simple clustering for source separation
        clustering = DBSCAN(eps=0.5, min_samples=5).fit(features_pca)
        labels = clustering.labels_

        # Reconstruct audio from clusters
        sources = {}
        for i in range(self.n_components):
            if i in labels:
                mask = labels == i
                source_segments = [segments[j] for j in range(len(segments)) if mask[j]]
                source_audio = np.concatenate(source_segments)
                sources[f'source_{i}'] = source_audio

        return sources

    def _segment_audio(self, audio: np.ndarray, segment_length: int = 1024) -> List[np.ndarray]:
        """
        Segment audio for processing.

        Args:
            audio: Audio signal
            segment_length: Segment length in samples

        Returns:
            List of audio segments
        """
        segments = []
        for i in range(0, len(audio) - segment_length, segment_length // 2):
            segments.append(audio[i:i + segment_length])
        return segments

    def _extract_features(self, segments: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from audio segments.

        Args:
            segments: List of audio segments

        Returns:
            Feature matrix
        """
        features = []
        for segment in segments:
            # Basic features: energy, spectral centroid, zero crossings
            energy = np.sum(segment ** 2)
            spectral_centroid = self._compute_spectral_centroid(segment)
            zero_crossings = np.sum(np.diff(np.sign(segment)) != 0)

            features.append([energy, spectral_centroid, zero_crossings])

        return np.array(features)

    def _compute_spectral_centroid(self, audio: np.ndarray) -> float:
        """
        Compute spectral centroid.

        Args:
            audio: Audio signal

        Returns:
            Spectral centroid frequency
        """
        spectrum = np.abs(np.fft.fft(audio))
        frequencies = np.fft.fftfreq(len(audio))
        return np.sum(spectrum * frequencies) / np.sum(spectrum)


class MultiModalFuser:
    """
    Enhanced multi-modal fusion engine.

    Integrates audio features with visual context for enhanced understanding,
    with advanced attention-based fusion capabilities.
    """

    def __init__(self, audio_weight: float = 0.7, visual_weight: float = 0.3,
                 visual_config: VisualConfig = None, learning_config: LearningConfig = None):
        """
        Initialize multi-modal fuser.

        Args:
            audio_weight: Weight for audio features
            visual_weight: Weight for visual features
            visual_config: Configuration for visual processing
            learning_config: Configuration for learning system
        """
        self.audio_weight = audio_weight
        self.visual_weight = visual_weight
        self.visual_config = visual_config or VisualConfig()
        self.learning_config = learning_config or LearningConfig()

        # Initialize visual fusion system
        self.visual_fusion = VisualFusion(self.visual_config)

        # Initialize learning system
        self.few_shot_learner = FewShotLearner(self.learning_config)

    def fuse_audio_visual(self, audio_features: Dict[str, float], visual_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fuse audio and visual features with enhanced integration.

        Args:
            audio_features: Audio feature dictionary
            visual_context: Visual context dictionary

        Returns:
            Fused context probabilities
        """
        # Process visual frame if raw frame is provided
        if 'frame' in visual_context:
            visual_state = self.visual_fusion.process_visual_frame(visual_context['frame'])
            visual_context = {
                'attention': visual_state.attention.value,
                'face_detected': visual_state.face_detected,
                'gaze_direction': visual_state.gaze_direction,
                'confidence': visual_state.face_confidence
            }

        # Extract audio features
        audio_confidence = self._compute_audio_confidence(audio_features)

        # Extract visual features
        visual_confidence = self._compute_visual_confidence(visual_context)

        # Get attention boost from visual context
        audio_context = audio_features.get('context', 'contact_call')
        attention_boost = self._get_attention_boost_from_context(audio_context, visual_context)

        # Fuse confidences
        fused_result = {
            'audio_confidence': audio_confidence,
            'visual_confidence': visual_confidence,
            'fused_confidence': self.audio_weight * audio_confidence + self.visual_weight * visual_confidence,
            'contact_probability': self._compute_contact_probability(audio_features, visual_context),
            'attention_boost': attention_boost,
            'visual_state': visual_context
        }

        return fused_result

    def _compute_audio_confidence(self, audio_features: Dict[str, float]) -> float:
        """
        Compute confidence from audio features.

        Args:
            audio_features: Audio features

        Returns:
            Confidence score (0-1)
        """
        f0 = audio_features.get('f0', 0)
        rms = audio_features.get('rms', 0)
        duration = audio_features.get('duration', 0)

        # Simple scoring based on typical animal vocalizations
        f0_score = 1.0 if 1000 <= f0 <= 20000 else 0.5
        rms_score = min(rms * 2, 1.0)  # Scale RMS to 0-1
        duration_score = min(duration / 0.5, 1.0)  # Normalize to 0.5s max

        return (f0_score + rms_score + duration_score) / 3

    def _compute_visual_confidence(self, visual_context: Dict[str, Any]) -> float:
        """
        Compute confidence from visual context.

        Args:
            visual_context: Visual context

        Returns:
            Confidence score (0-1)
        """
        gaze_direction = visual_context.get('gaze_direction', 'unknown')
        face_detected = visual_context.get('face_detected', False)
        confidence = visual_context.get('confidence', 0.0)
        attention = visual_context.get('attention', 'none')

        # Scoring based on visual cues
        if gaze_direction == 'toward':
            gaze_score = 1.0
        elif gaze_direction == 'center':
            gaze_score = 0.7
        elif gaze_direction == 'neutral':
            gaze_score = 0.5
        else:
            gaze_score = 0.1

        contact_score = 1.0 if face_detected else 0.1
        attention_score = {'high': 1.0, 'medium': 0.7, 'low': 0.3, 'none': 0.1}.get(attention, 0.5)

        # Combine scores with weight for confidence metric
        base_confidence = (gaze_score + contact_score + attention_score) / 3
        return base_confidence * (0.5 + 0.5 * confidence)  # Boost by confidence metric

    def _get_attention_boost_from_context(self, audio_context: str, visual_context: Dict[str, Any]) -> float:
        """Get attention boost from visual context using attention boost calculator."""
        # Convert audio_context format to match the expected format
        context_mapping = {
            'contact_call': 'contact_call',
            'alarm_call': 'alarm_call',
            'food_call': 'food_call'
        }

        mapped_context = context_mapping.get(audio_context, 'contact_call')
        return self.visual_fusion.get_attention_boost(mapped_context)

    def _compute_contact_probability(self, audio_features: Dict[str, float], visual_context: Dict[str, Any]) -> float:
        """
        Compute contact probability from audio-visual fusion.

        Args:
            audio_features: Audio features
            visual_context: Visual context

        Returns:
            Contact probability (0-1)
        """
        # Base probability from audio
        audio_contact_prob = self._audio_to_contact_prob(audio_features)

        # Adjustment from visual
        visual_adjustment = self._visual_to_contact_adjustment(visual_context)

        # Combine
        final_prob = audio_contact_prob * (1 + visual_adjustment)
        return max(0, min(1, final_prob))

    def _audio_to_contact_prob(self, audio_features: Dict[str, float]) -> float:
        """Convert audio features to contact probability."""
        f0 = audio_features.get('f0', 0)
        if 5000 <= f0 <= 8000:  # Typical contact call frequency
            return 0.8
        elif 4000 <= f0 <= 10000:
            return 0.5
        else:
            return 0.2

    def _visual_to_contact_adjustment(self, visual_context: Dict[str, Any]) -> float:
        """Convert visual context to contact probability adjustment."""
        eye_contact = visual_context.get('eye_contact', False)
        gaze_direction = visual_context.get('gaze_direction', 'unknown')

        if eye_contact and gaze_direction == 'toward':
            return 0.5  # High confidence boost
        elif not eye_contact and gaze_direction == 'away':
            return -0.3  # Reduce probability
        else:
            return 0.0  # No adjustment


class CognitiveLayer:
    """
    Enhanced main cognitive processing layer.

    Integrates few-shot learning, enhanced source separation, and multi-modal fusion
    for intelligent audio processing with visual integration.
    """

    def __init__(self, learning_rate: float = 0.01, adaptation_threshold: int = 5,
                 learning_config: LearningConfig = None, visual_config: VisualConfig = None,
                 separation_config: SourceSeparationConfig = None):
        """
        Initialize cognitive layer.

        Args:
            learning_rate: Learning rate for adaptation
            adaptation_threshold: Adaptation threshold
            learning_config: Configuration for learning system
            visual_config: Configuration for visual processing
            separation_config: Configuration for source separation
        """
        # Initialize learning system with both reinforcement and few-shot capabilities
        self.learning_config = learning_config or LearningConfig()
        self.visual_config = visual_config or VisualConfig()
        self.separation_config = separation_config or SourceSeparationConfig()

        # Initialize components with enhanced functionality
        self.online_learner = OnlineLearner(learning_rate, adaptation_threshold, self.learning_config)
        self.source_separator = SourceSeparator(self.separation_config)
        self.multi_modal_fuser = MultiModalFuser(
            visual_config=self.visual_config,
            learning_config=self.learning_config
        )

        # Initialize cognitive metrics
        self.cognitive_metrics = CognitiveMetrics()
        self.context_history = deque(maxlen=1000)
        self.is_active = True
        self.logger = logging.getLogger(__name__)

    def process_audio_with_learning(self, audio: np.ndarray, context: ContextType,
                                   f0: float, response_positive: bool = None,
                                   sample_rate: int = 44100) -> Dict[str, Any]:
        """
        Process audio with learning capabilities.

        Args:
            audio: Audio signal
            context: Context type
            f0: Fundamental frequency
            response_positive: Whether response was positive
            sample_rate: Audio sample rate

        Returns:
            Processing result with enhanced audio and adaptation info
        """
        start_time = time.time()

        # Apply source separation
        separated_sources = self.source_separator.separate_sources(audio)
        enhanced_audio = separated_sources.get('target', audio.copy())

        # Add experience if response is provided
        if response_positive is not None and self.learning_config.learning_enabled:
            self.online_learner.add_experience(audio, context, f0, response_positive, sample_rate)

        # Adapt if response is positive
        adaptation_result = None
        if response_positive and self.learning_config.learning_enabled:
            adaptation_result = self.online_learner.adapt_to_success(audio, context, sample_rate)

        # Update metrics
        processing_time = (time.time() - start_time) * 1000
        self.cognitive_metrics.processing_time_ms = processing_time
        self.cognitive_metrics.learning_events = self.online_learner.adaptation_count

        return {
            'enhanced_audio': enhanced_audio,
            'noise_estimate': separated_sources.get('noise'),
            'interferer_estimate': separated_sources.get('interferer'),
            'adaptation_result': adaptation_result,
            'learning_metrics': self.online_learner.get_adaptation_status(),
            'processing_time_ms': processing_time,
            'sample_rate': sample_rate
        }

    def process_visual_context(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process visual context for attention fusion."""
        visual_state = self.multi_modal_fuser.visual_fusion.process_visual_frame(frame)

        return {
            'visual_state': visual_state.__dict__,
            'attention_boost': self.multi_modal_fuser.visual_fusion.get_attention_boost('contact_call'),
            'face_tracking': self.multi_modal_fuser.visual_fusion.face_tracking_active
        }

    def calculate_adaptive_response(self, audio_context: str,
                                   visual_context: Optional[Dict] = None) -> Dict[str, Any]:
        """Calculate adaptive response based on audio and visual context."""
        base_response = {'urgency': 0.5, 'aggression': 0.2, 'playfulness': 0.5}

        # Apply attention boost from visual context
        attention_boost = 0.0
        if visual_context and 'attention_boost' in visual_context:
            attention_boost = visual_context['attention_boost']
            # Boost response probability for contact calls
            if audio_context == 'contact_call':
                base_response['urgency'] += attention_boost
                base_response['playfulness'] += attention_boost * 0.5

        # Apply adaptation from learning
        adaptation = self.online_learner.get_adaptation_status()
        if adaptation['success_rate'] > 0.7:
            base_response['confidence'] = adaptation['success_rate']

        # Normalize values
        for key in base_response:
            base_response[key] = max(0.0, min(1.0, base_response[key]))

        return {
            'adaptive_response': base_response,
            'attention_boost': attention_boost,
            'adaptation_confidence': adaptation['success_rate'] if adaptation['success_rate'] > 0 else 0.5,
            'context': audio_context
        }

    def get_adapted_parameters(self, context: str) -> Dict[str, Any]:
        """
        Get adapted parameters for context.

        Args:
            context: Context category

        Returns:
            Adapted parameters
        """
        return self.online_learner.get_adapted_parameters(context)

    def process_context(self, audio_features: Dict[str, float], visual_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process context with optional visual input.

        Args:
            audio_features: Audio features
            visual_context: Optional visual context

        Returns:
            Processed context
        """
        if visual_context is None:
            # Audio-only processing
            return {
                'context_type': 'audio_only',
                'audio_confidence': self.multi_modal_fuser._compute_audio_confidence(audio_features),
                'contact_probability': self.multi_modal_fuser._audio_to_contact_prob(audio_features),
                'processing_time_ms': self.cognitive_metrics.processing_time_ms
            }
        else:
            # Multi-modal processing
            return self.multi_modal_fuser.fuse_audio_visual(audio_features, visual_context)

    def save_learning_state(self, filepath: str):
        """Save learning state to file."""
        self.online_learner.save_learning_state(filepath)

    def load_learning_state(self, filepath: str):
        """Load learning state from file."""
        self.online_learner.load_learning_state(filepath)

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive cognitive performance report."""
        return {
            'learning_system': self.online_learner.get_adaptation_status(),
            'visual_system': {
                'current_state': self.multi_modal_fuser.visual_fusion.get_visual_state().__dict__,
                'attention_history': [att.value for att in self.multi_modal_fuser.visual_fusion.attention_history[-10:]]
            },
            'source_separation': self.source_separator.get_performance_metrics(),
            'cognitive_metrics': {
                'processing_time_ms': self.cognitive_metrics.processing_time_ms,
                'learning_events': self.cognitive_metrics.learning_events
            },
            'system_status': {
                'active': self.is_active,
                'learning_enabled': self.learning_config.learning_enabled,
                'visual_enabled': self.visual_config.tracking_enabled,
                'separation_enabled': self.separation_config.denoising_enabled
            }
        }

    def shutdown(self):
        """Shutdown cognitive layer systems."""
        self.is_active = False
        self.multi_modal_fuser.visual_fusion.shutdown()
        self.logger.info("Cognitive layer shutdown")


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Create enhanced cognitive layer system
    learning_config = LearningConfig(learning_mode=LearningMode.FEW_SHOT, adaptation_rate=0.1)
    visual_config = VisualConfig(tracking_enabled=True, fusion_enabled=True)
    separation_config = SourceSeparationConfig(denoising_enabled=True)

    system = CognitiveLayer(
        learning_rate=0.01,
        adaptation_threshold=5,
        learning_config=learning_config,
        visual_config=visual_config,
        separation_config=separation_config
    )

    # Generate test audio
    sample_rate = 44100
    duration = 0.1
    t = np.linspace(0, duration, int(sample_rate * duration))
    test_audio = 0.5 * np.sin(2 * np.pi * 6000 * t)  # 6kHz test tone

    # Process with learning
    context = ContextType.CONTACT_CALL
    f0 = 6000.0
    response_positive = True

    result = system.process_audio_with_learning(test_audio, context, f0, response_positive)

    # Print results
    print("\nEnhanced Cognitive Layer Results:")
    print(f"Processing time: {result['processing_time_ms']:.2f}ms")
    print(f"Enhanced audio length: {len(result['enhanced_audio'])} samples")
    print(f"Learning events: {result['learning_metrics']['learning_events']}")
    print(f"Success rate: {result['learning_metrics']['success_rate']:.2f}")

    # Test source separation
    mixed_audio = test_audio + 0.3 * np.sin(2 * np.pi * 4000 * t)  # Add interferer
    separated = system.separate_sources(mixed_audio)
    print("\nSource separation:")
    print(f"Target length: {len(separated['target'])}")
    print(f"Separation quality: {system.source_separator.get_performance_metrics()['avg_quality_score']:.2f}")

    # Test visual processing (if OpenCV is available)
    try:
        # Create a test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        visual_result = system.process_visual_context(test_frame)
        print("\nVisual processing:")
        print(f"Face detected: {visual_result['visual_state']['face_detected']}")
        print(f"Attention boost: {visual_result['attention_boost']:.3f}")
    except Exception as e:
        print(f"\nVisual processing test skipped: {e}")

    # Test audio-visual fusion
    audio_features = {
        'f0': 6000.0,
        'rms': 0.5,
        'duration': 0.1,
        'context': 'contact_call'
    }
    visual_context = {
        'gaze_direction': 'toward',
        'face_detected': True,
        'attention': 'high',
        'confidence': 0.8
    }
    fusion_result = system.fuse_audio_visual(audio_features, visual_context)
    print("\nAudio-visual fusion:")
    print(f"Fused confidence: {fusion_result['fused_confidence']:.2f}")
    print(f"Contact probability: {fusion_result['contact_probability']:.2f}")

    # Get performance report
    report = system.get_performance_report()
    print("\nPerformance Report:")
    print(f"Memory size: {report['learning_system']['memory_size']}")
    print(f"Separation quality: {report['source_separation']['avg_quality_score']:.2f}")

    # Save learning state
    system.save_learning_state("cognitive_learning_state.pkl")

    # Cleanup
    system.shutdown()
    print("\nCognitive layer example completed successfully!")
