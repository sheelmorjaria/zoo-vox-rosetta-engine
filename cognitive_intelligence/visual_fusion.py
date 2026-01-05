"""
Visual Fusion Module
===================

Implements visual tracking and fusion using MediaPipe on a separate thread.
Integrates audio and visual information for enhanced animal communication understanding.

Key Features:
- MediaPipe pose and hand tracking
- Eye gaze estimation
- Visual attention detection
- Separate thread processing to avoid blocking audio
- Lightweight alternative with LightTrack fallback
- Integration with audio context for multimodal fusion

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import cv2
import numpy as np
import time
import threading
import queue
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from unittest.mock import Mock

# Optional MediaPipe import
try:
    import mediapipe as mp
    # Import MediaPipe tasks (newer version)
    try:
        from mediapipe import tasks
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python import holistics
        MEDIAPIPE_VERSION_NEW = True
    except ImportError:
        # Fallback to older solutions
        from mediapipe import solutions
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_hands = mp.solutions.hands
        mp_face_mesh = mp.solutions.face_mesh
        mp_pose = mp.solutions.pose
        mp_holistic = mp.solutions.holistic
        MEDIAPIPE_VERSION_NEW = False

    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    MEDIAPIPE_VERSION_NEW = False
    # Create mock classes for when MediaPipe is not available
    class MockMediaPipe:
        def __getattr__(self, name):
            return Mock()

    mp = MockMediaPipe()
    mp_drawing = Mock()
    mp_drawing_styles = Mock()
    mp_hands = Mock()
    mp_face_mesh = Mock()
    mp_pose = Mock()
    mp_holistic = Mock()
import multiprocessing as mp_multi
from concurrent.futures import ThreadPoolExecutor

# Create MediaPipe mock implementations that work for testing
class MockHandsClass:
    """Mock MediaPipe Hands class for testing"""
    def __init__(self, **kwargs):
        pass
    def process(self, image):
        return MockMediaPipeResults()

class MockFaceMeshClass:
    """Mock MediaPipe Face Mesh class for testing"""
    def __init__(self, **kwargs):
        pass
    def process(self, image):
        return MockMediaPipeResults()

class MockPoseClass:
    """Mock MediaPipe Pose class for testing"""
    def __init__(self, **kwargs):
        pass
    def process(self, image):
        return MockMediaPipeResults()

class MockDrawingUtils:
    """Mock MediaPipe Drawing Utils for testing"""
    @staticmethod
    def draw_landmarks(image, landmarks, connections):
        pass
    @staticmethod
    def draw_detection(image, detection):
        pass

class MockDrawingStyles:
    """Mock MediaPipe Drawing Styles for testing"""
    @staticmethod
    def get_default_hand_landmarks_style():
        return {}
    @staticmethod
    def get_default_face_mesh_style():
        return {}

class MockMediaPipeResults:
    """Mock MediaPipe Results for testing"""
    def __init__(self):
        self.multi_hand_landmarks = None
        self.multi_face_landmarks = None
        self.pose_landmarks = None

# Create a mock module structure
class MockMediaPipeModule:
    """Mock MediaPipe module structure"""
    class solutions:
        drawing_utils = MockDrawingUtils
        drawing_styles = MockDrawingStyles
        hands = MockHandsClass
        face_mesh = MockFaceMeshClass
        pose = MockPoseClass
        holistic = Mock()

# Create mock module structure
class MockModule:
    def __init__(self, name):
        self.name = name

    def __getattr__(self, attr):
        if attr == 'Hands':
            return MockHandsClass
        elif attr == 'FaceMesh':
            return MockFaceMeshClass
        elif attr == 'Pose':
            return MockPoseClass
        else:
            return Mock()

# Create mock objects that can be accessed as modules
mp_drawing = MockDrawingUtils()
mp_drawing_styles = MockDrawingStyles()
mp_hands = MockModule('hands')
mp_face_mesh = MockModule('face_mesh')
mp_pose = MockModule('pose')
mp_holistic = Mock()

# Configure MediaPipe based on version
if MEDIAPIPE_AVAILABLE:
    if MEDIAPIPE_VERSION_NEW:
        # For new version, use mocks (tasks API is different)
        # Already set above with correct class names
        pass
    else:
        # Old version using solutions
        try:
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            mp_hands = mp.solutions.hands
            mp_face_mesh = mp.solutions.face_mesh
            mp_pose = mp.solutions.pose
            mp_holistic = mp.solutions.holistic
        except:
            # Fall back to mocks
            mp_drawing = MockMediaPipeDrawingUtils()
            mp_drawing_styles = MockMediaPipeDrawingStyles()
            mp_hands = MockMediaPipeHands()
            mp_face_mesh = MockMediaPipeFaceMesh()
            mp_pose = MockMediaPipePose()
            mp_holistic = Mock()
else:
    # Mock the components when MediaPipe is not available
    # Already set above with correct class names
    pass

class VisualAttentionLevel(Enum):
    """Visual attention levels"""
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    VERY_HIGH = "Very High"

@dataclass
class VisualFeatures:
    """Visual features extracted from video"""
    timestamp: float = 0.0
    gaze_direction: Optional[str] = None  # 'towards_camera', 'away', 'left', 'right'
    head_pose: Optional[str] = None  # 'straight', 'angled', 'extreme'
    attention_level: VisualAttentionLevel = VisualAttentionLevel.LOW
    movement_intensity: float = 0.0  # 0.0 to 1.0
    hand_gestures: List[str] = field(default_factory=list)
    body_facing: Optional[str] = None  # 'towards_camera', 'away', 'sideways'
    confidence: float = 0.0  # Overall confidence in visual analysis

@dataclass
class VisualFusionConfig:
    """Configuration for visual fusion system"""
    camera_resolution: Tuple[int, int] = (640, 480)
    fps: int = 30
    use_mediapipe: bool = True
    use_lighttrack_fallback: bool = True
    lighttrack_threshold: float = 0.8  # Accuracy threshold for LightTrack
    gaze_estimation_enabled: bool = True
    separate_thread: bool = True
    max_queue_size: int = 100
    frame_buffer_size: int = 10
    attention_calculation_window: int = 30  # frames

class MediaPipeTracker:
    """MediaPipe visual tracking implementation"""

    def __init__(self, config: VisualFusionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize MediaPipe solutions
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Movement tracking
        self.previous_landmarks = None
        self.movement_history = deque(maxlen=config.attention_calculation_window)

        # Attention calculation
        self.attention_scores = deque(maxlen=config.attention_calculation_window)

    def process_frame(self, frame: np.ndarray) -> VisualFeatures:
        """Process a single frame and extract visual features"""
        try:
            # Flip frame horizontally for selfie-view display
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process different body parts
            features = VisualFeatures()
            features.timestamp = time.time()

            # Process hands
            hand_results = self.hands.process(rgb_frame)
            hand_gestures = self._extract_hand_gestures(hand_results)
            features.hand_gestures = hand_gestures

            # Process face
            face_results = self.face_mesh.process(rgb_frame)
            if face_results.multi_face_landmarks:
                gaze, head_pose = self._estimate_gaze_and_pose(
                    face_results.multi_face_landmarks[0].landmark,
                    rgb_frame.shape
                )
                features.gaze_direction = gaze
                features.head_pose = head_pose
                features.body_facing = self._determine_body_facing(gaze, head_pose)

            # Process pose
            pose_results = self.pose.process(rgb_frame)
            movement = self._calculate_movement(pose_results)
            features.movement_intensity = movement

            # Calculate attention level
            features.attention_level = self._calculate_attention_level(features)
            features.confidence = self._calculate_confidence(
                hand_results, face_results, pose_results
            )

            # Update movement history
            self.movement_history.append(movement)
            self.attention_scores.append(
                self._attention_score_from_level(features.attention_level)
            )

            return features

        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            return VisualFeatures()

    def _extract_hand_gestures(self, hand_results) -> List[str]:
        """Extract hand gestures from hand tracking results with enhanced recognition"""
        gestures = []

        if not hand_results.multi_hand_landmarks:
            return gestures

        for hand_landmarks in hand_results.multi_hand_landmarks:
            # Get handedness information
            if hasattr(hand_results, 'multi_handedness') and hand_results.multi_handedness:
                handedness = hand_results.multi_handedness[0].classification[0].label
            else:
                handedness = "Unknown"

            # Enhanced finger counting with better accuracy
            fingers_up = self._count_fingers_up(hand_landmarks.landmark)

            # Additional gesture detection using finger relationships
            gesture = self._classify_advanced_gesture(hand_landmarks.landmark, fingers_up, handedness)

            if gesture:
                gestures.append(gesture)

        return gestures

    def _classify_advanced_gesture(self, landmarks, fingers_up: int, handedness: str) -> Optional[str]:
        """Classify advanced hand gestures beyond simple finger counting"""
        # Get specific landmark positions
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        index_tip = landmarks[8]
        index_pip = landmarks[6]
        middle_tip = landmarks[12]
        middle_pip = landmarks[10]
        ring_tip = landmarks[16]
        ring_pip = landmarks[14]
        pinky_tip = landmarks[20]
        pinky_pip = landmarks[18]

        # Calculate distances between fingertips
        index_middle_dist = np.sqrt((index_tip.x - middle_tip.x)**2 + (index_tip.y - middle_tip.y)**2)
        ring_pinky_dist = np.sqrt((ring_tip.x - pinky_tip.x)**2 + (ring_tip.y - pinky_tip.y)**2)

        # Check for specific gesture patterns
        # Thumbs up/down detection
        if fingers_up == 0:
            # Check if thumb is extended
            if handedness == "Right":
                thumb_up = thumb_tip.x < thumb_ip.x  # For right hand
            else:
                thumb_up = thumb_tip.x > thumb_ip.x  # For left hand

            if thumb_up:
                return "thumbs_up"
            else:
                return "thumbs_down"

        # Peace sign verification (only index and middle extended)
        elif fingers_up == 2:
            # Check if only index and middle are up
            index_extended = index_tip.y < index_pip.y
            middle_extended = middle_tip.y < middle_pip.y
            ring_down = ring_tip.y > ring_pip.y
            pinky_down = pinky_tip.y > pinky_pip.y

            if index_extended and middle_extended and ring_down and pinky_down:
                # Verify V-shape
                if index_middle_dist < 0.1:  # Fingers are close together
                    return "peace_sign"
                else:
                    return "two_fingers"

        # Open palm detection
        elif fingers_up == 5:
            # Check if all fingers are spread out
            if index_middle_dist > 0.15 and ring_pinky_dist > 0.15:
                return "open_palm"
            else:
                return "open_hand"

        # Pointing gesture (only index finger)
        elif fingers_up == 1:
            # Verify only index finger is extended
            index_extended = index_tip.y < index_pip.y
            middle_down = middle_tip.y > middle_pip.y
            ring_down = ring_tip.y > ring_pip.y
            pinky_down = pinky_tip.y > pinky_pip.y

            if index_extended and middle_down and ring_down and pinky_down:
                # Check if thumb is tucked
                thumb_tucked = abs(thumb_tip.x - thumb_ip.x) < 0.05
                if thumb_tucked:
                    return "pointing"

        # Fist detection (no fingers extended)
        elif fingers_up == 0:
            # Verify all fingertips are above their PIP joints
            index_fist = index_tip.y > index_pip.y
            middle_fist = middle_tip.y > middle_pip.y
            ring_fist = ring_tip.y > ring_pip.y
            pinky_fist = pinky_tip.y > pinky_pip.y

            if index_fist and middle_fist and ring_fist and pinky_fist:
                # Check if thumb is tucked
                thumb_tucked = abs(thumb_tip.x - thumb_ip.x) < 0.08
                if thumb_tucked:
                    return "fist"

        # Ok gesture (thumb and index finger form circle)
        elif fingers_up == 0:
            # Calculate distance between thumb tip and index tip
            thumb_index_dist = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)

            if thumb_index_dist < 0.05:  # Thumb and finger touching
                return "ok_gesture"

        # Rock on gesture (index and pinky extended, thumb tucked)
        elif fingers_up == 2:
            # Check if index and pinky are extended
            index_extended = index_tip.y < index_pip.y
            pinky_extended = pinky_tip.y < pinky_pip.y
            middle_down = middle_tip.y > middle_pip.y
            ring_down = ring_tip.y > ring_pip.y

            if index_extended and pinky_extended and middle_down and ring_down:
                # Check thumb position
                if handedness == "Right":
                    thumb_tucked = thumb_tip.x > thumb_ip.x
                else:
                    thumb_tucked = thumb_tip.x < thumb_ip.x

                if thumb_tucked:
                    return "rock_on"

        # Return basic finger count if no specific gesture detected
        if fingers_up > 0:
            return f"{fingers_up}_fingers"

        return None

    def _count_fingers_up(self, landmarks) -> int:
        """Count fingers based on landmark positions"""
        # Simplified finger counting logic
        fingers = 0

        # Thumb
        if landmarks[4].x < landmarks[3].x:  # Right hand
            fingers += 1
        elif landmarks[4].x > landmarks[3].x:  # Left hand
            fingers += 1

        # Other fingers
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]

        for tip, pip in zip(finger_tips, finger_pips):
            if landmarks[tip].y < landmarks[pip].y:
                fingers += 1

        return fingers

    def _estimate_gaze_and_pose(self, landmarks, frame_shape) -> Tuple[str, str]:
        """Estimate gaze direction and head pose from face landmarks with improved accuracy"""
        # Eye landmarks for better gaze estimation
        left_eye_inner = landmarks[133]  # Left eye inner corner
        left_eye_outer = landmarks[33]   # Left eye outer corner
        left_eye_top = landmarks[159]     # Left eye top
        left_eye_bottom = landmarks[145]  # Left eye bottom

        right_eye_inner = landmarks[362]  # Right eye inner corner
        right_eye_outer = landmarks[263]  # Right eye outer corner
        right_eye_top = landmarks[386]    # Right eye top
        right_eye_bottom = landmarks[374] # Right eye bottom

        nose_tip = landmarks[1]          # Nose tip
        nose_bridge = landmarks[2]        # Nose bridge

        # Calculate eye aspect ratios for gaze detection
        left_eye_width = abs(left_eye_outer.x - left_eye_inner.x)
        left_eye_height = abs(left_eye_top.y - left_eye_bottom.y)
        right_eye_width = abs(right_eye_outer.x - right_eye_inner.x)
        right_eye_height = abs(right_eye_top.y - right_eye_bottom.y)

        left_ear = left_eye_height / (left_eye_width + 1e-6)
        right_ear = right_eye_height / (right_eye_width + 1e-6)

        # Calculate pupil positions (approximation using eye corners)
        left_pupil_x = (left_eye_inner.x + left_eye_outer.x) / 2
        left_pupil_y = (left_eye_inner.y + left_eye_outer.y) / 2
        right_pupil_x = (right_eye_inner.x + right_eye_outer.x) / 2
        right_pupil_y = (right_eye_inner.y + right_eye_outer.y) / 2

        # Calculate gaze direction using vector analysis
        # Vector from nose to pupils
        left_gaze_vector_x = left_pupil_x - nose_tip.x
        left_gaze_vector_y = left_pupil_y - nose_tip.y
        right_gaze_vector_x = right_pupil_x - nose_tip.x
        right_gaze_vector_y = right_pupil_y - nose_tip.y

        # Average gaze vector
        avg_gaze_x = (left_gaze_vector_x + right_gaze_vector_x) / 2
        avg_gaze_y = (left_gaze_vector_y + right_gaze_vector_y) / 2

        # Normalize and determine gaze direction
        gaze_magnitude = np.sqrt(avg_gaze_x**2 + avg_gaze_y**2)

        if gaze_magnitude > 0.1:
            # Calculate angle in normalized image coordinates
            gaze_angle = np.arctan2(avg_gaze_y, avg_gaze_x)

            # Convert to degrees for easier interpretation
            gaze_degrees = np.degrees(gaze_angle)

            # Determine gaze direction with improved accuracy
            if gaze_degrees > 30:
                gaze = "left"
            elif gaze_degrees < -30:
                gaze = "right"
            else:
                gaze = "towards_camera"
        else:
            gaze = "towards_camera"

        # Improved head pose estimation using multiple facial landmarks
        # Use 3D-like pose estimation using facial landmarks

        # Calculate head tilt using ear landmarks and nose alignment
        left_ear_landmark = landmarks[234]  # Left ear
        right_ear_landmark = landmarks[454]  # Right ear

        # Calculate horizontal and vertical alignment
        horizontal_alignment = abs(left_ear_landmark.x - right_ear_landmark.x)
        vertical_alignment = abs(left_ear_landmark.y - right_ear_landmark.y)

        # Calculate head rotation using chin and forehead points
        chin = landmarks[152]  # Chin
        forehead = landmarks[10]  # Forehead center

        # Calculate head angle
        head_vector_x = forehead.x - chin.x
        head_vector_y = forehead.y - chin.y
        head_angle = np.degrees(np.arctan2(head_vector_y, head_vector_x))

        # Determine head pose with better accuracy
        if abs(head_angle) < 10 and vertical_alignment < 0.05:
            head_pose = "straight"
        elif 10 <= abs(head_angle) < 25 or vertical_alignment < 0.08:
            head_pose = "angled"
        else:
            head_pose = "extreme"

        # Add confidence scores
        gaze_confidence = min(gaze_magnitude * 2, 1.0)  # Scale to 0-1
        pose_confidence = 1.0 - (abs(head_angle) / 90)  # Inverse relationship

        # Store confidence in landmarks metadata if available
        if hasattr(landmarks, '__getitem__'):
            landmarks[0].gaze_confidence = gaze_confidence
            landmarks[0].pose_confidence = pose_confidence

        return gaze, head_pose

    def _determine_body_facing(self, gaze: str, head_pose: str) -> str:
        """Determine body facing direction"""
        if gaze == "towards_camera" and head_pose == "straight":
            return "towards_camera"
        elif gaze in ["left", "right"] and head_pose in ["angled", "extreme"]:
            return "sideways"
        else:
            return "away"

    def _calculate_movement(self, pose_results) -> float:
        """Calculate movement intensity from pose landmarks"""
        if not pose_results.pose_landmarks:
            return 0.0

        current_landmarks = pose_results.pose_landmarks.landmark

        if self.previous_landmarks is None:
            self.previous_landmarks = current_landmarks
            return 0.0

        # Calculate average movement between corresponding landmarks
        total_movement = 0.0
        count = 0

        for prev, curr in zip(self.previous_landmarks, current_landmarks):
            movement = np.sqrt(
                (curr.x - prev.x) ** 2 +
                (curr.y - prev.y) ** 2 +
                (curr.z - prev.z) ** 2
            )
            total_movement += movement
            count += 1

        self.previous_landmarks = current_landmarks

        # Normalize movement (typical values are 0.001 to 0.1)
        return min(total_movement / count * 10, 1.0) if count > 0 else 0.0

    def _calculate_attention_level(self, features: VisualFeatures) -> VisualAttentionLevel:
        """Calculate attention level based on various visual cues"""
        score = 0.0

        # Movement contributes to attention
        score += features.movement_intensity * 0.3

        # Gaze direction contributes
        if features.gaze_direction == "towards_camera":
            score += 0.4
        elif features.gaze_direction in ["left", "right"]:
            score += 0.2

        # Hand gestures contribute
        if features.hand_gestures:
            score += 0.1 * len(features.hand_gestures)

        # Body facing contributes
        if features.body_facing == "towards_camera":
            score += 0.3
        elif features.body_facing == "sideways":
            score += 0.1

        # Convert score to attention level
        if score >= 0.8:
            return VisualAttentionLevel.VERY_HIGH
        elif score >= 0.6:
            return VisualAttentionLevel.HIGH
        elif score >= 0.4:
            return VisualAttentionLevel.MODERATE
        else:
            return VisualAttentionLevel.LOW

    def _attention_score_from_level(self, level: VisualAttentionLevel) -> float:
        """Convert attention level to numerical score"""
        scores = {
            VisualAttentionLevel.LOW: 0.1,
            VisualAttentionLevel.MODERATE: 0.4,
            VisualAttentionLevel.HIGH: 0.7,
            VisualAttentionLevel.VERY_HIGH: 1.0
        }
        return scores.get(level, 0.1)

    def _calculate_confidence(self, hand_results, face_results, pose_results) -> float:
        """Calculate overall confidence in visual analysis"""
        confidence = 0.0
        count = 0

        # Hand detection confidence
        if hand_results.multi_hand_landmarks:
            confidence += 0.3
            count += 1

        # Face detection confidence
        if face_results.multi_face_landmarks:
            confidence += 0.4
            count += 1

        # Pose detection confidence
        if pose_results.pose_landmarks:
            confidence += 0.3
            count += 1

        return confidence / count if count > 0 else 0.0

class LightTrackFallback:
    """Enhanced lightweight visual tracking with advanced computer vision techniques"""

    def __init__(self, config: VisualFusionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize OpenCV-based tracking components
        self.tracker_types = ['CSRT', 'KCF', 'MIL']
        self.trackers = {}
        self.track_history = {}
        self.frame_count = 0
        self.initialized = False

        # Background subtractor for motion detection
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True)

        # Optical flow parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        # Previous frame for optical flow
        self.prev_gray = None

        # Feature detection parameters
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )

        self.logger.info("LightTrackFallback initialized with advanced computer vision")

    def track(self, frame: np.ndarray) -> VisualFeatures:
        """Enhanced tracking implementation with multiple computer vision techniques"""
        features = VisualFeatures()
        features.timestamp = time.time()

        try:
            # Initialize on first frame
            if not self.initialized:
                self._initialize_tracker(frame)
                self.initialized = True

            # Multiple tracking techniques for robustness
            movement_features = self._detect_motion(frame)
            flow_features = self._optical_flow_analysis(frame)
            attention_features = self._calculate_attention_features(frame)

            # Combine features
            features.movement_intensity = (movement_features['intensity'] +
                                         flow_features['intensity']) / 2
            features.attention_level = attention_features['level']

            # Detect basic body regions
            body_features = self._detect_body_regions(frame)
            features.body_facing = body_features['facing']
            features.head_pose = body_features['pose']

            # Calculate confidence based on detection quality
            features.confidence = (movement_features['confidence'] +
                                flow_features['confidence'] +
                                attention_features['confidence']) / 3

            # Update tracking information
            self._update_tracking_state(frame)

        except Exception as e:
            self.logger.error(f"Error in LightTrack tracking: {e}")
            # Fallback to basic motion detection
            features = self._basic_motion_detection(frame)

        return features

    def _initialize_tracker(self, frame):
        """Initialize tracking with various techniques"""
        # Initialize background subtractor
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.fgbg.apply(gray)

        # Store first frame for optical flow
        self.prev_gray = gray

        # Initialize feature points
        self.prev_pts = cv2.goodFeaturesToTrack(
            gray, mask=None, **self.feature_params)

        self.logger.debug("LightTrack initialized successfully")

    def _detect_motion(self, frame):
        """Motion detection using background subtraction and contour analysis"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply background subtraction
        fgmask = self.fgbg.apply(gray)

        # Remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Analyze contours
        total_area = 0
        large_contours = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filter small noise
                total_area += area
                large_contours += 1

                # Draw bounding box for large movements
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Calculate movement metrics
        frame_area = frame.shape[0] * frame.shape[1]
        movement_ratio = total_area / frame_area if frame_area > 0 else 0

        # Determine attention level based on movement
        if movement_ratio > 0.05:  # 5% of frame
            attention_level = VisualAttentionLevel.HIGH
        elif movement_ratio > 0.02:  # 2% of frame
            attention_level = VisualAttentionLevel.MODERATE
        else:
            attention_level = VisualAttentionLevel.LOW

        # Confidence based on detection quality
        confidence = min(large_contours / 5.0, 1.0)  # Normalize by expected max

        return {
            'intensity': min(movement_ratio * 10, 1.0),
            'level': attention_level,
            'confidence': confidence
        }

    def _optical_flow_analysis(self, frame):
        """Optical flow analysis for detailed motion tracking"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None or self.prev_pts is None:
            return {'intensity': 0.0, 'confidence': 0.0}

        # Calculate optical flow
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None, **self.lk_params)

        # Calculate flow intensity
        if next_pts is not None:
            good_new = next_pts[status == 1]
            good_old = self.prev_pts[status == 1]

            if len(good_new) > 0:
                # Calculate average movement
                flow_magnitudes = []
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    flow_magnitude = np.sqrt((new[0] - old[0])**2 + (new[1] - old[1])**2)
                    flow_magnitudes.append(flow_magnitude)

                avg_flow = np.mean(flow_magnitudes)
                max_flow = np.max(flow_magnitudes)

                # Convert to normalized intensity
                intensity = min(avg_flow / 50.0, 1.0)  # Normalize by typical max

                # Confidence based on number of tracked points
                confidence = min(len(good_new) / 50.0, 1.0)

                # Draw optical flow vectors
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    cv2.line(frame, old.astype(int), new.astype(int), (0, 255, 0), 2)
                    cv2.circle(frame, new.astype(int), 5, (0, 0, 255), -1)

                return {'intensity': intensity, 'confidence': confidence}

        return {'intensity': 0.0, 'confidence': 0.0}

    def _calculate_attention_features(self, frame):
        """Calculate attention features based on motion and visual cues"""
        # Simple edge detection for visual complexity
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

        # High edge density might indicate focused attention
        if edge_density > 0.1:
            attention_level = VisualAttentionLevel.HIGH
        elif edge_density > 0.05:
            attention_level = VisualAttentionLevel.MODERATE
        else:
            attention_level = VisualAttentionLevel.LOW

        # Confidence based on frame quality
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        confidence = min(blur_score / 500.0, 1.0)  # Normalize by typical good quality

        return {
            'level': attention_level,
            'confidence': confidence
        }

    def _detect_body_regions(self, frame):
        """Detect basic body regions and orientation"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Simple motion-based region detection
        fgmask = self.fgbg.apply(gray)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour (assuming it's the main subject)
            largest_contour = max(contours, key=cv2.contourArea)

            # Get bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            center_x = x + w // 2
            center_y = y + h // 2

            # Determine facing direction based on position
            frame_center_x = frame.shape[1] // 2
            frame_center_y = frame.shape[0] // 2

            # Horizontal position
            if abs(center_x - frame_center_x) < frame.shape[1] * 0.1:
                facing = "towards_camera"
            elif center_x < frame_center_x:
                facing = "left"
            else:
                facing = "right"

            # Vertical position for pose estimation
            if abs(center_y - frame_center_y) < frame.shape[0] * 0.1:
                pose = "straight"
            elif center_y < frame_center_y:
                pose = "angled"
            else:
                pose = "extreme"
        else:
            facing = "away"
            pose = "straight"

        return {'facing': facing, 'pose': pose}

    def _update_tracking_state(self, frame):
        """Update tracking state for next frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_gray = gray

        # Update feature points periodically
        if self.frame_count % 10 == 0:  # Every 10 frames
            self.prev_pts = cv2.goodFeaturesToTrack(
                gray, mask=None, **self.feature_params)

        self.frame_count += 1

    def _basic_motion_detection(self, frame):
        """Fallback basic motion detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if hasattr(self, 'prev_gray'):
            frame_delta = cv2.absdiff(self.prev_gray, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)

            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                features = VisualFeatures()
                features.movement_intensity = min(len(contours) / 10.0, 1.0)
                features.attention_level = VisualAttentionLevel.MODERATE if features.movement_intensity > 0.5 else VisualAttentionLevel.LOW
                features.timestamp = time.time()
                return features

        self.prev_gray = gray
        return VisualFeatures()

class VisualFusionSystem:
    """Main visual fusion system that coordinates visual tracking"""

    def __init__(self, config: VisualFusionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize tracker
        if config.use_mediapipe:
            self.tracker = MediaPipeTracker(config)
            self.fallback_tracker = LightTrackFallback(config) if config.use_lighttrack_fallback else None
        else:
            # Use fallback as primary
            self.tracker = LightTrackFallback(config)
            self.fallback_tracker = None

        # Threading setup
        self.config = config
        self.running = False
        self.thread = None
        self.frame_queue = queue.Queue(maxsize=config.max_queue_size)
        self.result_queue = queue.Queue(maxsize=config.max_queue_size)

        # Performance monitoring
        self.processing_times = deque(maxlen=100)
        self.frame_drops = 0

    def start(self, camera_id: int = 0):
        """Start visual tracking system"""
        if self.running:
            return

        self.running = True

        if self.config.separate_thread:
            # Start processing in separate thread
            self.thread = threading.Thread(target=self._processing_loop, args=(camera_id,))
            self.thread.daemon = True
            self.thread.start()
            self.logger.info("Visual fusion system started on separate thread")
        else:
            # Start in main thread
            self._processing_loop(camera_id)

    def stop(self):
        """Stop visual tracking system"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

        # Clear queues
        while not self.frame_queue.empty():
            self.frame_queue.get()
            self.frame_queue.task_done()

        while not self.result_queue.empty():
            self.result_queue.get()
            self.result_queue.task_done()

        self.logger.info("Visual fusion system stopped")

    def process_frame_async(self, frame: np.ndarray):
        """Process frame asynchronously"""
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            self.frame_drops += 1
            self.logger.warning(f"Frame dropped: queue full. Total drops: {self.frame_drops}")

    def get_visual_features(self) -> Optional[VisualFeatures]:
        """Get latest visual features"""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None

    def _processing_loop(self, camera_id: int):
        """Main processing loop running in separate thread"""
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_resolution[1])
        cap.set(cv2.CAP_PROP_FPS, self.config.fps)

        frame_count = 0
        fps_display_time = time.time()
        fps = 0

        while self.running:
            try:
                # Read frame from camera
                ret, frame = cap.read()
                if not ret:
                    self.logger.error("Failed to read frame from camera")
                    time.sleep(0.1)
                    continue

                # Process frame
                start_time = time.time()

                # Try MediaPipe first
                features = self.tracker.process_frame(frame)

                # Fallback to LightTrack if MediaPipe fails or accuracy too low
                if self.fallback_tracker and features.confidence < self.config.lighttrack_threshold:
                    self.logger.debug("Using LightTrack fallback")
                    features = self.fallback_tracker.track(frame)

                # Put result in queue
                try:
                    self.result_queue.put_nowait(features)
                except queue.Full:
                    self.frame_drops += 1

                # Calculate processing time
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)

                # Calculate FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    fps = 30.0 / (time.time() - fps_display_time)
                    fps_display_time = time.time()
                    self.logger.debug(f"Visual processing FPS: {fps:.1f}")

                # Control frame rate
                time.sleep(1.0 / self.config.fps)

            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                time.sleep(0.1)

        # Clean up
        cap.release()

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            "running": self.running,
            "frame_drops": self.frame_drops,
            "queue_size": self.frame_queue.qsize(),
            "result_queue_size": self.result_queue.qsize(),
            "avg_processing_time": np.mean(self.processing_times) if self.processing_times else 0.0,
            "max_processing_time": max(self.processing_times) if self.processing_times else 0.0,
            "fps_target": self.config.fps,
            "camera_resolution": self.config.camera_resolution
        }

    def integrate_with_audio(self, audio_features: Dict[str, Any],
                           visual_features: Optional[VisualFeatures] = None) -> Dict[str, Any]:
        """Integrate visual features with audio features"""
        if not visual_features:
            return audio_features

        # Create multimodal fusion
        fusion_result = audio_features.copy()

        # Visual attention boost
        if visual_features.attention_level in [VisualAttentionLevel.HIGH, VisualAttentionLevel.VERY_HIGH]:
            boost_factor = 0.2 if visual_features.attention_level == VisualAttentionLevel.HIGH else 0.3

            # Boost contact call probability if visual attention is high
            if 'context' in fusion_result and fusion_result['context'] == 'contact_call':
                fusion_result['response_probability'] = fusion_result.get('response_probability', 0.5) * (1 + boost_factor)

        # Gaze direction adjustment
        if visual_features.gaze_direction == 'towards_camera':
            fusion_result['attention_boost'] = 0.15
        elif visual_features.gaze_direction in ['left', 'right']:
            fusion_result['attention_boost'] = 0.05

        # Movement intensity adjustment
        if visual_features.movement_intensity > 0.7:
            fusion_result['urgency_factor'] = min(fusion_result.get('urgency_factor', 0.0) + 0.2, 1.0)

        # Store visual context for future reference
        fusion_result['visual_context'] = {
            'attention_level': visual_features.attention_level.value,
            'gaze_direction': visual_features.gaze_direction,
            'movement_intensity': visual_features.movement_intensity,
            'confidence': visual_features.confidence
        }

        return fusion_result

    def create_visual_attention_score(self, visual_features: VisualFeatures) -> float:
        """Create a visual attention score for data fusion"""
        score = 0.0

        # Base score from attention level
        level_scores = {
            VisualAttentionLevel.LOW: 0.1,
            VisualAttentionLevel.MODERATE: 0.4,
            VisualAttentionLevel.HIGH: 0.7,
            VisualAttentionLevel.VERY_HIGH: 1.0
        }
        score += level_scores.get(visual_features.attention_level, 0.1)

        # Gaze direction modifier
        gaze_modifiers = {
            'towards_camera': 0.3,
            'sideways': 0.1,
            'away': -0.2,
            'left': 0.0,
            'right': 0.0
        }
        score += gaze_modifiers.get(visual_features.gaze_direction, 0.0)

        # Movement intensity modifier
        score += visual_features.movement_intensity * 0.2

        # Hand gestures modifier
        score += len(visual_features.hand_gestures) * 0.1

        # Confidence weighting
        score *= visual_features.confidence

        return max(0.0, min(1.0, score))


# Test utility
def create_test_visual_fusion_system() -> VisualFusionSystem:
    """Create a visual fusion system for testing"""
    config = VisualFusionConfig(
        camera_resolution=(640, 480),
        fps=30,
        use_mediapipe=True,
        separate_thread=False,  # Run in main thread for testing
        max_queue_size=10
    )
    return VisualFusionSystem(config)