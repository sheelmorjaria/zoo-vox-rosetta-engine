"""
Adaptive Resonance Enhancement
Phase IV Feature Implementation

Implements Adaptive Resonance Theory (ART) for stable, fast learning
in real-time animal communication analysis.
"""

import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class ResonanceData:
    """Data structure for resonance information"""

    signal: List[float]
    features: Dict[str, Any]
    context: Dict[str, Any]
    resonance_score: float
    timestamp: float


class ResonanceNetwork:
    """Core adaptive resonance network based on ART principles"""

    def __init__(self, vigilance: float = 0.8, learning_rate: float = 0.1):
        self.vigilance = vigilance
        self.learning_rate = learning_rate
        self.prototypes = []  # List of prototype patterns
        self.context_weights = {}
        self.activation_history = []

    def compute_resonance(self, input_signal: List[float], prototype_signal: List[float]) -> float:
        """Compute resonance between input and prototype"""
        # Normalize signals
        input_norm = np.array(input_signal)
        prototype_norm = np.array(prototype_signal)

        # Compute similarity (resonance)
        if np.linalg.norm(prototype_norm) == 0:
            return 0.0

        resonance = np.dot(input_norm, prototype_norm) / (
            np.linalg.norm(input_norm) * np.linalg.norm(prototype_norm)
        )

        return max(0.0, min(1.0, resonance))

    def learn_prototype(self, signal: List[float], context: Dict[str, Any]) -> None:
        """Learn new prototype based on input signal"""
        # Find best matching prototype
        best_match_idx = -1
        best_match_score = 0.0

        for i, prototype in enumerate(self.prototypes):
            score = self.compute_resonance(signal, prototype["signal"])
            if score > best_match_score:
                best_match_score = score
                best_match_idx = i

        # Check vigilance criterion
        if best_match_idx >= 0 and best_match_score >= self.vigilance:
            # Update existing prototype
            prototype = self.prototypes[best_match_idx]
            old_signal = prototype["signal"]

            # Adaptive learning
            new_signal = [
                old * (1 - self.learning_rate) + new * self.learning_rate
                for old, new in zip(old_signal, signal)
            ]

            prototype["signal"] = new_signal
            prototype["context"] = context
            prototype["last_updated"] = time.time()

        else:
            # Create new prototype
            self.prototypes.append(
                {
                    "signal": signal.copy(),
                    "context": context,
                    "created": time.time(),
                    "last_updated": time.time(),
                    "usage_count": 1,
                }
            )

    def get_prototypes(self) -> List[Dict[str, Any]]:
        """Get all learned prototypes"""
        return self.prototypes

    def get_best_match(self, signal: List[float]) -> Tuple[int, float]:
        """Get best matching prototype index and score"""
        best_idx = -1
        best_score = 0.0

        for i, prototype in enumerate(self.prototypes):
            score = self.compute_resonance(signal, prototype["signal"])
            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx, best_score


class AdaptiveFilter:
    """Adaptive signal filter for preprocessing"""

    def __init__(self, filter_order: int = 3):
        self.filter_order = filter_order
        self.coefficients = [1.0]  # Simple moving average initially
        self.adaptation_gain = 0.1
        self.signal_history = []

    def apply_filter(self, input_signal: List[float], filter_type: str = "lowpass") -> List[float]:
        """Apply adaptive filter to signal"""
        if filter_type == "lowpass":
            return self._lowpass_filter(input_signal)
        elif filter_type == "bandpass":
            return self._bandpass_filter(input_signal)
        else:
            return input_signal.copy()

    def _lowpass_filter(self, signal: List[float]) -> List[float]:
        """Simple low-pass filter"""
        if len(self.coefficients) == 0:
            return signal.copy()

        filtered = []
        for i in range(len(signal)):
            start_idx = max(0, i - len(self.coefficients) + 1)
            window = signal[start_idx : i + 1]
            filtered.append(sum(window) / len(window))

        return filtered

    def _bandpass_filter(self, signal: List[float]) -> List[float]:
        """Simple band-pass filter (combination of high and low pass)"""
        # High pass: remove DC component
        high_passed = [
            signal[i] - signal[i - 1] if i > 0 else signal[0] for i in range(len(signal))
        ]

        # Then low pass
        return self._lowpass_filter(high_passed)

    def adapt_to_signal(self, signal: List[float]) -> None:
        """Adapt filter coefficients to signal characteristics"""
        self.signal_history.extend(signal)

        # Keep only recent history
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-1000:]

        # Simple adaptation based on signal statistics
        if len(self.signal_history) > 10:
            variance = np.var(self.signal_history[-100:])

            # Adjust adaptation gain based on signal variance
            if variance > 0.1:
                self.adaptation_gain = min(0.5, self.adaptation_gain * 1.1)
            else:
                self.adaptation_gain = max(0.01, self.adaptation_gain * 0.9)

    def get_adaptation_gain(self) -> float:
        """Get current adaptation gain"""
        return self.adaptation_gain


class FeatureDetector:
    """Feature detection and extraction system"""

    def __init__(self):
        self.feature_cache = {}

    def extract_features(self, signal: List[float]) -> Dict[str, Any]:
        """Extract acoustic features from signal"""
        if not signal:
            return {}

        # Basic statistical features
        signal_array = np.array(signal)

        features = {
            "mean": float(np.mean(signal_array)),
            "std": float(np.std(signal_array)),
            "variance": float(np.var(signal_array)),
            "min": float(np.min(signal_array)),
            "max": float(np.max(signal_array)),
            "range": float(np.max(signal_array) - np.min(signal_array)),
            "rms": float(np.sqrt(np.mean(signal_array**2))),
            "zero_crossings": self._count_zero_crossings(signal),
            "energy": float(np.sum(signal_array**2)),
        }

        # Frequency domain features if signal is long enough
        if len(signal) > 10:
            features.update(self._extract_frequency_features(signal))

        return features

    def _count_zero_crossings(self, signal: List[float]) -> int:
        """Count zero crossings in signal"""
        crossings = 0
        for i in range(1, len(signal)):
            if signal[i - 1] * signal[i] < 0:
                crossings += 1
        return crossings

    def _extract_frequency_features(self, signal: List[float]) -> Dict[str, Any]:
        """Extract frequency domain features"""
        # Simple FFT-based features
        fft = np.fft.fft(signal)
        magnitude = np.abs(fft[: len(fft) // 2])

        if len(magnitude) > 0:
            return {
                "spectral_centroid": float(
                    np.sum(magnitude * np.arange(len(magnitude))) / np.sum(magnitude)
                ),
                "spectral_spread": float(
                    np.sqrt(
                        np.sum(
                            (
                                np.arange(len(magnitude))
                                - np.sum(magnitude * np.arange(len(magnitude))) / np.sum(magnitude)
                            )
                            ** 2
                            * magnitude
                        )
                        / np.sum(magnitude)
                    )
                ),
                "spectral_rolloff": float(self._compute_spectral_rolloff(magnitude)),
            }
        return {}

    def _compute_spectral_rolloff(
        self, magnitude: np.ndarray, rolloff_percent: float = 0.85
    ) -> float:
        """Compute spectral rolloff frequency"""
        total_energy = np.sum(magnitude)
        threshold = total_energy * rolloff_percent

        cumulative_energy = 0
        rolloff_idx = len(magnitude) - 1

        for i, energy in enumerate(magnitude):
            cumulative_energy += energy
            if cumulative_energy >= threshold:
                rolloff_idx = i
                break

        return float(rolloff_idx)

    def match_features(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """Match two feature sets using cosine similarity"""
        # Get common features
        common_keys = set(features1.keys()) & set(features2.keys())

        if not common_keys:
            return 0.0

        # Normalize feature vectors
        vec1 = []
        vec2 = []

        for key in common_keys:
            val1 = features1.get(key, 0)
            val2 = features2.get(key, 0)

            # Normalize by max values
            max_val = max(abs(val1), abs(val2), 1e-8)
            vec1.append(val1 / max_val)
            vec2.append(val2 / max_val)

        # Compute cosine similarity
        vec1_array = np.array(vec1)
        vec2_array = np.array(vec2)

        if np.linalg.norm(vec1_array) == 0 or np.linalg.norm(vec2_array) == 0:
            return 0.0

        similarity = np.dot(vec1_array, vec2_array) / (
            np.linalg.norm(vec1_array) * np.linalg.norm(vec2_array)
        )

        return max(0.0, min(1.0, similarity))


class ResonanceMatcher:
    """Match signals against learned prototypes"""

    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold

    def find_best_match(
        self, input_signal: List[float], prototype_signals: List[List[float]]
    ) -> Dict[str, Any]:
        """Find best matching prototype"""
        if not prototype_signals:
            return {"index": -1, "score": 0.0}

        best_idx = -1
        best_score = 0.0

        for i, prototype in enumerate(prototype_signals):
            # Simple dot product similarity
            score = self._compute_similarity(input_signal, prototype)

            if score > best_score:
                best_score = score
                best_idx = i

        return {"index": best_idx, "score": best_score}

    def _compute_similarity(self, signal1: List[float], signal2: List[float]) -> float:
        """Compute similarity between two signals"""
        if len(signal1) != len(signal2):
            # Pad shorter signal
            max_len = max(len(signal1), len(signal2))
            signal1 = signal1 + [0] * (max_len - len(signal1))
            signal2 = signal2 + [0] * (max_len - len(signal2))

        vec1 = np.array(signal1)
        vec2 = np.array(signal2)

        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0

        # Cosine similarity
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

        return max(0.0, min(1.0, similarity))

    def match_with_threshold(self, input_signal: List[float], threshold: float) -> bool:
        """Check if signal matches any prototype above threshold"""
        # For now, use dummy implementation
        # In real implementation, this would compare against all prototypes
        return random.random() < threshold


class StabilityMonitor:
    """Monitor system stability and performance"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.resonance_readings = []
        self.alert_threshold = 0.3
        self.stability_threshold = 0.7

    def add_resonance_reading(self, resonance_value: float) -> None:
        """Add resonance reading to stability monitoring"""
        self.resonance_readings.append(resonance_value)

        # Keep only recent readings
        if len(self.resonance_readings) > self.window_size:
            self.resonance_readings = self.resonance_readings[-self.window_size :]

    def get_stability(self) -> float:
        """Compute system stability metric"""
        if len(self.resonance_readings) < 10:
            return 1.0  # Assume stable with limited data

        # Compute coefficient of variation
        mean = np.mean(self.resonance_readings)
        std = np.std(self.resonance_readings)

        if mean == 0:
            return 0.0

        cv = std / mean
        stability = max(0.0, 1.0 - cv)  # Higher CV = lower stability

        return stability

    def is_stable(self, threshold: float = 0.7) -> bool:
        """Check if system is stable"""
        return self.get_stability() >= threshold

    def detect_anomalies(self) -> List[int]:
        """Detect anomalous readings"""
        anomalies = []
        if len(self.resonance_readings) < 10:
            return anomalies

        mean = np.mean(self.resonance_readings)
        std = np.std(self.resonance_readings)

        for i, reading in enumerate(self.resonance_readings):
            z_score = abs(reading - mean) / (std + 1e-8)
            if z_score > 3.0:  # 3-sigma rule
                anomalies.append(i)

        return anomalies


class FastLearning:
    """Fast learning mechanisms for critical periods"""

    def __init__(self, base_learning_rate: float = 0.1):
        self.base_learning_rate = base_learning_rate
        self.critical_learning_rate = base_learning_rate * 10
        self.learning_rates = {}

    def get_adaptive_learning_rate(self, base_rate: float, iteration: int) -> float:
        """Get adaptive learning rate based on iteration"""
        # Simple decay schedule
        decay_factor = 1.0 / (1.0 + 0.01 * iteration)
        return base_rate * decay_factor

    def is_critical_period(self, signal_variance: float, context_similarity: float) -> bool:
        """Detect if current conditions indicate critical period"""
        # High variance + high context similarity = critical period
        return signal_variance > 0.3 and context_similarity > 0.8

    def get_critical_learning_rate(self) -> float:
        """Get learning rate for critical periods"""
        return self.critical_learning_rate


class NoiseRobustness:
    """Noise robustness features"""

    def __init__(self):
        self.noise_floor = 0.1
        self.adaptation_rate = 0.1

    def denoise_signal(self, noisy_signal: List[float], noise_level: float) -> List[float]:
        """Apply denoising to signal"""
        if noise_level < 0.1:
            return noisy_signal.copy()

        # Simple moving average denoising
        window_size = max(3, int(noise_level * 10))
        denoised = []

        for i in range(len(noisy_signal)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(noisy_signal), i + window_size // 2 + 1)
            window = noisy_signal[start_idx:end_idx]
            denoised.append(np.mean(window))

        return denoised

    def adapt_to_noise_level(self, noise_level: float) -> None:
        """Adapt to current noise level"""
        self.noise_floor = max(0.05, min(0.5, noise_level))

    def get_robustness_level(self) -> float:
        """Get current robustness level"""
        # Higher noise floor = higher robustness needed
        return 1.0 - (self.noise_floor / 0.5)


class ContextualModulation:
    """Context-based modulation of resonance"""

    def __init__(self):
        self.context_weights = {}
        self.default_weight = 1.0

    def apply_context_modulation(self, base_resonance: float, context: Dict[str, Any]) -> float:
        """Apply context-based modulation to resonance"""
        modulation_factor = self.default_weight

        # Social context modulation
        if context.get("social", False):
            modulation_factor *= 1.2  # Boost social contexts

        # Alert level modulation
        alert_level = context.get("alert_level", 0.5)
        modulation_factor *= 0.8 + 0.4 * alert_level

        # Time-based modulation
        time_of_day = context.get("time", 12) / 24.0  # Normalize to 0-1
        modulation_factor *= 0.9 + 0.2 * np.sin(time_of_day * 2 * np.pi)

        # Apply modulation
        return base_resonance * modulation_factor

    def learn_context_influence(self, context: Dict[str, Any], resonance_strength: float) -> None:
        """Learn influence of specific context patterns"""
        context_key = frozenset(context.items())
        self.context_weights[context_key] = resonance_strength


class DynamicAdaptation:
    """Dynamic adaptation to changing conditions"""

    def __init__(self):
        self.adaptation_parameters = {
            "learning_rate": 0.1,
            "vigilance": 0.8,
            "noise_threshold": 0.2,
        }
        self.adaptation_history = []

    def adapt_to_conditions(
        self, signal_statistics: Dict[str, float], performance_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Adapt to current signal and performance conditions"""
        # Extract statistics
        signal_variance = signal_statistics.get("variance", 0.1)
        performance_accuracy = performance_metrics.get("accuracy", 0.5)

        adaptation = {}

        # Adjust learning rate based on signal variance
        if signal_variance > 0.3:
            self.adaptation_parameters["learning_rate"] = min(
                0.5, self.adaptation_parameters["learning_rate"] * 1.1
            )
        else:
            self.adaptation_parameters["learning_rate"] = max(
                0.01, self.adaptation_parameters["learning_rate"] * 0.9
            )

        # Adjust vigilance based on performance
        if performance_accuracy < 0.7:
            self.adaptation_parameters["vigilance"] = min(
                0.95, self.adaptation_parameters["vigilance"] * 0.9
            )
        else:
            self.adaptation_parameters["vigilance"] = max(
                0.5, self.adaptation_parameters["vigilance"] * 1.05
            )

        adaptation["parameters_changed"] = self.adaptation_parameters.copy()

        # Record adaptation
        self.adaptation_history.append(
            {
                "timestamp": time.time(),
                "signal_variance": signal_variance,
                "performance_accuracy": performance_accuracy,
                "parameters_before": self.adaptation_parameters.copy(),
            }
        )

        return adaptation

    def get_adaptation_rate(self) -> float:
        """Get current adaptation rate"""
        return self.adaptation_parameters["learning_rate"]


class ResonanceOptimizer:
    """Optimize resonance parameters"""

    def __init__(self):
        self.optimization_history = []
        self.best_parameters = {}

    def optimize_parameters(
        self, current_params: Dict[str, float], performance_data: List[float]
    ) -> Dict[str, float]:
        """Optimize parameters based on performance data"""
        if not performance_data:
            return current_params

        # Simple gradient ascent on performance
        mean_performance = np.mean(performance_data)
        std_performance = np.std(performance_data)

        optimized_params = current_params.copy()

        # Adjust parameters based on performance
        for param_name, param_value in current_params.items():
            # Simple heuristic: increase good parameters, decrease bad ones
            performance_impact = mean_performance / (std_performance + 1e-8)

            if performance_impact > 1.0:
                optimized_params[param_name] = min(1.0, param_value * 1.1)
            else:
                optimized_params[param_name] = max(0.01, param_value * 0.9)

        # Record optimization
        self.optimization_history.append(
            {
                "timestamp": time.time(),
                "before_params": current_params,
                "after_params": optimized_params,
                "performance_mean": mean_performance,
                "performance_std": std_performance,
            }
        )

        return optimized_params

    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history"""
        return self.optimization_history


class ResonanceValidator:
    """Validate resonance computations and results"""

    @staticmethod
    def validate_resonance(resonance_value: float, threshold: float) -> bool:
        """Validate if resonance value meets threshold"""
        return resonance_value >= threshold

    @staticmethod
    def compute_resonance_statistics(resonance_values: List[float]) -> Dict[str, float]:
        """Compute statistics for resonance values"""
        if not resonance_values:
            return {}

        return {
            "mean": float(np.mean(resonance_values)),
            "std": float(np.std(resonance_values)),
            "min": float(np.min(resonance_values)),
            "max": float(np.max(resonance_values)),
            "median": float(np.median(resonance_values)),
            "q25": float(np.percentile(resonance_values, 25)),
            "q75": float(np.percentile(resonance_values, 75)),
        }


class AdaptiveResonance:
    """Main Adaptive Resonance System"""

    def __init__(self):
        self.resonance_network = ResonanceNetwork()
        self.feature_detector = FeatureDetector()
        self.adaptive_filter = AdaptiveFilter()
        self.resonance_matcher = ResonanceMatcher()
        self.stability_monitor = StabilityMonitor()
        self.fast_learning = FastLearning()
        self.noise_robustness = NoiseRobustness()
        self.contextual_modulation = ContextualModulation()
        self.dynamic_adaptation = DynamicAdaptation()
        self.resonance_optimizer = ResonanceOptimizer()
        self.validator = ResonanceValidator()

        self.real_time_mode = False
        self.processing_count = 0
        self.adaptation_progress = {}

    def initialize_resonance_system(self) -> None:
        """Initialize the adaptive resonance system"""
        # Initialize components
        self.resonance_network = ResonanceNetwork(vigilance=0.8, learning_rate=0.1)
        self.stability_monitor = StabilityMonitor()

    def process_adaptive_resonance(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input through adaptive resonance system"""
        signal = input_data.get("signal", [])
        features = input_data.get("features", {})
        context = input_data.get("context", {})

        # Pre-processing
        filtered_signal = self.adaptive_filter.apply_filter(signal)

        # Feature extraction
        if not features:
            features = self.feature_detector.extract_features(filtered_signal)

        # Compute resonance
        prototypes = self.resonance_network.get_prototypes()
        prototype_signals = [p["signal"] for p in prototypes]

        if prototype_signals:
            best_match = self.resonance_matcher.find_best_match(filtered_signal, prototype_signals)
            resonance_score = best_match["score"]
        else:
            resonance_score = 0.0

        # Apply contextual modulation
        modulated_resonance = self.contextual_modulation.apply_context_modulation(
            resonance_score, context
        )

        # Learn from this input
        self.resonance_network.learn_prototype(filtered_signal, context)

        # Update stability monitor
        self.stability_monitor.add_resonance_reading(modulated_resonance)

        # Update processing count
        self.processing_count += 1

        return {
            "resonance_score": modulated_resonance,
            "adapted_features": features,
            "filtered_signal": filtered_signal,
            "prototypes_count": len(prototypes),
            "stability": self.stability_monitor.get_stability(),
            "processing_count": self.processing_count,
        }

    def enable_real_time_mode(self) -> None:
        """Enable real-time processing mode"""
        self.real_time_mode = True
        self.processing_count = 0

    def real_time_process(self, signal: List[float], context: Dict[str, Any]) -> Dict[str, Any]:
        """Process single real-time input"""
        input_data = {"signal": signal, "features": {}, "context": context}

        return self.process_adaptive_resonance(input_data)

    def get_adaptation_progress(self) -> Dict[str, Any]:
        """Get adaptation progress metrics"""
        return {
            "iterations_completed": self.processing_count,
            "prototypes_learned": len(self.resonance_network.get_prototypes()),
            "current_stability": self.stability_monitor.get_stability(),
            "learning_rate": self.dynamic_adaptation.get_adaptation_rate(),
        }

    def get_system_stability(self) -> float:
        """Get overall system stability"""
        return self.stability_monitor.get_stability()

    def reset_system(self) -> None:
        """Reset the adaptive resonance system"""
        self.__init__()
