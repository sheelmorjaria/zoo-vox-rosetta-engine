"""
A/B Testing Controller Module
============================

Implements an advanced A/B Testing Controller with Blind Mode for unbiased
evaluation of algorithmic improvements and system enhancements.

Key Features:
- Multi-variant testing support (A/B, A/B/n testing)
- Blind mode for unbiased evaluation
- Statistical significance testing with p-values
- Real-time monitoring and dashboard
- Participant assignment and tracking
- Results export and analysis
- Confidence intervals and power analysis
- Randomized assignment with blocking
- Sample size calculation

Architecture:
```
ABTestingController
├── VariantManager
│   ├── Variant creation and management
│   └── Parameter validation
├── ParticipantAssigner
│   ├── Random assignment
│   ├── Blocking strategy
│   └── Assignment tracking
├── StatisticalAnalyzer
│   ├── Significance testing
│   ├── Confidence intervals
│   └── Power analysis
├── ResultTracker
│   ├── Result recording
│   └── Real-time updates
└── ExportManager
    ├── JSON export
    ├── CSV export
    └── Statistical summary
```

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import csv
import json
import logging
import math
import random
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import scipy.stats as stats


class TestStatus(Enum):
    """Test execution status"""

    NOT_STARTED = 1
    RUNNING = 2
    COMPLETED = 3
    PAUSED = 4
    CANCELLED = 5


class AssignmentStrategy(Enum):
    """Assignment strategies for participants"""

    RANDOM = 1
    WEIGHTED_RANDOM = 2
    BLOCKED_RANDOM = 3
    THROTTLED = 4


@dataclass
class Variant:
    """A/B test variant configuration"""

    variant_id: str
    name: str
    parameters: Dict[str, Any]
    weight: float = 1.0
    is_control: bool = False

    def __post_init__(self):
        if not self.variant_id:
            raise ValueError("Variant ID is required")
        if self.weight <= 0:
            raise ValueError("Weight must be positive")


@dataclass
class Participant:
    """Test participant information"""

    participant_id: str
    assigned_variant: str
    variant_name: str
    parameters: Dict[str, Any]
    assignment_timestamp: float
    group_metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.group_metadata is None:
            self.group_metadata = {}


@dataclass
class TestResult:
    """Single test result"""

    participant_id: str
    variant_id: str
    timestamp: float
    success: bool
    metrics: Dict[str, Any] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}


class ABTestingController:
    """Main A/B Testing Controller with Blind Mode"""

    def __init__(
        self,
        experiment_name: str,
        blind_mode: bool = True,
        significance_threshold: float = 0.05,
        confidence_level: float = 0.95,
        assignment_strategy: AssignmentStrategy = AssignmentStrategy.RANDOM,
        min_sample_size: int = 30,
    ):
        """
        Initialize A/B Testing Controller

        Args:
            experiment_name: Name of the experiment
            blind_mode: Enable blind mode to hide variant details
            significance_threshold: Threshold for statistical significance
            confidence_level: Confidence level for confidence intervals
            assignment_strategy: How to assign participants to variants
            min_sample_size: Minimum sample size per variant
        """
        self.experiment_name = experiment_name
        self.blind_mode = blind_mode
        self.significance_threshold = significance_threshold
        self.confidence_level = confidence_level
        self.assignment_strategy = assignment_strategy
        self.min_sample_size = min_sample_size

        # Core data structures
        self.variants: Dict[str, Variant] = {}
        self.participant_assignments: Dict[str, Participant] = {}
        self.test_results: Dict[str, List[TestResult]] = defaultdict(list)

        # State management
        self.status = TestStatus.NOT_STARTED
        self.experiment_start_time = None
        self.experiment_end_time = None

        # Configuration
        self.participant_group_size = 100
        self.rebalancing_interval = 3600  # 1 hour
        self.last_rebalance_time = time.time()

        # Threading (use RLock for reentrant locking)
        self._lock = threading.RLock()
        self.result_buffer = deque(maxlen=1000)

        # Logging
        self.logger = logging.getLogger(__name__)
        self.experiment_id = str(uuid.uuid4())[:8]

        # Statistical tracking
        self.success_counts = defaultdict(int)
        self.failure_counts = defaultdict(int)
        self.total_counts = defaultdict(int)
        self.metrics_history = defaultdict(lambda: deque(maxlen=100))

    def create_variant(
        self,
        variant_id: str,
        name: str,
        parameters: Dict[str, Any],
        weight: float = 1.0,
        is_control: bool = False,
    ) -> Variant:
        """Create a new test variant"""
        with self._lock:
            if variant_id in self.variants:
                raise ValueError(f"Variant {variant_id} already exists")

            variant = Variant(
                variant_id=variant_id,
                name=name,
                parameters=parameters,
                weight=weight,
                is_control=is_control,
            )

            self.variants[variant_id] = variant
            self.success_counts[variant_id] = 0
            self.failure_counts[variant_id] = 0
            self.total_counts[variant_id] = 0

            self.logger.info(f"Created variant {variant_id}: {name}")
            return variant

    def assign_participant(self, participant_id: str, group_metadata: Dict[str, Any] = None) -> str:
        """Assign a participant to a variant"""
        with self._lock:
            if participant_id in self.participant_assignments:
                return self.participant_assignments[participant_id].assigned_variant

            if not self.variants:
                raise ValueError("No variants available for assignment")

            # Choose variant based on assignment strategy
            variant_id = self._select_variant_for_participant()

            # Get variant information
            variant = self.variants[variant_id]

            # Create participant record
            participant = Participant(
                participant_id=participant_id,
                assigned_variant=variant_id,
                variant_name=f"Variant {variant_id}" if self.blind_mode else variant.name,
                parameters={} if self.blind_mode else variant.parameters.copy(),
                assignment_timestamp=time.time(),
                group_metadata=group_metadata or {},
            )

            # Store assignment
            self.participant_assignments[participant_id] = participant

            self.logger.info(f"Assigned participant {participant_id} to variant {variant_id}")
            return variant_id

    def _select_variant_for_participant(self) -> str:
        """Select variant for participant based on assignment strategy"""
        if self.assignment_strategy == AssignmentStrategy.RANDOM:
            return random.choice(list(self.variants.keys()))

        elif self.assignment_strategy == AssignmentStrategy.WEIGHTED_RANDOM:
            weights = [variant.weight for variant in self.variants.values()]
            return random.choices(list(self.variants.keys()), weights=weights)[0]

        elif self.assignment_strategy == AssignmentStrategy.BLOCKED_RANDOM:
            # Implement blocking based on current counts
            counts = [self.total_counts.get(variant_id, 0) for variant_id in self.variants.keys()]
            min_count = min(counts) if counts else 0

            # Select from variants with minimum count to maintain balance
            balanced_variants = [
                variant_id
                for variant_id, count in zip(self.variants.keys(), counts)
                if count == min_count
            ]
            return random.choice(balanced_variants)

        else:
            # Default to random
            return random.choice(list(self.variants.keys()))

    def record_result(self, participant_id: str, success: bool, metrics: Dict[str, Any] = None):
        """Record a test result for a participant"""
        with self._lock:
            if participant_id not in self.participant_assignments:
                raise ValueError(f"Participant {participant_id} not assigned to any variant")

            participant = self.participant_assignments[participant_id]
            variant_id = participant.assigned_variant

            # Create result
            result = TestResult(
                participant_id=participant_id,
                variant_id=variant_id,
                timestamp=time.time(),
                success=success,
                metrics=metrics or {},
            )

            # Store result
            self.test_results[variant_id].append(result)
            self.result_buffer.append(result)

            # Update counts
            self.total_counts[variant_id] += 1
            if success:
                self.success_counts[variant_id] += 1
            else:
                self.failure_counts[variant_id] += 1

            # Store metrics history
            if metrics:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self.metrics_history[f"{variant_id}_{key}"].append(value)

            # Check if we need to rebalance
            self._check_rebalance_needed()

            self.logger.debug(f"Recorded result for {participant_id}: {success}")

    def calculate_significance(self, variant_a: str, variant_b: str) -> Dict[str, float]:
        """Calculate statistical significance between two variants"""
        with self._lock:
            # Get results
            results_a = self.test_results[variant_a]
            results_b = self.test_results[variant_b]

            if not results_a or not results_b:
                return {"p_value": 1.0, "significant": False}

            # Extract success rates
            successes_a = sum(1 for r in results_a if r.success)
            successes_b = sum(1 for r in results_b if r.success)

            n_a = len(results_a)
            n_b = len(results_b)

            # Calculate proportions
            p_a = successes_a / n_a if n_a > 0 else 0
            p_b = successes_b / n_b if n_b > 0 else 0

            # Two-proportion z-test
            if n_a > 0 and n_b > 0:
                # Pooled proportion
                p_pooled = (successes_a + successes_b) / (n_a + n_b)

                # Standard error
                se = math.sqrt(p_pooled * (1 - p_pooled) * (1 / n_a + 1 / n_b))

                if se > 0:
                    z_score = (p_b - p_a) / se
                    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                else:
                    p_value = 1.0
            else:
                p_value = 1.0

            # Determine significance
            significant = p_value < self.significance_threshold

            return {
                "p_value": p_value,
                "significant": significant,
                "success_rate_a": p_a,
                "success_rate_b": p_b,
                "sample_size_a": n_a,
                "sample_size_b": n_b,
            }

    def calculate_confidence_intervals(self, variant_id: str) -> Dict[str, Any]:
        """Calculate confidence intervals for a variant"""
        with self._lock:
            results = self.test_results[variant_id]
            if not results:
                return {}

            successes = sum(1 for r in results if r.success)
            n = len(results)

            if n == 0:
                return {}

            p = successes / n

            # Wilson score interval
            z = stats.norm.ppf((1 + self.confidence_level) / 2)
            denominator = 1 + z**2 / n
            centre = p + z**2 / (2 * n)
            width = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)

            lower = (centre - width) / denominator
            upper = (centre + width) / denominator

            return {
                "success_rate": p,
                "confidence_interval": [max(0, lower), min(1, upper)],
                "sample_size": n,
                "confidence_level": self.confidence_level,
            }

    def get_participant_info(self, participant_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a participant"""
        with self._lock:
            if participant_id not in self.participant_assignments:
                return None

            participant = self.participant_assignments[participant_id]

            # Get participant results
            variant_results = self.test_results.get(participant.assigned_variant, [])

            # Calculate basic stats
            total_results = len(variant_results)
            successful_results = sum(1 for r in variant_results if r.success)

            info = {
                "participant_id": participant.participant_id,
                "assigned_variant": participant.assigned_variant,
                "variant_name": participant.variant_name,
                "assignment_timestamp": participant.assignment_timestamp,
                "total_results": total_results,
                "successful_results": successful_results,
                "success_rate": successful_results / total_results if total_results > 0 else 0,
                "group_metadata": participant.group_metadata,
            }

            return info

    def get_experiment_stats(self) -> Dict[str, Any]:
        """Get comprehensive experiment statistics"""
        with self._lock:
            # Calculate overall statistics
            total_participants = len(self.participant_assignments)
            total_results = sum(len(results) for results in self.test_results.values())

            # Calculate variant statistics
            variant_stats = {}
            for variant_id, variant in self.variants.items():
                total = self.total_counts.get(variant_id, 0)
                successes = self.success_counts.get(variant_id, 0)
                failures = self.failure_counts.get(variant_id, 0)

                variant_stats[variant_id] = {
                    "name": variant.name,
                    "total_participants": total,
                    "successful_results": successes,
                    "failed_results": failures,
                    "success_rate": successes / total if total > 0 else 0,
                    "parameters": variant.parameters.copy(),
                }

            # Calculate significance between variants
            significance_tests = {}
            variant_ids = list(self.variants.keys())
            for i, variant_a in enumerate(variant_ids):
                for variant_b in variant_ids[i + 1 :]:
                    if variant_a in self.variants and variant_b in self.variants:
                        significance = self.calculate_significance(variant_a, variant_b)
                        significance_tests[f"{variant_a}_vs_{variant_b}"] = significance

            return {
                "experiment_name": self.experiment_name,
                "experiment_id": self.experiment_id,
                "status": self.status.name,
                "total_participants": total_participants,
                "total_results": total_results,
                "blind_mode": self.blind_mode,
                "variants": variant_stats,
                "significance_tests": significance_tests,
                "start_time": self.experiment_start_time,
                "end_time": self.experiment_end_time,
                "duration_seconds": (self.experiment_end_time or time.time())
                - (self.experiment_start_time or 0),
            }

    def export_results(
        self, format: str = "json", filepath: str = None
    ) -> Union[str, Dict[str, Any]]:
        """Export experiment results"""
        with self._lock:
            stats = self.get_experiment_stats()

            if format.lower() == "json":
                if filepath:
                    with open(filepath, "w") as f:
                        json.dump(stats, f, indent=2, default=str)
                    return filepath
                else:
                    return stats

            elif format.lower() == "csv":
                if not filepath:
                    filepath = f"ab_test_results_{self.experiment_name}.csv"

                with open(filepath, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Variant", "Total", "Successes", "Failures", "Success Rate"])

                    for variant_id, variant_data in stats["variants"].items():
                        writer.writerow(
                            [
                                variant_id,
                                variant_data["total_participants"],
                                variant_data["successful_results"],
                                variant_data["failed_results"],
                                f"{variant_data['success_rate']:.4f}",
                            ]
                        )

                return filepath

            else:
                raise ValueError(f"Unsupported export format: {format}")

    def start_experiment(self):
        """Start the A/B test"""
        with self._lock:
            if self.status == TestStatus.RUNNING:
                return

            self.status = TestStatus.RUNNING
            self.experiment_start_time = time.time()
            self.logger.info(f"Started A/B test: {self.experiment_name}")

    def pause_experiment(self):
        """Pause the A/B test"""
        with self._lock:
            if self.status == TestStatus.RUNNING:
                self.status = TestStatus.PAUSED
                self.logger.info(f"Paused A/B test: {self.experiment_name}")

    def resume_experiment(self):
        """Resume the A/B test"""
        with self._lock:
            if self.status == TestStatus.PAUSED:
                self.status = TestStatus.RUNNING
                self.logger.info(f"Resumed A/B test: {self.experiment_name}")

    def stop_experiment(self):
        """Stop the A/B test"""
        with self._lock:
            self.status = TestStatus.COMPLETED
            self.experiment_end_time = time.time()
            self.logger.info(f"Stopped A/B test: {self.experiment_name}")

    def _check_rebalance_needed(self):
        """Check if participant assignment needs rebalancing"""
        current_time = time.time()

        if current_time - self.last_rebalance_time > self.rebalancing_interval:
            self._rebalance_assignments()
            self.last_rebalance_time = current_time

    def _rebalance_assignments(self):
        """Rebalance participant assignments if needed"""
        # Calculate current distribution
        variant_counts = {
            variant_id: len(self.test_results[variant_id]) for variant_id in self.variants.keys()
        }

        # Check if rebalancing is needed
        max_count = max(variant_counts.values())
        min_count = min(variant_counts.values())

        if max_count - min_count > self.participant_group_size // 2:
            self.logger.info(f"Rebalancing assignments. Distribution: {variant_counts}")
            # In a real implementation, you might implement rebalancing logic here
            # For now, just log the need for rebalancing

    def calculate_required_sample_size(
        self, baseline_rate: float, minimum_detectable_effect: float, power: float = 0.8
    ) -> int:
        """Calculate required sample size for desired power"""
        # Using standard sample size calculation for two proportions
        z_alpha = stats.norm.ppf(1 - self.significance_threshold / 2)
        z_beta = stats.norm.ppf(power)

        p1 = baseline_rate
        p2 = baseline_rate + minimum_detectable_effect

        # Standard formula for two-sample proportion test
        numerator = (
            z_alpha * math.sqrt(2 * p1 * (1 - p1))
            + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
        ) ** 2

        denominator = (p2 - p1) ** 2

        if denominator > 0:
            required_per_group = math.ceil(numerator / denominator)
        else:
            required_per_group = self.min_sample_size

        return max(required_per_group, self.min_sample_size)


# Test utility function
def create_test_ab_testing_controller() -> ABTestingController:
    """Create an A/B Testing Controller for testing"""
    controller = ABTestingController(
        experiment_name="test_experiment", blind_mode=True, significance_threshold=0.05
    )

    # Create test variants
    controller.create_variant("A", "Control", {"method": "traditional"})
    controller.create_variant("B", "Treatment", {"method": "enhanced"})

    return controller
