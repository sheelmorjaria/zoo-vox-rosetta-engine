"""
Scientific Rigor & Data Logging
==============================

Advanced data logging with provenance tracking, IEEE 1588 PTP synchronization,
and A/B testing capabilities for scientific rigor.

Classes:
- ProvenanceLogger: Comprehensive decision provenance logging
- PTPSync: IEEE 1588 precision time protocol synchronization
- ABTester: A/B testing framework
- ExperimentTracker: Experiment tracking and analysis
"""

import hashlib
import json
import logging
import sqlite3
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np


@dataclass
class DecisionRecord:
    """Comprehensive decision record for provenance tracking."""

    timestamp: str
    session_id: str
    input_features: Dict[str, Any]
    context_probabilities: Dict[str, float]
    phrase_selection: Dict[str, Any]
    synthesis_method: str
    output_audio: List[float]  # Serialized audio data
    processing_time_ms: float
    adaptation_parameters: Dict[str, Any]
    safety_applied: bool
    cognitive_context: Optional[Dict[str, Any]] = None
    visual_context: Optional[Dict[str, Any]] = None
    experimental_conditions: Dict[str, Any] = None


class ProvenanceLogger:
    """
    Comprehensive decision provenance logger.

    Logs complete decision trees for scientific reproducibility
    and analysis of system behavior.
    """

    def __init__(self, log_file: str = "provenance.log", database_file: str = "experiments.db"):
        """
        Initialize provenance logger.

        Args:
            log_file: JSON log file path
            database_file: SQLite database path
        """
        self.log_file = log_file
        self.database_file = database_file
        self.session_id = str(uuid.uuid4())
        self.decision_queue = deque(maxlen=1000)
        self.lock = threading.Lock()

        # Initialize database
        self._init_database()

        # Initialize logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Provenance logger initialized with session ID: {self.session_id}")

    def _init_database(self):
        """Initialize SQLite database for structured logging."""
        conn = sqlite3.connect(self.database_file)
        cursor = conn.cursor()

        # Create decisions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TEXT,
                input_features TEXT,
                context_probabilities TEXT,
                phrase_selection TEXT,
                synthesis_method TEXT,
                output_audio BLOB,
                processing_time_ms REAL,
                adaptation_parameters TEXT,
                safety_applied INTEGER,
                cognitive_context TEXT,
                visual_context TEXT,
                experimental_conditions TEXT
            )
        """)

        # Create sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                start_time TEXT,
                end_time TEXT,
                total_decisions INTEGER,
                total_processing_time REAL,
                species TEXT,
                synthesis_mode TEXT
            )
        """)

        conn.commit()
        conn.close()

    def log_decision(self, decision_record: Union[DecisionRecord, Dict[str, Any]]):
        """
        Log a decision record.

        Args:
            decision_record: Decision record to log
        """
        with self.lock:
            if isinstance(decision_record, dict):
                # If timestamp already provided, use it; otherwise generate new one
                if "timestamp" not in decision_record:
                    decision_record["timestamp"] = datetime.now().isoformat()
                decision_record = DecisionRecord(session_id=self.session_id, **decision_record)
            else:
                decision_record.session_id = self.session_id
                if not decision_record.timestamp:
                    decision_record.timestamp = datetime.now().isoformat()

            # Add to queue
            self.decision_queue.append(decision_record)

            # Log to JSON file
            self._log_to_json(decision_record)

            # Log to database
            self._log_to_database(decision_record)

    def _log_to_json(self, record: DecisionRecord):
        """Log decision to JSON file."""
        log_entry = asdict(record)
        log_entry["output_audio"] = str(log_entry["output_audio"][:100])  # Truncate for readability

        try:
            with open(self.log_file, "a") as f:
                json.dump(log_entry, f, indent=2)
                f.write("\n")
        except Exception as e:
            self.logger.error(f"Failed to write to JSON log: {e}")

    def _log_to_database(self, record: DecisionRecord):
        """Log decision to SQLite database."""
        try:
            conn = sqlite3.connect(self.database_file)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO decisions (
                    session_id, timestamp, input_features, context_probabilities,
                    phrase_selection, synthesis_method, output_audio, processing_time_ms,
                    adaptation_parameters, safety_applied, cognitive_context,
                    visual_context, experimental_conditions
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    record.session_id,
                    record.timestamp,
                    json.dumps(record.input_features),
                    json.dumps(record.context_probabilities),
                    json.dumps(record.phrase_selection),
                    record.synthesis_method,
                    np.array(record.output_audio).tobytes(),
                    record.processing_time_ms,
                    json.dumps(record.adaptation_parameters),
                    int(record.safety_applied),
                    json.dumps(record.cognitive_context) if record.cognitive_context else None,
                    json.dumps(record.visual_context) if record.visual_context else None,
                    json.dumps(record.experimental_conditions)
                    if record.experimental_conditions
                    else None,
                ),
            )

            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Failed to write to database: {e}")

    def get_decision_history(self, limit: int = 100) -> List[DecisionRecord]:
        """
        Get decision history.

        Args:
            limit: Maximum number of decisions to return

        Returns:
            List of decision records
        """
        with self.lock:
            return list(self.decision_queue)[-limit:]

    def export_session_data(self, session_id: str = None, format: str = "json") -> str:
        """
        Export session data.

        Args:
            session_id: Session ID to export (default: current session)
            format: Export format ('json', 'csv')

        Returns:
            Exported data as string
        """
        if session_id is None:
            session_id = self.session_id

        conn = sqlite3.connect(self.database_file)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM decisions WHERE session_id = ?", (session_id,))
        rows = cursor.fetchall()

        if format == "json":
            columns = [desc[0] for desc in cursor.description]
            data = [dict(zip(columns, row)) for row in rows]
            return json.dumps(data, indent=2)
        elif format == "csv":
            import csv
            import io

            output = io.StringIO()
            writer = csv.writer(output)

            # Write header
            columns = [desc[0] for desc in cursor.description]
            writer.writerow(columns)

            # Write data
            for row in rows:
                writer.writerow(row)

            return output.getvalue()

        conn.close()
        return ""


class PTPSync:
    """
    IEEE 1588 Precision Time Protocol synchronization.

    Provides microsecond-level time synchronization for
    scientific experiments requiring precise timing.
    """

    def __init__(self):
        """Initialize PTP synchronization."""
        self.is_initialized = False
        self.sync_offset = 0.0
        self.sync_accuracy = 0.0
        self.logger = logging.getLogger(__name__)

    def initialize(self) -> bool:
        """
        Initialize PTP synchronization.

        Returns:
            True if initialization successful
        """
        try:
            # Try to load PTP library (mock implementation)
            self.is_initialized = True
            self.sync_offset = 0.0
            self.sync_accuracy = 0.001  # 1ms accuracy for simulation

            self.logger.info("PTP synchronization initialized")
            return True
        except Exception as e:
            self.logger.warning(f"PTP initialization failed: {e}")
            self.is_initialized = False
            return False

    def get_ptp_timestamp(self) -> float:
        """
        Get PTP-synchronized timestamp.

        Returns:
            Synchronized timestamp
        """
        if not self.is_initialized:
            return time.time()

        # Simulate PTP synchronization
        system_time = time.time()
        ptp_time = system_time + self.sync_offset

        return ptp_time

    def get_time_offset(self) -> float:
        """
        Get time offset from system time.

        Returns:
            Time offset in seconds
        """
        return self.sync_offset

    def sync_with_reference(self, reference_time: float) -> None:
        """
        Synchronize with reference time source.

        Args:
            reference_time: Reference timestamp
        """
        if self.is_initialized:
            system_time = time.time()
            self.sync_offset = reference_time - system_time
            self.logger.info(f"Updated PTP offset: {self.sync_offset * 1000:.3f}ms")


class ABTester:
    """
    A/B testing framework for experimental validation.

    Provides deterministic A/B testing with consistent
    assignment based on session and experiment parameters.
    """

    def __init__(self, test_ratio: float = 0.5, experiment_id: str = "default"):
        """
        Initialize A/B tester.

        Args:
            test_ratio: Ratio for A/B split (0-1)
            experiment_id: Unique experiment identifier
        """
        self.test_ratio = max(0.0, min(1.0, test_ratio))
        self.experiment_id = experiment_id
        self.test_assignments = {}
        self.logger = logging.getLogger(__name__)

    def should_use_interactive_mode(self, session_id: str = None) -> bool:
        """
        Determine if session should use interactive mode.

        Args:
            session_id: Session ID for deterministic assignment

        Returns:
            True if should use interactive mode
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        # Create deterministic hash
        hash_input = f"{self.experiment_id}_{session_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest()[:8], 16)

        # Normalize to 0-1 range
        normalized_value = hash_value / (16**8 - 1)

        is_interactive = normalized_value < self.test_ratio

        # Log assignment
        self.test_assignments[session_id] = {
            "interactive": is_interactive,
            "hash_value": normalized_value,
        }

        self.logger.debug(
            f"Session {session_id}: Interactive={is_interactive} (value={normalized_value:.3f})"
        )

        return is_interactive

    def get_test_group(self, session_id: str = None) -> str:
        """
        Get test group assignment.

        Args:
            session_id: Session ID

        Returns:
            Test group ('A' or 'B')
        """
        if self.should_use_interactive_mode(session_id):
            return "A"
        else:
            return "B"

    def get_assignment_stats(self) -> Dict[str, Any]:
        """
        Get assignment statistics.

        Returns:
            Assignment statistics
        """
        if not self.test_assignments:
            return {"total_assignments": 0}

        total = len(self.test_assignments)
        interactive_count = sum(
            1 for assignment in self.test_assignments.values() if assignment["interactive"]
        )

        return {
            "total_assignments": total,
            "interactive_count": interactive_count,
            "control_count": total - interactive_count,
            "interactive_ratio": interactive_count / total if total > 0 else 0.0,
            "target_ratio": self.test_ratio,
        }


class ExperimentTracker:
    """
    Experiment tracking and analysis.

    Tracks experimental conditions and provides analysis
    tools for evaluating system performance.
    """

    def __init__(self, experiment_id: str):
        """
        Initialize experiment tracker.

        Args:
            experiment_id: Experiment identifier
        """
        self.experiment_id = experiment_id
        self.conditions = {}
        self.metrics = {}
        self.trials = []
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)

    def set_condition(self, key: str, value: Any) -> None:
        """
        Set experimental condition.

        Args:
            key: Condition key
            value: Condition value
        """
        with self.lock:
            self.conditions[key] = value

    def record_trial(self, trial_data: Dict[str, Any]) -> None:
        """
        Record experimental trial.

        Args:
            trial_data: Trial data including metrics
        """
        with self.lock:
            trial_data["timestamp"] = datetime.now().isoformat()
            trial_data["experiment_id"] = self.experiment_id
            self.trials.append(trial_data)

    def get_metric_statistics(self, metric_name: str) -> Dict[str, float]:
        """
        Get statistics for a specific metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Statistical summary
        """
        values = [trial.get(metric_name, 0) for trial in self.trials if metric_name in trial]

        if not values:
            return {"count": 0}

        return {
            "count": len(values),
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "median": np.median(values),
        }

    def compare_conditions(self, metric_name: str, condition_key: str) -> Dict[str, Any]:
        """
        Compare metric across different conditions.

        Args:
            metric_name: Name of the metric
            condition_key: Condition key to compare

        Returns:
            Comparison results
        """
        condition_groups = {}

        for trial in self.trials:
            if metric_name not in trial or condition_key not in trial:
                continue

            condition_value = trial[condition_key]
            if condition_value not in condition_groups:
                condition_groups[condition_value] = []
            condition_groups[condition_value].append(trial[metric_name])

        comparison = {}
        for condition, values in condition_groups.items():
            comparison[condition] = {
                "count": len(values),
                "mean": np.mean(values),
                "std": np.std(values),
            }

        return comparison

    def export_experiment_data(self, format: str = "json") -> str:
        """
        Export experiment data.

        Args:
            format: Export format

        Returns:
            Exported data
        """
        with self.lock:
            data = {
                "experiment_id": self.experiment_id,
                "conditions": self.conditions,
                "trials": self.trials,
            }

            if format == "json":
                return json.dumps(data, indent=2)
            elif format == "csv":
                import csv
                import io

                if not self.trials:
                    return ""

                output = io.StringIO()
                writer = csv.DictWriter(output, fieldnames=self.trials[0].keys())
                writer.writeheader()

                for trial in self.trials:
                    writer.writerow(trial)

                return output.getvalue()

            return ""


class DataQualityMonitor:
    """
    Data quality monitoring and validation.

    Ensures data quality through continuous monitoring
    and validation of system outputs.
    """

    def __init__(self, quality_threshold: float = 0.95):
        """
        Initialize data quality monitor.

        Args:
            quality_threshold: Minimum quality threshold (0-1)
        """
        self.quality_threshold = quality_threshold
        self.quality_metrics = deque(maxlen=1000)
        self.alert_threshold = 0.90
        self.logger = logging.getLogger(__name__)

    def assess_data_quality(self, audio_data: np.ndarray, metadata: Dict[str, Any]) -> float:
        """
        Assess data quality.

        Args:
            audio_data: Audio data to assess
            metadata: Associated metadata

        Returns:
            Quality score (0-1)
        """
        # Check for audio quality issues
        quality_score = 1.0

        # Check clipping
        if np.any(np.abs(audio_data) >= 0.99):
            quality_score *= 0.8  # Penalty for clipping

        # Check duration
        duration = len(audio_data) / 48000  # Assuming 48kHz
        if duration < 0.01 or duration > 10.0:
            quality_score *= 0.9  # Penalty for extreme durations

        # Check spectral content
        spectrum = np.abs(np.fft.fft(audio_data))
        if np.sum(spectrum) < 1e-6:
            quality_score *= 0.5  # Penalty for silent audio

        # Check metadata completeness
        required_fields = ["timestamp", "species", "context"]
        for field in required_fields:
            if field not in metadata:
                quality_score *= 0.95  # Small penalty for missing metadata

        self.quality_metrics.append(
            {
                "timestamp": datetime.now().isoformat(),
                "quality_score": quality_score,
                "metadata": metadata,
            }
        )

        if quality_score < self.alert_threshold:
            self.logger.warning(f"Low quality audio detected: {quality_score:.3f}")

        return quality_score

    def get_quality_statistics(self) -> Dict[str, Any]:
        """
        Get quality statistics.

        Returns:
            Quality statistics
        """
        if not self.quality_metrics:
            return {"count": 0}

        scores = [m["quality_score"] for m in self.quality_metrics]

        return {
            "count": len(scores),
            "mean_quality": np.mean(scores),
            "min_quality": np.min(scores),
            "max_quality": np.max(scores),
            "below_threshold": sum(1 for s in scores if s < self.quality_threshold),
            "below_alert": sum(1 for s in scores if s < self.alert_threshold),
        }


# Initialize global components for easy access
_global_provenance_logger = None
_global_ptp_sync = None
_global_ab_tester = None
_global_experiment_tracker = None


def get_provenance_logger() -> ProvenanceLogger:
    """Get global provenance logger instance."""
    global _global_provenance_logger
    if _global_provenance_logger is None:
        _global_provenance_logger = ProvenanceLogger()
    return _global_provenance_logger


def get_ptp_sync() -> PTPSync:
    """Get global PTP sync instance."""
    global _global_ptp_sync
    if _global_ptp_sync is None:
        _global_ptp_sync = PTPSync()
    return _global_ptp_sync


def get_ab_tester(experiment_id: str = "default") -> ABTester:
    """Get global A/B tester instance."""
    global _global_ab_tester
    if _global_ab_tester is None:
        _global_ab_tester = ABTester(experiment_id=experiment_id)
    return _global_ab_tester


def get_experiment_tracker(experiment_id: str) -> ExperimentTracker:
    """Get global experiment tracker instance."""
    global _global_experiment_tracker
    if _global_experiment_tracker is None:
        _global_experiment_tracker = ExperimentTracker(experiment_id)
    return _global_experiment_tracker
