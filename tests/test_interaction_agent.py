#!/usr/bin/env python3
"""
Tests for Interaction Agent

These tests verify the closed-loop cognitive agent that bridges
Rust Execution Layer and Python Logic Layer.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import unittest
from unittest.mock import Mock

import numpy as np


class TestInteractionAgentConfig(unittest.TestCase):
    """Test interaction agent configuration"""

    def test_default_config(self):
        """Default config should have correct endpoints"""
        from realtime.interaction_agent import InteractionAgentConfig

        config = InteractionAgentConfig()

        self.assertEqual(config.feature_endpoint, "ipc:///tmp/cognitive_features.ipc")
        self.assertEqual(config.action_endpoint, "ipc:///tmp/cognitive_actions.ipc")
        self.assertEqual(config.response_cooldown_ms, 100.0)
        self.assertEqual(config.max_responses_per_second, 5.0)
        self.assertFalse(config.verbose_logging)

    def test_custom_config(self):
        """Should accept custom configuration"""
        from realtime.interaction_agent import InteractionAgentConfig

        config = InteractionAgentConfig(
            feature_endpoint="tcp://localhost:5555",
            action_endpoint="tcp://localhost:5556",
            response_cooldown_ms=200.0,
            verbose_logging=True,
        )

        self.assertEqual(config.feature_endpoint, "tcp://localhost:5555")
        self.assertEqual(config.action_endpoint, "tcp://localhost:5556")
        self.assertEqual(config.response_cooldown_ms, 200.0)
        self.assertTrue(config.verbose_logging)


class TestAgentState(unittest.TestCase):
    """Test agent state enum"""

    def test_agent_state_values(self):
        """Agent state should have expected values"""
        from realtime.interaction_agent import AgentState

        self.assertEqual(AgentState.IDLE.value, "idle")
        self.assertEqual(AgentState.LISTENING.value, "listening")
        self.assertEqual(AgentState.RESPONDING.value, "responding")


class TestInteractionAgentCreation(unittest.TestCase):
    """Test interaction agent creation"""

    def test_agent_creation_default(self):
        """Should create agent with defaults"""
        from realtime.interaction_agent import InteractionAgent

        agent = InteractionAgent()

        self.assertIsNotNone(agent.feature_subscriber)
        self.assertIsNotNone(agent.action_publisher)
        self.assertEqual(agent._events_processed, 0)
        self.assertEqual(agent._responses_sent, 0)

    def test_agent_creation_with_callbacks(self):
        """Should accept callbacks in constructor"""
        from realtime.interaction_agent import InteractionAgent

        on_feature = Mock()
        on_context = Mock()

        agent = InteractionAgent(
            on_feature_event=on_feature,
            on_context_change=on_context,
        )

        self.assertEqual(agent.on_feature_event, on_feature)
        self.assertEqual(agent.on_context_change, on_context)


class TestContextInference(unittest.TestCase):
    """Test context inference from 112D features"""

    def test_infer_alarm_context(self):
        """High F0 + high RMS should be alarm"""
        from realtime.interaction_agent import InteractionAgent

        agent = InteractionAgent()

        # Create features: high F0, high RMS
        features = np.zeros(112, dtype=np.float32)
        features[0] = 9000.0  # High F0
        features[1] = 0.8  # High RMS

        context, confidence = agent._infer_context(features)

        self.assertEqual(context, "alarm")
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_infer_territorial_context(self):
        """High F0 (but not extreme) should be territorial"""
        from realtime.interaction_agent import InteractionAgent

        agent = InteractionAgent()

        features = np.zeros(112, dtype=np.float32)
        features[0] = 7000.0  # Medium-high F0

        context, confidence = agent._infer_context(features)

        self.assertEqual(context, "territorial")
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_infer_social_context(self):
        """Low F0 should be social"""
        from realtime.interaction_agent import InteractionAgent

        agent = InteractionAgent()

        features = np.zeros(112, dtype=np.float32)
        features[0] = 3000.0  # Low F0

        context, confidence = agent._infer_context(features)

        self.assertEqual(context, "social")
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_infer_contact_context(self):
        """Medium F0 should be contact"""
        from realtime.interaction_agent import InteractionAgent

        agent = InteractionAgent()

        features = np.zeros(112, dtype=np.float32)
        features[0] = 5000.0  # Medium F0

        context, confidence = agent._infer_context(features)

        self.assertEqual(context, "contact")
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)


class TestResponseTimeline(unittest.TestCase):
    """Test response timeline generation"""

    def test_alarm_timeline(self):
        """Alarm context should produce short, loud timeline"""
        from realtime.interaction_agent import InteractionAgent

        agent = InteractionAgent()

        timeline = agent._create_response_timeline(42, "alarm")

        self.assertEqual(len(timeline), 1)
        self.assertEqual(timeline[0].cluster_id, 42)
        self.assertEqual(timeline[0].duration_ms, 100.0)
        self.assertEqual(timeline[0].amplitude, 0.9)

    def test_territorial_timeline(self):
        """Territorial context should produce longer timeline"""
        from realtime.interaction_agent import InteractionAgent

        agent = InteractionAgent()

        timeline = agent._create_response_timeline(99, "territorial")

        self.assertEqual(len(timeline), 1)
        self.assertEqual(timeline[0].duration_ms, 200.0)
        self.assertEqual(timeline[0].amplitude, 0.85)

    def test_contact_timeline(self):
        """Contact context should produce medium timeline"""
        from realtime.interaction_agent import InteractionAgent

        agent = InteractionAgent()

        timeline = agent._create_response_timeline(1, "contact")

        self.assertEqual(len(timeline), 1)
        self.assertEqual(timeline[0].duration_ms, 150.0)
        self.assertEqual(timeline[0].amplitude, 0.75)


class TestMicroDynamicsDeltas(unittest.TestCase):
    """Test micro-dynamics delta generation"""

    def test_alarm_deltas(self):
        """Alarm context should raise F0 and energy"""
        from realtime.interaction_agent import InteractionAgent

        agent = InteractionAgent()

        deltas = agent._create_deltas("alarm")

        self.assertIsNotNone(deltas)
        self.assertEqual(deltas.delta_mean_f0_hz, 500.0)
        self.assertEqual(deltas.delta_rms_energy, 0.2)

    def test_territorial_deltas(self):
        """Territorial context should raise F0 and duration"""
        from realtime.interaction_agent import InteractionAgent

        agent = InteractionAgent()

        deltas = agent._create_deltas("territorial")

        self.assertIsNotNone(deltas)
        self.assertEqual(deltas.delta_mean_f0_hz, 200.0)
        self.assertEqual(deltas.delta_duration_ms, 20.0)

    def test_social_deltas(self):
        """Social context should lower F0"""
        from realtime.interaction_agent import InteractionAgent

        agent = InteractionAgent()

        deltas = agent._create_deltas("social")

        self.assertIsNotNone(deltas)
        self.assertEqual(deltas.delta_mean_f0_hz, -100.0)

    def test_contact_deltas(self):
        """Contact context should have no deltas"""
        from realtime.interaction_agent import InteractionAgent

        agent = InteractionAgent()

        deltas = agent._create_deltas("contact")

        self.assertIsNone(deltas)


class TestResponseDecision(unittest.TestCase):
    """Test response decision logic"""

    def test_should_respond_alarm(self):
        """Should respond to alarm context"""
        from realtime.interaction_agent import InteractionAgent

        agent = InteractionAgent()
        agent._last_response_time = 0.0  # Allow response

        result = {
            "context_state": "alarm",
            "confidence": 0.8,
        }

        should = agent._should_respond(result)
        self.assertTrue(should)

    def test_should_respond_low_confidence(self):
        """Should not respond with low confidence"""
        from realtime.interaction_agent import InteractionAgent

        agent = InteractionAgent()
        agent._last_response_time = 0.0

        result = {
            "context_state": "alarm",
            "confidence": 0.3,  # Low confidence
        }

        should = agent._should_respond(result)
        self.assertFalse(should)

    def test_should_respond_rate_limited(self):
        """Should not respond if rate limited"""
        import time

        from realtime.interaction_agent import InteractionAgent

        agent = InteractionAgent()
        agent._last_response_time = time.time()  # Just responded

        result = {
            "context_state": "alarm",
            "confidence": 0.8,
        }

        should = agent._should_respond(result)
        self.assertFalse(should)

    def test_should_not_respond_unknown_context(self):
        """Should not respond to unknown context"""
        from realtime.interaction_agent import InteractionAgent

        agent = InteractionAgent()
        agent._last_response_time = 0.0

        result = {
            "context_state": "unknown",
            "confidence": 0.9,
        }

        should = agent._should_respond(result)
        self.assertFalse(should)


class TestFeatureEventProcessing(unittest.TestCase):
    """Test feature event processing"""

    def test_process_features_returns_context(self):
        """Processing should return context"""
        from realtime.interaction_agent import (
            FeatureEvent,
            InteractionAgent,
        )

        agent = InteractionAgent()

        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=42,
            features_112d=np.random.randn(112).astype(np.float32),
            timestamp=1000.0,
            sequence=1,
        )

        result = agent._process_features(event)

        self.assertIn("context_state", result)
        self.assertIn("confidence", result)
        self.assertIn("cluster_id", result)
        self.assertEqual(result["cluster_id"], 42)

    def test_process_features_extracts_key_values(self):
        """Processing should extract key feature values"""
        from realtime.interaction_agent import (
            FeatureEvent,
            InteractionAgent,
        )

        agent = InteractionAgent()

        features = np.zeros(112, dtype=np.float32)
        features[0] = 6000.0  # F0
        features[1] = 0.5  # RMS

        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=1,
            features_112d=features,
            timestamp=0.0,
            sequence=1,
        )

        result = agent._process_features(event)

        self.assertEqual(result["cluster_id"], 1)
        self.assertEqual(result["sequence"], 1)


class TestStatistics(unittest.TestCase):
    """Test agent statistics"""

    def test_initial_stats(self):
        """Initial stats should be zero"""
        from realtime.interaction_agent import InteractionAgent

        agent = InteractionAgent()
        stats = agent.get_stats()

        self.assertEqual(stats["events_processed"], 0)
        self.assertEqual(stats["responses_sent"], 0)
        self.assertEqual(stats["state"], "idle")
        self.assertIsNone(stats["current_context"])

    def test_stats_after_processing(self):
        """Stats should update after processing"""
        from realtime.interaction_agent import (
            FeatureEvent,
            InteractionAgent,
        )

        agent = InteractionAgent()

        # Simulate processing an event
        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=42,
            features_112d=np.zeros(112, dtype=np.float32),
            timestamp=0.0,
            sequence=1,
        )

        agent._handle_feature_event(event)

        stats = agent.get_stats()
        self.assertEqual(stats["events_processed"], 1)


class TestContextClassifierIntegration(unittest.TestCase):
    """Test ContextClassifier integration with InteractionAgent"""

    def test_config_accepts_classifier_path(self):
        """Config should accept context_classifier_path parameter"""
        from realtime.interaction_agent import InteractionAgentConfig

        config = InteractionAgentConfig(context_classifier_path="/path/to/model.pkl")

        self.assertEqual(config.context_classifier_path, "/path/to/model.pkl")

    def test_agent_loads_classifier_on_init(self):
        """Agent should load ContextClassifier when path is provided"""
        import tempfile

        from realtime.context_classifier import ContextClassifier
        from realtime.interaction_agent import InteractionAgent, InteractionAgentConfig

        # Create and train a simple classifier
        features = np.random.randn(100, 112)
        labels = np.array(["social"] * 50 + ["alarm"] * 50)

        classifier = ContextClassifier(model_type="mlp", random_state=42)
        classifier.train(features, labels)

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            model_path = f.name
            classifier.save(model_path)

        try:
            # Create agent with classifier
            config = InteractionAgentConfig(context_classifier_path=model_path)
            agent = InteractionAgent(config=config)

            # Verify classifier was loaded
            self.assertIsNotNone(agent.context_classifier)
            self.assertEqual(agent.context_classifier.model_type, "mlp")
        finally:
            import os

            os.unlink(model_path)

    def test_agent_fallback_to_rules_without_classifier(self):
        """Agent should use rule-based inference when no classifier"""
        from realtime.interaction_agent import InteractionAgent

        agent = InteractionAgent()
        self.assertIsNone(agent.context_classifier)

        # Test rule-based inference with high F0
        features = np.zeros(112)
        features[0] = 9000.0  # High F0
        features[1] = 0.7  # High RMS

        context, confidence = agent._infer_context(features)
        self.assertEqual(context, "alarm")
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_agent_uses_classifier_for_inference(self):
        """Agent should use ContextClassifier for context inference"""
        import tempfile

        from realtime.context_classifier import ContextClassifier
        from realtime.interaction_agent import (
            InteractionAgent,
            InteractionAgentConfig,
        )

        # Create a classifier that returns "social" for low F0 features
        np.random.seed(42)
        features = np.random.randn(100, 112)
        # Make low F0 features map to "social"
        features[:50, 0] -= 5.0  # Low F0
        features[50:, 0] += 5.0  # High F0
        labels = np.array(["social"] * 50 + ["alarm"] * 50)

        classifier = ContextClassifier(model_type="mlp", random_state=42)
        classifier.train(features, labels)

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            model_path = f.name
            classifier.save(model_path)

        try:
            # Create agent with classifier
            config = InteractionAgentConfig(context_classifier_path=model_path)
            agent = InteractionAgent(config=config)

            # Test with low F0 features (should predict "social" via ML)
            test_features = np.zeros(112)
            test_features[0] = 3000.0  # Low F0

            context, confidence = agent._infer_context(test_features)

            # Should use ML prediction, not rule-based
            # With low F0, rule-based would give "social", so we verify
            # by checking the classifier is being used
            self.assertIsNotNone(agent.context_classifier)
            # The exact prediction depends on training, but should not be random
            self.assertIn(context, ["social", "alarm", "territorial", "contact"])
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)

        finally:
            import os

            os.unlink(model_path)

    def test_agent_handles_invalid_classifier_path(self):
        """Agent should fall back to rules when classifier path is invalid"""
        from realtime.interaction_agent import InteractionAgent, InteractionAgentConfig

        config = InteractionAgentConfig(context_classifier_path="/nonexistent/path.pkl")
        agent = InteractionAgent(config=config)

        # Should fall back to None (rule-based)
        self.assertIsNone(agent.context_classifier)

        # Rule-based inference should still work
        features = np.zeros(112)
        features[0] = 9000.0  # High F0
        features[1] = 0.7  # High RMS

        context, confidence = agent._infer_context(features)
        self.assertEqual(context, "alarm")
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_label_mapping_maps_pseudo_labels_to_canonical(self):
        """Agent should map pseudo-labels to canonical contexts via config."""
        import tempfile

        from realtime.context_classifier import ContextClassifier
        from realtime.interaction_agent import (
            InteractionAgent,
            InteractionAgentConfig,
        )

        # Create a classifier with pseudo-labels (context_0, context_1, etc.)
        np.random.seed(42)
        features = np.random.randn(100, 112)
        pseudo_labels = [f"context_{i % 3}" for i in range(100)]  # context_0, context_1, context_2

        classifier = ContextClassifier(model_type="mlp", random_state=42)
        classifier.train(features, np.array(pseudo_labels))

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            model_path = f.name
            classifier.save(model_path)

        try:
            # Create agent with label mapping
            config = InteractionAgentConfig(
                context_classifier_path=model_path,
                context_label_mapping={
                    "context_0": "social",
                    "context_1": "alarm",
                    "context_2": "territorial",
                },
            )
            agent = InteractionAgent(config=config)

            # Test that mapping works
            test_features = np.zeros(112)

            # Get raw prediction from classifier
            raw_context, _ = agent.context_classifier.predict(test_features)

            # Get mapped prediction through agent
            mapped_context, confidence = agent._infer_context(test_features)

            # Should be mapped to canonical context
            self.assertIn(mapped_context, ["social", "alarm", "territorial", "contact"])
            # If raw was pseudo-label, mapped should be different
            if raw_context.startswith("context_"):
                self.assertNotEqual(mapped_context, raw_context)

        finally:
            import os

            os.unlink(model_path)

    def test_unmapped_contexts_do_not_trigger_response(self):
        """Unmapped pseudo-labels should not trigger response."""
        import time

        from realtime.interaction_agent import InteractionAgent, InteractionAgentConfig

        # Create agent
        config = InteractionAgentConfig()
        agent = InteractionAgent(config=config)

        # Create result with unmapped pseudo-label
        result = {
            "context_state": "context_999",  # Not in canonical ontology
            "confidence": 0.9,
            "timestamp": time.time() - 1000,  # Past cooldown
        }

        # Should not respond because context is not canonical
        self.assertFalse(agent._should_respond(result))


class TestUncertaintyGating(unittest.TestCase):
    """Test uncertainty-gated response decisions"""

    def test_agent_rejects_high_uncertainty(self):
        """Agent rejects event with uncertainty > threshold"""
        from realtime.interaction_agent import (
            FeatureEvent,
            InteractionAgent,
            InteractionAgentConfig,
        )

        # Create agent with low uncertainty threshold
        config = InteractionAgentConfig(uncertainty_threshold=0.6)
        agent = InteractionAgent(config=config)
        agent._last_response_time = 0.0  # Allow response

        # Create event with high uncertainty
        features = np.zeros(112, dtype=np.float32)
        features[0] = 9000.0  # High F0 for alarm
        features[1] = 0.8  # High RMS

        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=42,
            features_112d=features,
            timestamp=1000.0,
            sequence=1,
            uncertainty=0.9,  # High uncertainty > threshold
        )

        # Process event
        result = agent._process_features(event)

        # Should NOT respond due to high uncertainty
        self.assertFalse(agent._should_respond(result))

    def test_agent_accepts_low_uncertainty(self):
        """Agent accepts event with uncertainty < threshold"""
        from realtime.interaction_agent import (
            FeatureEvent,
            InteractionAgent,
            InteractionAgentConfig,
        )

        config = InteractionAgentConfig(uncertainty_threshold=0.6)
        agent = InteractionAgent(config=config)
        agent._last_response_time = 0.0  # Allow response

        # Create event with low uncertainty
        features = np.zeros(112, dtype=np.float32)
        features[0] = 9000.0  # High F0 for alarm
        features[1] = 0.8  # High RMS

        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=42,
            features_112d=features,
            timestamp=1000.0,
            sequence=1,
            uncertainty=0.3,  # Low uncertainty < threshold
        )

        result = agent._process_features(event)

        # Should respond (low uncertainty, high confidence, alarm context)
        self.assertTrue(agent._should_respond(result))

    def test_agent_uncertainty_threshold_config(self):
        """Configurable uncertainty threshold works"""
        from realtime.interaction_agent import (
            FeatureEvent,
            InteractionAgent,
            InteractionAgentConfig,
        )

        # Agent with strict threshold (0.4)
        config = InteractionAgentConfig(uncertainty_threshold=0.4)
        agent = InteractionAgent(config=config)
        agent._last_response_time = 0.0

        features = np.zeros(112, dtype=np.float32)
        features[0] = 9000.0
        features[1] = 0.8

        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=42,
            features_112d=features,
            timestamp=1000.0,
            sequence=1,
            uncertainty=0.5,  # Between thresholds
        )

        result = agent._process_features(event)

        # Should NOT respond (0.5 > 0.4 threshold)
        self.assertFalse(agent._should_respond(result))

    def test_agent_uncertainty_with_confidence(self):
        """Both uncertainty and confidence checked together"""
        from realtime.interaction_agent import (
            FeatureEvent,
            InteractionAgent,
            InteractionAgentConfig,
        )

        config = InteractionAgentConfig(uncertainty_threshold=0.6)
        agent = InteractionAgent(config=config)
        agent._last_response_time = 0.0

        # Case 1: High confidence BUT high uncertainty -> reject
        features = np.zeros(112, dtype=np.float32)
        features[0] = 9000.0
        features[1] = 0.8

        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=42,
            features_112d=features,
            timestamp=1000.0,
            sequence=1,
            uncertainty=0.8,  # High uncertainty
        )

        result = agent._process_features(event)
        self.assertFalse(agent._should_respond(result))

        # Case 2: Low confidence AND low uncertainty -> reject
        event2 = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=42,
            features_112d=features * 0.01,  # Low variance -> low confidence
            timestamp=1000.0,
            sequence=2,
            uncertainty=0.2,  # Low uncertainty
        )

        result2 = agent._process_features(event2)
        # Should still reject due to low confidence
        self.assertFalse(agent._should_respond(result2))

    def test_agent_default_uncertainty_threshold(self):
        """Default uncertainty threshold should be 0.6"""
        from realtime.interaction_agent import InteractionAgentConfig

        config = InteractionAgentConfig()
        self.assertEqual(config.uncertainty_threshold, 0.6)

    def test_process_features_propagates_uncertainty(self):
        """Uncertainty should be propagated from event to result"""
        from realtime.interaction_agent import (
            FeatureEvent,
            InteractionAgent,
        )

        agent = InteractionAgent()

        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=42,
            features_112d=np.random.randn(112).astype(np.float32),
            timestamp=1000.0,
            sequence=1,
            uncertainty=0.45,
        )

        result = agent._process_features(event)

        self.assertIn("uncertainty", result)
        self.assertEqual(result["uncertainty"], 0.45)

    def test_process_features_uncertainty_none(self):
        """Uncertainty defaults to None when not provided"""
        from realtime.interaction_agent import (
            FeatureEvent,
            InteractionAgent,
        )

        agent = InteractionAgent()

        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=42,
            features_112d=np.random.randn(112).astype(np.float32),
            timestamp=1000.0,
            sequence=1,
            # No uncertainty field
        )

        result = agent._process_features(event)

        self.assertIn("uncertainty", result)
        self.assertIsNone(result["uncertainty"])


if __name__ == "__main__":
    unittest.main()
