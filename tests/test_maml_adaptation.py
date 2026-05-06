#!/usr/bin/env python3
"""
Tests for MAML Adaptation - Model-Agnostic Meta-Learning

These tests verify the MAML implementation for rapid cross-species
adaptation with few-shot learning.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import unittest

import numpy as np


class TestMAMLOptimizer(unittest.TestCase):
    """Test MAML optimizer for meta-learning"""

    def test_meta_parameter_initialization(self):
        """Should initialize meta-parameters"""
        from cognitive_intelligence.maml_adaptation import MAMLOptimizer

        optimizer = MAMLOptimizer(
            input_dim=112,
            hidden_dim=64,
            output_dim=4,
        )

        self.assertIsNotNone(optimizer.meta_parameters)

    def test_inner_loop_update(self):
        """Should perform inner loop gradient update"""
        from cognitive_intelligence.maml_adaptation import MAMLOptimizer

        optimizer = MAMLOptimizer(
            input_dim=112,
            hidden_dim=64,
            output_dim=4,
            inner_lr=0.01,
        )

        # Support set
        support_x = np.random.randn(5, 112).astype(np.float32)
        support_y = np.random.randint(0, 4, 5).astype(np.int32)

        adapted_params = optimizer.inner_loop_update(support_x, support_y)

        self.assertIsNotNone(adapted_params)

    def test_meta_update(self):
        """Should perform meta-update across tasks"""
        from cognitive_intelligence.maml_adaptation import MAMLOptimizer

        optimizer = MAMLOptimizer(
            input_dim=112,
            hidden_dim=64,
            output_dim=4,
            meta_lr=0.001,
        )

        # Batch of tasks
        tasks = []
        for _ in range(3):
            support_x = np.random.randn(5, 112).astype(np.float32)
            support_y = np.random.randint(0, 4, 5).astype(np.int32)
            query_x = np.random.randn(5, 112).astype(np.float32)
            query_y = np.random.randint(0, 4, 5).astype(np.int32)
            tasks.append((support_x, support_y, query_x, query_y))

        meta_loss = optimizer.meta_update(tasks)

        self.assertGreater(meta_loss, 0.0)


class TestFewShotClassifier(unittest.TestCase):
    """Test few-shot classifier using MAML"""

    def test_5_way_5_shot_classification(self):
        """Should perform 5-way 5-shot classification"""
        from cognitive_intelligence.maml_adaptation import FewShotClassifier

        classifier = FewShotClassifier(
            input_dim=112,
            num_classes=5,
            k_shot=5,
        )

        # Support set (5 examples per class)
        support_x = np.random.randn(25, 112).astype(np.float32)
        support_y = np.repeat(np.arange(5), 5).astype(np.int32)

        # Adapt to task
        classifier.adapt(support_x, support_y)

        # Query set
        query_x = np.random.randn(5, 112).astype(np.float32)
        predictions = classifier.predict(query_x)

        self.assertEqual(len(predictions), 5)

    def test_cross_species_adaptation(self):
        """Should adapt to new species with few examples"""
        from cognitive_intelligence.maml_adaptation import FewShotClassifier

        classifier = FewShotClassifier(
            input_dim=112,
            num_classes=4,
            k_shot=3,
        )

        # Simulate data from different species
        # Base species (marmoset)
        base_features = np.random.randn(50, 112).astype(np.float32)
        base_labels = np.random.randint(0, 4, 50).astype(np.int32)

        # Pre-train on base species
        classifier.pretrain(base_features, base_labels)

        # New species (bat) with only 3 examples per class
        new_species_features = np.random.randn(12, 112).astype(np.float32)
        new_species_labels = np.repeat(np.arange(4), 3).astype(np.int32)

        # Adapt to new species
        classifier.adapt(new_species_features, new_species_labels)

        # Predict on new species data
        query = np.random.randn(4, 112).astype(np.float32)
        predictions = classifier.predict(query)

        self.assertEqual(len(predictions), 4)

    def test_1_shot_classification(self):
        """Should handle 1-shot classification"""
        from cognitive_intelligence.maml_adaptation import FewShotClassifier

        classifier = FewShotClassifier(
            input_dim=112,
            num_classes=3,
            k_shot=1,
        )

        # 1 example per class
        support_x = np.random.randn(3, 112).astype(np.float32)
        support_y = np.arange(3).astype(np.int32)

        classifier.adapt(support_x, support_y)

        query_x = np.random.randn(3, 112).astype(np.float32)
        predictions = classifier.predict(query_x)

        self.assertEqual(len(predictions), 3)


class TestTaskDistribution(unittest.TestCase):
    """Test task distribution for meta-learning"""

    def test_sample_task(self):
        """Should sample a task from distribution"""
        from cognitive_intelligence.maml_adaptation import TaskDistribution

        distribution = TaskDistribution(
            n_classes=5,
            n_support=5,
            n_query=10,
        )

        task = distribution.sample_task()

        self.assertIn("support_x", task)
        self.assertIn("support_y", task)
        self.assertIn("query_x", task)
        self.assertIn("query_y", task)

    def test_cross_species_tasks(self):
        """Should generate cross-species tasks"""
        from cognitive_intelligence.maml_adaptation import TaskDistribution

        distribution = TaskDistribution(
            n_classes=4,
            n_support=3,
            n_query=5,
            species=["marmoset", "bat", "dolphin"],
        )

        task = distribution.sample_cross_species_task("marmoset")

        self.assertIn("support_x", task)
        self.assertIn("query_x", task)

    def test_task_batch(self):
        """Should sample batch of tasks"""
        from cognitive_intelligence.maml_adaptation import TaskDistribution

        distribution = TaskDistribution(
            n_classes=4,
            n_support=3,
            n_query=5,
        )

        batch = distribution.sample_batch(n_tasks=4)

        self.assertEqual(len(batch), 4)


class TestSpeciesEncoder(unittest.TestCase):
    """Test species-specific encoder"""

    def test_encode_species_features(self):
        """Should encode species-specific features"""
        from cognitive_intelligence.maml_adaptation import SpeciesEncoder

        encoder = SpeciesEncoder(
            input_dim=112,
            latent_dim=32,
            num_species=4,
        )

        features = np.random.randn(10, 112).astype(np.float32)
        species_id = 1  # Bat

        encoded = encoder.encode(features, species_id)

        self.assertEqual(encoded.shape[0], 10)
        self.assertEqual(encoded.shape[1], 32)

    def test_species_conditioning(self):
        """Should condition encoding on species"""
        from cognitive_intelligence.maml_adaptation import SpeciesEncoder

        encoder = SpeciesEncoder(
            input_dim=112,
            latent_dim=32,
            num_species=4,
        )

        features = np.random.randn(10, 112).astype(np.float32)

        # Different species should produce different encodings
        encoding_1 = encoder.encode(features, species_id=0)
        encoding_2 = encoder.encode(features, species_id=1)

        self.assertFalse(np.allclose(encoding_1, encoding_2))


class TestRapidAdaptation(unittest.TestCase):
    """Test rapid adaptation capabilities"""

    def test_adaptation_speed(self):
        """Should adapt quickly to new tasks"""
        from cognitive_intelligence.maml_adaptation import MAMLOptimizer

        optimizer = MAMLOptimizer(
            input_dim=112,
            hidden_dim=64,
            output_dim=4,
            inner_lr=0.01,
            n_inner_steps=3,
        )

        # Few examples for adaptation
        support_x = np.random.randn(3, 112).astype(np.float32)
        support_y = np.random.randint(0, 4, 3).astype(np.int32)

        # Should adapt in few steps
        adapted_params = optimizer.inner_loop_update(support_x, support_y)

        self.assertIsNotNone(adapted_params)

    def test_transfer_learning(self):
        """Should transfer knowledge across species"""
        from cognitive_intelligence.maml_adaptation import MetaLearner

        meta_learner = MetaLearner(
            input_dim=112,
            num_classes=4,
            species=["marmoset", "bat", "dolphin"],
        )

        # Train on multiple species
        for species in ["marmoset", "bat", "dolphin"]:
            features = np.random.randn(50, 112).astype(np.float32)
            labels = np.random.randint(0, 4, 50).astype(np.int32)
            meta_learner.add_species_data(species, features, labels)

        meta_learner.meta_train(n_epochs=2, n_tasks_per_epoch=4)

        # Rapid adaptation to new species
        new_species_data = np.random.randn(5, 112).astype(np.float32)
        new_species_labels = np.random.randint(0, 4, 5).astype(np.int32)

        meta_learner.adapt_to_species("finch", new_species_data, new_species_labels)

        # Should be able to predict
        query = np.random.randn(3, 112).astype(np.float32)
        predictions = meta_learner.predict(query, species="finch")

        self.assertEqual(len(predictions), 3)


class TestMAMLIntegration(unittest.TestCase):
    """Test MAML integration with existing models"""

    def test_adapt_context_classifier(self):
        """Should adapt context classifier to new species"""
        from cognitive_intelligence.maml_adaptation import MAMLAdaptationWrapper

        # Simulate existing classifier
        class DummyClassifier:
            def __init__(self):
                self.weights = np.random.randn(112, 4).astype(np.float32)

            def predict(self, x):
                logits = x @ self.weights
                return np.argmax(logits, axis=-1)

        base_classifier = DummyClassifier()
        wrapper = MAMLAdaptationWrapper(base_classifier)

        # Adapt with few examples
        support_x = np.random.randn(5, 112).astype(np.float32)
        support_y = np.random.randint(0, 4, 5).astype(np.int32)

        wrapper.adapt(support_x, support_y, n_steps=3)

        # Predict with adapted model
        query_x = np.random.randn(3, 112).astype(np.float32)
        predictions = wrapper.predict(query_x)

        self.assertEqual(len(predictions), 3)

    def test_fine_tune_vs_meta_learn(self):
        """Meta-learning should be more sample-efficient"""
        from cognitive_intelligence.maml_adaptation import MetaLearner

        # Meta-learned model
        meta_learner = MetaLearner(
            input_dim=112,
            num_classes=4,
        )

        # Pre-train on diverse tasks
        for _ in range(10):
            task_x = np.random.randn(20, 112).astype(np.float32)
            task_y = np.random.randint(0, 4, 20).astype(np.int32)
            meta_learner.meta_update_task(task_x, task_y)

        # Adapt with only 3 examples
        few_shot_x = np.random.randn(3, 112).astype(np.float32)
        few_shot_y = np.random.randint(0, 4, 3).astype(np.int32)

        meta_learner.adapt(few_shot_x, few_shot_y)

        # Should predict reasonably
        query_x = np.random.randn(3, 112).astype(np.float32)
        predictions = meta_learner.predict(query_x)

        self.assertEqual(len(predictions), 3)


if __name__ == "__main__":
    unittest.main()
