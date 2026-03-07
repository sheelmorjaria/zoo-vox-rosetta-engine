"""
Comprehensive Tests for Bio-Acoustic Interaction Agent

Tests the bio_acoustic_agent module covering:
- Enum and dataclass validation
- ContextDeltaCalculator logic
- FormantBarrierValidator barrier checks
- AcousticInventory management
- BioAcousticAgent lifecycle
- Synthesis planning
- Integration with cognitive layer

Author: Test Coverage Initiative
License: CC BY-ND 4.0 International
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Handle imports gracefully
try:
    from realtime.bio_acoustic_agent import (
        AcousticInventory,
        AcousticModality,
        AcousticPrototype,
        BioAcousticAgent,
        ContextDeltaCalculator,
        EnvState,
        FormantBarrierValidator,
        InteractionContext,
        MicroDynamicsDelta,
        ResponseModification,
        SemioticEnrichment,
        SourceMetadata,
        SynthesisPlan,
        ValidationResult,
    )

    BIO_ACOUSTIC_AGENT_AVAILABLE = True
except ImportError as e:
    BIO_ACOUSTIC_AGENT_AVAILABLE = False
    IMPORT_ERROR = str(e)


@unittest.skipIf(
    not BIO_ACOUSTIC_AGENT_AVAILABLE,
    f"Bio acoustic agent not available: {IMPORT_ERROR if not BIO_ACOUSTIC_AGENT_AVAILABLE else ''}",
)
class TestEnums(unittest.TestCase):
    """Test enum values"""

    def test_env_state_values(self):
        """EnvState should have expected values"""
        self.assertEqual(EnvState.QUIET.value, "Quiet")
        self.assertEqual(EnvState.WIND.value, "Wind")
        self.assertEqual(EnvState.RAIN.value, "Rain")
        self.assertEqual(EnvState.STORM.value, "Storm")
        self.assertEqual(EnvState.UNKNOWN.value, "Unknown")

    def test_interaction_context_values(self):
        """InteractionContext should have expected values"""
        self.assertEqual(InteractionContext.INITIATOR.value, "Initiator")
        self.assertEqual(InteractionContext.REPLY.value, "Reply")
        self.assertEqual(InteractionContext.SOLO.value, "Solo")
        self.assertEqual(InteractionContext.CHORUS.value, "Chorus")

    def test_acoustic_modality_values(self):
        """AcousticModality should have expected values"""
        self.assertEqual(AcousticModality.HARMONIC.value, "Harmonic")
        self.assertEqual(AcousticModality.TRANSIENT.value, "Transient")
        self.assertEqual(AcousticModality.MIXED.value, "Mixed")

    def test_response_modification_values(self):
        """ResponseModification should have expected values"""
        self.assertEqual(ResponseModification.NORMAL.value, "normal")
        self.assertEqual(ResponseModification.DECEPTION_ACKNOWLEDGE.value, "deception_ack")
        self.assertEqual(ResponseModification.EMERGENCE_LOG.value, "emergence_log")
        self.assertEqual(ResponseModification.URGENCY_BOOST.value, "urgency_boost")


@unittest.skipIf(
    not BIO_ACOUSTIC_AGENT_AVAILABLE,
    f"Bio acoustic agent not available: {IMPORT_ERROR if not BIO_ACOUSTIC_AGENT_AVAILABLE else ''}",
)
class TestDataClasses(unittest.TestCase):
    """Test dataclass creation and defaults"""

    def test_source_metadata_defaults(self):
        """SourceMetadata should have sensible defaults"""
        metadata = SourceMetadata()
        self.assertEqual(metadata.mean_f0_hz, 0.0)
        self.assertEqual(metadata.duration_ms, 0.0)
        self.assertEqual(metadata.f0_range_hz, 0.0)
        self.assertEqual(metadata.harmonic_to_noise_ratio, 0.0)
        self.assertEqual(metadata.rms_energy, 0.0)

    def test_source_metadata_from_dict(self):
        """SourceMetadata should be created from dict"""
        data = {
            "mean_f0_hz": 8000.0,
            "duration_ms": 200.0,
            "f0_range_hz": 1500.0,
        }
        metadata = SourceMetadata.from_dict(data)
        self.assertEqual(metadata.mean_f0_hz, 8000.0)
        self.assertEqual(metadata.duration_ms, 200.0)
        self.assertEqual(metadata.f0_range_hz, 1500.0)

    def test_source_metadata_get_modality_harmonic(self):
        """get_modality should return HARMONIC for tonal signals"""
        metadata = SourceMetadata(
            harmonic_to_noise_ratio=20.0,  # High HNR
            entropy=0.2,  # Low entropy = tonal
        )
        self.assertEqual(metadata.get_modality(), AcousticModality.HARMONIC)

    def test_source_metadata_get_modality_transient(self):
        """get_modality should return TRANSIENT for short noisy signals"""
        metadata = SourceMetadata(
            harmonic_to_noise_ratio=5.0,  # Low HNR
            entropy=0.7,  # High entropy = noisy
            duration_ms=50.0,  # Short
        )
        self.assertEqual(metadata.get_modality(), AcousticModality.TRANSIENT)

    def test_source_metadata_get_modality_mixed(self):
        """get_modality should return MIXED for in-between signals"""
        metadata = SourceMetadata(
            harmonic_to_noise_ratio=12.0,  # Medium HNR
            entropy=0.4,
            duration_ms=200.0,
        )
        self.assertEqual(metadata.get_modality(), AcousticModality.MIXED)

    def test_micro_dynamics_delta_defaults(self):
        """MicroDynamicsDelta should have sensible defaults"""
        delta = MicroDynamicsDelta()
        self.assertEqual(delta.delta_mean_f0_hz, 0.0)
        self.assertEqual(delta.delta_duration_ms, 0.0)
        self.assertEqual(delta.delta_loudness, 0.0)

    def test_micro_dynamics_delta_apply_to(self):
        """apply_to should apply delta to source metadata"""
        delta = MicroDynamicsDelta(
            delta_mean_f0_hz=100.0,
            delta_duration_ms=50.0,
        )
        source = SourceMetadata(mean_f0_hz=5000.0, duration_ms=200.0)
        result = delta.apply_to(source)
        self.assertEqual(result.mean_f0_hz, 5100.0)
        self.assertEqual(result.duration_ms, 250.0)

    def test_acoustic_prototype_creation(self):
        """AcousticPrototype should be created with proper fields"""
        prototype = AcousticPrototype(
            label="test_phee",
            sample_rate=48000,
            metadata=SourceMetadata(mean_f0_hz=8000.0),
            modality=AcousticModality.HARMONIC,
        )
        self.assertEqual(prototype.label, "test_phee")
        self.assertEqual(prototype.metadata.mean_f0_hz, 8000.0)
        self.assertEqual(prototype.modality, AcousticModality.HARMONIC)

    def test_acoustic_prototype_from_dict(self):
        """AcousticPrototype should be created from dict"""
        data = {
            "label": "phee_call",
            "sample_rate": 44100,
            "metadata": {"mean_f0_hz": 7000.0, "duration_ms": 300.0},
            "modality": "Harmonic",
        }
        prototype = AcousticPrototype.from_dict(data)
        self.assertEqual(prototype.label, "phee_call")
        self.assertEqual(prototype.sample_rate, 44100)
        self.assertEqual(prototype.metadata.mean_f0_hz, 7000.0)

    def test_validation_result(self):
        """ValidationResult should be created"""
        result = ValidationResult(
            is_valid=True,
            violations=[],
            recommended_action="Proceed with synthesis",
        )
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.violations), 0)
        self.assertEqual(result.recommended_action, "Proceed with synthesis")

    def test_validation_result_with_violations(self):
        """ValidationResult should track violations"""
        result = ValidationResult(
            is_valid=False,
            violations=["HNR change too large"],
            recommended_action="Reduce delta magnitude",
        )
        self.assertFalse(result.is_valid)
        self.assertEqual(len(result.violations), 1)

    def test_synthesis_plan_creation(self):
        """SynthesisPlan should be created with proper fields"""
        source = SourceMetadata(mean_f0_hz=5000.0)
        target = SourceMetadata(mean_f0_hz=5200.0)
        delta = MicroDynamicsDelta(delta_mean_f0_hz=200.0)
        validation = ValidationResult(is_valid=True, violations=[], recommended_action="Proceed")

        plan = SynthesisPlan(
            source_label="phee",
            source_metadata=source,
            delta=delta,
            target_metadata=target,
            validation=validation,
            description="Test plan",
        )
        self.assertEqual(plan.source_label, "phee")
        self.assertEqual(plan.source_metadata.mean_f0_hz, 5000.0)
        self.assertEqual(plan.target_metadata.mean_f0_hz, 5200.0)

    def test_semiotic_enrichment_defaults(self):
        """SemioticEnrichment should have sensible defaults"""
        enrichment = SemioticEnrichment()
        self.assertEqual(enrichment.deception_score, 0.0)
        self.assertEqual(enrichment.emergence_score, 0.0)
        self.assertEqual(enrichment.directed_score, 0.0)
        self.assertFalse(enrichment.deception_detected)
        self.assertEqual(enrichment.response_modification, ResponseModification.NORMAL)


@unittest.skipIf(
    not BIO_ACOUSTIC_AGENT_AVAILABLE,
    f"Bio acoustic agent not available: {IMPORT_ERROR if not BIO_ACOUSTIC_AGENT_AVAILABLE else ''}",
)
class TestContextDeltaCalculator(unittest.TestCase):
    """Test ContextDeltaCalculator class"""

    def test_calculator_creation(self):
        """ContextDeltaCalculator should be usable as static class"""
        # All methods are static, just verify we can call them
        delta = ContextDeltaCalculator.calculate(EnvState.QUIET, InteractionContext.SOLO)
        self.assertIsNotNone(delta)
        self.assertIsInstance(delta, MicroDynamicsDelta)

    def test_calculate_wind_environment(self):
        """calculate should adjust for wind environment"""
        delta = ContextDeltaCalculator.calculate(EnvState.WIND, InteractionContext.SOLO)

        # Wind should increase pitch for propagation
        self.assertGreater(delta.delta_mean_f0_hz, 0)
        self.assertGreater(delta.delta_loudness, 0)

    def test_calculate_storm_environment(self):
        """calculate should adjust for storm environment"""
        delta = ContextDeltaCalculator.calculate(EnvState.STORM, InteractionContext.SOLO)

        # Storm should increase entropy and loudness
        self.assertGreater(delta.delta_entropy, 0)
        self.assertGreater(delta.delta_loudness, 0)

    def test_calculate_reply_context(self):
        """calculate should adjust for reply context"""
        delta = ContextDeltaCalculator.calculate(EnvState.QUIET, InteractionContext.REPLY)

        # Reply should lower pitch for identity
        self.assertLess(delta.delta_mean_f0_hz, 0)

    def test_calculate_initiator_context(self):
        """calculate should adjust for initiator context"""
        delta = ContextDeltaCalculator.calculate(EnvState.QUIET, InteractionContext.INITIATOR)

        # Initiator should have increased sustain
        self.assertGreater(delta.delta_sustain_level, 0)

    def test_calculate_for_grading_high(self):
        """calculate_for_grading should handle high scores"""
        delta = ContextDeltaCalculator.calculate_for_grading(0.8)

        # High grading should add urgency
        self.assertGreater(delta.delta_jitter, 0)

    def test_calculate_for_grading_low(self):
        """calculate_for_grading should handle low scores"""
        delta = ContextDeltaCalculator.calculate_for_grading(0.2)

        # Low grading should be stable
        self.assertLess(delta.delta_jitter, 0)

    def test_combine_deltas(self):
        """combine should combine multiple deltas"""
        delta1 = MicroDynamicsDelta(delta_mean_f0_hz=100.0, delta_loudness=0.1)
        delta2 = MicroDynamicsDelta(delta_mean_f0_hz=50.0, delta_duration_ms=20.0)

        combined = ContextDeltaCalculator.combine([delta1, delta2])

        self.assertEqual(combined.delta_mean_f0_hz, 150.0)
        self.assertEqual(combined.delta_loudness, 0.1)
        self.assertEqual(combined.delta_duration_ms, 20.0)


@unittest.skipIf(
    not BIO_ACOUSTIC_AGENT_AVAILABLE,
    f"Bio acoustic agent not available: {IMPORT_ERROR if not BIO_ACOUSTIC_AGENT_AVAILABLE else ''}",
)
class TestFormantBarrierValidator(unittest.TestCase):
    """Test FormantBarrierValidator class"""

    def test_validate_similar_sources(self):
        """validate should pass similar source and target"""
        source = SourceMetadata(mean_f0_hz=5000.0, harmonic_to_noise_ratio=15.0, entropy=0.3)
        target = SourceMetadata(mean_f0_hz=5100.0, harmonic_to_noise_ratio=16.0, entropy=0.32)

        result = FormantBarrierValidator.validate(source, target)
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.violations), 0)

    def test_validate_hnr_violation(self):
        """validate should detect HNR change too large"""
        source = SourceMetadata(harmonic_to_noise_ratio=10.0)
        target = SourceMetadata(harmonic_to_noise_ratio=30.0)  # 20 dB change

        result = FormantBarrierValidator.validate(source, target)
        self.assertFalse(result.is_valid)
        self.assertTrue(any("HNR" in v for v in result.violations))

    def test_validate_entropy_violation(self):
        """validate should detect entropy change too large"""
        source = SourceMetadata(entropy=0.2)
        target = SourceMetadata(entropy=0.8)  # 0.6 change

        result = FormantBarrierValidator.validate(source, target)
        self.assertFalse(result.is_valid)
        self.assertTrue(any("Entropy" in v for v in result.violations))

    def test_validate_modality_crossing_harmonic_to_transient(self):
        """validate should detect modality crossing harmonic->transient"""
        # Harmonic source
        source = SourceMetadata(
            harmonic_to_noise_ratio=20.0,
            entropy=0.2,
            duration_ms=200.0,
        )
        # Transient target
        target = SourceMetadata(
            harmonic_to_noise_ratio=5.0,
            entropy=0.7,
            duration_ms=50.0,
        )

        result = FormantBarrierValidator.validate(source, target)
        self.assertFalse(result.is_valid)
        self.assertTrue(any("FORMANT BARRIER" in v for v in result.violations))

    def test_validate_recommended_action(self):
        """validate should provide recommended action"""
        source = SourceMetadata(harmonic_to_noise_ratio=10.0)
        target = SourceMetadata(harmonic_to_noise_ratio=30.0)

        result = FormantBarrierValidator.validate(source, target)
        self.assertIsNotNone(result.recommended_action)
        self.assertIsInstance(result.recommended_action, str)


@unittest.skipIf(
    not BIO_ACOUSTIC_AGENT_AVAILABLE,
    f"Bio acoustic agent not available: {IMPORT_ERROR if not BIO_ACOUSTIC_AGENT_AVAILABLE else ''}",
)
class TestAcousticInventory(unittest.TestCase):
    """Test AcousticInventory class"""

    def test_inventory_creation(self):
        """AcousticInventory should be created"""
        inventory = AcousticInventory(species="marmoset")
        self.assertEqual(inventory.species, "marmoset")
        self.assertEqual(len(inventory.prototypes), 0)

    def test_inventory_add_prototype(self):
        """add_prototype should store prototypes"""
        inventory = AcousticInventory()

        prototype = AcousticPrototype(
            label="phee",
            metadata=SourceMetadata(mean_f0_hz=8000.0),
        )

        inventory.add_prototype(prototype)

        # Should be retrievable
        retrieved = inventory.get_prototype("phee")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.label, "phee")

    def test_inventory_get_nonexistent(self):
        """get_prototype should return None for unknown labels"""
        inventory = AcousticInventory()
        result = inventory.get_prototype("nonexistent")
        self.assertIsNone(result)

    def test_inventory_available_labels(self):
        """available_labels should return all labels"""
        inventory = AcousticInventory()

        inventory.add_prototype(AcousticPrototype(label="phee"))
        inventory.add_prototype(AcousticPrototype(label="tsik"))

        labels = inventory.available_labels()
        self.assertEqual(set(labels), {"phee", "tsik"})

    def test_inventory_response_strategy(self):
        """set_response_strategy and get_response_label should work"""
        inventory = AcousticInventory()

        inventory.set_response_strategy("phee", "phee_reply")
        result = inventory.get_response_label("phee")

        self.assertEqual(result, "phee_reply")

    def test_inventory_get_response_label_nonexistent(self):
        """get_response_label should return None for unknown input"""
        inventory = AcousticInventory()
        result = inventory.get_response_label("unknown")
        self.assertIsNone(result)

    def test_inventory_save_load(self):
        """save and load should persist inventory"""
        inventory = AcousticInventory(species="test_species")
        inventory.add_prototype(
            AcousticPrototype(
                label="test_call",
                metadata=SourceMetadata(mean_f0_hz=5000.0),
            )
        )
        inventory.set_response_strategy("test_call", "test_response")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            inventory.save(path)
            loaded = AcousticInventory.load(path)

            self.assertEqual(loaded.species, "test_species")
            self.assertIn("test_call", loaded.prototypes)
            self.assertEqual(loaded.get_response_label("test_call"), "test_response")
        finally:
            path.unlink()


@unittest.skipIf(
    not BIO_ACOUSTIC_AGENT_AVAILABLE,
    f"Bio acoustic agent not available: {IMPORT_ERROR if not BIO_ACOUSTIC_AGENT_AVAILABLE else ''}",
)
class TestBioAcousticAgent(unittest.TestCase):
    """Test BioAcousticAgent class"""

    def test_agent_creation(self):
        """BioAcousticAgent should be created"""
        inventory = AcousticInventory()
        agent = BioAcousticAgent(inventory)
        self.assertIsNotNone(agent)
        self.assertIsNotNone(agent.inventory)

    def test_agent_plan_synthesis(self):
        """plan_synthesis should return SynthesisPlan"""
        inventory = AcousticInventory()

        # Add a prototype
        prototype = AcousticPrototype(
            label="phee",
            metadata=SourceMetadata(mean_f0_hz=5000.0),
            modality=AcousticModality.HARMONIC,
        )
        inventory.add_prototype(prototype)

        agent = BioAcousticAgent(inventory)

        # Plan synthesis
        plan = agent.plan_synthesis(
            label="phee",
            environment=EnvState.WIND,
            context=InteractionContext.INITIATOR,
        )
        self.assertIsNotNone(plan)
        self.assertEqual(plan.source_label, "phee")
        self.assertIsInstance(plan, SynthesisPlan)

    def test_agent_plan_synthesis_unknown_label(self):
        """plan_synthesis should raise for unknown label"""
        inventory = AcousticInventory()
        agent = BioAcousticAgent(inventory)

        with self.assertRaises(ValueError):
            agent.plan_synthesis(label="nonexistent")

    def test_agent_plan_synthesis_with_grading(self):
        """plan_synthesis should accept grading score"""
        inventory = AcousticInventory()
        inventory.add_prototype(
            AcousticPrototype(
                label="alarm",
                metadata=SourceMetadata(mean_f0_hz=8000.0),
            )
        )

        agent = BioAcousticAgent(inventory)

        plan = agent.plan_synthesis(label="alarm", grading=0.8)
        self.assertIsNotNone(plan)
        # Grading affects jitter
        self.assertNotEqual(plan.delta.delta_jitter, 0.0)

    def test_agent_plan_synthesis_with_pitch_offset(self):
        """plan_synthesis should accept pitch offset"""
        inventory = AcousticInventory()
        inventory.add_prototype(
            AcousticPrototype(
                label="phee",
                metadata=SourceMetadata(mean_f0_hz=5000.0),
            )
        )

        agent = BioAcousticAgent(inventory)

        plan = agent.plan_synthesis(label="phee", pitch_offset=300.0)
        self.assertIsNotNone(plan)
        # Pitch offset should be included in delta
        # The combined delta includes the pitch offset

    def test_agent_generate_response(self):
        """generate_response should return plan and label"""
        inventory = AcousticInventory()
        inventory.add_prototype(
            AcousticPrototype(
                label="phee",
                metadata=SourceMetadata(mean_f0_hz=5000.0),
            )
        )
        inventory.set_response_strategy("phee", "phee_reply")
        inventory.add_prototype(
            AcousticPrototype(
                label="phee_reply",
                metadata=SourceMetadata(mean_f0_hz=4800.0),
            )
        )

        agent = BioAcousticAgent(inventory)

        plan, response_label = agent.generate_response(
            input_label="phee",
            environment=EnvState.QUIET,
        )

        self.assertEqual(response_label, "phee_reply")
        self.assertIsInstance(plan, SynthesisPlan)

    def test_agent_generate_response_default_same(self):
        """generate_response should default to same label if no strategy"""
        inventory = AcousticInventory()
        inventory.add_prototype(
            AcousticPrototype(
                label="unknown_call",
                metadata=SourceMetadata(mean_f0_hz=5000.0),
            )
        )

        agent = BioAcousticAgent(inventory)

        plan, response_label = agent.generate_response(
            input_label="unknown_call",
            environment=EnvState.QUIET,
        )

        # Should default to same label
        self.assertEqual(response_label, "unknown_call")


@unittest.skipIf(
    not BIO_ACOUSTIC_AGENT_AVAILABLE,
    f"Bio acoustic agent not available: {IMPORT_ERROR if not BIO_ACOUSTIC_AGENT_AVAILABLE else ''}",
)
class TestBioAcousticAgentEdgeCases(unittest.TestCase):
    """Test edge cases in BioAcousticAgent"""

    def test_agent_empty_inventory(self):
        """Should handle empty inventory"""
        inventory = AcousticInventory()
        agent = BioAcousticAgent(inventory)

        with self.assertRaises(ValueError):
            agent.plan_synthesis(label="any_call")

    def test_agent_with_synthesizer(self):
        """Agent should accept optional synthesizer"""
        inventory = AcousticInventory()
        mock_synthesizer = MagicMock()

        agent = BioAcousticAgent(inventory, synthesizer=mock_synthesizer)
        self.assertEqual(agent.synthesizer, mock_synthesizer)

    def test_extreme_environment_values(self):
        """Should handle extreme environment values"""
        inventory = AcousticInventory()
        inventory.add_prototype(
            AcousticPrototype(
                label="alarm",
                metadata=SourceMetadata(mean_f0_hz=8000.0),
            )
        )

        agent = BioAcousticAgent(inventory)

        # Storm environment
        plan = agent.plan_synthesis(label="alarm", environment=EnvState.STORM)
        self.assertIsNotNone(plan)
        self.assertIn("Storm", plan.description)


if __name__ == "__main__":
    unittest.main()
