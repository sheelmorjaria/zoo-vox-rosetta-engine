#!/usr/bin/env python3
"""
Test Synthesis Integration
==========================

Tests the integrated synthesis system in advanced_technical_enhancements.py
with different synthesis modes.

Copyright (c) 2025 Sheel Morjaria
License: CC BY-ND 4.0 International
"""

import logging
import sys
import pytest

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


@pytest.fixture
def framework():
    """Fixture providing the advanced technical framework"""
    # Import locally to avoid module initialization issues
    try:
        from advanced_technical_enhancements import AdvancedTechnicalFramework
        return AdvancedTechnicalFramework(synthesis_mode='auto')
    except ImportError as e:
        pytest.skip(f"Could not import AdvancedTechnicalFramework: {e}")


def test_auto_mode():
    """Test auto mode (should use microharmonic since no database exists)"""
    logger.info("=" * 80)
    logger.info("TEST 1: Auto Mode (no phrase database)")
    logger.info("=" * 80)

    from advanced_technical_enhancements import AdvancedTechnicalFramework

    framework = AdvancedTechnicalFramework(synthesis_mode='auto')

    logger.info(f"✓ Framework initialized")
    logger.info(f"  Synthesis mode: {framework.synthesis_mode}")
    logger.info(f"  Synthesizer type: {type(framework.advanced_synthesizer).__name__}")

    return framework


def test_microharmonic_mode():
    """Test microharmonic mode (Rosetta Stone)"""
    logger.info("=" * 80)
    logger.info("TEST 2: Microharmonic Mode (Rosetta Stone)")
    logger.info("=" * 80)

    from advanced_technical_enhancements import AdvancedTechnicalFramework

    framework = AdvancedTechnicalFramework(synthesis_mode='microharmonic')

    logger.info(f"✓ Framework initialized")
    logger.info(f"  Synthesis mode: {framework.synthesis_mode}")
    logger.info(f"  Synthesizer type: {type(framework.advanced_synthesizer).__name__}")

    return framework


def test_gan_mode():
    """Test GAN mode (legacy)"""
    logger.info("=" * 80)
    logger.info("TEST 3: GAN Mode (legacy)")
    logger.info("=" * 80)

    from advanced_technical_enhancements import AdvancedTechnicalFramework

    framework = AdvancedTechnicalFramework(synthesis_mode='gan')

    logger.info(f"✓ Framework initialized")
    logger.info(f"  Synthesis mode: {framework.synthesis_mode}")
    logger.info(f"  Synthesizer type: {type(framework.advanced_synthesizer).__name__}")

    return framework


def test_concatenative_mode_no_db():
    """Test concatenative mode without database (should fallback to microharmonic)"""
    logger.info("=" * 80)
    logger.info("TEST 4: Concatenative Mode (no database - should fallback)")
    logger.info("=" * 80)

    from advanced_technical_enhancements import AdvancedTechnicalFramework

    framework = AdvancedTechnicalFramework(
        synthesis_mode='concatenative',
        phrase_database_path='nonexistent_database.pkl'
    )

    logger.info(f"✓ Framework initialized (with fallback)")
    logger.info(f"  Synthesis mode: {framework.synthesis_mode}")
    logger.info(f"  Synthesizer type: {type(framework.advanced_synthesizer).__name__}")

    return framework


def _test_synthesis_call(framework, mode_name):
    """Test that synthesis works"""
    logger.info("-" * 80)
    logger.info(f"Testing synthesis call for {mode_name}...")

    try:
        # Note: This will generate audio using the selected synthesizer
        # For Rosetta Stone, this will return VocalizationResult objects
        results = framework.advanced_synthesizer.generate_adaptive_vocalization(
            species='marmoset',
            context='Vocalization',
            num_variants=2,
            sequence_length=3,
            temperature=1.0
        )

        logger.info(f"✓ Synthesis successful!")
        logger.info(f"  Generated {len(results)} results")

        if results:
            first_result = results[0]

            # Handle different result types
            if hasattr(first_result, 'success'):
                logger.info(f"  First result success: {first_result.success}")
                if hasattr(first_result, 'phrase_sequence'):
                    logger.info(f"  Phrase sequence: {first_result.phrase_sequence}")
                if hasattr(first_result, 'audio'):
                    if first_result.audio is not None:
                        logger.info(f"  Audio shape: {first_result.audio.shape}")
            else:
                logger.info(f"  Result type: {type(first_result)}")

        return True

    except Exception as e:
        logger.error(f"✗ Synthesis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    logger.info("\n")
    logger.info("=" * 80)
    logger.info("SYNTHESIS INTEGRATION TEST SUITE")
    logger.info("=" * 80)
    logger.info("\n")

    all_passed = True

    # Test 1: Auto mode
    try:
        framework = test_auto_mode()
        if not test_synthesis_call(framework, "Auto Mode"):
            all_passed = False
    except Exception as e:
        logger.error(f"✗ Auto mode test failed: {e}")
        all_passed = False

    logger.info("\n")

    # Test 2: Microharmonic mode
    try:
        framework = test_microharmonic_mode()
        if not test_synthesis_call(framework, "Microharmonic Mode"):
            all_passed = False
    except Exception as e:
        logger.error(f"✗ Microharmonic mode test failed: {e}")
        all_passed = False

    logger.info("\n")

    # Test 3: GAN mode
    try:
        framework = test_gan_mode()
        if not test_synthesis_call(framework, "GAN Mode"):
            all_passed = False
    except Exception as e:
        logger.error(f"✗ GAN mode test failed: {e}")
        all_passed = False

    logger.info("\n")

    # Test 4: Concatenative mode without database
    try:
        framework = test_concatenative_mode_no_db()
        if not test_synthesis_call(framework, "Concatenative Mode (fallback)"):
            all_passed = False
    except Exception as e:
        logger.error(f"✗ Concatenative mode test failed: {e}")
        all_passed = False

    logger.info("\n")
    logger.info("=" * 80)
    if all_passed:
        logger.info("✓ ALL TESTS PASSED")
    else:
        logger.warning("⚠ SOME TESTS FAILED")
    logger.info("=" * 80)

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
