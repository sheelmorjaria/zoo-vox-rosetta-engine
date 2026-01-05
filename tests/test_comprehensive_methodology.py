#!/usr/bin/env python3
"""
Test suite for comprehensive METHODOLOGY_SUMMARY.md implementation.
"""

import sys
import os
sys.path.insert(0, '../..')

import numpy as np
import unittest
from collections import Counter
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Try to import from the correct path
try:
    from analysis.rosetta_stone.universal_rosetta_stone import (
        UniversalRosettaStone,
        Modality,
        PhraseSignature,
        Sentence
    )
except ImportError:
    try:
        from universal_rosetta_stone import (
            UniversalRosettaStone,
            Modality,
            PhraseSignature,
            Sentence
        )
    except ImportError:
        # Skip test if module not available
        import pytest
        pytestmark = pytest.mark.skip(
            reason="Module universal_rosetta_stone not found. "
                   "Expected at: analysis/rosetta_stone/universal_rosetta_stone.py"
        )

