#!/usr/bin/env python3
"""
**SKIPPED - Module not found**

This test suite references a rosetta_stone_base module that does not
exist in the current codebase. The base classes are implemented in
analysis/rosetta_stone/universal_rosetta_stone.py instead.

Archived by: Sheel Morjaria
License: CC BY-ND 4.0 International
Archive Date: 2025

Original Test Purpose:
Unit tests for Rosetta Stone base classes and interfaces.

Tests the abstract base class, data structures, and shared interfaces
that all species-specific modules inherit from.
"""

import pytest

# Skip entire module - referenced module does not exist
pytestmark = pytest.mark.skip(
    reason="Module 'rosetta_stone_base' not found. "
           "Base classes are in analysis/rosetta_stone/universal_rosetta_stone.py"
)

