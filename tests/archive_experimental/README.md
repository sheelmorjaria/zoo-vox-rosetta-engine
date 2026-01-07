# Archived Tests - Experimental Features

**Date Archived:** 2026-01-07

**Reason:** These tests reference experimental modules that were archived to `archive/experimental_analysis/` and `archive/experimental_realtime/`.

---

## Archived Test Files

| Test File | Module Tested | Location of Archived Module | Reason for Archiving |
|-----------|---------------|----------------------------|----------------------|
| `test_acoustic_algebra.py` | `analysis.rosetta_stone.acoustic_algebra` | `archive/experimental_analysis/acoustic_algebra.py` | Superseded by `high_dimensional_acoustic_algebra.py` (17D features) |
| `test_acoustic_algebra_contextual.py` | `realtime.acoustic_algebra_contextual` | `archive/experimental_realtime/acoustic_algebra_contextual.py` | Experimental feature not used in production |
| `test_annotation_loader.py` | `realtime.annotation_loader` | `archive/experimental_realtime/annotation_loader.py` | Experimental feature not used in production |
| `test_bio_acoustic_validation.py` | `realtime.bio_acoustic_validator` | `archive/experimental_realtime/bio_acoustic_validator.py` | Experimental feature not used in production |
| `test_high_dimensional_algebra.py` | `analysis.rosetta_stone.high_dimensional_acoustic_algebra` + `grain_based_grammar_discovery` | `archive/experimental_analysis/grain_based_grammar_discovery.py` | Depends on experimental `grain_based_grammar_discovery` |

---

## Active Replacements

### Acoustic Algebra
- **New Module:** `analysis/rosetta_stone/high_dimensional_acoustic_algebra.py`
- **Enhanced Features:** 17-dimensional feature vectors (vs. 7D in original)
- **Status:** Active and tested

### Other Features
- Annotation loading, bioacoustic validation, and grain-based grammar discovery remain experimental
- No active replacements currently deployed
- May be revisited in future research

---

## Migration Guide

If you need to restore these tests:

1. **For acoustic algebra:** Use `test_high_dimensional_algebra.py` instead (if grain_based_grammar_discovery dependency is resolved)
2. **For experimental features:** Restore modules from `archive/experimental_*/` to active directories
3. **Or delete these tests entirely** if the experimental features are deprecated

---

## Test Count Impact

- **Tests Archived:** 5 test files (~50+ test cases)
- **Tests Remaining:** 321 tests (from original 329)
- **Active Test Suite:** Fully functional

---

**Author:** Sheel Morjaria (sheelmorjaria@gmail.com)
**License:** CC BY-ND 4.0 International
