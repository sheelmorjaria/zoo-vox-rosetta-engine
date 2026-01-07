"""
Animal Vocalization Analysis Framework - Core Module

This framework provides comprehensive tools for analyzing animal vocalizations
using the Universal Rosetta Stone methodology, with support for cross-species
analysis and cognitive intelligence capabilities.
"""

# Core data models
# Import functionality
from .data_import import DataImporter as VocalizationDataImporter
from .data_models import (
    AcousticFeatures,
    GrammarRule,
    Phrase,
    PhraseContext,
    Sentence,
    Species,
    SpeciesData,
    VocalizationDatabase,
    VocalizationModality,
)

# Query interface
from .query_interface import VocalizationQueryInterface, get_query_interface

# Semiotic analysis
from .semiotics import (
    SemioticAnalysisResult,
    SemioticContext,
    SemioticEngine,
    SemioticRelation,
    SemioticState,
)

__version__ = "1.0.0"
__author__ = "Sheel Morjaria"

__all__ = [
    # Data models
    'Species',
    'VocalizationModality',
    'AcousticFeatures',
    'Phrase',
    'PhraseContext',
    'Sentence',
    'GrammarRule',
    'SpeciesData',
    'VocalizationDatabase',

    # Import functionality
    'VocalizationDataImporter',

    # Query interface
    'VocalizationQueryInterface',
    'get_query_interface',

    # Semiotic analysis
    'SemioticEngine',
    'SemioticState',
    'SemioticRelation',
    'SemioticContext',
    'SemioticAnalysisResult'
]
