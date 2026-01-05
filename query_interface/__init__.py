"""
Query Interface Module

This module provides high-performance query interfaces for accessing vocalization data
in real-time with various filtering, aggregation, and search capabilities.
"""

from .vocalization_query_interface import (
    VocalizationQueryInterface,
    get_query_interface,
    query_phrases_by_f0,
    query_phrases_by_duration,
    get_phrase_similarities,
    get_database_statistics
)

__all__ = [
    'VocalizationQueryInterface',
    'get_query_interface',
    'query_phrases_by_f0',
    'query_phrases_by_duration',
    'get_phrase_similarities',
    'get_database_statistics'
]