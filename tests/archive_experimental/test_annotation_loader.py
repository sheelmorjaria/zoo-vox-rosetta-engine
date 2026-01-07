#!/usr/bin/env python3
"""
Test Suite for Annotation Loader and Context Association
========================================================

Comprehensive tests for the annotation loader module supporting:
- ELAN .eaf format
- Praat .TextGrid format
- JSON format
- CSV format
- Context association functions

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from realtime.annotation_loader import (
    Annotation,
    AnnotationCollection,
    AnnotationLoader,
    AnnotationTrack,
    CSVAnnotationLoader,
    ELANAnnotationLoader,
    JSONAnnotationLoader,
    PraatTextGridLoader,
    associate_context_to_segments,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_eaf_content():
    """Sample ELAN .eaf file content."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<ANNOTATION_DOCUMENT AUTHOR="" DATE="2026-01-06" FORMAT="2.8" VERSION="2.8"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://www.mpi.nl/tools/elan/EAFv2.8.xsd">
    <HEADER MEDIA_FILE="" TIME_UNITS="milliseconds">
        <MEDIA_DESCRIPTOR MEDIA_URL="test.wav" MIME_TYPE="audio"/>
        <PROPERTY NAME="lastUsedAnnotationId">1</PROPERTY>
    </HEADER>
    <TIME_ORDER>
        <TIME_SLOT TIME_SLOT_ID="ts1" TIME_VALUE="0"/>
        <TIME_SLOT TIME_SLOT_ID="ts2" TIME_VALUE="500"/>
        <TIME_SLOT TIME_SLOT_ID="ts3" TIME_VALUE="1000"/>
        <TIME_SLOT TIME_SLOT_ID="ts4" TIME_VALUE="1500"/>
        <TIME_SLOT TIME_SLOT_ID="ts5" TIME_VALUE="2000"/>
    </TIME_ORDER>
    <TIER LINGUISTIC_TYPE_ID="default-lt" TIER_ID="context">
        <ALIGNABLE_ANNOTATION ANNOTATION_ID="a1"
            TIME_SLOT_REF1="ts1" TIME_SLOT_REF2="ts2">
            <ANNOTATION_VALUE>aggression</ANNOTATION_VALUE>
        </ALIGNABLE_ANNOTATION>
        <ALIGNABLE_ANNOTATION ANNOTATION_ID="a2"
            TIME_SLOT_REF1="ts3" TIME_SLOT_REF2="ts4">
            <ANNOTATION_VALUE>courtship:marmoset_A:male displaying</ANNOTATION_VALUE>
        </ALIGNABLE_ANNOTATION>
    </TIER>
    <TIER LINGUISTIC_TYPE_ID="default-lt" TIER_ID="individual">
        <ALIGNABLE_ANNOTATION ANNOTATION_ID="a3"
            TIME_SLOT_REF1="ts1" TIME_SLOT_REF2="ts2">
            <ANNOTATION_VALUE>marmoset_A</ANNOTATION_VALUE>
        </ALIGNABLE_ANNOTATION>
    </TIER>
</ANNOTATION_DOCUMENT>
"""


@pytest.fixture
def sample_textgrid_content():
    """Sample Praat .TextGrid file content."""
    return """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 2.5
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "context"
        xmin = 0
        xmax = 2.5
        intervals: size = 4
        intervals [1]:
            xmin = 0
            xmax = 0.5
            text = "aggression"
        intervals [2]:
            xmin = 0.5
            xmax = 1.0
            text = ""
        intervals [3]:
            xmin = 1.0
            xmax = 1.5
            text = "courtship:marmoset_B"
        intervals [4]:
            xmin = 1.5
            xmax = 2.5
            text = "food_discovery:marmoset_A:found fruit"
    item [2]:
        class = "IntervalTier"
        name = "individual"
        xmin = 0
        xmax = 2.5
        intervals: size = 3
        intervals [1]:
            xmin = 0
            xmax = 1.0
            text = "marmoset_A"
        intervals [2]:
            xmin = 1.0
            xmax = 1.5
            text = "marmoset_B"
        intervals [3]:
            xmin = 1.5
            xmax = 2.5
            text = "marmoset_A"
"""


@pytest.fixture
def sample_json_annotations():
    """Sample JSON annotations."""
    return {
        "metadata": {"species": "marmoset", "recording_date": "2026-01-06", "location": "cage_A"},
        "annotations": [
            {
                "id": "annot_1",
                "start_time_ms": 0.0,
                "end_time_ms": 500.0,
                "context": "aggression",
                "individual_id": "marmoset_A",
                "participant_role": "dominant",
                "notes": "Chase behavior",
                "confidence": 0.95,
            },
            {
                "id": "annot_2",
                "start_time_ms": 1000.0,
                "end_time_ms": 1500.0,
                "context": "courtship",
                "individual_id": "marmoset_B",
                "participant_role": "submissive",
                "notes": "Display behavior",
                "confidence": 0.88,
            },
            {
                "id": "annot_3",
                "start_time_ms": 2000.0,
                "end_time_ms": 2500.0,
                "context": "food_discovery",
                "individual_id": "marmoset_A",
                "notes": "Found fruit",
                "confidence": 1.0,
            },
        ],
    }


@pytest.fixture
def sample_csv_content():
    """Sample CSV annotations."""
    return """start_time_ms,end_time_ms,context,individual_id,notes,confidence
0.0,500.0,aggression,marmoset_A,Chase behavior,0.95
1000.0,1500.0,courtship,marmoset_B,Display behavior,0.88
2000.0,2500.0,food_discovery,marmoset_A,Found fruit,1.0
"""


# ============================================================================
# Annotation Data Structure Tests
# ============================================================================


class TestAnnotation:
    """Test Annotation dataclass."""

    def test_create_annotation(self):
        """Test creating a basic annotation."""
        annotation = Annotation(start_time_ms=0.0, end_time_ms=500.0, context="aggression")

        assert annotation.start_time_ms == 0.0
        assert annotation.end_time_ms == 500.0
        assert annotation.context == "aggression"
        assert annotation.individual_id is None

    def test_annotation_with_optional_fields(self):
        """Test creating annotation with optional fields."""
        annotation = Annotation(
            start_time_ms=0.0,
            end_time_ms=500.0,
            context="aggression",
            individual_id="marmoset_A",
            participant_role="dominant",
            location="cage_A",
            notes="Chase behavior",
            confidence=0.95,
        )

        assert annotation.individual_id == "marmoset_A"
        assert annotation.participant_role == "dominant"
        assert annotation.location == "cage_A"
        assert annotation.confidence == 0.95

    def test_overlaps_with(self):
        """Test overlaps_with method."""
        annotation = Annotation(start_time_ms=500.0, end_time_ms=1000.0, context="test")

        # Overlapping cases
        assert annotation.overlaps_with(400.0, 600.0) is True
        assert annotation.overlaps_with(500.0, 600.0) is True
        assert annotation.overlaps_with(600.0, 700.0) is True
        assert annotation.overlaps_with(900.0, 1100.0) is True

        # Non-overlapping cases
        assert annotation.overlaps_with(300.0, 400.0) is False
        assert annotation.overlaps_with(1100.0, 1200.0) is False
        assert annotation.overlaps_with(1000.0, 1100.0) is False  # end is exclusive

    def test_contains(self):
        """Test contains method."""
        annotation = Annotation(start_time_ms=500.0, end_time_ms=1000.0, context="test")

        # Contained cases
        assert annotation.contains(500.0) is True
        assert annotation.contains(600.0) is True
        assert annotation.contains(999.0) is True
        assert annotation.contains(1000.0) is True

        # Not contained cases
        assert annotation.contains(499.0) is False
        assert annotation.contains(1001.0) is False

    def test_to_dict(self):
        """Test serialization to dictionary."""
        annotation = Annotation(
            start_time_ms=100.0,
            end_time_ms=500.0,
            context="courtship",
            individual_id="marmoset_B",
            tier="behavior",
        )

        result = annotation.to_dict()

        assert result["start_time_ms"] == 100.0
        assert result["end_time_ms"] == 500.0
        assert result["context"] == "courtship"
        assert result["individual_id"] == "marmoset_B"
        assert result["tier"] == "behavior"


class TestAnnotationTrack:
    """Test AnnotationTrack class."""

    def test_create_and_add_annotations(self):
        """Test creating track and adding annotations."""
        track = AnnotationTrack(name="context")

        assert track.name == "context"
        assert len(track.annotations) == 0

        annotation = Annotation(start_time_ms=0.0, end_time_ms=500.0, context="aggression")

        track.add_annotation(annotation)

        assert len(track.annotations) == 1
        assert track.annotations[0].context == "aggression"

    def test_get_annotations_at_time(self):
        """Test getting annotations at a specific time."""
        track = AnnotationTrack(name="context")

        # Add annotations
        track.add_annotation(Annotation(0.0, 500.0, "aggression"))
        track.add_annotation(Annotation(500.0, 1000.0, "courtship"))
        track.add_annotation(Annotation(1000.0, 1500.0, "food"))

        # Query at various times
        assert len(track.get_annotations_at_time(250.0)) == 1
        assert len(track.get_annotations_at_time(750.0)) == 1
        assert len(track.get_annotations_at_time(1250.0)) == 1
        assert len(track.get_annotations_at_time(1600.0)) == 0  # Outside all

    def test_get_context_at_time(self):
        """Test getting context string at time."""
        track = AnnotationTrack(name="context")

        track.add_annotation(Annotation(0.0, 500.0, "aggression"))
        track.add_annotation(Annotation(500.0, 1000.0, "courtship"))

        assert track.get_context_at_time(250.0) == "aggression"
        assert track.get_context_at_time(750.0) == "courtship"
        assert track.get_context_at_time(1500.0) is None


class TestAnnotationCollection:
    """Test AnnotationCollection class."""

    def test_create_collection(self):
        """Test creating annotation collection."""
        collection = AnnotationCollection(source_file="test.json")

        assert collection.source_file == "test.json"
        assert len(collection.tracks) == 0
        assert isinstance(collection.tracks, dict)

    def test_add_and_get_tracks(self):
        """Test adding and retrieving tracks."""
        collection = AnnotationCollection(source_file="test.json")

        track1 = AnnotationTrack(name="context")
        track2 = AnnotationTrack(name="individual")

        collection.add_track(track1)
        collection.add_track(track2)

        assert len(collection.tracks) == 2
        assert collection.get_track("context") is track1
        assert collection.get_track("individual") is track2
        assert collection.get_track("nonexistent") is None

    def test_get_context_at_time(self):
        """Test getting context from specific track."""
        collection = AnnotationCollection(source_file="test.json")

        context_track = AnnotationTrack(name="context")
        context_track.add_annotation(Annotation(0.0, 1000.0, "aggression"))

        individual_track = AnnotationTrack(name="individual")
        individual_track.add_annotation(Annotation(0.0, 500.0, "marmoset_A"))

        collection.add_track(context_track)
        collection.add_track(individual_track)

        assert collection.get_context_at_time(500.0, "context") == "aggression"
        assert collection.get_context_at_time(500.0, "individual") == "marmoset_A"
        assert collection.get_context_at_time(500.0, "nonexistent") is None

    def test_get_primary_context(self):
        """Test getting primary context with priority."""
        collection = AnnotationCollection(source_file="test.json")

        track1 = AnnotationTrack(name="context")
        track1.add_annotation(Annotation(0.0, 1000.0, "aggression"))

        track2 = AnnotationTrack(name="behavior")
        track2.add_annotation(Annotation(0.0, 1000.0, "chase"))

        track3 = AnnotationTrack(name="event")
        track3.add_annotation(Annotation(0.0, 1000.0, "feeding"))

        collection.add_track(track1)
        collection.add_track(track2)
        collection.add_track(track3)

        # Default priority: context, behavior, event
        assert collection.get_primary_context(500.0) == "aggression"

        # Custom priority
        assert (
            collection.get_primary_context(500.0, priority=["event", "behavior", "context"])
            == "feeding"
        )

    def test_get_all_annotations_at_time(self):
        """Test getting all annotations from all tracks."""
        collection = AnnotationCollection(source_file="test.json")

        context_track = AnnotationTrack(name="context")
        context_track.add_annotation(Annotation(0.0, 1000.0, "aggression"))

        individual_track = AnnotationTrack(name="individual")
        individual_track.add_annotation(Annotation(0.0, 500.0, "marmoset_A"))

        collection.add_track(context_track)
        collection.add_track(individual_track)

        all_annotations = collection.get_all_annotations_at_time(250.0)

        assert "context" in all_annotations
        assert "individual" in all_annotations
        assert len(all_annotations["context"]) == 1
        assert len(all_annotations["individual"]) == 1


# ============================================================================
# ELAN Loader Tests
# ============================================================================


class TestELANAnnotationLoader:
    """Test ELAN .eaf file loader."""

    def test_load_eaf_file(self, sample_eaf_content):
        """Test loading ELAN .eaf file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".eaf", delete=False) as f:
            f.write(sample_eaf_content)
            temp_path = f.name

        try:
            loader = ELANAnnotationLoader()
            collection = loader.load(temp_path)

            assert collection.source_file == temp_path
            assert len(collection.tracks) > 0

            # Check context track
            context_track = collection.get_track("context")
            assert context_track is not None
            assert len(context_track.annotations) >= 1
            assert context_track.annotations[0].context == "aggression"

        finally:
            Path(temp_path).unlink()

    def test_parse_structured_context(self, sample_eaf_content):
        """Test parsing structured context (context:individual:notes)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".eaf", delete=False) as f:
            f.write(sample_eaf_content)
            temp_path = f.name

        try:
            loader = ELANAnnotationLoader()
            collection = loader.load(temp_path)

            context_track = collection.get_track("context")

            # Find annotation with structured context
            structured_annotation = None
            for annotation in context_track.annotations:
                if "courtship" in annotation.context:
                    structured_annotation = annotation
                    break

            assert structured_annotation is not None
            assert structured_annotation.context == "courtship"
            assert structured_annotation.individual_id == "marmoset_A"
            assert structured_annotation.notes == "male displaying"

        finally:
            Path(temp_path).unlink()


# ============================================================================
# Praat TextGrid Loader Tests
# ============================================================================


class TestPraatTextGridLoader:
    """Test Praat .TextGrid file loader."""

    def test_load_textgrid_file(self, sample_textgrid_content):
        """Test loading Praat .TextGrid file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".TextGrid", delete=False) as f:
            f.write(sample_textgrid_content)
            temp_path = f.name

        try:
            loader = PraatTextGridLoader()
            collection = loader.load(temp_path)

            assert collection.source_file == temp_path
            assert len(collection.tracks) >= 2

            # Check context track
            context_track = collection.get_track("context")
            assert context_track is not None

            # Should have 4 intervals (including empty)
            total_annotations = len(context_track.annotations)
            assert total_annotations >= 2  # At least 2 non-empty intervals

        finally:
            Path(temp_path).unlink()

    def test_parse_time_units(self, sample_textgrid_content):
        """Test that time units are correctly converted to milliseconds."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".TextGrid", delete=False) as f:
            f.write(sample_textgrid_content)
            temp_path = f.name

        try:
            loader = PraatTextGridLoader()
            collection = loader.load(temp_path)

            context_track = collection.get_track("context")

            # First interval: 0.0 to 0.5 seconds = 0 to 500 ms
            first_annotation = context_track.annotations[0]
            assert first_annotation.start_time_ms == 0.0
            assert first_annotation.end_time_ms == 500.0

        finally:
            Path(temp_path).unlink()


# ============================================================================
# JSON Loader Tests
# ============================================================================


class TestJSONAnnotationLoader:
    """Test JSON annotation loader."""

    def test_load_json_file(self, sample_json_annotations):
        """Test loading JSON annotations."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_json_annotations, f)
            temp_path = f.name

        try:
            loader = JSONAnnotationLoader()
            collection = loader.load(temp_path)

            assert collection.source_file == temp_path
            assert collection.metadata["species"] == "marmoset"
            assert len(collection.tracks) > 0

            # Check default track
            default_track = collection.get_track("default")
            assert default_track is not None
            assert len(default_track.annotations) == 3

        finally:
            Path(temp_path).unlink()

    def test_load_all_fields(self, sample_json_annotations):
        """Test that all JSON fields are loaded correctly."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_json_annotations, f)
            temp_path = f.name

        try:
            loader = JSONAnnotationLoader()
            collection = loader.load(temp_path)

            default_track = collection.get_track("default")

            # Check first annotation
            first = default_track.annotations[0]
            assert first.context == "aggression"
            assert first.individual_id == "marmoset_A"
            assert first.participant_role == "dominant"
            assert first.notes == "Chase behavior"
            assert first.confidence == 0.95
            assert first.annotation_id == "annot_1"

        finally:
            Path(temp_path).unlink()


# ============================================================================
# CSV Loader Tests
# ============================================================================


class TestCSVAnnotationLoader:
    """Test CSV annotation loader."""

    def test_load_csv_file(self, sample_csv_content):
        """Test loading CSV annotations."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(sample_csv_content)
            temp_path = f.name

        try:
            loader = CSVAnnotationLoader()
            collection = loader.load(temp_path)

            assert collection.source_file == temp_path
            assert len(collection.tracks) > 0

            default_track = collection.get_track("default")
            assert default_track is not None
            assert len(default_track.annotations) == 3

        finally:
            Path(temp_path).unlink()

    def test_parse_time_formats(self):
        """Test parsing different time formats."""
        csv_content = """start_time_ms,end_time_ms,context
0,500,aggression
1.0,1.5,courtship
00:00:02,00:00:02.5,food
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_path = f.name

        try:
            loader = CSVAnnotationLoader()
            collection = loader.load(temp_path)

            default_track = collection.get_track("default")

            # Milliseconds: 0-500
            assert default_track.annotations[0].start_time_ms == 0.0

            # Seconds (auto-detected and converted): 1.0s = 1000ms
            assert default_track.annotations[1].start_time_ms == 1000.0

            # HH:MM:SS format: 00:00:02 = 2s = 2000ms
            assert default_track.annotations[2].start_time_ms == 2000.0

        finally:
            Path(temp_path).unlink()


# ============================================================================
# Unified AnnotationLoader Tests
# ============================================================================


class TestAnnotationLoader:
    """Test unified annotation loader with auto-detect."""

    def test_auto_detect_eaf(self, sample_eaf_content):
        """Test auto-detection of .eaf files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".eaf", delete=False) as f:
            f.write(sample_eaf_content)
            temp_path = f.name

        try:
            loader = AnnotationLoader()
            collection = loader.load(temp_path)

            assert len(collection.tracks) > 0

        finally:
            Path(temp_path).unlink()

    def test_auto_detect_textgrid(self, sample_textgrid_content):
        """Test auto-detection of .TextGrid files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".TextGrid", delete=False) as f:
            f.write(sample_textgrid_content)
            temp_path = f.name

        try:
            loader = AnnotationLoader()
            collection = loader.load(temp_path)

            assert len(collection.tracks) >= 2

        finally:
            Path(temp_path).unlink()

    def test_auto_detect_json(self, sample_json_annotations):
        """Test auto-detection of .json files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_json_annotations, f)
            temp_path = f.name

        try:
            loader = AnnotationLoader()
            collection = loader.load(temp_path)

            assert len(collection.tracks) > 0

        finally:
            Path(temp_path).unlink()

    def test_auto_detect_csv(self, sample_csv_content):
        """Test auto-detection of .csv files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(sample_csv_content)
            temp_path = f.name

        try:
            loader = AnnotationLoader()
            collection = loader.load(temp_path)

            assert len(collection.tracks) > 0

        finally:
            Path(temp_path).unlink()

    def test_unsupported_format(self):
        """Test error handling for unsupported formats."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Not a valid annotation format")
            temp_path = f.name

        try:
            loader = AnnotationLoader()
            with pytest.raises(ValueError, match="Unsupported annotation format"):
                loader.load(temp_path)

        finally:
            Path(temp_path).unlink()

    def test_load_multiple_files(self):
        """Test loading and merging multiple annotation files."""
        # Create multiple JSON files
        json1 = {
            "metadata": {},
            "annotations": [{"start_time_ms": 0, "end_time_ms": 500, "context": "aggression"}],
        }

        json2 = {
            "metadata": {},
            "annotations": [{"start_time_ms": 1000, "end_time_ms": 1500, "context": "courtship"}],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f1:
            json.dump(json1, f1)
            path1 = f1.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f2:
            json.dump(json2, f2)
            path2 = f2.name

        try:
            loader = AnnotationLoader()
            merged = loader.load_multiple([path1, path2])

            # Should have tracks from both files, prefixed with filename
            assert len(merged.tracks) == 2

        finally:
            Path(path1).unlink()
            Path(path2).unlink()


# ============================================================================
# Context Association Tests
# ============================================================================


class TestContextAssociation:
    """Test context association helper functions."""

    def test_associate_context_to_segments(self):
        """Test associating context to audio segments."""
        # Create annotation collection
        collection = AnnotationCollection(source_file="test")

        context_track = AnnotationTrack(name="context")
        context_track.add_annotation(Annotation(0.0, 500.0, "aggression"))
        context_track.add_annotation(Annotation(500.0, 1000.0, "courtship"))
        context_track.add_annotation(Annotation(1000.0, 1500.0, "food"))

        # Use non-overlapping individual annotations
        individual_track = AnnotationTrack(name="individual")
        individual_track.add_annotation(Annotation(0.0, 500.0, "marmoset_A"))
        individual_track.add_annotation(Annotation(500.0, 1000.0, "marmoset_B"))
        individual_track.add_annotation(Annotation(1000.0, 1500.0, "marmoset_A"))

        collection.add_track(context_track)
        collection.add_track(individual_track)

        # Create segments with midpoints that clearly fall within one annotation each
        segments = [
            {
                "start_time_ms": 100.0,
                "end_time_ms": 400.0,
                "phrase_key": "phrase1",
            },  # midpoint: 250ms
            {
                "start_time_ms": 600.0,
                "end_time_ms": 900.0,
                "phrase_key": "phrase2",
            },  # midpoint: 750ms
            {
                "start_time_ms": 1100.0,
                "end_time_ms": 1400.0,
                "phrase_key": "phrase3",
            },  # midpoint: 1250ms
        ]

        # Associate context
        enriched = associate_context_to_segments(
            collection, segments, context_track="context", individual_track="individual"
        )

        # Verify associations
        assert enriched[0]["context"] == "aggression"
        assert enriched[0]["individual_id"] == "marmoset_A"

        assert enriched[1]["context"] == "courtship"
        assert enriched[1]["individual_id"] == "marmoset_B"

        assert enriched[2]["context"] == "food"
        assert enriched[2]["individual_id"] == "marmoset_A"


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for complete workflow."""

    def test_complete_workflow_json_to_context_association(self, sample_json_annotations):
        """Test complete workflow from JSON loading to context association."""
        # Save JSON to file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_json_annotations, f)
            temp_path = f.name

        try:
            # Load annotations
            loader = AnnotationLoader()
            collection = loader.load(temp_path)

            # Verify loading
            assert len(collection.tracks) > 0

            # Create segments for association
            segments = [
                {"start_time_ms": 100.0, "end_time_ms": 400.0, "phrase_key": "p1"},
                {"start_time_ms": 1200.0, "end_time_ms": 1400.0, "phrase_key": "p2"},
                {"start_time_ms": 2200.0, "end_time_ms": 2400.0, "phrase_key": "p3"},
            ]

            # Associate context
            enriched = associate_context_to_segments(collection, segments)

            # Verify
            assert enriched[0]["context"] == "aggression"
            assert enriched[1]["context"] == "courtship"
            assert enriched[2]["context"] == "food_discovery"

        finally:
            Path(temp_path).unlink()

    def test_serialization_roundtrip(self, sample_json_annotations):
        """Test that annotation collections can be serialized and deserialized."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_json_annotations, f)
            temp_path = f.name

        try:
            # Load
            loader = AnnotationLoader()
            collection = loader.load(temp_path)

            # Serialize
            serialized = collection.to_dict()

            # Verify structure
            assert "source_file" in serialized
            assert "tracks" in serialized
            assert "metadata" in serialized

            # Verify track data
            assert "default" in serialized["tracks"]
            assert len(serialized["tracks"]["default"]) == 3

        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
