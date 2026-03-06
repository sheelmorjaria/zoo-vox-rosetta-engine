# Semantic Reconstruction Pipeline

     4. ↮
     3. Corpus Analysis (Feature Vectors → Cluster IDs → N-gram Templates)
     4. Synthesis Output (Synthetic Audio Playback)
    2.
This pipeline implements TDD (Test-driven development) for the semantic reconstruction
    system described in the plan file.

---

## Overview

    This document describes the **Semantic Reconstruction Pipeline** (STAGE 4) of the audio synthesis workflow.
    The pipeline consists of **5 stages**:
        1.  **NBD Segmentation**: Raw Audio → Isolated Segments
        2.  **112D Feature Extraction**: Segments → Feature Vectors + audio buffers
        3.  **Corpus Analysis**: Feature vectors → Cluster IDs (k=1020) → N-gram templates
        4.  **Semantic Reconstruction**: Exemplar Manager + Metadata Mapper + Cached Granular Synthesizer
        5.  **Synthesis Output**: N-gram templates → Synthetic Audio

    The **Key Components:**
    - **ExemplarManager**: Stores best audio exemplar per cluster ID
    - **CachedGranularSynthesizer**: Caches audio buffers with metadata for synthesis
    - **SynthesisTimeline**: Represents synthesis timeline from N-gram templates
    - **SourceMetadata**: Maps 112D features to metadata
    - **TimelineEvent**: Individual events in the synthesis timeline
    - **MetadataMapper**: Translates 112D features to synthesis parameters

    The **Data Flow:**
    ```
    Raw Audio
        ↓
    [NBD Segmentation]
        ↓
    [112D Feature Extraction]
        ↓
    [Corpus Analysis (Clustering)]
        ↓
    [Semantic Reconstruction] ← THIS STAGE
        ↓
    [Synthesis Output]
    ```
    end-to-end pipeline:
    - Extract features from raw audio
    - Cluster similar segments
    - Synthesize new audio from clusters based on exemplars
    - Output final synthesized audio
