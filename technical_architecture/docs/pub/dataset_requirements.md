Minimal Viable Dataset Requirements
To replicate these findings and deploy cognitive bioacoustic engines, we define a "Data Spectrum" based on the analytical capabilities of the Neural Boundary Detection (NBD) system.

5.1 The Role of Segmentation (NBD)
Unlike standard pipelines that require pre-segmented audio clips, the Zoo Vox Rosetta Engine utilizes Neural Boundary Detection (NBD). This allows the system to ingest continuous, raw field recordings (e.g., 30-minute hydrophone dumps) and autonomously extract "acoustic gestures."

Implication:
The requirement for "Sequential Audio" is satisfied by NBD segmentation. The engine can create its own sequences from unsegmented files. Therefore, the critical bottleneck is not the segmentation of the signal, but the annotation of the context.

5.2 The Hierarchy of Data Requirements
Level 0: Structural Discovery (Raw Recordings Only)

Requirements: Raw, continuous audio recordings (no segmentation or labels required).
Engine Capability:
NBD segments the audio into atomic units.
Syntactic Mining extracts N-gram templates.
Outcome: The system can determine if a species uses Discrete Syntax or a Graded Continuum.
Limitation: Segments and N-grams cannot be mapped to behavioral contexts. The system detects structure but not meaning.
Level 1: Semantic Discovery (Raw Recordings + Context Labels)

Requirements: Raw audio recordings + Time-aligned Context Labels (e.g., "Feeding," "Traveling").
Engine Capability:
NBD creates the sequence.
Context Labels map the sequences to behaviors.
Outcome: The system can predict context from syntax (e.g., "Pattern A = Feeding").
Upgrade Path: Legacy datasets of isolated calls can be concatenated into "pseudo-sequences" to simulate this level if context is known.
Level 2: Functional (Raw Recordings + Context + Emitter ID)

Requirements: Level 1 + Emitter ID (Identification of the caller).
Engine Capability:
Builds Individual Voice Libraries.
Distinguishes between "Shared Syntax" (species-wide) and "Dialects" (individual-specific).
Why it matters: For species where "Identity" dominates (e.g., Bats), this level is required to decouple "Who is speaking" from "What is said."
Level 3: Interaction-Ready (Raw Recordings + Context + Emitter + Receiver ID)

Requirements: Level 2 + Receiver ID (Identification of the addressee).
Engine Capability: Models social dynamics (e.g., Turn-Taking, Addressing, Hub-and-Spoke topologies).
Why it matters: Required for two-way communication systems that need to decide who to reply to.