#!/usr/bin/env python3
"""
Persona-Based Voice Switching Synthesis Engine

Upgrades granular synthesis to support "voice switching" - hot-swapping source
buffers based on acoustic personas for contextually appropriate vocalization.

This implements Step 4 of the persona operationalization roadmap:
- Rust Execution Layer: Signal processing, grain generation, crossfading
- Python Logic Layer: Persona selection, grain scheduling, context interpretation

Architecture:
    Intent (Python) → PersonaRouter → PersonaBufferManager → RustSynthesisEngine → Audio

Design Philosophy:
- "Hard Truth" (Real Recordings): High-fidelity baseline granular buffers
- "Soft Variation" (Pitch/Time Shift): Organic variation around baseline
- "Voice Switching" (Persona Buffers): Context-appropriate source selection

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")

# Import persona router (new system with persona_source_map.json)
from .persona_router import ContextState as CommunicationContext
from .persona_router import PersonaRouter, RoutingDecision

# Try to import Rust granular synthesizer
try:
    from technical_architecture.synthesis import (
        GranularConcatenativeSynthesizer as RustGranularSynthesizer,
    )

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    import warnings

    warnings.warn("Rust GranularConcatenativeSynthesizer not available, using Python fallback")


class SynthesisMode(Enum):
    """Granular synthesis modes."""

    CONCATENATIVE = "concatenative"  # Sequential grain placement
    SUPERPOSITIONAL = "superpositional"  # Layered/overlapping grains
    TESSELLATED = "tessellated"  # Tiled mosaic patterns


@dataclass
class GrainParameters:
    """Parameters for individual grain generation."""

    size_ms: float = 50.0  # Grain duration
    window: str = "hann"  # Windowing function
    pitch_shift_semitones: float = 0.0  # Pitch modulation
    time_stretch_factor: float = 1.0  # Time modulation
    amplitude_db: float = 0.0  # Amplitude adjustment

    def to_dict(self) -> Dict:
        return {
            "size_ms": self.size_ms,
            "window": self.window,
            "pitch_shift_semitones": self.pitch_shift_semitones,
            "time_stretch_factor": self.time_stretch_factor,
            "amplitude_db": self.amplitude_db,
        }


@dataclass
class PersonaBuffer:
    """Audio buffer associated with a specific persona."""

    persona_id: str
    audio_data: np.ndarray  # Source audio buffer
    sample_rate: int
    metadata: Dict = field(default_factory=dict)

    # Derived statistics for intelligent grain placement
    mean_f0_hz: float = 0.0
    f0_range_hz: float = 0.0
    rms_level: float = 0.0
    duration_ms: float = 0.0

    def __post_init__(self):
        """Compute derived statistics from audio data."""
        self.duration_ms = len(self.audio_data) / self.sample_rate * 1000
        self.rms_level = np.sqrt(np.mean(self.audio_data**2))

        # Extract metadata if provided
        if "mean_f0_hz" in self.metadata:
            self.mean_f0_hz = self.metadata["mean_f0_hz"]
        if "f0_range_hz" in self.metadata:
            self.f0_range_hz = self.metadata["f0_range_hz"]


@dataclass
class SynthesisRequest:
    """Request for audio synthesis with persona context."""

    species: str
    context: CommunicationContext
    target_duration_ms: float

    # Contextual parameters for PersonaRouter
    arousal_level: float = 0.0
    comm_distance: str = "mid"
    social_complexity: str = "medium"

    # Synthesis parameters
    mode: SynthesisMode = SynthesisMode.CONCATENATIVE
    grain_params: Optional[GrainParameters] = None

    # Variation parameters (Soft Variation)
    pitch_variation_semitones: float = 2.0  # Random pitch variation range
    time_variation_factor: float = 0.1  # Random time stretch range


class PersonaBufferManager:
    """
    Manages persona-based audio buffers for voice switching.

    This is the Python Logic Layer that decides which "voice" to use.
    The Rust Execution Layer handles the actual grain processing.

    Integration with persona_source_map.json:
    - Loads persona definitions from JSON
    - Routes based on species, context, arousal level
    - Supports contextual extrapolation
    """

    def __init__(self, sample_rate: int = 48000, persona_map_path: Optional[str] = None):
        self.sample_rate = sample_rate
        self.buffers: Dict[str, PersonaBuffer] = {}
        self.router = PersonaRouter(persona_map_path)

    def register_buffer(self, buffer: PersonaBuffer):
        """Register a persona buffer for voice switching."""
        self.buffers[buffer.persona_id] = buffer

    def load_buffer_from_file(
        self, persona_id: str, file_path: str, metadata: Optional[Dict] = None
    ):
        """
        Load persona buffer from WAV file.

        Args:
            persona_id: Persona identifier (e.g., 'MARMOSET_PHEE')
            file_path: Path to WAV file
            metadata: Optional acoustic metadata (F0, range, etc.)
        """
        try:
            import scipy.io.wavfile as wavfile

            sr, audio = wavfile.read(file_path)

            # Convert to float32 normalized
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
            elif audio.dtype == np.float32 or audio.dtype == np.float64:
                audio = audio.astype(np.float32)
            else:
                raise ValueError(f"Unsupported audio dtype: {audio.dtype}")

            # Convert to mono if stereo
            if len(audio.shape) == 2:
                audio = np.mean(audio, axis=1)

            buffer = PersonaBuffer(
                persona_id=persona_id, audio_data=audio, sample_rate=sr, metadata=metadata or {}
            )

            self.register_buffer(buffer)
            print(f"Loaded {persona_id} from {file_path}")

        except Exception as e:
            print(f"Failed to load {file_path}: {e}")

    def load_all_persona_buffers(self, buffer_dir: Optional[str] = None):
        """
        Load all persona buffers defined in persona_source_map.json.

        Args:
            buffer_dir: Base directory for buffer files (defaults to project root)
        """
        if buffer_dir is None:
            buffer_dir = str(Path(__file__).parent.parent)

        # Get all persona definitions
        for persona_id in self.router.list_personas():
            persona_def = self.router.get_persona_definition(persona_id)
            if persona_def is None:
                continue

            # Resolve file path
            file_path = Path(buffer_dir) / persona_def.source_file

            if file_path.exists():
                self.load_buffer_from_file(
                    persona_id=persona_id,
                    file_path=str(file_path),
                    metadata=persona_def.acoustic_profile,
                )
            else:
                print(f"Warning: Buffer file not found: {file_path}")

    def select_buffer(
        self, species: str, context: CommunicationContext, **kwargs
    ) -> Optional[PersonaBuffer]:
        """Select appropriate persona buffer based on context."""
        decision = self.router.select_persona(species, context, **kwargs)

        if decision.persona_id in self.buffers:
            return self.buffers[decision.persona_id]

        # Try to load from file if not in memory
        persona_def = self.router.get_persona_definition(decision.persona_id)
        if persona_def:
            file_path = Path(__file__).parent.parent / persona_def.source_file
            if file_path.exists():
                self.load_buffer_from_file(
                    persona_id=decision.persona_id,
                    file_path=str(file_path),
                    metadata=persona_def.acoustic_profile,
                )
                return self.buffers.get(decision.persona_id)

        return None

    def get_routing_decision(
        self, species: str, context: CommunicationContext, **kwargs
    ) -> Optional[RoutingDecision]:
        """Get routing decision without loading buffer."""
        return self.router.select_persona(species, context, **kwargs)

    def get_buffer(self, persona_id: str) -> Optional[PersonaBuffer]:
        """Get buffer by persona ID."""
        return self.buffers.get(persona_id)

    def list_available_personas(self, species: Optional[str] = None) -> List[str]:
        """List available personas, optionally filtered by species."""
        return self.router.list_personas(species)


class GranularVoiceSynthesizer:
    """
    High-level granular synthesis engine with persona voice switching.

    This implements the "Hard Truth + Soft Variation" philosophy:
    - Hard Truth: Real recordings as base material (persona buffers)
    - Soft Variation: Granular pitch/time shifting for organic variation
    - Voice Switching: Context-aware buffer selection

    Performance Note: This Python version is for prototyping and validation.
    Production deployments should use the Rust execution layer for
    time-critical grain processing.
    """

    def __init__(self, buffer_manager: PersonaBufferManager, sample_rate: int = 48000):
        self.buffer_manager = buffer_manager
        self.sample_rate = sample_rate

    def synthesize(self, request: SynthesisRequest) -> Tuple[np.ndarray, Dict]:
        """
        Synthesize audio based on persona context.

        Returns:
            (audio_output, synthesis_metadata)
        """
        # Step 1: Select appropriate persona buffer
        buffer = self.buffer_manager.select_buffer(
            species=request.species,
            context=request.context,
            arousal_level=request.arousal_level,
            comm_distance=request.comm_distance,
            social_complexity=request.social_complexity,
        )

        if buffer is None:
            raise ValueError(
                f"No buffer available for species={request.species}, context={request.context}"
            )

        # Step 2: Generate grain schedule based on mode
        grain_schedule = self._generate_grain_schedule(request, buffer)

        # Step 3: Render grains (Python implementation - Rust in production)
        audio_output = self._render_grains(grain_schedule, buffer, request)

        # Step 4: Apply post-processing
        audio_output = self._apply_post_processing(audio_output)

        metadata = {
            "persona_id": buffer.persona_id,
            "species": request.species,
            "context": request.context.value,
            "duration_ms": len(audio_output) / self.sample_rate * 1000,
            "num_grains": len(grain_schedule),
            "buffer_stats": {
                "mean_f0_hz": buffer.mean_f0_hz,
                "f0_range_hz": buffer.f0_range_hz,
                "duration_ms": buffer.duration_ms,
            },
        }

        return audio_output, metadata

    def _generate_grain_schedule(
        self, request: SynthesisRequest, buffer: PersonaBuffer
    ) -> List[Dict]:
        """Generate grain placement schedule based on synthesis mode."""

        grain_params = request.grain_params or GrainParameters()

        if request.mode == SynthesisMode.CONCATENATIVE:
            return self._schedule_concatenative(request, buffer, grain_params)
        elif request.mode == SynthesisMode.SUPERPOSITIONAL:
            return self._schedule_superpositional(request, buffer, grain_params)
        else:  # TESSELLATED
            return self._schedule_tessellated(request, buffer, grain_params)

    def _schedule_concatenative(
        self, request: SynthesisRequest, buffer: PersonaBuffer, grain_params: GrainParameters
    ) -> List[Dict]:
        """Schedule grains sequentially (concatenative synthesis)."""

        grain_size_samples = int(grain_params.size_ms * self.sample_rate / 1000)
        target_duration_samples = int(request.target_duration_ms * self.sample_rate / 1000)

        # Overlap grains by 50% for smooth transitions
        overlap_samples = grain_size_samples // 2
        grain_spacing = grain_size_samples - overlap_samples

        num_grains = int(np.ceil(target_duration_samples / grain_spacing))
        schedule = []

        for i in range(num_grains):
            # Random position within buffer (Hard Truth)
            max_start = len(buffer.audio_data) - grain_size_samples
            if max_start <= 0:
                start_sample = 0
            else:
                start_sample = np.random.randint(0, max_start)

            # Soft Variation: Random pitch shift
            pitch_shift = np.random.uniform(
                -request.pitch_variation_semitones, request.pitch_variation_semitones
            )

            # Soft Variation: Random time stretch
            time_stretch = np.random.uniform(
                1.0 - request.time_variation_factor, 1.0 + request.time_variation_factor
            )

            schedule.append(
                {
                    "start_sample": start_sample,
                    "output_position": i * grain_spacing,
                    "pitch_shift_semitones": pitch_shift,
                    "time_stretch_factor": time_stretch,
                    "amplitude_db": grain_params.amplitude_db,
                }
            )

        return schedule

    def _schedule_superpositional(
        self, request: SynthesisRequest, buffer: PersonaBuffer, grain_params: GrainParameters
    ) -> List[Dict]:
        """Schedule grains with overlap (layered synthesis)."""

        grain_size_samples = int(grain_params.size_ms * self.sample_rate / 1000)
        target_duration_samples = int(request.target_duration_ms * self.sample_rate / 1000)

        # High overlap for layered effect
        overlap_samples = int(grain_size_samples * 0.75)
        grain_spacing = grain_size_samples - overlap_samples

        num_grains = int(np.ceil(target_duration_samples / grain_spacing))
        schedule = []

        for i in range(num_grains):
            max_start = len(buffer.audio_data) - grain_size_samples
            if max_start <= 0:
                start_sample = 0
            else:
                start_sample = np.random.randint(0, max_start)

            pitch_shift = np.random.uniform(
                -request.pitch_variation_semitones * 0.5,  # Less variation for layered
                request.pitch_variation_semitones * 0.5,
            )

            schedule.append(
                {
                    "start_sample": start_sample,
                    "output_position": i * grain_spacing,
                    "pitch_shift_semitones": pitch_shift,
                    "time_stretch_factor": 1.0,  # No time stretch for layered
                    "amplitude_db": grain_params.amplitude_db - 3,  # Lower amplitude per layer
                }
            )

        return schedule

    def _schedule_tessellated(
        self, request: SynthesisRequest, buffer: PersonaBuffer, grain_params: GrainParameters
    ) -> List[Dict]:
        """Schedule grains in tessellated mosaic pattern."""

        grain_size_samples = int(grain_params.size_ms * self.sample_rate / 1000)
        target_duration_samples = int(request.target_duration_ms * self.sample_rate / 1000)

        # Regular spacing with gaps
        grain_spacing = grain_size_samples * 2  # 50% density

        num_grains = int(np.ceil(target_duration_samples / grain_spacing))
        schedule = []

        for i in range(num_grains):
            max_start = len(buffer.audio_data) - grain_size_samples
            if max_start <= 0:
                start_sample = 0
            else:
                start_sample = np.random.randint(0, max_start)

            # Alternating pitch pattern
            pitch_shift = (
                request.pitch_variation_semitones
                if i % 2 == 0
                else -request.pitch_variation_semitones
            )

            schedule.append(
                {
                    "start_sample": start_sample,
                    "output_position": i * grain_spacing,
                    "pitch_shift_semitones": pitch_shift,
                    "time_stretch_factor": 1.0,
                    "amplitude_db": grain_params.amplitude_db,
                }
            )

        return schedule

    def _render_grains(
        self, schedule: List[Dict], buffer: PersonaBuffer, request: SynthesisRequest
    ) -> np.ndarray:
        """
        Render grain schedule to audio (Python implementation).

        NOTE: This is a simplified implementation for prototyping.
        Production deployments should use Rust execution layer for:
        - High-quality pitch shifting (phase vocoder)
        - Efficient time stretching
        - Parallel grain processing
        - Zero-copy operations
        """

        target_duration_samples = int(request.target_duration_ms * self.sample_rate / 1000)
        output = np.zeros(target_duration_samples)

        for grain in schedule:
            # Extract grain from buffer
            start = grain["start_sample"]
            grain_size_samples = (
                int(request.grain_params.size_ms * self.sample_rate / 1000)
                if request.grain_params
                else int(50 * self.sample_rate / 1000)
            )

            grain_audio = buffer.audio_data[start : start + grain_size_samples]

            if len(grain_audio) < grain_size_samples:
                continue  # Skip incomplete grains

            # Apply window
            window = np.hanning(len(grain_audio))
            grain_audio = grain_audio * window

            # Simple pitch shift (resampling - NOT phase vocoder quality)
            pitch_shift = grain["pitch_shift_semitones"]
            if abs(pitch_shift) > 0.1:
                pitch_factor = 2 ** (pitch_shift / 12.0)
                grain_audio = np.interp(
                    np.linspace(0, len(grain_audio), int(len(grain_audio) / pitch_factor)),
                    np.arange(len(grain_audio)),
                    grain_audio,
                )

            # Place grain in output
            output_pos = grain["output_position"]
            grain_end = output_pos + len(grain_audio)

            if grain_end > len(output):
                grain_end = len(output)

            # Mix with existing audio (additive for superposition)
            output[output_pos:grain_end] += grain_audio[: grain_end - output_pos]

        return output

    def _apply_post_processing(self, audio: np.ndarray) -> np.ndarray:
        """Apply normalization and limiting."""

        # Normalize to prevent clipping
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.95

        return audio


def create_demo_buffers(buffer_manager: PersonaBufferManager, sample_rate: int = 48000):
    """
    Create synthetic demo buffers for testing.

    NOTE: In production, these will be replaced with real recordings from
    the vocalization database. The paths in persona_source_map.json will
    point to actual WAV files extracted from the database.
    """

    duration = 1.0  # 1 second buffers
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Create synthetic buffers based on persona_source_map.json acoustic profiles

    # Marmoset Phee buffer (stable 6.5kHz tone, narrow range 427 Hz)
    phee_audio = 0.5 * np.sin(2 * np.pi * 6526 * t)
    phee_buffer = PersonaBuffer(
        persona_id="MARMOSET_PHEE",
        audio_data=phee_audio,
        sample_rate=sample_rate,
        metadata={"mean_f0_hz": 6526.0, "f0_range_hz": 427.0, "harmonicity": 0.95},
    )

    # Marmoset Alarm buffer (modulated 6kHz tone, wide range 3722 Hz)
    alarm_fm = 2 * np.pi * 6020 * t + 0.5 * np.sin(2 * np.pi * 50 * t)  # 50Hz modulation
    alarm_audio = 0.5 * np.sin(alarm_fm)
    alarm_buffer = PersonaBuffer(
        persona_id="MARMOSET_ALARM",
        audio_data=alarm_audio,
        sample_rate=sample_rate,
        metadata={"mean_f0_hz": 6020.0, "f0_range_hz": 3722.0, "harmonicity": 0.7},
    )

    # Bat Mid-FM buffer (frequency modulated sweep, 7.4kHz anchor)
    bat_fm_chirp = 2 * np.pi * (7437 + 5000 * np.sin(2 * np.pi * 20 * t)) * t
    bat_mid_audio = 0.5 * np.sin(bat_fm_chirp)
    bat_mid_buffer = PersonaBuffer(
        persona_id="BAT_MID_FM",
        audio_data=bat_mid_audio,
        sample_rate=sample_rate,
        metadata={"mean_f0_hz": 7437.0, "f0_range_hz": 9755.0, "harmonicity": 0.6},
    )

    # Bat Social Ultrasound buffer (stable 7.4kHz, very narrow range 24 Hz)
    bat_social_audio = 0.5 * np.sin(2 * np.pi * 7408 * t)
    bat_social_buffer = PersonaBuffer(
        persona_id="BAT_SOCIAL_US",
        audio_data=bat_social_audio,
        sample_rate=sample_rate,
        metadata={"mean_f0_hz": 7408.0, "f0_range_hz": 24.0, "harmonicity": 0.85},
    )

    # Bat Low Social buffer (2.9kHz, very wide FM)
    bat_low_fm = 2 * np.pi * (2884 + 6000 * np.sin(2 * np.pi * 30 * t)) * t
    bat_low_audio = 0.5 * np.sin(bat_low_fm)
    bat_low_buffer = PersonaBuffer(
        persona_id="BAT_LOW_SOCIAL",
        audio_data=bat_low_audio,
        sample_rate=sample_rate,
        metadata={"mean_f0_hz": 2884.0, "f0_range_hz": 11535.0, "harmonicity": 0.5},
    )

    for buf in [phee_buffer, alarm_buffer, bat_mid_buffer, bat_social_buffer, bat_low_buffer]:
        buffer_manager.register_buffer(buf)

    print("\nCreated synthetic demo buffers (production will use real recordings)")
    print(f"  Registered {len(buffer_manager.buffers)} personas")


def demonstrate_voice_switching():
    """
    Demonstrate persona-based voice switching with persona_source_map.json.

    This demo shows:
    1. Loading persona definitions from JSON
    2. Context-aware voice selection
    3. Granular synthesis with different personas
    4. Routing based on species, context, arousal level
    """

    print("\n" + "=" * 80)
    print("PERSONA VOICE SWITCHING DEMONSTRATION")
    print("Integrated with persona_source_map.json")
    print("=" * 80)

    # Setup
    sample_rate = 48000
    buffer_manager = PersonaBufferManager(sample_rate)

    # Try to load real buffers, fall back to synthetic
    print("\nAttempting to load real persona buffers...")
    try:
        buffer_manager.load_all_persona_buffers()
        if len(buffer_manager.buffers) == 0:
            print("No real buffers found, creating synthetic demo buffers")
            create_demo_buffers(buffer_manager, sample_rate)
    except Exception as e:
        print(f"Error loading buffers: {e}")
        print("Creating synthetic demo buffers")
        create_demo_buffers(buffer_manager, sample_rate)

    synthesizer = GranularVoiceSynthesizer(buffer_manager, sample_rate)

    # Test scenarios from persona_source_map.json routing rules
    test_scenarios = [
        # (species, context, params, description)
        (
            "marmoset",
            CommunicationContext.CONTACT,
            {"arousal_level": 0.3},
            "Marmoset neutral phee (should use MARMOSET_PHEE)",
        ),
        (
            "marmoset",
            CommunicationContext.ALARM,
            {"arousal_level": 0.9},
            "Marmoset high-arousal alarm (should use MARMOSET_ALARM)",
        ),
        (
            "egyptian_bat",
            CommunicationContext.NEUTRAL,
            {"comm_distance": "navigation"},
            "Bat navigation (should use BAT_MID_FM)",
        ),
        (
            "egyptian_bat",
            CommunicationContext.CONTACT,
            {"comm_distance": "roost"},
            "Bat roost communication (should use BAT_LOW_SOCIAL)",
        ),
        (
            "egyptian_bat",
            CommunicationContext.CONTACT,
            {"comm_distance": "close", "social_complexity": "high"},
            "Bat close-range social (should use BAT_SOCIAL_US)",
        ),
    ]

    print("\nSynthesis Scenarios:")
    print("-" * 80)

    for species, context, params, description in test_scenarios:
        request = SynthesisRequest(
            species=species,
            context=context,
            target_duration_ms=500.0,
            mode=SynthesisMode.CONCATENATIVE,
            grain_params=GrainParameters(size_ms=50.0),
            **params,
        )

        try:
            # Get routing decision first
            decision = buffer_manager.get_routing_decision(species, context, **params)

            audio, metadata = synthesizer.synthesize(request)

            print(f"\n{description}")
            print(f"  Persona: {metadata['persona_id']}")
            print(f"  Reasoning: {decision.reasoning if decision else 'N/A'}")
            print(f"  Duration: {metadata['duration_ms']:.1f} ms")
            print(f"  Grains: {metadata['num_grains']}")
            print(f"  Buffer F0: {metadata['buffer_stats']['mean_f0_hz']:.0f} Hz")
            print(f"  Output RMS: {np.sqrt(np.mean(audio**2)):.4f}")

        except ValueError as e:
            print(f"\n{description}")
            print(f"  ❌ Error: {e}")

    print("\n" + "=" * 80)
    print("\n💡 PRODUCTION DEPLOYMENT:")
    print("   ✓ Persona routing defined in persona_source_map.json")
    print("   ✓ Contextual extrapolation vectors configured")
    print("   → Add real WAV buffer files to buffers/ directory")
    print("   → Use Rust execution layer for grain processing")
    print("   → Implement phase vocoder for high-quality pitch shifting")
    print("   → Add zero-copy PyO3 bindings for Python-Rust integration")
    print()


if __name__ == "__main__":
    demonstrate_voice_switching()
