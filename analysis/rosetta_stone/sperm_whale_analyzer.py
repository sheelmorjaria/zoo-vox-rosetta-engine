#!/usr/bin/env python3
"""
Sperm Whale-Specific Analyzer

Combines click detection with modality classification for sperm whale vocalizations.
Handles the unique characteristics of sperm whale clicks and codas:
- Dense click trains (1,600+ clicks/minute in echolocation)
- Coda patterns (discrete click patterns with 3-200+ clicks)
- Inter-click intervals (ICIs) as the primary information carrier
- Frequency range: 2-15 kHz with peak around 8-10 kHz
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, hilbert

sys.path.insert(0, str(Path(__file__).parent))
from universal_rosetta_stone import UniversalRosettaStone


@dataclass
class Click:
    """Individual sperm whale click."""
    position_samples: int
    position_ms: float
    amplitude: float
    width_samples: int


@dataclass
class Coda:
    """Sperm whale coda (discrete click pattern)."""
    clicks: List[Click]
    start_ms: float
    end_ms: float
    duration_ms: float
    num_clicks: int
    inter_click_intervals_ms: List[float] = field(init=False)

    def __post_init__(self):
        """Calculate inter-click intervals."""
        self.inter_click_intervals_ms = []
        for i in range(1, len(self.clicks)):
            interval = self.clicks[i].position_ms - self.clicks[i-1].position_ms
            self.inter_click_intervals_ms.append(interval)

    @property
    def mean_ici_ms(self) -> float:
        """Mean inter-click interval in milliseconds."""
        if not self.inter_click_intervals_ms:
            return 0.0
        return np.mean(self.inter_click_intervals_ms)

    @property
    def std_ici_ms(self) -> float:
        """Standard deviation of inter-click intervals."""
        if not self.inter_click_intervals_ms:
            return 0.0
        return np.std(self.inter_click_intervals_ms)

    @property
    def rhythm_regularity(self) -> float:
        """
        Rhythm regularity score (0-1, higher = more regular).
        Based on coefficient of variation of ICIs.
        """
        if self.mean_ici_ms == 0:
            return 0.0
        cv = self.std_ici_ms / self.mean_ici_ms
        # Convert CV to regularity score (lower CV = higher regularity)
        regularity = 1.0 / (1.0 + cv)
        return regularity


@dataclass
class SpermWhaleAnalysis:
    """Complete analysis of a sperm whale recording."""
    filepath: str
    duration_sec: float
    sample_rate: int
    clicks: List[Click]
    codas: List[Coda]

    # Click statistics
    clicks_per_second: float = field(init=False)
    total_clicks: int = field(init=False)

    # Coda statistics
    total_codas: int = field(init=False)
    clicks_per_coda_mean: float = field(init=False)
    clicks_per_coda_std: float = field(init=False)

    # Energy distribution
    energy_0_2khz: float = 0.0
    energy_2_8khz: float = 0.0
    energy_8_15khz: float = 0.0
    energy_above_15khz: float = 0.0

    # Modality analysis (from Universal Rosetta Stone)
    overall_modality: str = "UNKNOWN"
    coda_modalities: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate derived statistics."""
        self.total_clicks = len(self.clicks)
        self.clicks_per_second = self.total_clicks / self.duration_sec if self.duration_sec > 0 else 0

        self.total_codas = len(self.codas)
        if self.codas:
            coda_lengths = [c.num_clicks for c in self.codas]
            self.clicks_per_coda_mean = np.mean(coda_lengths)
            self.clicks_per_coda_std = np.std(coda_lengths)
        else:
            self.clicks_per_coda_mean = 0
            self.clicks_per_coda_std = 0


class SpermWhaleAnalyzer:
    """
    Specialized analyzer for sperm whale vocalizations.

    Combines:
    1. Click detection (envelope peak detection)
    2. Coda segmentation (grouping clicks into patterns)
    3. Modality classification (Universal Rosetta Stone)
    """

    # Sperm whale click parameters
    MIN_CLICK_INTERVAL_MS = 5.0      # Minimum time between clicks
    MAX_CLICK_INTERVAL_MS = 100.0    # Maximum time to consider clicks in same coda (adaptive)
    DEFAULT_CLICK_THRESHOLD = 2.0    # Standard deviations above mean envelope

    # Coda classification thresholds (based on prior research)
    SHORT_CODA_MAX_CLICKS = 10       # SHORT codas: <10 clicks
    LONG_CODA_MIN_CLICKS = 50        # LONG codas: >=50 clicks

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.rosetta_stone = UniversalRosettaStone(sample_rate=sample_rate)

    def analyze(self, audio: np.ndarray, filepath: str = "") -> SpermWhaleAnalysis:
        """
        Perform complete analysis of sperm whale audio.

        Args:
            audio: Audio samples (mono)
            filepath: Optional file path for reference

        Returns:
            SpermWhaleAnalysis with complete results
        """
        duration_sec = len(audio) / self.sample_rate

        # Step 1: Detect clicks
        clicks = self._detect_clicks(audio)

        # Step 2: Segment codas
        codas = self._segment_codas(clicks)

        # Step 3: Analyze energy distribution
        energy_dist = self._analyze_energy_distribution(audio)

        # Step 4: Modality classification
        overall_modality = self.rosetta_stone.detect_modality(audio).name
        coda_modalities = self._analyze_coda_modalities(audio, codas)

        # Create analysis object
        analysis = SpermWhaleAnalysis(
            filepath=filepath,
            duration_sec=duration_sec,
            sample_rate=self.sample_rate,
            clicks=clicks,
            codas=codas,
            energy_0_2khz=energy_dist["0-2_khz"],
            energy_2_8khz=energy_dist["2-8_khz"],
            energy_8_15khz=energy_dist["8-15_khz"],
            energy_above_15khz=energy_dist["above_15_khz"],
            overall_modality=overall_modality,
            coda_modalities=coda_modalities
        )

        return analysis

    def _detect_clicks(
        self,
        audio: np.ndarray,
        threshold_sd: float = DEFAULT_CLICK_THRESHOLD
    ) -> List[Click]:
        """
        Detect individual clicks using envelope peak detection.

        Args:
            audio: Audio samples
            threshold_sd: Detection threshold in SD above mean envelope

        Returns:
            List of Click objects
        """
        # Compute analytic signal (envelope)
        envelope = np.abs(hilbert(audio))

        # Set threshold
        threshold = np.mean(envelope) + threshold_sd * np.std(envelope)

        # Find peaks (clicks)
        min_distance = int(self.MIN_CLICK_INTERVAL_MS * self.sample_rate / 1000)
        peaks, properties = find_peaks(
            envelope,
            height=threshold,
            distance=min_distance,
            width=10  # Minimum peak width
        )

        # Create Click objects
        clicks = []
        for idx, peak_idx in enumerate(peaks):
            click = Click(
                position_samples=peak_idx,
                position_ms=peak_idx / self.sample_rate * 1000,
                amplitude=envelope[peak_idx],
                width_samples=int(properties['widths'][idx])
            )
            clicks.append(click)

        return clicks

    def _segment_codas(
        self,
        clicks: List[Click],
        max_gap_ms: float = MAX_CLICK_INTERVAL_MS
    ) -> List[Coda]:
        """
        Group clicks into codas based on inter-click intervals.

        Args:
            clicks: List of detected clicks
            max_gap_ms: Maximum gap between clicks to consider same coda

        Returns:
            List of Coda objects
        """
        if len(clicks) < 2:
            return []

        codas = []
        current_coda_clicks = [clicks[0]]

        for i in range(1, len(clicks)):
            gap_ms = clicks[i].position_ms - clicks[i-1].position_ms

            if gap_ms <= max_gap_ms:
                # Part of same coda
                current_coda_clicks.append(clicks[i])
            else:
                # Gap too large - finalize current coda and start new one
                if len(current_coda_clicks) >= 2:  # Minimum 2 clicks for coda
                    coda = self._create_coda(current_coda_clicks)
                    codas.append(coda)
                current_coda_clicks = [clicks[i]]

        # Don't forget the last coda
        if len(current_coda_clicks) >= 2:
            coda = self._create_coda(current_coda_clicks)
            codas.append(coda)

        return codas

    def _create_coda(self, clicks: List[Click]) -> Coda:
        """Create a Coda object from a list of clicks."""
        start_ms = clicks[0].position_ms
        end_ms = clicks[-1].position_ms
        duration_ms = end_ms - start_ms

        return Coda(
            clicks=clicks,
            start_ms=start_ms,
            end_ms=end_ms,
            duration_ms=duration_ms,
            num_clicks=len(clicks)
        )

    def _analyze_energy_distribution(self, audio: np.ndarray) -> Dict[str, float]:
        """Calculate energy distribution across frequency bands."""
        fft_result = fft(audio)
        freqs = fftfreq(len(audio), 1/self.sample_rate)
        magnitude = np.abs(fft_result)

        pos_freqs = freqs[:len(freqs)//2]
        pos_magnitude = magnitude[:len(magnitude)//2]

        bands = [
            ("0-2_khz", 0, 2000),
            ("2-8_khz", 2000, 8000),
            ("8-15_khz", 8000, 15000),
            ("above_15_khz", 15000, self.sample_rate//2)
        ]

        energy_dist = {}
        total_energy = np.sum(pos_magnitude**2)

        for band_name, low, high in bands:
            mask = (pos_freqs >= low) & (pos_freqs < high)
            band_energy = np.sum(pos_magnitude[mask]**2)
            energy_dist[band_name] = band_energy / total_energy * 100

        return energy_dist

    def _analyze_coda_modalities(
        self,
        audio: np.ndarray,
        codas: List[Coda]
    ) -> Dict[str, int]:
        """
        Analyze modality for each coda.

        Extracts audio segments corresponding to codas and classifies modality.
        """
        modality_counts = {}

        for coda in codas:
            # Extract coda audio segment
            start_sample = int(coda.start_ms * self.sample_rate / 1000)
            end_sample = int(coda.end_ms * self.sample_rate / 1000)

            # Add padding
            pad_samples = int(50 * self.sample_rate / 1000)  # 50ms padding
            start_sample = max(0, start_sample - pad_samples)
            end_sample = min(len(audio), end_sample + pad_samples)

            coda_audio = audio[start_sample:end_sample]

            # Classify modality
            modality = self.rosetta_stone.detect_modality(coda_audio)
            modality_counts[modality.name] = modality_counts.get(modality.name, 0) + 1

        return modality_counts

    def print_summary(self, analysis: SpermWhaleAnalysis):
        """Print a formatted summary of the analysis."""
        print("\n" + "="*70)
        print(f"SPERM WHALE ANALYSIS: {Path(analysis.filepath).name}")
        print("="*70)

        print("\n📊 Recording Info:")
        print(f"  Duration: {analysis.duration_sec:.1f}s")
        print(f"  Sample rate: {analysis.sample_rate} Hz")
        print(f"  Overall modality: {analysis.overall_modality}")

        print("\n📊 Click Detection:")
        print(f"  Total clicks: {analysis.total_clicks}")
        print(f"  Click rate: {analysis.clicks_per_second:.1f} clicks/second")

        print("\n📊 Coda Analysis:")
        print(f"  Total codas: {analysis.total_codas}")
        if analysis.total_codas > 0:
            print(f"  Clicks per coda: {analysis.clicks_per_coda_mean:.1f} ± {analysis.clicks_per_coda_std:.1f}")

            # Coda length distribution
            short_codas = [c for c in analysis.codas if c.num_clicks < self.SHORT_CODA_MAX_CLICKS]
            medium_codas = [c for c in analysis.codas if self.SHORT_CODA_MAX_CLICKS <= c.num_clicks < self.LONG_CODA_MIN_CLICKS]
            long_codas = [c for c in analysis.codas if c.num_clicks >= self.LONG_CODA_MIN_CLICKS]

            print(f"    SHORT codas (<10 clicks): {len(short_codas)}")
            print(f"    MEDIUM codas (10-49 clicks): {len(medium_codas)}")
            print(f"    LONG codas (50+ clicks): {len(long_codas)}")

            # Rhythm analysis
            rhythm_scores = [c.rhythm_regularity for c in analysis.codas]
            print(f"  Mean rhythm regularity: {np.mean(rhythm_scores):.3f}")

        print("\n📊 Energy Distribution:")
        print(f"  0-2 kHz:    {analysis.energy_0_2khz:5.1f}%")
        print(f"  2-8 kHz:    {analysis.energy_2_8khz:5.1f}%  ⭐ Sperm whale range")
        print(f"  8-15 kHz:   {analysis.energy_8_15khz:5.1f}%")
        print(f"  >15 kHz:    {analysis.energy_above_15khz:5.1f}%")

        if analysis.coda_modalities:
            print("\n📊 Coda Modality Distribution:")
            for modality, count in sorted(analysis.coda_modalities.items()):
                percentage = count / analysis.total_codas * 100
                print(f"  {modality:15s}: {count:2d} ({percentage:5.1f}%)")

        print("="*70)


def demo_analysis():
    """Demo analysis on a sperm whale file."""
    try:
        import soundfile as sf
    except ImportError:
        print("soundfile library required for demo")
        return

    base_dir = Path.home() / "birdsong_analysis" / "data" / "Dominica_dataset" / "Signal_parts"

    if not base_dir.exists():
        print(f"Data directory not found: {base_dir}")
        return

    # Find a file with codas (SW_19 had 15 phrases in our test)
    filepath = base_dir / "SW_19_filtered.wav"

    if not filepath.exists():
        print(f"File not found: {filepath}")
        return

    print(f"Loading: {filepath.name}")
    audio, sr = sf.read(filepath)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Analyze
    analyzer = SpermWhaleAnalyzer(sample_rate=sr)
    analysis = analyzer.analyze(audio, str(filepath))
    analyzer.print_summary(analysis)


if __name__ == "__main__":
    demo_analysis()
