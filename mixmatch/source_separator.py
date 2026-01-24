"""
Audio source separation using Demucs.

Separates audio into stems (vocals, drums, bass, other) for improved
section detection and lyric transcription.
"""

import os
import tempfile
import shutil
from typing import Dict, Optional
from pathlib import Path

import numpy as np
import librosa
import torch


# Lazy load demucs to avoid heavy imports at module level
_model = None
_device = None


def _get_model():
    """Lazy load Demucs model (htdemucs)."""
    global _model, _device
    if _model is None:
        from demucs.pretrained import get_model
        from demucs.apply import apply_model

        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _model = get_model('htdemucs')
        _model.to(_device)
        _model.eval()
    return _model, _device


class StemCache:
    """Cache for separated stems to avoid re-processing."""

    def __init__(self):
        self._cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._sample_rates: Dict[str, int] = {}

    def get(self, audio_path: str) -> Optional[Dict[str, np.ndarray]]:
        """Get cached stems for audio file."""
        return self._cache.get(audio_path)

    def get_sr(self, audio_path: str) -> Optional[int]:
        """Get sample rate for cached audio."""
        return self._sample_rates.get(audio_path)

    def set(self, audio_path: str, stems: Dict[str, np.ndarray], sr: int):
        """Cache stems for audio file."""
        self._cache[audio_path] = stems
        self._sample_rates[audio_path] = sr

    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._sample_rates.clear()


# Global stem cache
_stem_cache = StemCache()


def separate_stems(
    audio_path: str,
    use_cache: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Separate audio into stems using Demucs htdemucs model.

    Returns vocals, drums, bass, and other as separate numpy arrays.
    Results are cached to avoid re-processing the same file.

    Args:
        audio_path: Path to audio file
        use_cache: Whether to use cached results

    Returns:
        Dictionary with keys: 'vocals', 'drums', 'bass', 'other'
        Each value is a numpy array (samples,) in mono at 22050 Hz
    """
    # Check cache first
    if use_cache:
        cached = _stem_cache.get(audio_path)
        if cached is not None:
            return cached

    from demucs.apply import apply_model

    model, device = _get_model()

    # Load audio with librosa (avoids torchaudio/TorchCodec issues)
    y, sr = librosa.load(audio_path, sr=model.samplerate, mono=False)

    # Convert to torch tensor
    # librosa returns (samples,) for mono or (channels, samples) for stereo
    if y.ndim == 1:
        # Mono: duplicate to stereo
        wav = torch.from_numpy(y).unsqueeze(0).repeat(2, 1)
    else:
        # Stereo or multi-channel
        wav = torch.from_numpy(y)
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2]

    # Add batch dimension and move to device
    wav = wav.unsqueeze(0).to(device)

    # Apply model
    with torch.no_grad():
        sources = apply_model(model, wav, device=device)

    # sources shape: (batch, num_sources, channels, samples)
    # htdemucs sources order: drums, bass, other, vocals
    source_names = model.sources  # ['drums', 'bass', 'other', 'vocals']

    stems = {}
    for i, name in enumerate(source_names):
        # Get source, convert to mono, move to CPU
        source = sources[0, i].mean(dim=0).cpu().numpy()

        # Resample to 22050 Hz for consistency
        if sr != 22050:
            source = librosa.resample(source, orig_sr=sr, target_sr=22050)

        stems[name] = source

    # Cache the results
    if use_cache:
        _stem_cache.set(audio_path, stems, 22050)

    return stems


def get_vocals(audio_path: str) -> np.ndarray:
    """
    Get isolated vocals from audio file.

    Args:
        audio_path: Path to audio file

    Returns:
        Numpy array of vocal audio (mono, 22050 Hz)
    """
    stems = separate_stems(audio_path)
    return stems.get('vocals', np.array([]))


def get_drums(audio_path: str) -> np.ndarray:
    """
    Get isolated drums from audio file.

    Args:
        audio_path: Path to audio file

    Returns:
        Numpy array of drum audio (mono, 22050 Hz)
    """
    stems = separate_stems(audio_path)
    return stems.get('drums', np.array([]))


def get_accompaniment(audio_path: str) -> np.ndarray:
    """
    Get accompaniment (everything except vocals) from audio file.

    Args:
        audio_path: Path to audio file

    Returns:
        Numpy array of accompaniment audio (mono, 22050 Hz)
    """
    stems = separate_stems(audio_path)

    # Combine drums, bass, and other
    drums = stems.get('drums', np.array([]))
    bass = stems.get('bass', np.array([]))
    other = stems.get('other', np.array([]))

    # Find max length and pad shorter arrays
    max_len = max(len(drums), len(bass), len(other))
    if max_len == 0:
        return np.array([])

    def pad_to_length(arr, length):
        if len(arr) == 0:
            return np.zeros(length)
        if len(arr) < length:
            return np.pad(arr, (0, length - len(arr)))
        return arr[:length]

    drums = pad_to_length(drums, max_len)
    bass = pad_to_length(bass, max_len)
    other = pad_to_length(other, max_len)

    return drums + bass + other


def analyze_drum_energy(
    audio_path: str,
    hop_length: int = 512,
    sr: int = 22050,
) -> np.ndarray:
    """
    Analyze drum energy over time for breakdown/bridge detection.

    Drops in drum energy often indicate:
    - Bridges/breakdowns (drum drops out)
    - Pre-choruses (drums may thin out before building)
    - Intros/outros

    Args:
        audio_path: Path to audio file
        hop_length: Hop length for RMS computation
        sr: Sample rate (should match stem extraction)

    Returns:
        Numpy array of drum RMS energy over time
    """
    drums = get_drums(audio_path)

    if len(drums) == 0:
        return np.array([])

    # Compute RMS energy
    rms = librosa.feature.rms(y=drums, hop_length=hop_length)[0]

    return rms


def detect_drum_drops(
    audio_path: str,
    threshold_percentile: float = 25.0,
    min_duration: float = 2.0,
    hop_length: int = 512,
    sr: int = 22050,
) -> list:
    """
    Detect sections where drums drop out significantly.

    Args:
        audio_path: Path to audio file
        threshold_percentile: Percentile below which drums are considered "dropped"
        min_duration: Minimum duration in seconds for a drum drop
        hop_length: Hop length for analysis
        sr: Sample rate

    Returns:
        List of (start_time, end_time) tuples for drum drop sections
    """
    drum_energy = analyze_drum_energy(audio_path, hop_length, sr)

    if len(drum_energy) == 0:
        return []

    # Find threshold for "low drums"
    threshold = np.percentile(drum_energy, threshold_percentile)

    # Find contiguous regions below threshold
    below_threshold = drum_energy < threshold

    # Convert to time
    frame_times = librosa.frames_to_time(
        np.arange(len(drum_energy)),
        sr=sr,
        hop_length=hop_length
    )

    # Find start and end of each low-drum region
    drops = []
    in_drop = False
    drop_start = 0.0

    for i, is_low in enumerate(below_threshold):
        time = frame_times[i]

        if is_low and not in_drop:
            # Start of drop
            in_drop = True
            drop_start = time
        elif not is_low and in_drop:
            # End of drop
            in_drop = False
            drop_duration = time - drop_start
            if drop_duration >= min_duration:
                drops.append((drop_start, time))

    # Handle case where song ends during a drop
    if in_drop:
        drop_duration = frame_times[-1] - drop_start
        if drop_duration >= min_duration:
            drops.append((drop_start, frame_times[-1]))

    return drops


def clear_cache():
    """Clear the stem cache."""
    _stem_cache.clear()
