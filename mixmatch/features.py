"""
Audio feature extraction for section detection.

Includes beat-synchronized features, HPSS, tempogram analysis,
MFCC features, and novelty curve computation.
"""

from typing import Tuple
import numpy as np
import librosa
from scipy.ndimage import median_filter


# Feature weights for novelty curve (tuned based on research)
NOVELTY_WEIGHTS = {
    'spectral': 0.25,
    'chroma': 0.35,
    'onset': 0.15,
    'mfcc': 0.25,
}


def compute_beat_synced_features(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute beat-synchronized RMS energy for cleaner section detection.

    Beat synchronization smooths out high-frequency variations and emphasizes
    patterns tied to musical structure, making chorus sections more distinguishable.

    Args:
        y: Audio signal
        sr: Sample rate
        hop_length: Hop length for analysis

    Returns:
        Tuple of (beat_synced_rms, beat_frames, beat_times)
    """
    # Get beat frames
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

    # Compute RMS
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    # Sync to beats (average RMS between beats)
    if len(beat_frames) > 1:
        beat_synced_rms = librosa.util.sync(rms.reshape(1, -1), beat_frames, aggregate=np.mean)[0]
    else:
        beat_synced_rms = rms

    return beat_synced_rms, beat_frames, beat_times


def extract_hpss_features(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separate harmonic and percussive components using HPSS.

    Useful because choruses often have distinct harmonic/percussive characteristics.
    Harmonic content captures melodic/tonal elements, percussive captures rhythmic elements.

    Args:
        y: Audio signal
        sr: Sample rate
        hop_length: Hop length for analysis

    Returns:
        Tuple of (rms_harmonic, rms_percussive)
    """
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # Compute RMS for each component
    rms_harmonic = librosa.feature.rms(y=y_harmonic, hop_length=hop_length)[0]
    rms_percussive = librosa.feature.rms(y=y_percussive, hop_length=hop_length)[0]

    return rms_harmonic, rms_percussive


def compute_tempogram_features(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute tempogram for rhythmic pattern analysis.

    Choruses often have more stable/consistent rhythmic patterns compared to
    verses or bridges. The tempogram captures rhythmic/tempo information over time.

    Args:
        y: Audio signal
        sr: Sample rate
        hop_length: Hop length for analysis

    Returns:
        Tuple of (onset_envelope, tempo_stability)
    """
    # Compute onset envelope (used for tempogram and as a feature)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # Compute tempogram
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length)

    # Get tempo stability (variance along tempo axis - lower = more stable)
    tempo_stability = np.std(tempogram, axis=0)

    return onset_env, tempo_stability


def compute_mfcc_flux(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    n_mfcc: int = 13,
) -> np.ndarray:
    """
    Compute MFCC flux (rate of change) for timbral boundary detection.

    MFCCs capture timbral characteristics that differ between sections.
    High MFCC flux indicates timbral changes, often at section boundaries.

    Args:
        y: Audio signal
        sr: Sample rate
        hop_length: Hop length for analysis
        n_mfcc: Number of MFCC coefficients

    Returns:
        MFCC flux array
    """
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)

    # Compute delta (rate of change) for boundary detection
    mfcc_flux = np.sqrt(np.sum(np.diff(mfccs, axis=1) ** 2, axis=0))
    mfcc_flux = np.concatenate([[0], mfcc_flux])

    return mfcc_flux


def compute_novelty_curve(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a novelty curve that peaks at section boundaries.

    Enhanced with multi-feature approach combining:
    - Spectral flux (frequency changes)
    - Chroma flux (harmonic/tonal changes)
    - Onset envelope (rhythmic changes)
    - MFCC flux (timbral changes)

    Args:
        y: Audio signal
        sr: Sample rate
        hop_length: Hop length for analysis

    Returns:
        Tuple of (novelty_curve, frame_times)
    """
    # Compute mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)

    # Compute spectral flux (frame-to-frame change)
    spectral_flux = np.sqrt(np.sum(np.diff(S_db, axis=1) ** 2, axis=0))
    spectral_flux = np.concatenate([[0], spectral_flux])

    # Compute chroma and chroma flux
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    chroma_flux = np.sqrt(np.sum(np.diff(chroma, axis=1) ** 2, axis=0))
    chroma_flux = np.concatenate([[0], chroma_flux])

    # Compute onset envelope (rhythmic changes)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # Compute MFCC flux (timbral changes)
    mfcc_flux = compute_mfcc_flux(y, sr, hop_length)

    # Normalize all features to [0, 1]
    spectral_flux = spectral_flux / (np.max(spectral_flux) + 1e-10)
    chroma_flux = chroma_flux / (np.max(chroma_flux) + 1e-10)
    onset_env = onset_env / (np.max(onset_env) + 1e-10)
    mfcc_flux = mfcc_flux / (np.max(mfcc_flux) + 1e-10)

    # Combine with research-based weights
    # Harmonic (chroma) and timbral (mfcc) changes are most important for section boundaries
    novelty = (
        NOVELTY_WEIGHTS['spectral'] * spectral_flux +
        NOVELTY_WEIGHTS['chroma'] * chroma_flux +
        NOVELTY_WEIGHTS['onset'] * onset_env +
        NOVELTY_WEIGHTS['mfcc'] * mfcc_flux
    )

    # Smooth with median filter
    novelty = median_filter(novelty, size=15)

    frame_times = librosa.frames_to_time(np.arange(len(novelty)), sr=sr, hop_length=hop_length)

    return novelty, frame_times


def compute_energy_per_section(
    y: np.ndarray,
    sr: int,
    boundaries: list,
    duration: float,
    hop_length: int = 512,
) -> list:
    """
    Compute average RMS energy for each section.

    Args:
        y: Audio signal
        sr: Sample rate
        boundaries: List of section boundary times
        duration: Total audio duration
        hop_length: Hop length for analysis

    Returns:
        List of average energy values per section
    """
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    energy_per_section = []

    for i in range(len(boundaries)):
        start_time = boundaries[i]
        end_time = boundaries[i + 1] if i + 1 < len(boundaries) else duration

        # Find frames in this section
        mask = (rms_times >= start_time) & (rms_times < end_time)

        # Total energy
        section_rms = rms[mask]
        section_energy = float(np.mean(section_rms)) if len(section_rms) > 0 else 0.0
        energy_per_section.append(section_energy)

    return energy_per_section


def compute_harmonic_energy_per_section(
    rms_harmonic: np.ndarray,
    sr: int,
    boundaries: list,
    duration: float,
    hop_length: int = 512,
) -> list:
    """
    Compute average harmonic energy for each section.

    Args:
        rms_harmonic: RMS of harmonic component
        sr: Sample rate
        boundaries: List of section boundary times
        duration: Total audio duration
        hop_length: Hop length for analysis

    Returns:
        List of average harmonic energy values per section
    """
    rms_times = librosa.frames_to_time(np.arange(len(rms_harmonic)), sr=sr, hop_length=hop_length)

    harmonic_per_section = []

    for i in range(len(boundaries)):
        start_time = boundaries[i]
        end_time = boundaries[i + 1] if i + 1 < len(boundaries) else duration

        mask = (rms_times >= start_time) & (rms_times < end_time)
        section_harmonic = rms_harmonic[mask]
        harmonic_energy = float(np.mean(section_harmonic)) if len(section_harmonic) > 0 else 0.0
        harmonic_per_section.append(harmonic_energy)

    return harmonic_per_section


def compute_stability_per_section(
    tempo_stability: np.ndarray,
    sr: int,
    boundaries: list,
    duration: float,
    hop_length: int = 512,
) -> list:
    """
    Compute average tempo stability for each section.

    Args:
        tempo_stability: Tempo stability array
        sr: Sample rate
        boundaries: List of section boundary times
        duration: Total audio duration
        hop_length: Hop length for analysis

    Returns:
        List of average stability values per section
    """
    times = librosa.frames_to_time(np.arange(len(tempo_stability)), sr=sr, hop_length=hop_length)

    stability_per_section = []

    for i in range(len(boundaries)):
        start_time = boundaries[i]
        end_time = boundaries[i + 1] if i + 1 < len(boundaries) else duration

        mask = (times >= start_time) & (times < end_time)
        section_stability = tempo_stability[mask]
        stability = float(np.mean(section_stability)) if len(section_stability) > 0 else 0.0
        stability_per_section.append(stability)

    return stability_per_section
