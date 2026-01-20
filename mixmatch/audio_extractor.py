"""
Local audio file extraction

Extracts BPM, sections (intro/verse/chorus/outro/bridge), and keys from audio files.
Uses self-similarity matrices and novelty detection for structural analysis.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import librosa
from scipy.ndimage import median_filter
from scipy.signal import find_peaks


# Pitch class names for key detection
PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Minimum section duration in seconds (sections shorter than this get merged)
MIN_SECTION_DURATION = 25.0

# Krumhansl-Schmuckler key profiles (defined once for efficiency)
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
MAJOR_PROFILE = MAJOR_PROFILE / np.linalg.norm(MAJOR_PROFILE)
MINOR_PROFILE = MINOR_PROFILE / np.linalg.norm(MINOR_PROFILE)


def detect_key(chroma: np.ndarray, weight_early: bool = False) -> str:
    """
    Detect the key using the Krumhansl-Schmuckler algorithm.

    Args:
        chroma: Chroma features (12, n_frames)
        weight_early: If True, weight the first 2/3 of the song more heavily
                     (useful for songs with key modulation at the end)

    Returns:
        Key string like "Am" or "C"
    """
    if chroma.shape[1] == 0:
        return "C"

    # Get average chroma across the song
    if weight_early and chroma.shape[1] > 100:
        # Weight the first 2/3 of the song more heavily to handle late modulations
        n_frames = chroma.shape[1]
        cutoff = int(n_frames * 2 / 3)

        # Weight: 70% early, 30% late
        early_avg = np.mean(chroma[:, :cutoff], axis=1)
        late_avg = np.mean(chroma[:, cutoff:], axis=1)
        chroma_avg = 0.7 * early_avg + 0.3 * late_avg
    else:
        chroma_avg = np.mean(chroma, axis=1)

    chroma_avg = chroma_avg / (np.linalg.norm(chroma_avg) + 1e-10)

    best_key = "C"
    best_corr = -1

    # Try all 12 major and minor keys
    for shift in range(12):
        # Rotate the chroma to test each key
        rotated = np.roll(chroma_avg, -shift)

        # Correlation with major profile
        corr_major = np.dot(rotated, MAJOR_PROFILE)
        if corr_major > best_corr:
            best_corr = corr_major
            best_key = PITCH_CLASSES[shift]

        # Correlation with minor profile
        corr_minor = np.dot(rotated, MINOR_PROFILE)
        if corr_minor > best_corr:
            best_corr = corr_minor
            best_key = f"{PITCH_CLASSES[shift]}m"

    return best_key


def compute_novelty_curve(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a novelty curve that peaks at section boundaries.
    Combines spectral and harmonic novelty.

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

    # Normalize both
    spectral_flux = spectral_flux / (np.max(spectral_flux) + 1e-10)
    chroma_flux = chroma_flux / (np.max(chroma_flux) + 1e-10)

    # Combine (harmonic changes are more important for section boundaries)
    novelty = 0.4 * spectral_flux + 0.6 * chroma_flux

    # Smooth with median filter
    novelty = median_filter(novelty, size=15)

    frame_times = librosa.frames_to_time(np.arange(len(novelty)), sr=sr, hop_length=hop_length)

    return novelty, frame_times


def detect_section_boundaries(
    y: np.ndarray,
    sr: int,
    min_section_duration: float = MIN_SECTION_DURATION,
) -> List[float]:
    """
    Detect section boundaries using novelty peaks.

    Args:
        y: Audio signal
        sr: Sample rate
        min_section_duration: Minimum time between boundaries in seconds

    Returns:
        List of boundary times in seconds
    """
    hop_length = 512
    novelty, frame_times = compute_novelty_curve(y, sr, hop_length)

    # Convert min duration to frames
    min_distance = int(min_section_duration * sr / hop_length)

    # Find peaks in novelty curve
    # Use a relatively high prominence to avoid false positives
    peaks, properties = find_peaks(
        novelty,
        distance=min_distance,
        prominence=0.1,
        height=np.percentile(novelty, 60),
    )

    # Convert to times
    boundary_times = [0.0]  # Always start at 0
    for peak in peaks:
        boundary_times.append(float(frame_times[peak]))

    return boundary_times


def build_recurrence_matrix(
    chroma: np.ndarray,
    hop_length: int,
    sr: int,
) -> np.ndarray:
    """
    Build a self-similarity matrix from chroma features.

    Args:
        chroma: Chroma features (12, n_frames)
        hop_length: Hop length used for chroma
        sr: Sample rate

    Returns:
        Recurrence matrix with affinity values
    """
    # Stack memory for temporal context
    chroma_stacked = librosa.feature.stack_memory(chroma, n_steps=4, delay=2)

    # Build recurrence matrix
    rec = librosa.segment.recurrence_matrix(
        chroma_stacked,
        k=None,  # Use all neighbors
        width=int(5 * sr / hop_length),  # 5 second minimum separation
        metric='cosine',
        mode='affinity',
        self=True,
    )

    return rec


def compute_section_similarity(
    rec_matrix: np.ndarray,
    boundaries: List[float],
    duration: float,
) -> np.ndarray:
    """
    Compute similarity matrix between sections.

    Args:
        rec_matrix: Self-similarity matrix
        boundaries: Section boundary times
        duration: Total duration

    Returns:
        Section-to-section similarity matrix
    """
    n_sections = len(boundaries)
    n_frames = rec_matrix.shape[0]

    # Convert times to frame indices
    boundary_frames = [int(t * n_frames / duration) for t in boundaries]
    boundary_frames.append(n_frames)

    similarity = np.zeros((n_sections, n_sections))

    for i in range(n_sections):
        for j in range(n_sections):
            if i == j:
                similarity[i, j] = 1.0
            else:
                # Get the block of the recurrence matrix for these two sections
                block = rec_matrix[
                    boundary_frames[i]:boundary_frames[i+1],
                    boundary_frames[j]:boundary_frames[j+1]
                ]
                similarity[i, j] = float(np.mean(block)) if block.size > 0 else 0.0

    return similarity


def label_sections(
    boundaries: List[float],
    section_similarity: np.ndarray,
    energy_per_section: List[float],
    duration: float,
) -> List[str]:
    """
    Label sections as intro/verse/chorus/bridge/outro.

    Strategy:
    - Intro: First section if short relative to song
    - Outro: Last section if starts late in song
    - Chorus: High energy sections that repeat (similar to other sections)
    - Bridge: Lower energy, appears once, typically after middle of song
    - Verse: Everything else

    Args:
        boundaries: Section boundary times
        section_similarity: Section-to-section similarity matrix
        energy_per_section: Average energy per section
        duration: Total duration

    Returns:
        List of section labels
    """
    n_sections = len(boundaries)
    labels = ['verse'] * n_sections

    if n_sections == 0:
        return labels

    # Normalize energy
    max_energy = max(energy_per_section) if energy_per_section else 1
    min_energy = min(energy_per_section) if energy_per_section else 0
    energy_range = max_energy - min_energy + 1e-10
    normalized_energy = [(e - min_energy) / energy_range for e in energy_per_section]

    # Compute repetition score for each section (how similar to other sections)
    repetition_scores = []
    for i in range(n_sections):
        # Average similarity to all other sections
        other_sims = [section_similarity[i, j] for j in range(n_sections) if j != i]
        rep_score = np.mean(other_sims) if other_sims else 0.0
        repetition_scores.append(rep_score)

    # First section is intro if it ends before the first chorus-like section
    # Typically intro is before 10% of the song or before ~20 seconds
    if n_sections > 1:
        first_section_end = boundaries[1] if len(boundaries) > 1 else duration
        if first_section_end < max(duration * 0.10, 25.0):
            labels[0] = 'intro'

    # Last section is outro if it starts after 85% of song
    if n_sections > 1 and boundaries[-1] > duration * 0.85:
        labels[-1] = 'outro'

    # Find chorus candidates: high energy + high repetition
    # Chorus typically appears multiple times and has high energy
    chorus_threshold_energy = 0.5  # Above median energy
    chorus_threshold_rep = np.percentile(repetition_scores, 60) if repetition_scores else 0.5

    chorus_indices = []
    for i in range(n_sections):
        if labels[i] not in ['intro', 'outro']:
            # Must be after first 15% of song
            if boundaries[i] >= duration * 0.15:
                if normalized_energy[i] >= chorus_threshold_energy and repetition_scores[i] >= chorus_threshold_rep:
                    chorus_indices.append(i)

    # If we found chorus candidates, mark them
    for idx in chorus_indices:
        labels[idx] = 'chorus'

    # Find bridge: lower energy section that appears once, typically in second half
    # Bridge is often between 50-75% of the song, before the final chorus
    bridge_candidates = []
    for i in range(n_sections):
        if labels[i] == 'verse':
            section_position = boundaries[i] / duration
            # Must be in middle-to-late portion of song (50-80%)
            if 0.50 <= section_position <= 0.80:
                # Lower repetition than choruses (more unique)
                if repetition_scores[i] < np.percentile(repetition_scores, 50):
                    # Check if it's between high-energy sections (choruses)
                    prev_is_chorus = i > 0 and labels[i-1] == 'chorus'
                    next_is_chorus = i < n_sections - 1 and labels[i+1] == 'chorus'

                    # Bridge often follows a chorus and precedes another chorus
                    if prev_is_chorus or next_is_chorus:
                        bridge_candidates.append((i, normalized_energy[i]))

    # Pick the lowest energy candidate as bridge
    if bridge_candidates:
        bridge_idx = min(bridge_candidates, key=lambda x: x[1])[0]
        labels[bridge_idx] = 'bridge'

    return labels


def detect_section_key(chroma: np.ndarray, start_frame: int, end_frame: int) -> str:
    """
    Detect the key for a specific section of the song.

    Args:
        chroma: Full chroma features (12, n_frames)
        start_frame: Starting frame index
        end_frame: Ending frame index

    Returns:
        Key string like "Am" or "C"
    """
    section_chroma = chroma[:, start_frame:end_frame]
    return detect_key(section_chroma)


def extract(audio_path: str) -> Dict:
    """
    Extract BPM, sections, and keys from audio file.

    Args:
        audio_path: Path to audio file

    Returns:
        Dictionary with structure:
        {
            "bpm": int,
            "key": str,
            "sections": [
                {"label": str, "start": float, "key": str},
                ...
            ]
        }
    """
    # Step 1: Load audio
    try:
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
    except Exception as e:
        raise ValueError(f"Failed to load audio: {e}")

    duration = librosa.get_duration(y=y, sr=sr)
    hop_length = 512

    # Step 2: Detect BPM
    try:
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        bpm = int(round(float(tempo)))
    except Exception as e:
        raise ValueError(f"Failed to detect BPM: {e}")

    # Step 3: Extract chroma features
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    except Exception as e:
        raise ValueError(f"Failed to extract chroma: {e}")

    n_chroma_frames = chroma.shape[1]

    # Step 4: Detect overall song key using Krumhansl-Schmuckler
    song_key = detect_key(chroma)

    # Step 5: Detect section boundaries using novelty curve
    boundaries = detect_section_boundaries(y, sr, min_section_duration=MIN_SECTION_DURATION)

    # If we got very few boundaries, try with shorter minimum
    if len(boundaries) < 4 and duration > 120:
        boundaries = detect_section_boundaries(y, sr, min_section_duration=10.0)

    # Step 6: Build self-similarity matrix
    rec_matrix = build_recurrence_matrix(chroma, hop_length, sr)

    # Step 7: Compute section similarity
    section_similarity = compute_section_similarity(rec_matrix, boundaries, duration)

    # Step 8: Compute energy per section
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    energy_per_section = []
    for i in range(len(boundaries)):
        start_time = boundaries[i]
        end_time = boundaries[i + 1] if i + 1 < len(boundaries) else duration

        # Find RMS frames in this section
        mask = (rms_times >= start_time) & (rms_times < end_time)
        section_rms = rms[mask]
        section_energy = float(np.mean(section_rms)) if len(section_rms) > 0 else 0.0
        energy_per_section.append(section_energy)

    # Step 9: Label sections
    labels = label_sections(boundaries, section_similarity, energy_per_section, duration)

    # Step 10: Build output sections with per-section key detection
    sections = []
    for i, boundary_time in enumerate(boundaries):
        # Convert time to chroma frame
        start_frame = int(boundary_time * n_chroma_frames / duration)
        if i + 1 < len(boundaries):
            end_frame = int(boundaries[i + 1] * n_chroma_frames / duration)
        else:
            end_frame = n_chroma_frames

        # Detect key for this section
        section_key = detect_section_key(chroma, start_frame, end_frame)

        sections.append({
            "label": labels[i],
            "start": round(boundary_time, 1),
            "key": section_key,
        })

    return {
        "bpm": bpm,
        "key": song_key,
        "sections": sections,
    }
