"""
Musical key detection using the Krumhansl-Schmuckler algorithm.

Detects major and minor keys from chroma features by correlating
with psychoacoustically-derived pitch profiles.
"""

import numpy as np


# Pitch class names
PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Krumhansl-Schmuckler key profiles (normalized)
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
MAJOR_PROFILE = MAJOR_PROFILE / np.linalg.norm(MAJOR_PROFILE)
MINOR_PROFILE = MINOR_PROFILE / np.linalg.norm(MINOR_PROFILE)


def detect_key(chroma: np.ndarray, weight_early: bool = False) -> str:
    """
    Detect the musical key using the Krumhansl-Schmuckler algorithm.

    The algorithm correlates the chroma distribution with pre-defined
    major and minor key profiles to find the best matching key.

    Args:
        chroma: Chroma features array of shape (12, n_frames)
        weight_early: If True, weight the first 2/3 of the song more heavily.
                     Useful for songs with key modulation at the end.

    Returns:
        Key string like "Am", "C", "F#m", etc.
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


def detect_section_key(chroma: np.ndarray, start_frame: int, end_frame: int) -> str:
    """
    Detect the key for a specific section of the song.

    Args:
        chroma: Full chroma features array of shape (12, n_frames)
        start_frame: Starting frame index
        end_frame: Ending frame index

    Returns:
        Key string like "Am", "C", "F#m", etc.
    """
    section_chroma = chroma[:, start_frame:end_frame]
    return detect_key(section_chroma)


def compute_key_invariant_chroma(chroma: np.ndarray) -> np.ndarray:
    """
    Make chromagram key-invariant by rotating to align with detected key.

    This helps identify similar sections even if they're in different keys,
    which is useful for detecting repeated chorus sections with key modulations.

    Args:
        chroma: Chroma features array of shape (12, n_frames)

    Returns:
        Key-invariant chroma features of shape (12, n_frames)
    """
    # Detect key (find the pitch class with highest energy)
    chroma_sum = np.sum(chroma, axis=1)
    key_shift = np.argmax(chroma_sum)

    # Rotate chroma to make it key-invariant (transpose to C)
    return np.roll(chroma, -key_shift, axis=0)
