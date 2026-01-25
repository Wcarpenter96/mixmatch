"""
Melody analysis using Spotify's Basic Pitch.

Extracts pitch/melody information from vocals and analyzes melodic patterns
to help identify repeated sections (choruses) based on melody similarity.
"""

import os
import tempfile
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import numpy as np
import soundfile as sf

# Lazy load basic_pitch to avoid heavy imports at module level
_basic_pitch_predict = None
_basic_pitch_model_path = None


@dataclass
class NoteEvent:
    """A single note event with timing and pitch information."""
    start: float  # Start time in seconds
    end: float  # End time in seconds
    pitch: int  # MIDI pitch (0-127)
    amplitude: float  # Note amplitude/confidence


@dataclass
class MelodyContour:
    """Melodic contour representation for a section."""
    pitches: np.ndarray  # Sequence of MIDI pitches
    times: np.ndarray  # Corresponding timestamps
    notes: List[NoteEvent]  # Raw note events


def _get_basic_pitch():
    """Lazy load Basic Pitch predict function and model path."""
    global _basic_pitch_predict, _basic_pitch_model_path
    if _basic_pitch_predict is None:
        from basic_pitch.inference import predict
        from basic_pitch import ICASSP_2022_MODEL_PATH
        _basic_pitch_predict = predict
        _basic_pitch_model_path = ICASSP_2022_MODEL_PATH
    return _basic_pitch_predict, _basic_pitch_model_path


def extract_melody_from_vocals(
    vocals: np.ndarray,
    sr: int = 22050,
) -> List[NoteEvent]:
    """
    Extract melody notes from isolated vocals using Basic Pitch.

    Args:
        vocals: Isolated vocals audio (mono, numpy array)
        sr: Sample rate of the audio

    Returns:
        List of NoteEvent objects representing the detected melody
    """
    predict, model_path = _get_basic_pitch()

    # Basic Pitch expects 22050 Hz audio
    if sr != 22050:
        import librosa
        vocals = librosa.resample(vocals, orig_sr=sr, target_sr=22050)
        sr = 22050

    # Write to temp file for Basic Pitch
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
        sf.write(temp_path, vocals, sr)

    try:
        # Run Basic Pitch inference with model path
        model_output, midi_data, note_events = predict(temp_path, model_path)

        # Convert to our NoteEvent format
        notes = []
        for note in note_events:
            notes.append(NoteEvent(
                start=note[0],  # start_time
                end=note[1],  # end_time
                pitch=int(note[2]),  # pitch (MIDI number)
                amplitude=note[3] if len(note) > 3 else 1.0,  # amplitude
            ))

        return notes
    finally:
        os.unlink(temp_path)


def notes_to_pitch_contour(
    notes: List[NoteEvent],
    hop_time: float = 0.1,
    duration: Optional[float] = None,
) -> MelodyContour:
    """
    Convert note events to a pitch contour (time series of pitches).

    Args:
        notes: List of NoteEvent objects
        hop_time: Time resolution for the contour in seconds
        duration: Total duration (if None, derived from notes)

    Returns:
        MelodyContour with pitch sequence
    """
    if not notes:
        return MelodyContour(
            pitches=np.array([]),
            times=np.array([]),
            notes=[],
        )

    if duration is None:
        duration = max(n.end for n in notes)

    # Create time grid
    times = np.arange(0, duration, hop_time)
    pitches = np.zeros(len(times))

    # Fill in pitches (use highest note if multiple overlap)
    for note in notes:
        start_idx = int(note.start / hop_time)
        end_idx = int(note.end / hop_time)
        for i in range(start_idx, min(end_idx + 1, len(pitches))):
            if note.pitch > pitches[i]:
                pitches[i] = note.pitch

    return MelodyContour(
        pitches=pitches,
        times=times,
        notes=notes,
    )


def extract_section_contour(
    contour: MelodyContour,
    start_time: float,
    end_time: float,
) -> MelodyContour:
    """
    Extract a portion of the melody contour for a specific section.

    Args:
        contour: Full song melody contour
        start_time: Section start time in seconds
        end_time: Section end time in seconds

    Returns:
        MelodyContour for just that section
    """
    mask = (contour.times >= start_time) & (contour.times < end_time)
    section_pitches = contour.pitches[mask]
    section_times = contour.times[mask] - start_time  # Normalize to section start

    section_notes = [
        NoteEvent(
            start=max(0, n.start - start_time),
            end=min(end_time - start_time, n.end - start_time),
            pitch=n.pitch,
            amplitude=n.amplitude,
        )
        for n in contour.notes
        if n.end > start_time and n.start < end_time
    ]

    return MelodyContour(
        pitches=section_pitches,
        times=section_times,
        notes=section_notes,
    )


def compute_pitch_histogram(contour: MelodyContour) -> np.ndarray:
    """
    Compute a normalized pitch class histogram from a melody contour.

    This is useful for comparing melodic content in a key-invariant way.

    Args:
        contour: MelodyContour to analyze

    Returns:
        12-element array representing pitch class distribution
    """
    histogram = np.zeros(12)

    for pitch in contour.pitches:
        if pitch > 0:  # Skip silence (pitch=0)
            pitch_class = int(pitch) % 12
            histogram[pitch_class] += 1

    # Normalize
    if histogram.sum() > 0:
        histogram = histogram / histogram.sum()

    return histogram


def compute_interval_sequence(contour: MelodyContour) -> np.ndarray:
    """
    Compute the sequence of melodic intervals (pitch differences).

    Intervals are more useful than absolute pitches for matching melodies
    in different keys.

    Args:
        contour: MelodyContour to analyze

    Returns:
        Array of intervals (semitones between consecutive pitches)
    """
    # Get non-zero pitches
    nonzero_pitches = contour.pitches[contour.pitches > 0]

    if len(nonzero_pitches) < 2:
        return np.array([])

    # Compute intervals
    intervals = np.diff(nonzero_pitches)

    return intervals


def compute_melody_similarity(
    contour1: MelodyContour,
    contour2: MelodyContour,
) -> float:
    """
    Compute similarity between two melody contours.

    Uses a combination of:
    - Pitch histogram correlation (captures overall pitch distribution)
    - Interval sequence correlation (captures melodic shape)

    Args:
        contour1: First melody contour
        contour2: Second melody contour

    Returns:
        Similarity score between 0 and 1
    """
    # Handle empty contours
    if len(contour1.pitches) == 0 or len(contour2.pitches) == 0:
        return 0.0

    # Pitch histogram similarity
    hist1 = compute_pitch_histogram(contour1)
    hist2 = compute_pitch_histogram(contour2)

    if hist1.sum() > 0 and hist2.sum() > 0:
        # Cosine similarity of histograms
        hist_similarity = np.dot(hist1, hist2) / (np.linalg.norm(hist1) * np.linalg.norm(hist2))
    else:
        hist_similarity = 0.0

    # Interval sequence similarity (more robust to transposition)
    intervals1 = compute_interval_sequence(contour1)
    intervals2 = compute_interval_sequence(contour2)

    if len(intervals1) > 5 and len(intervals2) > 5:
        # Use cross-correlation to find best alignment
        interval_similarity = _interval_correlation(intervals1, intervals2)
    else:
        interval_similarity = 0.0

    # Combine similarities
    return 0.4 * hist_similarity + 0.6 * interval_similarity


def _interval_correlation(intervals1: np.ndarray, intervals2: np.ndarray) -> float:
    """
    Compute correlation between two interval sequences.

    Uses sliding window cross-correlation to handle different lengths.
    """
    if len(intervals1) == 0 or len(intervals2) == 0:
        return 0.0

    # Normalize intervals
    intervals1 = intervals1 - np.mean(intervals1)
    intervals2 = intervals2 - np.mean(intervals2)

    std1 = np.std(intervals1)
    std2 = np.std(intervals2)

    if std1 == 0 or std2 == 0:
        return 0.0

    intervals1 = intervals1 / std1
    intervals2 = intervals2 / std2

    # Cross-correlation
    if len(intervals1) < len(intervals2):
        shorter, longer = intervals1, intervals2
    else:
        shorter, longer = intervals2, intervals1

    max_corr = 0.0
    for offset in range(len(longer) - len(shorter) + 1):
        segment = longer[offset:offset + len(shorter)]
        corr = np.corrcoef(shorter, segment)[0, 1]
        if not np.isnan(corr):
            max_corr = max(max_corr, abs(corr))

    return max_corr


def compute_section_melody_similarity(
    vocals: np.ndarray,
    boundaries: List[float],
    duration: float,
    sr: int = 22050,
) -> np.ndarray:
    """
    Compute pairwise melody similarity matrix between sections.

    Args:
        vocals: Isolated vocals audio
        boundaries: Section boundary times
        duration: Total audio duration
        sr: Sample rate

    Returns:
        n_sections x n_sections similarity matrix
    """
    n_sections = len(boundaries)
    similarity = np.zeros((n_sections, n_sections))

    if n_sections == 0:
        return similarity

    # Extract melody from full vocals
    try:
        notes = extract_melody_from_vocals(vocals, sr)
        full_contour = notes_to_pitch_contour(notes, hop_time=0.1, duration=duration)
    except Exception as e:
        print(f"Warning: Melody extraction failed: {e}")
        return similarity

    # Extract contours for each section
    section_contours = []
    for i, start_time in enumerate(boundaries):
        end_time = boundaries[i + 1] if i + 1 < len(boundaries) else duration
        section_contour = extract_section_contour(full_contour, start_time, end_time)
        section_contours.append(section_contour)

    # Compute pairwise similarities
    for i in range(n_sections):
        for j in range(n_sections):
            if i == j:
                similarity[i, j] = 1.0
            else:
                similarity[i, j] = compute_melody_similarity(
                    section_contours[i],
                    section_contours[j],
                )

    return similarity


def find_melodic_phrases(
    contour: MelodyContour,
    min_phrase_duration: float = 2.0,
    max_phrase_duration: float = 10.0,
) -> List[Tuple[float, float, np.ndarray]]:
    """
    Find candidate melodic phrases in a contour.

    Args:
        contour: MelodyContour to analyze
        min_phrase_duration: Minimum phrase duration in seconds
        max_phrase_duration: Maximum phrase duration in seconds

    Returns:
        List of (start_time, end_time, interval_pattern) tuples
    """
    phrases = []

    if len(contour.notes) < 3:
        return phrases

    # Group notes into phrases based on gaps
    current_phrase_notes = []
    phrase_start = None

    for note in contour.notes:
        if phrase_start is None:
            phrase_start = note.start
            current_phrase_notes = [note]
        elif note.start - current_phrase_notes[-1].end > 0.5:  # Gap > 0.5s = new phrase
            # End current phrase
            if current_phrase_notes:
                phrase_end = current_phrase_notes[-1].end
                phrase_duration = phrase_end - phrase_start

                if min_phrase_duration <= phrase_duration <= max_phrase_duration:
                    # Extract interval pattern
                    pitches = [n.pitch for n in current_phrase_notes]
                    if len(pitches) >= 3:
                        intervals = np.diff(pitches)
                        phrases.append((phrase_start, phrase_end, intervals))

            # Start new phrase
            phrase_start = note.start
            current_phrase_notes = [note]
        else:
            current_phrase_notes.append(note)

    # Handle last phrase
    if current_phrase_notes:
        phrase_end = current_phrase_notes[-1].end
        phrase_duration = phrase_end - phrase_start

        if min_phrase_duration <= phrase_duration <= max_phrase_duration:
            pitches = [n.pitch for n in current_phrase_notes]
            if len(pitches) >= 3:
                intervals = np.diff(pitches)
                phrases.append((phrase_start, phrase_end, intervals))

    return phrases


def find_repeated_melodic_phrases(
    vocals: np.ndarray,
    sr: int = 22050,
    similarity_threshold: float = 0.7,
) -> List[Tuple[List[Tuple[float, float]], np.ndarray]]:
    """
    Find melodic phrases that repeat throughout the song.

    Args:
        vocals: Isolated vocals audio
        sr: Sample rate
        similarity_threshold: Minimum similarity to consider phrases matching

    Returns:
        List of ([(start, end), ...], interval_pattern) for each repeated phrase
    """
    # Extract melody
    try:
        notes = extract_melody_from_vocals(vocals, sr)
        duration = len(vocals) / sr
        contour = notes_to_pitch_contour(notes, hop_time=0.1, duration=duration)
    except Exception as e:
        print(f"Warning: Melody extraction failed: {e}")
        return []

    # Find all phrases
    phrases = find_melodic_phrases(contour)

    if len(phrases) < 2:
        return []

    # Group similar phrases
    phrase_groups: Dict[int, List[Tuple[float, float]]] = {}
    phrase_patterns: Dict[int, np.ndarray] = {}

    for i, (start, end, pattern) in enumerate(phrases):
        matched = False

        for group_id, group_pattern in phrase_patterns.items():
            if len(pattern) == len(group_pattern):
                # Normalize and compare
                if len(pattern) > 0:
                    p1 = pattern - np.mean(pattern)
                    p2 = group_pattern - np.mean(group_pattern)

                    if np.std(p1) > 0 and np.std(p2) > 0:
                        corr = np.corrcoef(p1, p2)[0, 1]
                        if not np.isnan(corr) and corr > similarity_threshold:
                            phrase_groups[group_id].append((start, end))
                            matched = True
                            break

        if not matched:
            # New group
            group_id = len(phrase_patterns)
            phrase_patterns[group_id] = pattern
            phrase_groups[group_id] = [(start, end)]

    # Return only groups with multiple occurrences
    repeated = []
    for group_id, locations in phrase_groups.items():
        if len(locations) >= 2:
            repeated.append((locations, phrase_patterns[group_id]))

    # Sort by number of occurrences
    repeated.sort(key=lambda x: len(x[0]), reverse=True)

    return repeated
