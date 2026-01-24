"""
Section detection and labeling.

Detects section boundaries using novelty curves and energy analysis,
builds self-similarity matrices, and labels sections as
intro/verse/pre-chorus/chorus/bridge/outro.
"""

from typing import List, Optional, Tuple
import numpy as np
import librosa
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

from .features import compute_novelty_curve


# Minimum section duration in seconds
MIN_SECTION_DURATION = 15.0


def detect_section_boundaries(
    y: np.ndarray,
    sr: int,
    min_section_duration: float = MIN_SECTION_DURATION,
) -> List[float]:
    """
    Detect section boundaries using novelty peaks and energy analysis.

    Args:
        y: Audio signal
        sr: Sample rate
        min_section_duration: Minimum time between boundaries in seconds

    Returns:
        List of boundary times in seconds
    """
    hop_length = 512
    novelty, frame_times = compute_novelty_curve(y, sr, hop_length)

    # Compute energy-based boundaries (verse/chorus transitions)
    # Energy drops indicate transitions from chorus back to verse
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_normalized = (rms - np.min(rms)) / (np.max(rms) - np.min(rms) + 1e-10)

    # Smooth energy curve
    rms_smooth = median_filter(rms_normalized, size=11)

    # Find significant energy drops (indicating transition to verse)
    energy_diff = -np.diff(rms_smooth, prepend=rms_smooth[0])
    energy_diff_normalized = (energy_diff - np.min(energy_diff)) / (np.max(energy_diff) - np.min(energy_diff) + 1e-10)

    # Combine novelty and energy for better boundary detection
    combined_curve = 0.7 * novelty + 0.3 * energy_diff_normalized

    # Convert min duration to frames
    min_distance = int(min_section_duration * sr / hop_length)

    # Find peaks in novelty curve
    # Use moderate thresholds to detect main structural boundaries
    # Most pop songs have 6-12 main sections
    peaks, properties = find_peaks(
        novelty,
        distance=min_distance,
        prominence=0.1,
        height=np.percentile(novelty, 60),
    )

    # Convert to times
    boundary_times = [0.0]  # Always start at 0
    for peak in peaks:
        peak_time = float(frame_times[peak])
        # Only add boundary if it's far enough from the last one
        if peak_time - boundary_times[-1] >= min_section_duration:
            boundary_times.append(peak_time)

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
        k=None,
        width=int(5 * sr / hop_length),
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
    harmonic_energy_per_section: Optional[List[float]] = None,
    tempo_stability_per_section: Optional[List[float]] = None,
    lyric_similarity: Optional[np.ndarray] = None,
    lyrics_per_section: Optional[List[str]] = None,
    drum_energy_per_section: Optional[List[float]] = None,
) -> List[str]:
    """
    Label sections as intro/verse/pre-chorus/chorus/bridge/outro.

    Enhanced with research-based criteria:
    - Intensity shifts (energy changes)
    - Pattern repetition (similarity to other sections)
    - Harmonic content (choruses often more melodic)
    - Tempo stability (choruses often have stable rhythm)
    - Lyric similarity (choruses have repeated lyrics)
    - Drum energy (bridges often have drum drops)

    Args:
        boundaries: Section boundary times
        section_similarity: Section-to-section similarity matrix
        energy_per_section: Average energy per section
        duration: Total duration
        harmonic_energy_per_section: Optional harmonic component energy per section
        tempo_stability_per_section: Optional tempo stability per section
        lyric_similarity: Optional section-to-section lyric similarity matrix
        lyrics_per_section: Optional lyrics for each section
        drum_energy_per_section: Optional drum energy per section (from Demucs)

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

    # Compute repetition score for each section
    repetition_scores = []
    max_similarity_partner = []
    for i in range(n_sections):
        other_sims = [(section_similarity[i, j], j) for j in range(n_sections) if j != i]
        if other_sims:
            max_sim, partner = max(other_sims, key=lambda x: x[0])
            repetition_scores.append(max_sim)
            max_similarity_partner.append(partner)
        else:
            repetition_scores.append(0.0)
            max_similarity_partner.append(-1)

    # Normalize repetition scores
    max_rep = max(repetition_scores) if repetition_scores else 1
    min_rep = min(repetition_scores) if repetition_scores else 0
    rep_range = max_rep - min_rep + 1e-10
    normalized_rep = [(r - min_rep) / rep_range for r in repetition_scores]

    # Compute harmonic ratio if available
    normalized_harmonic = None
    if harmonic_energy_per_section:
        harmonic_ratio = [h / (e + 1e-10) for h, e in zip(harmonic_energy_per_section, energy_per_section)]
        max_hr = max(harmonic_ratio)
        min_hr = min(harmonic_ratio)
        hr_range = max_hr - min_hr + 1e-10
        normalized_harmonic = [(hr - min_hr) / hr_range for hr in harmonic_ratio]

    # Compute stability score if available
    normalized_stability = None
    if tempo_stability_per_section:
        max_stab = max(tempo_stability_per_section)
        min_stab = min(tempo_stability_per_section)
        stab_range = max_stab - min_stab + 1e-10
        normalized_stability = [1.0 - (s - min_stab) / stab_range for s in tempo_stability_per_section]

    # Compute lyric repetition scores if available
    normalized_lyric_rep = None
    if lyric_similarity is not None:
        lyric_rep_scores = []
        for i in range(n_sections):
            other_sims = [lyric_similarity[i, j] for j in range(n_sections) if j != i]
            lyric_rep_scores.append(max(other_sims) if other_sims else 0.0)
        max_lr = max(lyric_rep_scores) if lyric_rep_scores else 1
        min_lr = min(lyric_rep_scores) if lyric_rep_scores else 0
        lr_range = max_lr - min_lr + 1e-10
        normalized_lyric_rep = [(lr - min_lr) / lr_range for lr in lyric_rep_scores]

    # Normalize drum energy if available (for bridge detection)
    normalized_drum = None
    if drum_energy_per_section:
        max_drum = max(drum_energy_per_section)
        min_drum = min(drum_energy_per_section)
        drum_range = max_drum - min_drum + 1e-10
        normalized_drum = [(d - min_drum) / drum_range for d in drum_energy_per_section]

    # First section is intro if short relative to song
    if n_sections > 1:
        first_section_end = boundaries[1] if len(boundaries) > 1 else duration
        if first_section_end < max(duration * 0.10, 25.0):
            labels[0] = 'intro'

    # Last section is outro if it starts after 85% of song
    if n_sections > 1 and boundaries[-1] > duration * 0.85:
        labels[-1] = 'outro'

    # If intro has lyrics, it's actually a verse
    if labels[0] == 'intro' and lyrics_per_section:
        if lyrics_per_section[0] and len(lyrics_per_section[0].strip()) > 10:
            labels[0] = 'verse'

    # If outro has lyrics, it's actually a verse
    if labels[-1] == 'outro' and lyrics_per_section:
        if lyrics_per_section[-1] and len(lyrics_per_section[-1].strip()) > 10:
            labels[-1] = 'verse'

    # Compute composite chorus scores
    chorus_scores = []
    for i in range(n_sections):
        if labels[i] in ['intro', 'outro']:
            chorus_scores.append(-1.0)
            continue
        if boundaries[i] < duration * 0.15:
            chorus_scores.append(-1.0)
            continue

        if normalized_lyric_rep is not None:
            score = 0.35 * normalized_lyric_rep[i] + 0.25 * normalized_energy[i] + 0.25 * normalized_rep[i]
            if normalized_harmonic:
                score += 0.075 * normalized_harmonic[i]
            else:
                score += 0.075 * normalized_energy[i]
            if normalized_stability:
                score += 0.075 * normalized_stability[i]
            else:
                score += 0.075 * normalized_rep[i]
        else:
            score = 0.4 * normalized_energy[i] + 0.4 * normalized_rep[i]
            if normalized_harmonic:
                score += 0.1 * normalized_harmonic[i]
            else:
                score += 0.1 * normalized_energy[i]
            if normalized_stability:
                score += 0.1 * normalized_stability[i]
            else:
                score += 0.1 * normalized_rep[i]

        chorus_scores.append(score)

    # Mark sections with high composite scores as chorus
    valid_scores = [(i, s) for i, s in enumerate(chorus_scores) if s >= 0]
    if valid_scores:
        valid_scores.sort(key=lambda x: x[1], reverse=True)

        best_idx = valid_scores[0][0]
        labels[best_idx] = 'chorus'

        audio_sim_threshold = 0.3
        lyric_sim_threshold = 0.4

        for i in range(n_sections):
            if labels[i] in ['intro', 'outro']:
                continue
            if i == best_idx:
                continue

            audio_sim = section_similarity[i, best_idx]
            lyric_sim = 0.0
            if lyric_similarity is not None:
                lyric_sim = lyric_similarity[i, best_idx]

            if lyric_sim >= lyric_sim_threshold:
                labels[i] = 'chorus'
            elif audio_sim >= audio_sim_threshold and normalized_energy[i] >= 0.4:
                labels[i] = 'chorus'

        for i, score in valid_scores[1:]:
            if labels[i] != 'verse':
                continue
            has_audio_repeat = normalized_rep[i] >= 0.7
            has_lyric_repeat = normalized_lyric_rep is not None and normalized_lyric_rep[i] >= 0.7
            if (has_audio_repeat or has_lyric_repeat) and normalized_energy[i] >= 0.5:
                labels[i] = 'chorus'

    # Detect pre-chorus: section immediately before chorus with consistent pattern
    _detect_pre_chorus(labels, boundaries, section_similarity, lyric_similarity, normalized_energy)

    # Find bridge: unique section with low drum energy or unique lyrics
    _detect_bridge(
        labels, boundaries, duration, repetition_scores,
        normalized_rep, normalized_drum, lyrics_per_section
    )

    return labels


def _detect_pre_chorus(
    labels: List[str],
    boundaries: List[float],
    section_similarity: np.ndarray,
    lyric_similarity: Optional[np.ndarray],
    normalized_energy: List[float],
) -> None:
    """
    Detect pre-chorus sections (modifies labels in place).

    Pre-choruses:
    - Appear immediately before choruses
    - Often appear before multiple choruses with similar content
    - Have building energy
    """
    n_sections = len(labels)

    # Find all chorus indices
    chorus_indices = [i for i, label in enumerate(labels) if label == 'chorus']

    if len(chorus_indices) < 2:
        return

    # Find sections that appear before multiple choruses
    pre_chorus_candidates = {}

    for chorus_idx in chorus_indices:
        if chorus_idx == 0:
            continue

        prev_idx = chorus_idx - 1
        if labels[prev_idx] != 'verse':
            continue

        # Check if this section is similar to sections before other choruses
        for other_chorus_idx in chorus_indices:
            if other_chorus_idx == chorus_idx or other_chorus_idx == 0:
                continue

            other_prev_idx = other_chorus_idx - 1
            if other_prev_idx == prev_idx:
                continue

            # Check audio similarity
            audio_sim = section_similarity[prev_idx, other_prev_idx]

            # Check lyric similarity if available
            lyric_sim = 0.0
            if lyric_similarity is not None:
                lyric_sim = lyric_similarity[prev_idx, other_prev_idx]

            # If sections before choruses are similar, they're likely pre-choruses
            if audio_sim >= 0.3 or lyric_sim >= 0.4:
                if prev_idx not in pre_chorus_candidates:
                    pre_chorus_candidates[prev_idx] = 0
                pre_chorus_candidates[prev_idx] += 1

                if other_prev_idx not in pre_chorus_candidates:
                    pre_chorus_candidates[other_prev_idx] = 0
                pre_chorus_candidates[other_prev_idx] += 1

    # Mark sections that appear before multiple choruses as pre-chorus
    for idx, count in pre_chorus_candidates.items():
        if count >= 1:  # Appears before at least 2 choruses (count is incremented for pairs)
            labels[idx] = 'pre-chorus'


def _detect_bridge(
    labels: List[str],
    boundaries: List[float],
    duration: float,
    repetition_scores: List[float],
    normalized_rep: List[float],
    normalized_drum: Optional[List[float]],
    lyrics_per_section: Optional[List[str]],
) -> None:
    """
    Detect bridge sections (modifies labels in place).

    Bridges:
    - Unique (low similarity to other sections)
    - Often have drum drops (low drum energy)
    - May have vocalization patterns ("oh-oh")
    - Appear in middle-to-late portion of song
    """
    n_sections = len(labels)
    bridge_candidates = []

    for i in range(n_sections):
        if labels[i] != 'verse':
            continue

        section_position = boundaries[i] / duration

        # Must be in middle-to-late portion of song (50-85%)
        if not (0.50 <= section_position <= 0.85):
            continue

        # Check for uniqueness
        is_unique = repetition_scores[i] < np.percentile(repetition_scores, 40)

        # Check for drum drop (strong bridge indicator)
        has_drum_drop = False
        if normalized_drum is not None:
            has_drum_drop = normalized_drum[i] < 0.3

        # Check for vocalization patterns ("oh", "ah", etc.)
        has_vocalizations = False
        if lyrics_per_section and lyrics_per_section[i]:
            lyrics = lyrics_per_section[i].lower()
            vocalization_patterns = ['oh', 'ah', 'la', 'na', 'yeah', 'ooh', 'aah']
            word_count = len(lyrics.split())
            vocalization_count = sum(lyrics.count(p) for p in vocalization_patterns)
            if word_count > 0 and vocalization_count / word_count > 0.3:
                has_vocalizations = True

        # Score bridge candidates
        if is_unique or has_drum_drop or has_vocalizations:
            position_score = 1.0 - abs(section_position - 0.67) * 2
            uniqueness_score = 1.0 - normalized_rep[i]

            # Boost score for drum drops and vocalizations
            drum_bonus = 0.3 if has_drum_drop else 0.0
            vocal_bonus = 0.2 if has_vocalizations else 0.0

            bridge_score = 0.4 * uniqueness_score + 0.3 * position_score + drum_bonus + vocal_bonus
            bridge_candidates.append((i, bridge_score))

    # Pick the best bridge candidate
    if bridge_candidates:
        bridge_idx = max(bridge_candidates, key=lambda x: x[1])[0]
        labels[bridge_idx] = 'bridge'


def merge_adjacent_sections(sections: List[dict]) -> List[dict]:
    """
    Merge adjacent sections with the same label.

    Special handling for choruses: don't merge if far apart or different keys.

    Args:
        sections: List of section dictionaries with 'label', 'start', 'key', 'lyrics'

    Returns:
        Merged list of sections
    """
    merged = []

    for section in sections:
        if merged and merged[-1]["label"] == section["label"]:
            # For chorus sections, don't merge if they're far apart or have different keys
            if section["label"] == "chorus":
                time_gap = section["start"] - merged[-1]["start"]
                if time_gap > 30:
                    merged.append(section)
                    continue
                if merged[-1]["key"] != section["key"]:
                    merged.append(section)
                    continue

            # Merge with previous section - combine lyrics if both exist
            if merged[-1].get("lyrics") and section.get("lyrics"):
                merged[-1]["lyrics"] = merged[-1]["lyrics"] + " " + section["lyrics"]
            elif section.get("lyrics"):
                merged[-1]["lyrics"] = section["lyrics"]
        else:
            merged.append(section)

    return merged
