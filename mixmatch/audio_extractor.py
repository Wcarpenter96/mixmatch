"""
Local audio file extraction

Extracts BPM, sections (intro/verse/chorus/outro/bridge), and keys from audio files.
Uses self-similarity matrices and novelty detection for structural analysis.

Enhanced with techniques from chorus detection research:
- Beat-synchronized features for cleaner section detection
- Harmonic-percussive source separation (HPSS)
- Tempogram analysis for rhythmic patterns
- MFCC features for timbral characteristics
- Multi-feature novelty curve
- Composite chorus scoring
- Vocal separation and lyric transcription for improved chorus detection
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import librosa
from scipy.ndimage import median_filter
from scipy.signal import find_peaks


# Pitch class names for key detection
PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Minimum section duration in seconds (sections shorter than this get merged)
# Pop songs typically have 15-25 second sections
MIN_SECTION_DURATION = 15.0

# Feature weights for novelty curve (tuned based on research)
NOVELTY_WEIGHTS = {
    'spectral': 0.25,
    'chroma': 0.35,
    'onset': 0.15,
    'mfcc': 0.25,
}

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


def compute_key_invariant_chroma(
    chroma: np.ndarray,
) -> np.ndarray:
    """
    Make chromagram key-invariant by rotating to align with detected key.

    This helps identify similar sections even if they're in different keys,
    which is useful for detecting repeated chorus sections with key modulations.

    Args:
        chroma: Chroma features (12, n_frames)

    Returns:
        Key-invariant chroma features (12, n_frames)
    """
    # Detect key (find the pitch class with highest energy)
    chroma_sum = np.sum(chroma, axis=1)
    key_shift = np.argmax(chroma_sum)

    # Rotate chroma to make it key-invariant (transpose to C)
    return np.roll(chroma, -key_shift, axis=0)


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
        prominence=0.1,  # Lower prominence threshold
        height=np.percentile(novelty, 60),  # Top 40% of peaks
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
    harmonic_energy_per_section: Optional[List[float]] = None,
    tempo_stability_per_section: Optional[List[float]] = None,
    lyric_similarity: Optional[np.ndarray] = None,
    lyrics_per_section: Optional[List[str]] = None,
) -> List[str]:
    """
    Label sections as intro/verse/chorus/bridge/outro.

    Enhanced with research-based criteria:
    - Intensity shifts (energy changes)
    - Pattern repetition (similarity to other sections)
    - Harmonic content (choruses often more melodic)
    - Tempo stability (choruses often have stable rhythm)
    - Lyric similarity (choruses have repeated lyrics)

    Strategy:
    - Intro: First section if short relative to song
    - Outro: Last section if starts late in song
    - Chorus: High composite score (energy + repetition + harmonic ratio + stability + lyric repetition)
    - Bridge: Lower energy, appears once, typically after middle of song
    - Verse: Everything else

    Args:
        boundaries: Section boundary times
        section_similarity: Section-to-section similarity matrix
        energy_per_section: Average energy per section
        duration: Total duration
        harmonic_energy_per_section: Optional harmonic component energy per section
        tempo_stability_per_section: Optional tempo stability per section (lower = more stable)
        lyric_similarity: Optional section-to-section lyric similarity matrix

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
    # Key insight: choruses are similar TO EACH OTHER, not to all sections
    # So we look for MAX similarity to any other section (indicating repeated sections)
    repetition_scores = []
    max_similarity_partner = []  # Track which section each is most similar to
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

    # Compute harmonic ratio if available (choruses often more harmonic/melodic)
    normalized_harmonic = None
    if harmonic_energy_per_section:
        harmonic_ratio = [h / (e + 1e-10) for h, e in zip(harmonic_energy_per_section, energy_per_section)]
        max_hr = max(harmonic_ratio)
        min_hr = min(harmonic_ratio)
        hr_range = max_hr - min_hr + 1e-10
        normalized_harmonic = [(hr - min_hr) / hr_range for hr in harmonic_ratio]

    # Compute stability score if available (lower variance = more stable = higher score)
    normalized_stability = None
    if tempo_stability_per_section:
        # Invert so that lower variance = higher score
        max_stab = max(tempo_stability_per_section)
        min_stab = min(tempo_stability_per_section)
        stab_range = max_stab - min_stab + 1e-10
        # Invert: stable sections get higher scores
        normalized_stability = [1.0 - (s - min_stab) / stab_range for s in tempo_stability_per_section]

    # Compute lyric repetition scores if available
    # Choruses have highly similar lyrics to other choruses
    normalized_lyric_rep = None
    if lyric_similarity is not None:
        lyric_rep_scores = []
        for i in range(n_sections):
            # Max similarity to any other section (same logic as audio repetition)
            other_sims = [lyric_similarity[i, j] for j in range(n_sections) if j != i]
            lyric_rep_scores.append(max(other_sims) if other_sims else 0.0)
        # Normalize
        max_lr = max(lyric_rep_scores) if lyric_rep_scores else 1
        min_lr = min(lyric_rep_scores) if lyric_rep_scores else 0
        lr_range = max_lr - min_lr + 1e-10
        normalized_lyric_rep = [(lr - min_lr) / lr_range for lr in lyric_rep_scores]

    # First section is intro if it ends before the first chorus-like section
    # Typically intro is before 10% of the song or before ~20 seconds
    if n_sections > 1:
        first_section_end = boundaries[1] if len(boundaries) > 1 else duration
        if first_section_end < max(duration * 0.10, 25.0):
            labels[0] = 'intro'

    # Last section is outro if it starts after 85% of song
    if n_sections > 1 and boundaries[-1] > duration * 0.85:
        labels[-1] = 'outro'

    # If intro has lyrics, it's actually a verse (merge with next verse)
    # Intros are typically instrumental/short
    if labels[0] == 'intro' and lyrics_per_section:
        if lyrics_per_section[0] and len(lyrics_per_section[0].strip()) > 10:
            # Intro has substantial lyrics - it's really a verse
            labels[0] = 'verse'

    # If outro has lyrics, it's actually a verse (not an outro)
    if labels[-1] == 'outro' and lyrics_per_section:
        if lyrics_per_section[-1] and len(lyrics_per_section[-1].strip()) > 10:
            # Outro has substantial lyrics - it's really a verse
            labels[-1] = 'verse'

    # Compute composite chorus scores
    # Weights: lyric repetition is strongest signal, then energy and audio repetition
    chorus_scores = []
    for i in range(n_sections):
        if labels[i] in ['intro', 'outro']:
            chorus_scores.append(-1.0)
            continue
        if boundaries[i] < duration * 0.15:
            chorus_scores.append(-1.0)
            continue

        # If we have lyric similarity, use adjusted weights
        if normalized_lyric_rep is not None:
            # Lyric repetition (35%) + energy (25%) + audio repetition (25%) + other (15%)
            score = 0.35 * normalized_lyric_rep[i] + 0.25 * normalized_energy[i] + 0.25 * normalized_rep[i]
            # Add harmonic and stability bonuses (7.5% each)
            if normalized_harmonic:
                score += 0.075 * normalized_harmonic[i]
            else:
                score += 0.075 * normalized_energy[i]
            if normalized_stability:
                score += 0.075 * normalized_stability[i]
            else:
                score += 0.075 * normalized_rep[i]
        else:
            # Original weights when no lyrics available
            # Energy (40%) + repetition (40%) + harmonic (10%) + stability (10%)
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
    # Strategy: Find the best chorus candidate, then find all similar sections
    valid_scores = [(i, s) for i, s in enumerate(chorus_scores) if s >= 0]
    if valid_scores:
        # Sort by score descending
        valid_scores.sort(key=lambda x: x[1], reverse=True)

        # Mark the highest scoring section as chorus
        best_idx = valid_scores[0][0]
        labels[best_idx] = 'chorus'

        # Find all sections similar to the best chorus (repeated choruses)
        # Use both audio and lyric similarity (lyrics are stronger evidence)
        audio_sim_threshold = 0.3  # Sections with >30% audio similarity
        lyric_sim_threshold = 0.4  # Sections with >40% lyric similarity (lyrics are more reliable)

        for i in range(n_sections):
            if labels[i] in ['intro', 'outro']:
                continue
            if i == best_idx:
                continue

            # Check audio similarity to identified chorus
            audio_sim = section_similarity[i, best_idx]

            # Check lyric similarity if available
            lyric_sim = 0.0
            if lyric_similarity is not None:
                lyric_sim = lyric_similarity[i, best_idx]

            # Mark as chorus if either:
            # 1. High lyric similarity (repeated lyrics = likely chorus)
            # 2. High audio similarity AND reasonable energy
            if lyric_sim >= lyric_sim_threshold:
                labels[i] = 'chorus'
            elif audio_sim >= audio_sim_threshold and normalized_energy[i] >= 0.4:
                labels[i] = 'chorus'

        # Also check for sections that are similar to each other AND high energy
        # This catches choruses even if they're not the absolute highest scored
        for i, score in valid_scores[1:]:  # Skip the first (already labeled)
            if labels[i] != 'verse':
                continue
            # If this section has a strong repeat AND high energy
            has_audio_repeat = normalized_rep[i] >= 0.7
            has_lyric_repeat = normalized_lyric_rep is not None and normalized_lyric_rep[i] >= 0.7
            if (has_audio_repeat or has_lyric_repeat) and normalized_energy[i] >= 0.5:
                labels[i] = 'chorus'

    # Smart verse detection: consecutive sections with similar lyrics should both be verses
    # (not intro/outro/chorus)
    if lyric_similarity is not None:
        for i in range(n_sections - 1):
            current_label = labels[i]
            next_label = labels[i + 1]
            
            # If both sections are currently labeled as verse or unlabeled
            if current_label in ['verse'] and next_label in ['verse']:
                # Check if they have meaningful lyric similarity
                lyric_sim = lyric_similarity[i, i + 1]
                
                # If lyrics are moderately similar but NOT too similar (not chorus-repetition level)
                # this confirms they're both verses
                if 0.15 <= lyric_sim < 0.5:
                    # Both are likely verses - keep them as is
                    continue
                # If lyrics are very dissimilar, next one might be something else
                elif lyric_sim < 0.15:
                    # This is fine - verses can have different lyrics
                    continue

    # Find bridge: unique section in the second half, typically before final chorus
    # Bridges have low repetition (they only appear once) and often lower energy
    bridge_candidates = []
    for i in range(n_sections):
        if labels[i] == 'verse':
            section_position = boundaries[i] / duration
            # Must be in middle-to-late portion of song (50-85%)
            if 0.50 <= section_position <= 0.85:
                # Key characteristic: bridge is UNIQUE (low similarity to other sections)
                # This is the most reliable indicator
                is_unique = repetition_scores[i] < np.percentile(repetition_scores, 40)

                if is_unique:
                    # Score bridge candidates by how unique they are (lower rep = better bridge)
                    # Also factor in position (bridges usually around 60-75%)
                    position_score = 1.0 - abs(section_position - 0.67) * 2
                    uniqueness_score = 1.0 - normalized_rep[i]
                    bridge_score = 0.6 * uniqueness_score + 0.4 * position_score
                    bridge_candidates.append((i, bridge_score))

    # Pick the best bridge candidate (highest score = most unique + best position)
    if bridge_candidates:
        bridge_idx = max(bridge_candidates, key=lambda x: x[1])[0]
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
    Extract BPM, sections, keys, and lyrics from audio file.

    Enhanced with research-based techniques:
    - Multi-feature novelty detection (spectral, chroma, onset, MFCC)
    - Harmonic-percussive source separation (HPSS)
    - Tempogram analysis for rhythmic stability
    - Composite chorus scoring
    - Vocal separation and lyric transcription (Spleeter + Whisper)

    Args:
        audio_path: Path to audio file

    Returns:
        Dictionary with structure:
        {
            "bpm": int,
            "key": str,
            "sections": [
                {"label": str, "start": float, "key": str, "lyrics": str},
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

    # Step 5: Detect section boundaries using enhanced multi-feature novelty curve
    boundaries = detect_section_boundaries(y, sr, min_section_duration=MIN_SECTION_DURATION)

    # Don't use fallback thresholds - stick with the configured MIN_SECTION_DURATION
    # to maintain consistent section lengths across different songs

    # Step 6: Build self-similarity matrix using key-invariant chroma
    key_inv_chroma = compute_key_invariant_chroma(chroma)
    rec_matrix = build_recurrence_matrix(key_inv_chroma, hop_length, sr)

    # Step 7: Compute section similarity
    section_similarity = compute_section_similarity(rec_matrix, boundaries, duration)

    # Step 8: Extract HPSS features for harmonic/percussive analysis
    try:
        rms_harmonic, rms_percussive = extract_hpss_features(y, sr, hop_length)
    except Exception:
        # Fallback if HPSS fails
        rms_harmonic = None
        rms_percussive = None

    # Step 9: Extract tempogram features for rhythmic stability
    try:
        onset_env, tempo_stability = compute_tempogram_features(y, sr, hop_length)
    except Exception:
        # Fallback if tempogram fails
        tempo_stability = None

    # Step 10: Extract vocals and transcribe lyrics
    lyrics_per_section = None
    lyric_similarity = None
    try:
        from mixmatch.vocals_extractor import extract_vocals_and_lyrics
        lyrics_per_section, lyric_similarity = extract_vocals_and_lyrics(
            audio_path, boundaries, duration
        )
    except ImportError:
        # Spleeter or Whisper not installed
        print("Warning: Spleeter/Whisper not installed. Skipping lyrics extraction.")
    except Exception as e:
        # Vocals extraction failed - continue without lyrics
        print(f"Warning: Vocals extraction failed: {e}")

    # Step 11: Compute energy per section (total, harmonic, and tempo stability)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    energy_per_section = []
    harmonic_energy_per_section = []
    tempo_stability_per_section = []

    for i in range(len(boundaries)):
        start_time = boundaries[i]
        end_time = boundaries[i + 1] if i + 1 < len(boundaries) else duration

        # Find frames in this section
        mask = (rms_times >= start_time) & (rms_times < end_time)

        # Total energy
        section_rms = rms[mask]
        section_energy = float(np.mean(section_rms)) if len(section_rms) > 0 else 0.0
        energy_per_section.append(section_energy)

        # Harmonic energy (if available)
        if rms_harmonic is not None:
            section_harmonic = rms_harmonic[mask]
            harmonic_energy = float(np.mean(section_harmonic)) if len(section_harmonic) > 0 else 0.0
            harmonic_energy_per_section.append(harmonic_energy)

        # Tempo stability (if available)
        if tempo_stability is not None:
            section_stability = tempo_stability[mask]
            stability = float(np.mean(section_stability)) if len(section_stability) > 0 else 0.0
            tempo_stability_per_section.append(stability)

    # Step 12: Label sections with enhanced composite scoring
    labels = label_sections(
        boundaries,
        section_similarity,
        energy_per_section,
        duration,
        harmonic_energy_per_section=harmonic_energy_per_section if harmonic_energy_per_section else None,
        tempo_stability_per_section=tempo_stability_per_section if tempo_stability_per_section else None,
        lyric_similarity=lyric_similarity,
        lyrics_per_section=lyrics_per_section,
    )

    # Step 13: Build output sections with per-section key detection and lyrics
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

        # Get lyrics for this section (if available)
        section_lyrics = None
        if lyrics_per_section and i < len(lyrics_per_section):
            section_lyrics = lyrics_per_section[i] if lyrics_per_section[i] else None

        sections.append({
            "label": labels[i],
            "start": round(boundary_time, 1),
            "key": section_key,
            "lyrics": section_lyrics,
        })

    # Merge adjacent sections with the same label
    # But don't merge choruses that are far apart (they might be different sections)
    merged_sections = []
    for section in sections:
        if merged_sections and merged_sections[-1]["label"] == section["label"]:
            # For chorus sections, don't merge if they're far apart (> 30 seconds) or have different keys
            if section["label"] == "chorus":
                time_gap = section["start"] - merged_sections[-1]["start"]
                # Don't merge distant choruses - they're likely different instances
                if time_gap > 30:
                    merged_sections.append(section)
                    continue
                # If keys are different, don't merge
                if merged_sections[-1]["key"] != section["key"]:
                    merged_sections.append(section)
                    continue
            
            # Merge with previous section - combine lyrics if both exist
            if merged_sections[-1]["lyrics"] and section["lyrics"]:
                merged_sections[-1]["lyrics"] = merged_sections[-1]["lyrics"] + " " + section["lyrics"]
            elif section["lyrics"]:
                merged_sections[-1]["lyrics"] = section["lyrics"]
        else:
            merged_sections.append(section)

    return {
        "bpm": bpm,
        "key": song_key,
        "sections": merged_sections,
    }
