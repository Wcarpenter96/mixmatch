"""
Audio extraction orchestrator.

Coordinates the extraction of BPM, sections, keys, and lyrics from audio files.
Uses modular components for each analysis task.
"""

from typing import Dict, Optional
import numpy as np
import librosa

from .key_detector import detect_key, detect_section_key, compute_key_invariant_chroma
from .features import (
    extract_hpss_features,
    compute_tempogram_features,
    compute_energy_per_section,
    compute_harmonic_energy_per_section,
    compute_stability_per_section,
)
from .section_detector import (
    detect_section_boundaries,
    build_recurrence_matrix,
    compute_section_similarity,
    label_sections,
    merge_adjacent_sections,
    MIN_SECTION_DURATION,
)


def extract(audio_path: str, use_source_separation: bool = True) -> Dict:
    """
    Extract BPM, sections, keys, and lyrics from audio file.

    Uses modular components:
    - Demucs for vocal/drum isolation (improved transcription and bridge detection)
    - Whisper for lyric transcription
    - Multi-feature novelty detection for section boundaries
    - Composite scoring for section labeling

    Args:
        audio_path: Path to audio file
        use_source_separation: Whether to use Demucs for vocal isolation

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

    # Step 4: Detect overall song key
    song_key = detect_key(chroma)

    # Step 5: Detect section boundaries
    boundaries = detect_section_boundaries(y, sr, min_section_duration=MIN_SECTION_DURATION)

    # Step 6: Build self-similarity matrix
    key_inv_chroma = compute_key_invariant_chroma(chroma)
    rec_matrix = build_recurrence_matrix(key_inv_chroma, hop_length, sr)

    # Step 7: Compute section similarity
    section_similarity = compute_section_similarity(rec_matrix, boundaries, duration)

    # Step 8: Extract HPSS features
    rms_harmonic = None
    rms_percussive = None
    try:
        rms_harmonic, rms_percussive = extract_hpss_features(y, sr, hop_length)
    except Exception:
        pass

    # Step 9: Extract tempogram features
    tempo_stability = None
    try:
        _, tempo_stability = compute_tempogram_features(y, sr, hop_length)
    except Exception:
        pass

    # Step 10: Extract drum energy for bridge detection (using Demucs)
    drum_energy_per_section = None
    if use_source_separation:
        try:
            from .source_separator import analyze_drum_energy
            drum_rms = analyze_drum_energy(audio_path, hop_length, sr)
            if len(drum_rms) > 0:
                drum_times = librosa.frames_to_time(np.arange(len(drum_rms)), sr=sr, hop_length=hop_length)
                drum_energy_per_section = []
                for i in range(len(boundaries)):
                    start_time = boundaries[i]
                    end_time = boundaries[i + 1] if i + 1 < len(boundaries) else duration
                    mask = (drum_times >= start_time) & (drum_times < end_time)
                    section_drum = drum_rms[mask]
                    drum_energy = float(np.mean(section_drum)) if len(section_drum) > 0 else 0.0
                    drum_energy_per_section.append(drum_energy)
        except ImportError:
            pass
        except Exception as e:
            print(f"Warning: Drum analysis failed: {e}")

    # Step 11: Extract vocals and transcribe lyrics
    lyrics_per_section = None
    lyric_similarity = None
    try:
        from .vocals_extractor import extract_vocals_and_lyrics
        lyrics_per_section, lyric_similarity = extract_vocals_and_lyrics(
            audio_path, boundaries, duration, use_source_separation=use_source_separation
        )
    except ImportError:
        print("Warning: Whisper not installed. Skipping lyrics extraction.")
    except Exception as e:
        print(f"Warning: Vocals extraction failed: {e}")

    # Step 12: Compute energy per section
    energy_per_section = compute_energy_per_section(y, sr, boundaries, duration, hop_length)

    # Compute harmonic energy per section
    harmonic_energy_per_section = None
    if rms_harmonic is not None:
        harmonic_energy_per_section = compute_harmonic_energy_per_section(
            rms_harmonic, sr, boundaries, duration, hop_length
        )

    # Compute tempo stability per section
    tempo_stability_per_section = None
    if tempo_stability is not None:
        tempo_stability_per_section = compute_stability_per_section(
            tempo_stability, sr, boundaries, duration, hop_length
        )

    # Step 13: Label sections
    labels = label_sections(
        boundaries,
        section_similarity,
        energy_per_section,
        duration,
        harmonic_energy_per_section=harmonic_energy_per_section,
        tempo_stability_per_section=tempo_stability_per_section,
        lyric_similarity=lyric_similarity,
        lyrics_per_section=lyrics_per_section,
        drum_energy_per_section=drum_energy_per_section,
    )

    # Step 14: Build output sections with per-section key detection and lyrics
    sections = []
    for i, boundary_time in enumerate(boundaries):
        start_frame = int(boundary_time * n_chroma_frames / duration)
        if i + 1 < len(boundaries):
            end_frame = int(boundaries[i + 1] * n_chroma_frames / duration)
        else:
            end_frame = n_chroma_frames

        section_key = detect_section_key(chroma, start_frame, end_frame)

        section_lyrics = None
        if lyrics_per_section and i < len(lyrics_per_section):
            section_lyrics = lyrics_per_section[i] if lyrics_per_section[i] else None

        sections.append({
            "label": labels[i],
            "start": round(boundary_time, 1),
            "key": section_key,
            "lyrics": section_lyrics,
        })

    # Step 15: Merge adjacent sections with same label
    merged_sections = merge_adjacent_sections(sections)

    return {
        "bpm": bpm,
        "key": song_key,
        "sections": merged_sections,
    }
