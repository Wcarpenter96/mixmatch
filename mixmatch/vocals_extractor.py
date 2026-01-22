"""
Vocal extraction and lyrics transcription.

Uses Whisper for direct audio transcription.
Provides lyric similarity analysis to help identify repeated sections (choruses).
"""

import os
import tempfile
import shutil
from typing import List, Dict, Tuple, Optional
import numpy as np
import librosa
import soundfile as sf

# Lazy imports to avoid loading heavy models at module import time
_whisper_model = None


def _get_whisper():
    """Lazy load Whisper model (base size)."""
    global _whisper_model
    if _whisper_model is None:
        import whisper
        _whisper_model = whisper.load_model("base")
    return _whisper_model


def transcribe_audio(
    audio_path: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> str:
    """
    Transcribe audio to text using Whisper.

    Args:
        audio_path: Path to audio file
        start_time: Optional start time in seconds for segment
        end_time: Optional end time in seconds for segment

    Returns:
        Transcribed text
    """
    model = _get_whisper()

    # If segment times are specified, extract that portion
    if start_time is not None or end_time is not None:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        start_sample = int((start_time or 0) * sr)
        end_sample = int((end_time or len(y) / sr) * sr)
        y_segment = y[start_sample:end_sample]

        # Write to temp file for Whisper
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            sf.write(temp_path, y_segment, sr)

        try:
            result = model.transcribe(temp_path, language="en", fp16=False)
            return result["text"].strip()
        finally:
            os.unlink(temp_path)
    else:
        result = model.transcribe(audio_path, language="en", fp16=False)
        return result["text"].strip()


def transcribe_sections(
    vocals_path: str,
    boundaries: List[float],
    duration: float,
) -> List[str]:
    """
    Transcribe lyrics for each section.

    Args:
        vocals_path: Path to isolated vocals audio
        boundaries: List of section start times in seconds
        duration: Total audio duration in seconds

    Returns:
        List of transcribed lyrics for each section
    """
    lyrics = []

    for i, start_time in enumerate(boundaries):
        # Determine end time
        if i + 1 < len(boundaries):
            end_time = boundaries[i + 1]
        else:
            end_time = duration

        # Skip very short sections (likely intro/outro silence)
        if end_time - start_time < 3.0:
            lyrics.append("")
            continue

        try:
            section_lyrics = transcribe_audio(vocals_path, start_time, end_time)
            lyrics.append(section_lyrics)
        except Exception as e:
            print(f"Warning: Failed to transcribe section {i}: {e}")
            lyrics.append("")

    return lyrics


def compute_lyric_similarity(lyrics: List[str]) -> np.ndarray:
    """
    Compute pairwise similarity between section lyrics.

    Uses simple word overlap (Jaccard similarity) which works well
    for detecting repeated choruses.

    Args:
        lyrics: List of lyrics strings for each section

    Returns:
        n_sections x n_sections similarity matrix
    """
    n = len(lyrics)
    similarity = np.zeros((n, n))

    # Tokenize and normalize
    tokenized = []
    for lyric in lyrics:
        # Simple tokenization: lowercase, split on whitespace, remove punctuation
        words = set(
            word.lower().strip(".,!?\"'")
            for word in lyric.split()
            if len(word) > 2  # Skip very short words
        )
        tokenized.append(words)

    # Compute Jaccard similarity
    for i in range(n):
        for j in range(n):
            if i == j:
                similarity[i, j] = 1.0
            elif len(tokenized[i]) == 0 or len(tokenized[j]) == 0:
                similarity[i, j] = 0.0
            else:
                intersection = len(tokenized[i] & tokenized[j])
                union = len(tokenized[i] | tokenized[j])
                similarity[i, j] = intersection / union if union > 0 else 0.0

    return similarity


def extract_vocals_and_lyrics(
    audio_path: str,
    boundaries: List[float],
    duration: float,
) -> Tuple[List[str], np.ndarray]:
    """
    Full pipeline: transcribe sections and compute lyric similarity.

    Args:
        audio_path: Path to original audio file
        boundaries: Section boundary times
        duration: Total audio duration

    Returns:
        Tuple of (lyrics_per_section, lyric_similarity_matrix)
    """
    # Transcribe each section directly from the audio
    lyrics = transcribe_sections(audio_path, boundaries, duration)

    # Compute lyric similarity
    similarity = compute_lyric_similarity(lyrics)

    return lyrics, similarity
