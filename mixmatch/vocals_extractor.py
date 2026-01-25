"""
Vocal extraction and lyrics transcription.

Uses Demucs for vocal isolation and Whisper for transcription.
Provides lyric similarity analysis to help identify repeated sections (choruses).
Supports word-level timestamps for precise section boundary refinement.
"""

import os
import tempfile
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from collections import Counter
import numpy as np
import soundfile as sf

# Lazy imports to avoid loading heavy models at module import time
_whisper_model = None


@dataclass
class WordTiming:
    """Word with its timing information."""
    word: str
    start: float
    end: float


@dataclass
class TranscriptionResult:
    """Transcription with word-level timestamps."""
    text: str
    words: List[WordTiming]


def _get_whisper():
    """Lazy load Whisper model (base size)."""
    global _whisper_model
    if _whisper_model is None:
        import whisper
        _whisper_model = whisper.load_model("base")
    return _whisper_model


def transcribe_audio(
    audio: np.ndarray,
    sr: int = 22050,
) -> str:
    """
    Transcribe audio to text using Whisper.

    Args:
        audio: Audio signal as numpy array
        sr: Sample rate of the audio

    Returns:
        Transcribed text
    """
    model = _get_whisper()

    # Whisper expects 16kHz audio
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    # Write to temp file for Whisper
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
        sf.write(temp_path, audio, 16000)

    try:
        result = model.transcribe(temp_path, language="en", fp16=False)
        return result["text"].strip()
    finally:
        os.unlink(temp_path)


def transcribe_audio_with_timestamps(
    audio: np.ndarray,
    sr: int = 22050,
    time_offset: float = 0.0,
) -> TranscriptionResult:
    """
    Transcribe audio with word-level timestamps.

    Args:
        audio: Audio signal as numpy array
        sr: Sample rate of the audio
        time_offset: Offset to add to all timestamps (for section-relative times)

    Returns:
        TranscriptionResult with text and word timings
    """
    model = _get_whisper()

    # Whisper expects 16kHz audio
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    # Write to temp file for Whisper
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
        sf.write(temp_path, audio, 16000)

    try:
        result = model.transcribe(
            temp_path,
            language="en",
            fp16=False,
            word_timestamps=True,
        )

        words = []
        for segment in result.get("segments", []):
            for word_info in segment.get("words", []):
                words.append(WordTiming(
                    word=word_info["word"].strip(),
                    start=word_info["start"] + time_offset,
                    end=word_info["end"] + time_offset,
                ))

        return TranscriptionResult(
            text=result["text"].strip(),
            words=words,
        )
    finally:
        os.unlink(temp_path)


def transcribe_audio_file(
    audio_path: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> str:
    """
    Transcribe audio file to text using Whisper.

    Args:
        audio_path: Path to audio file
        start_time: Optional start time in seconds for segment
        end_time: Optional end time in seconds for segment

    Returns:
        Transcribed text
    """
    import librosa

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
    vocals: np.ndarray,
    boundaries: List[float],
    duration: float,
    sr: int = 22050,
) -> List[str]:
    """
    Transcribe lyrics for each section from isolated vocals.

    Args:
        vocals: Isolated vocals audio (from Demucs)
        boundaries: List of section start times in seconds
        duration: Total audio duration in seconds
        sr: Sample rate of vocals

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
            # Extract segment from vocals
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment = vocals[start_sample:end_sample]

            if len(segment) == 0:
                lyrics.append("")
                continue

            section_lyrics = transcribe_audio(segment, sr)
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


def _normalize_word(word: str) -> str:
    """Normalize a word for comparison."""
    return word.lower().strip(".,!?\"'()-")


def find_signature_phrases(
    transcriptions: List[TranscriptionResult],
    min_phrase_length: int = 3,
    max_phrase_length: int = 6,
    min_occurrences: int = 2,
) -> List[Tuple[Tuple[str, ...], List[Tuple[int, float]]]]:
    """
    Find repeated phrases across sections that could indicate chorus signatures.

    Args:
        transcriptions: List of TranscriptionResult for each section
        min_phrase_length: Minimum words in a phrase
        max_phrase_length: Maximum words in a phrase
        min_occurrences: Minimum times a phrase must appear

    Returns:
        List of (phrase_tuple, [(section_idx, start_time), ...]) sorted by frequency
    """
    # Collect all n-grams with their locations
    phrase_locations: Dict[Tuple[str, ...], List[Tuple[int, float]]] = {}

    for section_idx, transcription in enumerate(transcriptions):
        words = transcription.words
        if len(words) < min_phrase_length:
            continue

        # Extract n-grams of various lengths
        for phrase_len in range(min_phrase_length, min(max_phrase_length + 1, len(words) + 1)):
            for i in range(len(words) - phrase_len + 1):
                phrase_words = words[i:i + phrase_len]
                phrase_tuple = tuple(_normalize_word(w.word) for w in phrase_words)

                # Skip phrases with too many short/common words
                meaningful_words = [w for w in phrase_tuple if len(w) > 2]
                if len(meaningful_words) < phrase_len // 2:
                    continue

                if phrase_tuple not in phrase_locations:
                    phrase_locations[phrase_tuple] = []

                phrase_locations[phrase_tuple].append((
                    section_idx,
                    phrase_words[0].start,
                ))

    # Filter to phrases that appear multiple times and in different sections
    repeated_phrases = []
    for phrase, locations in phrase_locations.items():
        unique_sections = set(loc[0] for loc in locations)
        if len(locations) >= min_occurrences and len(unique_sections) >= min_occurrences:
            repeated_phrases.append((phrase, locations))

    # Sort by number of occurrences (most frequent first)
    repeated_phrases.sort(key=lambda x: len(x[1]), reverse=True)

    return repeated_phrases


def refine_boundaries_with_phrases(
    boundaries: List[float],
    labels: List[str],
    transcriptions: List[TranscriptionResult],
    duration: float,
    min_section_duration: float = 8.0,
) -> Tuple[List[float], List[str], List[str]]:
    """
    Refine section boundaries using repeated phrase detection.

    If a section labeled 'chorus' contains a signature phrase that starts
    mid-section, split the section so choruses align at the phrase.

    Args:
        boundaries: Original section boundary times
        labels: Section labels
        transcriptions: Word-level transcriptions per section
        duration: Total audio duration
        min_section_duration: Minimum duration for a section

    Returns:
        Tuple of (refined_boundaries, refined_labels, refined_lyrics)
    """
    if len(boundaries) == 0:
        return boundaries, labels, []

    # Find signature phrases (likely chorus hooks)
    signature_phrases = find_signature_phrases(transcriptions, min_phrase_length=3)

    if not signature_phrases:
        # No repeated phrases found, return original
        lyrics = [t.text for t in transcriptions]
        return boundaries, labels, lyrics

    # Get the most common phrase as the likely chorus signature
    chorus_phrase, phrase_locations = signature_phrases[0]

    # Find sections labeled as chorus
    chorus_indices = [i for i, label in enumerate(labels) if label == 'chorus']

    if len(chorus_indices) < 2:
        # Need at least 2 choruses to compare
        lyrics = [t.text for t in transcriptions]
        return boundaries, labels, lyrics

    # Find where the chorus phrase starts in each chorus section
    chorus_phrase_starts = {}
    for section_idx, phrase_start in phrase_locations:
        if labels[section_idx] == 'chorus':
            if section_idx not in chorus_phrase_starts:
                chorus_phrase_starts[section_idx] = phrase_start

    # Check if any chorus has the phrase starting significantly after the section start
    new_boundaries = list(boundaries)
    new_labels = list(labels)
    insertions = []  # (position, boundary_time, label) to insert

    for section_idx in sorted(chorus_phrase_starts.keys()):
        phrase_start = chorus_phrase_starts[section_idx]
        section_start = boundaries[section_idx]

        # If phrase starts more than 5 seconds after section start, split
        offset = phrase_start - section_start
        if offset > 5.0:
            # This section has non-chorus content before the actual chorus
            # We should split it
            insertions.append((section_idx, phrase_start, 'chorus'))
            # Mark the original section as something else (verse or pre-chorus)
            new_labels[section_idx] = 'pre-chorus'

    # Apply insertions (in reverse order to maintain indices)
    for position, boundary_time, label in reversed(insertions):
        # Insert new boundary after the current position
        insert_idx = position + 1
        new_boundaries.insert(insert_idx, boundary_time)
        new_labels.insert(insert_idx, label)

    # Regenerate lyrics for the new sections
    new_lyrics = []
    for i, transcription in enumerate(transcriptions):
        if i < len(new_boundaries):
            section_start = new_boundaries[i]
            section_end = new_boundaries[i + 1] if i + 1 < len(new_boundaries) else duration

            # Find words within this section
            section_words = [
                w for w in transcription.words
                if w.start >= section_start - 0.5 and w.start < section_end
            ]
            section_text = " ".join(w.word for w in section_words).strip()
            new_lyrics.append(section_text)

    # Handle any remaining sections
    while len(new_lyrics) < len(new_boundaries):
        new_lyrics.append("")

    return new_boundaries, new_labels, new_lyrics


def transcribe_full_audio_with_timestamps(
    vocals: np.ndarray,
    sr: int = 22050,
) -> TranscriptionResult:
    """
    Transcribe the full vocals track with word-level timestamps.

    Args:
        vocals: Isolated vocals audio
        sr: Sample rate

    Returns:
        TranscriptionResult with full text and all word timings
    """
    return transcribe_audio_with_timestamps(vocals, sr, time_offset=0.0)


def extract_vocals_and_lyrics(
    audio_path: str,
    boundaries: List[float],
    duration: float,
    use_source_separation: bool = True,
) -> Tuple[List[str], np.ndarray]:
    """
    Full pipeline: isolate vocals, transcribe sections, compute lyric similarity.

    Args:
        audio_path: Path to original audio file
        boundaries: Section boundary times
        duration: Total audio duration
        use_source_separation: Whether to use Demucs for vocal isolation

    Returns:
        Tuple of (lyrics_per_section, lyric_similarity_matrix)
    """
    if use_source_separation:
        try:
            from .source_separator import get_vocals
            vocals = get_vocals(audio_path)

            if len(vocals) > 0:
                # Transcribe each section from isolated vocals
                lyrics = transcribe_sections(vocals, boundaries, duration, sr=22050)
            else:
                # Fallback to direct transcription
                lyrics = _transcribe_sections_direct(audio_path, boundaries, duration)
        except ImportError as e:
            print(f"Warning: Source separation unavailable ({e}). Falling back to direct transcription.")
            lyrics = _transcribe_sections_direct(audio_path, boundaries, duration)
        except Exception as e:
            print(f"Warning: Vocal separation failed: {e}. Falling back to direct transcription.")
            lyrics = _transcribe_sections_direct(audio_path, boundaries, duration)
    else:
        lyrics = _transcribe_sections_direct(audio_path, boundaries, duration)

    # Compute lyric similarity
    similarity = compute_lyric_similarity(lyrics)

    return lyrics, similarity


def extract_vocals_lyrics_and_refine_boundaries(
    audio_path: str,
    boundaries: List[float],
    labels: List[str],
    duration: float,
    use_source_separation: bool = True,
) -> Tuple[List[float], List[str], List[str], np.ndarray]:
    """
    Full pipeline with boundary refinement based on repeated lyrics.

    Transcribes audio with word-level timestamps, finds repeated phrases
    (likely chorus hooks), and refines section boundaries so that choruses
    start at the same lyrical content.

    Args:
        audio_path: Path to original audio file
        boundaries: Initial section boundary times
        labels: Initial section labels
        duration: Total audio duration
        use_source_separation: Whether to use Demucs for vocal isolation

    Returns:
        Tuple of (refined_boundaries, refined_labels, lyrics_per_section, lyric_similarity_matrix)
    """
    vocals = None

    if use_source_separation:
        try:
            from .source_separator import get_vocals
            vocals = get_vocals(audio_path)
            if len(vocals) == 0:
                vocals = None
        except ImportError as e:
            print(f"Warning: Source separation unavailable ({e}).")
        except Exception as e:
            print(f"Warning: Vocal separation failed: {e}.")

    if vocals is not None:
        try:
            # Transcribe full audio with word-level timestamps
            full_transcription = transcribe_full_audio_with_timestamps(vocals, sr=22050)

            # Create per-section transcriptions from the full transcription
            section_transcriptions = []
            for i, start_time in enumerate(boundaries):
                end_time = boundaries[i + 1] if i + 1 < len(boundaries) else duration

                # Extract words for this section
                section_words = [
                    w for w in full_transcription.words
                    if w.start >= start_time - 0.5 and w.start < end_time
                ]
                section_text = " ".join(w.word for w in section_words).strip()
                section_transcriptions.append(TranscriptionResult(
                    text=section_text,
                    words=section_words,
                ))

            # Refine boundaries based on repeated phrases
            refined_boundaries, refined_labels, refined_lyrics = refine_boundaries_with_phrases(
                boundaries,
                labels,
                section_transcriptions,
                duration,
            )

            # Recompute lyrics for refined sections from full transcription
            final_lyrics = []
            for i, start_time in enumerate(refined_boundaries):
                end_time = refined_boundaries[i + 1] if i + 1 < len(refined_boundaries) else duration
                section_words = [
                    w for w in full_transcription.words
                    if w.start >= start_time - 0.5 and w.start < end_time
                ]
                section_text = " ".join(w.word for w in section_words).strip()
                final_lyrics.append(section_text)

            similarity = compute_lyric_similarity(final_lyrics)
            return refined_boundaries, refined_labels, final_lyrics, similarity

        except Exception as e:
            print(f"Warning: Word-level transcription failed ({e}). Using basic transcription.")

    # Fallback: use basic transcription without refinement
    lyrics, similarity = extract_vocals_and_lyrics(
        audio_path, boundaries, duration, use_source_separation=False
    )
    return boundaries, labels, lyrics, similarity


def _transcribe_sections_direct(
    audio_path: str,
    boundaries: List[float],
    duration: float,
) -> List[str]:
    """
    Transcribe sections directly from mixed audio (fallback).

    Args:
        audio_path: Path to audio file
        boundaries: Section boundary times
        duration: Total audio duration

    Returns:
        List of transcribed lyrics
    """
    lyrics = []

    for i, start_time in enumerate(boundaries):
        if i + 1 < len(boundaries):
            end_time = boundaries[i + 1]
        else:
            end_time = duration

        if end_time - start_time < 3.0:
            lyrics.append("")
            continue

        try:
            section_lyrics = transcribe_audio_file(audio_path, start_time, end_time)
            lyrics.append(section_lyrics)
        except Exception as e:
            print(f"Warning: Failed to transcribe section {i}: {e}")
            lyrics.append("")

    return lyrics
