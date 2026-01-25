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
) -> List[Tuple[Tuple[str, ...], List[Tuple[int, float, float]]]]:
    """
    Find repeated phrases across sections that could indicate chorus signatures.

    Args:
        transcriptions: List of TranscriptionResult for each section
        min_phrase_length: Minimum words in a phrase
        max_phrase_length: Maximum words in a phrase
        min_occurrences: Minimum times a phrase must appear

    Returns:
        List of (phrase_tuple, [(section_idx, start_time, end_time), ...]) sorted by score
    """
    # Collect all n-grams with their locations (including end times)
    phrase_locations: Dict[Tuple[str, ...], List[Tuple[int, float, float]]] = {}

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
                    phrase_words[-1].end,  # Track end time of phrase
                ))

    # Filter to phrases that appear multiple times and in different sections
    repeated_phrases = []
    for phrase, locations in phrase_locations.items():
        unique_sections = set(loc[0] for loc in locations)
        if len(locations) >= min_occurrences and len(unique_sections) >= min_occurrences:
            repeated_phrases.append((phrase, locations))

    # Sort by a composite score: prefer longer phrases, but also consider frequency
    # Score = occurrences * (1 + phrase_length / 10)
    # This gives a slight boost to longer phrases while still prioritizing frequency
    def phrase_score(item):
        phrase, locations = item
        occurrence_count = len(locations)
        phrase_length = len(phrase)
        return occurrence_count * (1 + phrase_length / 10)

    repeated_phrases.sort(key=phrase_score, reverse=True)

    return repeated_phrases


def refine_boundaries_with_phrases(
    boundaries: List[float],
    labels: List[str],
    transcriptions: List[TranscriptionResult],
    duration: float,
    full_transcription: Optional[TranscriptionResult] = None,
    melodic_phrase_starts: Optional[List[float]] = None,
    min_section_duration: float = 8.0,
) -> Tuple[List[float], List[str], List[str]]:
    """
    Refine section boundaries using repeated phrase detection (lyrics + melody).

    Finds the signature phrase (e.g., "What doesn't kill you makes you stronger")
    wherever it appears in the song, and uses those locations to properly
    align section boundaries so choruses start at the phrase. Also incorporates
    melodic phrase detection for cases where lyrics are unclear.

    Args:
        boundaries: Original section boundary times
        labels: Section labels
        transcriptions: Word-level transcriptions per section
        duration: Total audio duration
        full_transcription: Full song transcription with word timestamps
        melodic_phrase_starts: Start times of repeated melodic phrases
        min_section_duration: Minimum duration for a section

    Returns:
        Tuple of (refined_boundaries, refined_labels, refined_lyrics)
    """
    if len(boundaries) == 0:
        return boundaries, labels, []

    # Find signature phrases (likely chorus hooks) across ALL sections
    # Use longer phrases (4+ words) to get more distinctive hooks
    signature_phrases = find_signature_phrases(transcriptions, min_phrase_length=4, max_phrase_length=8)

    # Get lyric-based phrase starts AND ends
    lyric_phrase_intervals = []  # (start, end) tuples
    if signature_phrases:
        chorus_phrase, phrase_locations = signature_phrases[0]
        # Deduplicate by start time, keeping the longest end time
        start_to_end = {}
        for _, phrase_start, phrase_end in phrase_locations:
            if phrase_start not in start_to_end or phrase_end > start_to_end[phrase_start]:
                start_to_end[phrase_start] = phrase_end
        lyric_phrase_intervals = sorted([(s, e) for s, e in start_to_end.items()])

    lyric_phrase_starts = [s for s, e in lyric_phrase_intervals]

    # Combine lyric and melodic phrase starts
    all_phrase_starts = set(lyric_phrase_starts)

    if melodic_phrase_starts:
        # Add melodic phrase starts, but avoid duplicates that are too close
        for melodic_start in melodic_phrase_starts:
            # Check if there's already a lyric phrase start within 2 seconds
            is_duplicate = any(
                abs(melodic_start - lyric_start) < 2.0
                for lyric_start in lyric_phrase_starts
            )
            if not is_duplicate:
                all_phrase_starts.add(melodic_start)

    all_phrase_starts = sorted(all_phrase_starts)

    if not signature_phrases and not melodic_phrase_starts:
        # No repeated phrases found (lyric or melodic), return original
        lyrics = [t.text for t in transcriptions]
        return boundaries, labels, lyrics

    if len(all_phrase_starts) < 2:
        lyrics = [t.text for t in transcriptions]
        return boundaries, labels, lyrics

    # Estimate chorus duration based on the signature phrase occurrences
    # Use the average gap between phrase start and end, plus some buffer for the full chorus
    chorus_duration_estimate = 30.0  # Default estimate
    if lyric_phrase_intervals:
        phrase_durations = [e - s for s, e in lyric_phrase_intervals]
        avg_phrase_duration = sum(phrase_durations) / len(phrase_durations)
        # Chorus is typically 3-4x the length of the hook phrase
        chorus_duration_estimate = min(45.0, max(15.0, avg_phrase_duration * 4))

    # Build new boundaries based on where the signature phrase appears
    new_boundaries = []
    new_labels = []

    # Start with the first boundary (usually intro)
    if boundaries[0] < all_phrase_starts[0] - min_section_duration:
        new_boundaries.append(boundaries[0])
        new_labels.append(labels[0] if labels[0] in ['intro', 'verse'] else 'intro')

    # Track which original boundaries we've used
    used_boundaries = set()

    for idx, phrase_start in enumerate(all_phrase_starts):
        # Find the closest original boundary before this phrase
        prev_boundary = None
        prev_idx = None
        for i, b in enumerate(boundaries):
            if b < phrase_start:
                prev_boundary = b
                prev_idx = i
            else:
                break

        # If phrase starts significantly after a boundary, we may need an intermediate section
        if prev_boundary is not None and prev_idx not in used_boundaries:
            gap = phrase_start - prev_boundary
            if gap > min_section_duration:
                # Add the previous section
                new_boundaries.append(prev_boundary)
                orig_label = labels[prev_idx] if prev_idx < len(labels) else 'verse'
                # If this section runs into a chorus, it's probably verse or pre-chorus
                if orig_label == 'chorus':
                    new_labels.append('verse')
                else:
                    new_labels.append(orig_label)
                used_boundaries.add(prev_idx)

        # Add the chorus boundary at the phrase start
        new_boundaries.append(phrase_start)
        new_labels.append('chorus')

        # Check if we need a verse boundary after this chorus ends
        # Estimate where the chorus ends based on duration estimate
        estimated_chorus_end = phrase_start + chorus_duration_estimate

        # Determine next chorus start (if any)
        next_phrase_start = all_phrase_starts[idx + 1] if idx + 1 < len(all_phrase_starts) else duration

        # If there's a significant gap between estimated chorus end and next chorus,
        # add a verse boundary
        gap_after_chorus = next_phrase_start - estimated_chorus_end
        if gap_after_chorus > min_section_duration:
            new_boundaries.append(estimated_chorus_end)
            new_labels.append('verse')

    # Add any remaining sections after the last chorus phrase
    last_phrase = all_phrase_starts[-1] if all_phrase_starts else 0
    for i, b in enumerate(boundaries):
        if b > last_phrase + min_section_duration and i not in used_boundaries:
            new_boundaries.append(b)
            new_labels.append(labels[i] if i < len(labels) else 'outro')

    # Ensure boundaries are sorted and unique
    combined = sorted(set(zip(new_boundaries, new_labels)), key=lambda x: x[0])
    if combined:
        new_boundaries, new_labels = zip(*combined)
        new_boundaries = list(new_boundaries)
        new_labels = list(new_labels)
    else:
        new_boundaries = list(boundaries)
        new_labels = list(labels)

    # Filter out boundaries that would create sections shorter than min_section_duration
    if len(new_boundaries) > 1:
        filtered_boundaries = [new_boundaries[0]]
        filtered_labels = [new_labels[0]]

        for i in range(1, len(new_boundaries)):
            gap = new_boundaries[i] - filtered_boundaries[-1]
            if gap >= min_section_duration:
                # Section is long enough, add this boundary
                filtered_boundaries.append(new_boundaries[i])
                filtered_labels.append(new_labels[i])
            else:
                # Section too short - keep the one with the "stronger" label
                # Prefer chorus over verse, verse over intro/outro
                label_priority = {'chorus': 3, 'bridge': 2, 'verse': 1, 'pre-chorus': 1, 'intro': 0, 'outro': 0}
                current_priority = label_priority.get(filtered_labels[-1], 0)
                new_priority = label_priority.get(new_labels[i], 0)

                if new_priority > current_priority:
                    # Replace the previous boundary with this one (keep the later, stronger-labeled one)
                    filtered_labels[-1] = new_labels[i]
                # Otherwise keep the existing boundary and skip this one

        new_boundaries = filtered_boundaries
        new_labels = filtered_labels

    # Regenerate lyrics for the new sections using full transcription if available
    new_lyrics = []

    # Collect all words from all section transcriptions
    all_words = []
    for transcription in transcriptions:
        all_words.extend(transcription.words)

    # Use full transcription if provided, otherwise use collected words
    if full_transcription is not None:
        all_words = full_transcription.words

    # Sort words by start time
    all_words = sorted(all_words, key=lambda w: w.start)

    for i in range(len(new_boundaries)):
        section_start = new_boundaries[i]
        section_end = new_boundaries[i + 1] if i + 1 < len(new_boundaries) else duration

        # Find words within this section
        section_words = [
            w for w in all_words
            if w.start >= section_start - 0.5 and w.start < section_end
        ]
        section_text = " ".join(w.word for w in section_words).strip()
        new_lyrics.append(section_text)

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


def find_melodic_phrase_starts(
    vocals: np.ndarray,
    sr: int = 22050,
    min_occurrences: int = 2,
) -> List[float]:
    """
    Find start times of repeated melodic phrases.

    Uses Basic Pitch to analyze vocal melody and identify recurring
    melodic patterns that likely indicate chorus/hook sections.

    Args:
        vocals: Isolated vocals audio
        sr: Sample rate
        min_occurrences: Minimum times a phrase must repeat

    Returns:
        List of start times where repeated melodic phrases begin
    """
    try:
        from .melody_analyzer import find_repeated_melodic_phrases
        repeated_phrases = find_repeated_melodic_phrases(vocals, sr)

        if not repeated_phrases:
            return []

        # Get all start times from the most common repeated phrase
        # (likely the chorus hook melody)
        most_common_locations, _ = repeated_phrases[0]
        phrase_starts = sorted([start for start, end in most_common_locations])

        return phrase_starts
    except Exception as e:
        print(f"Warning: Melodic phrase detection failed ({e}).")
        return []


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
    Full pipeline with boundary refinement based on repeated lyrics and melody.

    Transcribes audio with word-level timestamps, analyzes melody patterns,
    finds repeated phrases (both lyrical and melodic), and refines section
    boundaries so that choruses start at the same content.

    Args:
        audio_path: Path to original audio file
        boundaries: Initial section boundary times
        labels: Initial section labels
        duration: Total audio duration
        use_source_separation: Whether to use Demucs for vocal isolation

    Returns:
        Tuple of (refined_boundaries, refined_labels, lyrics_per_section, similarity_matrix)
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

            # Find repeated melodic phrases for additional chorus detection
            melodic_phrase_starts = find_melodic_phrase_starts(vocals, sr=22050)

            # Refine boundaries based on repeated phrases (lyrics + melody)
            refined_boundaries, refined_labels, refined_lyrics = refine_boundaries_with_phrases(
                boundaries,
                labels,
                section_transcriptions,
                duration,
                full_transcription=full_transcription,
                melodic_phrase_starts=melodic_phrase_starts,
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

            # Compute lyric similarity
            lyric_similarity = compute_lyric_similarity(final_lyrics)

            # Compute melody similarity and combine with lyric similarity
            try:
                from .melody_analyzer import compute_section_melody_similarity
                melody_similarity = compute_section_melody_similarity(
                    vocals, refined_boundaries, duration, sr=22050
                )
                # Combine: weighted average of lyric and melody similarity
                similarity = 0.6 * lyric_similarity + 0.4 * melody_similarity
            except Exception as e:
                print(f"Warning: Melody analysis failed ({e}). Using lyrics only.")
                similarity = lyric_similarity

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
