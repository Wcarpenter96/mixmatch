"""
MixMatch: Extract BPM, sections, and keys from local audio files.

A lightweight library for DJ-centric audio analysis using signal processing.
No external APIs required - fully local analysis using librosa.

Features:
- BPM detection
- Section detection (intro, verse, pre-chorus, chorus, bridge, outro)
- Key detection (per-song and per-section)
- Lyric transcription using Whisper
- Source separation using Demucs (vocals, drums, bass, other)
"""

__version__ = "0.3.0"
__author__ = "Weston Carpenter"
__email__ = "wc.cc96@gmail.com"

from mixmatch.models import ExtractedAudio, AudioSection
from mixmatch.audio_extractor import extract

# Advanced imports for users who want individual components
from mixmatch.key_detector import detect_key, detect_section_key
from mixmatch.source_separator import (
    separate_stems,
    get_vocals,
    get_drums,
    detect_drum_drops,
)

__all__ = [
    # Main API
    "ExtractedAudio",
    "AudioSection",
    "extract",
    # Key detection
    "detect_key",
    "detect_section_key",
    # Source separation
    "separate_stems",
    "get_vocals",
    "get_drums",
    "detect_drum_drops",
]
