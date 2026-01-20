"""
MixMatch: Extract BPM, sections, and keys from local audio files.

A lightweight library for DJ-centric audio analysis using signal processing.
No external APIs required - fully local analysis using librosa.
"""

__version__ = "0.2.0"
__author__ = "Weston Carpenter"
__email__ = "wc.cc96@gmail.com"

from mixmatch.models import ExtractedAudio, AudioSection
from mixmatch.audio_extractor import extract

__all__ = [
    "ExtractedAudio",
    "AudioSection",
    "extract",
]
