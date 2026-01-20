"""
Data models for audio extraction.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class AudioSection:
    """A section of audio (intro, verse, chorus, outro)."""
    
    label: str  # "intro", "verse", "chorus", "outro"
    start: float  # Start time in seconds
    key: str  # Musical key (e.g., "Am", "C", "G")
    
    def __str__(self) -> str:
        return f"{self.label} @ {self.start:.1f}s in {self.key}"


@dataclass
class ExtractedAudio:
    """Extracted features from a local audio file."""
    
    file_path: str
    bpm: int  # Beats per minute
    sections: List[AudioSection]  # Sections with labels and keys
    
    def __str__(self) -> str:
        return f"{self.file_path} ({self.bpm} BPM, {len(self.sections)} sections)"
