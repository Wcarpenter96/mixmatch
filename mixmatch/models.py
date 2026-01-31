"""
Data models for audio extraction.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AudioSection:
    """A section of audio (intro, verse, chorus, bridge, outro)."""

    label: str  # "intro", "verse", "chorus", "bridge", "outro"
    start_seconds: float  # Start time in seconds
    key: str  # Musical key (e.g., "Am", "C", "G")
    lyrics: Optional[str] = None  # Transcribed lyrics for this section
    detection_method: Optional[dict] = None  # How this section was detected/classified

    @property
    def start(self) -> str:
        """Start time formatted as mm:ss."""
        minutes = int(self.start_seconds // 60)
        seconds = int(self.start_seconds % 60)
        return f"{minutes}:{seconds:02d}"

    def __str__(self) -> str:
        base = f"{self.label} @ {self.start} ({self.start_seconds:.1f}s) in {self.key}"
        if self.lyrics:
            # Show first 50 chars of lyrics
            preview = self.lyrics[:50] + "..." if len(self.lyrics) > 50 else self.lyrics
            base += f' "{preview}"'
        return base


@dataclass
class ExtractedAudio:
    """Extracted features from a local audio file."""
    
    file_path: str
    bpm: int  # Beats per minute
    sections: List[AudioSection]  # Sections with labels and keys
    
    def __str__(self) -> str:
        return f"{self.file_path} ({self.bpm} BPM, {len(self.sections)} sections)"
