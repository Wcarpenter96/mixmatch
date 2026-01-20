# MixMatch

> Extract BPM, sections, and keys from local audio files using signal processing. No external APIs required.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

MixMatch is a lightweight Python library for DJ-centric audio analysis. It extracts musical metadata directly from audio files using librosa and signal processing—no Spotify API or external services required.

## Features

✨ **Local Audio Analysis**
- Extract BPM (Beats Per Minute) using librosa beat tracking
- Detect song sections (intro, verse, chorus, outro) with energy + chroma analysis
- Identify musical keys per section using harmonic analysis
- Process any audio format librosa supports: MP3, WAV, FLAC, OGG, etc.

🚀 **Fast & Lightweight**
- ~150-300ms per track on CPU
- No API calls, no network latency
- Minimal dependencies: librosa, numpy, soundfile, click

🎵 **DJ-Centric Design**
- 8-bar phrase grid (DJ standard: 32 beats per mixing unit)
- Chord pattern detection and matching (24 major/minor templates)
- Chorus detection via repetition analysis
- Beat-synchronous chroma for phase-accurate key detection

🔓 **Open & Private**
- 100% offline operation
- No data sent to external services
- Full algorithmic control
- MIT licensed

## Installation

```bash
pip install mixmatch
```

Or from source:

```bash
git clone https://github.com/westoncarpenter/mixmatch.git
cd mixmatch
pip install -e .
```

## Quick Start

### Python API

```python
from mixmatch import extract

# Extract metadata from an audio file
result = extract("song.mp3")

print(f"BPM: {result['bpm']}")
print(f"Sections: {result['sections']}")

# Output:
# BPM: 124
# Sections: [
#   {'label': 'intro', 'start': 0.0, 'key': 'Am'},
#   {'label': 'verse', 'start': 32.0, 'key': 'Am'},
#   {'label': 'chorus', 'start': 64.0, 'key': 'C'},
#   {'label': 'outro', 'start': 96.0, 'key': 'Am'}
# ]
```

### Command Line

```bash
# Extract and display
mixmatch extract-cmd song.mp3

# Output as JSON
mixmatch extract-cmd song.mp3 --json
```

## Output Format

```python
{
    "bpm": 124,                          # Integer BPM
    "sections": [
        {
            "label": "intro",            # Section type: intro/verse/chorus/outro
            "start": 0.0,               # Start time in seconds
            "key": "Am"                 # Musical key (C, C#, D, ..., B, Cm, C#m, etc.)
        },
        # ... more sections
    ]
}
```

## How It Works

### Audio Processing Pipeline

1. **Load Audio** - Read MP3/WAV/FLAC using librosa
2. **Beat Detection** - Librosa beat tracking to find beat grid
3. **BPM Estimation** - Calculate tempo from beat frames
4. **Phrase Grid** - Build DJ-standard 8-bar phrases (32 beats)
5. **Chroma Analysis** - Extract beat-synchronous chroma (12-dimensional)
6. **Chord Detection** - Match chroma against 24 chord templates
7. **Energy Analysis** - Compute mel-spectrogram energy contours
8. **Chorus Detection** - Find repetitive high-energy sections
9. **Section Labeling** - Classify as intro/verse/chorus/outro
10. **Key Detection** - Extract dominant harmonic content per section

### Key Algorithm Details

**BPM Detection**
- Uses `librosa.beat.beat_track()` with dynamic programming
- Robust to tempo variations and syncopation

**Section Detection**
- Energy-based: differentiates chorus (high energy) from verse (lower)
- Chroma-based: detects harmonic stability for repetitive patterns
- Phrase-grid aligned: respects DJ mixing structure (8-bar phrases)

**Key Detection**
- Beat-synchronous CQT chroma (phase-accurate)
- Per-section key extraction (songs often modulate)
- Template-based chord matching (no ML models needed)

**Chorus Detection**
- Self-similarity matrix of chroma vectors
- Looks for diagonal patterns (repetition signature)
- Typically identifies first high-energy repetitive section after intro

## API Reference

### `extract(audio_path: str) -> Dict`

Extract all metadata from an audio file.

**Parameters:**
- `audio_path` (str): Path to audio file

**Returns:**
```python
{
    "bpm": int,
    "sections": [
        {"label": str, "start": float, "key": str},
        ...
    ]
}
```

**Raises:**
- `FileNotFoundError`: If audio file doesn't exist
- `ValueError`: If librosa fails to decode or analyze

**Example:**
```python
from mixmatch import extract

result = extract("track.mp3")
bpm = result["bpm"]
for section in result["sections"]:
    print(f"{section['label']}: {section['key']}")
```

## Data Models

### `ExtractedAudio`

```python
from mixmatch import ExtractedAudio, AudioSection

@dataclass
class ExtractedAudio:
    file_path: str              # Path to audio file
    bpm: int                    # Beats per minute
    sections: List[AudioSection]  # Detected sections
```

### `AudioSection`

```python
@dataclass
class AudioSection:
    label: str      # "intro", "verse", "chorus", "outro"
    start: float    # Start time in seconds
    key: str        # Musical key (e.g., "Am", "C", "G")
```

## Performance

| Metric | Value |
|--------|-------|
| Load time | ~2-3s (first run, librosa caching) |
| Process time | ~150-300ms per track |
| Memory | ~100-200 MB per 3-4 min track |
| CPU | Single-threaded, any modern CPU |

## Supported Audio Formats

librosa supports:
- MP3
- WAV / AIFF
- FLAC
- OGG Vorbis
- OPUS
- And more via ffmpeg

## Requirements

- Python 3.10+
- librosa >= 0.10.0
- numpy >= 1.24.0
- soundfile >= 0.12.0
- click >= 8.1.0

## Development

```bash
# Clone and create virtual environment
git clone https://github.com/westoncarpenter/mixmatch.git
cd mixmatch
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Code formatting
black mixmatch/
flake8 mixmatch/
```

## Use Cases

- 🎧 DJ Mix Preparation - Find compatible tracks for mashups
- 📊 Music Analysis - Extract structural information from songs
- 🎵 Playlist Generation - Organize tracks by key and tempo
- 🤖 Music Information Retrieval - Feed metadata to ML models
- 🎼 Harmonic Analysis - Study song structure and composition

## Limitations

- **No Lyrics** - Works only with audio, not metadata
- **Genre-Agnostic** - Accuracy varies by musical style
- **Mono Processing** - Converts stereo to mono internally
- **No Timing Quantization** - Section boundaries may not align to grid

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details

## Why MixMatch?

**Before (Spotify API Approach):**
- ❌ Rate limited
- ❌ API changes/deprecations
- ❌ Quota system
- ❌ Privacy concerns
- ❌ Cost per request

**After (Local Processing):**
- ✅ No rate limits
- ✅ Fully offline
- ✅ Complete algorithmic control
- ✅ Private by default
- ✅ Zero API costs

## Examples

### Extract and Process a Directory

```python
from pathlib import Path
from mixmatch import extract

audio_dir = Path("music_library")
for audio_file in audio_dir.glob("*.mp3"):
    result = extract(str(audio_file))
    print(f"{audio_file.name}: {result['bpm']} BPM")
```

### Find Compatible Key Signatures

```python
from mixmatch import extract

song1 = extract("track1.mp3")
song2 = extract("track2.mp3")

key1 = song1["sections"][1]["key"]  # Verse key
key2 = song2["sections"][1]["key"]

if key1 == key2:
    print("Keys match! Good mashup candidate.")
```

### Export as JSON

```python
import json
from mixmatch import extract

result = extract("song.mp3")
with open("metadata.json", "w") as f:
    json.dump(result, f, indent=2)
```

## Troubleshooting

**Q: "ModuleNotFoundError: No module named 'librosa'"**  
A: Install dependencies with `pip install mixmatch` or `pip install -e .` from source.

**Q: Audio file not detected**  
A: Ensure file exists and librosa supports the format. Try converting to MP3 or WAV first.

**Q: Incorrect BPM detected**  
A: Some audio has ambiguous tempo. The algorithm defaults to the most prominent beat. Consider manual adjustment.

**Q: Sections seem wrong**  
A: Section detection is heuristic-based and works best with structured pop/dance music. Electronic and ambient genres may vary.

## Roadmap

- [ ] Multi-threaded batch processing
- [ ] Caching layer for repeated extractions
- [ ] Web API (FastAPI endpoint)
- [ ] Visualization tools
- [ ] Advanced genre classification
- [ ] Confidence scores per detection

## Related Projects

- [librosa](https://librosa.org/) - Audio analysis library
- [essentia](https://essentia.upf.edu/) - Audio analysis framework
- [aubio](https://aubio.org/) - Real-time audio processing

---

**Made with ♪ for DJs, musicians, and audio enthusiasts.**
