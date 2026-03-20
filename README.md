# Slave to the Gods

A dark fantasy novel about Ra, an alien stranded on Earth for eight thousand years, and Issah, the human woman whose blood is the key to his survival—and his humanity.

## Audiobook Generation

This repository includes a complete pipeline for converting the manuscript into a professional audiobook using local AI text-to-speech.

### Overview

The audiobook is generated using **Qwen3-TTS** (1.7B parameter model) with two modes:

1. **VoiceDesign** (default) — Creates a custom narrator voice from a text description
2. **Voice Clone** — Uses your own narrator voice recording for consistent, realistic voice throughout

**VoiceDesign narrator voice:** *An epic older man's voice, deep and resonant with gravitas, warm yet commanding, measured cadence like a wise elder recounting ancient tales across centuries.*

### Voice Clone Mode (Recommended)

For the most natural-sounding audiobook, record yourself (or a voice actor) reading for 10-30 seconds, then use that as the reference. The TTS will clone your narrator's voice for the entire book.

**Recording tips:**
- Use a clean, quiet recording
- Speak in the tone and pace you want for the audiobook
- 10-30 seconds minimum, longer is better
- Save as WAV format (24kHz mono recommended)

### Sample Narrator Reference

A sample narrator reference is included at `narrator.wav` (~10 seconds, 24kHz mono). This was used to test and validate the voice cloning pipeline.

**Process used to validate:**
1. Record narrator reference (10-30 seconds, WAV format)
2. Test cloning with `test_narrator_clone.py` or `--test --narrator narrator.wav`
3. Compare original and clone using `voice_compare.wav`
4. If satisfied, run full generation with `--narrator narrator.wav`

### Requirements

- **OS:** Linux (tested on Ubuntu with kernel 6.17)
- **GPU:** AMD Radeon RX 7900 XTX (24GB VRAM) with ROCm
- **ROCm:** 7.2.0+ installed at `/opt/rocm-7.2.0`
- **Python:** 3.12+ via conda
- **Storage:** ~5GB for models + output files

### Software Setup

1. **Install ROCm PyTorch** (must be done before qwen-tts):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
```

2. **Install Qwen3-TTS**:
```bash
pip install qwen-tts soundfile
```

3. **Verify GPU detection**:
```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True AMD Radeon RX 7900 XTX
```

### Usage

#### Quick Test (100 words)
Test voice before processing the full manuscript:
```bash
# VoiceDesign mode (text description)
python3 manuscript_to_audiobook.py --test

# Voice Clone mode (use your narrator reference)
python3 manuscript_to_audiobook.py --test --narrator narrator.wav
```
Output: `audiobook/test_sample.wav`

#### Generate Full Audiobook
Process the entire manuscript:
```bash
# VoiceDesign mode
python3 manuscript_to_audiobook.py

# Voice Clone mode (recommended for consistent voice)
python3 manuscript_to_audiobook.py --narrator narrator.wav
```
Output: `audiobook/slave-to-the-gods-audiobook.wav`

#### Resume from Interruption
If interrupted, resume where you left off:
```bash
python3 manuscript_to_audiobook.py --resume
```

#### Process Specific Chapters
Generate only certain chapters:
```bash
python3 manuscript_to_audiobook.py --chapters 1,2,3
```

### Output Structure

```
audiobook/
├── test_sample.wav                    # Test output (if --test)
├── chapter_001_Introduction.wav     # Individual chapter files
├── chapter_002_The_Blessed.wav
├── ...
├── slave-to-the-gods-audiobook.wav  # Final merged audiobook
└── .checkpoint.json                   # Resume checkpoint (auto-generated)
```

### Processing Details

- **Modes:** VoiceDesign (text description) or Voice Clone (reference audio)
- **Chunk size:** 150 words per chunk (optimal for TTS quality)
- **Sample rate:** 24kHz, 16-bit PCM WAV
- **Estimated duration:** ~3.5-4 hours of audio
- **Processing time:** 4-6 hours on 7900 XTX
- **Output format:** Individual chapter files + final merged audiobook
- **Checkpointing:** Progress saved after every chunk (resumable with `--resume`)

### How It Works

1. **Parse manuscript** — Extracts chapters from markdown
2. **Clean text** — Removes markdown formatting, normalizes whitespace
3. **Smart chunking** — Splits at sentence boundaries (no mid-sentence cuts)
4. **TTS generation** — Each chunk processed through Qwen3-TTS
   - VoiceDesign mode: Uses text description to generate voice
   - Voice Clone mode: Uses x-vector embedding from your narrator reference
5. **Checkpointing** — Saves progress after every chunk (resumable)
6. **Chapter merge** — Combines chunks into chapter files
7. **Final merge** — Combines all chapters into complete audiobook

### Troubleshooting

**MIOpen workspace warnings** — Harmless ROCm noise, does not affect output.

**flash-attn warning** — Flash attention not available for ROCm. Falls back to standard PyTorch attention automatically.

**Out of memory** — Reduce `CHUNK_SIZE` in the script (default: 150 words).

**Model loading to RAM instead of VRAM** — Reinstall ROCm PyTorch:
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
```

### Technical Stack

| Component | Tool |
|-----------|------|
| TTS Model | Qwen3-TTS 1.7B (Base or VoiceDesign) |
| GPU Compute | AMD ROCm 7.2.0 |
| Voice Clone | x-vector embedding (x_vector_only_mode=True) |
| Audio Processing | soundfile |
| Checkpointing | JSON-based resume system |

### Manuscript

The full manuscript is in `manuscript/slave-to-the-gods.md` (~140KB, ~50,000 words).

## Story

**Genre:** Dark Fantasy / Mythological Fiction  
**Themes:** Power and sacrifice, the cost of immortality, love across impossible boundaries, the nature of divinity

### Synopsis

Ra is not a god. He is the last survivor of a dying world, stranded on Earth for eight thousand years, feeding on human blood to survive. His people—the Shetu—came as refugees when their sun began to die. They built temples over their crashed ships, and the humans who found them called them gods.

Issah was born to be fed upon. Raised in the Sacred Cattle caste, she has known since childhood that her blood belongs to the divine. But she is also a spy for the human resistance, gathering intelligence to overthrow the gods who have ruled Egypt since before memory.

When Ra claims her for his personal household, she sees her chance to strike at the heart of divine power. What she doesn't expect is to discover that the most powerful being on Earth is, at his core, a lonely exile who misses his home. Or that the monster she's been taught to hate is capable of love.

As the resistance prepares its final strike, Issah must choose: destroy the system that has enslaved her people, or save the being who has become the only real thing in a world of masks.

## Repository Structure

```
slave-to-the-gods/
├── manuscript/
│   └── slave-to-the-gods.md          # Full manuscript
├── chapters/                          # Chapter breakdowns (if any)
├── story-bible/                       # World-building notes
├── world-building/                    # Setting details
├── assets/                            # Images, covers, etc.
├── narrator.wav                       # Narrator reference for voice cloning
├── test_narrator_clone.py             # Test script for voice cloning
├── manuscript_to_audiobook.py          # Audiobook generation script
└── README.md                          # This file
```

## License

Copyright © 2026. All rights reserved.

---

*"The blood always sang to him."*
