# Slave to the Gods

A dark fantasy novel about Ra, an alien stranded on Earth for eight thousand years, and Issah, the human woman whose blood is the key to his survival—and his humanity.

## Audiobook Generation

This repository includes a complete pipeline for converting the manuscript into a professional audiobook using local AI text-to-speech.

### Overview

The audiobook is generated using **Qwen3-TTS** (1.7B parameter model) with **VoiceDesign** mode, which creates a custom narrator voice from a text description—no voice sample required.

**Narrator Voice:** *An epic older man's voice, deep and resonant with gravitas, warm yet commanding, measured cadence like a wise elder recounting ancient tales across centuries.*

### Requirements

- **OS:** Linux (tested on Ubuntu with kernel 6.17)
- **GPU:** AMD Radeon RX 7900 XTX (24GB VRAM) with ROCm
- **ROCm:** 7.2.0+ installed at `/opt/rocm-7.2.0`
- **Python:** 3.12 via conda
- **Storage:** ~5GB for model + output files

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
Test the voice before processing the full manuscript:
```bash
python3 manuscript_to_audiobook.py --test
```
Output: `audiobook/test_sample.wav`

#### Generate Full Audiobook
Process the entire manuscript:
```bash
python3 manuscript_to_audiobook.py
```
Output: `audiobook/slave-to-the-gods-audiobook.wav`

#### Resume from Interruption
If the process is interrupted, resume where it left off:
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

- **Chunk size:** 150 words per chunk (optimal for TTS quality)
- **Sample rate:** 24kHz, 16-bit PCM WAV
- **Estimated duration:** ~3.5-4 hours of audio
- **Processing time:** 4-6 hours on 7900 XTX
- **Output format:** Individual chapter files + final merged audiobook

### How It Works

1. **Parse manuscript** — Extracts chapters from markdown
2. **Clean text** — Removes markdown formatting, normalizes whitespace
3. **Smart chunking** — Splits at sentence boundaries (no mid-sentence cuts)
4. **TTS generation** — Each chunk processed through Qwen3-TTS VoiceDesign
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
| TTS Model | Qwen3-TTS 1.7B VoiceDesign |
| GPU Compute | AMD ROCm 7.2.0 |
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
├── manuscript_to_audiobook.py         # Audiobook generation script
└── README.md                          # This file
```

## License

Copyright © 2026. All rights reserved.

---

*"The blood always sang to him."*
