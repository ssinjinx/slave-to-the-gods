#!/usr/bin/env python3
"""Test voice cloning with narrator.wav reference using x_vector_only_mode."""
from qwen_tts import Qwen3TTSModel
import soundfile as sf

NARRATOR_WAV = "/home/ssinjin/Music/narrator.wav"
TEST_TEXT = "The old man walked through the darkness, his footsteps echoing against the ancient stone walls."
OUTPUT_PATH = "/home/ssinjin/projects/slave-to-the-gods/audiobook/test_clone.wav"

print("Loading Qwen3-TTS 1.7B Base model...")
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0"
)
print("Model loaded.")

print(f"Loading narrator reference: {NARRATOR_WAV}")
ref_audio, sr = sf.read(NARRATOR_WAV)
print(f"Reference audio: {len(ref_audio)} samples @ {sr}Hz ({len(ref_audio)/sr:.1f}s)")

print(f"\nGenerating voice clone with x_vector_only_mode...")
print(f"Text: {TEST_TEXT}")

audio, sr = model.generate_voice_clone(
    text=TEST_TEXT,
    language="english",
    ref_audio=(ref_audio, sr),
    ref_text=None,
    x_vector_only_mode=True  # Skip ICL, just use voice embedding
)

print(f"Generated {len(audio)} samples @ {sr}Hz ({len(audio)/sr:.1f}s)")
sf.write(OUTPUT_PATH, audio[0] if isinstance(audio, list) else audio, sr)
print(f"\nSaved to: {OUTPUT_PATH}")
