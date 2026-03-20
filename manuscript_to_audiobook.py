#!/usr/bin/env python3
"""
manuscript_to_audiobook.py
Convert Slave to the Gods manuscript to audiobook using Qwen3-TTS VoiceDesign.

Usage:
    python manuscript_to_audiobook.py                    # Process full manuscript
    python manuscript_to_audiobook.py --test             # Process first 100 words only
    python manuscript_to_audiobook.py --resume            # Resume from last checkpoint
    python manuscript_to_audiobook.py --chapters 1,2,3    # Process specific chapters only
"""

import argparse
import os
import re
import json
import sys
from pathlib import Path
from datetime import datetime

# Configuration
MANUSCRIPT_PATH = Path(__file__).parent / "manuscript" / "slave-to-the-gods.md"
OUTPUT_DIR = Path(__file__).parent / "audiobook"
CHECKPOINT_FILE = OUTPUT_DIR / ".checkpoint.json"
CHUNK_SIZE = 150  # words per chunk (optimal for TTS quality)
SAMPLE_RATE = 24000

# Voice description for epic older narrator
VOICE_DESCRIPTION = (
    "An epic older man's voice, deep and resonant with gravitas, "
    "warm yet commanding, measured cadence like a wise elder "
    "recounting ancient tales across centuries. "
    "Perfect for narrating dark fantasy and mythological fiction."
)

def log(msg):
    """Print with timestamp."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def split_into_chunks(text, chunk_size=CHUNK_SIZE):
    """Split text into chunks at sentence boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        word_count = len(sentence.split())
        
        if current_word_count + word_count > chunk_size and current_chunk:
            # Save current chunk and start new one
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_word_count = word_count
        else:
            current_chunk.append(sentence)
            current_word_count += word_count
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def extract_chapters(text):
    """Extract chapters from markdown manuscript."""
    # Split on chapter markers (## or #)
    chapter_pattern = r'(?:^|\n)(#{1,2}\s+[^\n]+)'
    parts = re.split(chapter_pattern, text)
    
    chapters = []
    current_title = "Introduction"
    current_content = ""
    
    for i, part in enumerate(parts):
        if part.startswith('#'):
            # Save previous chapter if exists
            if current_content.strip():
                chapters.append({
                    'title': current_title,
                    'content': current_content.strip()
                })
            current_title = part.lstrip('#').strip()
            current_content = ""
        else:
            current_content += part
    
    # Don't forget the last chapter
    if current_content.strip():
        chapters.append({
            'title': current_title,
            'content': current_content.strip()
        })
    
    return chapters

def clean_text(text):
    """Clean markdown formatting from text."""
    # Remove markdown headers
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    # Remove markdown formatting
    text = re.sub(r'\*\*|__|\*|_|`', '', text)
    # Remove links but keep text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Remove horizontal rules
    text = re.sub(r'^---+$', '', text, flags=re.MULTILINE)
    # Normalize whitespace
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def save_checkpoint(chapter_idx, chunk_idx, total_chunks):
    """Save progress checkpoint."""
    checkpoint = {
        'chapter_idx': chapter_idx,
        'chunk_idx': chunk_idx,
        'total_chunks': total_chunks,
        'timestamp': datetime.now().isoformat()
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f)
    log(f"Checkpoint saved: Chapter {chapter_idx + 1}, Chunk {chunk_idx + 1}/{total_chunks}")

def load_checkpoint():
    """Load progress checkpoint."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return None

def generate_audio_chunk(text, output_path, model):
    """Generate audio for a single text chunk."""
    import soundfile as sf
    
    audios, sr = model.generate_voice_design(
        text=text,
        instruct=VOICE_DESCRIPTION,
        language="english",
    )
    
    sf.write(str(output_path), audios[0], sr)
    return output_path

def merge_audio_files(audio_files, output_path):
    """Merge multiple WAV files into one."""
    import soundfile as sf
    import numpy as np
    
    combined = []
    sample_rate = None
    
    for audio_file in audio_files:
        data, sr = sf.read(str(audio_file))
        if sample_rate is None:
            sample_rate = sr
        combined.append(data)
        # Add 0.5 second silence between chunks
        combined.append(np.zeros(int(sr * 0.5)))
    
    # Remove trailing silence
    if combined:
        combined = combined[:-1]
    
    final_audio = np.concatenate(combined)
    sf.write(str(output_path), final_audio, sample_rate)
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Convert manuscript to audiobook")
    parser.add_argument("--test", action="store_true", help="Process only first 100 words")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--chapters", type=str, help="Process specific chapters (e.g., 1,2,3)")
    parser.add_argument("--output", type=str, default="slave-to-the-gods-audiobook.wav",
                        help="Output filename")
    args = parser.parse_args()
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Read manuscript
    log(f"Reading manuscript: {MANUSCRIPT_PATH}")
    with open(MANUSCRIPT_PATH) as f:
        manuscript = f.read()
    
    # Extract chapters
    chapters = extract_chapters(manuscript)
    log(f"Found {len(chapters)} chapters")
    
    # Filter chapters if specified
    if args.chapters:
        chapter_nums = [int(x.strip()) - 1 for x in args.chapters.split(',')]
        chapters = [chapters[i] for i in chapter_nums if i < len(chapters)]
        log(f"Processing {len(chapters)} selected chapters")
    
    # Test mode: only first 100 words
    if args.test:
        log("TEST MODE: Processing first 100 words only")
        full_text = clean_text(manuscript)
        words = full_text.split()[:100]
        test_text = ' '.join(words)
        
        log("Loading Qwen3-TTS VoiceDesign model...")
        from qwen_tts import Qwen3TTSModel
        model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            device_map="cuda:0"
        )
        
        output_path = OUTPUT_DIR / "test_sample.wav"
        log(f"Generating test audio: {output_path}")
        generate_audio_chunk(test_text, output_path, model)
        log(f"Test complete! Saved to: {output_path}")
        return
    
    # Load checkpoint if resuming
    checkpoint = load_checkpoint() if args.resume else None
    start_chapter = checkpoint['chapter_idx'] if checkpoint else 0
    start_chunk = checkpoint['chunk_idx'] + 1 if checkpoint else 0
    
    if checkpoint:
        log(f"Resuming from Chapter {start_chapter + 1}, Chunk {start_chunk}")
    
    # Load TTS model
    log("Loading Qwen3-TTS VoiceDesign model (1.7B)...")
    from qwen_tts import Qwen3TTSModel
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        device_map="cuda:0"
    )
    log("Model loaded successfully")
    
    # Process each chapter
    all_chapter_files = []
    
    for chapter_idx, chapter in enumerate(chapters[start_chapter:], start=start_chapter):
        log(f"\n{'='*60}")
        log(f"Processing Chapter {chapter_idx + 1}: {chapter['title']}")
        log(f"{'='*60}")
        
        # Clean and chunk the chapter
        clean_content = clean_text(chapter['content'])
        chunks = split_into_chunks(clean_content)
        log(f"Split into {len(chunks)} chunks")
        
        # Create chapter output directory
        chapter_dir = OUTPUT_DIR / f"chapter_{chapter_idx + 1:03d}"
        chapter_dir.mkdir(exist_ok=True)
        
        chapter_audio_files = []
        
        # Process chunks
        chunk_start = start_chunk if chapter_idx == start_chapter else 0
        
        for chunk_idx, chunk in enumerate(chunks[chunk_start:], start=chunk_start):
            chunk_file = chapter_dir / f"chunk_{chunk_idx + 1:04d}.wav"
            
            log(f"  Chunk {chunk_idx + 1}/{len(chunks)} ({len(chunk.split())} words)...")
            
            try:
                generate_audio_chunk(chunk, chunk_file, model)
                chapter_audio_files.append(chunk_file)
                
                # Save checkpoint after each chunk
                save_checkpoint(chapter_idx, chunk_idx, len(chunks))
                
            except Exception as e:
                log(f"ERROR processing chunk {chunk_idx + 1}: {e}")
                log("Checkpoint saved. Resume with --resume flag.")
                raise
        
        # Merge chapter audio
        if chapter_audio_files:
            chapter_output = OUTPUT_DIR / f"chapter_{chapter_idx + 1:03d}_{chapter['title'][:30].replace(' ', '_')}.wav"
            log(f"Merging chapter audio: {chapter_output.name}")
            merge_audio_files(chapter_audio_files, chapter_output)
            all_chapter_files.append(chapter_output)
            log(f"Chapter {chapter_idx + 1} complete!")
        
        # Reset start_chunk for subsequent chapters
        start_chunk = 0
    
    # Final merge: all chapters into one audiobook
    if all_chapter_files:
        final_output = OUTPUT_DIR / args.output
        log(f"\n{'='*60}")
        log(f"Creating final audiobook: {final_output.name}")
        log(f"{'='*60}")
        merge_audio_files(all_chapter_files, final_output)
        
        # Clean up checkpoint
        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()
        
        # Print summary
        log(f"\n{'='*60}")
        log("AUDIOBOOK GENERATION COMPLETE!")
        log(f"{'='*60}")
        log(f"Output: {final_output}")
        log(f"Chapters: {len(all_chapter_files)}")
        
        # Calculate approximate duration
        import soundfile as sf
        total_samples = 0
        for f in all_chapter_files:
            data, sr = sf.read(str(f))
            total_samples += len(data)
        duration_min = total_samples / sr / 60
        log(f"Approximate duration: {duration_min:.1f} minutes ({duration_min/60:.1f} hours)")

if __name__ == "__main__":
    main()
