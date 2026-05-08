from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.preprocessing.midi_parser import list_midi_files
from src.preprocessing.piano_roll import midi_to_pianoroll, segment_pianoroll
from src.preprocessing.tokenizer import build_tokenizer, midi_to_tokens
RAW_DIR = ROOT / "data" / "raw_midi" / "maestro-v3.0.0"
PROCESSED_ROLLS = ROOT / "data" / "processed" / "rolls"
PROCESSED_TOKENS = ROOT / "data" / "processed" / "tokens"


def main(sample_size: int = 8, seq_len: int = 128) -> None:
    midi_files = list_midi_files(RAW_DIR)
    if not midi_files:
        raise FileNotFoundError(f"No MIDI files found under {RAW_DIR}")

    random.seed(42)
    midi_files = midi_files[:sample_size]

    rolls = []
    for midi_path in midi_files:
        proll = midi_to_pianoroll(midi_path, fs=16, binarize=True)
        rolls.extend(segment_pianoroll(proll, seq_len=seq_len))

    if not rolls:
        raise RuntimeError("No piano-roll segments generated; try a larger sample size.")

    rolls = np.stack(rolls)
    split_idx = max(1, int(0.8 * len(rolls)))
    train_rolls, val_rolls = rolls[:split_idx], rolls[split_idx:]

    PROCESSED_ROLLS.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED_ROLLS / "train.npy", train_rolls)
    np.save(PROCESSED_ROLLS / "val.npy", val_rolls)

    positives = train_rolls.sum()
    total = train_rolls.size
    if positives > 0:
        pos_weight = (total - positives) / positives
        (PROCESSED_ROLLS / "pos_weight.txt").write_text(f"{pos_weight:.4f}")

    tokenizer = build_tokenizer()
    token_sequences: list[list[int]] = []
    for midi_path in midi_files:
        tokens = tokenizer(midi_path)
        candidates = tokens if isinstance(tokens, list) else [tokens]
        for cand in candidates:
            token_ids = cand.ids if hasattr(cand, "ids") else cand
            if len(token_ids) > 0:
                token_sequences.append(list(token_ids))
                break

    if not token_sequences:
        raise RuntimeError("No token sequences produced from MIDI files.")

    split_idx = max(1, int(0.8 * len(token_sequences)))
    train_tokens = np.array(token_sequences[:split_idx], dtype=object)
    val_tokens = np.array(token_sequences[split_idx:], dtype=object)

    PROCESSED_TOKENS.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED_TOKENS / "train.npy", train_tokens)
    np.save(PROCESSED_TOKENS / "val.npy", val_tokens)

    print("Smoke preprocessing complete:")
    print(f"- Rolls: {train_rolls.shape} train, {val_rolls.shape} val")
    print(f"- Tokens: {len(train_tokens)} train, {len(val_tokens)} val")


if __name__ == "__main__":
    main()
