from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pretty_midi
from tqdm import tqdm

from src.preprocessing.piano_roll import midi_to_pianoroll, segment_pianoroll
from src.preprocessing.tokenizer import build_tokenizer, normalize_token_ids


def resolve_maestro_root() -> Path:
    raw_root = Path("data") / "raw_midi"
    maestro_dir = raw_root / "maestro-v3.0.0"
    fallback_dir = Path("data") / "maestro-v3.0.0"
    metadata_file = "maestro-v3.0.0.csv"
    if (maestro_dir / metadata_file).exists():
        return maestro_dir
    if (fallback_dir / metadata_file).exists():
        return fallback_dir
    raise FileNotFoundError("MAESTRO metadata CSV not found in data/raw_midi or data/")


def load_splits(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata_path = data_dir / "maestro-v3.0.0.csv"
    metadata = pd.read_csv(metadata_path)
    train_meta = metadata[metadata["split"] == "train"]
    val_meta = metadata[metadata["split"] == "validation"]
    test_meta = metadata[metadata["split"] == "test"]

    split_dir = Path("data") / "train_test_split"
    split_dir.mkdir(parents=True, exist_ok=True)
    train_meta.to_csv(split_dir / "train.csv", index=False)
    val_meta.to_csv(split_dir / "val.csv", index=False)
    test_meta.to_csv(split_dir / "test.csv", index=False)

    return train_meta, val_meta, test_meta


def process_piano_roll(
    df: pd.DataFrame,
    data_dir: Path,
    fs: int,
    seq_len: int,
    sparsity_threshold: float,
) -> np.ndarray:
    windows: list[np.ndarray] = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        midi_file = data_dir / row["midi_filename"]
        try:
            proll = midi_to_pianoroll(midi_file, fs=fs, pitch_range=(21, 109), binarize=True)
            windows.extend(segment_pianoroll(proll, seq_len=seq_len, sparsity_threshold=sparsity_threshold))
        except Exception:
            continue
    return np.array(windows, dtype=np.float32)


def save_rolls(train_rolls: np.ndarray, val_rolls: np.ndarray, test_rolls: np.ndarray) -> None:
    processed_rolls = Path("data") / "processed" / "rolls"
    legacy_rolls = Path("data") / "processed_rolls"
    processed_rolls.mkdir(parents=True, exist_ok=True)
    legacy_rolls.mkdir(parents=True, exist_ok=True)

    for root in (processed_rolls, legacy_rolls):
        np.save(root / "train.npy", train_rolls)
        np.save(root / "val.npy", val_rolls)
        np.save(root / "test.npy", test_rolls)

    num_positive = np.sum(train_rolls == 1)
    num_negative = np.sum(train_rolls == 0)
    pos_weight = num_negative / max(num_positive, 1)
    for root in (processed_rolls, legacy_rolls):
        (root / "pos_weight.txt").write_text(str(pos_weight))

    print(
        f"Saved {len(train_rolls)} train, {len(val_rolls)} val, {len(test_rolls)} test rolls. "
        f"Suggested pos_weight: {pos_weight:.2f}"
    )


def process_tokens(
    df: pd.DataFrame,
    data_dir: Path,
    tokenizer,
    min_len: int,
) -> list[list[int]]:
    token_sequences: list[list[int]] = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        midi_file = data_dir / row["midi_filename"]
        try:
            tokens = tokenizer(str(midi_file))
            for seq in normalize_token_ids(tokens):
                if len(seq) >= min_len:
                    token_sequences.append(seq)
        except Exception:
            continue
    return token_sequences


def save_tokens(train_tokens, val_tokens, test_tokens) -> None:
    processed_tokens = Path("data") / "processed" / "tokens"
    legacy_tokens = Path("data") / "processed_tokens"
    processed_tokens.mkdir(parents=True, exist_ok=True)
    legacy_tokens.mkdir(parents=True, exist_ok=True)

    for root in (processed_tokens, legacy_tokens):
        np.save(root / "train.npy", np.array(train_tokens, dtype=object))
        np.save(root / "val.npy", np.array(val_tokens, dtype=object))
        np.save(root / "test.npy", np.array(test_tokens, dtype=object))

    print(
        f"Saved {len(train_tokens)} train, {len(val_tokens)} val, {len(test_tokens)} test sequences for Transformer."
    )


def save_eda_plots(data_dir: Path, train_meta: pd.DataFrame, sample_size: int = 200) -> None:
    plot_dir = Path("outputs") / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    durations = []
    pitches = []
    sample_rows = train_meta.head(sample_size)
    for _, row in tqdm(sample_rows.iterrows(), total=len(sample_rows)):
        midi_file = data_dir / row["midi_filename"]
        try:
            pm = pretty_midi.PrettyMIDI(str(midi_file))
            durations.append(pm.get_end_time())
            for inst in pm.instruments:
                pitches.extend([n.pitch for n in inst.notes])
        except Exception:
            continue

    if durations:
        plt.figure()
        plt.hist(durations, bins=30)
        plt.title("Duration Histogram")
        plt.tight_layout()
        plt.savefig(plot_dir / "eda_duration_hist.png")
        plt.savefig(plot_dir / "eda_duration_hist.pdf")
        plt.close()

    if pitches:
        plt.figure()
        plt.hist(pitches, bins=30)
        plt.title("Pitch Distribution Histogram")
        plt.tight_layout()
        plt.savefig(plot_dir / "eda_pitch_hist.png")
        plt.savefig(plot_dir / "eda_pitch_hist.pdf")
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Full MAESTRO preprocessing pipeline.")
    parser.add_argument("--fs", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--sparsity-threshold", type=float, default=0.02)
    parser.add_argument("--min-token-len", type=int, default=50)
    parser.add_argument("--num-velocities", type=int, default=32)
    parser.add_argument("--eda", action="store_true", help="Generate EDA plots.")
    args = parser.parse_args()

    data_dir = resolve_maestro_root()
    train_meta, val_meta, test_meta = load_splits(data_dir)
    print(
        f"Training files: {len(train_meta)}, Val files: {len(val_meta)}, Test files: {len(test_meta)}"
    )

    train_rolls = process_piano_roll(
        train_meta,
        data_dir,
        fs=args.fs,
        seq_len=args.seq_len,
        sparsity_threshold=args.sparsity_threshold,
    )
    val_rolls = process_piano_roll(
        val_meta,
        data_dir,
        fs=args.fs,
        seq_len=args.seq_len,
        sparsity_threshold=args.sparsity_threshold,
    )
    test_rolls = process_piano_roll(
        test_meta,
        data_dir,
        fs=args.fs,
        seq_len=args.seq_len,
        sparsity_threshold=args.sparsity_threshold,
    )
    save_rolls(train_rolls, val_rolls, test_rolls)

    tokenizer = build_tokenizer(num_velocities=args.num_velocities)
    train_tokens = process_tokens(train_meta, data_dir, tokenizer, min_len=args.min_token_len)
    val_tokens = process_tokens(val_meta, data_dir, tokenizer, min_len=args.min_token_len)
    test_tokens = process_tokens(test_meta, data_dir, tokenizer, min_len=args.min_token_len)
    save_tokens(train_tokens, val_tokens, test_tokens)

    if args.eda:
        save_eda_plots(data_dir, train_meta)


if __name__ == "__main__":
    main()
