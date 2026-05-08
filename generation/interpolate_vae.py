import argparse
import sys
from pathlib import Path

import numpy as np
import torch

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from generation.midi_export import piano_roll_to_midi, validate_midi
from src.models.vae import LSTMVAE
from src.preprocessing.piano_roll import midi_to_pianoroll, segment_pianoroll


def load_first_window(midi_path: Path, seq_len: int) -> torch.Tensor:
    roll = midi_to_pianoroll(midi_path, fs=16, pitch_range=(21, 109), binarize=True)
    windows = segment_pianoroll(roll, seq_len=seq_len)
    if not windows:
        raise ValueError(f"No valid piano-roll windows found for {midi_path}")
    return torch.tensor(windows[0], dtype=torch.float32).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description="Interpolate VAE latent codes and export MIDI.")
    parser.add_argument("--checkpoint", type=str, default=str(repo_root / "models" / "saved" / "vae.pth"))
    parser.add_argument("--midi-a", type=str, required=True)
    parser.add_argument("--midi-b", type=str, required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--out-dir", type=str, default=str(repo_root / "outputs" / "generated_midis" / "task2" / "interpolation"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMVAE(seq_len=args.seq_len, latent_dim=args.latent_dim).to(device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    roll_a = load_first_window(Path(args.midi_a), args.seq_len).to(device)
    roll_b = load_first_window(Path(args.midi_b), args.seq_len).to(device)

    with torch.no_grad():
        mu_a, _ = model.encode(roll_a)
        mu_b, _ = model.encode(roll_b)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    for i in range(args.steps):
        alpha = i / max(args.steps - 1, 1)
        z = (1 - alpha) * mu_a + alpha * mu_b
        with torch.no_grad():
            logits = model.decode(z).cpu().numpy()[0]
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        roll = (probs > args.threshold).astype(int)
        out_path = out_dir / f"interp_{i + 1:02d}.mid"
        piano_roll_to_midi(roll, str(out_path))
        if validate_midi(str(out_path)):
            generated += 1
        else:
            out_path.unlink(missing_ok=True)

    print(f"Generated {generated} interpolated MIDI files.")


if __name__ == "__main__":
    main()
