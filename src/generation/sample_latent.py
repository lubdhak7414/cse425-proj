# This file has been worked on by Safwan Usaid Lubdhak
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

repo_root = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_root))

from src.generation.midi_export import piano_roll_to_midi, validate_midi
from src.models.autoencoder import LSTMAutoencoder
from src.models.vae import LSTMVAE


def sample_latent(model, latent_dim, device):
    return torch.randn(1, latent_dim, device=device)


def main():
    parser = argparse.ArgumentParser(description="Sample latent codes and export MIDI.")
    parser.add_argument("--model-type", choices=["ae", "vae"], default="ae")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--out-dir", type=str, default=str(repo_root / "outputs" / "generated_midis" / "task1"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_type == "vae":
        model = LSTMVAE(seq_len=args.seq_len, latent_dim=args.latent_dim).to(device)
        default_ckpt = repo_root / "models" / "saved" / "vae.pth"
    else:
        model = LSTMAutoencoder(seq_len=args.seq_len, latent_dim=args.latent_dim).to(device)
        default_ckpt = repo_root / "models" / "saved" / "autoencoder.pth"

    ckpt_path = Path(args.checkpoint) if args.checkpoint else default_ckpt
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = 0
    for i in range(args.num_samples):
        z = sample_latent(model, args.latent_dim, device)
        with torch.no_grad():
            logits = model.decode(z).cpu().numpy()[0]
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        roll = (probs > args.threshold).astype(int)
        out_path = out_dir / f"sample_{i + 1}.mid"
        piano_roll_to_midi(roll, str(out_path))
        if validate_midi(str(out_path)):
            generated += 1
        else:
            out_path.unlink(missing_ok=True)
    print(f"Generated {generated} MIDI files.")


if __name__ == "__main__":
    main()
