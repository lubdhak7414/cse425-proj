import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

repo_root = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_root))

from src.config import DEVICE
from src.models.vae import build_vae


class PianoRollDataset(Dataset):
    def __init__(self, np_file):
        self.data = np.load(np_file).astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx])


def resolve_roll_paths(root_dir):
    processed_dir = root_dir / "data" / "processed" / "rolls"
    legacy_dir = root_dir / "data" / "processed_rolls"
    train_path = processed_dir / "train.npy"
    val_path = processed_dir / "val.npy"
    pos_weight_path = processed_dir / "pos_weight.txt"
    if not train_path.exists():
        train_path = legacy_dir / "train.npy"
        val_path = legacy_dir / "val.npy"
        pos_weight_path = legacy_dir / "pos_weight.txt"
    return train_path, val_path, pos_weight_path


def load_pos_weight(path, default=20.0):
    if not path.exists():
        return default
    try:
        return float(path.read_text().strip())
    except Exception:
        return default


def main():
    parser = argparse.ArgumentParser(description="Train LSTM VAE for piano-roll data.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--kl-warmup", type=int, default=10)
    args = parser.parse_args()

    root_dir = repo_root
    train_path, val_path, pos_weight_path = resolve_roll_paths(root_dir)
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("Missing processed piano-roll data. Expected train/val .npy files.")

    train_data = PianoRollDataset(train_path)
    val_data = PianoRollDataset(val_path)
    device = DEVICE
    print(f"Using device: {device}")
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    model = build_vae(
        input_dim=88,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        seq_len=args.seq_len,
    ).to(device)

    pos_weight_val = load_pos_weight(pos_weight_path)
    recon_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=device), reduction="sum")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    recon_hist = []
    kl_hist = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        beta = min(1.0, epoch / max(args.kl_warmup, 1))
        recon_total = 0.0
        kl_total = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits, mu, logvar = model(batch)
            recon = recon_loss_fn(logits, batch)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = (recon + beta * kl) / batch.size(0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            recon_total += recon.item() / batch.size(0)
            kl_total += kl.item() / batch.size(0)

        recon_hist.append(recon_total / len(train_loader))
        kl_hist.append(kl_total / len(train_loader))
        print(f"Epoch {epoch}: Recon {recon_hist[-1]:.2f} | KL {kl_hist[-1]:.2f} | Beta {beta:.2f}")

    plots_dir = root_dir / "outputs" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(recon_hist, label="Reconstruction")
    plt.plot(kl_hist, label="KL")
    plt.legend()
    plt.title("Task 2 VAE Loss Components")
    plt.tight_layout()
    plt.savefig(plots_dir / "task2_vae_losses.png")
    plt.savefig(plots_dir / "task2_vae_losses.pdf")
    plt.close()

    model_dir = root_dir / "models" / "saved"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "vae.pth")
    del model
    del train_loader
    del val_loader
    import gc
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
