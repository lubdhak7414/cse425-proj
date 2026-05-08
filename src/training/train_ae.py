import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

repo_root = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_root))

from src.models.autoencoder import build_autoencoder


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
    parser = argparse.ArgumentParser(description="Train LSTM autoencoder for piano-roll data.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=64)
    args = parser.parse_args()

    root_dir = repo_root
    train_path, val_path, pos_weight_path = resolve_roll_paths(root_dir)
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("Missing processed piano-roll data. Expected train/val .npy files.")

    train_data = PianoRollDataset(train_path)
    val_data = PianoRollDataset(val_path)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_autoencoder(
        input_dim=88,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        seq_len=args.seq_len,
    ).to(device)

    pos_weight_val = load_pos_weight(pos_weight_path)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=device))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_history = []
    val_history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * batch.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch)
                val_loss += criterion(logits, batch).item() * batch.size(0)

        train_history.append(train_loss / len(train_data))
        val_history.append(val_loss / len(val_data))
        print(f"Epoch {epoch}: Train {train_history[-1]:.4f} | Val {val_history[-1]:.4f}")

    plots_dir = root_dir / "outputs" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(train_history, label="Train")
    plt.plot(val_history, label="Validation")
    plt.legend()
    plt.title("Task 1 Loss")
    plt.tight_layout()
    plt.savefig(plots_dir / "task1_loss.png")
    plt.savefig(plots_dir / "task1_loss.pdf")
    plt.close()

    model_dir = root_dir / "models" / "saved"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "autoencoder.pth")


if __name__ == "__main__":
    main()
