import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from miditok import REMI, TokenizerConfig

from src.models.transformer import build_transformer


class TokenDataset(Dataset):
    def __init__(self, np_file, pad_token, seq_len=512, genre_file=None):
        self.data = np.load(np_file, allow_pickle=True)
        self.seq_len = seq_len
        self.pad_token = pad_token
        if genre_file and genre_file.exists():
            self.genres = np.load(genre_file)
        else:
            self.genres = np.zeros(len(self.data), dtype=int)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = list(self.data[idx])
        if len(seq) < self.seq_len:
            seq += [self.pad_token] * (self.seq_len - len(seq))
        genre = int(self.genres[idx]) if idx < len(self.genres) else 0
        return torch.tensor(seq[: self.seq_len], dtype=torch.long), torch.tensor(genre, dtype=torch.long)


def resolve_token_paths(root_dir):
    processed_dir = root_dir / "data" / "processed" / "tokens"
    legacy_dir = root_dir / "data" / "processed_tokens"
    train_path = processed_dir / "train.npy"
    genre_path = processed_dir / "genres.npy"
    if not train_path.exists():
        train_path = legacy_dir / "train.npy"
        genre_path = legacy_dir / "genres.npy"
    return train_path, genre_path


def main():
    parser = argparse.ArgumentParser(description="Train transformer for tokenized MIDI.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-velocities", type=int, default=32)
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[2]
    train_path, genre_path = resolve_token_paths(root_dir)
    if not train_path.exists():
        raise FileNotFoundError("Missing tokenized data. Expected data/processed/tokens/train.npy or legacy path.")

    config = TokenizerConfig(num_velocities=args.num_velocities, use_chords=False, use_programs=False)
    tokenizer = REMI(config)
    pad_token = tokenizer["PAD_None"]
    vocab_size = tokenizer.vocab_size

    dataset = TokenDataset(train_path, pad_token, seq_len=args.seq_len, genre_file=genre_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    genre_count = int(np.max(dataset.genres)) + 1 if len(dataset.genres) else 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_transformer(
        vocab_size=vocab_size,
        genre_count=genre_count,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        max_len=max(args.seq_len, 1024),
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_token)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    perplexity_hist = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for tokens, genres in loader:
            tokens = tokens.to(device)
            genres = genres.to(device)
            x_input, y_target = tokens[:, :-1], tokens[:, 1:]
            optimizer.zero_grad()
            logits = model(x_input, genres)
            loss = criterion(logits.reshape(-1, vocab_size), y_target.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        perplexity = math.exp(avg_loss)
        perplexity_hist.append(perplexity)
        print(f"Epoch {epoch}: Loss {avg_loss:.4f} | Perplexity {perplexity:.4f}")

    plots_dir = root_dir / "outputs" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(perplexity_hist, label="Perplexity")
    plt.legend()
    plt.title("Task 3 Perplexity")
    plt.tight_layout()
    plt.savefig(plots_dir / "task3_perplexity.png")
    plt.savefig(plots_dir / "task3_perplexity.pdf")
    plt.close()

    model_dir = root_dir / "models" / "saved"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "transformer.pth")


if __name__ == "__main__":
    main()
