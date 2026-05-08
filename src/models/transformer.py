import torch
from torch import nn


class GPTMusic(nn.Module):
    def __init__(
        self,
        vocab_size,
        genre_count,
        d_model=256,
        n_heads=8,
        num_layers=4,
        dropout=0.2,
        max_len=1024,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.genre_emb = nn.Embedding(genre_count, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x, genre_ids):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        genre_vec = self.genre_emb(genre_ids).unsqueeze(1)
        x_emb = self.token_emb(x) + self.pos_emb(positions) + genre_vec
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        out = self.transformer(x_emb, mask=mask, is_causal=True)
        return self.fc(out)


def build_transformer(
    vocab_size,
    genre_count,
    d_model=256,
    n_heads=8,
    num_layers=4,
    dropout=0.2,
    max_len=1024,
):
    return GPTMusic(
        vocab_size=vocab_size,
        genre_count=genre_count,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        dropout=dropout,
        max_len=max_len,
    )
