import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=88, hidden_dim=256, latent_dim=64, seq_len=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.encoder_lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.encoder_linear = nn.Linear(hidden_dim, latent_dim)

        self.decoder_lstm = nn.LSTM(
            latent_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder_linear = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        return self.encoder_linear(h_n[-1])

    def decode(self, z):
        z_repeated = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.decoder_lstm(z_repeated)
        return self.decoder_linear(out)

    def forward(self, x):
        return self.decode(self.encode(x))


def build_autoencoder(
    input_dim=88,
    hidden_dim=256,
    latent_dim=64,
    seq_len=128,
    num_layers=2,
    dropout=0.2,
):
    return LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        seq_len=seq_len,
        num_layers=num_layers,
        dropout=dropout,
    )
