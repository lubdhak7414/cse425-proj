import torch
from torch import nn


class LSTMVAE(nn.Module):
    def __init__(self, input_dim=88, hidden_dim=256, latent_dim=64, seq_len=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.LSTM(
            latent_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        _, (h, _) = self.encoder(x)
        h_last = h[-1]
        return self.fc_mu(h_last), self.fc_logvar(h_last)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z_rep = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.decoder(z_rep)
        return self.fc_out(out)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def build_vae(
    input_dim=88,
    hidden_dim=256,
    latent_dim=64,
    seq_len=128,
    num_layers=2,
    dropout=0.2,
):
    return LSTMVAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        seq_len=seq_len,
        num_layers=num_layers,
        dropout=dropout,
    )
