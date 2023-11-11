import torch
from torch import nn


class VariationalInformationBottleneck(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super().__init__()
        self.fc_mu = nn.Linear(input_dim, bottleneck_dim)
        self.fc_log_var = nn.Linear(input_dim, bottleneck_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var
