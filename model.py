import torch
from torch import nn

class FlexibleCompensator(nn.Module):
    def __init__(self, input_dim, hidden_sizes, output_dim=None, activation="GELU", norm="BatchNorm", dropout=0.2):
        super().__init__()
        layers = []
        act = {"ReLU": nn.ReLU(), "GELU": nn.GELU(), "SiLU": nn.SiLU(), "LeakyReLU": nn.LeakyReLU()}[activation]
        norm_fn = {
            "BatchNorm": nn.BatchNorm1d,
            "LayerNorm": nn.LayerNorm,
            "None": lambda x: nn.Identity()
        }[norm]

        prev_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            if norm != "None": layers.append(norm_fn(h))
            layers.append(act)
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        if output_dim is None:
            output_dim = input_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
