"""Compact, full-window models; no positional embedding or extra Mamba MLP."""

import torch
from torch import nn
from .models import wide_mamba_forward


def temporal_pool(x, method):
    if method == "max":
        return x.amax(dim=1)
    if method == "mean_std":
        # Population variance; clamp before sqrt avoids infinite derivative at zero.
        return torch.cat((x.mean(dim=1), x.var(dim=1, unbiased=False).clamp_min(1e-8).sqrt()), dim=-1)
    raise ValueError("Unknown pooling")


class SmallMamba(nn.Module):
    def __init__(self, width=64, layers=2, state=16, kernel=16, dropout=.2, pooling="mean_std"):
        super().__init__()
        from mamba_ssm import Mamba
        self.pooling = pooling
        self.projection = nn.Linear(18, width)
        self.input_norm = nn.LayerNorm(width)
        self.blocks = nn.ModuleList([Mamba(d_model=width, d_state=state, d_conv=kernel, expand=2)
                                     for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(width * (2 if pooling == "mean_std" else 1), 2)

    def forward(self, x):
        h = self.input_norm(self.projection(x))
        for block, norm in zip(self.blocks, self.norms):
            h = norm(h + wide_mamba_forward(block, h))
        return self.head(self.drop(temporal_pool(h, self.pooling)))


class TemporalBlock(nn.Module):
    """Two dilated causal convolutions, channel-only LayerNorm, residual path."""
    def __init__(self, width, kernel, dilation, dropout):
        super().__init__()
        self.pad = (kernel-1)*dilation
        self.convs = nn.ModuleList([nn.Conv1d(width, width, kernel, dilation=dilation,
                                            padding=self.pad) for _ in range(2)])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(2)])
        self.act, self.drop = nn.GELU(), nn.Dropout(dropout)

    def forward(self, x):
        h = x
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h.transpose(1, 2))[:, :, :x.shape[1]].transpose(1, 2)
            h = self.drop(self.act(norm(h)))
        return x + h


class CompactCNN(nn.Module):
    def __init__(self, width=32, layers=4, kernel=7, dropout=.2, pooling="mean_std"):
        super().__init__()
        self.pooling = pooling
        self.projection = nn.Linear(18, width)  # kernel-1 joint channel mixing
        self.norm = nn.LayerNorm(width)
        self.blocks = nn.ModuleList([TemporalBlock(width, kernel, 2**i, dropout) for i in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(width * (2 if pooling == "mean_std" else 1), 2)
        # Per-timestep receptive field before global pooling; no temporal normalization.
        self.receptive_field = 1 + 2*(kernel-1)*(2**layers-1)

    def forward(self, x):
        h = self.norm(self.projection(x))
        for block in self.blocks:
            h = block(h)
        return self.head(self.drop(temporal_pool(h, self.pooling)))


def build_candidate(candidate):
    kwargs = {k: candidate[k] for k in ("width", "layers", "kernel", "dropout", "pooling")}
    if candidate["family"] == "mamba":
        return SmallMamba(**kwargs, state=candidate["state"])
    if candidate["family"] == "cnn":
        return CompactCNN(**kwargs)
    raise ValueError("Unknown model family")