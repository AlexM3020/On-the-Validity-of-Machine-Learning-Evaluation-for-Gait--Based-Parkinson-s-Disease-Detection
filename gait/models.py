"""Reconstructed classifiers; no claim of exact historical reproducibility.

The Mamba dependency is imported only upon MambaClassifier construction.
No replacement backend or training is performed on import.
"""

import math
from numbers import Integral, Real

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _config(input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
    for name, value in (("input_dim", input_dim), ("hidden_dim", hidden_dim), ("num_layers", num_layers)):
        _positive_int(value, name)
    if isinstance(dropout, bool) or not isinstance(dropout, Real) or not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError("dropout must be finite and in [0, 1)")


def _lengths(x: torch.Tensor, lengths: torch.Tensor | None, input_dim: int) -> torch.Tensor:
    if x.ndim != 3 or x.shape[0] == 0 or x.shape[1] == 0 or x.shape[2] != input_dim:
        raise ValueError("x must have nonempty shape [batch, time, input_dim]")
    if not x.is_floating_point():
        raise ValueError("x must be a floating-point tensor")
    if lengths is None:
        return torch.full((x.shape[0],), x.shape[1], dtype=torch.long, device=x.device)
    if lengths.ndim != 1 or lengths.shape[0] != x.shape[0] or lengths.dtype not in (
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
    ):
        raise ValueError("lengths must be a one-dimensional integer tensor, one per sample")
    if torch.any(lengths <= 0).item() or torch.any(lengths > x.shape[1]).item():
        raise ValueError("Each length must be in [1, time]")
    return lengths.to(device=x.device, dtype=torch.long)


def wide_mamba_forward(block: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    """Full-sequence Mamba using PyTorch convolution and real selective scan.

    Wide depthwise kernels bypass causal-conv1d's width-2..4 optimized path.
    This independently written tensor formulation uses the block's existing
    parameters; it provides neither a replacement scan nor inference caching.
    """
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    time = hidden.shape[1]
    x, z = block.in_proj(hidden).chunk(2, dim=-1)
    x = x.transpose(1, 2).contiguous()
    z = z.transpose(1, 2).contiguous()
    x = block.act(block.conv1d(x)[:, :, :time]).contiguous()
    dt, state_b, state_c = block.x_proj(x.transpose(1, 2)).split(
        (block.dt_rank, block.d_state, block.d_state), dim=-1,
    )
    dt = nn.functional.linear(dt, block.dt_proj.weight).transpose(1, 2).contiguous()
    scan = selective_scan_fn(
        x, dt, -torch.exp(block.A_log.float()),
        state_b.transpose(1, 2).contiguous(),
        state_c.transpose(1, 2).contiguous(), block.D.float(),
        z=z, delta_bias=block.dt_proj.bias.float(), delta_softplus=True,
    )
    return block.out_proj(scan.transpose(1, 2))


class MambaClassifier(nn.Module):
    """Input projection/norm, residual Mamba/norm blocks, masked max, head.

    Default two blocks reconstruct the documented architecture. Trailing
    padding cannot affect valid outputs of the causal Mamba blocks, and is
    excluded from pooling. Real execution requires the Linux/CUDA Mamba
    backend; architecture-only tests may inject a clearly identified fake.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2,
        d_state: int = 32, d_conv: int = 32, expand: int = 2, dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _config(input_dim, hidden_dim, num_layers, dropout)
        for name, value in (("d_state", d_state), ("d_conv", d_conv), ("expand", expand)):
            _positive_int(value, name)
        try:
            from mamba_ssm import Mamba
        except (ImportError, OSError, RuntimeError) as exc:
            raise RuntimeError(
                "Mamba requires a working mamba_ssm Linux/CUDA installation and "
                "a compatible CUDA-enabled PyTorch environment. No fallback backend "
                "is provided; CPU baselines can use LSTM or GRU."
            ) from exc
        self.input_dim = input_dim
        self.d_conv = d_conv
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.norm_in = nn.LayerNorm(hidden_dim)
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.cls_head = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        lengths = _lengths(x, lengths, self.input_dim)
        hidden = self.norm_in(self.input_proj(x))
        for block, norm in zip(self.mamba_layers, self.norms):
            update = wide_mamba_forward(block, hidden) if self.d_conv > 4 else block(hidden)
            hidden = norm(hidden + update)
        valid = torch.arange(x.shape[1], device=x.device)[None, :] < lengths[:, None]
        pooled = hidden.masked_fill(~valid[..., None], -torch.inf).amax(dim=1)
        return self.cls_head(self.dropout(pooled))


class RecurrentClassifier(nn.Module):
    """Packed bidirectional LSTM/GRU with final-layer forward/backward states."""

    def __init__(
        self, model_name: str, input_dim: int, hidden_dim: int = 128,
        num_layers: int = 2, dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _config(input_dim, hidden_dim, num_layers, dropout)
        if not isinstance(model_name, str) or model_name.lower() not in ("lstm", "gru"):
            raise ValueError("Recurrent model_name must be 'lstm' or 'gru'")
        self.model_name = model_name.lower()
        self.input_dim = input_dim
        recurrent = nn.LSTM if self.model_name == "lstm" else nn.GRU
        self.recurrent = recurrent(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.cls_head = nn.Linear(2 * hidden_dim, 2)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        lengths = _lengths(x, lengths, self.input_dim)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, state = self.recurrent(packed)
        hidden = state[0] if self.model_name == "lstm" else state
        last_layer = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        return self.cls_head(self.dropout(last_layer))


def build_model(
    model_name: str, input_dim: int, hidden_dim: int = 128, num_layers: int = 2,
    d_state: int = 32, d_conv: int = 32, expand: int = 2, dropout: float = 0.0,
) -> nn.Module:
    """Build 'mamba', 'lstm', or 'gru' (case-insensitive), validating config.

    State/convolution/expansion settings apply only to Mamba, but must be
    positive integers for all names. Recurrent baselines never import Mamba.
    """
    if not isinstance(model_name, str) or model_name.lower() not in ("mamba", "lstm", "gru"):
        raise ValueError("model_name must be 'mamba', 'lstm', or 'gru'")
    _config(input_dim, hidden_dim, num_layers, dropout)
    for name, value in (("d_state", d_state), ("d_conv", d_conv), ("expand", expand)):
        _positive_int(value, name)
    if model_name.lower() == "mamba":
        return MambaClassifier(input_dim, hidden_dim, num_layers, d_state, d_conv, expand, dropout)
    return RecurrentClassifier(model_name, input_dim, hidden_dim, num_layers, dropout)
