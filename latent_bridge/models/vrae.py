from __future__ import annotations

import math

import torch
from torch import nn


def prepend_reference_frame_padding(
    images: torch.Tensor,
    *,
    compression_ratio: int = 4,
) -> torch.Tensor:
    """Pad an I2V clip so the reference frame owns one complete V-RAE chunk.

    A 49-frame clip ``[I0, I1, ..., I48]`` becomes
    ``[I0, I0, I0, I0, I1, ..., I48]`` for ratio four.  V-RAE therefore emits
    13 chunks, matching inference's one reference chunk plus 12 target chunks.
    """

    compression_ratio = int(compression_ratio)
    if images.ndim != 5:
        raise ValueError(f"Expected images [B,T,C,H,W], got {tuple(images.shape)}")
    if images.shape[1] <= 0 or compression_ratio <= 0:
        raise ValueError("images must contain frames and compression_ratio must be positive")
    prefix_frames = compression_ratio - 1
    if prefix_frames == 0:
        return images
    prefix = images[:, :1].expand(-1, prefix_frames, -1, -1, -1)
    return torch.cat((prefix, images), dim=1)


def crop_reference_frame_padding(
    features: torch.Tensor,
    *,
    compression_ratio: int = 4,
) -> torch.Tensor:
    """Remove synthetic prefix phases while retaining the real reference frame."""

    compression_ratio = int(compression_ratio)
    if features.ndim not in {4, 5}:
        raise ValueError(
            "Expected temporal features [B,T,N,C] or [B,T,C,H,W], got "
            f"{tuple(features.shape)}"
        )
    prefix_frames = compression_ratio - 1
    if compression_ratio <= 0 or features.shape[1] <= prefix_frames:
        raise ValueError(
            "features must outlive the synthetic prefix and compression_ratio must be positive"
        )
    return features[:, prefix_frames:].contiguous()


class VRAEToOmegaAdapter(nn.Module):
    """Residual per-patch alignment from V-RAE K7 features to Omega tokens."""

    def __init__(self, dim: int = 1024, hidden_dim: int = 2048) -> None:
        super().__init__()
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.norm = nn.LayerNorm(self.dim)
        self.fc1 = nn.Linear(self.dim, self.hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(self.hidden_dim, self.dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        # Zero residual makes the initial adapter an exact identity while still
        # allowing fc2 to receive gradients on the first optimization step.
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 4 or tokens.shape[-1] != self.dim:
            raise ValueError(
                f"Expected tokens [B,T,N,{self.dim}], got {tuple(tokens.shape)}"
            )
        residual = self.fc2(self.activation(self.fc1(self.norm(tokens))))
        return tokens + residual


class TemporalTokenAligner(nn.Module):
    """Residual temporal attention at each spatial patch, preserving every frame."""

    def __init__(self, dim: int = 1024, hidden_dim: int = 2048, num_heads: int = 16) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        if self.dim % self.num_heads:
            raise ValueError(f"dim={self.dim} must be divisible by num_heads={self.num_heads}")
        self.attention_norm = nn.LayerNorm(self.dim)
        self.attention = nn.MultiheadAttention(
            self.dim,
            self.num_heads,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(self.dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.dim),
        )
        # Both residual branches start at zero, so an Experiment A spatial
        # checkpoint is reproduced exactly before temporal training begins.
        nn.init.zeros_(self.attention.out_proj.weight)
        nn.init.zeros_(self.attention.out_proj.bias)
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    @staticmethod
    def _time_encoding(
        time: int,
        dim: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        position = torch.arange(time, device=device, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, dim, 2, device=device, dtype=torch.float32)
            * (-math.log(10_000.0) / dim)
        )
        encoding = torch.zeros(time, dim, device=device, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency)
        return encoding.to(dtype=dtype)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 4 or tokens.shape[-1] != self.dim:
            raise ValueError(
                f"Expected tokens [B,T,N,{self.dim}], got {tuple(tokens.shape)}"
            )
        batch, time, patches, channels = tokens.shape
        sequence = tokens.permute(0, 2, 1, 3).reshape(batch * patches, time, channels)
        normalized = self.attention_norm(sequence)
        position = self._time_encoding(
            time,
            channels,
            device=tokens.device,
            dtype=normalized.dtype,
        ).unsqueeze(0)
        attended, _ = self.attention(
            normalized + position,
            normalized + position,
            normalized,
            need_weights=False,
        )
        sequence = sequence + attended
        sequence = sequence + self.ffn(self.ffn_norm(sequence))
        return sequence.reshape(batch, patches, time, channels).permute(0, 2, 1, 3).contiguous()


class TemporalVRAEToOmegaAdapter(nn.Module):
    """Spatial feature mapping followed by frame-preserving temporal alignment."""

    def __init__(
        self,
        dim: int = 1024,
        hidden_dim: int = 2048,
        temporal_heads: int = 16,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.spatial = VRAEToOmegaAdapter(dim=dim, hidden_dim=hidden_dim)
        self.temporal = TemporalTokenAligner(
            dim=dim,
            hidden_dim=hidden_dim,
            num_heads=temporal_heads,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.temporal(self.spatial(tokens))


class CompressedLatentToOmegaAdapter(nn.Module):
    """Expand a compressed V-RAE latent grid into a per-frame feature grid.

    V-RAE produces one clean latent for every four RGB frames.  The learned
    phase projection predicts four frame-specific residuals per latent chunk;
    spatial token alignment and frame-level temporal attention are then trained
    jointly.  No RGB decoder or Omega image encoder is used on this path.
    """

    def __init__(
        self,
        dim: int = 1024,
        hidden_dim: int = 2048,
        temporal_heads: int = 16,
        compression_ratio: int = 4,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.compression_ratio = int(compression_ratio)
        if self.compression_ratio <= 0:
            raise ValueError("compression_ratio must be positive")
        self.phase_norm = nn.LayerNorm(self.dim)
        self.phase_projection = nn.Linear(
            self.dim,
            self.compression_ratio * self.dim,
        )
        self.phase_embedding = nn.Parameter(
            torch.zeros(self.compression_ratio, self.dim)
        )
        self.spatial = VRAEToOmegaAdapter(dim=dim, hidden_dim=hidden_dim)
        self.temporal = TemporalTokenAligner(
            dim=dim,
            hidden_dim=hidden_dim,
            num_heads=temporal_heads,
        )
        nn.init.zeros_(self.phase_projection.weight)
        nn.init.zeros_(self.phase_projection.bias)

    def _expand_time_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 4 or tokens.shape[-1] != self.dim:
            raise ValueError(
                f"Expected compressed tokens [B,Tc,N,{self.dim}], got {tuple(tokens.shape)}"
            )
        batch, chunks, patches, channels = tokens.shape
        phase_delta = self.phase_projection(self.phase_norm(tokens)).reshape(
            batch,
            chunks,
            patches,
            self.compression_ratio,
            channels,
        )
        phase_delta = phase_delta.permute(0, 1, 3, 2, 4)
        expanded = tokens.unsqueeze(2) + phase_delta
        expanded = expanded + self.phase_embedding.to(
            device=tokens.device,
            dtype=tokens.dtype,
        )[None, None, :, None, :]
        return expanded.reshape(
            batch,
            chunks * self.compression_ratio,
            patches,
            channels,
        )

    def forward(self, latent_grid: torch.Tensor) -> torch.Tensor:
        if latent_grid.ndim != 5 or latent_grid.shape[2] != self.dim:
            raise ValueError(
                "Expected latent_grid [B,Tc,C,H,W] with "
                f"C={self.dim}, got {tuple(latent_grid.shape)}"
            )
        batch, _, channels, height, width = latent_grid.shape
        tokens = (
            latent_grid.permute(0, 1, 3, 4, 2)
            .flatten(2, 3)
            .contiguous()
        )
        expanded = self._expand_time_tokens(tokens)
        aligned = self.temporal(self.spatial(expanded))
        return (
            aligned.reshape(batch, aligned.shape[1], height, width, channels)
            .permute(0, 1, 4, 2, 3)
            .contiguous()
        )
