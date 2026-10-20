from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class WanOmega35BridgeCfg:
    latent_channels: int = 16
    latent_frames: int = 13
    latent_height: int = 70
    latent_width: int = 70
    temporal_compression: int = 4
    output_frames: int = 49
    grid_height: int = 35
    grid_width: int = 35
    hidden_dim: int = 512
    output_dim: int = 1024
    num_blocks: int = 6
    num_heads: int = 8
    mlp_ratio: int = 4
    spatial_kernel_size: int = 7
    temporal_chunk_size: int = 256
    use_checkpoint: bool = True
    output_scale_init: float = 1.0
    output_bias_init: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class CausalTemporalPhaseUpsampler(nn.Module):
    """Expand causal Wan latents with four learned phases per later latent."""

    def __init__(self, channels: int, compression: int) -> None:
        super().__init__()
        if channels < 1 or compression < 1:
            raise ValueError("channels and compression must be positive")
        self.channels = channels
        self.compression = compression
        self.phase_projection = nn.Conv3d(
            channels,
            channels * compression,
            kernel_size=(3, 1, 1),
            padding=0,
        )
        self._initialize_as_causal_repeat()

    def _initialize_as_causal_repeat(self) -> None:
        with torch.no_grad():
            self.phase_projection.weight.zero_()
            self.phase_projection.bias.zero_()
            identity = torch.eye(
                self.channels,
                dtype=self.phase_projection.weight.dtype,
                device=self.phase_projection.weight.device,
            )
            for phase in range(self.compression):
                start = phase * self.channels
                self.phase_projection.weight[
                    start : start + self.channels, :, 2, 0, 0
                ].copy_(identity)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 5 or features.shape[1] != self.channels:
            raise ValueError(
                f"Expected [B,{self.channels},T,H,W], got {tuple(features.shape)}"
            )
        first = features[:, :, :1]
        if features.shape[2] == 1:
            return first
        causal = F.pad(features, (0, 0, 0, 0, 2, 0))
        phases = self.phase_projection(causal)[:, :, 1:]
        batch, _, groups, height, width = phases.shape
        phases = phases.view(
            batch,
            self.compression,
            self.channels,
            groups,
            height,
            width,
        )
        phases = phases.permute(0, 2, 3, 1, 4, 5).reshape(
            batch,
            self.channels,
            groups * self.compression,
            height,
            width,
        )
        return torch.cat([first, phases], dim=2)


class Omega35SpatialPacker(nn.Module):
    """Pack each exact 2x2 Wan latent phase into one Omega patch token."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        groups = min(32, output_channels)
        while output_channels % groups:
            groups -= 1
        self.input_channels = input_channels
        self.projection = nn.Sequential(
            nn.Conv2d(input_channels * 4, output_channels, kernel_size=1),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(),
            nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                groups=output_channels,
            ),
            nn.Conv2d(output_channels, output_channels, kernel_size=1),
        )

    def forward(self, latent_frames: Tensor) -> Tensor:
        if latent_frames.ndim != 4 or latent_frames.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected [N,{self.input_channels},H,W], got "
                f"{tuple(latent_frames.shape)}"
            )
        if latent_frames.shape[-2] % 2 or latent_frames.shape[-1] % 2:
            raise ValueError("Omega35 packing requires an even latent grid")
        packed = F.pixel_unshuffle(latent_frames, downscale_factor=2)
        return self.projection(packed)


class Omega35BridgeBlock(nn.Module):
    """Local spatial mixing plus frame attention at each Omega patch."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: int,
        spatial_kernel_size: int,
        temporal_chunk_size: int,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if spatial_kernel_size < 1 or spatial_kernel_size % 2 == 0:
            raise ValueError("spatial_kernel_size must be positive and odd")
        self.dim = dim
        self.temporal_chunk_size = temporal_chunk_size
        self.spatial_norm = nn.LayerNorm(dim)
        self.spatial_depthwise = nn.Conv2d(
            dim,
            dim,
            kernel_size=spatial_kernel_size,
            padding=spatial_kernel_size // 2,
            groups=dim,
        )
        self.spatial_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * mlp_ratio, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim * mlp_ratio, dim, kernel_size=1),
        )
        self.temporal_norm = nn.LayerNorm(dim)
        self.temporal_attention = nn.MultiheadAttention(
            dim,
            num_heads,
            batch_first=True,
        )
        self.channel_norm = nn.LayerNorm(dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def _temporal_attention(self, sequence: Tensor) -> Tensor:
        chunks: list[Tensor] = []
        for start in range(0, sequence.shape[0], self.temporal_chunk_size):
            chunk = sequence[start : start + self.temporal_chunk_size]
            normalized = self.temporal_norm(chunk)
            attended, _ = self.temporal_attention(
                normalized,
                normalized,
                normalized,
                need_weights=False,
            )
            chunks.append(chunk + attended)
        return torch.cat(chunks, dim=0)

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 5 or tokens.shape[-1] != self.dim:
            raise ValueError(
                f"Expected [B,T,H,W,{self.dim}], got {tuple(tokens.shape)}"
            )
        batch, frames, height, width, channels = tokens.shape

        spatial = self.spatial_norm(tokens).permute(0, 1, 4, 2, 3)
        spatial = spatial.reshape(batch * frames, channels, height, width)
        spatial = self.spatial_depthwise(spatial)
        spatial = self.spatial_mlp(spatial)
        spatial = spatial.reshape(batch, frames, channels, height, width)
        tokens = tokens + spatial.permute(0, 1, 3, 4, 2)

        temporal = tokens.permute(0, 2, 3, 1, 4).reshape(
            batch * height * width,
            frames,
            channels,
        )
        temporal = self._temporal_attention(temporal)
        tokens = temporal.reshape(batch, height, width, frames, channels)
        tokens = tokens.permute(0, 3, 1, 2, 4).contiguous()
        return tokens + self.channel_mlp(self.channel_norm(tokens))


class WanOmega35Bridge(nn.Module):
    """Map Wan 8x8x4 raw latents to native VGGT-Omega patch tokens."""

    def __init__(self, cfg: WanOmega35BridgeCfg | None = None) -> None:
        super().__init__()
        self.cfg = cfg or WanOmega35BridgeCfg()
        cfg = self.cfg
        expected_frames = 1 + (cfg.latent_frames - 1) * cfg.temporal_compression
        expected_height = cfg.latent_height // 2
        expected_width = cfg.latent_width // 2
        if cfg.output_frames != expected_frames:
            raise ValueError(
                f"output_frames must equal {expected_frames}, got {cfg.output_frames}"
            )
        if (cfg.grid_height, cfg.grid_width) != (expected_height, expected_width):
            raise ValueError(
                "Omega35 grid must be the exact 2x packed Wan latent grid: "
                f"expected {(expected_height, expected_width)}, got "
                f"{(cfg.grid_height, cfg.grid_width)}"
            )
        if cfg.hidden_dim % cfg.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if cfg.temporal_chunk_size < 1:
            raise ValueError("temporal_chunk_size must be positive")

        self.spatial_packer = Omega35SpatialPacker(
            cfg.latent_channels,
            cfg.hidden_dim,
        )
        self.temporal_upsampler = CausalTemporalPhaseUpsampler(
            cfg.hidden_dim,
            cfg.temporal_compression,
        )
        self.phase_embedding = nn.Parameter(
            torch.zeros(cfg.temporal_compression, cfg.hidden_dim)
        )
        self.register_buffer(
            "time_embedding",
            self._sinusoidal_time_embedding(cfg.output_frames, cfg.hidden_dim),
            persistent=False,
        )
        self.blocks = nn.ModuleList(
            Omega35BridgeBlock(
                cfg.hidden_dim,
                cfg.num_heads,
                cfg.mlp_ratio,
                cfg.spatial_kernel_size,
                cfg.temporal_chunk_size,
            )
            for _ in range(cfg.num_blocks)
        )
        self.output_projection = nn.Linear(cfg.hidden_dim, cfg.output_dim)
        self.output_norm = nn.LayerNorm(cfg.output_dim)
        nn.init.constant_(self.output_norm.weight, cfg.output_scale_init)
        nn.init.constant_(self.output_norm.bias, cfg.output_bias_init)

    @staticmethod
    def _sinusoidal_time_embedding(frames: int, dim: int) -> Tensor:
        positions = torch.arange(frames, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / dim)
        )
        embedding = torch.zeros(1, frames, 1, 1, dim, dtype=torch.float32)
        embedding[0, :, 0, 0, 0::2] = torch.sin(positions * frequencies)
        embedding[0, :, 0, 0, 1::2] = torch.cos(positions * frequencies)
        return embedding

    def forward_grid(self, latent: Tensor) -> Tensor:
        cfg = self.cfg
        expected = (
            latent.shape[0],
            cfg.latent_channels,
            cfg.latent_frames,
            cfg.latent_height,
            cfg.latent_width,
        )
        if tuple(latent.shape) != expected:
            raise ValueError(f"Expected Wan latent {expected}, got {tuple(latent.shape)}")
        if not torch.isfinite(latent).all():
            raise ValueError("Wan latent contains NaN or Inf")

        batch = latent.shape[0]
        frames = latent.permute(0, 2, 1, 3, 4).reshape(
            batch * cfg.latent_frames,
            cfg.latent_channels,
            cfg.latent_height,
            cfg.latent_width,
        )
        packed = self.spatial_packer(frames)
        packed = packed.reshape(
            batch,
            cfg.latent_frames,
            cfg.hidden_dim,
            cfg.grid_height,
            cfg.grid_width,
        ).permute(0, 2, 1, 3, 4)
        expanded = self.temporal_upsampler(packed)

        phase_ids = torch.arange(cfg.output_frames, device=latent.device)
        phase_ids = torch.where(
            phase_ids == 0,
            torch.zeros_like(phase_ids),
            (phase_ids - 1) % cfg.temporal_compression,
        )
        phase = self.phase_embedding[phase_ids].view(
            1,
            cfg.output_frames,
            1,
            1,
            cfg.hidden_dim,
        )
        tokens = expanded.permute(0, 2, 3, 4, 1)
        tokens = tokens + phase.to(dtype=tokens.dtype)
        tokens = tokens + self.time_embedding.to(
            device=tokens.device,
            dtype=tokens.dtype,
        )
        for block in self.blocks:
            if cfg.use_checkpoint and self.training:
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)

        tokens = self.output_projection(tokens)
        tokens = self.output_norm(tokens.float()).to(dtype=latent.dtype)
        if not torch.isfinite(tokens).all():
            raise RuntimeError("WanOmega35Bridge returned NaN or Inf")
        return tokens.permute(0, 1, 4, 2, 3).contiguous()

    def forward(self, latent: Tensor) -> Tensor:
        grid = self.forward_grid(latent)
        return grid.permute(0, 1, 3, 4, 2).flatten(2, 3).contiguous()
