"""External Omega integration without redistributing its source or weights."""
import importlib

import torch
from torch import nn


def load_omega(checkpoint, device):
    try:
        module = importlib.import_module("vggt_omega.models")
    except ImportError as error:
        raise ImportError("Install the separately licensed VGGT-Omega package; see docs/DEPENDENCIES.md") from error
    model = module.VGGTOmega(enable_camera=True, enable_depth=True, enable_alignment=False)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload.get("state_dict", payload.get("model", payload))
    expected = set(model.state_dict())
    # Extra alignment-head parameters are unused, not a partial geometry load.
    model.load_state_dict({key: value for key, value in state.items() if key in expected}, strict=True)
    return model.to(device).eval().requires_grad_(False)


class _PatchTokens(nn.Module):
    def __init__(self, tokens):
        super().__init__()
        self.tokens = tokens

    def forward(self, images):
        if images.shape[0] != self.tokens.shape[0] * self.tokens.shape[1]:
            raise ValueError("Omega patch-embedding frame count changed")
        return {"x_norm_patchtokens": self.tokens.flatten(0, 1)}


def forward_full_sequence(model, tokens, image_size=(560, 560)):
    if tokens.ndim != 4 or tokens.shape[1:] != (49, 1225, 1024):
        raise ValueError("Omega requires [B,49,1225,1024] native normalized patch tokens")
    if image_size != (560, 560) or not torch.isfinite(tokens).all():
        raise ValueError("Expected finite tokens for 560-square geometry")
    native = getattr(model, "forward_from_patch_tokens", None)
    if callable(native):
        return native(tokens, image_size=image_size, frames_chunk_size=None)
    # Reuse upstream aggregation/head execution while bypassing the image encoder.
    # The zero RGB tensor supplies only dimensions; it is NOT generated/encoded RGB.
    if not hasattr(model, "aggregator") or not hasattr(model.aggregator, "patch_embed"):
        raise TypeError("Unsupported Omega API: no aggregator.patch_embed")
    if getattr(model, "dense_head", None) is None:
        raise TypeError("Omega must expose its depth dense_head")
    original = model.aggregator.patch_embed
    model.aggregator.patch_embed = _PatchTokens(tokens)
    def full49(module, args, kwargs):
        kwargs["frames_chunk_size"] = None
        return args, kwargs
    hook = model.dense_head.register_forward_pre_hook(full49, with_kwargs=True)
    try:
        carrier = tokens.new_zeros((tokens.shape[0], 49, 3, *image_size))
        return model(carrier)
    finally:
        hook.remove()
        model.aggregator.patch_embed = original


def predict_geometry(model, tokens):
    predictions = forward_full_sequence(model, tokens)
    depth = predictions["depth"]
    if depth.ndim == 5 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    elif depth.ndim == 5 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    if tuple(depth.shape) != (1, 49, 560, 560):
        raise ValueError(f"Invalid depth layout {tuple(depth.shape)}")
    if not torch.isfinite(depth).all() or (depth <= 0).any():
        raise ValueError("Invalid geometry depth")
    from vggt_omega.utils.pose_enc import encoding_to_camera
    w2c, intrinsics = encoding_to_camera(predictions["pose_enc"].float(), image_size_hw=(560, 560))
    return depth[0].float(), w2c[0].float(), intrinsics[0].float()
