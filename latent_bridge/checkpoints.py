"""Strict, tensor-only public checkpoint format, independent of training paths."""
import hashlib
import json
from pathlib import Path

import torch

from .models import WanOmega35Bridge, WanOmega35BridgeCfg, CompressedLatentToOmegaAdapter
from .models.vrae import crop_reference_frame_padding

FORMAT = "latent_bridge.v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_config(path):
    value = json.loads(Path(path).read_text())
    if value.get("route") not in ("wan", "vrae") or not isinstance(value.get("model"), dict):
        raise ValueError("Config needs route=wan|vrae and an explicit model object")
    return value


def build_bridge(config):
    if config["route"] == "wan":
        return WanOmega35Bridge(WanOmega35BridgeCfg(**config["model"]))
    if config["route"] == "vrae":
        return CompressedLatentToOmegaAdapter(**config["model"])
    raise ValueError("Unsupported bridge route")


def bridge_tokens(model, latent, route):
    if latent.ndim != 5 or not torch.isfinite(latent).all():
        raise ValueError("Expected a finite five-dimensional latent tensor")
    if route == "wan":
        return model(latent)
    if route != "vrae":
        raise ValueError("Unknown route")
    # Thirteen V-RAE chunks expand to 52 phases; remove the three synthetic prefix phases.
    grid = crop_reference_frame_padding(model(latent), compression_ratio=model.compression_ratio)
    return grid.permute(0, 1, 3, 4, 2).flatten(2, 3).contiguous()


def save_bridge(path, model, config, step):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    payload = dict(format=FORMAT, route=config["route"], model_config=config["model"],
                   step=int(step), model={k: v.detach().cpu() for k, v in model.state_dict().items()})
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_bridge(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ValueError("Not a public bridge checkpoint; use export-checkpoint first")
    config = dict(route=payload["route"], model=payload["model_config"])
    model = build_bridge(config)
    model.load_state_dict(payload["model"], strict=True)
    for tensor in model.state_dict().values():
        if not torch.isfinite(tensor).all():
            raise ValueError("Nonfinite checkpoint parameter")
    return model.to(device).eval(), config, int(payload["step"])


def export_checkpoint(source, config_path, output, step=None, trust_pickle=False):
    config = read_config(config_path)
    # Full legacy optimizer/RNG checkpoints may require pickle. Never silently fall back.
    payload = torch.load(source, map_location="cpu", weights_only=not trust_pickle)
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint must be a mapping")
    state = payload.get("model", payload.get("state_dict", payload))
    model = build_bridge(config)
    model.load_state_dict(state, strict=True)
    if any(not torch.isfinite(x).all() for x in model.state_dict().values()):
        raise ValueError("Nonfinite checkpoint")
    saved_step = payload.get("step")
    if step is not None and saved_step is not None and step != saved_step:
        raise ValueError("Requested step disagrees with checkpoint metadata")
    step = saved_step if step is None else step
    if step is None or int(step) < 0:
        raise ValueError("An explicit nonnegative checkpoint step is required")
    save_bridge(output, model, config, step)
    return dict(format=FORMAT, step=int(step), route=config["route"], sha256=sha256(output))
