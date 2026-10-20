# Dependency and licensing boundaries

The Apache-2.0 grant applies to this project's code, not to every component of
an end-to-end system. No external model code or dataset assets are vendored.

| Component | Used for | Included here? |
|---|---|---|
| PyTorch, NumPy, SciPy | Bridge, geometry and reference nearest-neighbor metrics | Installed as dependencies, own licenses |
| VGGT-Omega | Frozen geometry aggregation/depth/camera decoding | No source or weights |
| DINOv3 | Omega teacher image encoder | No source or weights |
| Wan2.1 / generator implementation | Obtain native WAN latents | No source or weights |
| V-RAE / generator / normalizer | Obtain native V-RAE feature latents | No source or weights |
| Open3D, PyTorch3D | Optional voxel+FPS metric sampling | No source; optional external install |
| Gen3R, Aether, FantasyWorld | Historical comparison methods | Not implemented or bundled here |

## Installation

Cached-feature Bridge training and CPU tests do not need Omega. Raw-video cache
preparation and geometry inference do: first obtain an authorized Omega distribution and follow its dependency
instructions. Run these commands from this repository root if that distribution
is an installable Python package:

```bash
# Replace with your actual separately obtained source directory.
export OMEGA_SOURCE="$PWD/third_party/vggt-omega"
python -m pip install -e "$OMEGA_SOURCE"
python -c "from vggt_omega.models import VGGTOmega; from vggt_omega.utils.pose_enc import encoding_to_camera; print('Omega imports OK')"
python -m pip check
```

This placeholder directory is not bundled or downloaded. If the distribution has
no package metadata, follow its own instructions and add its Python package root
to `PYTHONPATH` instead. Prepare its authorized weights at `weights/omega.pt`.
There is no certified public backend revision/CUDA environment lock in this release;
successful imports do not establish real-weight inference compatibility.

Before CUDA execution check the active Python environment:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); print('BF16:', torch.cuda.is_bf16_supported())"
```

Remove `--bf16` for FP32 if unsupported. Do not silently split the full49 geometry
sequence to fit memory. CPU smoke tests establish neither GPU memory requirements
nor latency. For a validated environment, retain `python -m pip freeze` output.

## Omega backend contract

Install an authorized distribution exposing `vggt_omega.models.VGGTOmega` and
`vggt_omega.utils.pose_enc.encoding_to_camera`. The integration expects constructor
flags `enable_camera`, `enable_depth`, `enable_alignment`, plus `aggregator.patch_embed`
and `dense_head`. It accepts native `[B,49,1225,1024]` normalized DINO patch tokens.

If the backend exposes `forward_from_patch_tokens`, call it with
`frames_chunk_size=None`. Otherwise the integration temporarily replaces only
`aggregator.patch_embed` with a token-provider module, reusing upstream execution,
and forces the dense head to receive all frames. The original module and hook are
restored even on error. Use one model instance per worker; this temporary module
replacement is not thread-safe. A zero RGB shape carrier is allocated in this
fallback; it is not decoded RGB and the original DINO encoder does not execute.

An incompatible API raises an error. Before claiming production inference, pin
the exact externally licensed revision and verify parity with its RGB/token
reference path on real weights. The release does not invent an upstream commit
or claim public-backend parity from mock tests.

Check each model's own redistribution and research/commercial-use terms before
publishing weights. Pin and archive an approved dependency environment after
testing on the intended CUDA architecture. CPU bridge tests do not establish
GPU operator support, latency or peak memory for full geometry reconstruction.
