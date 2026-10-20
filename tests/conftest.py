import pytest
import torch

torch.set_num_threads(1)


@pytest.fixture(params=['wan', 'vrae'])
def bridge_case(request):
    if request.param == 'wan':
        config = dict(route='wan', model=dict(latent_channels=2, latent_frames=3,
                      latent_height=4, latent_width=4, grid_height=2, grid_width=2,
                      output_frames=9, hidden_dim=8, output_dim=8, num_blocks=1,
                      num_heads=2, mlp_ratio=2, spatial_kernel_size=3,
                      temporal_chunk_size=2, use_checkpoint=True))
        latent = torch.randn(1, 2, 3, 4, 4)
    else:
        config = dict(route='vrae', model=dict(dim=8, hidden_dim=16,
                      temporal_heads=2, compression_ratio=4))
        latent = torch.randn(1, 3, 8, 2, 2)
    return config, latent
