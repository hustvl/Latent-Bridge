import zipfile
import pytest
import torch
from torch import nn
from latent_bridge.omega import forward_full_sequence
from package_release import package


def tokens():
    return torch.zeros(1).expand(1, 49, 1225, 1024)


def test_native_omega_no_temporal_chunk():
    class Fake:
        def forward_from_patch_tokens(self, x, image_size, frames_chunk_size):
            assert x.shape[1] == 49 and image_size == (560, 560)
            assert frames_chunk_size is None
            return 'full49'
    assert forward_full_sequence(Fake(), tokens()) == 'full49'
    with pytest.raises(ValueError, match='49'):
        forward_full_sequence(Fake(), tokens()[:, :7])


@pytest.mark.parametrize('fail', [False, True])
def test_fallback_bypasses_encoder_and_restores(fail):
    class Encoder(nn.Module):
        def forward(self, x):
            raise AssertionError('Image encoder must not run')
    class Head(nn.Module):
        def forward(self, x, frames_chunk_size=7):
            assert frames_chunk_size is None
            if fail:
                raise RuntimeError('deliberate failure')
            return x
    class Fake(nn.Module):
        def __init__(self):
            super().__init__()
            self.aggregator = nn.Module()
            self.aggregator.patch_embed = Encoder()
            self.dense_head = Head()
        def forward(self, images):
            x = self.aggregator.patch_embed(images.flatten(0, 1))['x_norm_patchtokens']
            assert x.shape == (49, 1225, 1024)
            return self.dense_head(x, frames_chunk_size=7)
    model = Fake()
    original = model.aggregator.patch_embed
    if fail:
        with pytest.raises(RuntimeError, match='deliberate'):
            forward_full_sequence(model, tokens())
    else:
        assert forward_full_sequence(model, tokens()).shape[0] == 49
    assert model.aggregator.patch_embed is original
    assert not model.dense_head._forward_pre_hooks


def test_packaging(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    (root / 'README.md').write_text('safe source\n')
    zip_path = tmp_path / 'release.zip'
    digest = package(root, zip_path)
    assert len(digest) == 64
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.namelist() == ['latent-bridge/README.md']
        assert archive.testzip() is None
    with pytest.raises(ValueError, match='outside'):
        package(root, root / 'bad.zip')
    with pytest.raises(FileExistsError):
        package(root, zip_path)
    (root / 'weights.pt').write_bytes(b'binary')
    with pytest.raises(ValueError, match='source file type'):
        package(root, tmp_path / 'bad.zip')
    assert not (tmp_path / 'bad.zip').exists()


def test_packaging_excludes_generated_files(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    (root / 'train.py').write_text('pass\n')
    (root / '.DS_Store').write_bytes(b'\x00')
    for name in ('.git', '__pycache__', 'build', 'example.egg-info'):
        folder = root / name
        folder.mkdir()
        (folder / 'generated.bin').write_bytes(b'\x00')
    output = tmp_path / 'source.zip'
    package(root, output)
    with zipfile.ZipFile(output) as archive:
        assert archive.namelist() == ['latent-bridge/train.py']


def test_packaging_rejects_symlink(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    (root / 'linked.py').symlink_to(tmp_path / 'outside.py')
    with pytest.raises(ValueError, match='symlink'):
        package(root, tmp_path / 'source.zip')


@pytest.mark.parametrize('content', [b'\xff', b'\x00'])
def test_packaging_rejects_binary_text(tmp_path, content):
    root = tmp_path / 'source'
    root.mkdir()
    (root / 'config.txt').write_bytes(content)
    with pytest.raises(ValueError, match='binary|NUL'):
        package(root, tmp_path / 'source.zip')


def test_packaging_includes_method_figure(tmp_path):
    import base64

    root = tmp_path / 'source'
    assets = root / 'assets'
    assets.mkdir(parents=True)
    png = base64.b64decode(
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a6X8AAAAASUVORK5CYII=')
    (assets / 'method-overview.png').write_bytes(png)
    output = tmp_path / 'source.zip'
    package(root, output)
    with zipfile.ZipFile(output) as archive:
        assert archive.read('latent-bridge/assets/method-overview.png') == png
    (assets / 'unapproved.png').write_bytes(png)
    with pytest.raises(ValueError, match='source file type'):
        package(root, tmp_path / 'unapproved.zip')
