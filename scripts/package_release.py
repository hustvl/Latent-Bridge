"""Create a source archive, excluding history, caches and build outputs."""
import argparse
import hashlib
from pathlib import Path
import zipfile

SKIP = {'.git', '.venv', '__pycache__', '.pytest_cache', 'build', 'dist', '.DS_Store', '__MACOSX'}
EXTENSIONS = {'.py', '.md', '.json', '.jsonl', '.toml', '.yml', '.yaml', '.in', '.txt'}
SPECIAL = {'LICENSE', 'NOTICE', '.gitignore'}
DOCUMENT_ASSETS = {Path('assets/method-overview.png')}


def source_files(root):
    files = []
    for path in sorted(root.rglob('*')):
        if any(part in SKIP or part.endswith('.egg-info') for part in path.relative_to(root).parts):
            continue
        if path.is_symlink():
            raise ValueError(f'{path.relative_to(root)}: symlink not allowed')
        if not path.is_file():
            continue
        if path.relative_to(root) in DOCUMENT_ASSETS:
            if path.stat().st_size > 2_000_000 or not path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n'):
                raise ValueError(f'{path.relative_to(root)}: invalid documentation PNG')
            files.append(path)
            continue
        if path.suffix not in EXTENSIONS and path.name not in SPECIAL:
            raise ValueError(f'{path.relative_to(root)}: not an approved source file type')
        if path.stat().st_size > 2_000_000:
            raise ValueError(f'{path.relative_to(root)}: source file exceeds size limit')
        try:
            content = path.read_text(encoding='utf-8')
        except UnicodeDecodeError as error:
            raise ValueError(f'{path.relative_to(root)}: binary content') from error
        if '\x00' in content:
            raise ValueError(f'{path.relative_to(root)}: NUL byte in text')
        files.append(path)
    return files


def package(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or root in output.parents:
        raise ValueError('Archive must be outside the source tree')
    files = source_files(root)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            info = zipfile.ZipInfo('latent-bridge/' + path.relative_to(root).as_posix(), (2026, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    return hashlib.sha256(output.read_bytes()).hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(package(args.root, args.output))
