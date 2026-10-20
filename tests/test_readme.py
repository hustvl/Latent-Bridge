"""Check documentation syntax and CLI arguments without launching any jobs."""
import argparse
import ast
import importlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
from unittest.mock import patch
from urllib.parse import unquote, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = {'prepare_data.py': 'prepare', 'train.py': 'train', 'infer.py': 'infer',
           'evaluate.py': 'evaluate', 'export_checkpoint.py': 'export-checkpoint'}
DOCUMENTS = ['README.md', 'README.zh-CN.md', 'docs/DATA.md',
             'docs/DEPENDENCIES.md', 'docs/EVALUATION.md', 'docs/PREPARATION.md']


@pytest.mark.parametrize('name', DOCUMENTS)
def test_documented_cli_arguments(name):
    original = argparse.ArgumentParser.parse_args

    class Parsed(Exception):
        pass

    def parse_only(parser, *args, **kwargs):
        namespace = original(parser, *args, **kwargs)
        if hasattr(namespace, 'config'):
            assert (ROOT / namespace.config).is_file()
        # Stop before checkpoint access, GPU use, training, evaluation or export.
        raise Parsed()

    commands = set()
    document = (ROOT / name).read_text()
    for block in re.findall(r'```bash\n(.*?)```', document, flags=re.S):
        for line in block.replace('\\\n', ' ').splitlines():
            argv = shlex.split(line, comments=True)
            if not argv:
                continue
            assert argv[0] != 'latent-bridge', 'Use a plain Python script'
            assert 'latent_bridge.cli' not in argv, 'Use a plain Python script'
            if argv[0] == 'python' and len(argv) > 1 and argv[1] in SCRIPTS:
                assert (ROOT / argv[1]).is_file()
                script, arguments = argv[1], argv[2:]
            elif argv[0] == 'torchrun':
                position = argv.index('train.py')
                script, arguments = 'train.py', argv[position + 1:]
            else:
                continue
            commands.add(SCRIPTS[script])
            main = importlib.import_module(Path(script).stem).main
            with patch.object(argparse.ArgumentParser, 'parse_args', parse_only):
                try:
                    main(arguments)
                except Parsed:
                    pass
                except SystemExit as error:
                    assert '--help' in argv and error.code == 0
                else:
                    raise AssertionError('Command unexpectedly executed')
    if name.startswith('README'):
        assert commands == {'prepare', 'train', 'infer', 'evaluate'}
        assert 'export_checkpoint.py' not in document
        assert 'scripts/package_release.py' not in document


@pytest.mark.parametrize('name', DOCUMENTS)
def test_local_document_links_exist(name):
    document = ROOT / name
    for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', document.read_text()):
        link = urlsplit(target)
        if link.scheme or link.netloc or not link.path:
            continue
        assert (document.parent / unquote(link.path)).is_file(), (name, target)


@pytest.mark.parametrize('name', DOCUMENTS)
def test_documented_snippet_syntax(name):
    for language, block in re.findall(r'```(bash|python|jsonl|json)\n(.*?)```', (ROOT / name).read_text(), flags=re.S):
        if language == 'python':
            ast.parse(block)
        elif language == 'json':
            json.loads(block)
        elif language == 'jsonl':
            for line in block.splitlines():
                if line.strip():
                    json.loads(line)
        else:
            result = subprocess.run(['bash', '-n'], input=block, text=True, capture_output=True)
            assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('script', list(SCRIPTS))
def test_plain_python_entrypoint(script):
    result = subprocess.run([sys.executable, str(ROOT / script), '--help'],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert f'usage: python {script} ' in result.stdout


@pytest.mark.parametrize('script', list(SCRIPTS))
def test_entrypoints_own_their_parsers(script):
    source = (ROOT / script).read_text()
    tree = ast.parse(source)
    assert 'sys.path' not in source
    assert 'latent_bridge.cli' not in source
    assert '.add_subparsers(' not in source
    assert any(isinstance(node, ast.FunctionDef) and node.name == 'parse_args' for node in tree.body)
    assert not (ROOT / 'latent_bridge/cli.py').exists()
