"""Export trusted legacy Bridge weights to the public tensor-only format."""
import argparse
import json

from latent_bridge.checkpoints import export_checkpoint


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog='python export_checkpoint.py', description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--step', type=int)
    parser.add_argument('--trust-legacy-pickle', action='store_true', help='Unsafe for untrusted files: allows Python pickle execution')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = export_checkpoint(args.source, args.config, args.output, args.step, args.trust_legacy_pickle)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
