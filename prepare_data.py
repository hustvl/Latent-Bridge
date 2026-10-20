"""Create matched latent/teacher training pairs from raw videos."""
import argparse

from latent_bridge.prepare import prepare


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog='python prepare_data.py', description=__doc__)
    parser.add_argument('--config', required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--videos', help='Recursively scan a raw-video directory; one window per video')
    source.add_argument('--manifest', help='JSONL id/video/start_frame/stride for explicit windows')
    parser.add_argument('--output', required=True, help='New cache directory')
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--trust-legacy-pickle', action='store_true', help='Unsafe for untrusted V-RAE checkpoints; no automatic fallback')
    return parser.parse_args(argv)


def main(argv=None):
    return prepare(parse_args(argv))


if __name__ == '__main__':
    main()
