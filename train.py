"""Train a Bridge from cached native latents and teacher tokens."""
import argparse

from latent_bridge.train import train


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog='python train.py', description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--save-every', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--bf16', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    return train(parse_args(argv))


if __name__ == "__main__":
    main()
