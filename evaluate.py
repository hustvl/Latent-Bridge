"""Evaluate dense points with an explicit camera/filter/unit protocol."""
import argparse
import json
from pathlib import Path

import numpy as np

from latent_bridge.checkpoints import sha256
from latent_bridge.evaluation import evaluate_bundle


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog='python evaluate.py', description=__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--filter', choices=['none', 'log_mad', 'log_mad_multiview', 'log_mad_multiview_safe_v1'], default='none')
    parser.add_argument('--camera', choices=['omega', 'gt_rotation_direction'], default='omega')
    parser.add_argument('--sampling', choices=['all', 'voxel_fps'], default='all')
    parser.add_argument('--unit', choices=['native', 'meters'], required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    with np.load(args.bundle, allow_pickle=False) as bundle:
        report, mask = evaluate_bundle(bundle, filter_mode=args.filter, camera_mode=args.camera,
                                       sampling=args.sampling, unit=args.unit)
    report['bundle_sha256'] = sha256(args.bundle)
    np.savez_compressed(output / 'prediction_mask.npz', mask=mask)
    with (output / 'metrics.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
