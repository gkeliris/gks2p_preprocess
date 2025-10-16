"""Example script: call gks2p_smooth programmatically or via CLI-like args.

Usage (from project root):
    python scripts/gks2p_smooth_example.py --help

This script shows a minimal programmatic example and a CLI wrapper that
parses some arguments and calls `gks2p_smooth` from `gks2p.preprocess`.
"""

import argparse
import os
from pathlib import Path
from gks2p.preprocess import gks2p_smooth


def example_programmatic(ds, basepath):
    # Example: call block averaging x=2 on dataset
    gks2p_smooth(ds, basepath=basepath, pipeline='orig', method='block', method_kwargs={'x': 2}, inplace=False, update_ops=False, verbose=True)


def build_fake_ds(save_path0, fast_disk):
    # Minimal dataset-like object with iloc
    from types import SimpleNamespace
    class SimpleDF:
        def __init__(self, rows):
            self._rows = rows
        @property
        def iloc(self):
            return self
        def __len__(self):
            return len(self._rows)
        def __getitem__(self, idx):
            return self._rows[idx]
    row = SimpleNamespace(cohort='c', mouseID='m', day='d', session='s', expID='e', rawPath='')
    return SimpleDF([row])


def main():
    p = argparse.ArgumentParser(description='Example: call gks2p_smooth')
    p.add_argument('--basepath', required=False, default='.', help='Basepath where ops/save folders are located')
    p.add_argument('--method', choices=['block','gaussian','ema','median','savgol'], default='block')
    p.add_argument('--x', type=int, default=2, help='block size for block averaging')
    args = p.parse_args()

    # For demonstration, we assume user prepared save_path0 and fast_disk
    save_path0 = os.path.join(args.basepath, 'example_save')
    fast_disk = os.path.join(args.basepath, 'example_fast')

    ds = build_fake_ds(save_path0, fast_disk)
    method_kwargs = {'x': args.x} if args.method == 'block' else {}

    example_programmatic(ds, args.basepath)


if __name__ == '__main__':
    main()
