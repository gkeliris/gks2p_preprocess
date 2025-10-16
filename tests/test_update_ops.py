import os
import sys
import types
import numpy as np
from types import SimpleNamespace

# Minimal fakes for heavy optional modules
if 'suite2p' not in sys.modules:
    fake_suite2p = types.ModuleType('suite2p')
    # include a registration submodule to satisfy `from suite2p import registration`
    fake_suite2p.registration = types.ModuleType('suite2p.registration')
    fake_suite2p.io = types.ModuleType('suite2p.io')
    class _BinStub:
        def __init__(self, Ly=None, Lx=None, filename=None, n_frames=None):
            # We won't actually read frames in this test
            self.shape = (n_frames or 0, Ly or 0, Lx or 0)
    fake_suite2p.io.BinaryFile = _BinStub
    sys.modules['suite2p'] = fake_suite2p
    sys.modules['suite2p.registration'] = fake_suite2p.registration
    sys.modules['suite2p.io'] = fake_suite2p.io
if 'fissa' not in sys.modules:
    sys.modules['fissa'] = types.ModuleType('fissa')
if 'scanreader' not in sys.modules:
    sys.modules['scanreader'] = types.ModuleType('scanreader')
if 'natsort' not in sys.modules:
    natsort_mod = types.ModuleType('natsort')
    def natsorted(seq):
        return sorted(seq)
    natsort_mod.natsorted = natsorted
    sys.modules['natsort'] = natsort_mod

from gks2p.preprocess import gks2p_smooth
from gks2p.suite2p_temporal_smoothing import BinSpec


def make_small_bin(path, Lx, Ly, nchannels=1, dtype=np.int16, n_frames=8):
    P = Lx * Ly * nchannels
    tmp = np.arange(n_frames * P, dtype=np.int64) % (2**15)
    data = tmp.reshape(n_frames, P).astype(dtype, copy=False)
    mm = np.memmap(path, dtype=dtype, mode='w+', shape=(n_frames, P))
    mm[:] = data
    mm.flush()
    return path


def test_update_ops_rescales_fs(tmp_path, monkeypatch):
    tmpdir = str(tmp_path)
    save_path0 = os.path.join(tmpdir, 'save')
    fast_disk = os.path.join(tmpdir, 'fast')
    plane_folder = os.path.join(save_path0, 'suite2p_orig', 'plane0')
    os.makedirs(plane_folder, exist_ok=True)
    os.makedirs(os.path.join(fast_disk, 'suite2p', 'plane0'), exist_ok=True)

    # ops with explicit nframes and fs
    ops_plane = {
        'Lx': 4,
        'Ly': 2,
        'save_path0': save_path0,
        'save_folder': 'suite2p_orig',
        'fast_disk': fast_disk,
        'nframes': 8,
        'fs': 30.0,  # Hz
    }
    ops_path_plane = os.path.join(plane_folder, 'ops.npy')
    np.save(ops_path_plane, ops_plane, allow_pickle=True)

    # dataset-level ops
    ops_ds = {'dx': [0], 'save_path0': save_path0, 'fast_disk': fast_disk}
    np.save(os.path.join(save_path0, 'ops_orig.npy'), ops_ds, allow_pickle=True)

    binfile = os.path.join(fast_disk, 'suite2p', 'plane0', 'data_raw.bin')
    make_small_bin(binfile, Lx=ops_plane['Lx'], Ly=ops_plane['Ly'], n_frames=8)

    # Minimal dataset-like object
    row = SimpleNamespace(cohort='c', mouseID='m', day='d', session='s', expID='e', rawPath='')
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
    df = SimpleDF([row])

    monkeypatch.setattr('gks2p.preprocess.gks2p_path', lambda dat, basepath, pathType='save_path0': save_path0 if pathType=='save_path0' else fast_disk)

    # Run smoothing with downsample_factor (via method_kwargs). Use gaussian smoothing but small sigma.
    gks2p_smooth(df, basepath=tmpdir, pipeline='orig', method='gaussian', method_kwargs={'sigma_frames': 1.0, 'downsample_factor': 2, 'downsample_mode': 'decimate'}, inplace=False, update_ops=True, verbose=False)

    # Check ops updated
    updated_ops = np.load(ops_path_plane, allow_pickle=True).item()
    assert updated_ops['nframes'] == 4, 'nframes should be halved after downsampling by 2'
    # fs should be scaled by new/old = 4/8 = 0.5 => new fs = 15.0
    assert abs(updated_ops['fs'] - 15.0) < 1e-6, f"fs expected 15.0, got {updated_ops.get('fs')}"
