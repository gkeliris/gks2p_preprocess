import os
import sys
import types
import numpy as np
import tempfile

# Provide lightweight fakes for heavy optional imports used at module import time
# (preprocess.py imports `suite2p` and `fissa` at top-level). Inject minimal
# modules into sys.modules so tests can import the package without the real
# dependencies installed.
if 'suite2p' not in sys.modules:
    fake_suite2p = types.ModuleType('suite2p')
    fake_suite2p.registration = types.ModuleType('suite2p.registration')
    fake_suite2p.io = types.ModuleType('suite2p.io')
    # minimal BinaryFile stub used only for shape access in some flows
    class _BinStub:
        def __init__(self, Ly=None, Lx=None, filename=None, n_frames=None):
            self.shape = (n_frames or 0, Ly or 0, Lx or 0)
    fake_suite2p.io.BinaryFile = _BinStub
    sys.modules['suite2p'] = fake_suite2p
    sys.modules['suite2p.registration'] = fake_suite2p.registration
    sys.modules['suite2p.io'] = fake_suite2p.io

if 'fissa' not in sys.modules:
    sys.modules['fissa'] = types.ModuleType('fissa')

# scanreader may be imported by mkops; provide a minimal stub
if 'scanreader' not in sys.modules:
    sys.modules['scanreader'] = types.ModuleType('scanreader')

if 'natsort' not in sys.modules:
    sys.modules['natsort'] = types.ModuleType('natsort')
    # provide natsorted as identity sorted
    def natsorted(seq):
        return sorted(seq)
    sys.modules['natsort'].natsorted = natsorted

from gks2p.preprocess import gks2p_smooth


def make_small_bin(path, Lx, Ly, nchannels=1, dtype=np.int16, n_frames=8):
    P = Lx * Ly * nchannels
    # generate in a safe integer dtype then cast to target dtype
    tmp = np.arange(n_frames * P, dtype=np.int64) % (2**15)
    data = tmp.reshape(n_frames, P).astype(dtype, copy=False)
    mm = np.memmap(path, dtype=dtype, mode='w+', shape=(n_frames, P))
    mm[:] = data
    mm.flush()
    return path


def test_gks2p_smooth_helper_block(tmp_path, monkeypatch):
    """Create a tiny ops.npy and .bin and run gks2p_smooth with block averaging."""
    # Create a fake dataset row structure: minimal pandas-like access via list/iloc

    # temp directories mimic save_path0 and fast_disk
    tmpdir = str(tmp_path)
    save_path0 = os.path.join(tmpdir, 'save')
    fast_disk = os.path.join(tmpdir, 'fast')
    plane_folder = os.path.join(save_path0, 'suite2p_orig', 'plane0')
    os.makedirs(plane_folder, exist_ok=True)
    os.makedirs(os.path.join(fast_disk, 'suite2p', 'plane0'), exist_ok=True)

    # Build ops dict and save to plane orig ops and dataset-level ops
    ops_plane = {
        'Lx': 4,
        'Ly': 2,
        'save_path0': save_path0,
        'save_folder': 'suite2p_orig',
        'fast_disk': fast_disk,
    }
    ops_path_plane = os.path.join(plane_folder, 'ops.npy')
    np.save(ops_path_plane, ops_plane, allow_pickle=True)

    # dataset-level ops -> saved under save_path0 as ops_orig.npy
    ops_ds = {'dx': [0], 'save_path0': save_path0, 'fast_disk': fast_disk}
    np.save(os.path.join(save_path0, 'ops_orig.npy'), ops_ds, allow_pickle=True)

    # Build a tiny .bin in fast_disk
    binfile = os.path.join(fast_disk, 'suite2p', 'plane0', 'data_raw.bin')
    make_small_bin(binfile, Lx=ops_plane['Lx'], Ly=ops_plane['Ly'], n_frames=8)

    # Build a tiny dataset-like object with iloc access to avoid pandas dependency
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
    df = SimpleDF([row])

    # Monkeypatch gks2p_path to return the save_path0 and fast_disk we created
    from gks2p.preprocess import gks2p_path

    def fake_gks2p_path(dat, basepath, pathType='save_path0'):
        if pathType == 'save_path0':
            return save_path0
        elif pathType == 'fast_disk':
            return fast_disk
        return save_path0

    monkeypatch.setattr('gks2p.preprocess.gks2p_path', fake_gks2p_path)

    # Call smoothing: block averaging with x=2 -> output should be created
    gks2p_smooth(df, basepath=tmpdir, pipeline='orig', method='block', method_kwargs={'x': 2}, inplace=False, update_ops=False, verbose=False)

    # Verify that some output file exists (search for .bin with '.binavg2' tag)
    found = False
    for root, _, files in os.walk(fast_disk):
        for f in files:
            if '.binavg2' in f and f.endswith('.bin'):
                found = True
    assert found, 'Expected binavg output not found'
