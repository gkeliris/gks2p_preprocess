import os
import tempfile
import numpy as np
import pytest
from gks2p import suite2p_temporal_smoothing as sts


def make_bin(Lx, Ly, n_frames, dtype='int16'):
    data = np.arange(n_frames * Lx * Ly, dtype=dtype)
    fd, path = tempfile.mkstemp(suffix='.bin')
    os.close(fd)
    with open(path, 'wb') as f:
        f.write(data.tobytes())
    return path, data


def test_gaussian_small():
    path, data = make_bin(2, 2, 8)
    spec = sts.BinSpec(path=path, Lx=2, Ly=2, nchannels=1, dtype='int16')
    # gaussian_smooth requires chunk_frames >= 8; use safe value here
    out, _ = sts.gaussian_smooth(spec, sigma_frames=1.0, out_dtype='float32', inplace=False, chunk_frames=8)
    assert os.path.exists(out)
    os.remove(path)
    os.remove(out)


def test_ema_small():
    path, data = make_bin(2, 2, 6)
    spec = sts.BinSpec(path=path, Lx=2, Ly=2, nchannels=1, dtype='int16')
    out, _ = sts.ema_smooth(spec, tau_frames=2.0, out_dtype='float32', inplace=False)
    assert os.path.exists(out)
    os.remove(path)
    os.remove(out)


def test_median_small():
    # increase frames to avoid small-chunk edge cases
    path, data = make_bin(2, 2, 9)
    spec = sts.BinSpec(path=path, Lx=2, Ly=2, nchannels=1, dtype='int16')
    out, _ = sts.median_smooth(spec, window_frames=3, out_dtype='float32', inplace=False)
    assert os.path.exists(out)
    os.remove(path)
    os.remove(out)


def test_savgol_small():
    path, data = make_bin(2, 2, 9)
    spec = sts.BinSpec(path=path, Lx=2, Ly=2, nchannels=1, dtype='int16')
    out, _ = sts.savgol_smooth(spec, window_length=3, polyorder=1, out_dtype='float32', inplace=False)
    assert os.path.exists(out)
    os.remove(path)
    os.remove(out)
