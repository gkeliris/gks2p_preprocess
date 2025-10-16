import os
import tempfile
import numpy as np
from gks2p import suite2p_temporal_smoothing as sts


def test_block_average_small():
    # Create a small 1-channel bin with 6 frames, 2x2 pixels
    Lx, Ly = 2, 2
    n_frames = 6
    dtype = 'int16'
    data = np.arange(n_frames * Lx * Ly, dtype=dtype)

    fd, path = tempfile.mkstemp(suffix='.bin')
    os.close(fd)
    with open(path, 'wb') as f:
        f.write(data.tobytes())

    spec = sts.BinSpec(path=path, Lx=Lx, Ly=Ly, nchannels=1, dtype=dtype)
    out_path, backup = sts.block_average(spec, x=2, out_dtype='int16', inplace=False)

    # read out memmap and check shape
    mm = np.memmap(out_path, dtype=dtype, mode='r')
    # expected frames = 6 // 2 = 3, pixels = 4
    assert mm.size == 3 * (Lx * Ly)

    # load as frames
    frames = mm.reshape(3, Lx * Ly)
    # compute expected per-pixel averages across frames in each block
    frames_in = data.reshape(n_frames, Lx * Ly)
    expected_block0 = frames_in[0:2].mean(axis=0).astype(np.int16)
    assert frames[0,0] == int(expected_block0[0])

    # cleanup
    os.remove(path)
    os.remove(out_path)
