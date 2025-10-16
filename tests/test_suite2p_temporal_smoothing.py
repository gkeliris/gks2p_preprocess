import unittest
import os
import numpy as np
from gks2p import suite2p_temporal_smoothing as sts

class TestTemporalSmoothingImport(unittest.TestCase):
    def test_import_and_binspec(self):
        # Basic smoke test: create a dummy BinSpec and check methods exist
        tmp_path = os.path.join(os.path.dirname(__file__), 'dummy.bin')
        # create an empty small file matching dims
        Lx, Ly = 4, 3
        dtype = 'int16'
        n_frames = 2
        n_pixels = Lx * Ly
        arr = np.zeros(n_frames * n_pixels, dtype=dtype)
        with open(tmp_path, 'wb') as f:
            f.write(arr.tobytes())

        spec = sts.BinSpec(path=tmp_path, Lx=Lx, Ly=Ly, nchannels=1, dtype=dtype)
        self.assertEqual(spec.n_pixels(), n_pixels)
        self.assertEqual(spec.n_frames(), n_frames)
        os.remove(tmp_path)

if __name__ == '__main__':
    unittest.main()
