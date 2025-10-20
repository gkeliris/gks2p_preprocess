import os
import numpy as np
import types
import builtins

from gks2p import preprocess


def test_register_uses_ops_badframes(tmp_path, monkeypatch, capsys):
    """Ensure gks2p_register populates opsPlane['badframes'] from
    dataset-level .npy if present, and passes it into the suite2p registration
    wrapper.

    This test avoids importing real suite2p by monkeypatching a fake
    `suite2p` module with a registration_wrapper that captures the ops arg.
    """
    # Prepare a fake dataset (pandas-like) as a list-like with .iloc
    class FakeRow:
        def __init__(self, rawPath, cohort='c', mouseID='m', timepoint='t', session='s', expID='e'):
            self.rawPath = rawPath
            self.cohort = cohort
            self.mouseID = mouseID
            self.timepoint = timepoint
            self.session = session
            self.expID = expID

    class FakeDF:
        def __init__(self, rows):
            self._rows = rows
        def __len__(self):
            return len(self._rows)
        def iloc(self, i):
            return self._rows[i]
        # pandas uses .iloc[index] property access; emulate it
        @property
        def iloc(self):
            return self
        def __call__(self, i):
            return self._rows[i]
        def __getitem__(self, i):
            return self._rows[i]

    # Create a temporary rawPath and write bad_frames.npy there
    raw_dir = tmp_path / "rawdata"
    raw_dir.mkdir()
    bad_frames = np.array([0, 5, 6])
    np.save(os.path.join(raw_dir, 'bad_frames.npy'), bad_frames)

    # Create minimal ops structure files expected by gks2p_register
    save_path0 = tmp_path / "s2p_analysis" / "c" / "m" / "t" / "s" / "e"
    fast_disk = tmp_path / "s2p_binaries" / "c" / "m" / "t" / "s" / "e"
    plane_folder = save_path0 / 'suite2p_orig' / 'plane0'
    bin_folder = fast_disk / 'suite2p' / 'plane0'
    plane_folder.mkdir(parents=True)
    bin_folder.mkdir(parents=True)

    # Minimal ops dicts
    ops_dataset = {
        'save_path0': str(save_path0),
        'fast_disk': str(fast_disk),
        'save_folder': 'suite2p_orig',
        'save_path': str(plane_folder),
        'bruker': True,
        # include keys that gks2p_register expects
        'switch_chan': 0,
        'align_by_chan': False,
    }

    ops_plane = {
        'Ly': 10,
        'Lx': 10,
        'yrange': [0, 10],
        'xrange': [0, 10],
        'meanImg': np.zeros((10, 10), dtype=np.float32),
        'two_step_registration': False,
        'keep_movie_raw': False
    }

    # Save plane ops as suite2p_orig plane0 ops.npy
    np.save(os.path.join(plane_folder, 'ops.npy'), ops_plane)

    # Also create a dummy data_raw.bin file with shape metadata for BinaryFile
    # We'll simulate suite2p.io.BinaryFile by monkeypatching it later, so just create an empty file
    open(os.path.join(bin_folder, 'data_raw.bin'), 'wb').close()
    open(os.path.join(bin_folder, 'data.bin'), 'wb').close()

    # Build fake dataset and save ops file used by gks2p_loadOps
    fake_row = FakeRow(rawPath=str(raw_dir))
    ds = FakeDF([fake_row])

    # Make gks2p_loadOps return our dataset-level ops (ops_dataset)
    def fake_loadOps(_ds, _basepath, pipeline='orig'):
        return [ops_dataset]

    monkeypatch.setattr(preprocess, 'gks2p_loadOps', fake_loadOps)

    # Create a fake suite2p module to capture registration call
    captured = {}

    class FakeBinaryFile:
        def __init__(self, Ly, Lx, filename, n_frames=None):
            # when used as f.shape[0] in code, emulate shape attr
            self._n = 100
        @property
        def shape(self):
            return (self._n, )
        def __getitem__(self, key):
            # Return an array slice used for computing ref image; return zeros
            if isinstance(key, slice):
                return np.zeros((10, 10), dtype=np.int16)
            elif isinstance(key, (list, np.ndarray)):
                return np.zeros((len(key), 10, 10), dtype=np.int16)
            else:
                return np.zeros((10, 10), dtype=np.int16)

    def fake_registration_wrapper(f1_reg, f_raw=None, f_reg_chan2=None, f_raw_chan2=None, refImg=None, align_by_chan2=None, ops=None):
        # capture ops passed and return a dummy registration_outputs structure
        captured['ops_passed'] = ops
        return {'dummy': True}

    fake_suite2p = types.SimpleNamespace(io=types.SimpleNamespace(BinaryFile=FakeBinaryFile), registration_wrapper=fake_registration_wrapper, registration=types.SimpleNamespace(save_registration_outputs_to_ops=lambda a,b:None, compute_enhanced_mean_image=lambda a,b:np.zeros((10,10)), get_pc_metrics=lambda a,b: b))

    monkeypatch.setattr(preprocess, 'suite2p', fake_suite2p)

    # Now call gks2p_register and ensure the captured ops contain badframes
    preprocess.gks2p_register(ds, basepath=str(tmp_path))

    # After call, check captured ops
    assert 'ops_passed' in captured, "registration wrapper was not called"
    ops_used = captured['ops_passed']
    assert 'badframes' in ops_used, f"badframes not set in ops passed to registration: {list(ops_used.keys())}"
    # Check it matches the saved array
    np.testing.assert_array_equal(np.asarray(ops_used['badframes']), bad_frames)
