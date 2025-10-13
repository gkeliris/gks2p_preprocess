# gks2p_preprocess
---
This code uses modified suite2p that can be used in modules
---
## Setting up
### Installation
- create mamba environment
    > mamba create --name [envname] python=3.9 pandas scipy
- activate mamba environment
    > mamba activate [envname]
- install GAK:suite2p package directly from github
    > python -m pip install "suite2p[gui] @ git+https://github.com/gkeliris/suite2p.git@gks2p"
- install scanreader
    > pip3 install git+https://github.com/atlab/scanreader.git    
- install the gks2p_preprocess repository from github (choose one)
    > python -m pip install git+https://github.com/gkeliris/gks2p_preprocess.git@mecp2   (for MeCP2 datasets)

    > python -m pip install git+https://github.com/gkeliris/gks2p_preprocess.git@lrn2p   (for learning datasets)

    > python -m pip install git+https://github.com/gkeliris/gks2p_preprocess.git@tepi   (for temperature epilepsy)

- install FISSA
    > pip install fissa

    
### In case of problems / optional 
- optional in case not working
    > sudo apt-get install libegl1

    > conda install pyqt

    > pip uninstall PyQt6

    > pip install PyQt5
    
- install scipy / pandas / spyder in case not installed with environment
    > conda install pandas

    > conda install scipy

    > conda install spyder
- install VSCODE in case not already installed
    > sudo snap install code --classic

```markdown
# gks2p_preprocess
---
This code uses modified suite2p that can be used in modules
---
## Setting up
### Installation
- create mamba environment
    > mamba create --name [envname] python=3.9 pandas scipy
- activate mamba environment
    > mamba activate [envname]
- install GAK:suite2p package directly from github
    > python -m pip install "suite2p[gui] @ git+https://github.com/gkeliris/suite2p.git@gks2p"
- install scanreader
    > pip3 install git+https://github.com/atlab/scanreader.git    
- install the gks2p_preprocess repository from github (choose one)
    > python -m pip install git+https://github.com/gkeliris/gks2p_preprocess.git@mecp2   (for MeCP2 datasets)

    > python -m pip install git+https://github.com/gkeliris/gks2p_preprocess.git@lrn2p   (for learning datasets)

    > python -m pip install git+https://github.com/gkeliris/gks2p_preprocess.git@tepi   (for temperature epilepsy)

- install FISSA
    > pip install fissa

    
### In case of problems / optional 
- optional in case not working
    > sudo apt-get install libegl1

    > conda install pyqt

    > pip uninstall PyQt6

    > pip install PyQt5
    
- install scipy / pandas / spyder in case not installed with environment
    > conda install pandas

    > conda install scipy

    > conda install spyder
- install VSCODE in case not already installed
    > sudo snap install code --classic

- install CELLPOSE
    > pip install 'cellpose[gui]'

    

```

## CLI: temporal smoothing tool

This package installs a command-line helper for temporal smoothing of Suite2p .bin files.

Install the package (editable during development):

```bash
pip install -e .
```

After installation the script `gks2p-smooth` should be available on your PATH. Example usage:

```bash
# basic help
gks2p-smooth --help

# gaussian smoothing example
gks2p-smooth /path/to/data_raw.bin --Lx 512 --Ly 512 --method gaussian --sigma_frames 1.5 --out /path/to/output.bin
```

If the script is not on your PATH (for example when using a non-activated conda env), you can always run the CLI via Python:

```bash
python -m gks2p.suite2p_temporal_smoothing --help
```

## Python API: dataset-level smoothing helper

If you prefer to call smoothing from Python (for pipelines or scripts) there's a convenience
helper `gks2p_smooth` in `gks2p.preprocess` that mirrors the CLI and will locate `Lx`/`Ly`
from your `ops.npy` files when run against your dataset layout.

Example (block averaging, x=2):

```python
from gks2p.preprocess import gks2p_smooth

# ds: your dataset table (DataFrame-like with .iloc access)
# basepath: project base path used by gks2p_path/op helpers
gks2p_smooth(ds, basepath='/path/to/project', pipeline='orig', method='block', method_kwargs={'x': 2}, inplace=False, update_ops=True)
```

This is equivalent to running the CLI for a single plane's .bin file:

```bash
gks2p-smooth /path/to/plane0/data_raw.bin --Lx 512 --Ly 512 --method block --x 2 --out /path/to/output.bin
```

Notes:
- `gks2p_smooth` will try to find the per-plane `ops.npy` under your `save_path0` (or
    `suite2p_orig`) and the binary under `fast_disk/suite2p/plane*/`.
- Use `update_ops=True` to rewrite the plane `ops.npy` with updated `nframes` when smoothing
    changes frame count (e.g., downsampling).



