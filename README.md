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


    
