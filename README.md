# gks2p
---
This code uses modified suite2p that can be used in modules
---
## Setting up
### Installation
- create conda environment
    > conda create --name gks2p python=3.9 pandas scipy spyder
- activate conda environment
    > conda activate gks2p
- install GAK:suite2p package directly from github
    > python -m pip install "suite2p[gui] @ git+https://github.com/gkeliris/suite2p.git@gks2p"
- install scanreader
    > pip3 install git+https://github.com/atlab/scanreader.git    
- clone the gks2p repository from github    
    > git clone git@github.com:gkeliris/gks2p.git

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


    
