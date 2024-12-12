#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 08:33:43 2024

@author: georgioskeliris
"""
import os
import suite2p
import numpy as np

froot='/home/georgioskeliris/s2p_fastdisk/MECP2TUN/s2p_binaries/coh1/M12/w11/ses1_20210913/contrast/suite2p/plane5/'
fname=os.path.join(froot, 'data_raw.bin')
fname_ch2=os.path.join(froot, 'data_chan2_raw.bin')

ops = np.load('/home/georgioskeliris/Desktop/gkel@NAS/MECP2TUN/s2p_analysis/coh1/M12/w11/ses1_20210913/pipeline1/contrast/suite2p/plane5/ops.npy',allow_pickle=True).item()
Ly=ops['Ly']
Lx=ops['Lx']
print(Lx, Ly)

# Read in raw tif corresponding to our example tif
f_raw = suite2p.io.BinaryFile(Ly=Ly, Lx=Lx, filename=fname)
f_raw_chan2 = suite2p.io.BinaryFile(Ly=Ly, Lx=Lx, filename=fname_ch2)
# Create a binary file we will write our registered image to 
f_reg = suite2p.io.BinaryFile(Ly=Ly, Lx=Lx, filename=os.path.join(froot,'data.bin'), n_frames = f_raw.shape[0]) 
f_reg_chan2 = suite2p.io.BinaryFile(Ly=Ly, Lx=Lx, filename=os.path.join(froot,'data_chan2.bin'), n_frames = f_raw_chan2.shape[0]) 
out = suite2p.registration_wrapper(f_reg, f_raw=f_raw, f_reg_chan2=f_reg_chan2, 
                                                   f_raw_chan2=f_raw_chan2, refImg=None, 
                                                   align_by_chan2=False, ops=ops)