#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 17:49:05 2024

@author: georgioskeliris
"""
import numpy as np
import sys
import os, requests
from pathlib import Path
import matplotlib.pyplot as plt

froot = '/mnt/Smirnakis_lab_NAS/georgioskeliris/MECP2/M18/Contrast/'
fname = 'contrast_M18_00001_00001.tif'
from tifffile import imread
import suite2p
import json

data = imread(fname)
print('imaging data of shape: ', data.shape)
n_time, Ly, Lx = data.shape

data = imread(os.path.join(froot, fname))
f = open(os.path.join(froot, "ops.json"),'r')
ops = json.load(f)
defops = suite2p.default_ops()
ops = {**defops, **ops}

refImg, rmin, rmax, meanImg, rigid_offsets, \
nonrigid_offsets, zest, meanImg_chan2, badframes, \
yrange, xrange = suite2p.registration_wrapper(f1_reg, f_raw=f1, f_reg_chan2=None, 
                                                   f_raw_chan2=None, refImg=None, 
                                                   align_by_chan2=False, ops=ops)

fsall, ops1 = suite2p.io.utils.get_tif_list(ops)


f_raw = suite2p.io.BinaryFile(Ly=Ly, Lx=Lx, filename=fname)
f_reg = suite2p.io.BinaryFile(Ly=Ly, Lx=Lx, filename='reg_data.bin', n_frames = f_raw.shape[0])

def clear_all():
    """Clears all the variables from the workspace of the spyder application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]


# 1. REGISTRATION
pl='0'
froot = '/mnt/12TB_HDD_4/ThinkmateB_HDD4_Data/GKeliris/s2p_fastdisk/ses1/suite2p/plane' + pl
opsstr= os.path.join(froot, 'ops.npy')
ops=np.load(opsstr,allow_pickle=True).item()
Ly=ops['Ly']
Lx=ops['Lx']

f1 = suite2p.io.BinaryFile(Ly=Ly, Lx=Lx, filename=os.path.join(froot, 'data_raw.bin'))
f1_reg = suite2p.io.BinaryFile(Ly=Ly, Lx=Lx, filename=os.path.join(froot, 'data.bin'), n_frames = f1.shape[0])

refImg, rmin, rmax, meanImg, rigid_offsets, \
nonrigid_offsets, zest, meanImg_chan2, badframes, \
yrange, xrange = suite2p.registration_wrapper(f1_reg, f_raw=f1, f_reg_chan2=None, 
                                                   f_raw_chan2=None, refImg=None, 
                                                   align_by_chan2=False, ops=ops)

clear_all()
del ops

# 2. ROI DETECTION

pl='1'
opsstr='/mnt/Smirnakis_lab_NAS/georgioskeliris/MECP2/M11/Contrast_processed/suite2p/plane' + pl + '/ops.npy'
ops=np.load(opsstr,allow_pickle=True).item()
Ly=ops['Ly']
Lx=ops['Lx']


# Use default classification file provided by suite2p 
classfile = suite2p.classification.builtin_classfile
np.load(classfile, allow_pickle=True)[()]

f_reg = suite2p.io.BinaryFile(Ly=Ly, Lx=Lx, filename='data.bin')
ops, stat = suite2p.detection_wrapper(f_reg=f_reg, ops=ops, classfile=classfile)


# 3. Fluorescence Extraction
stat_after_extraction, F, Fneu, F_chan2, Fneu_chan2 = \
    suite2p.extraction_wrapper(stat, f_reg, f_reg_chan2 = None,ops=ops)
    
# 4. Cell Classification
iscell = suite2p.classify(stat=stat_after_extraction, classfile=classfile)

# 5. Spike Deconvolution
# Correct our fluorescence traces 
dF = F.copy() - ops['neucoeff']*Fneu
# 6.  Apply preprocessing step for deconvolution
dF = suite2p.extraction.preprocess(
        F=dF,
        baseline=ops['baseline'],
        win_baseline=ops['win_baseline'],
        sig_baseline=ops['sig_baseline'],
        fs=ops['fs'],
        prctile_baseline=ops['prctile_baseline']
    )
# Identify spikes
spks = suite2p.extraction.oasis(F=dF, batch_size=ops['batch_size'], tau=ops['tau'], fs=ops['fs'])



# Visualizations
output_ops=ops
# 1. REGISTRATION
plt.subplot(1, 4, 1)

plt.imshow(output_ops['refImg'], cmap='gray', )
plt.title("Reference Image for Registration");

# maximum of recording over time
plt.subplot(1, 4, 2)
plt.imshow(output_ops['max_proj'], cmap='gray')
plt.title("Registered Image, Max Projection");

plt.subplot(1, 4, 3)
plt.imshow(output_ops['meanImg'], cmap='gray')
plt.title("Mean registered image")

plt.subplot(1, 4, 4)
plt.imshow(output_ops['meanImgE'], cmap='gray')
plt.title("High-pass filtered Mean registered image");


plt.figure(figsize=(18,8))

plt.subplot(4,1,1)
plt.plot(output_ops['yoff'][:1000])
plt.ylabel('rigid y-offsets')

plt.subplot(4,1,2)
plt.plot(output_ops['xoff'][:1000])
plt.ylabel('rigid x-offsets')

plt.subplot(4,1,3)
plt.plot(output_ops['yoff1'][:1000])
plt.ylabel('nonrigid y-offsets')

plt.subplot(4,1,4)
plt.plot(output_ops['xoff1'][:1000])
plt.ylabel('nonrigid x-offsets')
plt.xlabel('frames')

plt.show()

#@title Run cell to look at registered frames
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from suite2p.io import BinaryFile

widget = widgets.IntSlider(
    value=7,
    min=0,
    max=10,
    step=1,
    description='Test:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)


def plot_frame(t):
    with BinaryFile(Ly=output_ops['Ly'],
                Lx=output_ops['Lx'],
                filename=output_ops['reg_file']) as f:
        plt.imshow(f[t])

interact(plot_frame, t=(0, output_ops['nframes']- 1, 1)); # zero-indexed so have to subtract 1

