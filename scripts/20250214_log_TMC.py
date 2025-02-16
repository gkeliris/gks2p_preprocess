#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:26:20 2025

@author: georgioskeliris
"""
import fissa
from gks2p.datasets import *
from gks2p.preprocess import *
from gks2p.mkops import *

basepath = '/mnt/NAS_DataStorage/Data_active/LRN2P/'
fastbase = '/mnt/NAS_DataStorage/Data_active/LRN2P/'
db = {
    'pipeline': 'orig',
    'tau': 1.5,
    'spatial_scale': 0,
    'reg_tif': True
}


ds = datasetQuery(cohort='lrn1', day='d6', mouseID='M827', experiment='FamNov2')
ops = gks2p_loadOps(ds, basepath)
svf = os.path.join(ops[0]['save_path0'],ops[0]['save_folder'])
opsPP = gks2p_loadOpsPerPlane(svf)

stat=np.load(os.path.join(opsPP[0]['save_path'],'stat.npy'),allow_pickle=True)
iscell=np.load(os.path.join(opsPP[0]['save_path'],'iscell.npy'),allow_pickle=True)[:,0]

# Get image size
Lx = opsPP[0]['Lx']
Ly = opsPP[0]['Ly']

# Get the cell ids
ncells = len(stat)
cell_ids = np.arange(ncells)  # assign each cell an ID, starting from 0.
cell_ids = cell_ids[iscell == 1]  # only take the ROIs that are actually cells.
num_rois = len(cell_ids)

# Generate ROI masks in a format usable by FISSA (in this case, a list of masks)
rois = [np.zeros((Ly, Lx), dtype=bool) for n in range(num_rois)]

for i, n in enumerate(cell_ids):
    # i is the position in cell_ids, and n is the actual cell number
    ypix = stat[n]["ypix"][~stat[n]["overlap"]]
    xpix = stat[n]["xpix"][~stat[n]["overlap"]]
    rois[i][ypix, xpix] = 1

imagesBin = suite2p.io.BinaryFile(Ly=Ly, Lx=Lx, filename=opsPP[0]['reg_file'])
images = os.path.join(opsPP[0]['save_path'],'reg_tif')
output_folder = os.path.join(opsPP[0]['save_path'],'FISSA')
experiment = fissa.Experiment(images, [rois[:5]], output_folder)
experiment.separate()
