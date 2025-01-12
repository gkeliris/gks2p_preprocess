#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 16:29:30 2024

 Analysis on ThinkmateC
 
@author: georgioskeliris
"""
from gks2p import *
basepath = '/mnt/NAS_UserStorage/georgioskeliris/MECP2TUN/'
fastbase = '/mnt/raid0/GKSMSLAB/MECP2TUN/'

### 30 Dec 2024 -> cellpose works
ds = datasetQuery(cohort='coh1', week='w11', mouseID='M10', ses='ses1', experiment='DR') 
ds = datasetQuery(cohort='coh1', week='w11', mouseID='M10', ses='ses1', experiment='OR2')

gks2p_segment(ds, basepath, iplaneList=None) # potentially can give a list of planes to process
gks2p_combine(ds, basepath)


ds = datasetQuery(cohort='coh1', week='w11', mouseID='M10', ses='ses1', experiment='OO') 

gks2p_updateOpsPaths(ds,basepath)
gks2p_updateOpsPerPlane(ds, basepath)
gks2p_segment(ds, basepath, iplaneList=None) # potentially can give a list of planes to process
gks2p_combine(ds, basepath)

### 1 Jan 2025
ds = datasetQuery(cohort='coh1', week='w22', mouseID='M11', experiment='SF')
gks2p_segment(ds, basepath, iplaneList=None)
ds = datasetQuery(cohort='coh1', week='w22', mouseID='M12', experiment='SF')
gks2p_segment(ds, basepath, iplaneList=None)
ds = datasetQuery(cohort='coh1', week='w22', mouseID='M18', experiment='SF')
gks2p_segment(ds, basepath, iplaneList=None)


## 9 Jan 2025
from gks2p import *
basepath = '/mnt/NAS_UserStorage/georgioskeliris/MECP2TUN/'
fastbase = '/mnt/NAS_DataStorage/Data_active/MECP2TUN/'
ds = datasetQuery(cohort='coh1', week='w22', experiment='TF')
gks2p_register(ds, basepath, iplaneList=None)

# 10 Jan 2025
gks2p_segment(ds, basepath, iplaneList=None)

#crashed on M20 plane6 - no ROIs detected

# continue for following datasets
ds=ds[ds.datID>127]
gks2p_register(ds, basepath, iplaneList=None) # this was by accident it should have been segment

# 11 Jan 2025
gks2p_segment(ds, basepath, iplaneList=None)

