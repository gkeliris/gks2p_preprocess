#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 07:17:51 2025

@author: georgioskeliris
"""
# 9 Jan 2025
from gks2p import *
basepath = '/mnt/NAS_UserStorage/georgioskeliris/MECP2TUN/'
fastbase = '/mnt/NAS_DataStorage/Data_active/MECP2TUN/'
ds = datasetQuery(cohort='coh1', week='w22', experiment='DR')
db = {
    'pipeline': 'orig',
    'tau': 1.5,
    'spatial_scale': 0,
    'reg_tif': True
}
# And make the ops from the .tif header (if not done)
gks2p_makeOps(ds, basepath, db=db, fastbase=fastbase)

gks2p_toBinary(ds, basepath)
gks2p_register(ds, basepath, iplaneList=None)

# 10 Jan 2025
gks2p_segment(ds, basepath, iplaneList=None)

