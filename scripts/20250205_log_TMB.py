#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:26:20 2025

@author: georgioskeliris
"""

from gks2p import *
basepath = '/mnt/NAS_DataStorage/Data_active/LRN2P/'
fastbase = '/mnt/NAS_DataStorage/Data_active/LRN2P/'

ds = datasetQuery(cohort='lrn1', day='d0', experiment='DR')
db = {
    'pipeline': 'orig',
    'tau': 1.5,
    'spatial_scale': 0,
    'reg_tif': True
}


gks2p_makeOps(ds, basepath, db=db, fastbase=fastbase)
