#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 09:16:49 2025

@author: georgioskeliris
"""

from gks2p import *

basepath = '/mnt/NAS_UserStorage/georgioskeliris/MECP2TUN/'
fastbase = '/mnt/NAS_DataStorage/Data_active/MECP2TUN/'

ds = datasetQuery(cohort='coh1', week='w22', experiment='TF')
db = {
    'pipeline': 'orig',
    'tau': 1.5,
    'spatial_scale': 0,
    'reg_tif': True
}

gks2p_makeOps(ds, basepath, db=db, fastbase=fastbase)

gks2p_toBinary(ds, basepath)

