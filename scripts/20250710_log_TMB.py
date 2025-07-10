#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:26:20 2025

@author: georgioskeliris
"""

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

ds = datasetQuery(cohort='lrn1', day='d0', experiment='DR')
gks2p_makeOps(ds, basepath, db=db, fastbase=fastbase)
gks2p_toBinary(ds, basepath)
gks2p_register(ds, basepath, iplaneList=None)
gks2p_segment(ds, basepath, iplaneList=None) # potentially can give a list of planes to process
# Feb 6, 2025
gks2p_combine(ds, basepath)


ds = datasetQuery(cohort='lrn1', day='d1', experiment='Familiar')
db = {
    'pipeline': 'orig',
    'tau': 1.5,
    'spatial_scale': 0,
    'reg_tif': True
}
gks2p_makeOps(ds, basepath, db=db, fastbase=fastbase)
gks2p_toBinary(ds, basepath)
gks2p_register(ds, basepath, iplaneList=None)
gks2p_segment(ds, basepath, iplaneList=None) # potentially can give a list of planes to process
gks2p_combine(ds, basepath)

ds = datasetQuery(cohort='lrn1', day='d1', mouseID='M827', experiment='Familiar_trial')
gks2p_makeOps(ds, basepath, db=db, fastbase=fastbase)
gks2p_toBinary(ds, basepath)
gks2p_register(ds, basepath, iplaneList=None)
gks2p_segment(ds, basepath, iplaneList=None) # potentially can give a list of planes to process
gks2p_combine(ds, basepath)

ds = datasetQuery(cohort='lrn1', day='d6', mouseID='M827')
gks2p_makeOps(ds, basepath, db=db, fastbase=fastbase)
gks2p_toBinary(ds, basepath)
gks2p_register(ds, basepath, iplaneList=None)
gks2p_segment(ds, basepath, iplaneList=None) # potentially can give a list of planes to process
gks2p_combine(ds, basepath)

ds = datasetQuery(cohort='lrn1', day='d2', mouseID='M827', experiment='Familiar')
gks2p_makeOps(ds, basepath, db=db, fastbase=fastbase)
gks2p_toBinary(ds, basepath)
gks2p_register(ds, basepath, iplaneList=None)
gks2p_segment(ds, basepath, iplaneList=None) # potentially can give a list of planes to process
gks2p_combine(ds, basepath)

ds = datasetQuery(cohort='lrn1', day='d4', mouseID='M827', experiment='Familiar')
gks2p_makeOps(ds, basepath, db=db, fastbase=fastbase)
gks2p_toBinary(ds, basepath)
gks2p_register(ds, basepath, iplaneList=None)
gks2p_segment(ds, basepath, iplaneList=None) # potentially can give a list of planes to process
gks2p_combine(ds, basepath)




# 27 Mar 2025
ds = datasetQuery(cohort='lrn1', day='d6', mouseID='M1')
db = {
    'pipeline': 'orig',
    'tau': 1.5,
    'spatial_scale': 0,
    'reg_tif': False
}
gks2p_makeOps(ds, basepath, db=db, fastbase=fastbase)
gks2p_toBinary(ds, basepath)
gks2p_register(ds, basepath, iplaneList=None)


# 28 Mar 2025
ds = datasetQuery(cohort='lrn1', day='d6', mouseID='M2')
db = {
    'pipeline': 'orig',
    'tau': 1.5,
    'spatial_scale': 0,
    'reg_tif': False
}
gks2p_makeOps(ds, basepath, db=db, fastbase=fastbase)
gks2p_toBinary(ds, basepath)
gks2p_register(ds, basepath, iplaneList=None)

# 29 Mar 2025
ds = datasetQuery(cohort='lrn1', day='d6', mouseID='M771')
db = {
    'pipeline': 'orig',
    'tau': 1.5,
    'spatial_scale': 0,
    'reg_tif': False
}
gks2p_makeOps(ds, basepath, db=db, fastbase=fastbase)
gks2p_toBinary(ds, basepath)
gks2p_register(ds, basepath, iplaneList=None)

ds = datasetQuery(cohort='lrn1', day='d6', mouseID='M797')
gks2p_makeOps(ds, basepath, db=db, fastbase=fastbase)
gks2p_toBinary(ds, basepath)
gks2p_register(ds, basepath, iplaneList=None)

ds = datasetQuery(cohort='lrn1', day='d6', mouseID='M801')
gks2p_makeOps(ds, basepath, db=db, fastbase=fastbase)
gks2p_toBinary(ds, basepath)
gks2p_register(ds, basepath, iplaneList=None)


# 11 Apr 2025
ds = datasetQuery(cohort='lrn1', day='d0', mouseID='M827', experiment='zstack')
gks2p_makeOps(ds, basepath, db=db, fastbase=fastbase)
gks2p_toBinary(ds, basepath)
gks2p_register(ds, basepath, iplaneList=None)

# 23 Jun 2025
ds = datasetQuery(cohort='lrn1', day='d0', mouseID='M852', experiment='DR')
gks2p_combine(ds, basepath)
