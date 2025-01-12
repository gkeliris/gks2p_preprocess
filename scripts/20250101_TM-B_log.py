#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 09:16:49 2025

@author: georgioskeliris
"""

from gks2p import *

basepath = '/mnt/NAS_UserStorage/georgioskeliris/MECP2TUN/'
fastbase = '/mnt/NAS_DataStorage/Data_active/MECP2TUN/'

ds = datasetQuery(cohort='coh1', week='w22', experiment='SF', mouseID='M19')
gks2p_register(ds, basepath, iplaneList=None)
gks2p_segment(ds, basepath, iplaneList=None)

ds = datasetQuery(cohort='coh1', week='w22', experiment='SF', mouseID='M11')
gks2p_segment(ds, basepath, iplaneList=None)

ds = datasetQuery(cohort='coh1', week='w22', experiment='SF', mouseID='M12')
gks2p_segment(ds, basepath, iplaneList=None)

ds = datasetQuery(cohort='coh1', week='w22', experiment='SF', mouseID='M18')
gks2p_segment(ds, basepath, iplaneList=None)

ds = datasetQuery(cohort='coh1', week='w22', experiment='SF', mouseID='M20')
gks2p_register(ds, basepath, iplaneList=None)
gks2p_segment(ds, basepath, iplaneList=None)

ds = datasetQuery(cohort='coh1', week='w22', experiment='SF', mouseID='M22')
gks2p_register(ds, basepath, iplaneList=None)
gks2p_segment(ds, basepath, iplaneList=None)

ds = datasetQuery(cohort='coh1', week='w22', experiment='SF', mouseID='M91')
gks2p_register(ds, basepath, iplaneList=None)
gks2p_segment(ds, basepath, iplaneList=None)

ds = datasetQuery(cohort='coh1', week='w22', experiment='SF', mouseID='M117')
gks2p_register(ds, basepath, iplaneList=None)
gks2p_segment(ds, basepath, iplaneList=None)
