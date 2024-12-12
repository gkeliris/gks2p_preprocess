#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 22:15:12 2024

@author: georgioskeliris
"""

folders = os.scandir('/mnt/12TB_HDD_6/ThinkmateB_HDD6_Data/stavroula-skarvelaki/Project_Thy1_Tg1/Contrast/')
for f in folders:
    dst=f.path
    mouse=f.name[:4]
    if mouse!='M415':
        ds = datasetQuery(cohort='coh3', week='w22', mouseID=mouse)
        dat=ds.iloc[0]
        gks2p_import(dat, dst, basepath, fastbase)

ds = datasetQuery(cohort='coh3', week='w22')
gks2p_updateOpsPerPlane(ds, basepath, fastbase=fastbase, pipeline="orig")




# IMPORT COH4
ds = datasetQuery(cohort='coh4')
dat = ds.iloc[17]
gks2p_import(dat,dat.rawPath + '_processed', basepath, fastbase=fastbase)
gks2p_import(dat,dat.rawPath + '_myprocessed', basepath, fastbase=fastbase)
