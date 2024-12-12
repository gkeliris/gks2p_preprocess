#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 17:49:05 2024

@author: georgioskeliris
"""
from ops.mkops import *
from ops.datasets import *

basepath = '/home/georgioskeliris/Desktop/gkel@NAS/MECP2TUN/'

#ds = getDataPaths('coh1','w11','contrast')
#ds = getOneExpPath('coh1', 'w11', 'M10', 'contrast')
ds = getOneSesPath('coh1', 'w11', 'M12', 'ses1', 'contrast')

#dsetOK = checkDatasets(ds.iloc[0].rawPath)

fileOK = checkTifFile(ds.rawPath, 131)
