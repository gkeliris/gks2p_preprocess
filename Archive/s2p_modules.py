#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 20:19:53 2024

@author: georgioskeliris
"""

import numpy as np
import sys
import os, requests
from pathlib import Path
import suite2p
import json


# 0. Read tiff files and create: ops
#-----------------------------------------------------------------------------
    # a) Use 'myOpsApp.m' that creates and saves ops.json
    # b) Create a python function that creates an ops (and saves ops.npy)

# 00. Read the ops from froot folder (TODO: a function that pick this from DES)
#-----------------------------------------------------------------------------
#froot = '/mnt/Smirnakis_lab_NAS/georgioskeliris/MECP2/M18/Contrast/'
froot ='/mnt/Smirnakis_lab_NAS/georgioskeliris/MECP2TUN/rawdata/coh1/M10/w11/ses1/contrast/'
froot = '/mnt/Toshiba_16TB_1/1.1 cohort1_hsynG7f_11weeks_raw tiff/M10_PVtd_3mm/RAW/Contrast/'
    # a) in case json
f = open(os.path.join(froot, "ops.json"),'r')
ops = json.load(f)
# fill in the fields that were not set with default values
defops = suite2p.default_ops()
ops = {**defops, **ops}


# 1. convert tiffs to binary using 1st step of suite2p pipeline in run_s2p
ops['batch_size']=500
suite2p.run_s2p_toBinary(ops=ops)

# 2.  
suite2p.run_planes(ops=ops)