#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 18:05:20 2023

@author: georgioskeliris
"""

import json
import numpy as np

# Opening JSON file
f = open('/mnt/Smirnakis_lab_NAS/georgioskeliris/MECP2/M19/contrast/ops.json')
 
# returns JSON object as 
# a dictionary
ops = json.load(f)

nplanes = ops['nplanes']
ops['iplane'] = np.zeros((nplanes * ops['nrois'],), np.int32)
for n in range(ops['nrois']):
    ops['iplane'][n::ops['nrois']] = np.arange(0, nplanes, 1, int)
    
dx = np.array([o['dx'] for o in ops])
