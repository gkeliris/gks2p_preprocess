#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 07:18:22 2023

@author: georgioskeliris
"""

from scipy.io import loadmat

matdat = loadmat('/home/georgioskeliris/Desktop/GKHub/gk2Putils/all_exp_description.mat','squeeze_me'='True')

matstruct = matdat['DES']
coh1=matstruct['coh1']
dat = coh1[0][0][0][0][1][0][0][1][0][0][1][0][0][1][0]

print(dat)