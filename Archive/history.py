#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 07:24:05 2024

@author: georgioskeliris
"""

    dT = np.diff(T[0])
    blocksSep = np.where(dT > 50000)
    blocksSep = np.append(blocksSep[0], len(dT))
    nBlocks = len(blocksSep)
    print('nBlocks = ', nBlocks)
    blocksSep = np.append(0, blocksSep)
    
    allRevs = np.where(dT > 5000)
    allRevs = np.append(0, allRevs)
    nRevsPerBlock = np.zeros([nBlocks], dtype='int64')
    for b in range(0, nBlocks):
        nRevsPerBlock[b] = len(np.squeeze(
            np.where((allRevs < blocksSep[b+1]) & (allRevs >= blocksSep[b]))))
    print('nRevsPerBlock = ', nRevsPerBlock)
    
    cumRevs = np.cumsum(nRevsPerBlock)
    revTrials = np.zeros([len(allRevs), 10000])
    revNum = []
    revInd = []
    for b in range(0, nBlocks):
        revNum.append(list(range(cumRevs[b]-nRevsPerBlock[b], cumRevs[b])))
        revInd = revInd + list(T[0][allRevs[revNum[b]]])
        for r in revNum[b]:
            try:
                revTrials[r] = data[revInd[r]:revInd[r]+10000, dataCh] - \
                    np.mean(data[revInd[r]:revInd[r]+1000, dataCh])
            except:
                print("Last trial was not complete")
                nRevsPerBlock[b] = nRevsPerBlock[b]-1
                print(nRevsPerBlock)
    return revTrials, revNum, nRevsPerBlock
trialsM6D1, revNumM6D1, nRevsM6D1 = getReversals(dataM4D1, 4, 0)
trialsM6D1, revNumM6D1, nRevsM6D1 = getReversals(dataM6D1, 4, 0)
trialsM6D1, revNumM6D1, nRevsM6D1 = getReversals(dataM6D1, 5, 0)
pltVEPs(trialsM6D1, revNumM6D1, 500)
M7D1 = '/mnt/Smirnakis_lab_NAS/georgioskeliris/Ephys/M7/20231213_day1/2023-12-13_10-42-05/'
dataM7D1, tsM7D1 = readVEPdata(M7D1)
trialsM7D1, revNumM7D1, nRevsM7D1 = getReversals(dataM7D1, 5, 0)
pltVEPs(trialsM7D1, revNumM7D1, 500)

## ---(Thu Dec 14 16:04:50 2023)---

import numpy as np
from matplotlib import pyplot as plt
from open_ephys.analysis import Session
#import statistics


def readVEPdata(directory):
    session = Session(directory)
    recordnode = session.recordnodes[0]
    recording = recordnode.recordings[0]
    data = recording.continuous[0].get_samples(0, None)
    timestamps = recording.continuous[0].timestamps
    return data, timestamps


def getReversals(data, triggerCh, dataCh):
    T = np.where(data[:, triggerCh] < 2)
    dT = np.diff(T[0])
    blocksSep = np.where(dT > 50000)
    blocksSep = np.append(blocksSep[0], len(dT))
    nBlocks = len(blocksSep)
    print('nBlocks = ', nBlocks)
    blocksSep = np.append(0, blocksSep)
    
    allRevs = np.where(dT > 5000)
    allRevs = np.append(0, allRevs)
    nRevsPerBlock = np.zeros([nBlocks], dtype='int64')
    for b in range(0, nBlocks):
        nRevsPerBlock[b] = len(np.squeeze(
            np.where((allRevs < blocksSep[b+1]) & (allRevs >= blocksSep[b]))))
    print('nRevsPerBlock = ', nRevsPerBlock)
    
    cumRevs = np.cumsum(nRevsPerBlock)
    revTrials = np.zeros([len(allRevs), 10000])
    revNum = []
    revInd = []
    for b in range(0, nBlocks):
        revNum.append(list(range(cumRevs[b]-nRevsPerBlock[b], cumRevs[b])))
        revInd = revInd + list(T[0][allRevs[revNum[b]]])
        for r in revNum[b]:
            try:
                revTrials[r] = data[revInd[r]:revInd[r]+10000, dataCh] - \
                    np.mean(data[revInd[r]:revInd[r]+1000, dataCh])
            except:
                print("Last trial was not complete")
                nRevsPerBlock[b] = nRevsPerBlock[b]-1
                print(nRevsPerBlock)
    return revTrials, revNum, nRevsPerBlock


def pltVEPsPerBlock(revTrials, revNum, rejectThr):
    
    maxAmp = np.max(np.abs(revTrials), 1)
    
    fig, ax = plt.subplots(1, 1)
    for b in range(0, len(revNum)):
        keep = np.where(maxAmp[revNum[b]] < rejectThr)
        print('BLK=', b, 'N=', len(keep[0]))
        indices = np.array(revNum[b])
        ax.errorbar(np.linspace(0, 10000/30, 10000)+(b)*10000/30,
                    np.mean(revTrials[indices[keep[0]], :], 0),
                    np.std(revTrials[indices[keep[0]], :], 0)/np.sqrt(len(keep[0])))


def pltVEPs(revTrials, revNum, rejectThr):
    maxAmp = np.max(np.abs(revTrials), 1)
    keep = np.where(maxAmp < rejectThr)
    plt.errorbar(np.linspace(0, 10000/30, 10000),
                 np.mean(revTrials[keep[0], :], 0),
                 np.std(revTrials[keep[0], :], 0)/np.sqrt(len(keep[0])))


def pltVEPsFAMvsNOV(revTrials, revNum, rejectThr):
    
    familialBlocks = range(0, len(revNum), 2)
    famIndex = []
    for b in familialBlocks:
        famIndex = famIndex + revNum[b]
    
    novelBlocks = range(1, len(revNum), 2)
    novIndex = []
    for b in novelBlocks:
        novIndex = novIndex + revNum[b]
    
    fam = np.array(famIndex)
    plt.errorbar(np.linspace(0, 10000/30, 10000), np.mean(revTrials[fam, :], 0),
                 np.std(revTrials[fam, :], 0)/np.sqrt(len(fam)))
    
    nov = np.array(novIndex)
    plt.errorbar(np.linspace(0, 10000/30, 10000), np.mean(revTrials[nov, :], 0),
                 np.std(revTrials[nov, :], 0)/np.sqrt(len(nov)))
M7D2='/home/georgioskeliris/Desktop/gkel@NAS/Ephys/M7/20231214_day2/2023-12-14_15-17-05/'
dataM7D2, tsM7D2 = readVEPdata(M7D2)
trialsM7D2, revNumM7D2, nRevsM7D2 = getReversals(dataM7D2, 5, 0)
pltVEPs(trialsM7D2, revNumM7D2, 500)
M2D2 = '/home/georgioskeliris/Desktop/gkel@NAS/Ephys/M2_25blocks/2023-12-04_16-19-18/'
dataM2D2, tsM2D2 = readVEPdata(M2D2)
trialsM2D2, revNumM2D2, nRevsM2D2 = getReversals(dataM2D2, 5, 0)
pltVEPs(trialsM2D2, revNumM2D2, 500)
trialsM2D2, revNumM2D2, nRevsM2D2 = getReversals(dataM2D2, 4, 0)
trialsM2D2, revNumM2D2, nRevsM2D2 = getReversals(dataM2D2, 3, 0)
M2D3 = '/home/georgioskeliris/Desktop/gkel@NAS/Ephys/M7/20231215_day3/2023-12-15_14-24-25/'
dataM2D3, tsM2D3 = readVEPdata(M2D3)
trialsM2D3, revNumM2D3, nRevsM2D3 = getReversals(dataM2D3, 3, 0)
pltVEPs(trialsM2D3, revNumM2D3, 500)
M7D3 = '/home/georgioskeliris/Desktop/gkel@NAS/Ephys/M7/20231215_day3/2023-12-15_14-24-25/'
dataM7D3, tsM7D3 = readVEPdata(M7D3)
trialsM7D3, revNumM7D3, nRevsM7D3 = getReversals(dataM7D3, 5, 0)
pltVEPs(trialsM7D3, revNumM7D3, 500)
dataM7D2, tsM7D2 = readVEPdata(M7D2)
trialsM7D2, revNumM7D2, nRevsM7D2 = getReversals(dataM7D2, 5, 0)
pltVEPs(trialsM7D2, revNumM7D2, 300)

dataM7D3, tsM7D3 = readVEPdata(M7D3)
trialsM7D3, revNumM7D3, nRevsM7D3 = getReversals(dataM7D3, 5, 0)
pltVEPs(trialsM7D3, revNumM7D3, 300)
M2D3 = '/home/georgioskeliris/Desktop/gkel@NAS/Ephys/M2_25blocks/2023-12-14_16-30-35/'
dataM2D3, tsM2D3 = readVEPdata(M2D3)
trialsM2D3, revNumM2D3, nRevsM2D3 = getReversals(dataM2D3, 5, 0)
pltVEPs(trialsM2D3, revNumM2D3, 500)

## ---(Tue Dec 19 04:12:21 2023)---
M7D4 = '/home/georgioskeliris/Desktop/gkel@NAS/Ephys/M7/20231216_day4/2023-12-16_14-50-04/'
M7D5 = '/home/georgioskeliris/Desktop/gkel@NAS/Ephys/M7/20231217_day5/2023-12-17_13-39-33/'
M7D6 = '/home/georgioskeliris/Desktop/gkel@NAS/Ephys/M7/20231218_day6/2023-12-18_18-49-56/'
dataM7D4, tsM7D4 = readVEPdata(M7D4)
trialsM7D4, revNumM7D4, nRevsM7D4 = getReversals(dataM7D4, 5, 0)
pltVEPs(trialsM7D4, revNumM7D4, 300)
import numpy as np
from matplotlib import pyplot as plt
from open_ephys.analysis import Session
#import statistics


def readVEPdata(directory):
    session = Session(directory)
    recordnode = session.recordnodes[0]
    recording = recordnode.recordings[0]
    data = recording.continuous[0].get_samples(0, None)
    timestamps = recording.continuous[0].timestamps
    return data, timestamps


def getReversals(data, triggerCh, dataCh):
    T = np.where(data[:, triggerCh] < 2)
    dT = np.diff(T[0])
    blocksSep = np.where(dT > 50000)
    blocksSep = np.append(blocksSep[0], len(dT))
    nBlocks = len(blocksSep)
    print('nBlocks = ', nBlocks)
    blocksSep = np.append(0, blocksSep)
    
    allRevs = np.where(dT > 5000)
    allRevs = np.append(0, allRevs)
    nRevsPerBlock = np.zeros([nBlocks], dtype='int64')
    for b in range(0, nBlocks):
        nRevsPerBlock[b] = len(np.squeeze(
            np.where((allRevs < blocksSep[b+1]) & (allRevs >= blocksSep[b]))))
    print('nRevsPerBlock = ', nRevsPerBlock)
    
    cumRevs = np.cumsum(nRevsPerBlock)
    revTrials = np.zeros([len(allRevs), 10000])
    revNum = []
    revInd = []
    for b in range(0, nBlocks):
        revNum.append(list(range(cumRevs[b]-nRevsPerBlock[b], cumRevs[b])))
        revInd = revInd + list(T[0][allRevs[revNum[b]]])
        for r in revNum[b]:
            try:
                revTrials[r] = data[revInd[r]:revInd[r]+10000, dataCh] - \
                    np.mean(data[revInd[r]:revInd[r]+1000, dataCh])
            except:
                print("Last trial was not complete")
                nRevsPerBlock[b] = nRevsPerBlock[b]-1
                print(nRevsPerBlock)
    return revTrials, revNum, nRevsPerBlock


def pltVEPsPerBlock(revTrials, revNum, rejectThr):
    
    maxAmp = np.max(np.abs(revTrials), 1)
    
    fig, ax = plt.subplots(1, 1)
    for b in range(0, len(revNum)):
        keep = np.where(maxAmp[revNum[b]] < rejectThr)
        print('BLK=', b, 'N=', len(keep[0]))
        indices = np.array(revNum[b])
        ax.errorbar(np.linspace(0, 10000/30, 10000)+(b)*10000/30,
                    np.mean(revTrials[indices[keep[0]], :], 0),
                    np.std(revTrials[indices[keep[0]], :], 0)/np.sqrt(len(keep[0])))


def pltVEPs(revTrials, revNum, rejectThr):
    maxAmp = np.max(np.abs(revTrials), 1)
    keep = np.where(maxAmp < rejectThr)
    plt.errorbar(np.linspace(0, 10000/30, 10000),
                 np.mean(revTrials[keep[0], :], 0),
                 np.std(revTrials[keep[0], :], 0)/np.sqrt(len(keep[0])))


def pltVEPsFAMvsNOV(revTrials, revNum, rejectThr):
    
    familialBlocks = range(0, len(revNum), 2)
    famIndex = []
    for b in familialBlocks:
        famIndex = famIndex + revNum[b]
    
    novelBlocks = range(1, len(revNum), 2)
    novIndex = []
    for b in novelBlocks:
        novIndex = novIndex + revNum[b]
    
    fam = np.array(famIndex)
    plt.errorbar(np.linspace(0, 10000/30, 10000), np.mean(revTrials[fam, :], 0),
                 np.std(revTrials[fam, :], 0)/np.sqrt(len(fam)))
    
    nov = np.array(novIndex)
    plt.errorbar(np.linspace(0, 10000/30, 10000), np.mean(revTrials[nov, :], 0),
                 np.std(revTrials[nov, :], 0)/np.sqrt(len(nov)))
M7D1 = '/mnt/Smirnakis_lab_NAS/georgioskeliris/Ephys/M7/20231213_day1/2023-12-13_10-42-05/'
M7D2='/home/georgioskeliris/Desktop/gkel@NAS/Ephys/M7/20231214_day2/2023-12-14_15-17-05/'
M7D3 = '/home/georgioskeliris/Desktop/gkel@NAS/Ephys/M7/20231215_day3/2023-12-15_14-24-25/'
M7D4 = '/home/georgioskeliris/Desktop/gkel@NAS/Ephys/M7/20231216_day4/2023-12-16_14-50-04/'
M7D5 = '/home/georgioskeliris/Desktop/gkel@NAS/Ephys/M7/20231217_day5/2023-12-17_13-39-33/'
M7D6 = '/home/georgioskeliris/Desktop/gkel@NAS/Ephys/M7/20231218_day6/2023-12-18_18-49-56/'

dataM7D1, tsM7D1 = readVEPdata(M7D1)
trialsM7D1, revNumM7D1, nRevsM7D1 = getReversals(dataM7D1, 5, 0)
pltVEPs(trialsM7D1, revNumM7D1, 300)

dataM7D2, tsM7D2 = readVEPdata(M7D2)
trialsM7D2, revNumM7D2, nRevsM7D2 = getReversals(dataM7D2, 5, 0)
pltVEPs(trialsM7D2, revNumM7D2, 300)

dataM7D3, tsM7D3 = readVEPdata(M7D3)
trialsM7D3, revNumM7D3, nRevsM7D3 = getReversals(dataM7D3, 5, 0)
pltVEPs(trialsM7D3, revNumM7D3, 300)

dataM7D4, tsM7D4 = readVEPdata(M7D4)
trialsM7D4, revNumM7D4, nRevsM7D4 = getReversals(dataM7D4, 5, 0)
pltVEPs(trialsM7D4, revNumM7D4, 300)

dataM7D5, tsM7D5 = readVEPdata(M7D5)
trialsM7D5, revNumM7D5, nRevsM7D5 = getReversals(dataM7D5, 5, 0)
pltVEPs(trialsM7D3, revNumM7D3, 300)
dataM7D6, tsM7D6 = readVEPdata(M7D6)
trialsM7D6, revNumM7D6, nRevsM7D6 = getReversals(dataM7D6, 5, 0)
pltVEPsFAMvsNOV(trialsM7D6, revNumM7D6, 300)
pltVEPsPerBlock(trialsM7D6, revNumM7D6, 300)

## ---(Mon Dec 25 19:42:18 2023)---
import pandas as pd
df = pd.read_csv('/mnt/Toshiba_16TB_1/MECP2_datasets.csv')
df.rawPath[o]
df.rawPath[0]
cd(df.rawPath[0])
cd df.rawPath[0]
a=df.rawPath[0]
cd (a)
print a
print(a)
cd(print(a))
cd ~
cd /home
pwd
cd ~
import numpy as np
p=df.rawPath[np.where(df.datID==1)]
np.where(df.datID == 1)
np.where(df.datID == 10)
df.rawPath[np.where(df.datID == 10)]
a = np.where(df.datID == 10)
a
a[0]
a[0][0]
a = np.where(df.expID == 'contrast')
a
a[0]
a[0][0]
a[0][1]
a[0][3]
a = np.where(df.expID == 'contrast') & np.where(df.mouse == 'M11')
df
df.mouseID
a = np.where(df.expID == 'contrast') & np.where(df.mouseID == 'M11')
a = np.where(df.expID == 'contrast' & df.mouseID == 'M11')
a = np.where((df.expID == 'contrast') & (df.mouseID == 'M11'))
a
a[0]
df.rawPath[a[0]]
a = np.where((df.expID == 'contrast') & (df.week == 'w11) (df.mouseID == 'M11'))
a = np.where((df.expID == 'contrast') & (df.week == 'w11) & (df.mouseID == 'M11'))
a = np.where((df.expID == 'contrast') & (df.week == 'w11') & (df.mouseID == 'M11'))
a
b = np.where((df.cohort == 'coh1') & (df.week == 'w11'))
b
b = np.where((df.cohort == 'coh1') & (df.week == 'w11') & (df.expID == 'contrast'))
b
print(df.mouseID[b[0]])
df[0]
b = np.where((df.cohort == 'coh1') & (df.week == 'w22') & (df.expID == 'contrast'))
print(df.mouseID[b[0]])
pip install /home/georgioskeliris/GKHub/MyPYCODE/gkLRNpkg/dist/gkLRNlib-0.1.0-py3-none-any.whl
import gkLRNlib as vep
M4D1 = '/home/georgioskeliris/Desktop/gkel@NAS/Ephys/M4_Day1/M4_blocks/2023-11-21_15-52-53/'
dataM4D1, tsM4D1 = vep.readVEPdata(M4D1)
dataM4D1, tsM4D1 = vep.gkLRNfunctions.readVEPdata(M4D1)
pip install /home/georgioskeliris/GKHub/MyPYCODE/gkLRNpkg/dist/gkLRNlib-0.1.0-py3-none-any.whl
pip install /home/georgioskeliris/GKHub/MyPYCODE/gkLRNpkg/dist/gkLRNlib-0.1.0-py3-none-any.whl --force-reinstall
exit
import gkLRNlib as vep
vep
M4D1 = '/home/georgioskeliris/Desktop/gkel@NAS/Ephys/M4_Day1/M4_blocks/2023-11-21_15-52-53/'
dataM4D1, tsM4D1 = vep.readVEPdata(M4D1)
exit
pip install /home/georgioskeliris/GKHub/MyPYCODE/gkLRNpkg/dist/gkLRNlib-0.1.0-py3-none-any.whl --force-reinstall
import gkLRNlib as vep
import gkLRNfunctions as vep
import gkLRNlib as vep
exit
import gkLRNlib
pip install /home/georgioskeliris/GKHub/MyPYCODE/gkLRNpkg/dist/gkLRNlib-0.1.0-py3-none-any.whl --force-reinstall
exit
import gkLRNlib as vp
gkLRNlib
vp
pip install /home/georgioskeliris/GKHub/MyPYCODE/packaging_tutorial/dist/example_package_GAK-0.0.1-py3-none-any.whl
exit
import example_package_GAK
add_one(5)
example_package_GAK.add_one(5)
example_package_GAK
import example_package_GAK.add_one as ao
import example_package_GAK.example as e
e.add_one(5)
example.add_one(4)
import example_package_GAK
example.add_one(5)
import example_package_GAK.example as ep
ep.add_one(9)
import gkLRNlib.gkLRNfunctions as vep
M4D1 = '/home/georgioskeliris/Desktop/gkel@NAS/Ephys/M4_Day1/M4_blocks/2023-11-21_15-52-53/'
vep.readVEPdata(M4D1)
vep.
dataM4D1, tsM4D1 = readVEPdata(M4D1)
dataM4D1, tsM4D1 = vep.readVEPdata(M4D1)
trialsM4D1, revNumM4D1, nRevsM4D1 = vep.getReversals(dataM4D1, 4, 1)
vep.pltVEPs(trialsM4D1, revNumM4D1, 200)
vep.pltVEPs(trialsM4D1, revNumM4D1, 500)
import csv
csvpath = '/home/georgioskeliris/Desktop/gkel@NAS/Ephys/M3/M3_tuning/'
file=open( csvpath +"tuning0.csv", "r")
reader = csv.reader(file)
for line in reader:
    t=line[1],line[2]
    print(t)
csvpath = '/home/georgioskeliris/Desktop/gkel@NAS/Ephys/M3_Day3/M3_tuning/'
file=open( csvpath +"tuning0.csv", "r")
reader = csv.reader(file)
for line in reader:
    t=line[1],line[2]
    print(t)
csvpath = '/home/georgioskeliris/Desktop/gkel@NAS/Ephys/M3_Day3/M3_tuning/'
file=open( csvpath +"M3_tuning0.csv", "r")
reader = csv.reader(file)
for line in reader:
    t=line[1],line[2]
    print(t)
import pandas as pd
df = pd.read_csv(file)
print(file)
df = pd.read_csv(csvpath +"M3_tuning0.csv")
print(df)
from ScanImageTiffReader import ScanImageTiffReader

reader = ScanImageTiffReader('/home/georgioskeliris/Desktop/gkel@NAS/MECP2/M19/Contrast/Contrast_M19_00001_00001.tif')
reader
reader.description
print(reader.description)
reader
reader.data
des = reader.description(0)
print(Des)
print(des)
reader
reader.metadata
reader.metadata(0)
reader.metadata()
h=reader.metadata()
print(h)
import json
json_obj=json.loads(h)
hl=h.splitlines()
hl
hl[0]
hl[1]
hl[2]
hl[3]
hl[4]
hl[5]
hl[6]
hl[-1]
h.find('{')
json_obj=json.loads(h[804:])
print(h[804:])
h[0]
h[804]
h[804::]
h[804:805]
h[804:809]
h[804:-1]
h[804:1000]
h
print(h)
h.find('{\n')
print(h[9915:-1])
json_obj=json.loads(h[9915:-1])
json_obj
json_obj('RoiGroups')
print(json_obj['RoiGroups'])
print(json_obj['RoiGroups'][imagingRoiGroup])
print(json_obj['RoiGroups']['imagingRoiGroup'])
print(json_obj['RoiGroups']['imagingRoiGroup']['centerXY'])
print(json_obj['RoiGroups']['imagingRoiGroup']['rois'])
print(h[9915:-1])
print(json_obj['RoiGroups']['imagingRoiGroup']['rois']['zs'])
print(json_obj['RoiGroups']['imagingRoiGroup']['ver'])
print(json_obj['RoiGroups']['imagingRoiGroup']['rois'])
print(json_obj['RoiGroups']['imagingRoiGroup']['rois']['ver'])
print(json_obj['RoiGroups']['imagingRoiGroup']['rois'][0])
print(json_obj['RoiGroups']['imagingRoiGroup']['rois'][1])
print(json_obj['RoiGroups']['imagingRoiGroup']['rois'][0]['zs'])
print(json_obj['RoiGroups']['imagingRoiGroup']['rois'].shape)
h.find(''hRoiManager.scanVolumeRate')
h.find('hRoiManager.scanVolumeRate')
js=h.find('{\n')
hh=h[:js-1]
hh
hh=h[:js-1].splitlines()
hh
hh[1]
hh[2]
hh=h[1:js-2].splitlines()
hh[0]
hh[-1]
hh[32]
hh.shape
len(hh)
import matlab.engine
hh[-1]
js=h.find('{\n')
js
json_obj=json.loads(h[js:-1])
len(h)
len(hh)
for s in hh:
    print(s)
s
s.find('=')
s(:23-1)
s[:23-1]
s[23+1:]
s[23+2:]
mdict={{}
}
mdict={}
mdict[s[:23-1]=s[23+2:]
]
mdict[s[:23-1]]=s[23+2:]
print(mdict)
hhDict={}
for s in hh:
    eq_ind = s.find('=')
    hhDict[s[:eq_ind-1]]=s[eq_ind+2:]
hhDict
hhDict[Si.loopAcqInterval]
hhDict['SI.loopAcqInterval']
fsV = hhDict['hRoiManager.scanVolumeRate']
fsV = hhDict['SI.hRoiManager.scanVolumeRate']
fsV
int(fsV)
fsV = float(hhDict['SI.hRoiManager.scanVolumeRate'])
fsF = float(hhDict['SI.hRoiManager.scanFrameRate'])
zs = hhDict['SI.hStackManager.zs']
zs
zs[1:0].split(" ")
zs[1:-2].split(" ")
zs[1:-1].split(" ")
zsstr = hhDict['SI.hStackManager.zs']
zs = zsstr[1:-1].split(" ")
zs
len(zs)
json_obj=json.loads(h[js:-1])
json_obj['RoiGroups']
len(json_obj['RoiGroups'])
len(json_obj['RoiGroups']['rois'])
json_obj['RoiGroups']['rois']
len(json_obj['RoiGroups']['imagingRoiGroup']['rois'])
si_rois = json_obj['RoiGroups']['imagingRoiGroup']['rois']
si_rois[1]
si_rois[0]
nrois = len(si_rois)
Ly = si_rois[0]['scannfields'][0]['pixelResolutionXY'][1]
Ly = si_rois[0]['scanfields'][0]['pixelResolutionXY'][1]
Ly
cXY = si_rois[0]['scanfields'][0]['centerXY']
szXY = si_rois[0]['scanfields'][0]['sizeXY']
runcell(0, '/home/georgioskeliris/GKHub/MyPYCODE/gk_ops.py')
for p in range(0,nplanes):
    for k in range(0,nrois):
        Ly[k] = si_rois[k]['scanfields'][p]['pixelResolutionXY'][1]
        Lx[k] = si_rois[k]['scanfields'][p]['pixelResolutionXY'][0]
        cXY[k] = si_rois[k]['scanfields'][p]['centerXY']
        szXY[k] = si_rois[k]['scanfields'][p]['sizeXY']
nplanes = len(zs)
nrois = len(si_rois)
for p in range(0,nplanes):
    for k in range(0,nrois):
        Ly[k] = si_rois[k]['scanfields'][p]['pixelResolutionXY'][1]
        Lx[k] = si_rois[k]['scanfields'][p]['pixelResolutionXY'][0]
        cXY[k] = si_rois[k]['scanfields'][p]['centerXY']
        szXY[k] = si_rois[k]['scanfields'][p]['sizeXY']
clear Ly
Ly
Lx
Ly[1]=200
for p in range(0,nplanes):
    for k in range(0,nrois):
        Ly[k] = si_rois[k]['scanfields'][p]['pixelResolutionXY'][1]
        Lx[k] = si_rois[k]['scanfields'][p]['pixelResolutionXY'][0]
        cXY[k] = si_rois[k]['scanfields'][p]['centerXY']
        szXY[k] = si_rois[k]['scanfields'][p]['sizeXY']
list(Ly)
Ly=[]
Ly
Ly[0]=1
Ly=list()
Ly[0]=1
Ly
for p in range(0,nplanes):
    Ly=[]
    Lx=[]
    for k in range(0,nrois):
        Ly.insert(k, si_rois[k]['scanfields'][p]['pixelResolutionXY'][1])
        Lx[k] = si_rois[k]['scanfields'][p]['pixelResolutionXY'][0]
        cXY[k] = si_rois[k]['scanfields'][p]['centerXY']
        szXY[k] = si_rois[k]['scanfields'][p]['sizeXY']
Ly
for p in range(0,nplanes):
    Ly=[], Lx=[], cXY=[], szXY=[]   
    for k in range(0,nrois):
        Ly.insert(k, si_rois[k]['scanfields'][p]['pixelResolutionXY'][1])
        Lx.insert(k, si_rois[k]['scanfields'][p]['pixelResolutionXY'][0])
        cXY.insert(k, si_rois[k]['scanfields'][p]['centerXY'])
        szXY.insert(k, si_rois[k]['scanfields'][p]['sizeXY'])
for p in range(0,nplanes):
    Ly=[]
    Lx=[]
    cXY=[]
    szXY=[]   
    for k in range(0,nrois):
        Ly.insert(k, si_rois[k]['scanfields'][p]['pixelResolutionXY'][1])
        Lx.insert(k, si_rois[k]['scanfields'][p]['pixelResolutionXY'][0])
        cXY.insert(k, si_rois[k]['scanfields'][p]['centerXY'])
        szXY.insert(k, si_rois[k]['scanfields'][p]['sizeXY'])
cXY
cXY[0]
szXY[0]/[Lx[0], Ly[0]]
Ly=np()
import numpy as np
Ly=np()
Ly=np.array()
Ly
Ly=np.array(Ly)
Ly
Lx
szXY
szXY=np.array(szXY)
szXY.shape
Lx=np.array(Lx)
szXY(1,:)
szXY[0,:]
szXY[0,:]/[Lx, Ly]
szXY[0,:]/[Lx[0], Ly[0]]
1000*szXY[0,:]/[Lx[0], Ly[0]]
Lx
Ly
umPerPix = 1000*szXY[0,:]/[Lx[0], Ly[0]]
umPerPix
aspect=[]
aspect.append(umPerPix[1]/upPerPix[0])
aspect.append(umPerPix[1]/umPerPix[0])
aspect
aspect=[]
for p in range(0,nplanes):
    Ly=[]
    Lx=[]
    cXY=[]
    szXY=[]   
    for k in range(0,nrois):
        Ly.insert(k, si_rois[k]['scanfields'][p]['pixelResolutionXY'][1])
        Lx.insert(k, si_rois[k]['scanfields'][p]['pixelResolutionXY'][0])
        cXY.insert(k, si_rois[k]['scanfields'][p]['centerXY'])
        szXY.insert(k, si_rois[k]['scanfields'][p]['sizeXY'])
    
    Ly=np.array(Ly)
    Lx=np.array(Lx)
    cXY=np.array(cXY)
    szXY=np.array(szXY)
    
    umPerPix = 1000*szXY[0,:]/[Lx[0], Ly[0]]
    aspect.append(umPerPix[1]/umPerPix[0])
aspect
szXY
umPerPix
aspect=[]
for p in range(0,nplanes):
    Ly=[]
    Lx=[]
    cXY=[]
    szXY=[]   
    for k in range(0,nrois):
        Ly.insert(k, si_rois[k]['scanfields'][p]['pixelResolutionXY'][1])
        Lx.insert(k, si_rois[k]['scanfields'][p]['pixelResolutionXY'][0])
        cXY.insert(k, si_rois[k]['scanfields'][p]['centerXY'])
        szXY.insert(k, si_rois[k]['scanfields'][p]['sizeXY'])
        mmPerPixX[k] = si_rois[k]['scanfields'][p]['pixelToRefTransform'][0][0]
aspect=[]
for p in range(0,nplanes):
    Ly=[]
    Lx=[]
    cXY=[]
    szXY=[]
    mmPerPix_X=[]
    for k in range(0,nrois):
        Ly.insert(k, si_rois[k]['scanfields'][p]['pixelResolutionXY'][1])
        Lx.insert(k, si_rois[k]['scanfields'][p]['pixelResolutionXY'][0])
        cXY.insert(k, si_rois[k]['scanfields'][p]['centerXY'])
        szXY.insert(k, si_rois[k]['scanfields'][p]['sizeXY'])
        mmPerPixX[k] = si_rois[k]['scanfields'][p]['pixelToRefTransform'][0][0]
aspect=[]
for p in range(0,nplanes):
    Ly=[]
    Lx=[]
    cXY=[]
    szXY=[]
    mmPerPix_X=[]
    for k in range(0,nrois):
        Ly.insert(k, si_rois[k]['scanfields'][p]['pixelResolutionXY'][1])
        Lx.insert(k, si_rois[k]['scanfields'][p]['pixelResolutionXY'][0])
        cXY.insert(k, si_rois[k]['scanfields'][p]['centerXY'])
        szXY.insert(k, si_rois[k]['scanfields'][p]['sizeXY'])
        mmPerPix_X[k] = si_rois[k]['scanfields'][p]['pixelToRefTransform'][0][0]

aspect=[]
for p in range(0,nplanes):
    Ly=[]
    Lx=[]
    cXY=[]
    szXY=[]
    mmPerPix_X=[]
    for k in range(0,nrois):
        Ly.insert(k, si_rois[k]['scanfields'][p]['pixelResolutionXY'][1])
        Lx.insert(k, si_rois[k]['scanfields'][p]['pixelResolutionXY'][0])
        cXY.insert(k, si_rois[k]['scanfields'][p]['centerXY'])
        szXY.insert(k, si_rois[k]['scanfields'][p]['sizeXY'])
        mmPerPix_X.insert(k, si_rois[k]['scanfields'][p]['pixelToRefTransform'][0][0])
mmPerPix_X
aspect=[]
for p in range(0,nplanes):
    Ly=[]
    Lx=[]
    cXY=[]
    szXY=[]
    mmPerPIx_Y=[]
    mmPerPix_X=[]
    for k in range(0,nrois):
        Ly.insert(k, si_rois[k]['scanfields'][p]['pixelResolutionXY'][1])
        Lx.insert(k, si_rois[k]['scanfields'][p]['pixelResolutionXY'][0])
        cXY.insert(k, si_rois[k]['scanfields'][p]['centerXY'])
        szXY.insert(k, si_rois[k]['scanfields'][p]['sizeXY'])
        mmPerPix_Y.insert(k, si_rois[k]['scanfields'][p]['pixelToRefTransform'][1][1])
        mmPerPix_X.insert(k, si_rois[k]['scanfields'][p]['pixelToRefTransform'][0][0])
aspect=[]
for p in range(0,nplanes):
    Ly=[]
    Lx=[]
    cXY=[]
    szXY=[]
    mmPerPix_Y=[]
    mmPerPix_X=[]
    for k in range(0,nrois):
        Ly.insert(k, si_rois[k]['scanfields'][p]['pixelResolutionXY'][1])
        Lx.insert(k, si_rois[k]['scanfields'][p]['pixelResolutionXY'][0])
        cXY.insert(k, si_rois[k]['scanfields'][p]['centerXY'])
        szXY.insert(k, si_rois[k]['scanfields'][p]['sizeXY'])
        mmPerPix_Y.insert(k, si_rois[k]['scanfields'][p]['pixelToRefTransform'][1][1])
        mmPerPix_X.insert(k, si_rois[k]['scanfields'][p]['pixelToRefTransform'][0][0])
mmPerPix_Y
x=[], y=[]
x=[]; y=[]
x
y
x=y=[]
x
y
Lx=Ly=cXY=szXY=mmPerPix_X=mmPerPix_Y=aspect=[]
Lx=Ly=cXY=szXY=mmPerPix_X=mmPerPix_Y=[]
for p in range(0,nplanes):
    for k in range(0,nrois):
        Ly.append(si_rois[k]['scanfields'][p]['pixelResolutionXY'][1])
        Lx.append(si_rois[k]['scanfields'][p]['pixelResolutionXY'][0])
        cXY.append(si_rois[k]['scanfields'][p]['centerXY'])
        szXY.append(si_rois[k]['scanfields'][p]['sizeXY'])
        mmPerPix_Y.append(si_rois[k]['scanfields'][p]['pixelToRefTransform'][1][1])
        mmPerPix_X.insert(si_rois[k]['scanfields'][p]['pixelToRefTransform'][0][0])
Lx=Ly=cXY=szXY=mmPerPix_X=mmPerPix_Y=[]
for p in range(0,nplanes):
    for k in range(0,nrois):
        Ly.append(si_rois[k]['scanfields'][p]['pixelResolutionXY'][1])
        Lx.append(si_rois[k]['scanfields'][p]['pixelResolutionXY'][0])
        cXY.append(si_rois[k]['scanfields'][p]['centerXY'])
        szXY.append(si_rois[k]['scanfields'][p]['sizeXY'])
        mmPerPix_Y.append(si_rois[k]['scanfields'][p]['pixelToRefTransform'][1][1])
        mmPerPix_X.append(si_rois[k]['scanfields'][p]['pixelToRefTransform'][0][0])
mmPerPix_Y
Lx
Lx=[]; Ly=[]; cXY=[]; szXY=[]; mmPerPix_X=[]; mmPerPix_Y=[]
for p in range(0,nplanes):
    for k in range(0,nrois):
        Ly.append(si_rois[k]['scanfields'][p]['pixelResolutionXY'][1])
        Lx.append(si_rois[k]['scanfields'][p]['pixelResolutionXY'][0])
        cXY.append(si_rois[k]['scanfields'][p]['centerXY'])
        szXY.append(si_rois[k]['scanfields'][p]['sizeXY'])
        mmPerPix_Y.append(si_rois[k]['scanfields'][p]['pixelToRefTransform'][1][1])
        mmPerPix_X.append(si_rois[k]['scanfields'][p]['pixelToRefTransform'][0][0])
Lx
Ly
mmPerPixY
mmPerPix_Y
mmPerPix_X
aspect=[];
aspect=np.array(mmPerPix_Y) / np.array(mmPerPix_X)
aspect
aspect=np.array(mmPerPix_Y) / np.array(mmPerPix_X)
aspect
szXY
cXY
cXY  = np.array(cXY) - np.array(szXY)/2
cXY
cXY.min
a = cXY.min
a
a[0]
a[0][0]
a = cXY.min()
a
a = cXY.min(dim=1)
a = cXY.min('dim'=1)
min(cXY)
cXY
sum(Ly)
a
cXY
diameter=[60, 12, 50, 12]
del(diameter[0:1])
diameter
diameter=[60, 12, 50, 12]
del(diameter[0:2])
diameter
op['diameter']=[60, 12, 50, 12]
op = {'diameter' : [60, 12, 50, 12]}
op['diameter']
del(op['diameter'][0:2])
op['diameter']
op = {'diameter' : [60, 12, 50, 12]}
del(op['diameter'][0:0])
op['diameter']
for key in op
for key in op:
    if key is 'diameter':
        del(op[key][0:2])
for key in op:
    if key == 'diameter':
        del(op[key][0:2])
        print(op[key])
op
op = {'diameter' : [60, 12, 50, 12]}
for key in op:
    if key == 'diameter':
        del(op[key][0:2])
        print(op[key])
fname = '/mnt/Smirnakis_lab_NAS/georgioskeliris/MECP2/M18/Contrast/contrast_M18_00001_00001.tif'
from tifffile import imread
data = imread(fname)
print('imaging data of shape: ', data.shape)
n_time, Ly, Lx = data.shape
n_time, n_channels, Ly, Lx = data.shape
import suite2p
f_raw = suite2p.io.BinaryFile(Ly=Ly, Lx=Lx, filename=fname)
f_reg = suite2p.io.BinaryFile(Ly=Ly, Lx=Lx, filename='registered_data.bin', n_frames = f_raw.shape[0]) 
ls
import matplotlib as mpl
mpl.plt(f_reg[:,500,500])
mpl.pyplot(f_reg[:,500,500])
f_reg[0,0,0]
f_reg[500,300,300]
f_raw[500,300,300]
import numpy as np
import sys
import os, requests
from pathlib import Path
import matplotlib.pyplot as plt
pwd
cd Desktop/gkel@NAS/
cd MECP2/M18/Contrast/
ops=np.load('ops.json')
ls
ops=json.load('ops.json')
dir
expPath = '/mnt/Smirnakis_lab_NAS/georgioskeliris/MECP2/M18/Contrast/'
fname = 'contrast_M18_00001_00001.tif'
data = imread(os.path.join(froot, fname))
froot = '/mnt/Smirnakis_lab_NAS/georgioskeliris/MECP2/M18/Contrast/'
fname = 'contrast_M18_00001_00001.tif'
data = imread(os.path.join(froot, fname))
ops1 = np.load(os.path.join(froot, "ops.json"))
ops1 = json.load(os.path.join(froot, "ops.json"))
f = open(os.path.join(froot, "ops.json"),'r')
ops = json.load(f)
ops
ops['diameters']
refImg, rmin, rmax, meanImg, rigid_offsets, \
nonrigid_offsets, zest, meanImg_chan2, badframes, \
yrange, xrange = suite2p.registration_wrapper(f_reg, f_raw=f_raw, f_reg_chan2=None, 
                                                   f_raw_chan2=None, refImg=None, 
                                                   align_by_chan2=False, ops=ops)
ops = {**default_ops(), **ops}
defops = suite2p.default_ops()
ops = {**defops, **ops}
ps
ops
refImg, rmin, rmax, meanImg, rigid_offsets, \
nonrigid_offsets, zest, meanImg_chan2, badframes, \
yrange, xrange = suite2p.registration_wrapper(f_reg, f_raw=f_raw, f_reg_chan2=None, 
                                                   f_raw_chan2=None, refImg=None, 
                                                   align_by_chan2=False, ops=ops)
plt.imshow(refImg)
ops['input_format']
ops["input_format"]
ops
fsall, ops1 = get_tif_list(ops)
fsall, ops1 = utils.get_tif_list(ops)
fsall, ops1 = suite2p.utils.get_tif_list(ops)
fsall, ops1 = suite2p.io.utils.get_tif_list(ops)
fsall
fsall[0]
fsall[1]
fsall[-1]

## ---(Wed Jan 10 17:41:28 2024)---
import numpy as np
cd ~
cd s2p_fastdisk/M11/
ls
cd suite2p/
cd plane0/
ls
cd ~
cd Desktop/gkel@NAS/
dir
ls
cd MECP2/
ls
cd M11/Contrast_processed/
ls
cd suite2p/
ls
cd plane0/
ops=np.load('ops.npy')
ops=np.load('ops.npy',allow_pickle=True)
ops.shape
ops
ops.file_path
ops['diameter']
ops['registration']
ops
ops=np.load('ops.npy',allow_pickle=True)
ops[1]
ops[0]
ops
ops.nframes
import suite2p
defops=suite2p.default_ops()
ops=np.load('ops.npy',allow_pickle=False)
ops.getfield
ops.get('diameter')
ops.getfield('diameter')
ops=np.load('ops.npy',allow_pickle=True).item()
ops['diameter']
ops['diameters']
ops.keys
ops.keys()
ops.keys[0]
ops.keys(0)
ops['meanImg']
ops['meanImg'].shape
pwd
plt.imshow(ops['meanImg'])
import matplotlib.pyplot as plt
plt.imshow(ops['meanImg'])
cd ..
pwd
cd plane2
ops=np.load('ops.npy',allow_pickle=True).item()
ops
ops['Ly']
ops['Lx']
ops['nrois']
print(ops)
ops.keys
ops.keys()
ops['save_path0']
ops.keys()
ops['nplanes']
ops['nrois']
ops['aspect']
ops['dx']
ops['dy']
ops['lines']
ops.keys()
ops['iplance']
ops['iplane']
ops['ops_path']
pwd
ls
ops=np.load('ops.npy',allow_pickle=True).item()
Lx=ops['Lx']
Ly=ops['Ly']
cd ~
cd s2p_fastdisk/
ls
cd M11
ls
cd suite2p/plane2/
ls
f_raw = suite2p.io.BinaryFile(Ly=Ly, Lx=Lx, filename='data_raw.bin')
f_reg = suite2p.io.BinaryFile(Ly=Ly, Lx=Lx, filename='reg_data.bin', n_frames = f_raw.shape[0])
ls
refImg, rmin, rmax, meanImg, rigid_offsets, \
nonrigid_offsets, zest, meanImg_chan2, badframes, \
yrange, xrange = suite2p.registration_wrapper(f_reg, f_raw=f_raw, f_reg_chan2=None, 
                                                   f_raw_chan2=None, refImg=None, 
                                                   align_by_chan2=False ops=ops)
refImg, rmin, rmax, meanImg, rigid_offsets, \
nonrigid_offsets, zest, meanImg_chan2, badframes, \
yrange, xrange = suite2p.registration_wrapper(f_reg, f_raw=f_raw, f_reg_chan2=None, 
                                                   f_raw_chan2=None, refImg=None, 
                                                   align_by_chan2=False, ops=ops)