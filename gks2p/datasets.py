#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 13:35:10 2024

@author: georgioskeliris
"""

import pandas as pd
import numpy as np
import suite2p
from ScanImageTiffReader import ScanImageTiffReader
defaultCsvPath = '/mnt/NAS_DataStorage/Data_raw/Temperature_epilepsy/TEPI_datasets.csv'

def datasetQuery(csvFilePath=defaultCsvPath,
                 cohort=[], timepoint=[], mouseID=[], ses=None, experiment=None):
    # read the datasets.csvs
    df = pd.read_csv(csvFilePath)
    if type(cohort) == str:
        cohort = [cohort]
    if type(timepoint) == str:
        timepoint = [timepoint]
    if type(mouseID) == str:
        mouseID = [mouseID]
        
    if cohort == []:
        cohort = list(df["cohort"].unique())
    if timepoint == []:
        timepoint = list(df["timepoint"].unique())
    if mouseID == []:
        mouseID = list(df["mouseID"].unique())
    if experiment is None:
        exps = list(df["expID"].unique())
    else:
        # account for possibility for exp2, exp3, exp4
        exps = [(experiment + str(x)) for x in range(2,6)]
        exps.append(experiment)
        
    ds = df[(df["cohort"].isin(cohort)) & (df["timepoint"].isin(timepoint)) & 
            (df["mouseID"].isin(mouseID)) & (df["expID"].isin(exps))]
    
    datIDs=[]
    if ses is not None:
        if len(ses)>4:
            ds = ds[(ds["session"]==ses)]
        else:
            for d in range(0,len(ds)):
                if ds["session"].iloc[d][0:4]==ses:
                    datIDs.append(ds["datID"].iloc[d])       
            ds = ds[(ds["datID"].isin(datIDs))]
    return ds

def getDataPaths(cohort, day, experiment, csvFilePath=defaultCsvPath):
    df = pd.read_csv(csvFilePath)
    # account for possibility for exp2, exp3, exp4
    exps = [(experiment + str(x)) for x in range(2,6)]
    exps.append(experiment)
    
    ds = df[(df["cohort"] == cohort) & (df["day"]== day) & (df["expID"].isin(exps))]
    return ds

def getOneExpPath(cohort, day, mouseID, experiment, csvFilePath=defaultCsvPath):
    df = pd.read_csv(csvFilePath)
    # account for possibility for exp2, exp3, exp4
    exps = [(experiment + str(x)) for x in range(2,6)]
    exps.append(experiment)
    
    ds = df[(df["cohort"] == cohort) & (df["day"]== day) & (df["mouseID"]==mouseID) & (df["expID"].isin(exps))]
    return ds

def getOneSesPath(cohort, day, mouseID, sesX, experiment):
    ds = getOneExpPath(cohort, day, mouseID, experiment)
    # account for possibility for exp2, exp3, exp4
    exps = [(experiment + str(x)) for x in range(2,6)]
    exps.append(experiment)
    
    for d in range(0,len(ds)):
        if ds["session"].iloc[d][0:4]==sesX:
            ds=ds[(ds["session"] == ds["session"].iloc[d])]
            break
    return ds

def checkDatasets(ds_path):
    fs, dif = suite2p.io.utils.list_files(ds_path,False,["*.tif","*.tiff"])
    print('\n\nCHECKING DATA INTEGRITY:')
    out = {'PASS':[],'PASS_ind':[],'FAIL':[],'FAIL_ind':[],'EXCEPTION':[]}
    for t in range(0,len(fs)):
        try: 
            ScanImageTiffReader(fs[t])
            print(fs[t][-8:],'\t-> OK')
            out['PASS'].append(fs[t][-8:])
            out['PASS_ind'].append(t)
        except Exception as error:
            print(fs[t][-8:],'\t-> CORRUPTED')
            out['FAIL'].append(fs[t][-8:])
            out['FAIL_ind'].append(t)
            out['EXCEPTION'].append(error)
            # handle the exception
            print("\t\t\tAn exception occurred:", type(error).__name__, "-", error)
    return out

def checkTifFile(ds_path, fileIndex):
    fs, dif = suite2p.io.utils.list_files(ds_path,False,["*.tif","*.tiff"])
    print('\n\nCHECKING DATA INTEGRITY:')
    try:
        ScanImageTiffReader(fs[fileIndex])
        print(fs[fileIndex][-8:],'\t-> OK')
        out=True
    except Exception as error:
        print(fs[fileIndex][-8:],'\t-> CORRUPTED')
        out=False
    return out

def getTimeStamps(ds_path, fileIndex):
    fs, dif = suite2p.io.utils.list_files(ds_path,False,["*.tif","*.tiff"])
    reader = ScanImageTiffReader(fs[fileIndex])
    stack = reader.data()
    timestamps=np.empty(stack.shape[0])
    for fr in range(stack.shape[0]):
        des = reader.description(fr)
        desList = des.splitlines()
        tmp = desList[3]
        tmp2 = tmp.split(" = ")
        timestamps[fr] = float(tmp2[1])
    
    return timestamps
        