#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 13:07:12 2023

@author: georgioskeliris
"""
from ScanImageTiffReader import ScanImageTiffReader
import os
import json
import numpy as np


def mkops(basepath, dat):

    reader = ScanImageTiffReader(os.path.join(dat.rawPath, dat.firstTiff))
    tiff_dim=reader.shape()
    h=reader.metadata()
    js=h.find('{\n')
    hh=h[1:js-2].splitlines()
    json_obj=json.loads(h[js:-1])

    hhDict={}
    for s in hh:
        eq_ind = s.find('=')
        hhDict[s[:eq_ind-1]] = s[eq_ind+2:]
        
    fsV = float(hhDict['SI.hRoiManager.scanVolumeRate'])
    fsF = float(hhDict['SI.hRoiManager.scanFrameRate'])
    zsstr = hhDict['SI.hStackManager.zs']
    zs = zsstr[1:-1].split(" ")
    nplanes = len(zs)

    si_rois = json_obj['RoiGroups']['imagingRoiGroup']['rois']
    nrois = len(si_rois)


    imin=[]; irow=[]; aspect=[]
    for p in range(0,nplanes):
        LXY=[]; Lx=[]; Ly=[]; cXY=[]; szXY=[]; mmPerPix_X=[]; mmPerPix_Y=[]
        for k in range(0,nrois):
            Ly.append(si_rois[k]['scanfields'][p]['pixelResolutionXY'][1])
            Lx.append(si_rois[k]['scanfields'][p]['pixelResolutionXY'][0])
            LXY.append([Lx[k], Ly[k]])
            cXY.append(si_rois[k]['scanfields'][p]['centerXY'])
            szXY.append(si_rois[k]['scanfields'][p]['sizeXY'])
        
            mmPerPix_Y=si_rois[k]['scanfields'][p]['pixelToRefTransform'][1][1]
            mmPerPix_X=si_rois[k]['scanfields'][p]['pixelToRefTransform'][0][0]
        
            aspect.append(np.array(mmPerPix_Y) / np.array(mmPerPix_X)) 
            
        cXY = np.array(cXY) - np.array(szXY)/2
        cXY = np.array(cXY) - cXY.min(axis=0) 
        mu = np.array(LXY) / np.array(szXY)
        imin_tmp = np.array(cXY) * np.array(mu)
        imin.append(imin_tmp)
        
        if p == 0:
            n_rows_sum = sum(Ly)
            n_flyback = (tiff_dim[1] - n_rows_sum) / max(1, (nrois - 1))
            
        irow_tmp = np.delete(np.insert(np.cumsum(np.array(Ly) + n_flyback), 0, 0),-1,0)
        irow_tmp2=irow_tmp + np.array(Ly)
        irow.append(irow_tmp)
        irow.append(irow_tmp2)

    diameter=[]
    for a in range(0,nplanes*nrois):
        diameter.append(10)
        diameter.append(round(10*aspect[a]))
        
        
    ops={}
    if nplanes == 1:
        ops['fs'] = fsF
    else:
        ops['fs'] = fsV

    ops['nchannels'] = int(hhDict['SI.hChannels.channelsAvailable'])
    ops['nplanes'] = nplanes
    ops['nrois'] = nrois
    if nrois == 1:
        ops['mesoscan'] = 0
    else:
        ops['mesoscan'] = 1
    ops['diameter']=diameter[:2]
    ops['diameters']=diameter
    ops['num_workers_roi'] = 5
    ops['keep_movie_raw'] = 1
    ops['delete_bin'] = 0
    ops['batch_size'] = 500
    ops['nimg_init'] = 300
    ops['tau'] = 1.5
    ops['combined'] = 1
    ops['nonrigid'] = 1
    ops['save_mat'] = 0
    ops['anatomical_only'] = 0 
    ops['cellprob_threshold'] = 0.5
    ops['aspect'] = 1.0

    if ops['mesoscan']:
        dxdy=np.reshape(np.array(imin), [nplanes*2,2])
        dx=[]; dy=[]
        for i in range(0,len(irow)):
            dx.append(int(np.round(dxdy[i,0])))
            dy.append(int(np.round(dxdy[i,1])))
            
        ops['dx'] = dx
        ops['dy'] = dy
    ops['save_path0'] = os.path.join(basepath, 's2p_analysis', dat.cohort, dat.mouseID, dat.week, dat.session,'pipeline1', dat.expID)
    os.makedirs(ops['save_path0'], exist_ok=True)
    print(ops['save_path0'])
    ops['fastdisk'] = os.path.join(basepath, 's2p_binaries', dat.cohort, dat.mouseID, dat.week, dat.session, dat.expID)
    os.makedirs(ops['fastdisk'], exist_ok=True)
    np.save(os.path.join(ops['save_path0'], 'ops_orig.npy'),ops)
    print('\n')
    print(ops)
    return ops