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
import suite2p


def mkops_old(basepath, dat, db={}):
    
    if 'pipeline' not in db.keys():
        db['pipeline']='orig'
    
    # Create a reader object. Currently using ScanImageTiffReader. Should be
    # adapted for other tiff images.
    reader = ScanImageTiffReader(os.path.join(dat.rawPath, dat.firstTiff))
    # Get the tiff dimensions. Currently not used I think
    tiff_dim=reader.shape()
    # Read the header info and convert to a json object
    h=reader.metadata()
    js=h.find('{\n')
    hh=h[1:js-2].splitlines()
    json_obj=json.loads(h[js:-1])

    # Create and fill a dictionary with key : value pairs
    hhDict={}
    for s in hh:
        eq_ind = s.find('=')
        hhDict[s[:eq_ind-1]] = s[eq_ind+2:]
    
    # Get the sampling rate
    fsV = float(hhDict['SI.hRoiManager.scanVolumeRate'])
    fsF = float(hhDict['SI.hRoiManager.scanFrameRate'])
    # Get the z planes
    zsstr = hhDict['SI.hStackManager.zs']
    if zsstr[0] == '[':
        zs = zsstr[1:-1].split(" ")
    else:
        zs = [zsstr]
    # This is to avoid the fact of a very long list of the same z in some datasets
    zs = list(dict.fromkeys(zs))
    print('zs=' + str(zs))
    # This is another way if previous problematic
    # tmpzs = []
    # for x in zs:
    #     if x not in tmpzs:
    #         tmpzs.append(x)
    # zs=tmpzs

    # Calculate the number of planes
    nplanes = len(zs)
    # Get the number of channels
    if hhDict['SI.hChannels.channelsActive']=='[1;2]':
        nChannels=2;
    else:
        nChannels=1;
    
    # Decode the ROI info
    si_rois = json_obj['RoiGroups']['imagingRoiGroup']['rois']
    if type(si_rois) is dict:
        si_rois = [si_rois]
    nrois = len(si_rois)
    rois_zs = [si_rois[f]['zs'] for f in range(nrois)]
    if type(rois_zs[0]) is list:
        rois_zs = set(sum(rois_zs,[]))

    # Exceptions
    if (dat.cohort=='coh1') & (dat.mouseID=='M25') & (dat.week=='w11'):
        whichplane=[0, 0, 1, 1]
        whichroi=[0, 1, 0, 1]
        whichscanfield=[0, 0, 0, 0]
        nroisPerPlane=np.array([2, 2])
    elif (dat.cohort=='coh1') & (dat.mouseID=='M117') & (dat.week=='w22'):
        whichplane=[0, 0, 1, 1, 2, 2]
        whichroi=[0, 1, 0, 1, 0, 1]
        whichscanfield=[0, 0, 0, 0, 0, 0]
        nroisPerPlane=np.array([2, 2, 2])
    elif len(rois_zs) < len(zs): # This probably solves the above two as well
        whichplane =sum([[f] * nrois for f in range(nplanes)],[])
        whichroi=sum([[f for f in range(nrois)] * nplanes],[])
        whichscanfield=[0] * (nrois * nplanes)
        nroisPerPlane=np.array([nrois] * nplanes)
        print('special case\n')
    else: # read values of rois per plane etc. to later loop and get info
        nroisPerPlane = np.zeros(nplanes)
        whichplane = []
        whichroi = []
        whichscanfield=[]
        for p in range(0, nplanes):
            for kk in range(0, nrois):
                if np.any(np.isin(si_rois[kk]['zs'], int(zs[p]))):
                    nroisPerPlane[p] = nroisPerPlane[p]+1
                    whichplane.append(p)
                    whichroi.append(kk)
                    tmp=np.where(np.array(si_rois[kk]['zs'])==int(zs[p]))
                    whichscanfield.append(tmp[0][0])
    
    #nplanes = len(np.unique(whichplane))
    
    # Loop across planes and extact info of each ROI
    pln = [{'plane':x,'Ly':[], 'Lx':[], 'cY':[], 'cX':[], 'szY':[], 'szX':[]} for x in range(0,nplanes)]
    aspect=[]
    for i in range(0,len(whichplane)):
        if len(np.unique(whichscanfield)) > 1:
            pln[whichplane[i]]['Ly'].append(si_rois[whichroi[i]]['scanfields'][whichscanfield[i]]['pixelResolutionXY'][1])
            pln[whichplane[i]]['Lx'].append(si_rois[whichroi[i]]['scanfields'][whichscanfield[i]]['pixelResolutionXY'][0])
            pln[whichplane[i]]['cY'].append(si_rois[whichroi[i]]['scanfields'][whichscanfield[i]]['centerXY'][1])
            pln[whichplane[i]]['cX'].append(si_rois[whichroi[i]]['scanfields'][whichscanfield[i]]['centerXY'][0])
            pln[whichplane[i]]['szY'].append(si_rois[whichroi[i]]['scanfields'][whichscanfield[i]]['sizeXY'][1])
            pln[whichplane[i]]['szX'].append(si_rois[whichroi[i]]['scanfields'][whichscanfield[i]]['sizeXY'][0])
            
            mmPerPix_Y=si_rois[whichroi[i]]['scanfields'][whichscanfield[i]]['pixelToRefTransform'][1][1]
            mmPerPix_X=si_rois[whichroi[i]]['scanfields'][whichscanfield[i]]['pixelToRefTransform'][0][0]
            
        else:
            pln[whichplane[i]]['Ly'].append(si_rois[whichroi[i]]['scanfields']['pixelResolutionXY'][1])
            pln[whichplane[i]]['Lx'].append(si_rois[whichroi[i]]['scanfields']['pixelResolutionXY'][0])
            pln[whichplane[i]]['cY'].append(si_rois[whichroi[i]]['scanfields']['centerXY'][1])
            pln[whichplane[i]]['cX'].append(si_rois[whichroi[i]]['scanfields']['centerXY'][0])
            pln[whichplane[i]]['szY'].append(si_rois[whichroi[i]]['scanfields']['sizeXY'][1])
            pln[whichplane[i]]['szX'].append(si_rois[whichroi[i]]['scanfields']['sizeXY'][0])
            mmPerPix_Y=si_rois[whichroi[i]]['scanfields']['pixelToRefTransform'][1][1]
            mmPerPix_X=si_rois[whichroi[i]]['scanfields']['pixelToRefTransform'][0][0]
        
        aspect.append(np.array(mmPerPix_Y) / np.array(mmPerPix_X)) 
            

    # Calculate the flyback lines to remove from lines in the ops
    n_rows_sum = np.zeros(nplanes)
    n_flybackP = np.zeros(nplanes)
    for p in range(0,nplanes):
        n_rows_sum[p] = sum(pln[p]['Ly'])
        n_flybackP[p] = (tiff_dim[1] - n_rows_sum[p]) / max(1, (nroisPerPlane[p] - 1))

    # deduce flyback from most filled z-plane    
    n_flyback = np.min(n_flybackP)

    # Calculate some values to extract the rows in the tiff files for each ROI
    muX=[]; muY=[]; iminX=[]; iminY=[]; irow1=[]; irow2=[]; allLx=[]; allLy=[]
    for p in range(0,nplanes):
        pln[p]['cX'] = np.array(pln[p]['cX']) - np.array(pln[p]['szX'])/2
        pln[p]['cX'] = pln[p]['cX'] - np.min(pln[p]['cX']) 
        pln[p]['cY'] = np.array(pln[p]['cY']) - np.array(pln[p]['szY'])/2
        pln[p]['cY'] = pln[p]['cY'] - np.min(pln[p]['cY'])    
        for r in range(0,int(nroisPerPlane[p])):
            allLx.append(pln[p]['Lx'][r])
            allLy.append(pln[p]['Ly'][r])
            muX = np.array(pln[p]['Lx'][r])/np.array(pln[p]['szX'][r])
            muY = np.array(pln[p]['Ly'][r])/np.array(pln[p]['szY'][r])
            iminX.append(pln[p]['cX'][r] * muX)
            iminY.append(pln[p]['cY'][r] * muY)
        
        irow_tmp = np.delete(np.insert(np.cumsum(np.array(pln[p]['Ly']) + n_flyback), 0, 0),-1,0)
        irow_tmp2=irow_tmp + np.array(pln[p]['Ly'])
        irow1.append(list(irow_tmp))
        irow2.append(list(irow_tmp2))

    irow1=[item for row in irow1 for item in row]
    irow2=[item for row in irow2 for item in row]

    # Use the diameter in ops to indicate the aspect ratio of pixels (suite2p convention)
    # Note: suite2p only uses a list of two values (for the first plane) and 
    #       assumes it is the same for other planes (copies). I added diameters
    #       that includes values for each plane and sligtly modified suite2p to
    #       update diameter with the right info for the currently run plane
    diameter=[]
    for a in range(0,len(aspect)):
        diameter.append(10)
        diameter.append(round(10*aspect[a]))
    
    # added nroisPerPlane to help detect/run multi-plane datasets in suite2p
    nroisPerPlane=nroisPerPlane[nroisPerPlane>0]

    # create OPS according to the default in suite2p and overwrite some values
    ops=suite2p.default_ops()
    if nplanes == 1:
        ops['fs'] = fsF
    else:
        ops['fs'] = fsV

    ops['nchannels'] = nChannels
    
    ops['nrois'] = nrois
    ops['nroisPerPlane'] = list(np.int16(nroisPerPlane))
    ops['iplane'] = whichplane
    ops['nplanes'] = len(whichplane)
    if nrois * nplanes == 1:
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
    ops['aspects'] = aspect
    if ops['mesoscan']:
        ops['allLx']=allLx
        ops['allLy']=allLy
        ops['dx']=[]; ops['dy']=[]; ops['lines']=[]
        for i in range(0,len(aspect)):
            ops['dx'].append(int(iminX[i]))
            ops['dy'].append(int(iminY[i]))
            ops['lines'].append([x for x in range(int(irow1[i]),int(irow2[i]))])
    # add the paths for the data        
    ops['data_path'] = [dat.rawPath]
    ops['save_path0'] = os.path.join(basepath, 's2p_analysis', dat.cohort, 
            dat.mouseID, dat.week, dat.session, dat.expID)
    ops['save_folder'] = 'suite2p_' + db['pipeline']
    ops['fast_disk'] = os.path.join(basepath, 's2p_binaries', dat.cohort, 
                            dat.mouseID, dat.week, dat.session, dat.expID)
    
    ops = {**ops, **db}
    # create folders
    os.makedirs(ops['save_path0'], exist_ok=True)
    os.makedirs(ops['fast_disk'], exist_ok=True)
    # save the ops
    np.save(os.path.join(ops['save_path0'], 'ops_' + db['pipeline']),ops)

    return ops
