#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 19:08:18 2024

@author: georgioskeliris
"""
import numpy as np
import suite2p
from suite2p import registration
import fissa
import sys
from pathlib import Path 
from natsort import natsorted
import shutil
import os
from gks2p.mkops import mkops

def clear_all():
    """Clears all the variables from the workspace of the spyder application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]

# CHANGE THIS ACCORDING TO THE DATA STRUCTURE YOU HAVE
def gks2p_path(dat, basepath, pathType="save_path0"):
    if pathType == "save_path0":
        outpath = os.path.join(basepath, 's2p_analysis', dat.cohort, 
                dat.mouseID, dat.day, dat.session, dat.expID)
    elif pathType == "fast_disk":
        outpath = os.path.join(basepath, 's2p_binaries', dat.cohort, 
                dat.mouseID, dat.day, dat.session, dat.expID)
    else:
        print("unknown type of path")
    return outpath
    
def gks2p_makeOps(ds, basepath, db={}, fastbase=None):
    if fastbase == None:
        fastbase = basepath
    for d in range(0,len(ds)):
        print('\n\nPROCESSING:')
        print(ds.iloc[d])
        try:
            ops = mkops(gks2p_path(ds.iloc[d],basepath), ds.iloc[d], db, 
                        fastdisk=gks2p_path(ds.iloc[d],fastbase,'fast_disk'))
        except Exception as error:
        # handle the exception
            print('\n****** -> PROBLEM WITH THIS DATASET ****\n')
            print("An exception occurred:", type(error).__name__, "-", error)
    return

def gks2p_loadOps(ds, basepath, pipeline="orig"):
    opsPath = []
    for d in range(len(ds)):
        dat = ds.iloc[d]  # Convert the pandas DataFrame to a pandas Series
        opsPath.append(os.path.join(gks2p_path(dat,basepath), 'ops_' + pipeline + '.npy'))
    ops = [np.load(f, allow_pickle=True).item() for f in opsPath]
    return ops

def gks2p_loadOpsPerPlane(save_folder):
    plane_folders = natsorted([ f.path for f in os.scandir(save_folder) if f.is_dir() and f.name[:5]=='plane'])
    ops1 = [np.load(os.path.join(f, 'ops.npy'), allow_pickle=True).item() for f in plane_folders]
    
    return ops1

def gks2p_updateOpsPaths(ds, basepath, fastbase=None, pipeline="orig"):
    if fastbase==None:
        fastbase=basepath
    for d in range(len(ds)):
        dat = ds.iloc[d]  # Convert the pandas DataFrame to a pandas Series
        npops = np.load(os.path.join(gks2p_path(dat,basepath), \
                    'ops_' + pipeline + '.npy'), allow_pickle=True)
        ops=npops.item()
        ops['save_path0']= gks2p_path(dat,basepath)
        ops['fast_disk'] = gks2p_path(ds.iloc[d],fastbase,'fast_disk')
        np.save(os.path.join(ops['save_path0'], 'ops_' + pipeline),ops)
    return

def gks2p_updateOpsPerPlane(ds, basepath, fastbase=None, pipeline="orig"):
    if fastbase==None:
        fastbase=basepath
    ops = gks2p_loadOps(ds, basepath, pipeline=pipeline)
    for d in range(len(ops)):
        save_folder = os.path.join(ops[d]['save_path0'],ops[d]['save_folder'])
        plane_folders = natsorted([ f.path for f in os.scandir(save_folder) if f.is_dir() \
                    and (f.name[:5]=='plane' or f.name=='combined')])
        for f in plane_folders:
            head, curFolder = os.path.split(f)
            ops1 = [np.load(os.path.join(f, 'ops.npy'), allow_pickle=True).item()]
            #ops1[0] = {**ops1[0], **ops[d]}
            ops1[0]['save_path']=f
            ops1[0]['save_path0']=gks2p_path(ds.iloc[d],basepath)
            ops1[0]['fast_disk']=gks2p_path(ds.iloc[d],fastbase,'fast_disk')
            ops1[0]['ops_path']=os.path.join(f,'ops.npy')
            ops1[0]['save_folder']= 'suite2p_' + pipeline
            if curFolder != 'combined':
                if 'raw_file' in ops1[0]:
                    head, raw_file = os.path.split(ops1[0]['raw_file'])
                    ops1[0]['raw_file']=os.path.join(ops1[0]['fast_disk'],'suite2p',curFolder,raw_file)
                if 'reg_file' in ops1[0]:
                    head, reg_file = os.path.split(ops1[0]['reg_file'])
                    ops1[0]['reg_file']=os.path.join(ops1[0]['fast_disk'],'suite2p',curFolder,reg_file)            
            np.save(ops1[0]['ops_path'],ops1[0])
    return

def gks2p_import(dat, import_folder, basepath, fastbase=None, db={}):
    if fastbase==None:
        fastbase=basepath
    ops = mkops(gks2p_path(dat,basepath), dat, db, 
                fastdisk=gks2p_path(dat,fastbase,'fast_disk'))
    os.makedirs(os.path.join(gks2p_path(dat,basepath),'matlabana'), exist_ok=True)
    [shutil.copy(mf,os.path.join(gks2p_path(dat,basepath),'matlabana')) for mf in \
         os.scandir(import_folder) if mf.name[-4:]=='.mat']
    plane_folders_src = natsorted([ f.path for f in \
                os.scandir(os.path.join(import_folder,'suite2p')) if f.is_dir() \
                and (f.name[:5]=='plane' or f.name=='combined')])
    for f in plane_folders_src:
        head, fld = os.path.split(f)
        dst = os.path.join(ops['save_path0'],'suite2p_orig',fld)
        os.makedirs(dst, exist_ok=True)
        [shutil.copy(ff,dst) for ff in os.scandir(f) if ff.name[-4:]=='.npy']
        if fld!='combined':
            dstbin = os.path.join(ops['fast_disk'],'suite2p',fld)
            os.makedirs(dstbin, exist_ok=True)
            [shutil.copy(ff,dstbin) for ff in os.scandir(f) if ff.name[-4:]=='.bin']
    
    return
    
def gks2p_toBinary(ds, basepath):
    opsList = gks2p_loadOps(ds, basepath)
    for d in range(len(ds)):
        dat=ds.iloc[d] # Convert the pandas DataFrame to a pandas Series
        print(dat)
        ops = opsList[d]
        suite2p.run_s2p_toBinary(ops=ops)
        #suite2p.run_planes(ops=ops)
    return


def gks2p_register(ds, basepath, pipeline='orig', iplaneList=None):
    
    opsList = gks2p_loadOps(ds, basepath, pipeline)
    for d in range(len(ds)):
        dat=ds.iloc[d] # Convert the pandas DataFrame to a pandas Series
        print(dat)
        ops = opsList[d]
        
        if iplaneList is None:
            cur_iplaneList=[x for x in range(len(ops['dx']))]
        else:
            cur_iplaneList=iplaneList
        
        for iplane in cur_iplaneList:
            print("\nREGISTERING: plane" + str(iplane))
            #opsstr= os.path.join(ops['save_path0'],'suite2p_' + pipeline,'plane' + str(iplane), 'ops.npy')
            pathstr= os.path.join(ops['save_path0'],
                                  'suite2p_' + pipeline,'plane' + str(iplane))
            opsstr=os.path.join(pathstr,'ops.npy')
            if not os.path.isfile(opsstr):
                opsstr_orig= os.path.join(ops['save_path0'],'suite2p_orig','plane' + str(iplane), 'ops.npy')
                os.makedirs(pathstr, exist_ok=True)
                opsPlane=np.load(opsstr_orig,allow_pickle=True).item()
            else:
                opsPlane=np.load(opsstr,allow_pickle=True).item()
            opsPlane = {**opsPlane, **ops}
            Ly=opsPlane['Ly']
            Lx=opsPlane['Lx']
            
            f1 = suite2p.io.BinaryFile(Ly=Ly, Lx=Lx,
                            filename=os.path.join(ops['fast_disk'],'suite2p',
                            'plane' + str(iplane), 'data_raw.bin'))
            n_frames = f1.shape[0]
            if pipeline=='orig':
                reg_file="data.bin"
            else:
                reg_file="data_" + pipeline + ".bin"
            f1_reg = suite2p.io.BinaryFile(Ly=Ly, Lx=Lx,
                            filename=os.path.join(ops['fast_disk'],'suite2p',
                            'plane' + str(iplane), reg_file), n_frames = n_frames)
        
            registration_outputs = suite2p.registration_wrapper(f1_reg, f_raw=f1, f_reg_chan2=None, 
                                                               f_raw_chan2=None, refImg=None, 
                                                               align_by_chan2=False, ops=opsPlane)
            
            suite2p.registration.register.save_registration_outputs_to_ops(registration_outputs, opsPlane)
            # add enhanced mean image
            meanImgE = suite2p.registration.compute_enhanced_mean_image(
                            opsPlane["meanImg"].astype(np.float32), opsPlane)
            opsPlane["meanImgE"] = meanImgE
            np.save(opsstr,opsPlane)
            
            if opsPlane["two_step_registration"] and opsPlane["keep_movie_raw"]:
                print("----------- REGISTRATION STEP 2")
                print("(making mean image (excluding bad frames)")
                nsamps = min(n_frames, 1000)
                inds = np.linspace(0, n_frames, 1 + nsamps).astype(np.int64)[:-1]

                refImg = f1_reg[inds].astype(np.float32).mean(axis=0)
                registration_outputs = suite2p.registration_wrapper(
                    f1_reg, f_raw=None, f_reg_chan2=None, f_raw_chan2=None,
                    refImg=refImg, align_by_chan2=False, ops=opsPlane)
                np.save(opsstr,opsPlane)
            
            # compute metrics for registration
            if ops.get("do_regmetrics", True) and n_frames >= 1500:
                
                # n frames to pick from full movie
                nsamp = min(2000 if n_frames < 5000 or Ly > 700 or Lx > 700 else 5000,
                            n_frames)
                inds = np.linspace(0, n_frames - 1, nsamp).astype("int")
                mov = f1_reg[inds]
                mov = mov[:, opsPlane["yrange"][0]:opsPlane["yrange"][-1],
                          opsPlane["xrange"][0]:opsPlane["xrange"][-1]]
                opsPlane = suite2p.registration.get_pc_metrics(mov, opsPlane)
                np.save(opsstr,opsPlane)
    return

def gks2p_segment(ds, basepath, pipeline='orig', iplaneList=None):

    opsList = gks2p_loadOps(ds, basepath, pipeline)    
    for d in range(len(ds)):
        dat=ds.iloc[d] # Convert the pandas DataFrame to a pandas Series
        print(dat)
        ops = opsList[d]

        if iplaneList is None:
            cur_iplaneList=[x for x in range(len(ops['dx']))]
        else:
            cur_iplaneList=iplaneList
        
       # original = sys.stdout
       # sys.stdout = open(os.path.join(ops['save_path0'], ops['save_folder'], "run.log"), "a")
        
        for iplane in cur_iplaneList:
            print("\nSEGMENTING: plane" + str(iplane))
            pathstr= os.path.join(ops['save_path0'],
                                  'suite2p_' + pipeline,'plane' + str(iplane))
            opsstr=os.path.join(pathstr,'ops.npy')
            if not os.path.isfile(opsstr):
                opsstr_orig= os.path.join(ops['save_path0'],'suite2p_orig','plane' + str(iplane), 'ops.npy')
                os.makedirs(pathstr, exist_ok=True)
                opsPlane=np.load(opsstr_orig,allow_pickle=True).item()
            else:
                opsPlane=np.load(opsstr,allow_pickle=True).item()
            opsPlane = {**opsPlane, **ops}
            Ly=opsPlane['Ly']
            Lx=opsPlane['Lx']

            # Use default classification file provided by suite2p 
            classfile = suite2p.classification.builtin_classfile
            #np.load(classfile, allow_pickle=True)[()]
            
            f_reg = suite2p.io.BinaryFile(Ly=Ly, Lx=Lx, 
                    filename=os.path.join(opsPlane['fast_disk'],'suite2p', 
                                      'plane' + str(iplane), 'data.bin'))
            
            opsPlane, stat = suite2p.detection_wrapper(f_reg=f_reg, 
                                            ops=opsPlane, classfile=classfile)
            
            np.save(opsstr,opsPlane)
            np.save(os.path.join(pathstr,'stat.npy'),stat)
                
            # Fluorescence Extraction
            stat_after_extraction, F, Fneu, F_chan2, Fneu_chan2 = \
                suite2p.extraction_wrapper(stat, f_reg, f_reg_chan2 = None,
                                           ops=opsPlane)
            
            np.save(os.path.join(pathstr,'stat.npy'),stat_after_extraction)
            np.save(os.path.join(pathstr,'F.npy'),F)
            np.save(os.path.join(pathstr,'Fneu.npy'),Fneu)
            
            
            # Cell Classification
            iscell = suite2p.classify(stat=stat_after_extraction, classfile=classfile)
            np.save(os.path.join(pathstr,'iscell.npy'),iscell)
        
            # Spike Deconvolution
            # Correct our fluorescence traces 
            dF = F.copy() - opsPlane['neucoeff']*Fneu
            # Apply preprocessing step for deconvolution
            dF = suite2p.extraction.preprocess(
                    F=dF,
                    baseline=opsPlane['baseline'],
                    win_baseline=opsPlane['win_baseline'],
                    sig_baseline=opsPlane['sig_baseline'],
                    fs=opsPlane['fs'],
                    prctile_baseline=opsPlane['prctile_baseline']
                )
            # Identify spikes
            spks = suite2p.extraction.oasis(F=dF, batch_size=opsPlane['batch_size'], 
                                            tau=opsPlane['tau'], fs=opsPlane['fs'])
            np.save(os.path.join(pathstr,'spks.npy'),spks)
    
    #sys.stdout =  original
    return

def gks2p_classify(ds, basepath, pipeline="orig", iplaneList=None, classfile=None):
    
    if classfile is None:
        classfile = suite2p.classification.builtin_classfile
    np.load(classfile, allow_pickle=True)[()]
    
    opsList = gks2p_loadOps(ds, basepath, pipeline)
    for d in range(len(ds)):
        dat=ds.iloc[d] # Convert the pandas DataFrame to a pandas Series
        print(dat)
        ops = opsList[d]
        
        if iplaneList is None:
            iplaneList=[x for x in range(len(ops['dx']))]
            
        for iplane in iplaneList:
            print("\nCLASSIFYING: plane" + str(iplane))
            pathstr= os.path.join(ops['save_path0'],
                                  'suite2p_' + pipeline,'plane' + str(iplane))
            stat_after_extraction = np.load(os.path.join(pathstr,'stat.npy'), allow_pickle=True)
            iscell = suite2p.classify(stat=stat_after_extraction, classfile=classfile)
            np.save(os.path.join(pathstr,'iscell.npy'),iscell)
    return

def gks2p_deconvolve(ds, basepath, tau, pipeline="orig", iplaneList=None):
    
    opsList = gks2p_loadOps(ds, basepath, pipeline)
    for d in range(len(ds)):
        dat=ds.iloc[d] # Convert the pandas DataFrame to a pandas Series
        print(dat)
        ops = opsList[d]
        ops['tau_deconvolution'] = tau
        np.save(os.path.join(ops['save_path0'], 'ops_' + ops['pipeline']),ops)
        if iplaneList is None:
            iplaneList=[x for x in range(len(ops['dx']))]
            
        for iplane in iplaneList:
            print("\nDECONVOLVING: plane" + str(iplane))
            pathstr= os.path.join(ops['save_path0'],
                                  'suite2p_' + pipeline,'plane' + str(iplane))
            opsstr=os.path.join(pathstr,'ops.npy')
            opsPlane=np.load(opsstr,allow_pickle=True).item()
            opsPlane['tau_deconvolution'] = tau
            np.save(opsstr,opsPlane)
            
            F = np.load(os.path.join(pathstr,'F.npy'))
            Fneu = np.load(os.path.join(pathstr,'Fneu.npy'))
            
            # Spike Deconvolution
            # Correct our fluorescence traces 
            dF = F.copy() - opsPlane['neucoeff']*Fneu
            # Apply preprocessing step for deconvolution
            dF = suite2p.extraction.preprocess(
                    F=dF,
                    baseline=opsPlane['baseline'],
                    win_baseline=opsPlane['win_baseline'],
                    sig_baseline=opsPlane['sig_baseline'],
                    fs=opsPlane['fs'],
                    prctile_baseline=opsPlane['prctile_baseline']
                )
            # Identify spikes
            spks = suite2p.extraction.oasis(F=dF, batch_size=opsPlane['batch_size'], 
                                            tau=tau, fs=opsPlane['fs'])
            np.save(os.path.join(pathstr,'spks.npy'),spks)
        
        print('\nCombining planes...\n')
        out = suite2p.io.combined(os.path.join(ops['save_path0'],ops['save_folder']), save=True)
    return
            
def gks2p_combine(ds, basepath, pipeline="orig"):
    opsList = gks2p_loadOps(ds, basepath)
    for d in range(len(ds)):
        ops = opsList[d]
        out = suite2p.io.combined(os.path.join(ops['save_path0'],ops['save_folder']), save=True)
    return

'''
def gks2p_opsPerPlane(ds, basepath, pipeline="orig", iplaneList=None):
    
    for d in range(len(ds)):
        dat=ds.iloc[d] # Convert the pandas DataFrame to a pandas Series
        print(dat)
        opsPath = os.path.join(basepath, 's2p_analysis', dat.cohort,
                               dat.mouseID, dat.week, dat.session,
                               dat.expID, 'ops_' + pipeline + '.npy')
        ops = np.load(opsPath, allow_pickle=True).item() # Load ops as a dict
        
        if iplaneList is None:
            iplaneList=[x for x in range(len(ops['dx']))]
        
        
        for iplane in iplaneList:
            pathstr= os.path.join(ops['save_path0'],
                                  'suite2p_' + pipeline,'plane' + str(iplane))
            opsstr=os.path.join(pathstr,'ops.npy')
            opsPlane=np.load(opsstr,allow_pickle=True).item()
    
        ops1=suite2p.io.utils.init_ops(ops)
'''

def gks2p_correctOpsPerPlane(ds, basepath):
    for d in range(len(ds)):
        ops = mkops(basepath, ds.iloc[d])
        head, n = os.path.split(ops['save_path0'])
        src = os.path.join(head, 'pipeline1', n, 'suite2p')
        dst = os.path.join(head, n, 'suite2p_orig')
        shutil.move(src, dst)
        ops1 = gks2p_loadOpsPerPlane(os.path.join(ops['save_path0'],ops['save_folder']))
        for i in range(len(ops1)):
            pathstr= os.path.join(ops['save_path0'],
                                  'suite2p_orig','plane' + str(i))
            ops1[i]['save_path0']=ops['save_path0']
            ops1[i]['save_folder']=ops['save_folder']
            ops1[i]['save_path']=pathstr
            np.save(os.path.join(ops1[i]['save_path'],'ops.npy'),ops1[i])
    return


# To correct some datasets that were directing to a different fastdisk
'''
ops = gks2p_loadOps(ds, basepath)
ops1=gks2p_loadOpsPerPlane(os.path.join(ops[0]['save_path0'],ops[0]['save_folder']))
for i in range(len(ops1)):
    #ops1[i]['fast_disk'] = os.path.join(basepath, 's2p_binaries', dat.cohort, 
    #                    dat.mouseID, dat.week, dat.session, dat.expID)
    ops1[i]['spatial_scale']=2

    np.save(os.path.join(ops1[i]['save_path'],'ops.npy'),ops1[i])
'''
    
    
def gks2p_fissa(ds, basepath, iplaneList=None):
    opsList = gks2p_loadOps(ds, basepath)
    for d in range(len(ds)):
        ops = opsList[d]
        svf = os.path.join(ops['save_path0'],ops['save_folder'])
        opsPPlist = gks2p_loadOpsPerPlane(svf)
        if iplaneList is None:
            iplaneList=[x for x in range(len(opsPPlist))]
        for p in iplaneList:
            opsPP = opsPPlist[p]
            stat=np.load(os.path.join(opsPP['save_path'],'stat.npy'),allow_pickle=True)
            iscell=np.load(os.path.join(opsPP['save_path'],'iscell.npy'),allow_pickle=True)[:,0]

            # Get image size
            Lx = opsPP['Lx']
            Ly = opsPP['Ly']
            
            # Get the cell ids
            ncells = len(stat)
            cell_ids = np.arange(ncells)  # assign each cell an ID, starting from 0.
            cell_ids = cell_ids[iscell == 1]  # only take the ROIs that are actually cells.
            num_rois = len(cell_ids)
            
            # Generate ROI masks in a format usable by FISSA (in this case, a list of masks)
            rois = [np.zeros((Ly, Lx), dtype=bool) for n in range(num_rois)]
            
            empty_rois=[]
            for i, n in enumerate(cell_ids):
                # i is the position in cell_ids, and n is the actual cell number
                ypix = stat[n]["ypix"][~stat[n]["overlap"]]
                if len(ypix)==0:
                    empty_rois.append(n)
                xpix = stat[n]["xpix"][~stat[n]["overlap"]]
                rois[i][ypix, xpix] = 1
            
            if len(empty_rois):
                print("\nThe following empty ROIs were found:\n")
                print(empty_rois)
                print("\nPlease correct in suite2p GUI and try again\n")
                return
            
            #imagesBin = suite2p.io.BinaryFile(Ly=Ly, Lx=Lx, filename=opsPP[0]['reg_file'])
            images = os.path.join(opsPP['save_path'],'reg_tif')
            output_folder = os.path.join(opsPP['save_path'],'FISSA')
            experiment = fissa.Experiment(images, [rois[:ncells]], output_folder)
            experiment.separate()

    return