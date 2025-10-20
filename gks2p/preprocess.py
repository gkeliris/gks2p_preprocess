#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 19:08:18 2024

@author: georgioskeliris
"""
import numpy as np
import sys
from pathlib import Path
from natsort import natsorted
import shutil
import os

# Import helpers from mkops; also import tifffile for compatibility with some readers
from gks2p.mkops import mkops, generate_ops_from_metadata, generate_ops_from_metadata2, parse_bruker_xml
import tifffile
from typing import Optional, Dict

# Optional heavy dependencies: allow importing preprocess even when
# suite2p/fissa are not installed. Functions that need them will raise
# informative ImportError at runtime.
try:
    import suite2p
    from suite2p import registration
except Exception:
    suite2p = None
    registration = None

try:
    import fissa
except Exception:
    fissa = None

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
                dat.mouseID, dat.timepoint, dat.session, dat.expID)
    elif pathType == "fast_disk":
        outpath = os.path.join(basepath, 's2p_binaries', dat.cohort, 
                dat.mouseID, dat.timepoint, dat.session, dat.expID)
    else:
        print("unknown type of path")
    return outpath
    
def gks2p_makeOps(ds, basepath, db={}, fastbase=None, combine_folder=None):
    if fastbase == None:
        fastbase = basepath
    if combine_folder is not None:
        subfolders = []
        for d in range(0,len(ds)):
            subfolders.append(ds.iloc[d].rawPath)
        db_combine = {
            'look_one_level_down': True,
            'subfolders': subfolders
        }
        db = {**db, **db_combine}
        head, tail = os.path.split(gks2p_path(ds.iloc[0],basepath))
        save_path0 = os.path.join(head,combine_folder)
        head, tail = os.path.split(gks2p_path(ds.iloc[0],fastbase,'fast_disk'))
        fastdisk = os.path.join(head,combine_folder)
        ops = mkops(save_path0, ds.iloc[0], db, fastdisk=fastdisk)
    else:
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
    return ops

def gks2p_makeOps_bruker(ds, basepath, db={}, fastbase=None):
    for d in range(0,len(ds)):
        print('\n\nPROCESSING:')
        print(ds.iloc[d])
        try:
            ops = generate_ops_from_metadata(gks2p_path(ds.iloc[d],basepath), ds.iloc[d], db, 
                            fastdisk=gks2p_path(ds.iloc[d],fastbase,'fast_disk'))
        except Exception as error:
            # handle the exception
            print('\n****** -> PROBLEM WITH THIS DATASET ****\n')
            print("An exception occurred:", type(error).__name__, "-", error)
    return ops


def gks2p_makeOps_bruker_multifile(ds, basepath, db={}, fastbase=None):
    """Compatibility helper: same as `gks2p_makeOps_bruker` but calls the
    multifile-aware generator `generate_ops_from_metadata2`.

    This mirrors older code paths and is provided for notebooks/scripts that
    expect the `_multifile` helper to exist.
    """
    for d in range(0, len(ds)):
        print('\n\nPROCESSING:')
        print(ds.iloc[d])
        try:
            ops = generate_ops_from_metadata2(
                gks2p_path(ds.iloc[d], basepath), ds.iloc[d], db,
                fastdisk=gks2p_path(ds.iloc[d], fastbase, 'fast_disk')
            )
        except Exception as error:
            # handle the exception
            print('\n****** -> PROBLEM WITH THIS DATASET ****\n')
            print("An exception occurred:", type(error).__name__, "-", error)
    return ops

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

def gks2p_import(dat, import_folder, basepath, fastbase=None, db={}, bruker=False):
    if fastbase==None:
        fastbase=basepath
    if bruker:
        ops = generate_ops_from_metadata(gks2p_path(dat,basepath), dat, db, 
                fastdisk=gks2p_path(dat,fastbase,'fast_disk'))
    else:
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
        print(dat.rawPath)
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
            if ops['bruker']:
                cur_iplaneList = [0]  # For Bruker data, we only register the first plane   
            else:
                cur_iplaneList=[x for x in range(len(ops['dx']))]
        else:
            cur_iplaneList=iplaneList
        
        for iplane in cur_iplaneList:
            print("\nREGISTERING: plane" + str(iplane))
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

            # Channel 1 (main)
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

            # Channel 2 (optional)
            chan2_path = os.path.join(ops['fast_disk'],'suite2p',
                'plane' + str(iplane), 'data_chan2_raw.bin')
            if os.path.isfile(chan2_path):
                f2 = suite2p.io.BinaryFile(Ly=Ly, Lx=Lx, filename=chan2_path)
                if pipeline=='orig':
                    reg_file2="data_chan2.bin"
                else:
                    reg_file2="data_chan2" + pipeline + ".bin"
                print(f"Found channel 2 data for plane {iplane}. Registering both channels.")

                f2_reg = suite2p.io.BinaryFile(Ly=Ly, Lx=Lx,
                    filename=os.path.join(ops['fast_disk'],'suite2p',
                    'plane' + str(iplane), reg_file2), n_frames = n_frames)
            else:
                f2 = None
                f2_reg = None

            # Switch channels if specified
            if ops['switch_chan'] == 1:
                f1, f2 = f2, f1
                print("Switched channels for registration.")
            
            registration_outputs = suite2p.registration_wrapper(
                f1_reg, f_raw=f1, f_reg_chan2= f2_reg,
                f_raw_chan2=f2, refImg=None,
                align_by_chan2=ops['align_by_chan'], ops=opsPlane
            )

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
                    f1_reg, f_raw=None, f_reg_chan2=f2_reg, f_raw_chan2=None,
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


def gks2p_smooth(
    ds,
    basepath,
    pipeline='orig',
    method='block',
    method_kwargs: Optional[Dict] = None,
    bin_target: str = 'auto',
    iplaneList=None,
    inplace: bool = True,
    update_ops: bool = True,
    verbose: bool = True,
):
    """
    High-level helper to apply temporal smoothing to Suite2p binaries for datasets.

    Parameters
    - ds, basepath: same as other helpers (dataset dataframe and base path)
    - pipeline: which suite2p pipeline folder to target (e.g. 'orig' or custom)
    - method: smoothing method passed to the smoothing wrapper (block, gaussian, ...)
    - method_kwargs: dict of method-specific kwargs (e.g., x=2 or sigma_frames=1.5)
    - bin_target: control which binaries to smooth when multiple exist per plane.
        * 'auto' (default): prefer *_raw binaries if present, otherwise registered
        * 'raw_only': require *_raw binaries; skip registered ones
        * 'registered_only': process registered binaries (data.bin / data_{pipeline}.bin)
        * 'both': process both raw and registered variants (raw first)
    - iplaneList: list of plane indices to process; defaults to all planes in ops['dx']
    - inplace: if True, replace input .bin (smooth wrapper handles rename/backup)
    - update_ops: if True, update the per-plane ops.npy nframes key after smoothing
    - verbose: passed through to the smoothing routine

    This function mirrors the pattern used by `gks2p_register` / `gks2p_segment`:
    it finds the per-plane `ops.npy` (falling back to suite2p_orig), merges with the
    dataset-level ops, determines Lx/Ly and the binary path and then calls the
    smoothing wrapper from `gks2p.suite2p_temporal_smoothing`.
    """
    if method_kwargs is None:
        method_kwargs = {}

    valid_targets = {'auto', 'raw_only', 'registered_only', 'both'}
    if bin_target not in valid_targets:
        raise ValueError(f"bin_target must be one of {valid_targets}, got '{bin_target}'")

    opsList = gks2p_loadOps(ds, basepath, pipeline)
    for d in range(len(ds)):
        dat = ds.iloc[d]
        print(dat)
        ops = opsList[d]

        if iplaneList is None:
            # Try a sequence of fallbacks to determine number of planes so the
            # function works for single-plane Bruker datasets (where 'dx' may
            # not be present) as well as multi-plane Mesoscope datasets.
            if ops.get('bruker'):
                # Bruker (single-plane) — default to plane 0
                cur_iplaneList = [0]
            else:
                # Preferred: use ops['dx'] if available
                dx = ops.get('dx', None)
                if dx is not None:
                    try:
                        cur_iplaneList = list(range(len(dx)))
                    except Exception:
                        cur_iplaneList = [0]
                else:
                    # Try common n-planes keys
                    nplanes = None
                    for k in ('nplanes', 'nPlanes', 'nPlanesTot', 'nPlanes_total'):
                        if k in ops:
                            try:
                                nplanes = int(ops[k])
                                break
                            except Exception:
                                nplanes = None
                    if nplanes is not None and nplanes > 0:
                        cur_iplaneList = list(range(nplanes))
                    else:
                        # Fallback: inspect the suite2p folder for plane subfolders
                        cur_iplaneList = None
                        save_path0 = ops.get('save_path0')
                        save_folder = ops.get('save_folder', 'suite2p_orig')
                        if save_path0 is not None:
                            candidate = os.path.join(save_path0, save_folder)
                            try:
                                if os.path.isdir(candidate):
                                    plane_dirs = natsorted([f.name for f in os.scandir(candidate) if f.is_dir() and (f.name[:5] == 'plane' or f.name == 'combined')])
                                    plane_idxs = []
                                    for name in plane_dirs:
                                        if name == 'combined':
                                            continue
                                        try:
                                            plane_idxs.append(int(name.replace('plane', '')))
                                        except Exception:
                                            pass
                                    if plane_idxs:
                                        cur_iplaneList = sorted(plane_idxs)
                            except Exception:
                                cur_iplaneList = None

                        # If all else fails assume single plane
                        if cur_iplaneList is None:
                            cur_iplaneList = [0]
        else:
            cur_iplaneList = iplaneList

        for iplane in cur_iplaneList:
            print("\nSMOOTHING: plane" + str(iplane))
            pathstr = os.path.join(ops['save_path0'], 'suite2p_' + pipeline, 'plane' + str(iplane))
            opsstr = os.path.join(pathstr, 'ops.npy')
            if not os.path.isfile(opsstr):
                opsstr_orig = os.path.join(ops['save_path0'], 'suite2p_orig', 'plane' + str(iplane), 'ops.npy')
                os.makedirs(pathstr, exist_ok=True)
                opsPlane = np.load(opsstr_orig, allow_pickle=True).item()
                # Persist a copy into the pipeline folder so updates operate on the
                # pipeline ops file (ops.npy) rather than only the original copy.
                try:
                    np.save(opsstr, opsPlane, allow_pickle=True)
                except Exception as e:
                    print(f"Warning: failed to write pipeline ops file at {opsstr}: {e}")
            else:
                opsPlane = np.load(opsstr, allow_pickle=True).item()

            # Merge dataset-level ops into plane ops (plane ops take precedence)
            opsPlane = {**opsPlane, **ops}

            Ly = opsPlane['Ly']
            Lx = opsPlane['Lx']
            nch = opsPlane.get('nchannels', 1)
            # dtype may be stored under different names; try common keys then fallback
            dtype = opsPlane.get('dtype', opsPlane.get('movie_dtype', 'int16')) or 'int16'

            # Prefer the registered/processed movie for the chosen pipeline, fall back to raw
            reg_file = 'data.bin' if pipeline == 'orig' else f"data_{pipeline}.bin"
            bin_path = os.path.join(ops['fast_disk'], 'suite2p', 'plane' + str(iplane), reg_file)
            if not os.path.isfile(bin_path):
                alt = os.path.join(ops['fast_disk'], 'suite2p', 'plane' + str(iplane), 'data_raw.bin')
                if os.path.isfile(alt):
                    bin_path = alt
                else:
                    print(f"Bin file not found for plane {iplane}: {bin_path} (skipping)")
                    continue

            # Import lazily to avoid heavy deps at module import time
            from gks2p.suite2p_temporal_smoothing import (
                smooth_suite2p_bin,
                update_ops_nframes,
                BinSpec,
                _ensure_parent_dir,
            )

            # Prepare smoothing kwargs for this plane without mutating caller input
            call_kwargs = dict(method_kwargs)
            # Ensure sensible defaults for smoothing parameters. For the
            # 'block' method require an integer 'x' >= 2; default to 5 if not
            # provided so single-plane calls without explicit kwargs work.
            if method == 'block' and call_kwargs.get('x', None) is None:
                call_kwargs['x'] = 5
            if inplace and call_kwargs.get('out_dtype', None) is None:
                # Preserve Suite2p expectations when replacing the original binary
                call_kwargs['out_dtype'] = dtype

            # Prepare a list of candidate binary filenames to smooth for this plane
            # Build registered/raw variants; ordering decided by bin_target
            reg_file = 'data.bin' if pipeline == 'orig' else f"data_{pipeline}.bin"
            reg_file2 = 'data_chan2.bin' if pipeline == 'orig' else f"data_chan2_{pipeline}.bin"
            reg_candidates = [reg_file, reg_file2]
            raw_candidates = ['data_raw.bin', 'data_chan2_raw.bin']

            def _exists(name: str) -> bool:
                return os.path.isfile(os.path.join(ops['fast_disk'], 'suite2p', f'plane{iplane}', name))

            if bin_target == 'raw_only':
                candidates = raw_candidates
            elif bin_target == 'registered_only':
                candidates = reg_candidates
            elif bin_target == 'both':
                candidates = raw_candidates + reg_candidates
            else:  # 'auto'
                if any(_exists(f) for f in raw_candidates):
                    candidates = raw_candidates
                elif any(_exists(f) for f in reg_candidates):
                    candidates = reg_candidates
                else:
                    # fallback: preserve legacy behaviour (try everything)
                    candidates = raw_candidates + reg_candidates

            processed = []
            backups = []
            for fname in candidates:
                bin_candidate = os.path.join(ops['fast_disk'], 'suite2p', 'plane' + str(iplane), fname)
                if not os.path.isfile(bin_candidate):
                    continue
                try:
                    final_path, backup = smooth_suite2p_bin(
                        bin_candidate,
                        Lx=Lx,
                        Ly=Ly,
                        nchannels=nch,
                        dtype=dtype,
                        method=method,
                        out_path=call_kwargs.get('out_path', None),
                        out_dtype=call_kwargs.get('out_dtype', 'float32'),
                        inplace=inplace,
                        downsample_factor=call_kwargs.get('downsample_factor', None),
                        downsample_mode=call_kwargs.get('downsample_mode', 'decimate'),
                        keep_remainder=call_kwargs.get('keep_remainder', False),
                        x=call_kwargs.get('x', None),
                        sigma_frames=call_kwargs.get('sigma_frames', None),
                        tau_frames=call_kwargs.get('tau_frames', None),
                        window_frames=call_kwargs.get('window_frames', None),
                        sg_window=call_kwargs.get('sg_window', None),
                        sg_polyorder=call_kwargs.get('sg_polyorder', None),
                        truncate=call_kwargs.get('truncate', 3.0),
                        chunk_frames=call_kwargs.get('chunk_frames', 64),
                        mode=call_kwargs.get('mode', 'reflect'),
                        verbose=verbose,
                    )
                    processed.append((bin_candidate, final_path, backup))
                    if backup is not None:
                        backups.append(backup)
                    print(f"Smoothed {bin_candidate} -> {final_path} (backup={backup})")
                except Exception as e:
                    print(f"Error while smoothing {bin_candidate}: {e}")

            # Write a short smooth.txt log in the plane folder describing what happened
            try:
                from datetime import datetime
                log_lines = [f"timestamp: {datetime.utcnow().isoformat()}Z", f"method: {method}", f"method_kwargs: {call_kwargs}", "processed:"]
                for p in processed:
                    log_lines.append(f"  - input: {p[0]} -> output: {p[1]} backup: {p[2]}")
                log_path = os.path.join(pathstr, 'smooth.txt')
                _ensure_parent_dir(log_path)
                with open(log_path, 'a') as fh:
                    fh.write('\n'.join(log_lines) + '\n')
            except Exception:
                pass

            # Optionally update ops.npy with the new frame count for this plane
            # and also attempt to update ops files two levels up (if present).
            if update_ops and processed:
                # Prefer first processed output to compute new nframes
                final_path = processed[0][1]
                # Use backup from the first processed file if available
                backup = processed[0][2] if processed[0][2] is not None else (backups[0] if backups else None)
                spec_new = BinSpec(path=final_path, Lx=Lx, Ly=Ly, nchannels=nch, dtype=call_kwargs.get('out_dtype', dtype))
                try:
                    new_T = spec_new.n_frames()
                    # Backup and update plane ops (opsstr)
                    if os.path.isfile(opsstr):
                        try:
                            # create a timestamped backup of the ops file before changing
                            from datetime import datetime
                            ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
                            plane_ops_backup = opsstr + f'.bak.{ts}'
                            shutil.copy2(opsstr, plane_ops_backup)
                        except Exception:
                            plane_ops_backup = None

                        # determine original nframes if available so updater can rescale fs
                        original_n = None
                        for k in ('nframes', 'nFrames', 'nFramesTot'):
                            if k in opsPlane:
                                try:
                                    original_n = int(opsPlane[k])
                                    break
                                except Exception:
                                    original_n = None

                        if original_n is None:
                            try:
                                spec_old = BinSpec(path=processed[0][0], Lx=Lx, Ly=Ly, nchannels=nch, dtype=opsPlane.get('dtype', 'int16'))
                                original_n = spec_old.n_frames()
                            except Exception:
                                original_n = None

                        scale_sampling = call_kwargs.get('scale_sampling', True)
                        update_ops_nframes(opsstr, new_T, original_nframes=original_n, scale_sampling=bool(scale_sampling), backup_path=backup)
                        print(f"Updated {opsstr} with nframes={new_T} (scale_sampling={bool(scale_sampling)})")
                    else:
                        print(f"ops file not found at {opsstr}; skipping ops update")

                    # Try to update ops files two levels up (e.g., suite2p pipeline-level ops)
                    two_up = os.path.dirname(os.path.dirname(pathstr))
                    try:
                        # look for ops*.npy files in this folder
                        from datetime import datetime
                        for f in os.listdir(two_up):
                            if f.startswith('ops') and f.endswith('.npy'):
                                candidate_ops = os.path.join(two_up, f)
                                try:
                                    # backup
                                    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
                                    candidate_backup = candidate_ops + f'.bak.{ts}'
                                    shutil.copy2(candidate_ops, candidate_backup)
                                    update_ops_nframes(candidate_ops, new_T, original_nframes=original_n, scale_sampling=bool(scale_sampling), backup_path=backup)
                                    print(f"Updated {candidate_ops} with nframes={new_T} (backup saved: {candidate_backup})")
                                except Exception as e:
                                    print(f"Failed to update ops file {candidate_ops}: {e}")
                    except Exception:
                        pass
                except Exception as e:
                    print(f"Failed to compute/update nframes for {final_path}: {e}")

    return

def gks2p_segment(ds, basepath, pipeline='orig', iplaneList=None):

    opsList = gks2p_loadOps(ds, basepath, pipeline)    
    for d in range(len(ds)):
        dat=ds.iloc[d] # Convert the pandas DataFrame to a pandas Series
        print(dat)
        ops = opsList[d]

        if iplaneList is None:
            if ops['bruker']:
                cur_iplaneList = [0]  # For Bruker data, we only register the first plane   
            else:
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
    
    
def gks2p_fissa(ds, basepath, iplaneList=None, nCores=None, use_reg_tif=False, redo_prep=False):
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
            
            if use_reg_tif:
                images = os.path.join(opsPP['save_path'],'reg_tif')
            else:
                tmp = suite2p.io.BinaryFile(Ly=Ly, Lx=Lx, filename=opsPP['reg_file'])
                images = [np.array(tmp[:,:,:])]

            output_folder = os.path.join(opsPP['save_path'],'FISSA')
            experiment = fissa.Experiment(
                images,
                [rois[:ncells]],
                output_folder,
                ncores_preparation=nCores,
                ncores_separation=nCores,
                verbosity=6
            )
            experiment.separate(redo_prep=redo_prep)
            
            sampling_frequency = ops['fs']  # Hz
            experiment.calc_deltaf(freq=sampling_frequency)
            
            experiment.to_matfile()

    return experiment


def gks2p_split_bruker_multipage_tif(input_folder):
    """
    Splits all multipage TIFF files in a folder into individual OME-TIFF files,
    one for each frame, and saves them to the same folder.
    Moves the original TIFF files to a subfolder called 'original_tiffs'.
    """
    import shutil

    # Create subfolder for original TIFFs
    single_tiffs_folder = os.path.join(input_folder, 'single_frame_tiffs')
    os.makedirs(single_tiffs_folder, exist_ok=True)

    # Get a list of all TIFF files in the input folder
    tif_files = [f for f in os.listdir(input_folder) if f.endswith('.tif') or f.endswith('.tiff')]

    # Initialize frame counters for each channel
    frame_counters = {'Ch1': 0, 'Ch2': 0, 'other': 0}

    # Iterate over each TIFF file in the input folder
    for file in tif_files:
        file_path = os.path.join(input_folder, file)

        # Check if the file name includes Ch1 or Ch2 to determine which counter to use
        if 'Ch1' in file:
            counter_key = 'Ch1'
        elif 'Ch2' in file:
            counter_key = 'Ch2'
        else:
            counter_key = 'other'

        # Load the multipage TIFF file
        with tifffile.TiffFile(file_path) as tif:
            num_frames = len(tif.pages)
            for frame_index in range(num_frames):
                filename, file_extension = os.path.splitext(file)
                new_file = f"{filename}_frame_{frame_counters[counter_key]:06d}{file_extension}"
                new_file_path = os.path.join(input_folder, single_tiffs_folder, new_file)
                frame_data = tif.pages[frame_index].asarray()
                tifffile.imwrite(new_file_path, frame_data)
                frame_counters[counter_key] += 1

        # Move the original TIFF file to the 'original_tiffs' subfolder
        # shutil.move(file_path, os.path.join(original_tiffs_folder, file))
