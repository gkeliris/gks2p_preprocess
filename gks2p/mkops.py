#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 13:07:12 2023

@author: georgioskeliris
"""
#from ScanImageTiffReader import ScanImageTiffReader
import os
#import json
import numpy as np
import scipy.io
import suite2p
import scanreader

def mkops(savepath0, dat, db={}, fastdisk=None):
    
    """
    Generates and saves a dictionary of operational parameters for Suite2P processing.

    Args:
        savepath0 (str): The base directory where the output ops file will be saved.
        dat (pandas.Series): A pandas Series containing dataset information, including `rawPath` and `firstTiff` attributes.
        db (dict, optional): A dictionary containing database parameters for Suite2P. Defaults to an empty dict.
        fastdisk (str, optional): The directory for fast disk storage. Defaults to None.

    Returns:
        dict: A dictionary containing the operational parameters for Suite2P.

    Notes:
        - The function calculates various parameters required for Suite2P using scan data.
        - It modifies the 'ops' dictionary with custom parameters and saves it in both .npy and .mat formats.
    """

    if 'pipeline' not in db.keys():
        db['pipeline']='orig'
    
    # Create a reader object. Currently using ScanImageTiffReader. Should be
    # adapted for other tiff images.
    #reader = ScanImageTiffReader(os.path.join(dat.rawPath, dat.firstTiff))
    # Get the tiff dimensions. Currently not used I think
    #tiff_dim=reader.shape()

    scan = scanreader.read_scan(os.path.join(dat.rawPath, dat.firstTiff))
    
    # Use the diameter in ops to indicate the aspect ratio of pixels (suite2p convention)
    # Note: suite2p only uses a list of two values (for the first plane) and 
    #       assumes it is the same for other planes (copies). I added diameters
    #       that includes values for each plane and sligtly modified suite2p to
    #       update diameter with the right info for the currently run plane
    diameter=[]; aspect=[]; left_x=[]; top_y=[]
    width_in_degrees=[]; height_in_degrees=[]
    for idx in range(scan.num_fields):
        aspect.append(
            (scan.field_heights_in_microns[idx] / scan.field_heights[idx]) / 
            (scan.field_widths_in_microns[idx] / scan.field_widths[idx])
        )
        diameter.append(10)
        diameter.append(round(10*aspect[idx]))
        # From the centers and dimensions of each field we calculate the location 
        # of the top left corner of each field. This will help us later to convert 
        # this into pixels i.e. dx, dy from the top-left corner of the most
        # left and top field.
        width_in_degrees.append(scan.fields[idx].width_in_degrees)
        height_in_degrees.append(scan.fields[idx].height_in_degrees)
        left_x.append(scan.fields[idx].x-width_in_degrees[idx]/2)
        top_y.append(scan.fields[idx].y-height_in_degrees[idx]/2)
        

    # create OPS according to the default in suite2p and overwrite some values
    ops=suite2p.default_ops()

    ops['fs'] = scan.fps
    ops['nchannels'] = scan.num_channels
    ops['num_scanning_depths'] = scan.num_scanning_depths
    ops['nrois'] = scan.num_rois
    plane_nrois = {i:scan.field_depths.count(i) for i in scan.field_depths}
    ops['nroisPerPlane'] = [plane_nrois[i] for i in plane_nrois.keys()]
    ops['iplane'] = scan.field_slices
    ops['nplanes'] = scan.num_fields
    if scan.is_multiROI:
        ops['mesoscan'] = 1
    else:
        ops['mesoscan'] = 0
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
    ops['aspects'] = [float(x) for x in aspect]
    if ops['mesoscan']:
        ops['allLx']=scan.field_widths
        ops['allLy']=scan.field_heights
        ops['dx']=[]; ops['dy']=[]; ops['slices']=[]; ops['lines']=[]
        for field_idx, field_info in enumerate(scan.fields):
            # NOTE: when the pixel dimensions of the fields are different
            #       it is impossible to calculate the dx, dy correctly without
            #       upsampling to a higher resolution single matrix. Here we 
            #       calculate the dx, dy based on the resolution of the most 
            #       upper-left field in each plane
            cur_plane = scan.field_slices[field_idx]
            idx_cur_plane = np.array(scan.field_slices)==cur_plane
            idx_selected_ref_field_cur_plane = np.argmin(
                np.array(top_y)[idx_cur_plane]
                [np.array(left_x)[idx_cur_plane]==
                 np.min(np.array(left_x)[idx_cur_plane])]
                )
            pix_dx_plane = np.array(width_in_degrees)[idx_cur_plane] \
                [idx_selected_ref_field_cur_plane] \
                / np.array(scan.field_widths)[idx_cur_plane] \
                [idx_selected_ref_field_cur_plane]
            pix_dy_plane = np.array(height_in_degrees)[idx_cur_plane] \
                [idx_selected_ref_field_cur_plane] \
                / np.array(scan.field_heights)[idx_cur_plane] \
                [idx_selected_ref_field_cur_plane]
            left_x_plane = np.array(left_x)[idx_cur_plane] \
                [idx_selected_ref_field_cur_plane]
            top_y_plane = np.array(top_y)[idx_cur_plane] \
                [idx_selected_ref_field_cur_plane]
            
            ops['dx'].append(int((left_x[field_idx]-left_x_plane)/pix_dx_plane))
            ops['dy'].append(int((top_y[field_idx]-top_y_plane)/pix_dy_plane))
            ops['slices'].append(field_info.slice_id)
            ops['lines'].append(
                np.arange(field_info.yslices[0].start, field_info.yslices[0].stop)
            )
        #remove potentially negative numbers from ops['dx'] and ops['dy']
        ops['dx']=[int(x) for x in list(np.array(ops['dx'])-min(np.array(ops['dx'])))]
        ops['dy']=[int(x) for x in list(np.array(ops['dy'])-min(np.array(ops['dy'])))]
        print(ops['dx'])
        print(ops['dy'])

    # add the paths for the data        
    ops['data_path'] = [dat.rawPath]
    ops['save_path0'] = savepath0
    ops['save_folder'] = 'suite2p_' + db['pipeline']
    if fastdisk != None:
        ops['fast_disk'] = fastdisk
    
    ops = {**ops, **db}
    # create folders
    os.makedirs(ops['save_path0'], exist_ok=True)
    if fastdisk != None:
        os.makedirs(ops['fast_disk'], exist_ok=True)
    # save the ops
    np.save(os.path.join(ops['save_path0'], 'ops_' + db['pipeline']),ops)
    scipy.io.savemat(os.path.join(ops['save_path0'], 'ops_' + db['pipeline']) + ".mat", {"ops": ops})
    return ops
