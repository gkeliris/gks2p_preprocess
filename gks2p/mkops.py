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
import xml.etree.ElementTree as ET

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

def get_namespace(tag):
    """Extract XML namespace from tag"""
    if tag.startswith('{'):
        return tag[1:].split('}')[0]
    return ''

def parse_ome_metadata(ome_path):
    tree = ET.parse(ome_path)
    root = tree.getroot()
    namespace = get_namespace(root.tag)
    ns = {'ome': namespace}

    pixels = root.find('.//ome:Pixels', ns)
    if pixels is None:
        raise ValueError("Could not find <Pixels> element in the OME-XML. Check namespace or file structure.")

    try:
        size_x = int(pixels.attrib['SizeX'])
        size_y = int(pixels.attrib['SizeY'])
        size_c = int(pixels.attrib['SizeC'])
        size_z = int(pixels.attrib['SizeZ'])
        size_t = int(pixels.attrib['SizeT'])
    except KeyError as e:
        raise ValueError(f"Missing attribute in <Pixels>: {e}")

    # Attempt to get DeltaT for frame rate
    delta_t = None
    planes = pixels.findall('ome:Plane', ns)
    try:
        delta_t = float(planes[size_c].attrib['DeltaT']) / float(planes[size_c].attrib['TheT'])
    except KeyError as e:
        raise ValueError(f"Missing attribute DeltaT: {e}")

    fs = 1.0 / delta_t

    # Get TIFF filename
    tiffdata = pixels.find('ome:TiffData', ns)
    file_name = None
    if tiffdata is not None:
        uuid_elem = tiffdata.find('ome:UUID', ns)
        if uuid_elem is not None:
            file_name = uuid_elem.attrib.get('FileName')
        else:
            # Try fallback path
            file_name = tiffdata.attrib.get('FileName')
    if file_name is None:
        raise ValueError("Could not extract TIFF file name from <TiffData> or <UUID>.")

    return {
        'Lx': size_x,
        'Ly': size_y,
        'nchannels': size_c,
        'nplanes': size_z,
        'nframes': size_t,
        'fs': fs,
        'tiff_file': file_name
    }


def parse_bruker_xml(xml_path, with_timestamps=False):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    metadata = {}

    # --- Grab header parameters from the first PVStateShard ---
    pvstate = root.find(".//PVStateShard")
    if pvstate is not None:
        for pv in pvstate.findall("PVStateValue"):
            key = pv.get("key")
            val = pv.get("value")

            if key in ["activeMode", "framePeriod", "dwellTime", "samplesPerPixel", "scanLinePeriod"] and val is not None:
                try:
                    val = float(val)
                    if val.is_integer():
                        val = int(val)
                except Exception:
                    pass
                metadata[key] = val

            elif key == "linesPerFrame" and val is not None:
                metadata["Ly"] = int(val)

            elif key == "pixelsPerLine" and val is not None:
                metadata["Lx"] = int(val)

            elif key == "micronsPerPixel":
                for idx in pv.findall("IndexedValue"):
                    if idx.get("index") == "XAxis":
                        metadata["umPerPx_X"] = float(idx.get("value"))
                    elif idx.get("index") == "YAxis":
                        metadata["umPerPx_Y"] = float(idx.get("value"))

            elif key == "laserWavelength":
                idx = pv.find("IndexedValue")
                if idx is not None:
                    metadata["laserWavelength"] = float(idx.get("value"))

            elif key == "zStack" and val is not None:
                try:
                    metadata["nplanes"] = int(val)
                except Exception:
                    pass

    # --- Compute fs from framePeriod ---
    if "framePeriod" in metadata:
        try:
            metadata["fs"] = 1.0 / float(metadata["framePeriod"])
        except ZeroDivisionError:
            metadata["fs"] = None

    # --- Collect frame info ---
    frames = []
    for f in root.findall(".//Frame"):
        if with_timestamps:
            rel = f.get("relativeTime")
            abs_ = f.get("absoluteTime")
            frames.append({
                "index": int(f.get("index")),
                "relativeTime": float(rel) if rel else None,
                "absoluteTime": float(abs_) if abs_ else None
            })
        else:
            frames.append(None)

    metadata["nframes"] = len(frames)

    if with_timestamps:
        metadata["frameTimestamps"] = frames

    # --- Count unique channels ---
    channels = set()
    for f in root.findall(".//Frame"):
        for file in f.findall("File"):
            ch = file.get("channelName")
            if ch:
                channels.add(ch)
    metadata["nchannels"] = len(channels)

    # --- Default nplanes if not detected ---
    if "nplanes" not in metadata:
        metadata["nplanes"] = 1

    return metadata

def generate_ops_from_metadata2(savepath0, dat, db={}, fastdisk=None):
    """
    Generates a Suite2p ops file by extracting metadata from OME or Bruker XML files.
    This is your function, integrated with the new parser.
    """
    if 'pipeline' not in db.keys():
        db['pipeline'] = 'orig'

    # Look for OME sidecar first
    ome_files= [f for f in os.listdir(dat.rawPath) if f.endswith('.ome')]
    if len(ome_files) > 0:
        ometif_files = [f for f in os.listdir(dat.rawPath) if f.endswith('.ome.tif') or f.endswith('.ome.tiff')]
        if len(ometif_files) > 0:
            # This is a simplification; you might need a more robust way to find the right OME file
            ome_path = os.path.join(dat.rawPath, ome_files[0])
            ome_data = parse_ome_metadata(ome_path)  # your existing function
            print(f"Using metadata from {ome_path}")
        else:
            print("No .ome.tif files found, cannot use .ome metadata.")
            ome_data = None    
    else:
        # Look for Bruker XML fallback
        xml_files = [f for f in os.listdir(dat.rawPath) 
             if f.endswith('.xml') and "VoltageRecording" not in f]
        if not xml_files:
            raise FileNotFoundError(f"No .ome or .xml metadata found in {dat.rawPath}")
        xml_path = os.path.join(dat.rawPath, xml_files[0])
        ome_data = parse_bruker_xml(xml_path)
        print(f"Using metadata from {xml_path}")
    
    if not ome_data:
        print("Failed to extract metadata. Aborting ops generation.")
        return None

    # Check for single-frame TIFFs (if not present will stay with dat.rawPath)
    single_frame_tiffs_path = os.path.join(dat.rawPath, "single_frame_tiffs")
    if os.path.isdir(single_frame_tiffs_path):
        dat.rawPath = single_frame_tiffs_path
        print(f"Using single frame TIFFs from {dat.rawPath}")

    # Create OPS according to Suite2p defaults
    ops = suite2p.default_ops()
    ops['data_path'] = [dat.rawPath]
    ops['save_path0'] = savepath0
    ops['fast_disk'] = fastdisk if fastdisk else savepath0
    ops['bruker'] = True
    ops['input_format'] = 'bruker' # or 'ome.tif' if you handle that
    ops['bruker_bidirectional'] = True # Common for resonant scanning

    # Populate metadata
    ops.update(ome_data)

    # Other useful defaults
    ops['num_workers_roi'] = 5
    ops['keep_movie_raw'] = True
    ops['delete_bin'] = False
    ops['batch_size'] = 500
    ops['nimg_init'] = 300
    ops['tau'] = 1.5
    ops['combined'] = False
    ops['nonrigid'] = True
    ops['save_mat'] = False
    ops['anatomical_only'] = 0 
    ops['cellprob_threshold'] = 0.5
    ops['aspect'] = 1.0
    ops['diameter'] = 10
    ops['diameters'] = [10, 10]  # Assuming square pixels; adjust if needed

    # Merge user-provided db params
    ops.update(db)
    ops['save_folder'] = os.path.join(savepath0, 'suite2p')

    # Create output folders
    os.makedirs(ops['save_path0'], exist_ok=True)
    if fastdisk is not None:
        os.makedirs(ops['fast_disk'], exist_ok=True)

    # Save the ops
    pipeline_name = db.get('pipeline', 'orig')
    ops_out_path = os.path.join(ops['save_path0'], f'ops_{pipeline_name}.npy')
    mat_out_path = os.path.join(ops['save_path0'], f'ops_{pipeline_name}.mat')
    
    np.save(ops_out_path, ops)
    scipy.io.savemat(mat_out_path, {"ops": ops})
    print(f"Saved ops file to {ops_out_path}")
    print(f"Saved MATLAB ops file to {mat_out_path}")

    return ops


def generate_ops_from_metadata(savepath0, dat, db={}, fastdisk=None):
    if 'pipeline' not in db.keys():
        db['pipeline'] = 'orig'

    # Look for OME sidecar first
    ome_files = [f for f in os.listdir(dat.rawPath) if f.endswith('.ome')]
    if len(ome_files) == 1:
        ome_path = os.path.join(dat.rawPath, ome_files[0])
        ome_data = parse_ome_metadata(ome_path)  # your existing function
        print(f"Using metadata from {ome_path}")
    else:
        # Look for Bruker XML fallback
        xml_files = [f for f in os.listdir(dat.rawPath) 
             if f.endswith('.xml') and "VoltageRecording" not in f]
        if not xml_files:
            raise FileNotFoundError(f"No .ome or .xml metadata found in {dat.rawPath}")
        xml_path = os.path.join(dat.rawPath, xml_files[0])
        ome_data = parse_bruker_xml(xml_path)
        print(f"Using metadata from {xml_path}")
    
    if not ome_data:
        print("Failed to extract metadata. Aborting ops generation.")
        return None

    # Check for single-frame TIFFs
    single_frame_tiffs_path = os.path.join(dat.rawPath, "single_frame_tiffs")
    if os.path.isdir(single_frame_tiffs_path):
        dat.rawPath = single_frame_tiffs_path
        print(f"Using single frame TIFFs from {dat.rawPath}")

    # Create OPS according to Suite2p defaults
    ops = suite2p.default_ops()
    ops['data_path'] = [dat.rawPath]
    ops['save_path0'] = savepath0
    ops['fast_disk'] = fastdisk if fastdisk else savepath0
    ops['input_format'] = 'bruker'
    ops['bruker'] = True

    # Populate metadata
    ops['fs'] = ome_data['fs']
    ops['nframes'] = ome_data['nframes']
    ops['nplanes'] = ome_data['nplanes']
    ops['nchannels'] = ome_data['nchannels']
    ops['Lx'] = ome_data['Lx']
    ops['Ly'] = ome_data['Ly']

    # Other useful defaults
    #ops['num_workers_roi'] = 5
    ops['keep_movie_raw'] = 1
    ops['delete_bin'] = 0
    ops['batch_size'] = 500
    ops['nimg_init'] = 300
    ops['tau'] = 1.5
    ops['combined'] = 0
    ops['nonrigid'] = 1
    ops['save_mat'] = 0
    ops['anatomical_only'] = 0 
    ops['cellprob_threshold'] = 0.5
    ops['aspect'] = 1.0
    ops['diameter'] = 10
    ops['diameters'] = [10, 10]

    # Merge user-provided db params
    ops = {**ops, **db}
    ops['save_folder'] = 'suite2p_' + db['pipeline']

    # Create output folders
    os.makedirs(ops['save_path0'], exist_ok=True)
    if fastdisk is not None:
        os.makedirs(ops['fast_disk'], exist_ok=True)

    # Save the ops
    np.save(os.path.join(ops['save_path0'], 'ops_' + db['pipeline']), ops)
    scipy.io.savemat(os.path.join(ops['save_path0'], 'ops_' + db['pipeline']) + ".mat", {"ops": ops})

    return ops