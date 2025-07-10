import os
import numpy as np
import xml.etree.ElementTree as ET
import suite2p


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

def generate_ops_from_ome(ome_path, output_dir):
    ome_dir = os.path.dirname(ome_path)
    ome_data = parse_ome_metadata(ome_path)

    tiff_file = ome_data['tiff_file']
    tiff_path = os.path.abspath(os.path.join(ome_dir, tiff_file))
    
    # create OPS according to the default in suite2p and overwrite some values
    ops=suite2p.default_ops()

    db = {
        'nframes': ome_data['nframes'],
        'nplanes': ome_data['nplanes'],
        'nchannels': ome_data['nchannels'],
        'functional_chan': 2,
        'fs': ome_data['fs'],
        'Lx': ome_data['Lx'],
        'Ly': ome_data['Ly'],
        'diameter': 10,
        'diameters:': [10, 10],
        'data_path': [os.path.dirname(tiff_path)],
        #'tiff_list': [tiff_path],
        'planes_to_process': [0],
        'save_path0': output_dir,
        'look_one_level_down': False,
        'do_registration': True,
        'keep_movie_raw': False,
        'roidetect': True,
        'neuropil_extract': True,
        'spatial_scale': 1.0,
        'fast_disk': [output_dir],
        'input_format': 'bruker',
    }
    ops = {**ops, **db}
    output_path = os.path.join(output_dir, 'ops.npy')
    np.save(output_path, ops)
    print(f"Saved ops.npy to: {output_path}")
