# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from datetime import datetime
from dateutil import tz
from pathlib import Path
from neuroconv import ConverterPipe
from neuroconv.datainterfaces import TiffImagingInterface, Suite2pSegmentationInterface

file_path = OPHYS_DATA_PATH / "imaging_datasets" / "Tif" / "demoMovie.tif"
interface_tiff = TiffImagingInterface(file_path=file_path, sampling_frequency=15.0, verbose=False)

folder_path= OPHYS_DATA_PATH / "segmentation_datasets" / "suite2p"
interface_suit2p = Suite2pSegmentationInterface(folder_path=folder_path, verbose=False)

# Now that we have defined the two interfaces we pass them to the ConverterPipe which will coordinate the
# concurrent conversion of the data
converter = ConverterPipe(data_interfaces=[interface_tiff, interface_suit2p], verbose=False)

# Extract what metadata we can from the source files
metadata = converter.get_metadata()
# For data provenance we add the time zone information to the conversion
session_start_time = datetime(2020, 1, 1, 12, 30, 0, tzinfo=tz.gettz("US/Pacific"))
metadata["NWBFile"].update(session_start_time=session_start_time)

# Choose a path for saving the nwb file and run the conversion
nwbfile_path = f"{path_to_save_nwbfile}"
converter.run_conversion(nwbfile_path=nwbfile_path,  metadata=metadata)
