#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2010-2018 PyTroll Community

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>
#   Adam Bybbroe <adam.dybbroe@smhi.se>
#   Sauli Joro <sauli.joro@eumetsat.int>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""HDF5 format reader.

References:
    MSG Level 1.5 Image Data FormatDescription

TODO:
- HRV navigation

"""

import logging
from datetime import datetime, timedelta

import numpy as np
import h5py

from pyresample import geometry
from satpy.readers.hdf5_utils import HDF5FileHandler

from satpy.readers.seviri_base import SEVIRICalibrationHandler
from satpy.readers.seviri_base import (CHANNEL_NAMES, CALIB, SATNUM)
from satpy.readers.eum_base import timecds2datetime

logger = logging.getLogger("hdf5_msg")


def make_time_cds_expanded(tcds_array):
    return (datetime(1958, 1, 1) +
            timedelta(days=int(tcds_array["days"]),
                      milliseconds=int(tcds_array["milliseconds"]),
                      microseconds=float(tcds_array["microseconds"] +
                                         tcds_array["nanoseconds"] / 1000.)))


def subdict(keys, value):
    """
    Takes a list and a value and if the list contains more than one key
    subdictionarys for each key will be created.

    Parameters
    ----------
    keys : list of str
        List of one or more strings
    value : any
        Value to assign to (sub)dictionary key

    Returns
    -------
    dict
        Dict or dict of dicts

    """
    tdict = {}
    key = keys[0].strip()
    if len(keys) == 1:
        tdict[key] = value
    else:
        keys.remove(key)
        tdict[key] = subdict(keys, value)
    return tdict

import collections
from collections import defaultdict


def dict_merge(dct, merge_dct):
    """
    Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    
    Parameters
    ----------
    dct : dict
        dict onto which the merge is executed
    merge_dct : dct
        dct merged into dct
        
    Returns
    -------
    None
    """
    for k, v in merge_dct.items():# merge_dct.iteritems():
        if (k in dct and isinstance(dct[k], dict) and isinstance(merge_dct[k], collections.Mapping)):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]


def rec2dict(arr):
    """
    Converts an array of attributes to a dictionary.

    Parameters
    ----------
    arr : ndarray
        DESCR array from hdf5 MSG data file

    Returns
    -------
    dict

    """
    res = {}
    for dtuple in arr:
        fullkey = dtuple[0].decode(encoding="utf-8").split("-")
        key = fullkey[0]
        data = dtuple[1]
        ndict = subdict(fullkey, data)
        dict_merge(res, ndict)
    return res


class HDF5SEVIRIFileHandler(HDF5FileHandler):

    """SEVIRI HDF5 format reader
    """

    def __init__(self, filename, filename_info, filetype_info, ext_calib_coefs=None):
        """Initialize the reader."""
        super(HDF5SEVIRIFileHandler, self).__init__(filename, filename_info, filetype_info)

        self._filename_info = filename_info
        self.ext_calib_coefs = ext_calib_coefs or {}

        self._get_header()

    def get_metadata(self, ds_id, ds_info):
        """
        Get the metadata for specific dataset listed in yaml config


        :param ds_id:
        :param ds_info:
        :return:
        """
        pass


    def _get_header(self):
        """Read the header info, and fill the metadata dictionary"""

        self.mda = defaultdict(dict)

        for k, d in self.file_content.items():
            if isinstance(d, h5py.Dataset) and k.endswith("DESCR") or k.endswith("SUBSET"):
                group = k.split("/")[-2]
                key = k.split("/")[-1]
                dset = np.array(self[k])
                self.mda[group][key] = rec2dict(dset)

        self.calib_coeffs = np.array(self["U-MARF/MSG/Level1.5/METADATA/HEADER/RadiometricProcessing/Level15ImageCalibration_ARRAY"])

        earth_model = self.mda["GeometricProcessing"]["GeometricProcessing_DESCR"]["EarthModel"]
        self.mda["offset_corrected"] = int(earth_model["TypeOfEarthModel"]) == 1
        b = (float(earth_model["NorthPolarRadius"]) + float(earth_model["SouthPolarRadius"])) / 2.0 * 1000
        self.mda["projection_parameters"]["a"] = float(earth_model["EquatorialRadius"]) * 1000
        self.mda["projection_parameters"]["b"] = b
        self.mda["projection_parameters"]["h"] = 35785831.0
        ssp = self.mda["ImageDescription"]["ImageDescription_DESCR"]["ProjectionDescription"]["LongitudeOfSSP"]
        self.mda["projection_parameters"]["SSP_longitude"] = float(ssp)
        self.mda["projection_parameters"]["SSP_latitude"] = 0.0
        self.platform_id = int(self.mda["ImageProductionStats"]["ImageProductionStats_DESCR"]["SatelliteId"])
        self.platform_name = "Meteosat-" + SATNUM[self.platform_id]
        self.mda["platform_name"] = self.platform_name
        #service = self._filename_info["service"]
        #if service == "":
        self.mda["service"] = "0DEG"
        #else:
         #   self.mda["service"] = service
        #self.channel_name = CHANNEL_NAMES[self.mda["spectral_channel_id"]]

    @property
    def start_time(self):
        time = self.mda["ImageProductionStats"]["ImageProductionStats_DESCR"]["ActualScanningSummary"]["ForwardScanStart"]

        return timecds2datetime({k.capitalize(): v for k,v in time.items()})

    @property
    def end_time(self):
        time = self.mda["ImageProductionStats"]["ImageProductionStats_DESCR"]["ActualScanningSummary"]["ForwardScanEnd"]

        return timecds2datetime({k.capitalize(): v for k,v in time.items()})

    #from nc reader
    def get_area_extent(self, dsid):

        # following calculations assume grid origin is south-east corner
        # section 7.2.4 of MSG Level 1.5 Image Data Format Description
        # origins = {0: 'NW', 1: 'SW', 2: 'SE', 3: 'NE'}
        # grid_origin = self.nc.attrs['vis_ir_grid_origin']
        # grid_origin = int(grid_origin, 16)
        # if grid_origin != 2:
        #     raise NotImplementedError(
        #         'Grid origin not supported number: {}, {} corner'
        #         .format(grid_origin, origins[grid_origin])
        #     )

        ds_type = "VIS_IR"

        if dsid["name"] == "HRV":
            ds_type = "HRV"

        refGrid = self.mda["ImageDescription"]["ImageDescription_DESCR"]["ReferenceGrid" + ds_type]

        center_point = 3712/2

        column_step = float(refGrid["ColumnDirGridStep"]) * 1000

        line_step = float(refGrid["LineDirGridStep"]) * 1000

        # check for Earth model as this affects the north-south and
        # west-east offsets
        # section 3.1.4.2 of MSG Level 1.5 Image Data Format Description
        earth_model = int(self.mda["GeometricProcessing"]["GeometricProcessing_DESCR"]["EarthModel"]["TypeOfEarthModel"]) #int(self.nc.attrs['type_of_earth_model'], 16)
        if earth_model == 2:
            ns_offset = 0  # north +ve
            we_offset = 0  # west +ve
        elif earth_model == 1:
            ns_offset = -0.5  # north +ve
            we_offset = 0.5  # west +ve
        else:
            raise NotImplementedError(
                'unrecognised earth model: {}'.format(earth_model)
            )

        #from nc file reader
        #self.north = int(self.nc.attrs['north_most_line'])
        #self.east = int(self.nc.attrs['east_most_pixel'])
        #self.west = int(self.nc.attrs['west_most_pixel'])
        #self.south = int(self.nc.attrs['south_most_line'])


        # hdf5 files attributes for cropped data
        # if "METADATA" in self.mda.keys():
        #     subset = self.mda["METADATA"]["SUBSET"]
        #     ll_x = int(subset[ds_type + "WestColumnSelectedRectangle"])
        #     ll_y = int(subset[ds_type + "SouthLineSelectedRectangle"])
        #     ur_x = int(subset[ds_type + "EastColumnSelectedRectangle"])
        #     ur_y = int(subset[ds_type + "NorthLineSelectedRectangle"])
        #     bounds = (ncols - ll_x, nlines - ll_y, ncols - ur_x, nlines - ur_y)
        #     ncols = ll_x - ur_x + 1
        #     nlines = ur_y - ll_y + 1

        #hardwired to fulldisk
        west = 3712
        south = 1
        east = 1
        north = 3712

        # section 3.1.5 of MSG Level 1.5 Image Data Format Description
        ll_c = (center_point - west - 0.5 + we_offset) * column_step
        ll_l = (south - center_point - 0.5 + ns_offset) * line_step
        ur_c = (center_point - east + 0.5 + we_offset) * column_step
        ur_l = (north - center_point + 0.5 + ns_offset) * line_step
        area_extent = (ll_c, ll_l, ur_c, ur_l)

        return area_extent

    def get_area_def(self, dsid):
        """Get the area definition of the band."""
        ds_type = "VIS_IR"

        if dsid["name"] == "HRV":
            ds_type = "HRV"

        refGrid = self.mda["ImageDescription"]["ImageDescription_DESCR"]["ReferenceGrid" + ds_type]

        nlines = int(refGrid["NumberOfLines"])
        ncols = int(refGrid["NumberOfColumns"])


        area_extent = self.get_area_extent(dsid)

        b = self.mda["projection_parameters"]["b"]
        a = self.mda["projection_parameters"]["a"]
        h = self.mda["projection_parameters"]["h"]
        lon_0 = self.mda["projection_parameters"]["SSP_longitude"]

        proj_dict = {"a": float(a),
                     "b": float(b),
                     "lon_0": float(lon_0),
                     "h": float(h),
                     "proj": "geos",
                     "units": "m"}

        area = geometry.AreaDefinition(
            "some_area_name",
            "On-the-fly area",
            "geosmsg",
            proj_dict,
            ncols,
            nlines,
            area_extent)

        self.area = area
        return area


    def get_dataset(self, dataset_id, ds_info):
        ds_path = ds_info.get("file_key", "{}".format(dataset_id))
        channel_id = int(self.mda[ds_path]["LineSideInfo_DESCR"]["ChannelId"])
        res = self["U-MARF/MSG/Level1.5/DATA/" + ds_path + "/IMAGE_DATA"][::-1,:]
        res = self.calibrate(res, dataset_id, channel_id) #key.calibration)
        #res.attrs.update(ds_info) #as alternative to the following three lines but updates to much info
        res.attrs["units"] = ds_info["units"]
        res.attrs["wavelength"] = ds_info["wavelength"]
        res.attrs["standard_name"] = ds_info["standard_name"]
        res.attrs["platform_name"] = self.platform_name
        res.attrs["sensor"] = "seviri"
        res.attrs["orbital_parameters"] = {
            "projection_longitude": self.mda["projection_parameters"]["SSP_longitude"],
            "projection_latitude": self.mda["projection_parameters"]["SSP_latitude"],
            "projection_altitude": self.mda["projection_parameters"]["h"]}

        return res


    def calibrate(self, dataset, dataset_id, channel_id):
        """Calibrate the data."""
        calibration = dataset_id["calibration"]

        if calibration == "counts":
            res = dataset
        elif calibration in ["radiance", "reflectance", "brightness_temperature"]:
            dataset = dataset.where(dataset != 0).astype(np.float32)

        channel = dataset_id["name"]

        calib = SEVIRICalibrationHandler(
            platform_id=int(self.platform_id),
            channel_name=channel,
            coefs=self._get_calib_coefs(dataset, channel_id),
            calib_mode="NOMINAL",
            scan_time=self.start_time
        )

        return calib.calibrate(dataset, calibration)


    def _get_calib_coefs(self, dataset, channel_id):
        """Get coefficients for calibration from counts to radiance.
        MPEF MSG calibration coefficients (gain and offset).
        """
        channel = CHANNEL_NAMES[channel_id]
        offset = self.calib_coeffs["Cal_Offset"][channel_id - 1]
        gain = self.calib_coeffs["Cal_Slope"][channel_id - 1]

        cal_type_list = list(int(x) for x in self.mda["ImageDescription"]["ImageDescription_DESCR"]["Level 1_5 ImageProduction"]["PlannedChanProcessing"].decode(encoding="utf-8").split(","))
        cal_type = cal_type_list[channel_id - 1]
        # Only one calibration (GSICS) available
        return {
            "coefs": {
                "GSICS": {
                    "gain": gain,
                    "offset": offset
                },
                "NOMINAL": {
                    "gain": gain,
                    "offset": offset
                },
                "EXTERNAL": self.ext_calib_coefs.get(channel, {})
            },
            "radiance_type": cal_type
        }


def show(data, negate=False):
    """Show the stretched data.
    """
    from PIL import Image as pil
    data = np.array((data - data.min()) * 255.0 /
                    (data.max() - data.min()), np.uint8)
    if negate:
        data = 255 - data
    img = pil.fromarray(data)
    img.show()
