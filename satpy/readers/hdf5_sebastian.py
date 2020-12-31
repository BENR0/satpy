#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2010-2018 PyTroll Community

# Author(s):

#   Benjamin Roesner <benjamin.roesner@geo.uni-marburg.de>

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
    for k, v in merge_dct.iteritems():
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
        fullkey = dtuple[0].split("-")
        key = fullkey[0]
        data = dtuple[1]
        ndict = subdict(fullkey, data)
        dict_merge(res, ndict)
    return res


class HDF5MSGFileHandler(HDF5FileHandler, SEVIRICalibrationHandler):

    """MSG HDF5 format reader
    """

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize the reader."""
        super(HDF5MSGFileHandler, self).__init__(filename, filename_info, filetype_info)

        self._filename_info = filename_info

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

        self.mda["projection_parameters"]["a"] = 6378169.0
        self.mda["projection_parameters"]["b"] = 6356583.8
        self.mda["projection_parameters"]["h"] = 35785831.0
        self.mda["projection_parameters"]["SSP_longitude"] = 0.0
        self.mda["projection_parameters"]["SSP_latitude"] = 0.0
        self.platform_id = 324
        self.platform_name = "Meteosat-" + SATNUM[self.platform_id]
        self.mda["platform_name"] = self.platform_name
        self.mda["service"] = "0DEG"


    @property
    def start_time(self):
        return self._filename_info["slot_time"]


    @property
    def end_time(self):
        return self._filename_info["slot_time"]


    def get_xy_from_linecol(self, line, col, offsets, factors):
        """Get the intermediate coordinates from line & col.

        Intermediate coordinates are actually the instruments scanning angles.
        """
        loff, coff = offsets
        lfac, cfac = factors
        x__ = (col - coff) / cfac * 2**16
        y__ = - (line - loff) / lfac * 2**16

        return x__, y__


    def from_msg_space_coordinate(self, x, y, gridsteps):
        COLUMN_DIR_GRID_STEP, LINE_DIR_GRID_STEP = gridsteps
        return x * LINE_DIR_GRID_STEP, y * COLUMN_DIR_GRID_STEP


    def from_top_left_of_north_west_pixel_zero_based(self, msg_x, msg_y, offsets, gridsteps):
        """
        Calculate coordinates based on pixel count and gridstep.

        Parameters
        ----------
        msg_x : int
            Pixel count in x direction with origin top left
        msg_y : int
            Pixel count in y direction with origin top left
        offsets : tuple of int
            (column offset, line offset)
        gridsteps : tuple of int
            (column gridstep, line gridstep)

        Returns
        -------
        tuple of float
            Coordinates in geostationary projection (x,y)

        """
        COFF, LOFF = offsets
        msg_x_coord = (msg_x - COFF) - 0.5
        msg_y_coord = (LOFF - msg_y) + 0.5

        return self.from_msg_space_coordinate(msg_x_coord, msg_y_coord, gridsteps)


    def get_area_extent(self, bounds, offsets, gridsteps):
        """Get the area extent of the file."""

        ll_x, ll_y = self.from_top_left_of_north_west_pixel_zero_based(bounds[0], bounds[1], offsets, gridsteps)

        ur_x, ur_y = self.from_top_left_of_north_west_pixel_zero_based(bounds[2], bounds[3], offsets, gridsteps)

        return ll_x, ll_y, ur_x, ur_y


    def get_area_def(self, dsid):
        """Get the area definition of the band."""
        ds_type = "VIS_IR"

        nlines = 3712
        ncols = 3712
        linegridstep = 3.00040316582 * 1000
        colgridstep = 3.00040316582 * 1000
        gridsteps = (colgridstep, linegridstep)

        loff = nlines/2
        coff = ncols/2
        offsets = (coff, loff)

        ll_x = 1489
        ll_y = 764
        ur_x = 2456
        ur_y = 54
        bounds = (ll_x, ll_y, ur_x, ur_y)
        ncols = ur_x -ll_x
        nlines = ll_y - ur_y


        area_extent = self.get_area_extent(bounds, offsets, gridsteps)

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
        res = self[ds_path]
        res.attrs["units"] = ds_info["units"]
        res.attrs["wavelength"] = ds_info["wavelength"]
        res.attrs["standard_name"] = ds_info["standard_name"]
        res.attrs["platform_name"] = self.platform_name
        res.attrs["sensor"] = "seviri"
        res.attrs["modifiers"] = ("sunz_corrected",)
        res.attrs["orbital_parameters"] = {
            "projection_longitude": self.mda["projection_parameters"]["SSP_longitude"],
            "projection_latitude": self.mda["projection_parameters"]["SSP_latitude"],
            "projection_altitude": self.mda["projection_parameters"]["h"]}

        return res


    def calibrate(self, data, calibration, channel_id):
        """Calibrate the data."""

        return NotImplementedError
