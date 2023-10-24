#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023 Satpy developers
#
# This file is part of satpy.
#
# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.
"""Reader for Himawari L2 cloud products from NOAA's big data programme.

For more information about the data, see: <https://registry.opendata.aws/noaa-himawari/>.

These products are generated by the NOAA enterprise cloud suite and have filenames like:
AHI-CMSK_v1r1_h09_s202308240540213_e202308240549407_c202308240557548.nc

The second letter grouping (CMSK above) indicates the product type:
    CMSK - Cloud mask

    CHGT - Cloud height

    CPHS - Cloud type and phase

These products are generated from the AHI sensor on Himawari-8 and Himawari-9, and are
produced at the native instrument resolution for the IR channels (2km at nadir).

NOTE: This reader is currently only compatible with full disk scenes. Unlike level 1 himawari
data, the netCDF files do not contain the required metadata to produce an appropriate area
definition for the data contents, and hence the area definition is hardcoded into the reader.

A warning is displayed to the user highlighting this. The assumed area definition is a full
disk image at the nominal subsatellite longitude of 140.7 degrees East.

All the simple data products are supported here, but multidimensional products are not yet
supported. These include the CldHgtFlag and the CloudMaskPacked variables.
"""

import logging
from datetime import datetime

import xarray as xr

from satpy._compat import cached_property
from satpy.readers._geos_area import get_area_definition, get_area_extent
from satpy.readers.file_handlers import BaseFileHandler

logger = logging.getLogger(__name__)

EXPECTED_DATA_AREA = 'Full Disk'


class HIML2NCFileHandler(BaseFileHandler):
    """File handler for Himawari L2 NOAA enterprise data in netCDF format."""

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize the reader."""
        super().__init__(filename, filename_info, filetype_info)
        self.nc = xr.open_dataset(self.filename,
                                  decode_cf=True,
                                  mask_and_scale=False,
                                  chunks={"xc": "auto", "yc": "auto"})

        # Check that file is a full disk scene, we don't know the area for anything else
        if self.nc.attrs['cdm_data_type'] != EXPECTED_DATA_AREA:
            raise ValueError('File is not a full disk scene')

        self.sensor = self.nc.attrs['instrument_name'].lower()
        self.nlines = self.nc.dims['Columns']
        self.ncols = self.nc.dims['Rows']
        self.platform_name = self.nc.attrs['satellite_name']
        self.platform_shortname = filename_info['platform']
        self._meta = None

    @property
    def start_time(self):
        """Start timestamp of the dataset."""
        dt = self.nc.attrs['time_coverage_start']
        return datetime.strptime(dt, '%Y-%m-%dT%H:%M:%SZ')

    @property
    def end_time(self):
        """End timestamp of the dataset."""
        dt = self.nc.attrs['time_coverage_end']
        return datetime.strptime(dt, '%Y-%m-%dT%H:%M:%SZ')

    def get_dataset(self, key, info):
        """Load a dataset."""
        var = info['file_key']
        logger.debug('Reading in get_dataset %s.', var)
        variable = self.nc[var]

        # Data has 'Latitude' and 'Longitude' coords, these must be replaced.
        variable = variable.rename({'Rows': 'y', 'Columns': 'x'})

        variable = variable.drop('Latitude')
        variable = variable.drop('Longitude')

        variable.attrs.update(key.to_dict())
        return variable

    @cached_property
    def area(self):
        """Get AreaDefinition representing this file's data."""
        return self._get_area_def()

    def get_area_def(self, dsid):
        """Get the area definition."""
        del dsid
        return self.area

    def _get_area_def(self):
        logger.info('The AHI L2 cloud products do not have the metadata required to produce an area definition.'
                    ' Assuming standard Himawari-8/9 full disk projection.')

        # Basic check to ensure we're processing a full disk (2km) scene.n
        if self.nlines != 5500 or self.ncols != 5500:
            raise ValueError("Input L2 file is not a full disk Himawari scene. Only full disk data is supported.")

        pdict = {'cfac': 20466275, 'lfac': 20466275, 'coff': 2750.5, 'loff': 2750.5, 'a': 6378137.0, 'h': 35785863.0,
                 'b': 6356752.3, 'ssp_lon': 140.7, 'nlines': self.nlines, 'ncols': self.ncols, 'scandir': 'N2S'}

        aex = get_area_extent(pdict)

        pdict['a_name'] = 'Himawari_Area'
        pdict['a_desc'] = "AHI Full Disk area"
        pdict['p_id'] = f'geos{self.platform_shortname}'

        return get_area_definition(pdict, aex)