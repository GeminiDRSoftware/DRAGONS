import traceback
from datetime import datetime

import astropy
from astropy.table import Table

from astrodata import AstroDataFits
import numpy as np

from gempy.utils import logutils
from recipe_system.utils.md5 import md5sum

log = logutils.get_logger(__name__)


def get_provenance(ad):
    retval = list()
    try:
        if isinstance(ad, AstroDataFits):
            if 'GEM_PROVENANCE' in ad:
                provenance = ad.GEM_PROVENANCE
                pass
                for row in provenance:
                    timestamp = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S.%f")
                    filename = row[1]
                    md5 = row[2]
                    primitive = row[3]
                    retval.append({"timestamp": timestamp, "filename": filename, "md5": md5, "primitive": primitive})
    except Exception as e:
        pass
    return retval


def get_provenance_history(ad):
    retval = list()
    try:
        if isinstance(ad, AstroDataFits):
            if 'GEM_PROVENANCE_HISTORY' in ad:
                provenance_history = ad.GEM_PROVENANCE_HISTORY
                for row in provenance_history:
                    timestamp_start = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S.%f")
                    timestamp_end = datetime.strptime(row[1], "%Y-%m-%d %H:%M:%S.%f")
                    primitive = row[2]
                    args = row[3]
                    retval.append({"timestamp_start": timestamp_start, "timestamp_end": timestamp_end,
                                   "primitive": primitive, "args": args})
    except Exception as e:
        pass
    return retval


def add_provenance(ad, timestamp, filename, md5, primitive):
    try:
        if isinstance(ad, AstroDataFits):
            if timestamp is None:
                timestamp_str = ""
            else:
                timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")
            if md5 is None:
                md5 = ""

            if 'GEM_PROVENANCE' not in ad:
                timestamp_data = np.array([timestamp_str])
                filename_data = np.array([filename])
                md5_data = np.array([md5])
                primitive_data = np.array([primitive])
                my_astropy_table = Table([timestamp_data, filename_data, md5_data, primitive_data],
                                         names=('timestamp', 'filename', 'md5', 'primitive'),
                                         dtype=('S28', 'S128', 'S128', 'S128'))
                ad.append(my_astropy_table, name='GEM_PROVENANCE')
                pass
            else:
                provenance = ad.GEM_PROVENANCE
                provenance.add_row((timestamp_str, filename, md5, primitive))
                pass
        else:
            log.warn("Not a FITS AstroData, add provenance does nothing")
    except Exception as e:
        pass

def add_provenance_history(ad, timestamp_start, timestamp_end, primitive, args):
    if isinstance(ad, AstroDataFits):
        timestamp_start_str = ""
        if timestamp_start is not None:
            timestamp_start_str = timestamp_start.strftime("%Y-%m-%d %H:%M:%S.%f")
        timestamp_end_str = ""
        if timestamp_end is not None:
            timestamp_end_str = timestamp_end.strftime("%Y-%m-%d %H:%M:%S.%f")
        if 'GEM_PROVENANCE_HISTORY' not in ad:
            timestamp_start_data = np.array([timestamp_start_str])
            timestamp_end_data = np.array([timestamp_end_str])
            primitive_data = np.array([primitive])
            args_data = np.array([args])

            my_astropy_table = Table([timestamp_start_data, timestamp_end_data, primitive_data, args_data],
                                     names=('timestamp_start', 'timestamp_end', 'primitive', 'args'),
                                     dtype=('S28', 'S28', 'S128', 'S128'))
            # astrodata.add_header_to_table(my_astropy_table)
            ad.append(my_astropy_table, name='GEM_PROVENANCE_HISTORY', header=astropy.io.fits.Header())
        else:
            history = ad.GEM_PROVENANCE_HISTORY
            history.add_row((timestamp_start_str, timestamp_end_str, primitive, args))
    else:
        log.warn("Not a FITS AstroData, add provenance history does nothing")
