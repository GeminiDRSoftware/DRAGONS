import inspect
import traceback
from datetime import datetime

import astropy
from astropy.table import Table

import geminidr
from astrodata import AstroDataFits
import numpy as np

from gempy.utils import logutils
from recipe_system.utils.md5 import md5sum

log = logutils.get_logger(__name__)


# def top_level_primitive(called_from_decorator=False):
#     """ Check if we are a 'top-level' primitive.
#
#     Returns true if this primitive was called directly or from a recipe, rather
#     than being invoked by another primitive.
#     """
#     # if we were called from a decorator, we flag that we already "saw" our own
#     # method since it's not in the stack
#     if called_from_decorator:
#         saw_primitive = True
#     else:
#         saw_primitive = False
#     for trace in inspect.stack():
#         if "self" in trace[0].f_locals:
#             inst = trace[0].f_locals["self"]
#             if isinstance(inst, geminidr.PrimitivesBASE):
#                 if saw_primitive:
#                     return False
#                 saw_primitive = True
#     # if we encounter no primitives above this decorator, then this is a top level primitive call
#     return True


# def get_provenance_history(ad):
#     retval = list()
#     try:
#         if isinstance(ad, AstroDataFits):
#             if 'GEM_PROVENANCE_HISTORY' in ad:
#                 provenance_history = ad.GEM_PROVENANCE_HISTORY
#                 for row in provenance_history:
#                     timestamp_start = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S.%f")
#                     timestamp_end = datetime.strptime(row[1], "%Y-%m-%d %H:%M:%S.%f")
#                     provenance_added_by = row[2]
#                     args = row[3]
#                     retval.append({"timestamp_start": timestamp_start, "timestamp_end": timestamp_end,
#                                    "provenance_added_by": provenance_added_by, "args": args})
#     except Exception as e:
#         pass
#     return retval
#
#
# def add_provenance_history(ad, timestamp_start, timestamp_end, primitive, args):
#     if isinstance(ad, AstroDataFits):
#         timestamp_start_str = ""
#         if timestamp_start is not None:
#             timestamp_start_str = timestamp_start.strftime("%Y-%m-%d %H:%M:%S.%f")
#         timestamp_end_str = ""
#         if timestamp_end is not None:
#             timestamp_end_str = timestamp_end.strftime("%Y-%m-%d %H:%M:%S.%f")
#         if 'GEM_PROVENANCE_HISTORY' not in ad:
#             timestamp_start_data = np.array([timestamp_start_str])
#             timestamp_end_data = np.array([timestamp_end_str])
#             primitive_data = np.array([primitive])
#             args_data = np.array([args])
#
#             my_astropy_table = Table([timestamp_start_data, timestamp_end_data, primitive_data, args_data],
#                                      names=('timestamp_start', 'timestamp_end', 'primitive', 'args'),
#                                      dtype=('S28', 'S28', 'S128', 'S128'))
#             # astrodata.add_header_to_table(my_astropy_table)
#             ad.append(my_astropy_table, name='GEM_PROVENANCE_HISTORY', header=astropy.io.fits.Header())
#         else:
#             history = ad.GEM_PROVENANCE_HISTORY
#             history.add_row((timestamp_start_str, timestamp_end_str, primitive, args))
#     else:
#         log.warn("Not a FITS AstroData, add provenance history does nothing")
