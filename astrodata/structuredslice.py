#
#                                                        gemini_python/astrodata
#                                                   astrodata.structuredslice.py
#                                                                        08-2013
# ------------------------------------------------------------------------------
# $Id$
# ------------------------------------------------------------------------------
__version__      = '$Revision$'[11:-2]
__version_date__ = '$Date$'[7:-2]
# ------------------------------------------------------------------------------
from Structures import centralStructureIndex, retrieve_structure_obj

csi = centralStructureIndex

for sname in csi:
    struct = retrieve_structure_obj(sname)
    exec("%s = struct" % (sname))
