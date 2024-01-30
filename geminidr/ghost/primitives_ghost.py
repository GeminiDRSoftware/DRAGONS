#
#                                                                  gemini_python
#
#                                                            primitives_ghost.py
# ------------------------------------------------------------------------------
from geminidr.gemini.primitives_gemini import Gemini
from geminidr.core.primitives_ccd import CCD
from .primitives_calibdb_ghost import CalibDBGHOST

from . import parameters_ghost

from .lookups import keyword_comments, timestamp_keywords as ghost_stamps

from recipe_system.utils.decorators import parameter_override

import re
import astrodata
from gempy.gemini import gemini_tools as gt
import numpy as np
# ------------------------------------------------------------------------------
_HDR_SIZE_REGEX = re.compile(r'^\[(?P<x1>[0-9]*)\:'
                             r'(?P<x2>[0-9]*),'
                             r'(?P<y1>[0-9]*)\:'
                             r'(?P<y2>[0-9]*)\]$')


def filename_updater(ad, **kwargs):
    origname = ad.filename
    ad.update_filename(**kwargs)
    rv = ad.filename
    ad.filename = origname
    return rv


@parameter_override
class GHOST(Gemini, CCD, CalibDBGHOST):
    """
    Top-level primitives for handling GHOST data

    The primitives in this class are applicable to all flavours of GHOST data.
    All other GHOST primitive classes inherit from this class.
    """
    tagset = set()  # Cannot be assigned as a class

    def _initialize(self, adinputs, **kwargs):
        self.inst_lookups = 'geminidr.ghost.lookups'
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_ghost)
        # Add GHOST-specific timestamp keywords
        self.timestamp_keys.update(ghost_stamps.timestamp_keys)
        self.keyword_comments.update(keyword_comments.keyword_comments)

    @staticmethod
    def _has_valid_extensions(ad):
        return len(ad) > 0
