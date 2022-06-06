#
#                                                                  gemini_python
#
#                                                          primtives_f2_spect.py
# ------------------------------------------------------------------------------
import numpy as np

from importlib import import_module
import os

from geminidr.core import Spect
from .primitives_f2 import F2
from . import parameters_f2_spect

#
#                                                                  gemini_python
#
#                                                        primtives_gnirs_spect.py
# ------------------------------------------------------------------------------
import os

import numpy as np
from copy import copy
from importlib import import_module
from datetime import datetime
from functools import reduce
from copy import deepcopy

from functools import partial

from gempy.library import astrotools as at

from astropy.table import Table

from astropy.modeling import models, Model
from astropy import units as u
from scipy.interpolate import UnivariateSpline

import geminidr.interactive.server
from geminidr.core import Spect
from gempy.library.fitting import fit_1D
from .primitives_f2 import F2
from . import parameters_f2_spect

from geminidr.gemini.lookups import DQ_definitions as DQ
from geminidr.gmos.lookups import geometry_conf as geotable

from gempy.gemini import gemini_tools as gt
from gempy.library import astromodels as am
from gempy.library import transform, wavecal

from recipe_system.utils.decorators import parameter_override, capture_provenance
from ..interactive.fit.wavecal import WavelengthSolutionVisualizer
from ..interactive.interactive import UIParameters
from ..core import NearIR

from recipe_system.utils.decorators import parameter_override, capture_provenance


# ------------------------------------------------------------------------------
@parameter_override
@capture_provenance
class F2Spect(F2, Spect):
    """
    This is the class containing all of the preprocessing primitives
    for the F2Spect level of the type hierarchy tree. It inherits all
    the primitives from the level above
    """
    tagset = {"GEMINI", "F2", "SPECT"}

    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_f2_spect)

    def standardizeWCS(self, adinputs=None, suffix=None):
        """
        This primitive updates the WCS attribute of each NDAstroData extension
        in the input AstroData objects. For spectroscopic data, it means
        replacing an imaging WCS with an approximate spectroscopic WCS.

        This is an F2-specific primitive due to the need to apply an offset to the
        central wavelength derived from image header, which for F2 is specified for the middle of
        the grism+filter transmission window, not for the central row.

        Parameters
        ----------
        suffix: str/None
            suffix to be added to output files

        """

        log = self.log
        timestamp_key = self.timestamp_keys[self.myself()]
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        for ad in adinputs:
            log.stdinfo(f"Adding spectroscopic WCS to {ad.filename}")
            # Apply central wavelength offset
            cenwave = ad.central_wavelength(asNanometers=True) + abs(ad.dispersion(asNanometers=True)[0]) * ad.cenwave_offset()
            transform.add_longslit_wcs(ad, central_wavelength=cenwave)

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)
        return adinputs

    def _get_arc_linelist(self, waves=None, ad=None):
        lookup_dir = os.path.dirname(import_module('.__init__',
                                                   self.inst_lookups).__file__)

        if 'ARC' in ad.tags:
            linelist = 'lowresargon.dat'
        else:
            linelist = 'sky.dat'

        filename = os.path.join(lookup_dir, linelist)
        return wavecal.LineList(filename)




