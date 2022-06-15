#
#                                                                 gemini_python
#
#                                                     primitives_f2_longslit.py
# -----------------------------------------------------------------------------

from astropy.table import Table
import numpy as np

from gempy.gemini import gemini_tools as gt
from recipe_system.utils.decorators import (parameter_override,
                                            capture_provenance)

from . import parameters_f2_longslit
from .primitives_f2_spect import F2Spect

# -----------------------------------------------------------------------------

# This dictionary contains the following information for each slit configuration:
# x-coordinate of the pixel in the center of the slit, slit length in arcsec,
# slit length in pixels. The F2 pixel scale is 0.18 arsec/pixel, according to
# the instrument webpage.
f2_slit_info = {'1pix': (763.5, 271.62, 1509),
                '2pix': (769.5, 265.14, 1473),
                '3pix': (770.0, 271.80, 1510),
                '4pix': (772.5, 270.54, 1503),
                '6pix': (777.0, 271.80, 1510),
                '8pix': (771.0, 271.80, 1510)}

@parameter_override
@capture_provenance
class F2Longslit(F2Spect):
    """This class contains all of the processing primitives for the F2Longslit
    level of the type hiearchy tree. It inherits all the primitives from the
    above level.
    """
    tagset = {'GEMINI', 'F2', 'SPECT', 'LS'}
    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_f2_longslit)

    def addMDF(self, adinputs=None, suffix=None, mdf=None):
        """
        This F2-specific implementation of addMDF() adds a "virtual MDF" (as
        in, created from data in this module rather than pulled from another
        file) to F2 data. It calls primitives_gemini._addMDF() on each astrodata
        object to attach the MDFs, then performs corrections depending on the
        data. It also attaches two columns, 'slitsize_arcsec' and 'slitsize_pixels'
        with the length of the slit in arcseconds and pixels, respectively.

        Parameters
        ----------
        suffix : str
            suffix to be added to output files
        mdf : str/None
            name of MDF to add (None => use default)

        """

        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        mdf_list = mdf or self.caldb.get_calibrations(adinputs,
                                                      caltype="mask").files

        for ad, mdf in zip(*gt.make_lists(adinputs, mdf_list, force_ad=True)):

            # F2 doesn't have actual mask defintiion files, so this won't add
            # anyything, but it will check if the file already has an MDF table
            self._addMDF(ad, suffix, mdf)

            try:
                mdf = ad.MDF
                continue
            except AttributeError:
                log.stdinfo(f"Creating MDF table for F2 file {ad.filename}.")

                maskname = ad.phu['MASKNAME'].split('-')[0]

                mdf_table = Table(np.array(f2_slit_info[maskname]),
                                  names=('x_ccd', 'slitlength_arcsec',
                                         'slitlength_pixels'))
                ad.MDF = mdf_table
