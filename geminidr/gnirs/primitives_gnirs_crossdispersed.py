#
#                                                                 gemini_python
#
#                                            primitives_gnirs_crossdispersed.py
# -----------------------------------------------------------------------------

from astropy.table import Table, hstack

from gempy.gemini import gemini_tools as gt
from recipe_system.utils.decorators import (parameter_override,
                                            capture_provenance)
from geminidr.gemini.lookups import DQ_definitions as DQ

from .primitives_gnirs_spect import GNIRSSpect
from . import parameters_gnirs_crossdispersed
from geminidr.core.primitives_crossdispersed import CrossDispersed
from .lookups.MDF_XD_GNIRS import slit_info

# -----------------------------------------------------------------------------
@parameter_override
@capture_provenance
class GNIRSCrossDispersed(GNIRSSpect, CrossDispersed):
    """This class contains all of the preprocessing primitives for the
    GNIRSCrossDispersed level of the type hierarchy tree. It inherits all the
    primitives from the above level.
    """
    tagset = {"GEMINI", "GNIRS", "SPECT", "XD"}

    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_gnirs_crossdispersed)

    def _map_spec_order(self, ext_id):
        """Return the spectral order corresponding to the given extension ID

        This provides a mapping between extension ID and spectral order; for
        GNIRS cross-dispersed this is a simple linear relation. The orders that
        are traced are 3-8.
        """
        return ext_id + 2

    # TODO: handle _fields_overlap()


    def addMDF(self, adinputs=None, suffix=None, mdf=None):
        """
        This GNIRS XD-specific implementation of addMDF() calls
        primitives_gemini._addMDF() on each astrodata object to attach the MDFs.
        It also attaches two columns, 'slitlength_arcsec' and 'slitlength_pixels',
        with the length of the slit in arcseconds and pixels, respectively.

        Any parameters given will be passed to primitives_gemini._addMDF().

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        mdf: str/None
            name of MDF to add (None => use default)
        """

        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        mdf_list = mdf or self.caldb.get_calibrations(adinputs,
                                                      caltype="mask").files

        columns = ('slit_id', 'x_ccd', 'y_ccd', 'specorder',
                   'slitlength_asec', 'slitlength_pixels')

        for ad, mdf in zip(*gt.make_lists(adinputs, mdf_list, force_ad=True)):

            # GNIRS XD doesn't use mask definition files, so this won't add
            # anything, but it will check if the file already has an MDF table.
            self._addMDF(ad, suffix, mdf)

            if hasattr(ad, 'MDF'):
                log.fullinfo(f"{ad.filename} already has an MDF table.")
                continue
            else:
                mdf_key_parts = ('telescope', '_prism', 'decker',
                                 '_grating', 'camera')
                mdf_key = "_".join(getattr(ad, desc)()
                                   for desc in mdf_key_parts)
                ad.MDF = Table(slit_info[mdf_key],
                               names=columns)
                log.stdinfo(f"Added MDF table for {ad.filename}")

        return adinputs


    def tracePinholeApertures(self, adinputs=None, **params):
        """
        This primitive exists to provide some mode-specific values to the
        tracePinholeApertures() in primitives_spect, since the modes in GNIRS
        cross-dispersed are different enough to warrant having different values.
        """

        camera = getattr(adinputs[0], 'camera')()

        if 'Short' in camera:
            # In the short camera configuration there are four good pinholes
            # and one that's right on the edge of the slit and isn't consistently
            # picked up. This setting stops it from being used in the orders it
            # is found in since it produces a sketchy fit.
            if params['debug_max_trace_pos'] is None:
                params['debug_max_trace_pos'] = 4
                self.log.debug("Setting debug_max_trace_pos to 4 for Short "
                               "camera.")

        elif 'Long' in camera:
            # In the long camera configuration the 5th and 6th slits run off the
            # side of the array, necessitating a start point much closer to the
            # bottom instead of the default middle-of-the-array.
            if params['start_pos'] is None:
                params['start_pos'] = 150
                self.log.fullinfo("Setting trace start location to row 150 for "
                                  "Long camera.")

        # Call the parent primitive with the new parameter values.
        return super().tracePinholeApertures(adinputs=adinputs, **params)


    def _get_order_information_key(self):
        """
        This function provides a key to the order-specific information needed
        for updating the WCS when cutting out slits in XD data.

        Returns
        -------
        tuple
            A tuple of strings representing the attributes of the dict key for
            information on the orders

        """

        return ('telescope', '_prism', 'decker', '_grating', 'camera')
