#
#                                                                  gemini_python
#
#                                                        primtives_gnirs_spect.py
# ------------------------------------------------------------------------------
import os


from importlib import import_module

from geminidr.core import Spect

from .primitives_gnirs import GNIRS
from . import parameters_gnirs_spect

from gempy.gemini import gemini_tools as gt
from gempy.library import transform, wavecal

from recipe_system.utils.decorators import parameter_override, capture_provenance


@parameter_override
@capture_provenance
class GNIRSSpect(Spect, GNIRS):
    """
    This is the class containing all of the preprocessing primitives
    for the GNIRSSpect level of the type hierarchy tree. It inherits all
    the primitives from the level above
    """
    tagset = {"GEMINI", "GNIRS", "SPECT"}

    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_gnirs_spect)

    def standardizeWCS(self, adinputs=None, **params):
        """
        This primitive updates the WCS attribute of each NDAstroData extension
        in the input AstroData objects. For spectroscopic data, it means
        replacing an imaging WCS with an approximate spectroscopic WCS.

        Parameters
        ----------
        suffix: str/None
            suffix to be added to output files

        """
        log = self.log
        timestamp_key = self.timestamp_keys[self.myself()]
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        super().standardizeWCS(adinputs, **params)

        for ad in adinputs:
            log.stdinfo(f"Adding spectroscopic WCS to {ad.filename}")
            cenwave = ad.central_wavelength(asNanometers=True)
            transform.add_longslit_wcs(ad, central_wavelength=cenwave,
                                       pointing=ad[0].wcs(512, 511))

            # Timestamp. Suffix was updated in the super() call
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
        return adinputs

    def _get_arc_linelist(self, waves=None, ad=None):
        lookup_dir = os.path.dirname(import_module('.__init__',
                                                   self.inst_lookups).__file__)

        is_lowres = ad.disperser(pretty=True).startswith('10') or \
                    (ad.disperser(pretty=True).startswith('32') and
                        ad.camera(pretty=True).startswith('Short'))

        if 'ARC' in ad.tags:
            if 'Xe' in ad.object():
                linelist ='Ar_Xe.dat'
            elif "Ar" in ad.object():
                if is_lowres:
                    linelist = 'lowresargon.dat'
                else:
                    linelist = 'argon.dat'
            else:
                raise ValueError(f"No default line list found for {ad.object()}-type arc. Please provide a line list.")

        else:
            if ad.filter_name(pretty=True).startswith('M'):
                resolution = self._get_resolution(ad)
                if resolution >= 5000:
                    linelist = 'sky_M_band_high_res.dat'
                elif (2000 <= resolution < 5000):
                    linelist = 'sky_M_band_med_res.dat'
                elif (500 <= resolution < 2000):
                    linelist = 'sky_M_band_low_res.dat'
                elif resolution < 500:
                    linelist = 'sky_M_band_very_low_res.dat'
            elif ad.filter_name(pretty=True).startswith('L'):
                resolution = self._get_resolution(ad)
                if resolution >=10000:
                    linelist = 'sky_L_band_high_res.dat'
                elif (3000 <= resolution < 10000):
                    linelist = 'sky_L_band_med_res.dat'
                elif (1000 <= resolution < 3000):
                    linelist = 'sky_L_band_low_res.dat'
                elif resolution < 1000:
                    linelist = 'sky_L_band_very_low_res.dat'

            else:
                linelist = 'nearIRsky.dat'

        self.log.debug(f"Using linelist '{linelist}'")
        filename = os.path.join(lookup_dir, linelist)

        return wavecal.LineList(filename)


    def _get_resolution(self, ad=None):
        resolution_2pix_slit = {('M, 10/mm, 0.05'): 1200,
                                ('M, 32/mm, 0.15'): 1240,
                                ('M, 32/mm, 0.05'): 3700,
                                ('M, 111/mm, 0.15'): 4300,
                                ('M, 111/mm, 0.05'): 12800,
                                ('L, 10/mm, 0.05'): 1800,
                                ('L, 32/mm, 0.15'): 1800,
                                ('L, 32/mm, 0.05'): 5400,
                                ('L, 111/mm, 0.15'): 6400,
                                ('L, 111/mm, 0.05'): 19000}

        filter = str(ad.filter_name(pretty=True))[0]
        grating = ad._grating(pretty=True, stripID=True)
        pix_scale = ad.pixel_scale()
        config = f"{filter}, {grating}, {pix_scale}"

        resolution_2pix = resolution_2pix_slit.get(config)
        slit_width_pix = ad.slit_width()/pix_scale

        return resolution_2pix * 2 / slit_width_pix

    def _get_cenwave_accuracy(self, ad=None):
        # Accuracy of central wavelength (nm) for a given instrument/setup.
        # According to GNIRS instrument pages "wavelength settings are accurate
        # to better than 5 percent of the wavelength coverage".
        # However using 7% covers more cases. For the arcs dc0=10 works just fine for all modes.

        mband = ad.filter_name(pretty=True).startswith('M')
        lband = ad.filter_name(pretty=True).startswith('L')

        if 'ARC' in ad.tags or not (mband or lband):
            dcenwave = 10
        else:
            dcenwave = abs(ad.dispersion(asNanometers=True)) * 1024 * 0.07
        return dcenwave
