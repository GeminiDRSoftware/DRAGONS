#
#                                                                  gemini_python
#
#                                                      primitives_gsaoi_image.py
# ------------------------------------------------------------------------------
import numpy as np
from astropy.table import vstack

from geminidr.core import NearIR
from geminidr.gemini.primitives_gemini import Gemini
from gempy.gemini import gemini_tools as gt
from recipe_system.utils.decorators import parameter_override, capture_provenance
from gemini_instruments.gsaoi import lookup as adlookup

from . import parameters_gsaoi


@parameter_override
@capture_provenance
class GSAOI(Gemini, NearIR):
    """
    This is the class containing all of the preprocessing primitives
    for the F2 level of the type hierarchy tree. It inherits all
    the primitives from the level above
    """
    tagset = {"GEMINI", "GSAOI"}

    def _initialize(self, adinputs, **kwargs):
        self.inst_lookups = 'geminidr.gsaoi.lookups'
        self.inst_adlookup = adlookup
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_gsaoi)

    def standardizeInstrumentHeaders(self, adinputs=None, suffix=None):
        """
        This primitive is used to make the changes and additions to the
        keywords in the headers of GSAOI data, specifically.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        for ad in adinputs:
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by "
                            "standardizeInstrumentHeaders".format(ad.filename))
                continue

            # Standardize the headers of the input AstroData object. Update the
            # keywords in the headers that are specific to GSAOI.
            log.status("Updating keywords that are specific to GSAOI")

            # Filter name (required for IRAF?)
            ad.phu.set('FILTER', ad.filter_name(stripID=True, pretty=True),
                       self.keyword_comments['FILTER'])

            # Pixel scale
            ad.phu.set('PIXSCALE', ad.pixel_scale(),
                       self.keyword_comments['PIXSCALE'])

            for desc in ('read_noise', 'gain', 'non_linear_level',
                         'saturation_level'):
                kw = ad._keyword_for(desc)
                for ext, value in zip(ad, getattr(ad, desc)()):
                    ext.hdr.set(kw, value, self.keyword_comments[kw])

            # Move BUNIT from PHU to the extension HDUs
            try:
                bunit = ad.phu['BUNIT']
            except KeyError:
                pass
            else:
                del ad.phu['BUNIT']
                ad.hdr.set('BUNIT', bunit, self.keyword_comments['BUNIT'])

            # There is a bug in GSAOI data where the central 1K x 1K ROI is
            # read out (usually for photometric standards) and the CCDSEC
            # keyword needs to be fixed for this type of data so the auxiliary
            # data (i.e. BPM and flats) match. Best guess date for start of
            # problem is currently May 15 2013. It is not clear if this bug is
            # in the detector controller code or the SDSU board timing.
            if ad.phu.get('DATE-OBS') >= '2013-05-13':
                for i, ext in enumerate(ad):
                    if ext.array_section(pretty=True) == '[513:1536,513:1536]':
                        if i == 0:
                            log.stdinfo("Updating the CCDSEC for central ROI data")

                        for sec_name in ('array_section', 'detector_section'):
                            kw = ad._keyword_for(sec_name)
                            sec = getattr(ext, sec_name)()

                            y1o = (sec.y1 + 1) if i < 2 else (sec.y1 - 1)
                            y2o = y1o + 1024

                            secstr = "[{}:{},{}:{}]".format(
                                sec.x1 + 1, sec.x2, y1o + 1, y2o)

                            ext.hdr.set(kw, secstr, self.keyword_comments[kw])

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)

        return adinputs

    def makeBPM(self, adinputs=None, **params):
        """
        To be run from recipe makeProcessedBPM.

        The main input is a flat field image that has been constructed by
        stacking the differences of lamp on / off exposures in a given filter
        and normalizing the resulting image to unit average.

        A 'darks' stream must also be provided, containing a single image
        constructed by stacking short darks.


        Parameters
        ----------
        override_thresh: bool
            Override GSAOI default dark threshold calculation with a
            user-specified `dark_hi_thresh`? With the default of False, hot
            pixels are considered to be those with a level greater than 75%
            of (the mean + 3 sigma); note that this calculated level varies
            significantly between quadrants.
        dark_hi_thresh: float, optional
            Maximum data value above which pixels in the input dark are to be
            considered bad, if `override_thresh` is set. This is always an
            absolute limit in ADUs. If None, no limit (+Inf) is applied and
            the dark does not contribute to the bad pixel mask.
        flat_lo_thresh: float, optional
            Minimum (unit-normalized) data value below which pixels in the
            input flat are considered to be bad (default 0.5). If None, no
            limit (-Inf) is applied and the flat does not contribute to the
            bad pixel mask.

        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        # This GSAOI version approximately follows a procedure documented on
        # the GSAOI Web pages (GSAOI_BPM_forweb.pdf). That document is
        # inconsistent on whether a mean or median is used for the dark, but
        # it seems to make very little difference to the result (and the actual
        # example commands shown appear to use a mean).

        override = params['override_thresh']
        dark_hi = params['dark_hi_thresh']
        flat_lo = params['flat_lo_thresh']

        if dark_hi is None:
            dark_hi = float('Inf')
        if flat_lo is None:
            flat_lo = float('-Inf')

        # Get the stacked flat and dark; these are single-element lists
        try:
            flat = adinputs[0]
        except IndexError:
            raise ValueError("A SET OF FLATS IS REQUIRED INPUT")
        try:
            dark = self.streams['darks'][0]
        except (KeyError, TypeError, IndexError):
            raise ValueError("A SET OF DARKS IS REQUIRED INPUT")

        for dark_ext, flat_ext in zip(dark, flat):

            msg = "BPM flat mask lower limit: {}"
            log.stdinfo(msg.format(flat_lo))

            flat_mask = flat_ext.data < flat_lo  # (already normalized)

            msg = "BPM dark mask upper limit: {:.2f} ADU ({:.2f})"

            bunit = dark_ext.hdr.get('BUNIT', 'ADU').upper()
            if bunit in ('ELECTRON', 'ELECTRONS'):
                conv = dark_ext.gain()
            elif bunit == 'ADU' or override is False:
                conv = 1.
            else:
                raise ValueError("Input units for dark should be ADU or "
                                 "ELECTRON, not {}".format(bunit))

            if override:
                # Convert a user-specified threshold from ADUs:
                dark_lim = conv * dark_hi
            else:
                # Use the "standard" calculation for GSAOI:
                dark_lim = 0.75 * (np.mean(dark_ext.data) \
                                   + 3 * np.std(dark_ext.data))

            log.stdinfo(msg.format(dark_lim / conv, dark_lim))

            dark_mask = dark_ext.data > dark_lim

            # combine masks and write to bpm file
            data_mask = np.ma.mask_or(dark_mask, flat_mask, shrink=False)
            flat_ext.reset(data_mask.astype(np.int16), mask=None, variance=None)

        flat.update_filename(suffix="_bpm", strip=True)
        flat.phu.set('OBJECT', 'BPM')
        gt.mark_history(flat, primname=self.myself(), keyword=timestamp_key)
        return [flat]

    def _nonlinearity_coeffs(self, ad):
        """
        For each extension, return a tuple (a0,a1,a2) of coefficients such
        that the linearized counts are a0 + a1*c _ a2*c^2 for raw counts c

        Returns
        -------
        tuple/list
            coefficients
        """
        return ad._look_up_arr_property('coeffs')

    @staticmethod
    def _has_valid_extensions(ad):
        """Check the AD has a valid number of extensions"""
        return len(ad) == 4

