#
#                                                                  gemini_python
#
#                                                            primitives_gnirs.py
# ------------------------------------------------------------------------------
from os import path
from gempy.gemini import gemini_tools as gt

from ..core import NearIR
from ..gemini.primitives_gemini import Gemini
from . import parameters_gnirs

from recipe_system.utils.decorators import parameter_override
# ------------------------------------------------------------------------------
@parameter_override
class GNIRS(Gemini, NearIR):
    """
    This is the class containing all of the preprocessing primitives
    for the F2 level of the type hierarchy tree. It inherits all
    the primitives from the level above
    """
    tagset = set(["GEMINI", "GNIRS"])

    def __init__(self, adinputs, **kwargs):
        super(GNIRS, self).__init__(adinputs, **kwargs)
        self.inst_lookups = 'geminidr.gnirs.lookups'
        self._param_update(parameters_gnirs)

    def standardizeInstrumentHeaders(self, adinputs=None, suffix=None):
        """
        This primitive is used to make the changes and additions to the
        keywords in the headers of GNIRS data, specifically.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        """
        # Instantiate the log
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
            # keywords in the headers that are specific to GNIRS.
            log.status("Updating keywords that are specific to GNIRS")

            # Filter name (required for IRAF?)
            ad.phu.set('FILTER', ad.filter_name(stripID=True, pretty=True),
                       self.keyword_comments['FILTER'])

            # Pixel scale (CJS: I'm putting this in the extension too!)
            pixel_scale = ad.pixel_scale()
            ad.phu.set('PIXSCALE', pixel_scale, self.keyword_comments['PIXSCALE'])
            ad.hdr.set('PIXSCALE', pixel_scale, self.keyword_comments['PIXSCALE'])

            for desc in ('read_noise', 'gain', 'non_linear_level',
                         'saturation_level'):
                kw = ad._keyword_for(desc)
                ad.hdr.set(kw, getattr(ad, desc)()[0], self.keyword_comments[kw])
                try:
                    ad.phu.remove(kw)
                except (KeyError, AttributeError):
                    pass

            # The exposure time keyword in raw data the exptime of each coadd
            # but the data have been summed, not averaged, so it needs to be
            # reset to COADDS*EXPTIME. The descriptor always returns that value,
            # regardless of whether the data are prepared or unprepared.
            kw = ad._keyword_for('exposure_time')
            ad.phu.set(kw, ad.exposure_time(), self.keyword_comments[kw])

            if 'SPECT' in ad.tags:
                kw = ad._keyword_for('dispersion_axis')
                self.hdr.set(kw, 2, self.keyword_comments(kw))

            # Adding the WCS information to the pixel data header, since for
            # GNIRS images at least, it may be only in the PHU
            for kw in ('CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 'CD1_1',
                      'CD2_1', 'CD1_2', 'CD2_2', 'MJD-OBS', 'CTYPE1', 'CTYPE2'):
                value = ad.phu.get(kw)
                if value is not None:
                    if ad[0].hdr.get(kw) is None:
                        ad.hdr.set(kw, value, self.keyword_comments.get(kw))

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)
        return adinputs
