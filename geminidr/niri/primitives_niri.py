import numpy as np

from gempy.gemini import gemini_tools as gt

from ..core import NearIR
from ..gemini.primitives_gemini import Gemini
from .parameters_niri import ParametersNIRI

from recipe_system.utils.decorators import parameter_override
# ------------------------------------------------------------------------------
@parameter_override
class NIRI(Gemini, NearIR):
    """
    This is the class containing all of the preprocessing primitives
    for the F2 level of the type hierarchy tree. It inherits all
    the primitives from the level above
    """
    tagset = set(["GEMINI", "NIRI"])

    def __init__(self, adinputs, **kwargs):
        super(NIRI, self).__init__(adinputs, **kwargs)
        self.parameters = ParametersNIRI

    def nonlinearityCorrect(self, adinputs=None, **params):
        """
        Run on raw or nprepared Gemini NIRI data, this script calculates and
        applies a per-pixel linearity correction based on the counts in the
        pixel, the exposure time, the read mode, the bias level and the ROI.
        Pixels over the maximum correctable value are set to BADVAL unless
        given the force flag. Note that you may use glob expansion in infile,
        however, any pattern matching characters (*,?) must be either quoted
        or escaped with a backslash. Do we need a badval parameter that defines
        a value to assign to uncorrectable pixels, or do we want to just add
        those pixels to the DQ plane with a specific value?

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
                            "already been processed by nonlinearityCorrect".
                            format(ad.filename))
                continue

            total_exptime = ad.exposure_time()
            coadds = ad.coadds()
            # Check the raw exposure time (i.e., per coadd). First, convert
            # the total exposure time returned by the descriptor back to
            # the raw exposure time
            exptime = total_exptime / coadds
            if exptime > 600.:
                log.warning("Exposure time {} for {} is outside the range "
                             "used to derive correction.",format(exptime,
                                                                 ad.filename))

            for ext, coeffs in zip(ad, ad.nonlinearity_coeffs()):
                raw_pixel_data = ext.data / coadds
                
                raw_mean_value = np.mean(raw_pixel_data, dtype=np.float64)
                log.fullinfo("The mean value of the raw pixel data in " \
                             "{} is {:.8f}".format(ext.filename, raw_mean_value))
                
                log.fullinfo("Coefficients used = {:.12f} {:.9e} {:.9e}".
                            format(coeffs.time_delta, coeffs.gamma, coeffs.eta))
                
                # Create a new array that contains the corrected pixel data
                corrected_pixel_data = raw_pixel_data * (1 + raw_pixel_data *
                        (coeffs.gamma + coeffs.eta * raw_pixel_data)) * coadds
                
                # Correct VAR plane; additive correction means this works
                # even if read noise has been added. Data must be in ADU.
                if ext.variance is not None:
                    ext.variance += (corrected_pixel_data -
                                raw_pixel_data*coadds) / ext.gain()
                    
                # Write the corrected pixel data to the output object
                ext.data = corrected_pixel_data

                # Correct for the exposure time issue by scaling the counts
                # to the nominal exposure time
                ext.multiply(exptime / (exptime + coeffs.time_delta))
                
                # Determine the mean of the corrected pixel data
                corrected_mean_value = np.mean(ext.data, dtype=np.float64)
                log.fullinfo("The mean value of the corrected pixel data in "
                        "{} is {:.8f}".format(ext.filename, corrected_mean_value))
            
            # Correct the exposure time by adding coeff1 * coadds
            total_exptime = total_exptime + coeffs.time_delta * coadds
            log.fullinfo("The true total exposure time = {}".format(total_exptime))
            
            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.filename = gt.filename_updater(adinput=ad, suffix=params["suffix"],
                                              strip=True)
        return adinputs
    
    def standardizeInstrumentHeaders(self, adinputs=None, **params):
        """
        This primitive is used to make the changes and additions to the
        keywords in the headers of NIRI data, specifically.

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
            # keywords in the headers that are specific to FLAMINGOS-2.
            log.status("Updating keywords that are specific to NIRI")

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
                self.hdr.set(kw, 1, self.keyword_comments(kw))

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.filename = gt.filename_updater(adinput=ad, suffix=params["suffix"],
                                              strip=True)
        return adinputs