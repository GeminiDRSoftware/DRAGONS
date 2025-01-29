#
#                                                                  gemini_python
#
#                                                             primitives_niri.py
# ------------------------------------------------------------------------------
import numpy as np
from os import path

from gempy.gemini import gemini_tools as gt
from gemini_instruments.niri import lookup as adlookup

from ..core import NearIR
from ..gemini.primitives_gemini import Gemini
from . import parameters_niri

from recipe_system.utils.decorators import parameter_override, capture_provenance


# ------------------------------------------------------------------------------
@parameter_override
@capture_provenance
class NIRI(Gemini, NearIR):
    """
    This is the class containing all of the preprocessing primitives
    for the F2 level of the type hierarchy tree. It inherits all
    the primitives from the level above
    """
    tagset = {"GEMINI", "NIRI"}

    def _initialize(self, adinputs, **kwargs):
        self.inst_lookups = 'geminidr.niri.lookups'
        self.inst_adlookup = adlookup
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_niri)

    def nonlinearityCorrect(self, adinputs=None, suffix=None):
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

        def linearize(counts, coeffs):
            """Return a linearized version of the counts in electrons per coadd"""
            return counts * (1 + counts * (coeffs.gamma + counts * coeffs.eta))

        for ad in adinputs:
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by nonlinearityCorrect".
                            format(ad.filename))
                continue

            # This assumes all extensions have the same dtype
            extension_data_type = ad[0].data.dtype.type
            total_exptime = extension_data_type(ad.exposure_time())
            coadds = extension_data_type(ad.coadds())

            # Check the raw exposure time (i.e., per coadd). First, convert
            # the total exposure time returned by the descriptor back to
            # the raw exposure time
            exptime = total_exptime / coadds
            if exptime > 600:
                log.warning(f"Exposure time {exptime} for {ad.filename} is "
                            "outside the range used to derive correction.")

            for ext, gain, coeffs in zip(ad, ad.gain(), self._nonlinearity_coeffs(ad)):
                if coeffs is None:
                    log.warning("No nonlinearity coefficients found for "
                                f"{ad.filename} extension {ext.id} - "
                                "no correction applied")
                    continue

                gain = extension_data_type(gain)

                raw_mean_value = np.mean(ext.data) / coadds
                log.fullinfo("The mean value of the raw pixel data in " \
                             "{} is {:.8f}".format(ext.filename, raw_mean_value))

                log.fullinfo("Coefficients used = {:.12f} {:.9e} {:.9e}".
                            format(coeffs.time_delta, coeffs.gamma, coeffs.eta))

                # Create a new array that contains the corrected pixel data
                raw_pixel_data = ext.data * gain / coadds
                corrected_pixel_data = linearize(raw_pixel_data, coeffs) * coadds / gain

                # Try to do something useful with the VAR plane, if it exists
                # Since the data are fairly pristine, VAR will simply be the
                # Poisson noise (divided by gain if in ADU), possibly plus RN**2
                # So making an additive correction will sort this out,
                # irrespective of whether there's read noise
                if ext.variance is not None:
                    ext.variance += (corrected_pixel_data - ext.data) / gain
                # Now update the SCI extension
                ext.data = corrected_pixel_data

                # Correct for the exposure time issue by scaling the counts
                # to the nominal exposure time
                time_delta = extension_data_type(coeffs.time_delta)
                ext.multiply(exptime / (exptime + time_delta))

                # Determine the mean of the corrected pixel data
                corrected_mean_value = np.mean(ext.data) / coadds
                log.fullinfo("The mean value of the corrected pixel data in "
                        "{} is {:.8f}".format(ext.filename, corrected_mean_value))

                # Correct the exposure time by adding coeff1 * coadds
                total_exptime = total_exptime + time_delta * coadds

                # Update descriptors for saturation and nonlinear thresholds
                log.fullinfo(f"The true total exposure time = {total_exptime}")
                for desc in ('saturation_level', 'non_linear_level'):
                    current_value = getattr(ext, desc)()
                    current_value = extension_data_type(current_value)
                    new_value = linearize(
                        current_value * gain / coadds, coeffs) * coadds / gain
                    ext.hdr[ad._keyword_for(desc)] = np.round(new_value, 3)

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)
        return adinputs

    def standardizeInstrumentHeaders(self, adinputs=None, suffix=None):
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
                ad.hdr.set(kw, 1, self.keyword_comments[kw])

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)
        return adinputs

    def _nonlinearity_coeffs(self, ad):
        """
        Returns a list of namedtuples containing the necessary information to
        perform a nonlinearity correction. The list contains one namedtuple
        per extension (although normal NIRI data only has a single extension).

        Returns
        -------
        list of namedtuples
            nonlinearity info (max counts, exptime correction, gamma, eta)
        """
        read_mode = ad.read_mode()
        well_depth = ad.well_depth_setting()
        naxis2 = ad.hdr.get('NAXIS2')
        return [self.inst_adlookup.nonlin_coeffs.get((read_mode, size, well_depth))
                    for size in naxis2]

