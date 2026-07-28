import numpy as np

from gempy.gemini import gemini_tools as gt

from ..gemini.primitives_gemini import Gemini
from . import parameters_irdc

from recipe_system.utils.decorators import parameter_override, capture_provenance


@parameter_override
@capture_provenance
class IRDC(Gemini):
    """
    This is the class containing all of the preprocessing primitives
    for instruments that use the Gemini Infrared Detector Controller (IRDC).
    """
    tagset = {"GEMINI"}

    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_irdc)

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
            # The coefficients are in the form of a {region: coeffs} dictionary.
            corrected_counts = np.empty_like(counts)
            for _slice, coeff in coeffs.items():
                log.debug("Coefficients for {} rows = {:.6f} "
                             "{:.9e} {:.9e}".format(
                    _slice, coeff.time_delta, coeff.gamma, coeff.eta))
                corrected_counts[_slice] = (
                        counts[_slice] * (1 + counts[_slice] *
                                          (np.float32(coeff.gamma) +
                                           counts[_slice] * np.float32(coeff.eta))))
            return corrected_counts

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
            exptime = np.float32(total_exptime /
                                 (coadds if ad.is_coadds_summed() else 1))
            if exptime > 600:
                log.warning(f"Exposure time {exptime} for {ad.filename} is "
                            "outside the range used to derive correction.")

            for ext, gain, coeffs in zip(ad, ad.gain(), self._nonlinearity_coeffs(ad)):
                if coeffs is None:
                    log.warning("No nonlinearity coefficients found for "
                                f"{ad.filename} extension {ext.id} - "
                                "no correction applied")
                    continue
                elif not isinstance(coeffs, dict):
                    # coeffs apply to entire array
                    coeffs = {None: coeffs}

                raw_mean_value = np.mean(ext.data) / coadds
                log.fullinfo("The mean value of the raw pixel data in " \
                             "{} is {:.8f}".format(ext.filename, raw_mean_value))

                # Create a new array that contains the corrected pixel data.
                # Remember that gain() returns 1.0 after ADUToElectrons
                raw_pixel_data = ext.data * np.float32(gain / coadds)
                corrected_pixel_data = (linearize(raw_pixel_data, coeffs) *
                                        np.float32(coadds / gain))

                # Try to do something useful with the VAR plane, if it exists
                # Since the data are fairly pristine, VAR will simply be the
                # Poisson noise (divided by gain if in ADU), possibly plus RN**2
                # So making an additive correction will sort this out,
                # irrespective of whether there's read noise
                if ext.variance is not None:
                    ext.variance += (corrected_pixel_data - ext.data) / np.float32(gain)
                # Now update the SCI extension
                ext.data = corrected_pixel_data

                # Correct for the exposure time issue by scaling the counts
                # to the nominal exposure time
                time_delta = np.float32(np.mean([v.time_delta for v in coeffs.values()]))
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
                    new_value = linearize(
                        np.array([current_value * gain / coadds]),
                        coeffs)[0] * coadds / gain
                    ext.hdr[ad._keyword_for(desc)] = np.round(new_value, 3)

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)
        return adinputs
