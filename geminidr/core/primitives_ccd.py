#
#                                                                  gemini_python
#
#                                                              primitives_ccd.py
# ------------------------------------------------------------------------------
from contextlib import suppress

import numpy as np

from astrodata.provenance import add_provenance
from gempy.gemini import gemini_tools as gt
from gempy.library import astrotools as at
from gempy.library.fitting import fit_1D

from geminidr import PrimitivesBASE, CalibrationNotFoundError
from recipe_system.utils.md5 import md5sum
from . import parameters_ccd

from recipe_system.utils.decorators import parameter_override, capture_provenance


# ------------------------------------------------------------------------------
@parameter_override
@capture_provenance
class CCD(PrimitivesBASE):
    """
    This is the class containing all of the primitives used for generic CCD
    reduction.
    """
    tagset = None

    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_ccd)

    def biasCorrect(self, adinputs=None, suffix=None, bias=None, do_cal=None):
        """
        The biasCorrect primitive will subtract the science extension of the
        input bias frames from the science extension of the input science
        frames. The variance and data quality extension will be updated, if
        they exist. If no bias is provided, the calibration database(s) will
        be queried.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        bias: str/list of str
            bias(es) to subtract
        do_cal: str
            perform bias subtraction?
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        if do_cal == 'skip':
            log.warning("Bias correction has been turned off.")
            return adinputs

        if bias is None:
            bias_list = self.caldb.get_processed_bias(adinputs)
        else:
            bias_list = (bias, None)

        # Provide a bias AD object for every science frame, and an origin
        for ad, bias, origin in zip(*gt.make_lists(adinputs, *bias_list,
                                    force_ad=(1,))):
            if ad.phu.get(timestamp_key):
                log.warning(f"{ad.filename}: already processed by "
                            "biasCorrect. Continuing.")
                continue

            if bias is None:
                if 'sq' in self.mode or do_cal == 'force':
                    raise CalibrationNotFoundError("No processed bias listed "
                                                   f"for {ad.filename}")
                else:
                    log.warning(f"No changes will be made to {ad.filename}, "
                                "since no bias was specified")
                    continue

            try:
                gt.check_inputs_match(ad, bias, check_filter=False,
                                      check_units=True)
            except ValueError:
                bias = gt.clip_auxiliary_data(ad, aux=bias, aux_type='cal')
                # An Error will be raised if they don't match now
                gt.check_inputs_match(ad, bias, check_filter=False,
                                      check_units=True)

            origin_str = f" (obtained from {origin})" if origin else ""
            log.stdinfo(f"{ad.filename}: subtracting the bias "
                         f"{bias.filename}{origin_str}")
            ad.subtract(bias)

            # If there's no OVERSCAN keyword on the bias extension, then it
            # hasn't been overscan-subtracted and we are subtracting a value
            # from the data, which needs to be recorded
            for ext, ext_bias in zip(ad, bias):
                if 'OVERSCAN' not in ext_bias.hdr:
                    bias_level = np.median(ext_bias.data)
                    ext.hdr.set('OVERSCAN', bias_level,
                                self.keyword_comments['OVERSCAN'])
                    for desc in ('saturation_level', 'non_linear_level'):
                        with suppress(AttributeError, KeyError):
                            ext.hdr[ad._keyword_for(desc)] -= bias_level

            # Record bias used, timestamp, and update filename
            ad.phu.set('BIASIM', bias.filename, self.keyword_comments['BIASIM'])
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)
            if bias.path:
                add_provenance(ad, bias.filename, md5sum(bias.path) or "", self.myself())

        return adinputs

    def overscanCorrect(self, adinputs=None, **params):
        adinputs = self.subtractOverscan(adinputs,
                    **self._inherit_params(params, "subtractOverscan"))
        adinputs = self.trimOverscan(adinputs, suffix=params["suffix"])
        return adinputs

    def subtractOverscan(self, adinputs=None, **params):
        """
        This primitive subtracts the overscan level from the image. The
        level for each row (currently the primitive requires that the overscan
        region be a vertical strip) is determined in one of the following
        ways, according to the *function* and *order* parameters:

        :"poly":   a polynomial of degree *order* (1=linear, etc)
        :"spline": using *order* equally-sized cubic spline pieces or, if
                  order=None or 0, a spline that provides a reduced chi^2=1
        :"none":   no function is fit, and the value for each row is determined
                  by the overscan pixels in that row

        The fitting is done iteratively but, in the first instance, a running
        median of the rows is calculated and rows that deviate from this median
        are rejected (and used in place of the actual value if function="none")

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        niterate: int
            number of rejection iterations
        high_reject: float/None
            number of standard deviations above which to reject high pixels
        low_reject: float/None
            number of standard deviations above which to reject low pixels
        nbiascontam: int/None
            number of columns adjacent to the illuminated region to reject
        function: str/None
            function to fit ("chebyshev" | "spline" | "none")
        order: int/None
            order of polynomial fit or number of spline pieces
        bias_type: str
            For multiple overscan regions, selects which one to use
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        sfx = params["suffix"]
        fit1d_params = fit_1D.translate_params(params)
        # We need some of these parameters for pre-processing

        function = (fit1d_params.pop("function") or "none").lower()
        lsigma = params["lsigma"]
        hsigma = params["hsigma"]
        order = params["order"]
        nbiascontam = params["nbiascontam"]
        bias_type = params.get("bias_type")

        for ad in adinputs:
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by subtractOverscan".
                            format(ad.filename))
                continue

            if bias_type:
                osec_list = ad.overscan_section()[bias_type]
            else:
                osec_list = ad.overscan_section()
            for ext, ext_osec in zip(ad, osec_list):
                ext.data = ext.data.astype(np.float32)
                mapped_asec = gt.map_array_sections(ext)
                ext_rdnoise = ext.read_noise()
                ext_gain = ext.gain()

                # GMOS for example returns a single Section.  To simplify
                # the rest of the script, make that a list (to be use in the
                # for-loop below).
                if not isinstance(ext_osec, list):
                    ext_osec = [ext_osec]
                    mapped_asec = [mapped_asec]

                # if there are several amps, the readnoise and gain are likely
                # to be different for each.  If there's only one value, just
                # use it for all amps, if not ensure that the number matches
                # the number of overscan section and arrays.
                if not isinstance(ext_rdnoise, list):
                    ext_rdnoise = [ext_rdnoise] * len(ext_osec)
                elif len(ext_rdnoise) != len(ext_osec):
                    raise ValueError('Readnoise descriptor does not match overscan.')

                if not isinstance(ext_gain, list):
                    ext_gain = [ext_gain] * len(ext_osec)
                elif len(ext_gain) != len(ext_osec):
                    raise ValueError('Gain descriptor does not match overscan.')


                for osec, asec, rdnoise, gain in zip(ext_osec, mapped_asec, ext_rdnoise, ext_gain):
                    x1, x2, y1, y2 = osec.x1, osec.x2, osec.y1, osec.y2
                    axis = np.argmin([y2 - y1, x2 - x1])
                    if axis == 1:
                        if x1 > asec.x1:  # Bias on right
                            x1 += nbiascontam
                            x2 -= 1
                        else:  # Bias on left
                            x1 += 1
                            x2 -= nbiascontam
                        sigma = rdnoise / np.sqrt(x2 - x1)
                        # need to match asec location and size
                        pixels = np.arange(asec.y1, asec.y2)
                        data = np.mean(ext.data[asec.y1:asec.y2, x1:x2], axis=axis)
                        stds = np.std(ext.data[asec.y1:asec.y2, x1:x2], axis=axis)
                    else:
                        if y1 > asec.y1:  # Bias on top
                            y1 += nbiascontam
                            y2 -= 1
                        else:  # Bias on bottom
                            y1 += 1
                            y2 -= nbiascontam

                        sigma = rdnoise / np.sqrt(y2 - y1)
                        # needs to match asec location and size
                        pixels = np.arange(asec.x1, asec.x2)
                        data = np.mean(ext.data[y1:y2, asec.x1:asec.x2], axis=axis)
                        stds = np.std(ext.data[y1:y2, asec.x1:asec.x2], axis=axis)

                    # We have 1 sample (of x2-x1 values) from each of N
                    # populations each of which has a different population mean,
                    # (ie the bias leve for that row) but we assume they all
                    # have the same standard deviation (ie the read noise) and
                    # we want to estimate that population standard deviation.
                    # We've calculated the mean (data) and standard deviation
                    # (stds) of each of the samples, and want to estimate the
                    # population standard deviation. Because the means are
                    # different, we can't just calculate the std of all the
                    # samples.
                    overstd = np.sqrt(np.mean(stds * stds))

                    # Readnoise descriptor always returns value in electrons
                    if ext.is_in_adu():
                        sigma /= gain

                    # The UnivariateSpline will make reduced-chi^2=1 so it will
                    # fit bad rows. Need to mask these before starting, so use a
                    # running median. Probably a good starting point for all fits.
                    runmed = at.boxcar(data, operation=np.median, size=2)
                    residuals = data - runmed
                    mask = np.logical_or(residuals > hsigma * sigma
                                         if hsigma is not None else False,
                                         residuals < -lsigma * sigma
                                         if lsigma is not None else False)
                    if "spline" in function and order is None:
                        data = np.where(mask, runmed, data)

                    if function == "none":
                        bias = data

                    else:
                        fit1d = fit_1D(np.ma.masked_array(data, mask=mask),
                                       points=pixels,
                                       weights=np.full_like(data, 1. / sigma),
                                       function=function, **fit1d_params)
                        bias = fit1d.evaluate(np.arange(data.size))
                        sigma = fit1d.rms


                    # using "-=" won't change from int to float
                    if axis == 1:
                        ext.data[asec.y1:asec.y2, asec.x1:asec.x2] = \
                            ext.data[asec.y1:asec.y2, asec.x1:asec.x2] - \
                            bias[:, np.newaxis].astype(np.float32)
                        # KL: useful when debugging
                        # ext.data[asec.y1:asec.y2, x1:x2] = \
                        #     ext.data[asec.y1:asec.y2, x1:x2] - \
                        #     bias[:, np.newaxis].astype(np.float32)
                    else:
                        ext.data[asec.y1:asec.y2, asec.x1:asec.x2] = \
                            ext.data[asec.y1:asec.y2, asec.x1:asec.x2] - \
                            bias.astype(np.float32)
                        # KL: useful when debugging
                        # ext.data[y1:y2, asec.x1:asec.x2] = \
                        #     ext.data[y1:y2, asec.x1:asec.x2] - \
                        #     bias.astype(np.float32)


                    ext.hdr.set('OVERSEC', f'[{x1+1}:{x2},{y1+1}:{y2}]',
                                self.keyword_comments['OVERSEC'])
                    # Some shenanigans to deal with the case where the user
                    # subtracts a non-corrected bias from a non-corrected
                    # science and then overscanCorrects the result
                    previous_overscan = ext.hdr.get('OVERSCAN', 0)
                    bias_level = np.median(data)
                    ext.hdr.set('OVERSCAN', previous_overscan + bias_level,
                                self.keyword_comments['OVERSCAN'])
                    ext.hdr.set('OVERRMS', sigma, self.keyword_comments['OVERRMS'])
                    ext.hdr.set('OVERRDNS', overstd, self.keyword_comments['OVERRDNS'])
                    ext.hdr.set('RDNOISEM', overstd*ext.gain(), self.keyword_comments['RDNOISEM'])
                    for desc in ('saturation_level', 'non_linear_level'):
                        with suppress(AttributeError, KeyError):
                            ext.hdr[ad._keyword_for(desc)] -= bias_level


            # Timestamp, and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)

        return adinputs

    def trimOverscan(self, adinputs=None, suffix=None):
        """
        The trimOverscan primitive trims the overscan region from the input
        AstroData object and updates the headers.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        for ad in adinputs:
            if ad.phu.get(timestamp_key) is not None:
                log.warning('No changes will be made to {}, since it has '
                            'already been processed by trimOverscan'.
                            format(ad.filename))
                continue

            ad = gt.trim_to_data_section(
                ad, keyword_comments=self.keyword_comments)

            # Delete overscan_section keyword so no attempt is made to measure it
            with suppress(AttributeError, KeyError):
                del ad.hdr[ad._keyword_for('overscan_section')]

            # Set keyword, timestamp, and update filename
            ad.phu.set('TRIMMED', 'yes', self.keyword_comments['TRIMMED'])
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)
        return adinputs
