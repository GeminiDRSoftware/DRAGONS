#
#                                                                  gemini_python
#
#                                                              primitives_ccd.py
# ------------------------------------------------------------------------------
from datetime import datetime

import numpy as np

from astropy.modeling import models, fitting
from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline

from astrodata.provenance import add_provenance
from astrodata import wcs as adwcs
from gempy.gemini import gemini_tools as gt

from geminidr import PrimitivesBASE
from recipe_system.utils.md5 import md5sum
from . import parameters_ccd

from recipe_system.utils.decorators import parameter_override
# ------------------------------------------------------------------------------
@parameter_override
class CCD(PrimitivesBASE):
    """
    This is the class containing all of the primitives used for generic CCD
    reduction.
    """
    tagset = None

    def __init__(self, adinputs, **kwargs):
        super().__init__(adinputs, **kwargs)
        self._param_update(parameters_ccd)

    def biasCorrect(self, adinputs=None, suffix=None, bias=None, do_bias=True):
        """
        The biasCorrect primitive will subtract the science extension of the
        input bias frames from the science extension of the input science
        frames. The variance and data quality extension will be updated, if
        they exist. If no bias is provided, getProcessedBias will be called
        to ensure a bias exists for every adinput.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        bias: str/list of str
            bias(es) to subtract
        do_bias: bool
            perform bias subtraction?
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        if not do_bias:
            log.warning("Bias correction has been turned off.")
            return adinputs

        if bias is None:
            self.getProcessedBias(adinputs, refresh=False)
            bias_list = self._get_cal(adinputs, 'processed_bias')
        else:
            bias_list = bias

        # Provide a bias AD object for every science frame
        for ad, bias in zip(*gt.make_lists(adinputs, bias_list, force_ad=True)):
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by biasCorrect".
                            format(ad.filename))
                continue

            if bias is None:
                if 'qa' in self.mode:
                    log.warning("No changes will be made to {}, since no "
                                "bias was specified".format(ad.filename))
                    continue
                else:
                    raise OSError('No processed bias listed for {}'.
                                  format(ad.filename))

            try:
                gt.check_inputs_match(ad, bias, check_filter=False,
                                      check_units=True)
            except ValueError:
                bias = gt.clip_auxiliary_data(ad, aux=bias, aux_type='cal')
                # An Error will be raised if they don't match now
                gt.check_inputs_match(ad, bias, check_filter=False,
                                      check_units=True)

            log.fullinfo('Subtracting this bias from {}:\n{}'.
                         format(ad.filename, bias.filename))
            ad.subtract(bias)

            # Record bias used, timestamp, and update filename
            ad.phu.set('BIASIM', bias.filename, self.keyword_comments['BIASIM'])
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)
            if bias.path:
                add_provenance(ad, bias.filename, md5sum(bias.path) or "", self.myself())

            timestamp = datetime.now()
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

        "poly":   a polynomial of degree *order* (1=linear, etc)
        "spline": using *order* equally-sized cubic spline pieces or, if
                  order=None or 0, a spline that provides a reduced chi^2=1
        "none":   no function is fit, and the value for each row is determined
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
            function to fit ("poly" | "spline" | "none")
        order: int/None
            order of polynomial fit or number of spline pieces
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        sfx = params["suffix"]
        niterate = params["niterate"]
        lo_rej = params["low_reject"]
        hi_rej = params["high_reject"]
        order = params["order"] or 0  # None is the same as 0
        func = (params["function"] or 'none').lower()
        nbiascontam = params["nbiascontam"]

        for ad in adinputs:
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by subtractOverscan".
                            format(ad.filename))
                continue

            osec_list = ad.overscan_section()
            dsec_list = ad.data_section()
            for ext, osec, dsec in zip(ad, osec_list, dsec_list):
                x1, x2, y1, y2 = osec.x1, osec.x2, osec.y1, osec.y2
                if x1 > dsec.x1:  # Bias on right
                    x1 += nbiascontam
                    x2 -= 1
                else:  # Bias on left
                    x1 += 1
                    x2 -= nbiascontam

                row = np.arange(y1, y2)
                data = np.mean(ext.data[y1:y2, x1:x2], axis=1)
                # Weights are used to determine number of spline pieces
                # should be the estimate of the mean
                wt = np.sqrt(x2 - x1) / ext.read_noise()
                if ext.is_in_adu():
                    wt *= ext.gain()

                medboxsize = 2  # really 2n+1 = 5
                for iter in range(niterate+1):
                    # The UnivariateSpline will make reduced-chi^2=1 so it will
                    # fit bad rows. Need to mask these before starting, so use a
                    # running median. Probably a good starting point for all fits.
                    if iter == 0 or func == 'none':
                        medarray = np.full((medboxsize * 2 + 1, y2 - y1), np.nan)
                        for i in range(-medboxsize, medboxsize + 1):
                            mx1 = max(i, 0)
                            mx2 = min(y2 - y1, y2 - y1 + i)
                            medarray[medboxsize + i, mx1:mx2] = data[:mx2 - mx1]
                        runmed = np.ma.median(np.ma.masked_where(np.isnan(medarray),
                                                                 medarray), axis=0)
                        residuals = data - runmed
                        sigma = np.sqrt(x2 - x1) / wt  # read noise

                    mask = np.logical_or(residuals > hi_rej * sigma
                                        if hi_rej is not None else False,
                                        residuals < -lo_rej * sigma
                                        if lo_rej is not None else False)

                    # Don't clip any pixels if iter==0
                    if func == 'none' and iter < niterate:
                        # Replace bad data with running median
                        data = np.where(mask, runmed, data)
                    elif func != 'none':
                        if func == 'spline':
                            if order:
                                # Equally-spaced knots (like IRAF)
                                knots = np.linspace(row[0], row[-1], order+1)[1:-1]
                                bias = LSQUnivariateSpline(row[~mask], data[~mask], knots)
                            else:
                                bias = UnivariateSpline(row[~mask], data[~mask],
                                                        w=[wt]*np.sum(~mask))
                        else:
                            bias_init = models.Chebyshev1D(degree=order,
                                                           c0=np.median(data[~mask]))
                            fit_f = fitting.LinearLSQFitter()
                            bias = fit_f(bias_init, row[~mask], data[~mask])

                        residuals = data - bias(row)
                        sigma = np.std(residuals[~mask])

                # using "-=" won't change from int to float
                if func != 'none':
                    data = bias(np.arange(0, ext.data.shape[0]))
                ext.data = ext.data - np.tile(data,
                                        (ext.data.shape[1],1)).T.astype(np.float32)

                ext.hdr.set('OVERSEC', '[{}:{},{}:{}]'.format(x1+1,x2,y1+1,y2),
                            self.keyword_comments['OVERSEC'])
                ext.hdr.set('OVERSCAN', np.mean(data),
                            self.keyword_comments['OVERSCAN'])
                ext.hdr.set('OVERRMS', sigma, self.keyword_comments['OVERRMS'])

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

            ad = gt.trim_to_data_section(ad,
                                    keyword_comments=self.keyword_comments)
            # HACK! Need to update FITS header because imaging primitives edit it
            if 'IMAGE' in ad.tags:
                for ext in ad:
                    if ext.wcs is not None:
                        wcs_dict = adwcs.gwcs_to_fits(ext, ad.phu)
                        ext.hdr.update(wcs_dict)

            # Set keyword, timestamp, and update filename
            ad.phu.set('TRIMMED', 'yes', self.keyword_comments['TRIMMED'])
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)
        return adinputs
