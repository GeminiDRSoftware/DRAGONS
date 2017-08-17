#
#                                                                  gemini_python
#
#                                                              primitives_ccd.py
# ------------------------------------------------------------------------------
import numpy as np

from astropy.modeling import models, fitting
from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline

from gempy.gemini import gemini_tools as gt

from geminidr import PrimitivesBASE
from .parameters_ccd import ParametersCCD

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
        super(CCD, self).__init__(adinputs, **kwargs)
        self.parameters = ParametersCCD

    def biasCorrect(self, adinputs=None, **params):
        self.getProcessedBias(adinputs)
        adinputs = self.subtractBias(adinputs, **params)
        return adinputs

    def overscanCorrect(self, adinputs=None, **params):
        adinputs = self.subtractOverscan(adinputs, **params)
        adinputs = self.trimOverscan(adinputs, **params)
        return adinputs

    def subtractBias(self, adinputs=None, **params):
        """
        The subtractBias primitive will subtract the science extension of the
        input bias frames from the science extension of the input science
        frames. The variance and data quality extension will be updated, if
        they exist.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        bias: str/list of str
            bias(es) to subtract
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        bias_list = params["bias"] if params["bias"] else [
            self._get_cal(ad, 'processed_bias') for ad in adinputs]

        # Provide a bias AD object for every science frame
        for ad, bias in zip(*gt.make_lists(adinputs, bias_list, force_ad=True)):
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by subtractBias".
                            format(ad.filename))
                continue

            if bias is None:
                if 'qa' in self.context:
                    log.warning("No changes will be made to {}, since no "
                                "bias was specified".format(ad.filename))
                    continue
                else:
                    raise IOError('No processed bias listed for {}'.
                                  format(ad.filename))

            try:
                gt.check_inputs_match(ad, bias, check_filter=False)
            except ValueError:
                bias = gt.clip_auxiliary_data(ad, bias, aux_type='cal',
                                    keyword_comments=self.keyword_comments)
                # An Error will be raised if they don't match now
                gt.check_inputs_match(ad, bias, check_filter=False)

            log.fullinfo('Subtracting this bias from {}:\n{}'.
                         format(ad.filename, bias.filename))
            ad.subtract(bias)

            # Record bias used, timestamp, and update filename
            ad.phu.set('BIASIM', bias.filename, self.keyword_comments['BIASIM'])
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.filename = gt.filename_updater(adinput=ad, suffix=params["suffix"],
                                              strip=True)
        return adinputs

    def subtractOverscan(self, adinputs=None, **params):
        """
        Subtract the overscan level from the image by fitting a polynomial
        to the overscan region.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        niterate: int
            number of rejection iterations
        high_reject: float
            number of standard deviations above which to reject high pixels
        low_reject: float
            number of standard deviations above which to reject low pixels
        overscan_section: str/None
            comma-separated list of IRAF-style overscan sections
        nbiascontam: int/None
            number of columns adjacent to the illuminated region to reject
        function: str
            function to fit ("polynomial" | "spline" | "none")
        order: int
            order of Chebyshev fit or spline/None
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        sfx = params["suffix"]
        niterate = params["niterate"]
        lo_rej = params["low_reject"]
        hi_rej = params["high_reject"]
        order = params["order"]
        func = (params["function"] or 'none').lower()
        nbiascontam = params["nbiascontam"]

        if lo_rej < 0:
            log.warning("Low rejection threshold set to invalid value. "
                        "Ignorning.")
            lo_rej = None
        if hi_rej < 0:
            log.warning("High rejection threshold set to invalid value. "
                        "Ignorning.")
            hi_rej = None

        if not(func.startswith('poly') or func == 'spline' or func=='none'):
            log.warning("Unrecognized function {}.".format(func))
            func = 'none'

        for ad in adinputs:
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by subtractOverscan".
                            format(ad.filename))
                continue

            # Use gireduce defaults if values aren't specified
            if 'GMOS' in ad.tags:
                detname = ad.detector_name(pretty=True)
                if order is None and func.startswith('poly'):
                    order = 6 if detname.startswith('Hamamatsu') else 0
                if nbiascontam is None:
                    nbiascontam = 5 if detname == 'e2vDD' else 4
            else:
                detname = ''
                nbiascontam = 0
                order = 1

            osec_list = ad.overscan_section()
            dsec_list = ad.data_section()
            ybinning = ad.detector_y_bin()
            for i, (ext, osec, dsec) in enumerate(zip(ad, osec_list, dsec_list)):
                x1, x2, y1, y2 = osec.x1, osec.x2, osec.y1, osec.y2
                if x1 > dsec.x1:  # Bias on right
                    x1 += nbiascontam
                    x2 -= 1
                else:  # Bias on left
                    x1 += 1
                    x2 -= nbiascontam

                if detname.startswith('Hamamatsu') and func.startswith('poly'):
                    y1 = max(y1, 48 // ybinning)
                    if i == 0:  # Don't log for every extension
                        log.fullinfo('Ignoring bottom 48 rows of {}'.
                                    format(ad.filename))

                row = np.arange(y1, y2)
                data = np.mean(ext.data[y1:y2, x1:x2], axis=1)
                # Weights are used to determine number of spline pieces
                # should be the estimate of the mean
                wt = np.sqrt(x2-x1-1) / ext.read_noise()
                if ext.hdr.get('BUNIT', 'adu').lower() == 'adu':
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
                        sigma = np.sqrt(x2 - x1 + 1) / wt  # read noise

                    mask = np.where(np.logical_or(residuals > hi_rej * sigma
                                        if hi_rej is not None else False,
                                        residuals < -lo_rej * sigma
                                        if lo_rej is not None else False), True, False)

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
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)

        return adinputs

    def trimOverscan(self, adinputs=None, **params):
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
        sfx = params["suffix"]

        for ad in adinputs:
            if ad.phu.get(timestamp_key) is not None:
                log.warning('No changes will be made to {}, since it has '
                            'already been processed by trimOverscan'.
                            format(ad.filename))
                continue

            ad = gt.trim_to_data_section(ad,
                                    keyword_comments=self.keyword_comments)

            # Set keyword, timestamp, and update filename
            ad.phu.set('TRIMMED', 'yes', self.keyword_comments['TRIMMED'])
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
        return adinputs
