from copy import deepcopy

import numpy as np

from geminidr import PrimitivesBASE
from gempy.gemini import gemini_tools as gt
from recipe_system.utils.decorators import parameter_override, capture_provenance

from . import parameters_stats

@parameter_override
class Stats(PrimitivesBASE):

    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_stats)

    def stats(self, adinputs=None, **params):
        """
        Adds headers to the AD object giving some statistics of the pixels
        :param adinputs:
        :param params:
        :return:
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        for ad in adinputs:
            for ext in ad:
                ext.hdr['PIXMEAN'] = (np.mean(ext.data[ext.mask==0]),
                                      "Mean of unmasked pixel values")
                ext.hdr['PIXSTDEV'] = (np.std(ext.data[ext.mask==0]),
                                       "Standard Deviation of pixel values")
                ext.hdr['PIXMED'] = (np.median(ext.data[ext.mask == 0]),
                                     "Median of unmasked of pixel values")
        return adinputs


    def signaltonoiseratio(self, adinputs=None, **param):
        """
        Divide the signal by the noise.

        :param adinputs:
        :param param:
        :return:
        """

        # Make a signal-to-noise-ratio image
        for ad in adinputs:
            for ext in ad:
                ext.data /= np.sqrt(ext.variance)

                ext.hdr['SNRMEAN'] = (np.mean(ext.data[ext.mask==0]),
                                      'Mean Signal-to-Noise Ratio')

                # count the number of good pixels where -3 < SNR > 3

                goodpix = np.abs(ext.data[ext.mask==0])

                threesig = np.where(goodpix > 3.0, 1, 0)

                good = np.where(ext.mask==0, 1, 0)

                ext.hdr['FSNRGT3'] = (
                    np.count_nonzero(threesig) / np.count_nonzero(good),
                    "Fraction of pixels more than 3-sigma away from zero"
                )

            ad.update_filename(suffix="_snr")

        return adinputs
