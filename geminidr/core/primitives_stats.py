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
        Adds headers to the AD object giving some statistics of the unmasked
        pixel values
        By default, adds:
        PIXMEAN - the arithmetic mean of the pixel values
        PIXSTDV - the standard deviation of the pixel values
        PIXMED - the median of the pixel values.

        Parameters
        ----------

        adinputs: list of :class:`~astrodata.AstroData`

        prefix: Prefix for header keywords. Maximum of 4 characters, defaults
                to PIX.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        pre = params['prefix']

        if pre is None:
            pre = 'PIX'
        else:
            pre = pre[:4]

        for ad in adinputs:
            for ext in ad:
                try:
                    ext.hdr[pre+'MEAN'] = (np.mean(ext.data[ext.mask==0]),
                                          "Mean of unmasked pixel values")
                except ValueError:
                    # Things like NaN can't be written to FITS headers
                    pass
                try:
                    ext.hdr[pre+'STDV'] = (np.std(ext.data[ext.mask==0]),
                                           "Standard Deviation of pixel values")
                except ValueError:
                    pass
                try:
                    ext.hdr[pre+'MED'] = (np.median(ext.data[ext.mask == 0]),
                                         "Median of unmasked of pixel values")
                except ValueError:
                    pass
        return adinputs
