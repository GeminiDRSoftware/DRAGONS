# This parameter file contains the parameters related to the primitives located
# in the primitives_crossdispersed.py file, in alphabetical order.

from geminidr.core import parameters_generic
from gempy.library import config
from astrodata import AstroData

from geminidr.core import parameters_spect


class findAperturesConfig(parameters_spect.findAperturesConfig):
    ext = config.RangeField("Extension (1-indexed) to use for finding apertures",
                            int, None, optional=True, min=1, inclusiveMin=True)
    comp_method = config.Field("Comparison method to find 'best' order ('sum', 'median')",
                               str, "sum", optional=True)


class resampleToCommonFrameConfig(parameters_spect.resampleToCommonFrameConfig):
    single_wave_scale = config.Field("Resample all orders to a single wavelength scale?",
                                     bool, False)

    def validate(self):
        super().validate()
        if self.single_wave_scale and self.force_linear == False:
            raise ValueError("Incompatible parameters: single_wave_scale=True"
                             " and force_linear=False")
        if (not self.single_wave_scale and
                [self.w1, self.w2, self.dw, self.npix].count(None) not in (1, 4)):
            raise ValueError("Must specify 0 or 3 resampling parameters "
                             "(w1, w2, dw, npix) if single_wave_scale=False")
