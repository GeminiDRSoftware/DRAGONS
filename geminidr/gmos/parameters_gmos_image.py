# This parameter file contains the parameters related to the primitives located
# in the primitives_gmos_image.py file, in alphabetical order.
from gempy.library import config
from geminidr.core import parameters_image, parameters_standardize, parameters_stack

class addOIWFSToDQConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_oiwfsDqAdded", optional=True)
    contrast = config.RangeField("Fractional decrease in sky level", float, 0.2, min=0.001, max=1.)
    convergence = config.RangeField("Convergence required in sky level to stop dilation",
                                    float, 2.0, min=0.001)

class addDQConfig(parameters_standardize.addDQConfig):
    def setDefaults(self):
        self.add_illum_mask = True

class makeFringeFrameConfig(parameters_image.makeFringeFrameConfig):
    subtract_median_image = config.Field("Subtract median image?", bool, None, optional=True)

class scaleFlatsConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_scaled", optional=True)

class stackFlatsConfig(parameters_stack.core_stacking_config):
    def setDefaults(self):
        del (self.apply_dq, self.hsigma, self.lsigma, self.nlow, self.nhigh,
             self.max_iters, self.reject_method, self.save_rejection_map, self.statsec)
