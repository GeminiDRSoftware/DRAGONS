from functools import partial

from geminidr.core.primitives_telluric import LineSpreadFunction

from gempy.library import convolution


def lsf_factory(classname):
    """In case the LSF depends on the class (Longslit v MOS v IFU)"""
    return IGRINSLineSpreadFunction


class IGRINSLineSpreadFunction(LineSpreadFunction):
    parameters = ["lsf_scaling"]

    def __init__(self, ext):
        super().__init__(ext)
        self.resolution = 35000
        self.mean_resolution = self.resolution

    def convolutions(self, lsf_scaling=1):
        resolution = self.resolution / lsf_scaling
        gaussian_func = partial(convolution.gaussian_constant_r, r=resolution)
        gaussian_dw = 3 * self.all_waves.max() / resolution
        convolutions = [(gaussian_func, gaussian_dw)]
        #print("CONVOLUTIONS", convolutions)
        return convolutions
