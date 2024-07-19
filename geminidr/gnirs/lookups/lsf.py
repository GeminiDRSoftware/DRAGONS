from functools import partial

from geminidr.core.primitives_telluric import LineSpreadFunction

from gempy.library import convolution


def lsf_factory(classname):
    return GNIRSLineSpreadFunction


class GNIRSLineSpreadFunction(LineSpreadFunction):
    parameters = ["lsf_scaling"]

    def __init__(self, ext):
        super().__init__(ext)
        self.slit_width_arcsec = float(ext.slit(pretty=True).replace('arcsec', ''))
        self.slit_width_pix =  self.slit_width_arcsec / ext.pixel_scale()
        self.grating = float(ext.disperser(pretty=True).split('/')[0])

    def convolutions(self, lsf_scaling=1):
        resolution = 1700 * (0.3 / self.slit_width_arcsec) * (self.grating / 32) / lsf_scaling
        gaussian_func = partial(convolution.gaussian_constant_r, r=resolution)
        gaussian_dw = 3 * self.all_waves.max() / resolution
        convolutions = [(gaussian_func, gaussian_dw)]
        print("CONVOLUTIONS", convolutions)
        return convolutions
