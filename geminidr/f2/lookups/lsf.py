from functools import partial

import numpy as np
from scipy.special import erf

from geminidr.core.primitives_telluric import LineSpreadFunction

from gempy.library import convolution


def lsf_factory(classname):
    """In case the LSF depends on the class (Longslit v MOS v IFU)"""
    return F2LineSpreadFunction


class F2LineSpreadFunction(LineSpreadFunction):
    parameters = ["lsf_scaling"]

    res_2pix = {'JH':  1200,
                'HK':  1200,
                'R3K': 3200}

    slit_row = 1061  # location of slit on the detector

    def __init__(self, ext):
        super().__init__(ext)
        self.slit_width_pix = int(ext.focal_plane_mask().replace('pix-slit', ''))
        self.grism = ext.disperser(pretty=True)
        self.resolution = self.res_2pix[self.grism] * 2 / self.slit_width_pix
        self.orig_dispersion = abs(ext.dispersion(asNanometers=True))
        self.orig_cenwave = ext.actual_central_wavelength(asNanometers=True)

        # skew is in the sense of wavelength, and y is the *original* row
        # so large y => -ve skew because they are the bluest wavelengths
        self.skew = lambda y: (1 - y / self.slit_row) * 12
        self.omega = lambda y: (5.0 + 4.0 * (2 * (y / self.slit_row - 1) ** 2 - 1)) * self.dispersion

        # Estimate the mean resolution at an "average" row
        # This attribute is needed for the wavecal reference spectrum plot
        alpha = self.skew(1450)
        omega = self.omega(1450)
        # From https://en.wikipedia.org/wiki/Skew_normal_distribution
        var_skew = omega ** 2 * (1 - alpha ** 2 / (np.pi * (1 + alpha ** 2)))
        var_slit = (self.slit_width_pix * self.orig_dispersion) ** 2
        self.mean_resolution = self.all_waves.mean() / np.sqrt(var_skew + var_slit)

    def skew_normal(self, w0, dw, scale=1):
        # actual_central_wavelength corresponds to row 1024, not the slit row
        y = 1024 - (w0 - self.orig_cenwave) / self.orig_dispersion
        skew, omega = self.skew(y), self.omega(y) * scale
        xnorm = dw / omega + skew / np.sqrt(0.5 * np.pi * (1 + skew * skew))
        phi = np.exp(-0.5 * xnorm * xnorm) * (1 + erf(0.70710678 * skew * xnorm))
        return phi / phi.sum()

    def convolutions(self, lsf_scaling=1):
        #boxcar_func = partial(convolution.boxcar, width=slit_width*lsf_scaling)
        #convolutions = [(partial(self.skew_normal, scale=1), 25 * self.dispersion),
        #                #boxcar_func, 0.5 * slit_width * lsf_scaling)]
        fwhm = self.slit_width_pix * self.orig_dispersion
        gaussian_func = partial(convolution.gaussian_constant_fwhm,
                                fwhm=fwhm*lsf_scaling)
        gaussian_dw = 2 * fwhm
        convolutions = [(partial(self.skew_normal, scale=1), 25 * self.dispersion),
                        (gaussian_func, gaussian_dw)]
        print("CONVOLUTIONS", convolutions)
        return convolutions
