from functools import partial

import numpy as np
from scipy.special import erf

from geminidr.core.primitives_telluric import LineSpreadFunction

from gempy.library import convolution


class F2LineSpreadFunction(LineSpreadFunction):
    parameters = ["lsf_scaling"]

    # These are the c2 Chebyshev1D coeffs multiplied by 2/1061**2
    quad_coeffs = {
        ("R3K", "J"): 1.485e-6,
        ("R3K", "H"): 2.065e-6,
        ("R3K", "K-long"): 3.035e-6,
    }

    def __init__(self, ext):
        super().__init__(ext)
        self.slit_width_pix = int(ext.focal_plane_mask().replace('pix-slit', ''))
        key = (ext.disperser(pretty=True), ext.filter_name(pretty=True))
        try:
            quad_coeff = self.quad_coeffs[key]
        except KeyError:
            raise KeyError(f"Unsupported configuration {key}")
        self.skew = lambda y: (y / 1061 - 1) * 9.5
        self.omega = lambda y: 0.9 * self.dispersion + quad_coeff * (y - 1061) ** 2

    def skew_normal(self, w0, dw, scale=1):
        y = np.argmin(abs(w0 - self.all_waves))
        skew, omega = self.skew(y), self.omega(y) * scale
        #print(w0, y, skew, omega)
        xnorm = dw / omega + skew / np.sqrt(0.5 * np.pi * (1 + skew * skew))
        phi = np.exp(-0.5 * xnorm * xnorm) * (1 + erf(0.70710678 * skew * xnorm))
        return phi / phi.sum()

    def convolutions(self, lsf_scaling=1):
        slit_width = self.slit_width_pix * self.dispersion
        print("SLIT WIDTH", slit_width, self.slit_width_pix, self.dispersion)
        boxcar_func = partial(convolution.boxcar, width=slit_width)
        convolutions = [(partial(self.skew_normal, scale=lsf_scaling), 20*self.dispersion),
                        (boxcar_func, 0.5*slit_width)]
        print("CONVOLUTIONS", convolutions)
        return convolutions
