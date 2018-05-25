#
#                                                                  gemini_python
#
#                                                       primitives_niri_image.py
# ------------------------------------------------------------------------------
from .primitives_niri import NIRI
from ..core import Image, Photometry
from . import parameters_niri_image

from gempy.gemini import gemini_tools as gt
from gempy.library.nddops import NDStacker

import numpy as np
from functools import partial
from itertools import product as cart_product
from astropy.stats import sigma_clip
import warnings

from recipe_system.utils.decorators import parameter_override
# ------------------------------------------------------------------------------
@parameter_override
class NIRIImage(NIRI, Image, Photometry):
    """
    This is the class containing all of the preprocessing primitives
    for the F2Image level of the type hierarchy tree. It inherits all
    the primitives from the level above
    """
    tagset = set(["GEMINI", "NIRI", "IMAGE"])

    def __init__(self, adinputs, **kwargs):
        super(NIRIImage, self).__init__(adinputs, **kwargs)
        self._param_update(parameters_niri_image)

    def removePatternNoise(self, adinputs=None, **params):
        """
        This attempts to remove the pattern noise in NIRI/GNIRS data. In each
        quadrant, boxes of a specified size are extracted and, for each pixel
        location in the box, the median across all the boxes is determined.
        The resultant median is then tiled to the size of the quadrant and
        subtracted. Optionally, the median of each box can be subtracted
        before performing the operation.

        Based on Andy Stephens's "cleanir"

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        force: bool
            perform operation even if standard deviation in quadrant increases?
        hsigma/lsigma: float
            sigma-clipping limits
        pattern_x_size: int
            size of pattern "box" in x direction
        pattern_y_size: int
            size of pattern "box" in y direction
        subtract_background: bool
            remove median of each "box" before calculating pattern noise?
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        hsigma, lsigma = params["hsigma"], params["lsigma"]
        pxsize, pysize = params["pattern_x_size"], params["pattern_y_size"]
        bgsub = params["subtract_background"]
        force = params["force"]
        stack_function = NDStacker(combine='median', reject='sigclip',
                                   hsigma=hsigma, lsigma=lsigma)
        sigclip = partial(sigma_clip, sigma_lower=lsigma, sigma_upper=hsigma)
        zeros = None  # will remain unchanged if not subtract_background

        for ad in adinputs:
            for ext in ad:
                qysize, qxsize = [size // 2 for size in ext.data.shape]
                yticks = [(y, y + pysize) for y in range(0, qysize, pysize)]
                xticks = [(x, x + pxsize) for x in range(0, qxsize, pxsize)]
                for ystart in (0, qysize):
                    for xstart in (0, qxsize):
                        quad = ext.nddata[ystart:ystart + qysize, xstart:xstart + qxsize]
                        sigma_in = sigclip(np.ma.masked_array(quad.data, quad.mask)).std()
                        print sigma_in
                        blocks = [quad[tuple(slice(start, end)
                                             for (start, end) in coords)]
                                  for coords in cart_product(yticks, xticks)]
                        if bgsub:
                            # If all pixels are masked in a box, we'll try to
                            # take the mean of an empty slice. Suppress warning.
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", category=RuntimeWarning)
                                zeros = np.nan_to_num([-np.mean(block.data[block.mask == 0])
                                                       for block in blocks])
                        out = stack_function(blocks, zero=zeros).data
                        out_quad = (quad.data + np.mean(out) -
                                    np.tile(out, (len(yticks), len(xticks))))
                        sigma_out = sigclip(np.ma.masked_array(out_quad, quad.mask)).std()
                        print sigma_in, sigma_out
                        if sigma_out < sigma_in or force:
                            ext.data[ystart:ystart + qysize, xstart:xstart + qxsize] = out_quad

            ad.update_filename(suffix=params["suffix"], strip=True)
        return adinputs
