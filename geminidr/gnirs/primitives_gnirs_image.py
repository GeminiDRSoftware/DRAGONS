#
#                                                                  gemini_python
#
#                                                      primitives_gnirs_image.py
# ------------------------------------------------------------------------------
import os

import numpy as np
from scipy import ndimage

import astrodata, gemini_instruments
from geminidr.gemini.lookups import DQ_definitions as DQ
from gempy.gemini import gemini_tools as gt

from .primitives_gnirs import GNIRS
from ..core import Image, Photometry
from . import parameters_gnirs_image
from .lookups import maskdb

from recipe_system.utils.decorators import parameter_override, capture_provenance


# ------------------------------------------------------------------------------
@parameter_override
@capture_provenance
class GNIRSImage(GNIRS, Image, Photometry):
    """
    This is the class containing all of the preprocessing primitives
    for the GNIRSImage level of the type hierarchy tree. It inherits all
    the primitives from the level above
    """
    tagset = {"GEMINI", "GNIRS", "IMAGE"}

    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_gnirs_image)

    def addIllumMaskToDQ(self, adinputs=None, **params):
        """
        This primitive combines the illumination mask from the lookup directory
        into the DQ plane

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        # Since this primitive needs a reference, it must no-op without any
        if not adinputs:
            return adinputs

        # Get list of input and identify a suitable reference frame.
        # In most case it will be the first image, but for lamp-on, lamp-off
        # flats, one wants the reference frame to be a lamp-on since there's
        # next to no signal in the lamp-off.
        #
        # BEWARE: See note on the only GCAL_IR_OFF case below this block.
        lampons = self.selectFromInputs(adinputs, 'GCAL_IR_ON')
        reference = lampons[0] if lampons else adinputs[0]

        # When only a GCAL_IR_OFF is available:
        # To cover the one-at-a-time mode check for a compatible list
        # if list found, try to find a lamp-on in there to use as
        # reference for the mask and the shifts.
        # The mask's name and the shifts should stored in the headers
        # of the reference to simplify this and speed things up.
        #
        # NOT NEEDED NOW because we calling addIllumMask after the
        # lamp-offs have been subtracted.  But kept the idea here
        # in case we needed.

        # Fetching a corrected illumination mask with a keyhole that aligns
        # with the science data
        illum_mask = _create_illum_mask(reference, log, xshift=params["xshift"],
                                        yshift=params["yshift"])
        if illum_mask is None:
            log.warning(f"No illumination mask found for {reference.filename},"
                        " no mask can be added to the DQ planes of the inputs")
            return adinputs

        illum_mask = illum_mask.astype(DQ.datatype) * DQ.unilluminated
        for ad in adinputs:
            # binary_OR the illumination mask or create a DQ plane from it.
            if ad[0].mask is None:
                ad[0].mask = illum_mask
            else:
                ad[0].mask |= illum_mask

            # Update the filename
            ad.update_filename(suffix=params["suffix"], strip=True)
        return adinputs

    def _fields_overlap(self, ad1, ad2, frac_FOV=1.0):
        """
        Checks whether the fields of view of two F2 images overlap
        sufficiently to be considered part of a single ExposureGroup.
        GNIRSImage requires its own code since it has a weird FOV.

        Parameters
        ----------
        ad1: AstroData
            one of the input AD objects
        ad2: AstroData
            the other input AD object
        frac_FOV: float (0 < frac_FOV <= 1)
            fraction of the field of view for an overlap to be considered. If
            frac_FOV=1, *any* overlap is considered to be OK

        Returns
        -------
        bool: do the fields overlap sufficiently?
        """
        center = np.array([512, 512])
        # We inverse-scale by frac_FOV
        shift = -(ad2[0].wcs.invert(*ad1[0].wcs(*center)) - center) / frac_FOV
        illum_data = _create_illum_mask(ad1, self.log, align=False)
        shifted_data = ndimage.shift(illum_data, shift, order=0,
                                     cval=DQ.unilluminated)
        return np.any(~(illum_data | shifted_data))

    def _get_illum_mask_filename(self, ad):
        """Should never be called"""
        raise NotImplementedError("GNIRS IMAGE has no illum_mask files")

##############################################################################
# Below are the helper functions for the user level functions in this module #
##############################################################################


def _create_illum_mask(ad, log, align=True, xshift=0, yshift=0):
    """
    Creates a keyhole mask for GNIRS imaging using geometric shapes. If
    requested this is aligned with an AD instance based on its illuminated
    pixels.

    Parameters
    ----------
    adinput: AstroData
        single AD instance to which the keyhole should be matched
    log: logger
        the log
    align: bool
        align the mask to the AD instance? (set to False for testing whether
        images overlap)
    xshift: int
        pixel shift in x compared to automated placement
    yshift: int
        pixel shift in y compared to automated placement

    Returns
    -------
    mask: a boolean ndarray of the masked pixels
    """
    # Center of Mass X and Y and number of illuminated pixels
    # Apparently no difference between the two ShortBlue cameras but we keep
    # the component number in here in case that's not true of future cameras
    keyhole_dict = {('LongBlue_G5542', False):  (537.2, 454.8, 205000),
                    ('LongBlue_G5542', True):   (518.4, 452.9, 275000),
                    ('LongRed_G5543', False):   (451.0, 429.1, 198000),
                    ('LongRed_G5543', True):    (532.4, 507.7, 272000),
                    ('ShortBlue_G5538', False): (504.6, 446.8, 25000),
                    ('ShortBlue_G5538', True):  (506.8, 457.1, 49000),
                    ('ShortBlue_G5540', False): (504.6, 446.8, 25000),
                    ('ShortBlue_G5540', True):  (506.8, 457.1, 49000),
                    ('ShortRed_G5539', True):   (519.3, 472.6, 58000)}

    pixscale = ad.pixel_scale()
    wings = 'PHOT' not in ad.filter_name(pretty=True)
    try:
        comx0, comy0, ngood = keyhole_dict[ad.camera(), wings]
    except KeyError:
        return
    comx, comy = comx0, comy0

    addata = ad[0].data
    if align:
        med = max(np.median(addata), 0.0)
        # Get read noise in appropriate units (descriptor returns electrons)
        rdnoise = ad.read_noise()[0]
        if ad.is_in_adu():
            rdnoise /= ad.gain()[0]

        # Do a bit of binary dilation of unilluminated pixels in order to
        # eliminate small groups of bright bad pixels
        threshold = med + 3 * rdnoise
        structure = np.ones((5, 5))
        regions, nregions = ndimage.label(
            ~ndimage.binary_dilation(addata < threshold, structure))
        # Find the region with the most pixels (but not region 0)
        keyhole_region = np.argmax([(regions == i).sum()
                                    for i in range(1, nregions+1)]) + 1

        n_inregion = (regions == keyhole_region).sum()
        if n_inregion > 0.5 * ngood:
            comy, comx = ndimage.center_of_mass(regions == keyhole_region)
        else:
            log.warning(f"Only {n_inregion} illuminated pixels detected in "
                        f"{ad.filename}, when {ngood} are expected. Using the "
                        "default illumination mask position.")

    r1 = 14.7 / pixscale  # keyhole circle has radius of 14.7 arcsec
    width = 9.6 / pixscale  # "bar" is 9.3 arcsec wide
    length = 97.8 / pixscale  # "bar" is 97.8 arcsec long

    y, x = np.mgrid[:ad[0].shape[0], :ad[0].shape[1]]
    mask = np.ones_like(addata, dtype=bool)
    xedges = comx + np.array([-0.5, 0.5]) * length + xshift
    xc = comx + 2 + xshift
    yc = comy + 3.2 / pixscale + yshift

    # Not much data so this is a bit ad hoc
    if 'Red' in ad.camera():
        xedges -= 5
        xc -= 5
        yc -= 4

    if wings:
        # Having wings means the CoM is much higher than the circle centre
        wingshift = 15 * (min(xedges[1], 1023) - max(xedges[0], 0)) / 1023
        yc -= wingshift
        # If the bar covers the full width of the detector, then the CoM's
        # horizontal position is not at the centre of the circle
        if xedges[0] <= 0:  # only Long cameras
            xc += (comx - 512) * 2.5
    elif pixscale < 0.1:
        yc -= 4

    # The "bar"
    mask[np.logical_and.reduce([xedges[0] <= x, x <= xedges[1],
                                y > yc - 0.5 * width - 0.01 * (comx - x),
                                y <= yc + 0.5 * width - 0.01 * (comx - x)])] = False
    # The keyhole
    mask[np.logical_and((x - xc) ** 2 + (y - yc) ** 2 < r1 * r1,
                        y < yc)] = False
    if not wings:
        r2 = (1.13 if pixscale < 0.1 else 1.29) * r1
        mask[(x - xc) ** 2 + (y - yc) ** 2 > r2 * r2] = True

    if align:
        dx, dy = comx - comx0, comy - comy0
        log.stdinfo("Applying shifts to the illumination mask: "
                    f"dx = {dx:.1f}px, dy = {dy:.1f}px.")

    return mask
