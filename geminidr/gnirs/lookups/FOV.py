from os import path

import numpy as np
from scipy import ndimage

from geminidr.gemini.lookups import DQ_definitions as DQ


# ------------------------------------------------------------------------------
# This code is looked up by gempy as part of the configuration for the
# appropriate instrument and evaled by the infrastructure. It has initially
# been written to support gemini_tools.ExposureGroup.
#
# All calculations are now done in PIXELS

def pointing_in_field(ad, refpos, frac_FOV=1.0, frac_slit=1.0):

    """
    See gemini_tools.pointing_in_field() for the API. This is an
    instrument-specific back end that you shouldn't be calling directly.

    No inputs are validated at this level; that's the responsibility of the
    calling function, for reasons of efficiency.
    
    The GNIRS FOV is determined by whether the calculated center point 
    (according to the center of mass of the illumination mask) of the
    image falls within the illumination mask of the reference image.
    
    :param pos: AstroData instance to be checked for whether it belongs
                in the same sky grouping as refpos
    :type pos: AstroData instance
    
    :param refpos: This is the POFFSET and QOFFSET of the reference image
    :type refpos: tuple of floats
    
    :param frac_FOV: For use with spectroscopy data
    :type frac_FOV: float
    
    :param frac_slit: For use with spectroscopy data
    :type frac_slit: float
    """
    # Object location in AD = refpos + shift
    xshift = ad.detector_x_offset() - refpos[0]
    yshift = ad.detector_y_offset() - refpos[1]

    # We inverse-scale by frac_FOV
    fov_xshift = -int(xshift / frac_FOV)
    fov_yshift = -int(yshift / frac_FOV)
    # Imaging:
    if 'IMAGE' in ad.tags:
        if (abs(fov_yshift) >= ad[0].shape[1] or
                abs(fov_xshift) >= ad[0].shape[1]):
            return False

        illum_data = _create_illum_mask(ad, None, align=False)
        if illum_data is None:
            raise OSError("Problem creating illumination mask for "
                          f"{ad.filename}")

        # Shift the illumination mask and see if the shifted keyhole
        # overlaps with the original keyhole
        shifted_data = ndimage.shift(illum_data, (fov_yshift, fov_xshift),
                                     order=0, cval=DQ.unilluminated)
        return (illum_data | shifted_data == 0).any()

    # Spectroscopy:
    elif 'SPECT' in ad.tags:
        raise NotImplementedError("FOV lookup not yet supported for GNIRS "
                                  "Spectroscopy")

    # Some engineering observation or bad mask value etc.:
    else:
        raise ValueError("Can't determine FOV for unrecognized GNIRS config "
          "({}, {})".format(ad.focal_plane_mask(), ad.disperser()))


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
