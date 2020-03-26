#
#                                                                  gemini_python
#
#                                                      primitives_gnirs_image.py
# ------------------------------------------------------------------------------
import numpy as np
import scipy.ndimage
from skimage.morphology import binary_dilation

import astrodata
import gemini_instruments
from gempy.gemini import gemini_tools as gt

from .lookups import FOV as fov

from .primitives_gnirs import GNIRS
from ..core import Image, Photometry
from . import parameters_gnirs_image

from recipe_system.utils.decorators import parameter_override
# ------------------------------------------------------------------------------
@parameter_override
class GNIRSImage(GNIRS, Image, Photometry):
    """
    This is the class containing all of the preprocessing primitives
    for the GNIRSImage level of the type hierarchy tree. It inherits all
    the primitives from the level above
    """
    tagset = set(["GEMINI", "GNIRS", "IMAGE"])

    def __init__(self, adinputs, **kwargs):
        super().__init__(adinputs, **kwargs)
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
        illum = self._get_illum_mask_filename(reference)
        if illum is None:
            log.warning("No illumination mask found for {}, no mask can "
                        "be added to the DQ planes of the inputs".
                        format(reference.filename))
            return adinputs

        illum_ad = astrodata.open(illum)
        corr_illum_ad = _position_illum_mask(reference, illum_ad, log)

        for ad in adinputs:
            final_illum = gt.clip_auxiliary_data(ad, aux=corr_illum_ad,
                    aux_type="bpm")

            # binary_OR the illumination mask or create a DQ plane from it.
            if ad[0].mask is None:
                ad[0].mask = final_illum[0].data
            else:
                ad[0].mask |= final_illum[0].data

            # Update the filename
            ad.update_filename(suffix=params["suffix"], strip=True)
        return adinputs

    def _get_illum_mask_filename(self, ad):
        """
        Gets the illumMask filename for an input science frame, using
        illumMask_dict in geminidr.<instrument>.lookups.maskdb.py and looks
        for a key <INSTRUMENT>_<MODE>_<XBIN><YBIN>. This file will be sent
        to clip_auxiliary_data for a subframe ROI.

        Returns
        -------
        str/None: Filename of the appropriate illumination mask
        """
        # TODO: Look at the whole pointing_in_field situation
        return fov.get_illum_mask_filename(ad)

##############################################################################
# Below are the helper functions for the user level functions in this module #
##############################################################################

def _position_illum_mask(adinput, illum, log, max_dy=20):
    """
    This function is used to reposition a GNIRS illumination mask so that
    the keyhole matches with the science data.

    Parameters
    ----------
    adinput: AstroData
        single AD instance to which the keyhole should be matched
    illum: AstroData
        the standard illumination mask
    log: logger
        the log
    max_dy: int
        maximum y shift allowed
    """
    # Normalizing and thresholding the science data to get a rough
    # illumination mask. A 5x5 box around non-illuminated pixels is also
    # flagged as non-illuminated to better handle the edge effects. The
    # limit for thresholding is twice the median *and* more than 3x the
    # read noise. A check is made that a reasonable number of pixels
    # pass this test, since center_of_mass() will always return a result
    # even when it really shouldn't.
    addata = adinput[0].data
    med = max(np.median(addata), 0.0)
    # Get read noise in appropriate units (descriptor returns electrons)
    rdnoise = adinput.read_noise()[0]
    if adinput.is_in_adu():
        rdnoise /= adinput.gain()[0]
    # Set UNilliminated pixels here, to allow the binary_dilation to work
    threshpixdata = np.where(np.logical_and(addata>2*med, addata>3*rdnoise),
                             np.int16(0), np.int16(1))
    structure = np.ones((5,5))
    threshpixdata = binary_dilation(threshpixdata, structure)

    # There can be stray illumination (ghosts?) on the array which will upset
    # the centre-of-mass measurement, so disregard pixels more than a certain
    # distance from the keyhole in the illumination mask
    illum_shape = illum[0].data.shape
    data_shape = threshpixdata.shape
    first, last = np.argmin(illum[0].data), np.argmin(illum[0].data[::-1])
    ymin = np.unravel_index(first, illum_shape)[0]
    ymax = illum_shape[0] - np.unravel_index(last, illum_shape)[0]
    if ymin > max_dy:
        threshpixdata[:ymin-max_dy] = np.ones((ymin-max_dy, data_shape[1]))
    if ymax + max_dy < data_shape[0]:
        threshpixdata[ymax+max_dy:] = np.ones((data_shape[0]-ymax-max_dy,
                                               data_shape[1]))

    # Invert to set illuminated pixels. Demand a minimum number of such pixels
    keyhole = 1 - threshpixdata
    numpix_mask_illum = np.sum(illum[0].data==0)
    numpix_data_illum = np.sum(keyhole)
    if numpix_data_illum < 0.5 * numpix_mask_illum:
        log.warning("Only {} illuminated pixels detected in {}, when {} are"
                    "expected. Using the illumination mask without adjustment"
                    ".".format(numpix_data_illum, adinput.filename,
                               numpix_mask_illum))
        return illum

    # Finding the centre of mass of the rough pixel mask and using
    # this in comparison with the centre of mass of the illumination
    # mass to adjust the keyholes to align. Note that the
    # center_of_mass function has switched x and y axes compared to normal.
    comx_illummask = illum.phu['CENMASSX']
    comy_illummask = illum.phu['CENMASSY']
    comy, comx = scipy.ndimage.measurements.center_of_mass(keyhole)
    if not np.isnan(comx) and not np.isnan(comy):
        dx = int(comx - comx_illummask)
        dy = int(comy - comy_illummask)
    else:
        log.warning("The centre of mass of {} cannot be measured, so "
                "the illumination mask cannot be positioned and "
                "will be used without adjustment".format(adinput.filename))
        return illum

    # Recording the shifts in the header of the illumination mask
    log.stdinfo("Applying shifts to the illumination mask: dx = {}px, dy = "
                "{}px.".format(dx, dy))
    illum.phu.set('OBSHIFTX', dx, "Relative x shift to object frame")
    illum.phu.set('OBSHIFTY', dy, "Relative y shift to object frame")

    # Applying the offsets to the illumination mask
    illumpixdata1 = illum[0].data
    illumpixdata2 = np.roll(illumpixdata1, dx, 1)
    illumpixdata3 = np.roll(illumpixdata2, dy, 0)
    illum[0].data = illumpixdata3

    return illum
