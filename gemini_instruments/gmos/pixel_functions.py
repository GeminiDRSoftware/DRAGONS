#
#                                                                  gemini_python
#
#                                                                   gempy.gemini
#                                                    pixel_functions.py
# ------------------------------------------------------------------------------
# $Id$
# ------------------------------------------------------------------------------
# The gemini_data_calculations module contains functions that calculate values
# from Gemini data, only get_bias_level() needs to be called

import numpy as np
from datetime import date
from gemini_instruments.gmos import lookup
# ------------------------------------------------------------------------------


def get_bias_level(adinput=None, estimate=True):
    # Temporarily only do this for GMOS data. It would be better if we could do
    # checks on oberving band / type (e.g., OPTICAL / IR) and have this
    # function call an appropriate function. This is in place due to the call
    # to the function from the primitives_qa module - MS 2014-05-14 see
    # Trac #683
    try:  # the header keyword gets deleted when trimming, so it is there?
        adinput.hdr[adinput._keyword_for('overscan_section')]
    except KeyError:  # no, it isn't!
        estimate = True  # so we have to estimate
    __ALLOWED_TYPES__ = ["GMOS"]
    if not set(__ALLOWED_TYPES__).issubset(adinput.tags):
        msg = "{}.{} only works for {} data".format(__name__,
                                                    "get_bias_level",
                                                    __ALLOWED_TYPES__)
        raise NotImplementedError(msg)
    elif estimate:
        bias_level = _get_bias_level_estimate(adinput=adinput)
    else:
        bias_level = _get_bias_level(adinput=adinput)
    return bias_level


def _get_bias_level(adinput=None):
    """
    Determine the bias level value from the science extensions of the input
    AstroData object. The bias level is equal to the median of the overscan
    region. A float or list of floats is returned, depdending on whether the
    input is a single-extension slice.

    Parameters
    ----------
    adinput: AstroDataGmos
        The thing we want to know the bias level of

    Returns
    -------
    float/list
        the bias level(s)
    """
    # Get the overscan section value using the appropriate descriptor
    overscans = adinput.overscan_section()
    if not isinstance(overscans, list):
        overscans = [overscans]

    bias_level = []

    # Only measure if *all* extensions return an overscan section
    if all(ov is not None for ov in overscans):
        # The type of CCD determines the number of contaminated columns in the
        # overscan region. Get the pretty detector name value using the
        # appropriate descriptor.
        detector_name = adinput.detector_name(pretty=True)
        if detector_name == "EEV":
            nbiascontam = 4
        elif detector_name == "e2vDD":
            nbiascontam = 5
        elif detector_name.startswith("Hamamatsu"):
            nbiascontam = 4
        else:
            nbiascontam = 4

        # adinput is always iterable (even if _single); make sure overscans is
        for ext, overscan_section in zip(adinput, overscans):
            # Turn tuple into list to make mutable
            osec = list(overscan_section)
            # Don't include columns at edges
            if osec[0] == 0:
                # Overscan region is on the left
                osec[1] -= nbiascontam
                osec[0] += 1
            else:
                # Overscan region is on the right
                osec[0] += nbiascontam
                osec[1] -= 1

            # Extract overscan data. In numpy arrays, y indices come first.
            overscan_data = ext.data[osec[2]:osec[3], osec[0]:osec[1]]
            bias_level.append(np.median(overscan_data))

        # Turn single-element list into a value if sent a single-extension slice
        if adinput.is_single:
            bias_level = bias_level[0]
    else:
        bias_level = _get_bias_level_estimate(adinput=adinput)

    return bias_level


def _get_bias_level_estimate(adinput=None):
    """
    Determine an estiamte of the bias level value from GMOS data.

    Parameters
    ----------
    adinput: AstroDataGmos
        Image/slice we want to determine the bias level of

    Returns
    -------
    float/list
        Estimate of the bias level
    """
    # We use OVERSCAN if it exists, otherwise use RAWBIAS if it exists,
    # otherwise use the LUT value. To avoid slow slicing later, get the LUT
    # values now, even if we're not going to need them
    ut_date = adinput.ut_date()
    if ut_date >= date(2023, 12, 14):
        bias_dict = lookup.gmosampsBias
    elif ut_date >= date(2017, 2, 24):
        bias_dict = lookup.gmosampsBiasBefore20231214
    elif ut_date >= date(2015, 8, 26):
        bias_dict = lookup.gmosampsBiasBefore20170224
    elif ut_date >= date(2006, 8, 31):
        bias_dict = lookup.gmosampsBiasBefore20150826
    else:
        bias_dict = lookup.gmosampsBiasBefore20060831

    read_speed_setting = adinput.read_speed_setting()
    gain_setting = adinput.gain_setting()
    # This may be a list
    ampname = adinput.array_name()

    # Create appropriate object (list/value) of LUT bias level(s)
    if isinstance(ampname, list):
        lut_bias = [bias_dict.get((read_speed_setting, gain_setting, a))
                for a in ampname]
    else:
        lut_bias = bias_dict.get((read_speed_setting, gain_setting, ampname))

    # Get the overscan value and the raw bias level from the extension headers
    overscan_values = adinput.hdr.get('OVERSCAN')
    raw_bias_levels = adinput.hdr.get('RAWBIAS')

    try:
        bias_level = [o if o is not None else
                      r if r is not None else b
                      for o,r,b in zip(overscan_values, raw_bias_levels,
                                       lut_bias)]
    except TypeError:
        bias_level = overscan_values if overscan_values is not None else \
            raw_bias_levels if raw_bias_levels is not None else \
            lut_bias

    return bias_level
