# Testing support functions

import pyfits
from astrodata import AstroData

def dummy_ad():

    phu = pyfits.PrimaryHDU(data=None)
    sci = pyfits.ImageHDU(data=None)
    sci.header['EXTNAME'] = 'SCI'
    sci.header['EXTVER'] = 1
    hdulist = pyfits.HDUList([phu, sci])

    # Note: even if I specify the badly-documented 'exts' parameter here,
    # AstroData seems not to set its _extver attribute and complains about
    # it later if one tries to specify ad['SCI'].

    return AstroData(hdulist)


def get_ad_sublist(adlist, names):

    """
    Select a sublist of AstroData instances from the input list using a list
    of filename strings. Any filenames that don't exist in the AstroData list
    are just ignored.
    """

    outlist = []

    for ad in adlist:
        if ad.filename in names:
            outlist.append(ad)

    return outlist

def same_lists(first, second):

    """Do the 2 input lists contain equivalent objects?"""

    nobjs = len(first)
    if len(second) != nobjs:
        return False

    # This is a bit of a brute-force approach since I'm not sure whether
    # ExposureGroups will get ordered consistently by a sort operation if
    # their internal ordering varies without defining the appropriate
    # special methods, but no big deal. Equivalence *does* work where the
    # internal ordering of ExposureGroups varies.
    for obj1 in first:
        found = False
        for obj2 in second:
            if obj1 == obj2:
                found = True
                break
        if not found:
            return False

    return True

