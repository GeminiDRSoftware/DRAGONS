#!/usr/bin/env python
#
#                                                                  gemini_python
#
#                                                                  autoMosaic.py
# ------------------------------------------------------------------------------
"""
This module provides the AutoMosaic class and command line interface to
receive a list of files to process with gemini_python mosaic package, currently
under gemini_python/trunk/devel/mosaic. The mosaic package currently supports
GMOS and GSAOI images.

The AutoMosaic class provides one (1) public method, auto_mosaic() is all
the caller needs to call on a list of input images. Command line users  may
pass one (1) or more FITS files to the command line. mosaicFactory will process
all input files. Efficient use of this module will see a list of input images
passed, avoiding re-importing astrodata and the MosaicAD class.

"""
from __future__ import print_function
from builtins import object
# ------------------------------------------------------------------------------
__version__ = "2.0.0 (beta)"
# ------------------------------------------------------------------------------
import os
import sys

import astrodata
import gemini_instruments

from gempy.mosaic.mosaicAD import MosaicAD
from gempy.mosaic.gemMosaicFunction import gemini_mosaic_function

from gempy.utils import logutils

# ------------------------------------------------------------------------------
log = logutils.get_logger(__name__)
# ------------------------------------------------------------------------------
class AutoMosaic(object):
    def __init__(self, args):
        """
        Parameters:
        -----------
        args: list of input filenames
        type: <list>

        Returns:
        --------
        <void>

        """
        self.files = args.infiles
        self.img = args.img
        self.tile = args.tile

    def auto_mosaic(self):
        for in_fits in self.files:
            out_fits = self._set_out_fits(in_fits)
            log.stdinfo("{}, {}".format("AutoMosaic", __version__))
            log.stdinfo("\tWorking on {}".format(os.path.split(in_fits)[1]))
            ad = astrodata.open(in_fits)

            log.stdinfo("\tAstroData object built")
            log.stdinfo("\tWorking on type: {}".format(self._check_tags(ad.tags)))
            log.stdinfo("\tConstructing MosaicAD instance ...")
            mos = MosaicAD(ad, mosaic_ad_function=gemini_mosaic_function)

            log.stdinfo("\tMaking mosaic, converting data ...")
            adout = mos.as_astrodata(tile=self.tile, doimg=self.img)

            log.stdinfo("\tWriting file ...")
            adout.write(out_fits)
            log.stdinfo("\tMosaic fits image written: {}".format(out_fits))

        return

    def _check_tags(self, tlist):
        """
        Parameters:
        ----------
        tlist: set of astrodata tags
        type: <Set> or <list>

        Returns:
        -------
        item: compatible tags, 'GMOS IMAGE' or 'GSAOI IMAGE'
        type: <str>

        Raises:
        ------
        TypeError on incompatible tag sets.

        """
        item = None
        err  = "File does not appear to be a supported type."

        if "GMOS" in tlist and "IMAGE" in tlist:
            item = 'GMOS IMAGE'
        elif "GSAOI" in tlist and "IMAGE" in tlist:
            item = 'GSAOI IMAGE'

        if not item:
            raise TypeError(err)

        return item

    def _set_out_fits(self, in_fits):
        """
        Parameters:
        ----------
        in_fits: input filename
        type:    <str>

        Returns:
        -------
        out_fits: output filename
        type: <str>

        """
        head, tail = os.path.split(in_fits)
        fnam, fext = os.path.splitext(tail)
        out_fits   = fnam + "_mosaic" + fext
        return out_fits
