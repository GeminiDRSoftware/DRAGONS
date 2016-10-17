#!/usr/bin/env python
#
#                                                                     QAP Gemini
#
#                                                               mosaicFactory.py
#                                                               Kenneth Anderson
#                                                                        07-2013
#                                                          <kanderso@gemini.edu>
# ------------------------------------------------------------------------------

# $Id: mosaicFactory.py 4197 2013-07-09 18:42:54Z kanderson $
# ------------------------------------------------------------------------------
__version__      = '$Revision: 4197 $'[11:-2]
__version_date__ = '$Date: 2013-07-09 14:42:54 -0400 (Tue, 09 Jul 2013) $'[7:-2]
__author__       = "k.r. anderson, <kanderso@gemini.edu>"
#__mods__  : Imports edits to gempy location
# ------------------------------------------------------------------------------
# This module provides the MosaicFactory class and command line interface to 
# receive a list of files to process with gemini_python mosaic package,currently
# under gemini_python/trunk/devel/mosaic. The mosaic package currently supports
# GMOS and GSAOI images.
#
# The MosaicFactory class provides one (1) public method, auto_mosaic() is all 
# the caller needs to call on a list of input images. Command line users  may
# pass one (1) or more FITS files to the command line. mosaicFactory will process
# all input files. Efficient use of this module will see a list of input images
# passed, avoiding re-importing AstroData and Mosaic classes. 
#
# Eg.,
# $ infile=`ls *.fits`
# $ mosaicFactory.py $infiles
#
# usage: mosaicFactory [-h] [-v] [infiles [infiles ...]]
#
# Auto mosaic builder.
#
# positional arguments:
#   infiles        infile1 [infile2 ...]
#
# optional arguments:
#   -h, --help     show this help message and exit
#   -v, --version  show program's version number and exit
#
# Mosaic-ed FITS  images are written to cwd(), like
#
# inputImage.fits --> inputImage_mosaic.fits
# ------------------------------------------------------------------------------
#__version__ = "0.1" # Using svn revision

import os
import sys

from datetime import datetime
from argparse import ArgumentParser

from astrodata import AstroData

# ------------------------------------------------------------------------------
def ptime():
    """
    parameters: <void>
    return:     <string>, ISO 8601 'now' timestamp
    """
    return datetime.isoformat(datetime.now())


def buildNewParser(version=__version__):
    """
    parameters: <string>, defaulted optional version.
    return:     <instance>, ArgumentParser instance
    """
    parser = ArgumentParser(description="Auto mosaic builder.",
                            prog="mosaicFactory")
    parser.add_argument("-v", "--version", action="version",
                        version="%(prog)s @r" + version)
    parser.add_argument('infiles', nargs='*', default= '',
                        help="infile1 [infile2 ...]")
    return parser


def handleCLargs():
    """
    parameters: None
    return:     <instance>, Namespace instance.
    """
    parser = buildNewParser()
    args = parser.parse_args()
    return args


class MosaicFactory(object):

    def __init__(self):
        self._set_mosaic_environ()


    def auto_mosaic(self, infiles):
        """
        parameters: <list>, list of input filenames
        return:     <void>
        """
        for in_fits in infiles:
            out_fits = self._set_out_fits(in_fits)
                
            print ptime(),"\tWorking on ...", os.path.split(in_fits)[1]
            ad = AstroData(in_fits)
            
            print ptime(),"\tAstroData object built"
            print ptime(),"\tWorking on type:", self._get_adtype(ad.types)
            print ptime(),"\tConstructing MosaicAD instance ..."
            mos = self.MosaicAD(ad, mosaic_ad_function=self.gemini_mosaic_function)

            print ptime(),"\tMaking mosaic ..."
            mos.mosaic_image_data()

            print ptime(), "\tConverting data ..."
            adout = mos.as_astrodata()

            print ptime(),"\tWriting file ..."
            adout.write(out_fits)

            print ptime(),"\tMosaic fits image written:", out_fits
        return


    def _get_adtype(self, tlist):
        """
        parameters: <list>,   list of strings, AstroData types
        return:     <string>, containing type incl *_IMAGE
        """
        item = None
        err  = "File does not appear to be a supported type."
        for item in tlist:
            if "_IMAGE" in item:
                break
        if not item:
            raise TypeError(err)
        return item


    def _set_mosaic_environ(self):
        """
        parameters: <void>
        return:     <void>
        """
        try:
            from gempy.adlibrary.mosaicAD import MosaicAD
            from gempy.gemini.gemMosaicFunction import gemini_mosaic_function
        except ImportError:
            self._add_mosaic_path()
            from mosaicAD import MosaicAD
            from gemMosaicFunction import gemini_mosaic_function

        self.MosaicAD = MosaicAD
        self.gemini_mosaic_function = gemini_mosaic_function

        return


    def _add_mosaic_path(self):
        """Add the current configuration path for
        the Mosaic package (gemini_python/trunk/devel/mosaic)
        to sys.path when ImportError is raised on Mosaic imports.

        parameters: <void>
        return:     <void>
        """
        _mos_path = None
        _mos_msg = "gemini_python installation not found."
        for ppath in sys.path:
            if 'gemini_python' in ppath:
                _mos_path = os.path.join(ppath, 'devel','mosaic')
                sys.path.append(_mos_path)
                break
            else: continue
        try:                  
            assert(_mos_path)
            # An AssertionError here should never happen.
            # This implies that astrodata is not available, 
            # which means the astrodata import would fail upfront.
        except AssertionError:       
            raise EnvironmentError(_mos_msg)
        return


    def _set_out_fits(self, in_fits):
        """
        parameters: <string>, filename
        return:     <string>, output filename
        """
        head, tail = os.path.split(in_fits)
        fnam, fext = os.path.splitext(tail)
        out_fits   = fnam + "_mosaic" + fext
        return out_fits



if __name__ == '__main__':
    args = handleCLargs()
    mos_factory = MosaicFactory()
    mos_factory.auto_mosaic(args.infiles)
