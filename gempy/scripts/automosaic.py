#!/usr/bin/env python
#
#                                                                  gemini_python
#
#                                                                  automosaic.py
# ------------------------------------------------------------------------------
"""
usage: automosaic [-h] [-i] [-t] [-v] [infiles [infiles ...]]

Auto mosaic builder.

positional arguments:
  infiles        infile1 [infile2 ...]

optional arguments:
  -h, --help     show this help message and exit
  -i, --image    Tranform image (SCI) data only.
  -t, --tile     Tile data only.
  -v, --version  show program's version number and exit

Mosaicked FITS images are written to cwd(), like

inputImage.fits --> inputImage_mosaic.fits

"""
from builtins import object
# ------------------------------------------------------------------------------
__version__ = "2.0.0 (beta)"
# ------------------------------------------------------------------------------
import os
import sys
from argparse import ArgumentParser

from gempy.mosaic.autoMosaic import AutoMosaic

# ------------------------------------------------------------------------------
def buildNewParser(version=__version__):
    """
    parameters: <string>, defaulted optional version.
    return:     <instance>, ArgumentParser instance

    """
    parser = ArgumentParser(description="Auto mosaic builder.",
                            prog="automosaic")
    parser.add_argument("-i", "--image", dest='img', action="store_true",
                        help="Tranform image (SCI) data only.")
    parser.add_argument("-t", "--tile", dest='tile', action="store_true",
                        help="Tile data only.")
    parser.add_argument("-v", "--version", action="version",
                        version="mosaicFactory, v{}".format(version))
    parser.add_argument('infiles', nargs='*', default= '',
                        help="infile1 [infile2 ...]")
    return parser

def handleCLargs():
    """
    Returns:
    -------
    args: Namespace instance
    type: <instance>

    """
    parser = buildNewParser()
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = handleCLargs()
    mos_factory = AutoMosaic(args)
    sys.exit(mos_factory.auto_mosaic())
