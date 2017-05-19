#!/usr/bin/env python
#
#                                                                  gemini_python
#
#                                                                    showpars.py
# ------------------------------------------------------------------------------
import sys
from argparse import ArgumentParser

import astrodata
import gemini_instruments

from recipe_system.mappers import primitiveMapper
# ------------------------------------------------------------------------------
__version__ = '2.0.0 (beta)'
# ------------------------------------------------------------------------------
def buildArgs():
    parser = ArgumentParser(
        description="Primitive parameter display, v{}".format(__version__))
    parser.add_argument(
        "-v", "--version", action="version", version="v{}".format(__version__))
    parser.add_argument('filen', nargs=1, help="filename")
    parser.add_argument('primn', nargs=1, help="primitive name")
    args = parser.parse_args()
    return args

def get_pars(filename):
    ad = astrodata.open(filename)
    pm = primitiveMapper.PrimitiveMapper([ad])
    p = pm.get_applicable_primitives()
    return p.parameters

def showpars(pobj, primname):
    """

    """
    for i in dir(pobj):
        if i.startswith("_"):
            continue

    pars = getattr(pobj, primname)
    print
    print "Settable parameters on primitive '{}':".format(primname)
    print "="*32
    print " Name       Current setting"
    print
    for k,v in pars.items():
        print "{} :         {}".format(k,v)

    print
    return

if __name__ == '__main__':
    args = buildArgs()
    fname = args.filen[0]
    pname = args.primn[0]
    paro = get_pars(fname)
    sys.exit(showpars(paro, pname))
