#!/usr/bin/env python
#
#                                                                        DRAGONS
#                                                                  gempy.scripts
#                                                                    showpars.py
# ------------------------------------------------------------------------------

import sys
from argparse import ArgumentParser

import astrodata
import gemini_instruments

from gempy import __version__

from recipe_system.mappers import primitiveMapper
# ------------------------------------------------------------------------------

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
    dtags = set(list(ad.tags)[:])
    instpkg = ad.instrument(generic=True).lower()
    pm = primitiveMapper.PrimitiveMapper(dtags, instpkg)
    pclass = pm.get_applicable_primitives()
    p = pclass([ad])
    return p.params, dtags

def showpars(pobj, primname, tags):
    for i in dir(pobj):
        if i.startswith("_"):
            continue

    pars = pobj[primname]
    print("Dataset tagged as {}".format(tags))
    print("Settable parameters on '{}':".format(primname))
    print("="*40)
    print(" Name\t\t\tCurrent setting")
    print()
    for k, v in pars.items():
        if not k.startswith("debug"):
            print("{:20s} {:20s} {}".format(k, repr(v), pars.doc(k)))
    print()
    return

if __name__ == '__main__':
    args = buildArgs()
    fname = args.filen[0]
    pname = args.primn[0]
    paro, tags = get_pars(fname)
    sys.exit(showpars(paro, pname, tags))
