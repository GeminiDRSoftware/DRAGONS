#!/usr/bin/env python
#
#                                                                       DRAGONS
#                                                                 gempy.scripts
#                                                                   showpars.py
# -----------------------------------------------------------------------------

import sys
import textwrap
from argparse import ArgumentParser
from importlib import import_module

import astrodata
import gemini_instruments  # noqa
from gempy import __version__
from recipe_system.mappers import primitiveMapper

# -----------------------------------------------------------------------------


def main(args=None):
    parser = ArgumentParser(
        description=f"Primitive parameter display, v{__version__}")
    parser.add_argument("-v", "--version", action="version",
                        version=f"v{__version__}")
    parser.add_argument('filename', help="filename")
    parser.add_argument('primitive', help="primitive name")
    parser.add_argument("-d", "--doc", action="store_true",
                        help="show the full docstring")
    parser.add_argument('--adpkg', help='Name of the astrodata instrument '
                        'package to use if not gemini_instruments')
    parser.add_argument('--drpkg', help='Name of the DRAGONS instrument '
                        'package to use if not geminidr')
    args = parser.parse_args(args)
    pobj, tags = get_pars(args.filename, adpkg=args.adpkg, drpkg=args.drpkg)
    return showpars(pobj, args.primitive, tags, args.doc)


def get_pars(filename, adpkg=None, drpkg=None):
    if adpkg is not None:
        import_module(adpkg)

    ad = astrodata.from_file(filename)

    dtags = set(list(ad.tags)[:])
    instpkg = ad.instrument(generic=True).lower()
    if drpkg is None:
        pm = primitiveMapper.PrimitiveMapper(dtags, instpkg)
    else:
        pm = primitiveMapper.PrimitiveMapper(dtags, instpkg, drpkg=drpkg)
    pclass = pm.get_applicable_primitives()
    pobj = pclass([ad])
    return pobj, dtags


def showpars(pobj, primname, tags, show_docstring):
    print(f"Dataset tagged as {tags}\n")
    if primname not in pobj.params:
        raise KeyError(f"{primname} doesn't exist for "
                       "this data type.")

    print(f"Settable parameters on '{primname}':")
    print("=" * 40)
    print(f"{'Name':20s} {'Current setting':20s} Description\n")

    params = pobj.params[primname]
    for k, v in params.items():
        if not k.startswith("debug"):
            print(f"{k:20s} {v!r:20s} {params.doc(k)}")

    if show_docstring:
        print(f"\nDocstring for '{primname}':")
        print("=" * 40)
        print(textwrap.dedent(getattr(pobj, primname).__doc__))


if __name__ == '__main__':
    sys.exit(main())
