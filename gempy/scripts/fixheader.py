#!/usr/bin/env python
#
#                                                                       DRAGONS
#                                                                 gempy.scripts
#                                                                    fixheader.py
# -----------------------------------------------------------------------------
"""
This command will fix the headers of FITS files by editing or adding a keyword
to the primary header or one or more of the extension headers. Without the
-a/--add option, the keyword must already exist: if it is present in the
primary header, it will be modified there; otherwise it will be edited in the
headers of the extensions where it is already present. With the -a/--add
option, the keyword will be added to the primary header by default: to add it
to a specific extension, append ':extid' (where 'extid' is usually identical
to the EXTVER) to the filename; to add it to all extensions, just use ':'
"""

import sys
from argparse import ArgumentParser
import textwrap

import astrodata, gemini_instruments
from gempy.adlibrary.fixheader import modify_header
from gempy import __version__


def main(args=None):
    parser = ArgumentParser(
        description=f"Header keyword editor, v{__version__}",
        epilog=textwrap.dedent(__doc__))
    parser.add_argument("-v", "--version", action="version",
                        version=f"v{__version__}")
    parser.add_argument('filename', help="filename or filename[:[extension]]")
    parser.add_argument('keyword', help="keyword to change")
    parser.add_argument('value', help="new value")
    parser.add_argument("-d", "--dtype", help="data type (int, float, str)")
    parser.add_argument("-a", "--add", action="store_true",
                        help="add (rather than replace) new keyword")
    args = parser.parse_args(args)
    if ":" in args.filename:
        filename, extid = args.filename.split(":")
    else:
        filename = args.filename
        extid = None
    update_header(filename, extid, args.keyword, args.value, args.add, args.dtype)


def update_header(filename, extid, keyword, value, add, dtype=None):
    ad = astrodata.open(filename)
    modify_header(ad, extid=extid, keyword=keyword, value=value, add=add, dtype=dtype)
    ad.write(overwrite=True)


if __name__ == '__main__':
    sys.exit(main())
