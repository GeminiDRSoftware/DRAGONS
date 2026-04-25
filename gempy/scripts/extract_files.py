#!/usr/bin/env python
#
#                                                                       DRAGONS
#                                                                 gempy.scripts
#                                                                    fixheader.py
# -----------------------------------------------------------------------------
"""
This command will extract files that have been embedded within the FITS file
using the gempy embedded_files module. This is typically used to capture PDF
or other figures generated during data reduction and store them in the
reduced data fits file.
"""
from argparse import ArgumentParser
import textwrap
import sys

import astrodata
from gempy import __version__

from gempy.adlibrary.embedded_files import list_files, extract_files

def main(args=None):
    parser = ArgumentParser(
        description=f"Embedded file extractor, v{__version__}",
        epilog=textwrap.dedent(__doc__))
    parser.add_argument("-v", "--version", action="version",
                        version=f"v{__version__}")
    parser.add_argument('--list', help="List embedded files, "
                                     "as opposed to extracting them to disk",
                        action='store_true')
    parser.add_argument('filename', help='Fits file to extract from')

    args = parser.parse_args(args)

    ad = astrodata.open(args.filename)

    if args.list:
        for d in list_files(ad, fullinfo=True):
            print(f"{d['filename']}: {d['Content-Type']}, {d['size']} bytes")
    else:
        for i in extract_files(ad):
            print(f"Exracting: {i}")

if __name__ == '__main__':
    sys.exit(main())