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
from gempy import __version__


DTYPES = {'int': int, 'float': float, 'str': str}


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

    def coerce(value, dtype):
        if dtype is not None:
            if isinstance(dtype, str):
                dtype = DTYPES[dtype]
            return dtype(value)
        try:
            v = int(value)
        except ValueError:
            try:
                v = float(value)
            except ValueError:
                return value
            else:
                return v
        return v

    ad = astrodata.open(filename)
    if extid is None:  # PHU
        if keyword in ad.phu or add:
            if keyword in ad.phu and dtype is None:
                dtype = type(ad.phu[keyword])
            print(f"Updating {keyword}={value} in PHU")
            ad.phu[keyword] = coerce(value, dtype)
        elif ad.hdr.get(keyword).count(None) == 0:  # try all extensions
            extid = ''
        else:
            raise KeyError(f"{keyword} is not present in all extensions")
    elif extid:  # one header only
        try:
            index = [ext.id for ext in ad].index(int(extid))
        except ValueError:
            raise ValueError(f"{extid} not a valid extension id in {filename}")
        if keyword in ad[index].hdr or add:
            if keyword in ad[index].hdr and dtype is None:
                dtype = type(ad[index].hdr[keyword])
            print(f"Updating {keyword}={ad[index].hdr[keyword]} -> {value} "
                  f"in extension {ad[index].id}")
            ad[index].hdr[keyword] = coerce(value, dtype)
        else:
            raise KeyError(f"{keyword} not found in {filename}:{extid} and "
                           "'--add' not selected")
    if extid == '':  # all headers
        already_values = ad.hdr.get(keyword)
        if already_values.count(None) > 0 and not add:
            print(f"{keyword} exists only in extensions "+
                  ", ".join([str(ext.id) for ext in ad if keyword in ext.hdr]))
        if not add:
            for ext, already in zip(ad, already_values):
                if already is not None:
                    print(f"Updating {keyword}={ext.hdr[keyword]} -> {value} "
                          f"in extension {ext.id}")
                    ext.hdr[keyword] = coerce(value, type(ext.hdr[keyword])
                                              if dtype is None else dtype)
        else:
            if dtype is not None:
                ad.hdr[keyword] = coerce(value, dtype)
            else:
                dtypes = set([type(ext.hdr[keyword]) for ext in ad]) - {type(None)}
                if len(dtypes) > 1:
                    raise ValueError(f"{keyword} does not have a unique datatype; "
                                     "please specify on the command line")
                ad.hdr[keyword] = coerce(value, dtypes.pop())
            print(f"Updating {keyword}={value} in all extensions")
    ad.write(overwrite=True)


if __name__ == '__main__':
    sys.exit(main())
