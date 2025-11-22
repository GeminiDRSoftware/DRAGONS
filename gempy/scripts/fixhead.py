#!/usr/bin/env python
#
#                                                                       DRAGONS
#                                                                 gempy.scripts
#                                                                    fixhead.py
# -----------------------------------------------------------------------------

import sys
from argparse import ArgumentParser

import astrodata, gemini_instruments
from gempy import __version__


DTYPES = {'int': int, 'float': float, 'str': str}


def main(args=None):
    parser = ArgumentParser(
        description=f"Primitive parameter display, v{__version__}")
    parser.add_argument("-v", "--version", action="version",
                        version=f"v{__version__}")
    parser.add_argument('filename', help="filename or filename:[extension]")
    parser.add_argument('keyword', help="keyword")
    parser.add_argument('value', help="new value")
    parser.add_argument("-d", "--dtype", help="data type")
    parser.add_argument("-a", "--add", action="store_true",
                        help="add new keyword")
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
            ad[index].hdr[keyword] = coerce(value, dtype)
            print(f"Updating {keyword}={value} in extension {ad[index].id}")
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
