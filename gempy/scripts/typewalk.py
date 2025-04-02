#!/usr/bin/env python
#
#                                                                  gemini_python
#
#                                                                    typewalk.py
# ------------------------------------------------------------------------------

from astrodata import version
__version__ = version()
# ------------------------------------------------------------------------------
desc = """
Description:
  typewalk examines files in a directory or directory tree and reports the
  types and status values through the AstroDataType classification scheme.
  Files are selected and reported through a regular expression mask, which by
  default, finds all ".fits" and ".FITS" files. Users can change this mask
  with the -f --filemask option.

  By default, typewalk will recurse all subdirectories under the current
  directory. Users may specify an explicit directory with the -d --dir option.

  A user may request that an output file is written when AstroDataType
  qualifiers are passed by the --types option. An output file is specified
  through the -o --out option. Output files are formatted so they may
  be passed directly to the reduce command line via that applications
  'at-file' (@file) facility. See reduce help for more on that facility.

  Users may select type matching logic with the --or switch. By default,
  qualifying logic is AND. I.e. the logic specifies that *all* types must be
  present (x AND y); --or specifies that ANY types, enumerated with --types,
  may be present (x OR y). --or is only effective when --types is used.

  For example, find all gmos images from Cerro Pachon in the top level
  directory and write out the matching files, then run reduce on them,

    $ typewalk -n --tags SOUTH GMOS IMAGE --out gmos_images_south
    $ reduce @gmos_images_south

  This will also report match results to stdout, colourized if requested (-c).
"""
# ------------------------------------------------------------------------------
import os
import re
import sys
import time

from importlib import import_module

import astrodata
import gemini_instruments

from astrodata import AstroDataError

# ------------------------------------------------------------------------------
batchno = 100


def typewalk_argparser():

    from argparse import ArgumentParser
    from argparse import RawDescriptionHelpFormatter

    parser = ArgumentParser(description=desc, prog='typewalk',
                            formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument("-b", "--batch", dest="batchnum", default=100,
                      help="In shallow walk mode, number of files to process "
                      "at a time in the current directory. Controls behavior "
                      "in large data directories. Default = 100.")

    parser.add_argument("-d", "--dir", dest="twdir", default=os.getcwd(),
                      help="Walk this directory and report types. "
                      "default is cwd.")

    parser.add_argument("-f", "--filemask", dest="filemask", default=None,
                      help="Show files matching regex <FILEMASK>. Default "
                        "is all .fits and .FITS files.")

    parser.add_argument("-n", "--norecurse", dest="stayTop", action="store_true",
                      help="Do not recurse subdirectories.")

    parser.add_argument("--or", dest="or_logic", action="store_true",
                        help= "Use OR logic on 'types' criteria. If not "
                        "specified, matching logic is AND (See --types). "
                        "Eg., --or --types SOUTH GMOS IMAGE will report "
                        "datasets that are one of SOUTH *OR* GMOS *OR* IMAGE.")

    parser.add_argument("-o", "--out", dest="outfile", default=None,
                      help= "Write reported files to this file. "
                        "Effective only with --tags option.")

    parser.add_argument("--tags", dest="tags", nargs='+', default='all',
                      help= "Find datasets that match only these tag criteria. "
                        "Eg., --tags SOUTH GMOS IMAGE will report datasets "
                        "that are all tagged SOUTH *and* GMOS *and* IMAGE.")

    parser.add_argument("--xtags", dest="xtags", nargs='+', default=None,
                        help="Exclude <xtags> from reporting.")

    parser.add_argument("--adpkg", dest="adpkg", nargs=1, action="store",
                        required=False,
                        help='Name of the astrodata instrument package to use'
                             'if not gemini_instruments')

    return parser.parse_args()


def path2list(path):
    if path[-1] == os.sep:
        path = path[:-1]
    upath = path
    palist = []
    while True:
        upath, tail = os.path.split(upath)
        if tail == "":
            break
        else:
            palist.insert(0, tail)
    return palist


def shallow_walk(directory):
    global batchno
    ld = os.listdir(directory)
    root = directory
    dirn = []
    files = []

    for li in ld:
        if os.path.isdir(li):
            dirn.append(li)
        else:
            files.append(li)

        if len(files) > batchno:
            yield (root, [], files)
            files = []
    yield (root, [], files)


def generate_outfile(outfile, olist, match_type, logical_or, xtypes):
    with open(outfile, "w") as ofile:
        ofile.write("# Auto-generated by typewalk, v" + __version__ + "\n")
        ofile.write("# Written: " + time.ctime(time.time()) + "\n")
        ofile.write("# Qualifying types: " + "  ".join(match_type) + "\n")
        if logical_or:
            ofile.write("# Qualifying logic: OR\n")
        else:
            ofile.write("# Qualifying logic: AND\n")
        if xtypes:
            ofile.write("# Excluded types:   " + " ".join(xtypes) + "\n")
        ofile.write("# -----------------------\n")
        for ffile in olist:
            ofile.write(ffile)
            ofile.write("\n")
    return


class Faces:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class DataSpider:
    """
    DataSpider() providing one (1) method, typewalk,  that will walk a
    directory and report types via AstroData.

    """
    def typewalk(self, directory=os.getcwd(), only=None, filemask=None,
                 or_logic=False, outfile=None, stayTop=False, batchnum=100,
                 xtypes=None, adpkg=None):
        """
        Recursively walk <directory> and put type information to stdout

        """
        if adpkg is not None:
            import_module(adpkg)

        directory = os.path.abspath(directory)

        # This accumulates files that match --types type if --out is
        # specified.
        outfile_list = []
        batchno = batchnum

        if stayTop == True:
            walkfunc = shallow_walk
        else:
            walkfunc = os.walk

        for root, dirn, files in walkfunc(directory):

            fullroot = os.path.abspath(root)

            if root == ".":
                rootln = "\n{}directory: {}. ({})".format(
                    Faces.CYAN, Faces.END, fullroot)

            else:
                rootln = "\n{}directory: {} {}".format(
                    Faces.CYAN, Faces.END, root)

            firstfile = True
            files.sort()

            for tfile in files:

                if filemask is None:
                    mask = r".*?\.(fits|FITS|fz)$"
                else:
                    mask = filemask

                try:
                    matched = re.match(mask, tfile)
                except:
                    print("BAD FILEMASK (must be a valid regexp):", mask)
                    return str(sys.exc_info()[1])

                if re.match(mask, tfile):

                    fname = os.path.join(root, tfile)

                    try:
                        fl = astrodata.from_file(fname)
                        dtypes = list(fl.tags)
                    except AttributeError:
                        print("     Bad headers in file: {}".format(tfile))
                        continue
                    except OSError:
                        print("     Could not open file: {}".format(fname))
                        continue
                    except ValueError as err:
                        print("     Failed to open: {}, {}".format(fname, str(err)))
                        continue
                    except AstroDataError:
                        print("AstroData failed to open file: {}".format(fname))
                        continue

                    # exclude if dtypes has any xtypes
                    if xtypes:
                        try:
                            assert set(dtypes).intersection(set(xtypes))
                            continue
                        except AssertionError:
                            pass

                    # Here we are looking to match *all* caller types.
                    # Logical AND, not OR.
                    if only == "all":
                        found = True
                    else:
                        found = False
                        if or_logic:
                            try:
                                assert(set(only).intersection(set(dtypes)))
                                found = True
                                if outfile:
                                    outfile_list.append(fname)
                            except AssertionError:
                                pass
                        else:
                            if set(only).issubset(dtypes):
                                found = True
                                if outfile:
                                    outfile_list.append(fname)

                    if not found:
                        continue

                    if firstfile:
                        print(rootln)

                    firstfile = False
                    # PRINTING OUT THE FILE AND TYPE INFO
                    indent = 5
                    pwid = 40
                    fwid = pwid - indent
                    while len(tfile) >= (fwid - 1):
                        print("     {}{}{}".format(Faces.BLUE, tfile, Faces.END))
                        tfile = ""

                    if len(tfile) > 0:
                        prlin = "     {} ".format(tfile)
                        prlincolor = "     {}{}{} ".format(Faces.BLUE, tfile,
                                                           Faces.END)
                    else:
                        prlin = "     "
                        prlincolor = "     "

                    empty = " " * indent + "." * fwid
                    fwid = pwid + indent
                    lp = len(prlin)
                    nsp = pwid - (lp % pwid)
                    print(prlincolor+("."*nsp)+"{}".format(Faces.END), end=' ')
                    tstr = ""
                    astr = ""
                    dtypes.sort()
                    for dtype in dtypes:
                        if (dtype is not None):
                            newtype = "({}) ".format(dtype)
                        else:
                            newtype = "(Unknown) "

                        astr += newtype

                    print("{}{}{}".format(Faces.RED, astr, Faces.END))

        if outfile and outfile_list:
            generate_outfile(outfile, outfile_list, only, or_logic, xtypes)
        return


def main(options):

    if options.adpkg is not None:
        options.adpkg = options.adpkg[0]

    # remove current working directory from PYTHONPATH to speed up import in
    # gigantic data directories
    curpath = os.getcwd()

    # @@REVIEW Note: This is here because it's very confusing when someone
    # runs a script IN the package itself.
    if curpath in sys.path:
        sys.path.remove(curpath)

    print(options.adpkg)
    # Gemini Specific class code
    dt = DataSpider()
    try:
        dt.typewalk(directory=options.twdir,
                    only=options.tags,
                    or_logic=options.or_logic,
                    outfile=options.outfile,
                    filemask=options.filemask,
                    stayTop=options.stayTop,
                    batchnum=int(options.batchnum)-1,
                    xtypes=options.xtags,
                    adpkg=options.adpkg,
        )
        print("Done DataSpider.typewalk(..)")
    except KeyboardInterrupt:
        print("Interrupted by Control-C")
    return


if __name__ == '__main__':
    args = typewalk_argparser()
    sys.exit(main(args))
