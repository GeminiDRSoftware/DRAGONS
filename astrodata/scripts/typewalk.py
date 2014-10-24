#!/usr/bin/env python
#
#                                                                  gemini_python
#
#                                                                    typewalk.py
# ------------------------------------------------------------------------------
# $Id$
# ------------------------------------------------------------------------------
__version__      = '$Revision$'[11:-2]
__version_date__ = '$Date$'[7:-2]
# ------------------------------------------------------------------------------
# this script was developed to exercise the AstroDataType class but now serves a
# general purpose and as  demo for AstroData... see options documentation.
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

    $ typewalk -n --types GEMINI_SOUTH GMOS_IMAGE --out gmos_images_south
    $ reduce @gmos_images_south

  This will also report match results to stdout, colourized if requested (-c).
"""
# ------------------------------------------------------------------------------
import os
import re
import sys
import time

from astrodata.AstroData import AstroData
from astrodata.Errors import AstroDataError

from astrodata.adutils import terminal

# ------------------------------------------------------------------------------
from astrodata.LocalCalibrationService import CalibrationService
from astrodata.CalibrationDefinitionLibrary import CalibrationDefinitionLibrary

batchno = 100

# ------------------------------------------------------------------------------
def typewalk_argparser():
    from argparse import ArgumentParser
    from argparse import RawDescriptionHelpFormatter

    parser = ArgumentParser(description=desc, prog='typewalk',
                            formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument("-b", "--batch", dest="batchnum", default=100,
                      help="In shallow walk mode, number of files to process "
                      "at a time in the current directory. Controls behavior "
                      "in large data directories. Default = 100.")

    parser.add_argument("--calibrations", dest="showCals", action="store_true",
                      help="Show local calibrations (NOT IMPLEMENTED).")

    parser.add_argument("-c", "--color", dest="usecolor", action="store_true",
                      help="Colorize display")

    parser.add_argument("-d", "--dir", dest="twdir", default=os.getcwd(),
                      help="Walk this directory and report types. "
                      "default is cwd.")

    parser.add_argument("-f", "--filemask", dest="filemask", default=None,
                      help="Show files matching regex <FILEMASK>. Default "
                        "is all .fits and .FITS files.")

    parser.add_argument("-i", "--info", dest="showinfo", action="store_true",
                      help="Show file meta information.")

    parser.add_argument("--keys", metavar="KEY", nargs='+', dest="phukeys",
                        default=None,
                        help= "Print keyword values for reported files."
                        "Eg., --keys TELESCOP OBJECT")

    parser.add_argument("-n", "--norecurse", dest="stayTop", action="store_true",
                      help="Do not recurse subdirectories.")

    parser.add_argument("--or", dest="or_logic", action="store_true",
                        help= "Use OR logic on 'types' criteria. If not "
                        "specified, matching logic is AND (See --types). "
                        "Eg., --or --types GEMINI_SOUTH GMOS_IMAGE will report "
                        "datasets that are either GEMINI_SOUTH *OR* GMOS_IMAGE.")

    parser.add_argument("-o", "--out", dest="outfile", default=None,
                      help= "Write reported files to this file. "
                        "Effective only with --types option.")

    parser.add_argument("--raise", dest="raiseExcept", action="store_true",
                      help="Raise descriptor exceptions.")

    parser.add_argument("--types", dest="types", nargs='+', default='all',
                      help= "Find datasets that match only these type criteria. "
                        "Eg., --types GEMINI_SOUTH GMOS_IMAGE will report datasets "
                        "that are both GEMINI_SOUTH *and* GMOS_IMAGE.")

    parser.add_argument("--status", dest="onlyStatus", action="store_true",
                      help="Report data processing status only.")

    parser.add_argument("--typology", dest="onlyTypology", action="store_true",
                      help="Report data typologies only.")

    parser.add_argument("--xcal", dest="xcal", action="store_true",
                        help="Exclude calibration ('CAL') types from reporting.")

    return parser.parse_args()

# ------------------------------------------------------------------------------
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

def generate_outfile(outfile, olist, match_type, logical_or, xcal):
    with open(outfile, "w") as ofile:
        ofile.write("# Auto-generated by typewalk, v" + __version__ + "\n")
        ofile.write("# Written: " + time.ctime(time.time()) + "\n")
        ofile.write("# Qualifying types: " + "  ".join(match_type) + "\n")
        if logical_or:
            ofile.write("# Qualifying logic: OR\n")
        else:
            ofile.write("# Qualifying logic: AND\n")
        if xcal:
            ofile.write("# Calibration (CAL) types not included.\n")
        ofile.write("# -----------------------\n")
        for ffile in olist:
            ofile.write(ffile)
            ofile.write("\n")
    return

# ------------------------------------------------------------------------------
class DataSpider(object):
    """
    DataSpider() providing one (1) method, typewalk,  that will walk a 
    directory and report types via AstroData.
    """
    def __init__(self, context=None):
        nocal_msg = ("\tLocal calibration service not implemented in this "
                     "version of astrodata")
        nocal_msg += ", GP-X1, r" + __version__
        nocal_msg += "\n\tCalibrations cannot be displayed at this time."

        self.contextType = context
        self.show_cals_off = False
        self.calDefLib = CalibrationDefinitionLibrary()

        # LocalCalibrationService Not Implemented
        try:
            self.calService = CalibrationService()
        except NotImplementedError, err:
            self.nocal_msg = "\t" + str(err) + "\n" + nocal_msg
            self.show_cals_off = True

    def typewalk(self, directory=os.getcwd(), only="all", or_logic=False,
                 pheads=None, showinfo=False, onlyStatus=False, outfile=None,
                 onlyTypology=False, filemask=None, showCals=False,
                 stayTop=False, raiseExcept=False, batchnum=100, xcal=False):
        """
        Recursively walk <directory> and put type information to stdout
        """
        directory = os.path.abspath(directory)

        # This accumulates files that match --types type if --out is
        # specified.
        outfile_list = []

        if showCals and self.show_cals_off:
            self._spool_nocal_msg()

        global batchno
        if batchnum:
            batchno = batchnum

        if raiseExcept:
            from astrodata.debugmodes import set_descriptor_throw
            set_descriptor_throw(True)

        if stayTop == True:
            walkfunc = shallow_walk
        else:
            walkfunc = os.walk

        for root, dirn, files in walkfunc(directory):
            if (".svn" not in root):
                ## CREATE THE LINE WRITTEN FOR EACH DIRECTORY RECURSED
                fullroot = os.path.abspath(root)
                if root == ".":
                    rootln = "\n${NORMAL}${BOLD}directory: ${NORMAL}. (" + \
                             fullroot + ")${NORMAL}"
                else:
                    rootln = "\n${NORMAL}${BOLD}directory: ${NORMAL}" + \
                             root + "${NORMAL}"

                firstfile = True
                for tfile in files:
                    if filemask is None:
                        mask = r".*?\.(fits|FITS)$"
                    else:
                        mask = filemask

                    try:
                        matched = re.match(mask, tfile)
                    except:
                        print "BAD FILEMASK (must be a valid regexp):", mask
                        return str(sys.exc_info()[1])

                    if (re.match(mask, tfile)) :
                        fname = os.path.join(root, tfile)
                        try:
                            fl = AstroData(fname)
                        except IOError:
                            print "Could not open file: %s" % fname
                            continue
                        except AstroDataError:
                            print "AstroData failed to open file: %s" % fname
                            continue

                        if (onlyTypology == onlyStatus):
                            dtypes = fl.types
                        elif (onlyTypology):
                            dtypes = fl.type()
                        elif (onlyStatus):
                            dtypes = fl.status()

                        # xcal indicates no reporting CAL types
                        if xcal:
                            if dtypes and 'CAL' in dtypes:
                                continue

                        # Here we are looking to match *all* caller types.
                        # Logical AND, not OR.
                        if dtypes:
                            found = False
                            if (only == "all"):
                                found = True
                            else:
                                found = False
                                if or_logic:
                                    for only_type in only:
                                        if only_type in dtypes:
                                            found = True
                                            if outfile:
                                                outfile_list.append(fname)
                                            break
                                else:
                                    if set(only).issubset(dtypes):
                                        found = True
                                        if outfile:
                                            outfile_list.append(fname)

                            if not found:
                                continue
                            if firstfile:
                                print rootln

                            firstfile = False
                            # PRINTING OUT THE FILE AND TYPE INFO
                            indent = 5
                            pwid = 40
                            fwid = pwid - indent
                            while len(tfile) >= (fwid - 1):
                                print "     ${BG_WHITE}%s${NORMAL}" % tfile
                                tfile = ""

                            if len(tfile) > 0:
                                prlin = "     %s " % tfile
                                prlincolor = "     ${BG_WHITE}%s${NORMAL} " % tfile
                            else:
                                prlin = "     "
                                prlincolor = "     "

                            empty = " "*indent + "."*fwid
                            fwid  = pwid+indent
                            lp    = len(prlin)
                            nsp   = pwid - ( lp % pwid )
                            print prlincolor+("."*nsp)+"${NORMAL}",
                            tstr = ""
                            termsize = terminal.getTerminalSize()
                            maxlen   = termsize[0] - pwid -1
                            dtypes.sort()
                            for dtype in dtypes:
                                if (dtype != None):
                                    newtype = "(%s) " % dtype
                                else:
                                    newtype = "(Unknown) "

                                astr = tstr + newtype
                                if len(astr) >= maxlen:
                                    print "${BLUE}"+ tstr + "${NORMAL}"
                                    tstr = newtype
                                    print empty,
                                else:
                                    tstr = astr

                            if tstr != "":
                                print "${BLUE}"+ tstr + "${NORMAL}"
                                tstr = ""
                                astr = ""

                            if showinfo:
                                print "-"*40
                                fl.hdulist.info()
                                print "-"*40

                            if pheads:
                                print "\t${UNDERLINE}PHU Headers${NORMAL}"
                                for headkey in pheads:
                                    try:
                                        print "\t%s = (%s)" % \
                                            (headkey, fl.phu.header[headkey])
                                    except KeyError:
                                        print "\t%s not present in PHU of %s" % \
                                            (headkey, tfile)

                            if showCals:
                                if self.show_cals_off:
                                    continue
                                # calurls = localCalibrationSearch(fl)
                                # print "\t${BOLD}Local Calibration Search${NORMAL}"
                                # if calurls:
                                #     for caltyp in calurls.keys():
                                #         print "\t${BOLD}%s${NORMAL}: %s" + \
                                #                % (caltyp, calurls[caltyp])
                                # else:
                                #     print "\t${RED}No Calibrations Found${NORMAL}"

        if outfile and outfile_list:
            generate_outfile(outfile, outfile_list, only, or_logic, xcal)
        return

    def _spool_nocal_msg(self):
        print self.nocal_msg
        return

# ------------------------------------------------------------------------------
def main(options):
    # remove current working directory from PYTHONPATH to speed up import in
    # gigantic data directories
    curpath = os.getcwd()

    # @@REVIEW Note: This is here because it's very confusing when someone
    # runs a script IN the package itself.
    if (curpath in sys.path):
        sys.path.remove(curpath)

    if not options.usecolor:
        os.environ["TERM"] = ""

    REASLSTDOUT = sys.stdout
    REALSTDERR  = sys.stderr
    fstdout     = terminal.FilteredStdout()
    fstdout.addFilter(terminal.ColorFilter())
    sys.stdout = fstdout

    # Gemini Specific class code
    dt = DataSpider()
    try:
        dt.typewalk(directory=options.twdir,
                    only=options.types,
                    or_logic=options.or_logic,
                    outfile=options.outfile,
                    pheads=options.phukeys,
                    showinfo=options.showinfo,
                    onlyStatus=options.onlyStatus,
                    onlyTypology=options.onlyTypology,
                    filemask=options.filemask,
                    showCals=options.showCals,
                    stayTop=options.stayTop,
                    raiseExcept=options.raiseExcept,
                    batchnum=int(options.batchnum)-1,
                    xcal=options.xcal
        )
        print "Done DataSpider.typewalk(..)"
    except KeyboardInterrupt:
        print "Interrupted by Control-C"
    return


if __name__ == '__main__':
    args = typewalk_argparser()
    sys.exit(main(args))
