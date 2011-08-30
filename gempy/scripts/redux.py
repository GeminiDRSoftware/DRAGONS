#!/usr/bin/env python

import sys
import subprocess
from optparse import OptionParser
from astrodata import AstroData
from astrodata.adutils import gemutil as gu

# Define site here
localsite = "Gemini-North"

if __name__=='__main__':

    # Get command line options and arguments
    parser = OptionParser()
    parser.set_usage(parser.get_usage()[:-1] + " infile")
    parser.add_option("-c", "--clean", action="store_true",
                      dest="clean", default=False,
                      help="Restore to default state: remove all caches, " + \
                           "reduce instances, and adcc instances")
    parser.add_option("-d", "--directory", action="store",
                      dest="directory", default=None,
                      help="Specify an input data directory, if not adata " + \
                           "and not included in name.")
    parser.add_option("-p", "--prefix", action="store",
                      dest="prefix", default=None,
                      help="Specify a file prefix if not auto " + \
                           "(ie. (N/S)YYYYMMDDS).")
    (options,args) = parser.parse_args()

    # If cleaning desired, call superclean to clear out cache and
    # kill old reduce and adcc processes
    if options.clean:
        print "\nRestoring redux to default state:" + \
              "\nall stack and calibration associations will be lost;" + \
              "\nall temporary files will be removed.\n"

        subprocess.call(["superclean", "-rad"])
        subprocess.call(["rm", "-f", "tmp*"])
        print ""
        sys.exit()

    
    if len(args)!=1:
        print "ERROR - No input file specified"
        parser.print_help()
        sys.exit()

    # Get data directory from user or from site defaults
    directory = "."
    if options.directory:
        directory = options.directory
    elif localsite=="Gemini-North":
        directory = "/net/archie/staging/perm/"
    elif localsite=="Gemini-South":
        directory = "/net/reggie/staging/perm/"

    # Get file prefix from user; otherwise, use auto
    prefix = "auto"
    if options.prefix:
        prefix = options.prefix
    
    # Convert argument into valid file name
    try:
        imgpath,imgname = gu.imageName(args[0], rawpath=directory,
                                       prefix=prefix,
                                       observatory=localsite, verbose=False)
    except IOError:
        print "\nFile %s was not found.\n" % args[0]
        sys.exit()

    # Check that file is a GMOS IMAGE; other types are not yet supported
    # (allow biases and flats, too)
    ad = AstroData(imgpath)
    if (("GMOS_IMAGE" in ad.types and ad.focal_plane_mask()!="Imaging") or
        "GMOS_BIAS" in ad.types or 
        "GMOS_IMAGE_FLAT" in ad.types):

        # Call reduce with auto-selected reduction recipe
        print "\nBeginning reduction for file %s, %s\n" % (imgname,ad.data_label()) 
        reduce_cmd = ["reduce", 
                      "--context","QA",
                      "--logLevel","stdinfo",
                      "-p", "clobber=True",
                      imgpath]
        subprocess.call(reduce_cmd)

        print ""

    else:
        print "\nFile %s is not a supported type." % imgname + \
              "\nOnly GMOS images can be reduced at this time.\n"
        sys.exit()

