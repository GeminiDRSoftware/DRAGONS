#!/usr/bin/env python

import os
import sys
import re
import time
import subprocess
import datetime
from optparse import OptionParser
from astrodata import AstroData
from astrodata.adutils import gemutil as gu

# Some global variables
GEMINI_NORTH = 'gemini-north'
GEMINI_SOUTH = 'gemini-south'
OPSDATAPATH = { GEMINI_NORTH : '/net/archie/staging/perm/',
                GEMINI_SOUTH : '/net/reggie/staging/perm/' }
OBSPREF = { GEMINI_NORTH : 'N',
            GEMINI_SOUTH : 'S' }

# Borrowed this little function from fitsstore: importing directly
# caused undesired messages to console.
numre = re.compile('^\d+$')
datecre=re.compile('^20\d\d[01]\d[0123]\d$')
def gemini_date(string):
    """
    A utility function for matching dates of the form YYYYMMDD
    also supports today, yesterday
    returns the YYYYMMDD string, or '' if not a date
    May need modification to make today and yesterday work usefully 
    for Chile
    """
    if(datecre.match(string)):
        return string
    if(string == 'today'):
        now=datetime.datetime.utcnow().date()
        return now.strftime('%Y%m%d')
    if(string == 'yesterday'):
        then=datetime.datetime.utcnow() - datetime.timedelta(days=1)
        return then.date().strftime('%Y%m%d')
    return ''


if __name__=='__main__':

    # Set usage message
    parser = OptionParser()
    usage = parser.get_usage()[:-1] + " [date] file" + \
  """

  Given a number, redux will attempt to construct a Gemini file name from
  the current date.  An alternate date can be specified either as an argument
  or with the prefix (-p) option.  The default directory is the operations
  directory; an alternate directory can be specified with the -d option.
  Files may also be specified directly with a full directory path."""

    parser.set_usage(usage)

    # Get command line options and arguments
    parser.add_option("-d", "--directory", action="store",
                      dest="directory", default=None,
                      help="Specify an input data directory, if not adata " + \
                           "and not included in name.")
    parser.add_option("-p", "--prefix", action="store",
                      dest="prefix", default=None,
                      help="Specify a file prefix if not auto " + \
                           "(eg. (N/S)YYYYMMDDS).")
    parser.add_option("-c", "--clean", action="store_true",
                      dest="clean", default=False,
                      help="Restore current directory to default state: " + \
                           "remove all caches and kill all " + \
                           "reduce and adcc instances")
    parser.add_option("-u", "--upload", action="store_true",
                      dest="upload", default=False,
                      help="Upload any generated calibrations to the " + \
                           "calibration service")
    (options,args) = parser.parse_args()

    # If cleaning desired, call superclean to clear out cache and
    # kill old reduce and adcc processes
    if options.clean:
        print "\nRestoring redux to default state:" + \
              "\nall stack and calibration associations will be lost;" + \
              "\nall temporary files will be removed.\n"

        subprocess.call(["superclean", "--safe"])
        print ""
        sys.exit()


    # Get local site
    if time.timezone / 3600 == 10:        # HST = UTC+10
        localsite = GEMINI_NORTH
    elif time.timezone / 3600 == 4:       # CST = UTC+4
        localsite = GEMINI_SOUTH
    else:
        print "ERROR - timezone is not HST or CST. Local site cannot " + \
              "be determined"
        sys.exit()

    # Check arguments
    if len(args)<1:
        print "ERROR - No input file specified"
        parser.print_help()
        sys.exit()
    elif len(args)>2:
        print "ERROR - Too many arguments"
        parser.print_help()
        sys.exit()
    elif len(args)==2:
        date = gemini_date(args[0])
        if date!="":
            prefix = OBSPREF[localsite]+date+"S"
        else:
            print "ERROR - Date %s not recognized." % args[0]
            parser.print_help()
            sys.exit()

        filenm = prefix + "%04i" % int(args[1])
    else:
        filenm = args[0]

    # Get file prefix from user; otherwise, use auto    
    prefix = "auto"
    if options.prefix:
        prefix = options.prefix

    # Get data directory from user or from site defaults
    directory = "."
    if options.directory:
        directory = options.directory
    else:
        directory = OPSDATAPATH[localsite]
    directory = directory.rstrip("/") + "/"

    # If filenm is a number, use it to construct a name
    if numre.match(filenm):
        # Convert argument into valid file name
        try:
            imgpath,imgname = gu.imageName(filenm, rawpath=directory,
                                           prefix=prefix,
                                           observatory=localsite, 
                                           verbose=False)
        except:
            print "\nFile %s was not found.\n" % filenm
            sys.exit()
    else:

        if prefix!="auto":
            filenm = prefix + filenm

        # Check argument to see if it is a valid file
        if os.path.exists(filenm):
            imgpath = filenm
        elif os.path.exists(filenm + ".fits"):
            imgpath = filenm + ".fits"
        elif os.path.dirname(filenm)=="." and not os.path.exists(filenm):
            # Check for case that current directory was explicitly
            # specified and file does not exist -- otherwise, 
            # it might match directory + ./ + filenm when it should
            # fail
            print "\nFile %s was not found.\n" % filenm
            sys.exit()
        elif os.path.exists(directory + filenm):
            imgpath = directory + filenm
        elif os.path.exists(directory + filenm + ".fits"):
            imgpath = directory + filenm + ".fits"
        else:
            print "\nFile %s was not found.\n" % filenm
            sys.exit()

    imgname = os.path.basename(imgpath)

    print "Image path: "+imgpath

    # Check that file is a GMOS IMAGE; other types are not yet supported
    # (allow biases and flats, too)
    try:
        ad = AstroData(imgpath)
    except IOError:
        print "\nProblem accessing file %s.\n" % filenm
        sys.exit()

    if (("GMOS_IMAGE" in ad.types and
         ad.focal_plane_mask()=="Imaging" and
         "GMOS_DARK" not in ad.types) or
        "GMOS_BIAS" in ad.types or 
        "GMOS_IMAGE_FLAT" in ad.types):

        # Call reduce with auto-selected reduction recipe
        print "\nBeginning reduction for file %s, %s\n" % (imgname,
                                                           ad.data_label()) 
        if options.upload:
            context = "QA,upload"
        else:
            context = "QA"

        reduce_cmd = ["reduce", 
                      "--context",context,
                      "--logLevel","stdinfo",
                      "-p", "clobber=True",
                      imgpath]
            
        subprocess.call(reduce_cmd)

        print ""

    else:
        print "\nFile %s is not a supported type." % imgname + \
              "\nOnly GMOS images can be reduced at this time."

        print "\nTypes:",ad.types
        print "Focal plane mask:",ad.focal_plane_mask(),"\n"

        sys.exit()


