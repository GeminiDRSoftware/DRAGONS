#!/usr/bin/env python
#
#                                                                  gempy/scripts
#                                                                       redux.py
#                                                                        08-2013
# ------------------------------------------------------------------------------
# $Id$
# ------------------------------------------------------------------------------
__version__      = '$Revision$'[11:-2]
__version_date__ = '$Date$'[7:-2]
# ------------------------------------------------------------------------------
import os
import re
import sys
import time
import datetime
import subprocess

from optparse import OptionParser

from astrodata import AstroData
from astrodata.adutils import gemutil as gu

from gempy.gemini.opsdefs import GEMINI_NORTH, GEMINI_SOUTH
from gempy.gemini.opsdefs import OPSDATAPATH, OPSDATAPATHBKUP, OBSPREF
# ------------------------------------------------------------------------------
def handleClArgs():
    parser = OptionParser()
    usage = parser.get_usage()[:-1] + " [date] file" + \
            """
            Given a number, redux will attempt to construct a Gemini file name 
            from the current date.  An alternate date can be specified either as 
            an argument or with the prefix (-p) option.  The default directory 
            is the operations directory; an alternate directory can be specified 
            with the -d option. Files may also be specified directly with a full 
            directory path.
            """
    parser.set_usage(usage)
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
    parser.add_option("-s", "--stack", action="store_true",
                      dest="stack", default=False,
                      help="Perform stacking of all previously reduced "+ \
                           "images associated with the current image")
    (options, args) = parser.parse_args()
    return options, args, parser

# ------------------------------------------------------------------------------
# Borrowed function from fitsstore. 
# Direct import caused undesired messages to console.

def gemini_date(string):
    """
    A utility function for matching dates of the form YYYYMMDD.
    Also supports today, yesterday returns the YYYYMMDD string, or '' 
    if not a date. May need modification to make today and yesterday work 
    usefully for Chile.
    """
    datecre=re.compile('^20\d\d[01]\d[0123]\d$')
    if(datecre.match(string)):
        return string
    if(string == 'today'):
        now=datetime.datetime.utcnow().date()
        return now.strftime('%Y%m%d')
    if(string == 'yesterday'):
        then=datetime.datetime.utcnow() - datetime.timedelta(days=1)
        return then.date().strftime('%Y%m%d')
    return ''
# ------------------------------------------------------------------------------
# This function contains the logic necessary for full flexibility in
# input specification
def image_path(filenm, directory, prefix="auto", localsite=None, suffix=None):
    imgpath = None
    numre   = re.compile('^\d+$')
    # If filenm is a number, use it to construct a name
    if numre.match(filenm):
        # Convert argument into valid file name
        try:
            imgpath, imgname = gu.imageName(filenm, rawpath=directory,
                                            prefix=prefix,
                                            suffix=suffix,
                                            observatory=localsite, 
                                            verbose=False)
        except:
            imgpath = None
    else:

        if prefix!="auto":
            filenm = prefix + filenm
        if suffix!=None:
            (fn,ext) = os.path.splitext(filenm)
            filenm = fn + suffix + ext

        # Check argument to see if it is a valid file
        if os.path.exists(filenm):
            imgpath = filenm
        elif os.path.exists(filenm + ".fits"):
            imgpath = filenm + ".fits"
        elif os.path.dirname(filenm) == "." and not os.path.exists(filenm):
            # Check for case that current directory was explicitly
            # specified and file does not exist -- otherwise, 
            # it might match directory + ./ + filenm when it should
            # fail
            imgpath = None
        elif os.path.exists(directory + filenm):
            imgpath = directory + filenm
        elif os.path.exists(directory + filenm + ".fits"):
            imgpath = directory + filenm + ".fits"

    return imgpath
# ------------------------------------------------------------------------------
def main():
    options, args, parser = handleClArgs()

    # If clean, call superclean to clear out cache and kill old reduce
    # and adcc processes.
    if options.clean:
        print "\nRestoring redux to default state:" + \
              "\nall stack and calibration associations will be lost;" + \
              "\nall temporary files will be removed.\n"
        subprocess.call(["superclean", "--safe"])
        print
        sys.exit()

    # Get local site
    if time.timezone / 3600 == 10:      # HST = UTC + 10
        localsite = GEMINI_NORTH
    elif time.timezone / 3600 == 4:     # CST = UTC + 4
        localsite = GEMINI_SOUTH
    else:
        print "ERROR - timezone is not HST or CST. Local site cannot " + \
              "be determined"
        sys.exit()

    # Check arguments
    if len(args) < 1:
        print "ERROR - No input file specified"
        parser.print_help()
        sys.exit()
    elif len(args) > 2:
        print "ERROR - Too many arguments"
        parser.print_help()
        sys.exit()
    elif len(args) == 2:
        date = gemini_date(args[0])
        if date != "":
            prefix = OBSPREF[localsite] + date + "S"
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
        if os.path.exists(OPSDATAPATH[localsite]):
            directory = OPSDATAPATH[localsite]
        elif os.path.exists(OPSDATAPATHBKUP[localsite]):
            directory = OPSDATAPATHBKUP[localsite]
        else:
            print "Cannot find %s or %s. Please specify a directory." % \
                  (OPSDATAPATH[localsite], OPSDATAPATHBKUP[localsite])
    directory = directory.rstrip("/") + "/"

    # Check for stacking option
    recipe = None
    if options.stack:

        # Check for _forStack image in local directory
        if os.path.exists(filenm):
            stack_filenm = os.path.basename(filenm)
        else:
            stack_filenm = filenm

        imgpath = image_path(stack_filenm, ".", 
                             prefix=prefix, localsite=localsite,
                             suffix="_forStack")
        if imgpath is None:
            # If not found, check for raw image in specified directory
            # and use full reduction recipe
            imgpath = image_path(filenm, directory, 
                                 prefix=prefix, localsite=localsite)
            recipe = "qaReduceAndStack"
        else:
            # If found, just use stacking recipe
            recipe = "qaStack"
    else:
        imgpath = image_path(filenm, directory, 
                             prefix=prefix, localsite=localsite)

    if imgpath is None:
        print "\nFile %s was not found.\n" % filenm
        sys.exit()

    imgname = os.path.basename(imgpath)
    print "Image path: "+imgpath

    # Check that file is GMOS LONGSLIT or IMAGE; other types are not yet supported
    # (but do allow biases and flats, too)
    try:
        ad = AstroData(imgpath)
    except IOError:
        print "\nProblem accessing file %s.\n" % filenm
        sys.exit()

    try:
        fp_mask = ad.focal_plane_mask().as_pytype()
    except:
        fp_mask = None

    if "RAW" not in ad.types and "PREPARED" in ad.types:
        print "AstroDataType 'RAW'  not found in types."
        print "AstroDataType 'PREPARED' found in types."
        print "\nFile %s appears to have been processed." % imgname
        print "redux halting ..."
        sys.exit()
    
    # Good grief!  This is a messy way to do things.  I'll just try to add 
    # NIRI to this GMOS-centric piece of "logic".  Badly needs to be 
    # rewritten. KL March 2014
    # I also remove the GMOS Longslit stuff since, well, we don't support
    # it yet!
 
    if "GMOS" not in ad.types and "NIRI" not in ad.types:
        print "\nFile %s is neither a GMOS or a NIRI file." % imgname
        print "Only GMOS and NIRI images can be reduced at this time.\n"
        sys.exit()
    elif "GMOS_DARK" in ad.types:
        print "\nFile %s is a GMOS dark." % imgname
        print "Only GMOS images can be reduced at this time.\n"
        sys.exit()
    elif ("GMOS_IMAGE" in ad.types and
          fp_mask!="Imaging"):
        print "\nFile %s is a slit image." % imgname
        print "Only GMOS images can be reduced at this time.\n"
        sys.exit()
    elif ("NIRI_IMAGE" in ad.types and 
          not re.compile('-cam_').findall(fp_mask)):
        print "\nFile %s is a slit image." % imgname
        print "Only NIRI images can be reduced at this time.\n"
        sys.exit()
    elif (("GMOS_IMAGE" in ad.types and
           fp_mask=="Imaging" and 
           "GMOS_DARK" not in ad.types) or 
          "GMOS_BIAS" in ad.types or 
          "GMOS_IMAGE_FLAT" in ad.types): # or 
          #"GMOS_LS" in ad.types):

        # Test for 3-amp mode with e2vDD CCDs. NOT commissioned.
        dettype = ad.phu_get_key_value("DETTYPE")
        if dettype=="SDSU II e2v DD CCD42-90":
            namps = ad.phu_get_key_value("NAMPS")
            if namps is not None and int(namps)==1:
                print "\nERROR: The GMOS e2v detectors should " \
                      "not use 3-amp mode!"
                print "Please set the GMOS CCD Readout Characteristics " \
                      "to use 6 amplifiers.\n"
                sys.exit()
        
        OK_launch_reduce = True

    elif "NIRI_IMAGE" in ad.types:
        OK_launch_reduce = True
    else:
        OK_launch_reduce = False

    if OK_launch_reduce:
        print "\nBeginning reduction for file %s, %s\n" % (imgname,
                                                           ad.data_label()) 
        if options.upload:
            context = "QA,upload"
        else:
            context = "QA"

        # Check for an alternate recipe for science reductions
        if ("GMOS_IMAGE" in ad.types and
            "GMOS_BIAS" not in ad.types and 
            "GMOS_IMAGE_FLAT" not in ad.types and
            recipe is not None):
            reduce_cmd = ["reduce",
                          "-r", recipe,
                          "--context",context,
                          "--loglevel","stdinfo",
                          "--logfile","gemini.log",
                          "-p", "clobber=True",
                          imgpath]
        elif ("NIRI_IMAGE" in ad.types and
              "NIRI_DARK" not in ad.types and
              "NIRI_IMAGE_FLAT" not in ad.types and
              recipe is not None):
            reduce_cmd = ["reduce",
                          "-r", recipe,
                          "--context",context,
                          "--loglevel","stdinfo",
                          "--logfile","gemini.log",
                          "-p", "clobber=True",
                          imgpath]
        else:
            # Otherwise call reduce with auto-selected reduction recipe
            reduce_cmd = ["reduce", 
                          "--context",context,
                          "--loglevel","stdinfo",
                          "--logfile","gemini.log",
                          "-p", "clobber=True",
                          imgpath]
            
        subprocess.call(reduce_cmd)

        print ""

    else:
        print "\nFile %s is not a supported type." % imgname
        print "Only GMOS and NIRI images can be reduced at this time.\n"
        sys.exit()


if __name__ == '__main__':
    main()

