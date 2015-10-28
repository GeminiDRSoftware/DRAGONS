#!/usr/bin/env python
#
#
#                                                                     QAP Gemini
#                                                                  gempy/scripts
#                                                                   autoredux.py
#                                                                        09-2013
# ------------------------------------------------------------------------------
# $Id$
# ------------------------------------------------------------------------------
__version__      = '$Revision$'[11:-2]
__version_date__ = '$Date$'[7:-2]
# ------------------------------------------------------------------------------
# Updated to run and poll continuously across operational day boundaries.
#
# Prior to Rev.4416, user parameter 'upload_metrics' was always passed as True.
# Now, autoredux provides a command line argument to toggle this value:
#
#  -n, --noqa,  Do not upload QA metrics to FITSStore
#
# Default behaviour is to upload. User must toggle for no QA metric upload.
#
# * NOTE: argparse is used if available, but falls back to optparse, which is
#   depracated in 2.7. The get_args() function returns "args" in either case.
#   In the case of optparse, positional arguments are set to equal 'n_args', 
#   matching the args attribute of the ArgumentParser object.
#   Interface to args attributes is the same in both instances.
# ------------------------------------------------------------------------------
import os
import sys
import re
import json
import time
import urllib2

from gempy.gemini.opsdefs import GEMINI_NORTH, GEMINI_SOUTH, OBSPREF
# ------------------------------------------------------------------------------
def buildArgParser():
    from argparse import ArgumentParser

    usage = "autoredux [-h] [-r RECIPE] [-d DIRECTORY] [-c] [-u] [-s SUFFIX]\n"\
            "\t\t [YYYYMMDD] [filenumbers]\n\n"\
            "With no arguments, this will reduce all files in the ops\n"\
            "directory matching the current date. If a date is given\n"\
            "it will reduce all files from the selected date. If a single\n"\
            "file number is given, it will start reducing at that file.\n"\
            "If multiple file numbers are given as a range or a comma\n"\
            "separated list (eg. 1-10,42-46), only those files will be\n"\
            "reduced.\n\nPLEASE NOTE: the adcc must be started in the desired\n"\
            "output directory before launching autoredux.\n"\
            "To start the adcc, type 'adcc >> adcclog 2>&1 &  (in bash)'.\n"
    parser = ArgumentParser(usage=usage)

    parser.add_argument("n_args", metavar="YYYYMMDD or filenumbers",nargs="*")

    parser.add_argument("-r", "--recipe", action="store", dest="recipe", 
                        default=None,
                        help="Specify an alternate processing recipe")

    parser.add_argument("-d", "--directory", action="store", dest="directory", 
                        default=None, 
                        help="Data location. Default OPS directory.")

    parser.add_argument("-c", "--calibrations", action="store_true",
                        dest="calibrations", default=False,
                        help="Reduce calibration files (eg. biases, flats).")

    parser.add_argument("-u", "--upload", action="store_true", dest="upload", 
                        default=False, 
                        help="Upload generated calibrations")

    parser.add_argument("-n", "--noqa", action="store_true", dest="noqa",
                        help="Do not upload QA metrics to FITSStore.")

    parser.add_argument("-s", "--suffix", action="store", dest="suffix", 
                        default="", 
                        help="Specify a filename suffix")

    parser.add_argument("--noskip_backlog", action="store_false", dest="skip_backlog",
                        default=True,
                        help="Do not skip ahead to latest file to avoid backlog")
    parser.add_argument("--quiet", action="store_true", dest="quiet",
                        default=False,
                        help="Run quietly without stdout or display.")

    args = parser.parse_args()

    # arg value checks ...
    if len(args.n_args) == 0:
        pass
    elif len(args.n_args) == 1:
        if re.match("^\d{8}$", args.n_args[0]):
            pass
        elif re.match("^[0-9]+([-,][0-9]+)*$", args.n_args[0]):
            pass
        else:
            parser.error("Bad date or file number: " + args.n_args[0])
    elif len(args.n_args) == 2:
        if not re.match("^\d{8}$", args.n_args[0]):
            parser.error("Bad date: " + fakedate)
        if not re.match("^[0-9]+([-,][0-9]+)*$", args.n_args[1]):
            parser.error("Bad file number: " + args.n_args[1])
    else:
        parser.error("Wrong number of arguments")    

    # args.n_args are okay.
    return args

# ------------------------------------------------------------------------------
def buildOptParser():
    from optparse import OptionParser

    parser = OptionParser()
    usage = parser.get_usage()[:-1] + \
        " [YYYYMMDD] [filenumbers]\n\n"\
        "With no arguments, this will reduce all files in the ops\n"\
        "directory matching the current date. If a date is given\n"\
        "it will reduce all files from the selected date. If a single file\n"\
        "number is given, it will start reducing at that file. If multiple\n"\
        "file numbers are given as a range or a comma separated list,\n"\
        "(eg. 1-10,42-46) only those files will be reduced.\n\n"\
        "PLEASE NOTE: the adcc must be started in the desired output\n"\
        "directory before launching autoredux.  To start the adcc, type\n"\
        "'adcc'.\n"
    parser.set_usage(usage)

    # Get options
    parser.add_option("-r", "--recipe", action="store", dest="recipe", 
                      default=None,
                      help="Specify an alternate processing recipe")

    parser.add_option("-d", "--directory", action="store", dest="directory", 
                      default=None,
                      help="Specify a data directory. Default is ops "\
                           "directory.")

    parser.add_option("-c", "--calibrations", action="store_true", 
                      dest="calibrations", default=False,
                      help="Reduce calibration files (eg. biases and flats).")

    parser.add_option("-u", "--upload", action="store_true", dest="upload", 
                      default=False,
                      help="Upload any generated calibrations to the " + \
                           "calibration service")

    parser.add_option("-n", "--noqa", action="store_true", dest="noqa",
                        help="Do not upload QA metrics to FITSStore.")

    parser.add_option("-s", "--suffix", action="store", dest="suffix", 
                      default="", help="Specify a filename suffix")
    
    parser.add_option("--noskip_backlog", action="store_false", dest="skip_backlog",
                        default=True,
                        help="Do not skip ahead to latest file to avoid backlog")
    parser.add_option("--quiet", action="store_true", dest="quiet",
                      default=False,
                      help="Run quietly without stdout or display.")


    args, pos_args = parser.parse_args()
    args.n_args    = pos_args

    # arg value checks ...
    if len(args.n_args) == 0:
        pass
    elif len(args.n_args) == 1:
        if re.match("^\d{8}$", args.n_args[0]):
            pass
        elif re.match("^[0-9]+([-,][0-9]+)*$", args.n_args[0]):
            pass
        else:
            parser.error("Bad date or file number: " + args.n_args[0])
    elif len(args.n_args) == 2:
        if not re.match("^\d{8}$", args.n_args[0]):
            parser.error("Bad date: " + fakedate)
        if not re.match("^[0-9]+([-,][0-9]+)*$", args.n_args[1]):
            parser.error("Bad file number: " + args.n_args[1])
    else:
        parser.error("Wrong number of arguments")    

    return args

# ------------------------------------------------------------------------------
def get_args():
    try:
        args = buildArgParser()
    except ImportError:
        args = buildOptParser()
    return args

# ------------------------------------------------------------------------------
def ping_adcc():
    # Check that there is an adcc running by requesting its site
    # information
    is_adcc = False
    url = "http://localhost:8777/rqsite.json"
    try:
        rq = urllib2.Request(url)
        u = urllib2.urlopen(rq)
        site = u.read()
        u.close()
    except (urllib2.HTTPError, urllib2.URLError):
        site = None
    if site:
        is_adcc = True
    return is_adcc

# ------------------------------------------------------------------------------
def file_list(str_list):
    # Parse a string with comma-separated ranges of file numbers 
    # into a list of file numbers
    nums = []
    comma_sep = str_list.split(',')
    for item in comma_sep:
        if re.match('^[0-9]+-[0-9]+$', item):
            endpt = item.split('-')
            itemlist = range(int(endpt[0]), int(endpt[1])+1)
            nums += itemlist
        else:
            nums.append(int(item))

    # Eliminate duplicates and sort
    nums = list(set(nums))
    nums.sort()
    return nums

# ------------------------------------------------------------------------------
def check_and_run(filepath, options=None):
    new_file = os.path.basename(filepath)
    if options is not None:
        cal   = options.calibrations
        upl   = options.upload
        rec   = options.recipe
        noqa  = options.noqa
        quiet = options.quiet

    if not quiet:
        print "..."
    
    if os.path.exists(filepath):
        if not quiet:
            print "Checking %s" % new_file
        ok = verify_file(filepath, quiet=quiet)
        if(ok):
            supported, reasons = check_supported_data(filepath, cal)
            
            if supported:
                if not quiet:
                    print "Reducing %s%s" % (new_file, "," if reasons else ""), \
                        ", ".join(reasons)
                launch_reduce(filepath, noqa, upload=upl, recipe=rec, quiet=quiet)
            else:
                if not quiet:
                    print "Ignoring %s%s" % (new_file, "," if reasons else ""), \
                        ", ".join(reasons)
        else:
            if not quiet:
                print "Ignoring %s, not a valid fits file" % new_file
    else:
        if not quiet:
            print "Ignoring %s, does not exist" % new_file
    return

# ------------------------------------------------------------------------------
def verify_file(filepath, quiet=False):
    from gempy.library import fitsverify as fv
    # Repeatedly Fits verify it until it passes
    tries = 10
    ok = False
    while (not ok and tries > 0):
        tries -= 1
        
        # fv_check is a 4 element array:
        # [0]: boolean says whether fitsverify belives it
        #      to be a fits file
        # [1]: number of fitsverify warnings
        # [2]: number of fitsverify errors
        # [3]: full text of fitsverify report

        fv_check = fv.fitsverify(filepath)

        if (fv_check is None or fv_check[0] == False or int(fv_check[2]) > 0):
            ok = False
        else:
            ok = True

        if (tries == 0):
            if not quiet:
                print "ERROR: File %s never did pass fitsverify%s" % \
                    (filepath, "." if fv_check is None else ":\n"+fv_check[3])

        if not ok:
            time.sleep(1)
    return ok

# ------------------------------------------------------------------------------
def check_supported_data(filepath, calibrations=False):
    from astrodata import AstroData
    
    reasons = []
    try:
        ad = AstroData(filepath)
    except:
        reasons.append("Unable to load file.")
        return False, reasons
    
    if "GMOS" in ad.types:
        # just mimicing what was implemented, I don't understand the purpose
        # of this logic structure.  I don't know why the reason of check_gmos_image
        # needs to be added to the reason of check_gmos_longslit. KL March 2014
        #
        # ... in fact I can confirm that this logic is broken.
        gmi, reason = check_gmos_image(ad, calibrations=calibrations)
        reasons.append(reason)
        if gmi:
            return gmi, reasons
        else:
            gmls, reason = check_gmos_longslit(ad)
            reasons.append(reason)
            #return gmls, reasons
            # deactivate GMOS longslit  KL March 2014  (why the heck was it ever
            #        activated?  We don't support GMOS LS yet.)
            return False, reasons
        
    elif "NIRI" in ad.types:
        niri_image, reason = check_niri_image(ad, calibrations=calibrations)
        reasons.append(reason)
        return niri_image, reasons

    else:
        return False, ["Unsupported data type."]
        

# ------------------------------------------------------------------------------
def check_gmos_image(ad, calibrations=False):
    #from astrodata import AstroData
    reason = "GMOS image"
    #try:
    #    ad = AstroData(filepath)
    #except:
    #    reason = "can't load file"
    #    return False, reason
    
    try:
        fp_mask = ad.focal_plane_mask().as_pytype()
    except:
        fp_mask = None

    if "GMOS" not in ad.types:
        reason = "not GMOS"
        return False, reason
    elif "GMOS_DARK" in ad.types:
        reason = "GMOS dark"
        return False, reason
    elif "GMOS_BIAS" in ad.types and not calibrations:
        reason = "GMOS bias"
        return False, reason
    elif "GMOS_IMAGE_FLAT" in ad.types and not calibrations:
        # KL - that doesn't work well.  If calibration is true,
        # you don't get to know that it's a flat, you just get
        # that it's an image.
        reason = "GMOS flat"
        return False, reason
    elif ("GMOS_IMAGE" in ad.types and
          fp_mask!="Imaging"):
        reason = "GMOS slit image"
        return False, reason
    elif (("GMOS_IMAGE" in ad.types and
           fp_mask == "Imaging" and
           "GMOS_DARK" not in ad.types) or
          "GMOS_BIAS" in ad.types or 
          "GMOS_IMAGE_FLAT" in ad.types):

        # Test for 3-amp mode with e2vDD CCDs
        # This mode has not been commissioned.
        dettype = ad.phu_get_key_value("DETTYPE")
        if dettype == "SDSU II e2v DD CCD42-90":
            namps = ad.phu_get_key_value("NAMPS")
            if namps is not None and int(namps)==1:
                reason = "uncommissioned 3-amp mode"
                return False, reason
            else:
                return True, reason
        else:
            return True, reason
    else:
        reason = "not GMOS image"
        return False, reason

# ------------------------------------------------------------------------------
def check_gmos_longslit(ad):
    #from astrodata import AstroData
    reason = "GMOS longslit"

    #try:
    #    ad = AstroData(filepath)
    #except:
    #    reason = "can't load file"
    #    return False, reason
    
    try:
        fp_mask = ad.focal_plane_mask().as_pytype()
    except:
        fp_mask = None

    if "GMOS" not in ad.types:
        reason = "not GMOS"
        return False, reason
    elif "GMOS_DARK" in ad.types:
        reason = "GMOS dark"
        return False, reason
    elif "GMOS_BIAS" in ad.types:
        reason = "GMOS bias"
        return False, reason
    elif "GMOS_NODANDSHUFFLE" in ad.types:
        reason = "GMOS nod and shuffle"
        return False, reason
    elif "GMOS_LS" in ad.types:

        # Test for 3-amp mode with e2vDD CCDs
        # This mode has not been commissioned.
        dettype = ad.phu_get_key_value("DETTYPE")
        if dettype == "SDSU II e2v DD CCD42-90":
            namps = ad.phu_get_key_value("NAMPS")
            if namps is not None and int(namps)==1:
                reason = "uncommissioned 3-amp mode"
                return False, reason
            else:
                return True, reason
        else:
            return True, reason
    else:
        reason = "not GMOS longslit"
        return False, reason

# ------------------------------------------------------------------------------
def check_niri_image(ad, calibrations=False):
    # From what I understand, calibrations controls whether we
    # want autoredux to reduce calibration files.
    # The argument should be renamed to "do_calibrations" or something.
    #
    # For the record, while I'm mimicking the check_gmos functions because
    # we need to deploy asap, I find this whole data checking inefficient,
    # cumbersome, and there must be a better way to do this.
    #
    # - KL March 2014
    
    import re
        
    # check whether this is a through-slit image
    try:
        fp_mask = ad.focal_plane_mask().as_pytype()
    except:
        fp_mask = None
    
    if "NIRI" not in ad.types:
        reason = "not NIRI"
        return False, reason
    elif "NIRI_DARK" in ad.types:
        reason = "NIRI dark"
        if calibrations:
            return True, reason
        else:
            return False, reason
    elif "NIRI_IMAGE_FLAT" in ad.types:
        reason = "NIRI FLAT"
        if calibrations:
            return True, reason
        else:
            return False, reason
    elif "NIRI_IMAGE" in ad.types:
        if fp_mask is None:
            # malformed header, eg. dataset take from DM screen 
            # rather than seqexec.
            reason = 'Malformed headers (from DM screen?)'
            return False, reason
        elif re.compile('-cam_').findall(fp_mask):
            # fp_mask is in the 'cam'era setting
            reason = "NIRI Image"
            return True, reason
        else:
            # fp_mask is a slit
            reason = "NIRI slit image"
            return False, reason
    else:
        reason = "not NIRI image"
        return False, reason


# ------------------------------------------------------------------------------
def launch_reduce(filepath, noqa, upload=False, recipe=None, quiet=False):
    if noqa:
        if not quiet:
            print "Received a noqa signal, no upload of QA metrics"
        up_metrics = False
    else:
        up_metrics = True

    if upload:
        context = "QA,upload"
    else:
        context = "QA"

    if quiet:
        logmode = "quiet"
        nodisplay = True
    else:
        logmode = "standard"
        nodisplay = False

    options = {"context":context,
               "loglevel":"stdinfo",
               "logmode":logmode
              }
    
    if recipe is not None:
        options["recipe"] = recipe

    # QA metrics uploaded to fitsstore with 'upload_metrics' param
    #
    # Re: ignore.  When we switch to the new reduce, the setting
    # should be "display:ignore":quiet and nodisplay variable removed.
    # Here the global setting works because only display has that
    # parameter anyway.  The specific setting is not supported in 
    # the old reduce.
    param_dict = {"filepath":filepath,
                  "parameters":{"clobber":True,
                                "upload_metrics": up_metrics,
                                "ignore":nodisplay
                            },
                  "options":options,
                 }

    postdata = json.dumps(param_dict)
    url = "http://localhost:8777/runreduce/"

    try:
        rq = urllib2.Request(url)
        u = urllib2.urlopen(rq, postdata)
    except urllib2.HTTPError, error:
        contens = error.read()
        if not quiet:
            print contens
        sys.exit()
    u.read()
    u.close()
    return

# ------------------------------------------------------------------------------
def date_and_fileno(nargs):
    """Caller sends a list of postional arguments as supplied by the parser,
    which has ensured that any supplied arguments match format criteria.
    Returns a date string of the form 'YYYYMMDD', a file number as a <str>,

    parameters: <list>         pos args (args.n_args)
    return:     <str>,  <str>  date, filenumber
    """
    fakedate = None
    filenum  = None

    if len(nargs) == 0:
        pass
    elif len(nargs) == 1:
        date_or_num = nargs[0]
        if re.match("^\d{8}$", date_or_num):
            fakedate = date_or_num
        elif re.match("^[0-9]+([-,][0-9]+)*$", date_or_num):
            filenum = date_or_num
    elif len(nargs) == 2:
        fakedate = nargs[0]
        filenum  = nargs[1]

    return fakedate, filenum

# ------------------------------------------------------------------------------
def get_localsite():
    """Returns a Gemini local_site string, one of
    'gemini-north'
    'gemini-south'
    
    parameters: <void>
    return:     <str>
    """
    if time.timezone / 3600 == 10:        # HST = UTC+10
        localsite = GEMINI_NORTH
    elif time.timezone / 3600 in [3,4]:   # CST = UTC+4
        # Set to 3 or 4 due to inconsitent setting of timezones due to DST in
        # Chile being extended at will by the Chilean government.
        localsite = GEMINI_SOUTH
    else:
        print "ERROR - TZ not HST or CST. Local site cannot " + \
            "be determined"
        sys.exit()

    return localsite

# ------------------------------------------------------------------------------
def build_prefix(date, site):
    """Caller passes a date string of the form YYYYMMDD and the local site
    string as returned by get_localste(). Returns a full file prefix.

    parameters: <str>, <str>, YYYYMMDD, local site identifier
    return:     <str>,        file prefix, eg. 'S20130907S'
    """
    try:
        prefix = OBSPREF[site]
    except KeyError:
        print "ERROR - Unrecognized local site:", site
        sys.exit()

    return prefix + date + "S"

# ------------------------------------------------------------------------------
def get_filelist(prefix, suffix, filenum):
    """Caller passes a string as produce by build_prefix(), a string file suffix
    as parsed by the command line parser (i.e. args.suffix), a number string.

    parameters: <str>, <str>, <str>
    return:     <list> or <Nonetype>, <str>
    """
    filelist = None
    if filenum:
        if re.match("^\d{1,4}$", filenum):
            filenum = "%s%.4d%s.fits" % (prefix, int(filenum), suffix)
        else:
            nums = file_list(filenum)
            filelist = ["%s%.4d%s.fits" % (prefix, num, suffix) for num in nums]
    return filelist, filenum


def get_directory(directory, localsite, quiet=False):
    """Caller passes a directory, which may be None, and a localsite string
    as returned by get_localsite(). Returns a directory path for file searching.
    If None is passed as directory, default paths in gemini.opsdef are supplied.

    parameters: <str> or <Nonetype>, <str>
    return:     <str>
    """
    from gempy.gemini.opsdefs import OPSDATAPATH, OPSDATAPATHBKUP

    if not directory:
        if os.path.exists(OPSDATAPATH[localsite]):
            directory = OPSDATAPATH[localsite]
        elif os.path.exists(OPSDATAPATHBKUP[localsite]):
            directory = OPSDATAPATHBKUP[localsite]
        else:
            if not quiet:
                print "Cannot find %s or %s. Please specify a directory." % \
                    (OPSDATAPATH[localsite], OPSDATAPATHBKUP[localsite])
    return directory

# ------------------------------------------------------------------------------
def process_explicit(directory, files, args):
    """Caller passes a list of files and python object, on which the
    pipeline will be called. A nominal object will be a command line parser
    object, where command line arguments are avaible as attributes of the 
    object. Eg., args.suffix.

    parameters: >str>, <list>, <obj>, path, file list, parser object
    return:     <void>
    """
    for new_file in files:
        filepath = os.path.join(directory, new_file)
        check_and_run(filepath, args)
    print "..."
    sys.exit()
    return
# ------------------------------------------------------------------------------
def build_day_list(path, pattern):
    """Build a list of files in <path> matching <pattern> 

    parameters: <str>, <str> path to search, search pattern
    return:     <list>,      matching files in path
    """
    days_list = []
    day_regex = re.compile(pattern)
    path_list = os.listdir(path)

    for ffile in path_list:
        if day_regex.match(ffile):
            days_list.append(ffile)
    days_list.sort()
    return days_list
# ------------------------------------------------------------------------------
def main():
    args = get_args()
    
    quiet = args.quiet

    # Now, check for an adcc
    if not ping_adcc():
        raise RuntimeError("No adcc found at port 8777")

    # Get gempy stuff now. Else, too long to get to the help/error messages)
    from gempy.gemini.gemini_metadata_utils import gemini_date

    # Check for a date or file number argument
    # If a date has been passed, only that day is processed.
    # Boolean 'single_day' indicates.
    fakedate, filenum = date_and_fileno(args.n_args)
    if fakedate:
        single_day = True
        # skipping to last dataset when all datasets already there doesn't
        # make much sense
        args.skip_backlog = False
    else:
        single_day = False
        fakedate   = gemini_date()

    localsite = get_localsite()
    directory = get_directory(args.directory, localsite, quiet=quiet)
    prefix    = build_prefix(fakedate, localsite)
    files, filenum = get_filelist(prefix, args.suffix, filenum)

    # If files were specified explicitly, loop through, then exit
    if files:
        process_explicit(directory, files, args)
 
    # Otherwise, enter while-loop to check for new files
    last_index   = None
    printed_none = False
    printed_wait = False    

    # regex for filename
    regex_patt = '^' + prefix + '\d{4}' + args.suffix + '.fits$'
    file_cre   = re.compile(regex_patt)

    
    # BEFORE modifying this file monitoring algorithm, contact
    # Kathleen, she has a new, more sane logic flow.  Flow charts
    # for both this non-sense and the new proposed flow available.
    while(True):
        today = build_day_list(directory, regex_patt)
        if(len(today) > 0):           # Any files for 'today'?
            new_file = None           # Did we just start up?
            if last_index is None:
                if filenum is not None:
                    try:
                        last_index = today.index(filenum)
                    except ValueError:
                        if not quiet:
                            print "File %s not found" % filenum
                        sys.exit()
                else:
                    last_index = 0

                new_file = today[last_index]
                if not quiet:
                    print "Starting from file: %s" % os.path.basename(new_file)
            else:
                # Did we find something new?
                if len(today) > last_index + 1:
                    if args.skip_backlog:
                        # skip ahead to the last file
                        last_index = len(today) - 1
                    else:
                        last_index += 1
                        
                    new_file = today[last_index]
                    printed_wait = False

            if new_file is not None:
                new_file_path = os.path.join(directory, new_file)
                check_and_run(new_file_path, args)

        else:
            if not printed_none:
                if not quiet:
                    print "No files with the prefix: %s" % prefix
                printed_none = True

        # Wait 1 second before looping again if working on the last file
        if last_index is None or last_index == len(today) - 1:
            # Check the date, if it is not the current one, exit -- there
            # will be no more files to process
            check_date = gemini_date()
            if check_date != fakedate or args.suffix != "":
                if not quiet:
                    print "...\nNo more files to check from %s." % fakedate
                    print "Operational day %s terminated at %s" % \
                        (fakedate, time.ctime(time.time()))

                # If a date argument has been passed, stop.
                if single_day:
                    sys.exit("Finished processing " + fakedate)
                else:
                    fakedate = gemini_date() 

                if not quiet:
                    print "Monitoring operational day %s\n..." % fakedate
                prefix = build_prefix(fakedate, localsite)
                regex_patt = '^' + prefix + '\d{4}' + args.suffix + '.fits$'
                # Reset loop markers for new day.
                last_index   = None
                printed_none = False
                printed_wait = False
            else:
                if not printed_wait:
                    if not quiet:
                        print "...\nWaiting for more files"
                    printed_wait = True
                time.sleep(3)
    return



if __name__ == '__main__':
    main()

