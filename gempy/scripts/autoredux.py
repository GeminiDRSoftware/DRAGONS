#!/usr/bin/env python

import os
import sys
import re
import time
import subprocess
import datetime
import json
import urllib,urllib2
from optparse import OptionParser
from astrodata import AstroData
from gempy import gemini_metadata_utils as gmu
from gempy import fitsverify as fv
from gempy.gemini.opsdefs import GEMINI_NORTH, GEMINI_SOUTH, OPSDATAPATH, OPSDATAPATHBKUP, OBSPREF

def main():

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
    parser.add_option("-r", "--recipe", action="store",
                      dest="recipe", default=None,
                      help="Specify an alternate processing recipe")
    parser.add_option("-d", "--directory", action="store",
                      dest="directory", default=None,
                      help="Specify a data directory. Default is ops "\
                           "directory.")
    parser.add_option("-c", "--calibrations", action="store_true",
                      dest="calibrations", default=False,
                      help="Reduce calibration files (eg. biases and flats).")
    parser.add_option("-u", "--upload", action="store_true",
                      dest="upload", default=False,
                      help="Upload any generated calibrations to the " + \
                           "calibration service")
    parser.add_option("-s", "--suffix", action="store",
                      dest="suffix", default="",
                      help="Specify a filename suffix")
    (options, args) = parser.parse_args()

    # Check for a date or file number argument
    if len(args)==0:
        fakedate = None
        filenum = None
    elif len(args)==1:
        date_or_num = args[0]
        if re.match("^\d{8}$",date_or_num):
            fakedate = date_or_num
            filenum = None
        elif re.match("^[0-9]+([-,][0-9]+)*$",date_or_num):
            filenum = date_or_num
            fakedate = None
        else:
            parser.error("Bad date or file number: "+date_or_num)
    elif len(args)==2:
        fakedate = args[0]
        filenum = args[1]
        if not re.match("^\d{8}$",fakedate):
            parser.error("Bad date: "+fakedate)
        if not re.match("^[0-9]+([-,][0-9]+)*$",filenum):
            parser.error("Bad file number: "+filenum)
    else:
        parser.error("Wrong number of arguments")

    # Before doing anything else, check for an adcc
    is_adcc = ping_adcc()
    if not is_adcc:
        parser.error("No adcc found at port 8777")

    if fakedate is None:
        fakedate = gmu.gemini_date()

    # Get local site
    if time.timezone / 3600 == 10:        # HST = UTC+10
        localsite = GEMINI_NORTH
        prefix = "N"
    elif time.timezone / 3600 == 4:       # CST = UTC+4
        localsite = GEMINI_SOUTH
        prefix = "S"
    else:
        print "ERROR - timezone is not HST or CST. Local site cannot " + \
              "be determined"
        sys.exit()

    # Construct the file prefix
    prefix = prefix + fakedate + "S"

    # Regular expression for filenames
    file_cre = re.compile('^'+prefix+'\d{4}'+options.suffix+'.fits$')

    # Construct the file name to start from, if desired
    files = None
    if filenum is not None:
        if re.match("^\d{1,4}$",filenum):
            filenum = "%s%.4d.fits"% (prefix,int(filenum))
        else:
            nums = file_list(filenum)
            files = ["%s%.4d.fits"% (prefix,num) for num in nums]

    # Get directory
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

    # If files were specified explicitly, loop through them, then exit
    if files is not None:
        for new_file in files:
            filepath = directory + new_file
            check_and_run(filepath,options)
        print "..."
        sys.exit()

    # Otherwise, enter while-loop to check for new files
    last_index = None
    printed_none = False
    printed_wait = False
    while(True):

        # Get a directory listing
        list = os.listdir(directory)

        # Filter down to fits files matching the prefix and sort
        today = []
        for i in list:
            if file_cre.match(i):
                today.append(i)
        today.sort()

        # Did we find anything at all?
        if(len(today) > 0):

            # Did we just start up?
            new_file = None
            if last_index is None:
                if filenum is not None:
                    try:
                        last_index = today.index(filenum)
                    except ValueError:
                        print "File %s not found" % filenum
                        sys.exit()
                else:
                    last_index = 0
                new_file = today[last_index]
                print "Starting from file: %s" % new_file
            else:
                # Did we find something new?
                if len(today)>last_index+1:
                    last_index +=1
                    new_file = today[last_index]
                    printed_wait = False

            if new_file is not None:
                filepath = directory + new_file
                check_and_run(filepath,options)

        else:
            if not printed_none:
                print "No files with the prefix: %s" % prefix
                printed_none = True

        # Wait 1 second before looping again if working on the last file
        if last_index is None or last_index==len(today)-1:
            # Check the date, if it is not the current one, exit -- there
            # will be no more files to process
            check_date = gmu.gemini_date()
            if check_date!=fakedate:
                print "...\nNo more files to check from %s\n..." % fakedate 
                sys.exit()
            else:
                if not printed_wait:
                    print "...\nWaiting for more files"
                    printed_wait = True
                time.sleep(1)


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

def file_list(str_list):
    # Parse a string with comma-separated ranges of file numbers 
    # into a list of file numbers
    nums = []
    comma_sep = str_list.split(',')
    for item in comma_sep:
        if re.match('^[0-9]+-[0-9]+$', item):
            endpt = item.split('-')
            itemlist = range(int(endpt[0]),int(endpt[1])+1)
            nums += itemlist
        else:
            nums.append(int(item))

    # Eliminate duplicates and sort
    nums = list(set(nums))
    nums.sort()
    return nums

def check_and_run(filepath,options=None):
    new_file = os.path.basename(filepath)
    if options is not None:
        cal = options.calibrations
        upl = options.upload
        rec = options.recipe

    print "..."
    
    if os.path.exists(filepath):
        ok = verify_file(filepath)
        if(ok):
            print "Checking %s" % new_file
        
            gmi,reason = check_gmos_image(filepath,
                                          calibrations=cal)
            if gmi:
                print "Reducing %s" % new_file
                launch_reduce(filepath,upload=upl,recipe=rec)
            else:
                print "Ignoring %s, %s" % (new_file, reason)

        else:
            print "Ignoring %s, not a valid fits file" % new_file
    else:
        print "Ignoring %s, does not exist" % new_file

def verify_file(filepath):

    # Repeatedly Fits verify it until it passes
    tries = 10
    ok = False
    while (not ok and tries > 0):
        tries -= 1
        
        # This module call returns a 4 element array:
        # [0]: boolean says whether fitsverify belives it
        #      to be a fits file
        # [1]: number of fitsverify warnings
        # [2]: number of fitsverify errors
        # [3]: full text of fitsverify report

        fv_check = fv.fitsverify(filepath)
        if(fv_check[0] == False):
            ok = False
        elif(int(fv_check[2]) > 0):
            ok = False
        else:
            ok = True

        if(tries==0):
            print "ERROR: File %s never did pass fitsverify:\n%s" % \
                  fv_check[3]
            
        if not ok:
            time.sleep(1)

    return ok


def check_gmos_image(filepath, calibrations=False):

    reason = "is GMOS image"

    try:
        ad = AstroData(filepath)
    except:
        return False,reason
    
    try:
        fp_mask = ad.focal_plane_mask().as_pytype()
    except:
        fp_mask = None

    if "GMOS" not in ad.types:
        reason = "not GMOS"
        return False,reason
    elif "GMOS_DARK" in ad.types:
        reason = "GMOS dark"
        return False,reason
    elif "GMOS_BIAS" in ad.types and not calibrations:
        reason = "GMOS bias"
        return False,reason
    elif "GMOS_IMAGE_FLAT" in ad.types and not calibrations:
        reason = "GMOS flat"
        return False,reason
    elif ("GMOS_IMAGE" in ad.types and
          fp_mask!="Imaging"):
        reason = "GMOS slit image"
        return False,reason
    elif (("GMOS_IMAGE" in ad.types and
           fp_mask=="Imaging" and
           "GMOS_DARK" not in ad.types) or
          "GMOS_BIAS" in ad.types or 
          "GMOS_IMAGE_FLAT" in ad.types):

        # Test for 3-amp mode with e2vDD CCDs
        # This mode has not been commissioned.
        dettype = ad.phu_get_key_value("DETTYPE")
        if dettype=="SDSU II e2v DD CCD42-90":
            namps = ad.phu_get_key_value("NAMPS")
            if namps is not None and int(namps)==1:
                reason = "uncommissioned 3-amp mode"
                return False,reason
            else:
                return True,reason
        else:
            return True,reason
    else:
        reason = "not GMOS image"
        return False,reason

def launch_reduce(filepath, upload=False, recipe=None):

    if upload:
        context = "QA,upload"
    else:
        context = "QA"
    if recipe is not None:
        options = {"context":context,
                   "loglevel":"stdinfo",
                   "recipe": recipe}
    else:
        options = {"context":context,
                   "loglevel":"stdinfo"}

    param_dict = {"filepath":filepath,
                  "parameters":{"clobber":True},
                  "options":options,
              }

    postdata = json.dumps(param_dict)

    url = "http://localhost:8777/runreduce/"
    try:
        rq = urllib2.Request(url)
        u = urllib2.urlopen(rq, postdata)
    except urllib2.HTTPError, error:
        contens = error.read()
        print contens
        sys.exit()
    
    u.read()
    u.close()


if __name__=='__main__':
    main()

