#!/usr/bin/env python

import os
import sys
import re
import time
import subprocess
import datetime
import json
import urllib,urllib2
from astrodata import AstroData
from astrodata.adutils import gemutil as gu
import FitsVerify


# Some global variables
GEMINI_NORTH = 'gemini-north'
GEMINI_SOUTH = 'gemini-south'
OPSDATAPATH = { GEMINI_NORTH : '/net/wikiwiki/vol/dhs_perm/',
                GEMINI_SOUTH : '/net/petrohue/tier1/gem/dhs/perm/' }
OBSPREF = { GEMINI_NORTH : 'N',
            GEMINI_SOUTH : 'S' }

def main():

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

    # Use ops directory
    directory = OPSDATAPATH[localsite]

    # Get current date file prefix
    pyraf, gemini, yes, no = gu.pyrafLoader()
    pyraf.iraf.getfakeUT()
    fakedate = pyraf.iraf.getfakeUT.fakeUT

    prefix = prefix + fakedate + "S"

    # Regular expression for filenames
    file_cre = re.compile('^'+prefix+'\d{4}.fits$')

    # Enter while-loop to check for files
    last_index = None
    while(True):

        # Get a directory listing
        list = os.listdir(directory)

        # filter down to fits files matching the prefix and sort
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
                last_index = 0
                new_file = today[last_index]
                print "Starting from file: %s" % new_file
            else:
                if len(today)>last_index+1:
                    last_index +=1
                    new_file = today[last_index]

            if new_file is not None:
                
                filepath = directory + new_file

                ok = verify_file(filepath)
                if(ok):
                    print "Checking %s" % new_file

                    gmi = check_gmos_image(filepath)
                    if gmi:
                        print "Reducing %s" % new_file
                        launch_reduce(filepath)
                    else:
                        print "Ignoring %s, not a GMOS image" % new_file

                else:
                    print "Ignoring %s, not a valid fits file" % new_file

        else:
            print "No files with the prefix: %s" % prefix

        # Wait 1 second before looping again if working on the last file
        if last_index==len(today)-1:
            time.sleep(1)


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

        fv=FitsVerify.fitsverify(filepath)
        if(fv[0] == False):
            ok = False
        elif(int(fv[2]) > 0):
            ok = False
        else:
            ok = True

        if(tries==0):
            print "ERROR: File %s never did pass fitsverify:\n%s" % \
                  fv[3]
            
        if not ok:
            time.sleep(1)

    return ok


def check_gmos_image(filepath):

    try:
        ad = AstroData(filepath)
    except:
        return False
    
    try:
        fp_mask = ad.focal_plane_mask().as_pytype()
    except:
        fp_mask = None

    if "GMOS" not in ad.types:
        return False
    elif "GMOS_DARK" in ad.types:
        return False
    elif ("GMOS_IMAGE" in ad.types and
          fp_mask!="Imaging"):
        return False
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
                return False
            else:
                return True
        else:
            return True
    else:
        return False

def launch_reduce(filepath):

    param_dict = {"filepath":filepath,
                  "parameters":{"clobber":True},
                  "options":{"context":"QA",
                             "loglevel":"stdinfo"},
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
    
    u.close()


if __name__=='__main__':
    main()

