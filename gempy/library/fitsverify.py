"""
This is the fitsverify module. It provides a python interface to the
fitsverify command.

The fitsverify_bin string contains the path to the fitsverify
executable
"""

import subprocess
import os
import re

# the path to the fitsverify binary
#FITSVERIFY_BIN = '/opt/fitsverify/fitsverify'
FITSVERIFY_BIN = 'fitsverify'

# Compile the regular expression here for efficiency
NFRE = re.compile(r'This does not look like a FITS file.')
WERE = re.compile(r'\*\*\*\* Verification found (\d*) warning\(s\) and (\d*) error\(s\). \*\*\*\*')

def fitsverify(filename):
    """
    Runs the fitsverify command on the filename argument.
    Returns a 4 component array containing

    * a boolean that is true if the argument is a fits file
    * an integer giving the number of warnings
    * an integer giving the number of errors
    * a string containing the full fitsverify report
    """

    # First check that the filename exists is readable and is a file
    exists = os.access(filename, os.F_OK | os.R_OK)
    isfile = os.path.isfile(filename)
    if not (exists and isfile):
        print("%s is not readable or is not a file" % (filename))
        return

    # Fire off the subprocess and capture the output
    subproc = subprocess.Popen([FITSVERIFY_BIN, filename], stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    (stdoutstring, stderrstring) = subproc.communicate()
   
    stdoutstring += stderrstring

    # Check to see if we got a not a fits file situation
    nfmatch = NFRE.search(stdoutstring.decode('utf-8'))
    if nfmatch:
        isfits = 0
    else:
        isfits = 1

    # If it is a fits file, parse how many warnings and errors we got
    if isfits:
        #print stdoutstring
        match = WERE.search(stdoutstring.decode('utf-8'))
        if match:
            warnings = match.group(1)
            errors = match.group(2)
        else:
            print("Could not match warnings and errors string")
            warnings = 255
            errors = 255

    return [isfits, warnings, errors, stdoutstring]
