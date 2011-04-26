"""
This is the CadcCRC module. It provides an interface to the CRC function
used by CADC at the GSA.

The full path filename of the executable to run is contained in the
cadcCRC_bin string.

This module also contains an md5sum function for convenience when we
move to md5
"""
import subprocess
import os
import re
from FitsStorageConfig import *

# the path to the fitsverify binary
cadcCRC_bin = '/opt/cadc/cadcCRC'

# Compile the regular expression here for efficiency
cre=re.compile('\S*\s*([0123456789abcdef]*)\n')

def cadcCRC(filename):
  """
  Runs the executable on the specified filename.
  Retuns a string containing the CRC string.
  """
  if fsc_localmode == True:
    return None
  # First check that the filename exists is readable and is a file
  exists = os.access(filename, os.F_OK | os.R_OK)
  isfile = os.path.isfile(filename)
  if(not(exists and isfile)):
    print "%s is not readable or is not a file" % (filename)
    return

  # Fire off the subprocess and capture the output
  sp = subprocess.Popen([cadcCRC_bin, filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  (stdoutstring, stderrstring) = sp.communicate()

  match=cre.match(stdoutstring)
  if(match):
    retary = match.group(1)
  else:
    print "Could not match cadcCRC return value"
    retary=None

  return retary
  
# commented out but left for reference, how an AstroData ID could be
# gotten using the IDFactory
#def cadcCRC_ADID(filename):
#    from astrodata import AstroData
#    from astrodata import IDFactory
#    ad = AstroData(filename)
#    retary = IDFactory.generateFingerprint( ad)
#    return retary

def md5sumfile(f):
  """
  Generates the md5sum of the thing represented by the file object f
  """
  import md5
  m = md5.new()

  block = 64*1024
  data = f.read(block)
  m.update(data)
  while(data):
    data = f.read(block)
    m.update(data)

  return m.hexdigest()

def md5sum(filename):
  """
  Generates the md5sum of the filename, returns the hex string.
  """
  f = open(filename, 'r')
  m = md5sumfile(f)
  f.close()
  return m


