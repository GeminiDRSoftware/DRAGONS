"""
This is the "CadcWMD" module. It provides a python interface to the 
CADC provided wmd tool.

This module provides one function, cadcWMD(filename) which causes the 
wmd command to be run on filename, and returns two values, a boolean 
and a string. 

The boolean is true if wmd says the file IS ready to be ingested into 
the GSA, and false if wmd says IS NOT ready for ingestion.  

The string is the text output of the wmd report.

The path to the executable (java in this case) and all the necessary 
arguments to make it run are contained in the wmd array.
"""


import subprocess
import os
import re

from FitsStorageConfig import *

# the path to the fitsverify binary
wmd = ['/astro/i686/jre1.5.0_03/bin/java', '-Djava.library.path=/opt/cadc/mdIngest/lib.x86_fedora', '-Dca.nrc.cadc.configDir=/opt/cadc/mdIngest/config', '-jar', '/opt/cadc/mdIngest/lib/mdIngest.jar', '--archive=GEMINI', '-c', '-d', '--log=/dev/null']

# Compile the regular expression here for efficiency
cre=re.compile('File \S* (IS|IS NOT) ready for ingestion')

def cadcWMD(filename):
  """
  Run the wmd command on filename.
  Returns a (boolean, string) pair. 
  The boolean is true if wmd says the file IS ready to be ingested 
  into the GSA, and false if wmd says IS NOT ready for ingestion. 
  The string is the text output of the wmd report.
  """
  
  if fsc_localmode:
    return (False, "No WMD Report, local mode")
  # First check that the filename exists is readable and is a file
  exists = os.access(filename, os.F_OK | os.R_OK)
  isfile = os.path.isfile(filename)
  if(not(exists and isfile)):
    print "%s is not readable or is not a file" % (filename)
    return

  wmd_arg = "--file=%s" % filename
  wmdcmd=list(wmd)
  wmdcmd.append(wmd_arg)

  env=os.environ
  env['LD_LIBRARY_PATH']='/opt/cadc/mdIngest/lib.x86_fedora'
  # Fire off the subprocess and capture the output
  sp = subprocess.Popen(wmdcmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  (stdoutstring, stderrstring) = sp.communicate()

  match=cre.search(stdoutstring)
  if(match):
    isit = match.group(1)
    itis=0
    if(isit=="IS"):
      itis=1
    if(isit=="IS NOT"):
      itis=0
  else:
    print "Could not match cadcWMD return value"
    itis = 0

  return (itis, stdoutstring)
