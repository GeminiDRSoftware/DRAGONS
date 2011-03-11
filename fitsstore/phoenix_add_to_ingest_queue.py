import sys
sys.path=['/opt/sqlalchemy/lib/python2.5/site-packages', '/astro/iraf/x86_64/gempylocal/lib/stsci_python/lib/python2.5/site-packages']+sys.path

import FitsStorage
import FitsStorageConfig
from FitsStorageLogger import *
from FitsStorageUtils import *
import os
import re
import datetime
import time
import subprocess

# Option Parsing
from optparse import OptionParser
parser = OptionParser()
parser.add_option("--file-re", action="store", type="string", dest="file_re", help="python regular expression string to select files by. Special values are today, twoday, fourday to include only files from today, the last two days, or the last four days respectively (days counted as UTC days)")
parser.add_option("--debug", action="store_true", dest="debug", help="Increase log level to debug")
parser.add_option("--demon", action="store_true", dest="demon", help="Run as a background demon, do not generate stdout")
parser.add_option("--date", action="store", type="string", dest="date", help="date(s) to process. Special values are today, twoday, fourday to include only files from today, the last two days, or the last four days respectively")

(options, args) = parser.parse_args()

# Logging level to debug? Include stdio log?
setdebug(options.debug)
setdemon(options.demon)

# Annouce startup
now = datetime.datetime.now()
logger.info("*********  phoenix_add_to_ingest_queue.py - starting up at %s" % now)

# First, we need to parse the date string to figure out what we're doing.
datestring = options.date
dates = []

# Handle the today and twoday options
now=datetime.datetime.utcnow()
delta=datetime.timedelta(days=1)

if(options.date == "today"):
  dates.append(now.date())

if(options.date == "twoday"):
  then=now-delta
  dates.append(now.date())
  dates.append(then.date())

if(options.date == "fourday"):
  dates.append(now.date())
  then=now-delta
  dates.append(then.date())
  then=then-delta
  dates.append(then.date())
  then=then-delta
  dates.append(then.date())

# now loop through the dates, calling add_to_ingest_queue.py
# with the appropriate options
for date in dates:
  datestr = date.strftime("%Y%b%d")
  cmd = []
  cmd.append("/astro/iraf/x86_64/gempylocal/bin/python")
  cmd.append("/opt/FitsStorage/add_to_ingestqueue.py")
  if(options.file_re):
    cmd.append("--file-re=%s" % options.file_re)
  if(options.debug):
    cmd.append("--debug")
  if(options.demon):
    cmd.append("--demon")
  path = os.path.join("phoenix", datestr)
  cmd.append("--path=%s" % path)

  logger.info("Running: %s" % str(cmd))
  # Fire off the subprocess and capture the output
  sp = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  (stdoutstring, stderrstring) = sp.communicate()
  logger.info(stdoutstring)
  logger.info(stderrstring)

now=datetime.datetime.now()
logger.info("*** phoenix_add_to_ingestqueue.py exiting normally at %s" % now)

