import sys
sys.path=['/opt/sqlalchemy/lib/python2.5/site-packages', '/astro/iraf/x86_64/gempylocal/lib/stsci_python/lib/python2.5/site-packages']+sys.path

import FitsStorage
import FitsStorageConfig
from FitsStorageLogger import *
from FitsStorageUtils.AddToIngestQueue import *
import os
import re
import datetime
import time

# Option Parsing
from optparse import OptionParser
parser = OptionParser()
parser.add_option("--file-re", action="store", type="string", dest="file_re", help="python regular expression string to select files by. Special values are today, twoday, fourday to include only files from today, the last two days, or the last four days respectively (days counted as UTC days)")
parser.add_option("--debug", action="store_true", dest="debug", help="Increase log level to debug")
parser.add_option("--demon", action="store_true", dest="demon", help="Run as a background demon, do not generate stdout")
parser.add_option("--path", action="store", dest="path", default = "", help="Use given path relative to storage root")

(options, args) = parser.parse_args()
path = options.path

# Logging level to debug? Include stdio log?
setdebug(options.debug)
setdemon(options.demon)

# Annouce startup
now = datetime.datetime.now()
logger.info("*********  add_to_ingest_queue.py - starting up at %s" % now)

# Get a list of all the files in the datastore
# We assume this is just one dir (ie non recursive) for now.

fulldirpath = os.path.join(FitsStorage.storage_root, path)
logger.info("Queueing files for ingest from: %s" % fulldirpath)

file_re = options.file_re
# Handle the today and twoday options
now=datetime.datetime.utcnow()
delta=datetime.timedelta(days=1)
if(options.file_re == "today"):
  file_re=now.date().strftime("%Y%m%d")

if(options.file_re == "twoday"):
  then=now-delta
  a=now.date().strftime("%Y%m%d")
  b=then.date().strftime("%Y%m%d")
  file_re="%s|%s" % (a, b)

if(options.file_re == "fourday"):
  a=now.date().strftime("%Y%m%d")
  then=now-delta
  b=then.date().strftime("%Y%m%d")
  then=then-delta
  c=then.date().strftime("%Y%m%d")
  then=then-delta
  d=then.date().strftime("%Y%m%d")
  file_re="%s|%s|%s|%s" % (a, b, c, d)

if(file_re):
  cre=re.compile(file_re)

filelist = os.listdir(fulldirpath)

files=[]
if(file_re):
  for filename in filelist:
    if(cre.search(filename)):
      files.append(filename)
else:
  files = filelist

# Skip files with tmp in the filename
# Also require .fits in the filename
thefiles=[]
tmpcre = re.compile("tmp")
fitscre = re.compile(".fits")
logger.info("Checking for tmp files")
for filename in files:
  if(tmpcre.search(filename) or not fitscre.search(filename)):
    logger.info("skipping tmp file: %s" % filename)
  else:
    thefiles.append(filename)

i=0
n=len(thefiles)
# print what we're about to do, and give abort opportunity
logger.info("About to scan %d files" % n)
if (n>5000):
  logger.info("That's a lot of files. Hit ctrl-c within 5 secs to abort")
  time.sleep(6)

session = sessionfactory()

for filename in thefiles:
  i+=1
  logger.info("Queueing for Ingest: (%d/%d): %s" % (i, n, filename))
  addto_ingestqueue(session, filename, path)

session.close()
now=datetime.datetime.now()
logger.info("*** add_to_ingestqueue.py exiting normally at %s" % now)

