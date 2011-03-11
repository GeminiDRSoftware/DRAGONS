import sys
sys.path=['/opt/sqlalchemy/lib/python2.5/site-packages', '/astro/iraf/x86_64/gempylocal/lib/stsci_python/lib/python2.5/site-packages']+sys.path

import FitsStorage
from FitsStorageConfig import *
from FitsStorageLogger import *
from FitsStorageUtils import *
import CadcCRC
import datetime
import urllib
from xml.dom.minidom import parseString
import os
import smtplib


# Option Parsing
from optparse import OptionParser
parser = OptionParser()
parser.add_option("--tapeserver", action="store", type="string", dest="tapeserver", default="hbffitstape1", help="The Fits Storage Tape server to use to check the files are on tape")
parser.add_option("--file-pre", action="store", type="string", dest="filepre", help="File prefix to operate on, eg N20090130, N200812 etc")
parser.add_option("--notpresent", action="store_true", dest="notpresent", help="Include files that are marked as not present")
parser.add_option("--mintapes", action="store", type="int", dest="mintapes", default=2, help="Minimum number of tapes file must be on to be eligable for deletion")
parser.add_option("--tapeset", action="store", type="int", dest="tapeset", help="Only consider tapes in this tapeset")
parser.add_option("--debug", action="store_true", dest="debug", help="Increase log level to debug")
parser.add_option("--demon", action="store_true", dest="demon", help="Run as a background demon, do not generate stdout")

(options, args) = parser.parse_args()

# Logging level to debug? Include stdio log?
setdebug(options.debug)
setdemon(options.demon)


msg = ""
# Annouce startup
logger.info("*********  check_on_tape.py - starting up at %s" % datetime.datetime.now())

session = sessionfactory()

query = session.query(DiskFile.id).select_from(join(File, DiskFile)).filter(DiskFile.canonical==True)

if(options.filepre):
  likestr = "%s%%" % options.filepre
  query = query.filter(File.filename.like(likestr))

if(not options.notpresent):
  query = query.filter(DiskFile.present==True)

query = query.order_by(File.filename)

diskfileids = query.all()

if(len(diskfileids) == 0):
  logger.info("No Files found matching file-pre. Exiting")
  session.close()
  sys.exit(0)

logger.info("Got %d files to check" % len(diskfileids))

sumbytes = 0
sumfiles = 0

for diskfileid in diskfileids:

  diskfile = session.query(DiskFile).filter(DiskFile.id == diskfileid).one()

  fullpath = diskfile.file.fullpath()
  dbmd5 = diskfile.md5
  dbfilename = diskfile.file.filename

  url = "http://%s/fileontape/%s" % (options.tapeserver, dbfilename)
  logger.debug("Querying tape server DB at %s" % url)

  u = urllib.urlopen(url)
  xml = u.read()
  u.close()

  dom = parseString(xml)

  fileelements = dom.getElementsByTagName("file")

  tapeids = []
  for fe in fileelements:
    filename = fe.getElementsByTagName("filename")[0].childNodes[0].data
    md5 = fe.getElementsByTagName("md5")[0].childNodes[0].data
    tapeid = int(fe.getElementsByTagName("tapeid")[0].childNodes[0].data)
    tapeset = int(fe.getElementsByTagName("tapeset")[0].childNodes[0].data)
    logger.debug("Filename: %s; md5=%s, tapeid=%d, tapeset=%d" % (filename, md5, tapeid, tapeset))
    if((filename == dbfilename) and (md5 == dbmd5) and (tapeid not in tapeids)):
      logger.debug("Found it on tape id %d" % tapeid)
      if(options.tapeset is not None and tapeset != options.tapeset):
        logger.debug("But this tape id is not in the requested tapeset")
      else:
        tapeids.append(tapeid)

  if(len(tapeids) < options.mintapes):
    sumbytes += diskfile.size
    sumfiles += 1
    logger.info("*** File %s - %s needs to go to tape, it is on %d tapes: %s" % (fullpath, dbmd5, len(tapeids), tapeids))
  else:
    logger.info("File %s - %s is OK, it already is on %d tapes: %s" % (fullpath, dbmd5, len(tapeids), tapeids))
    

logger.info("Found %d files totalling %.2f GB that should go to tape" % (sumfiles, sumbytes/1.0E9))

session.close()

logger.info("**check_on_tape.py exiting normally")
