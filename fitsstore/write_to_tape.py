import sys
sys.path=['/opt/sqlalchemy/lib/python2.5/site-packages', '/astro/iraf/x86_64/gempylocal/lib/stsci_python/lib/python2.5/site-packages']+sys.path

import FitsStorage
from FitsStorageConfig import *
from FitsStorageLogger import *
from FitsStorageUtils import *
from FitsStorageTape import TapeDrive
import CadcCRC
import os
import re
import datetime
import time
import subprocess
import tarfile
import urllib
from xml.dom.minidom import parseString


# Option Parsing
from optparse import OptionParser
parser = OptionParser()
parser.add_option("--diskserver", action="store", type="string", dest="diskserver", default="fits", help="The Fits Storage Disk server to get the files from")
parser.add_option("--selection", action="store", type="string", dest="selection", help="the file selection criteria to use. This is a / separated list like in the URLs. Can be a date or daterange for example")
parser.add_option("--tapedrive", action="append", type="string", dest="tapedrive", help="tapedrive to use. Give this option multiple times to specify multiple drives")
parser.add_option("--tapelabel", action="append", type="string", dest="tapelabel", help="tape label of tape. Give this option multiple times to specify multiple tapes. Give the tapedrive and tapelabel arguments in the same order.")
parser.add_option("--dryrun", action="store_true", dest="dryrun", help="Dry Run - do not actually do anything")
parser.add_option("--dontcheck", action="store_true", dest="dontcheck", help="Don't rewind and check the tape label in the drive, go direct to eod and write")
parser.add_option("--debug", action="store_true", dest="debug", help="Increase log level to debug")
parser.add_option("--demon", action="store_true", dest="demon", help="Run as a background demon, do not generate stdout")

(options, args) = parser.parse_args()

# Logging level to debug? Include stdio log?
setdebug(options.debug)
setdemon(options.demon)

# Annouce startup
logger.info("*********  write_to_tape.py - starting up at %s" % datetime.datetime.now())

if(not options.selection):
  logger.error("You must specify a file selection")
  sys.exit(1)

if(len(options.tapedrive) < 1):
  logger.error("You must specify a tape drive")
  sys.exit(1)

if(len(options.tapedrive) != len(options.tapelabel)):
  logger.error("You must specify the same number of tape drives as tape labels")
  sys.exit(1)

logger.info("TapeDrive: %s; TapeLabel: %s" % (options.tapedrive, options.tapelabel))

logger.info("Fetching file list from disk server...")
# Get the list of files to put on tape from the server
url = "http://" + options.diskserver + "/xmlfilelist/" + options.selection
logger.debug("file list url: %s" % url)

u = urllib.urlopen(url)
xml = u.read()
u.close()

dom = parseString(xml)
files = []
totalsize=0
for fe in dom.getElementsByTagName("file"):
  dict = {}
  dict['filename']=fe.getElementsByTagName("filename")[0].childNodes[0].data
  dict['size']=int(fe.getElementsByTagName("size")[0].childNodes[0].data)
  dict['ccrc']=fe.getElementsByTagName("ccrc")[0].childNodes[0].data
  dict['lastmod']=fe.getElementsByTagName("lastmod")[0].childNodes[0].data
  files.append(dict)
  totalsize += dict['size']

numfiles = len(files)
logger.info("Got %d files totalling %.2f GB to write to tape" % (numfiles, (totalsize / 1.0E9)))
if(numfiles == 0):
  logger.info("Exiting - no files")
  exit(0)
 
# Make a list containing the tape device objects
tds = []
for i in range(0, len(options.tapedrive)):
  tds.append(TapeDrive(options.tapedrive[i], fits_tape_scratchdir))

session = sessionfactory()

# Get the database tape object for each tape label given
logger.debug("Finding tape records in DB")
tapes = []
for i in range(0, len(options.tapelabel)):
  query = session.query(Tape).filter(Tape.label == options.tapelabel[i]).filter(Tape.active == True)
  if(query.count() == 0):
    logger.error("Could not find active tape with label %s" % options.tapelabel[i])
    session.close()
    sys.exit(1)
  if(query.count() > 1):
    logger.error("Multiple active tapes with label %s:" % options.tapelabel[i])
    session.close()
    sys.exit(1)
  tapes.append(query.one())
  logger.debug("Found tape id in database: %d, label: %s" % (tapes[i].id, tapes[i].label))
  if(tapes[i].full):
    logger.error("Tape with label %s is full according to the DB. Exiting" % tapes[i].label)
    sys.exit(2)

# Check the tape label in the drives
if(not options.dontcheck):
  for i in range(0, len(options.tapelabel)):
    logger.info("Checking tape label in drive %s" % tds[i].dev)
    if(tds[i].online() == False):
      logger.error("No tape in drive %s" % tds[i].dev)
      session.close()
      sys.exit(1)
    thislabel = tds[i].readlabel()
    if(thislabel != options.tapelabel[i]):
      logger.error("Label of tape in drive %s: %s does not match label given as %s" % (tds[i].dev, thislabel, options.tapelabel[i]))
      session.close()
      sys.exit(1)
    logger.info("OK - found tape in drive %s with label: %s" % (tds[i].dev, thislabel))
    
# Copy the files to the local scratch, and check CRCs.
try:
  logger.info("Fetching files to local disk")
  tds[0].cdworkingdir()
  for f in files:
    filename = f['filename']
    size = int(f['size'])
    ccrc = f['ccrc']
    url="http://%s/file/%s" % (options.diskserver, filename)
    logger.info("Fetching file: %s from %s" % (filename, url))
    retcode=subprocess.call(['/usr/bin/curl', '-s', '-b', 'gemini_fits_authorization=good_to_go', '-O', '-f', url])
    if(retcode):
      # Curl command failed. Bail out
      logger.error("Fetch failed for url: %s" % url)
      tds[0].cdback()
      tds[0].cleanup()
      session.close()
      sys.exit(1)
    else:
      # Curl command suceeded.
      # Check the CRC of the file we got against the DB
      filecrc = CadcCRC.cadcCRC(filename)
      if(filecrc != ccrc):
        logger.error("CRC mismatch for file %s: file: %s, database: %s" % (filename, filecrc, ccrc))
        tds[0].cdback()
        tds[0].cleanup()
        session.close()
        sys.exit(1)
      # Check the md5sum of the file we got against the DB
      # Actually, the DB doesn't have md5 yet, so just calcultate it here for use later.
      md5sum = CadcCRC.md5sum(filename)
      f['md5sum'] = md5sum
  logger.info("All files fetched OK")
except:
  logger.error("Problem Fetching Files, aborting")
  tds[0].cdback()
  tds[0].cleanup()
  session.close()
  sys.exit(1)


# Now loop through the tapes, doing all the stuff on each
for i in range(0, len(tds)):
  td = tds[i]
  tape = tapes[i]

  logger.debug("About to write on tape label %s in drive %s" % (tape.label, td.dev))
  # Position Tape
  if(not options.dryrun):
    logger.info("Positioning Tape %s" % td.dev)
    td.setblk0()
    td.eod(fail=True)

    if(td.eot()):
      logger.error("Tape %s in %s is at End of Tape. Tape is Full. Marking tape as full in DB and aborting" % (tape.label, td.dev))
      tape.full = True
      session.commit()
      td.cleanup()
      td.cdback()
      session.close()
      sys.exit(1)

    # Update tape first/lastwrite
    logger.debug("Updating tape record for tape label %s" % tape.label)
    if(tape.firstwrite == None):
      tape.firstwrite = datetime.datetime.utcnow()
    tape.lastwrite = datetime.datetime.utcnow()
    session.commit()

    # Create tapewrite record
    logger.debug("Creating TapeWrite record for tape %s" % tape.label)
    tw = FitsStorage.TapeWrite()
    tw.tape_id = tape.id
    session.add(tw)
    session.commit()
    # Update tapewrite values pre-write
    tw.beforestatus = td.status()
    tw.filenum = td.fileno()
    tw.startdate = datetime.datetime.utcnow()
    tw.hostname = os.uname()[1]
    tw.tapedrive = td.dev
    tw.suceeded = False
    session.commit()

    # Write the tape.
    bytecount = 0
    blksize = 64 * 1024
    tarok = True
  
    logger.info("Creating tar archive on tape %s on drive %s" % (tape.label, td.dev))
    try:
      tar = tarfile.open(name=td.dev, mode='w|', bufsize=blksize)
    except:
      logger.error("Error opening tar archive - Exception: %s : %s" % (sys.exc_info()[0], sys.exc_info()[1]))
      tarok = False
    for f in files:
      filename = f['filename']
      size = int(f['size'])
      ccrc = f['ccrc']
      md5 = f['md5sum']
      lastmod = f['lastmod']
      logger.info("Adding %s to tar file on tape %s in drive %s" % (filename, tape.label, td.dev))
      try:
      # the filename is a unicode string, and tarfile cannot handle this, convert to ascii
        filename = filename.encode('ascii')
        tar.add(filename)
      except:
        logger.error("Error adding file to tar archive - Exception: %s : %s" % (sys.exc_info()[0], sys.exc_info()[1]))
        logger.info("Probably the tape filled up - Marking tape as full in the DB - label: %s" % tape.label)
        tape.full = True
        session.commit()
        tarok = False
        break
      # Create the TapeFile entry and add to DB
      tapefile = FitsStorage.TapeFile()
      tapefile.tapewrite_id = tw.id
      tapefile.filename = filename
      tapefile.ccrc = ccrc
      tapefile.md5 = md5
      tapefile.lastmod = lastmod
      tapefile.size = size
      session.add(tapefile)
      session.commit()
      # Keep a running total of bytes written
      bytecount += size
    logger.info("Completed writing tar archive on tape %s in drive %s" % (tape.label, td.dev))
    logger.info("Wrote %d bytes = %.2f GB" % (bytecount , (bytecount/1.0E9)))
    try:
      tar.close()
    except:
      logger.error("Error closing tar archive - Exception: %s : %s" % (sys.exc_info()[0], sys.exc_info()[1]))
      tarok = False

    # update records post-write
    logger.debug("Updating tapewrite record")
    tw.enddate = datetime.datetime.utcnow()
    logger.debug("Suceeded: %s" % tarok)
    tw.suceeded = tarok
    tw.afterstatus = td.status()
    tw.size = bytecount
    session.commit()
   
logger.info("Cleaning up disk staging files")
tds[0].cleanup()
tds[0].cdback()
session.close()
logger.info("*** write_to_tape exiting normally at %s" % datetime.datetime.now())

