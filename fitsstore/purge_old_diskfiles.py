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

# Option Parsing
from optparse import OptionParser
parser = OptionParser()
parser.add_option("--dryrun", action="store_true", dest="dryrun", help="Dry Run - do not actually do anything")
parser.add_option("--file-pre", action="store", dest="filepre", help="filename prefix to elimanate non canonical diskfiles for")
parser.add_option("--debug", action="store_true", dest="debug", help="Increase log level to debug")
parser.add_option("--demon", action="store_true", dest="demon", help="Run as a background demon, do not generate stdout")

(options, args) = parser.parse_args()

# Logging level to debug? Include stdio log?
setdebug(options.debug)
setdemon(options.demon)

# Annouce startup
logger.info("*********  purge_old_diskfiles.py - starting up at %s" % datetime.datetime.now())

if(len(options.filepre) < 1):
  logger.error("You must specify a file-pre")
  sys.exit(1)

session = sessionfactory()

logger.info("Getting list of file_ids to check")
likestr = "%s%%" % options.filepre
logger.debug("Matching File.filename LIKE %s" % likestr)
query = session.query(File.id).filter(File.filename.like(likestr))
fileids = query.all()

logger.info("Got %d files to check" % len(fileids))

if(len(fileids)==0):
  session.close()
  logger.info("No files to check, exiting")
  sys.exit(0)


# Loop through the file ids
for fileid in fileids:
  #logger.debug("Checking file_id %d" % fileid)
  query = session.query(DiskFile).filter(DiskFile.file_id==fileid).filter(DiskFile.present==False).filter(DiskFile.canonical==False)
  todelete = query.all()
  if(len(todelete)>0):
    logger.info("Found diskfiles to delete for file_id %d" % fileid)
    for diskfile in todelete:
      logger.debug("Need to delete diskfile id %d" % diskfile.id)
      # Are there any header rows?
      hquery = session.query(Header).filter(Header.diskfile_id==diskfile.id)
      for header in hquery.all():
        logger.debug("Need to delete header id %d" % header.id)
        # Are there any instrument headers that need deleting?
        gquery = session.query(Gmos).filter(Gmos.header_id==header.id)
        for g in gquery.all():
          if(not options.dryrun):
            logger.debug("Deleting GMOS id %d" % g.id)
            session.delete(g)
            session.commit()
          else:
            logger.debug("Dry Run - would delete GMOS id %d" % g.id)
        nquery = session.query(Niri).filter(Niri.header_id==header.id)
        for n in nquery.all():
          if(not options.dryrun):
            logger.debug("Deleting NIRI id %d" % n.id)
            session.delete(n)
            session.commit()
          else:
            logger.debug("Dry Run - would delete Niri id %d" % n.id)
        if(not options.dryrun):
          logger.debug("Deleting header id %d" % header.id)
          session.delete(header)
          session.commit()
        else:
          logger.debug("Dry Run - not deleting Header id %d" % header.id)
      if(not options.dryrun):
        logger.debug("Deleting diskfile id %d" % diskfile.id)
        session.delete(diskfile)
        session.commit()
      else:
        logger.debug("Dry Run - not deleting DiskFile id %d" % diskfile.id)

session.close()
logger.info("*** purge_old_diskfiles exiting normally at %s" % datetime.datetime.now())

