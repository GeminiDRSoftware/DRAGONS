#!/usr/bin/env python
import sys
sys.path=['/opt/sqlalchemy/lib/python2.5/site-packages', '/astro/iraf/x86_64/gempylocal/lib/stsci_python/lib/python2.5/site-packages']+sys.path

import FitsStorage
import FitsStorageConfig
from FitsStorageUtils import *
from FitsStorageLogger import *
import signal
import os
import re
import datetime
import time
import traceback

from optparse import OptionParser

fsc_localmode = FitsStorageConfig.fsc_localmode
parser = OptionParser()
parser.add_option("--force-crc", action="store_true", dest="force_crc", help="Force crc check on pre-existing files")
parser.add_option("--skip-fv", action="store_true", dest="skip_fv", help="Do not run fitsverify on the files")
parser.add_option("--skip-wmd", action="store_true", dest="skip_wmd", help="Do not run a wmd check on the files")
parser.add_option("--debug", action="store_true", dest="debug", help="Increase log level to debug")
parser.add_option("--demon", action="store_true", dest="demon", help="Run as a background demon, do not generate stdout")
parser.add_option("--lockfile", action="store", dest="lockfile", help="Use this as a lockfile to limit instances")
parser.add_option("--empty", action="store_true", default=False, dest="empty", help="This flag indicates that service ingest queue should empty the current queue and then exit.")
(options, args) = parser.parse_args()

# Logging level to debug? Include stdio log?
setdebug(options.debug)
setdemon(options.demon)

# Define signal handler. This allows us to bail out neatly if we get a signal
def handler(signum, frame):
  logger.info("Received signal: %d " % signum)
  raise KeyboardInterrupt('Signal', signum)

# Set handlers for the signals we want to handle
signal.signal(signal.SIGHUP, handler)
signal.signal(signal.SIGINT, handler)
signal.signal(signal.SIGQUIT, handler)
signal.signal(signal.SIGTERM, handler)

# Annouce startup
now = datetime.datetime.now()
logger.info("*********  service_ingest_queue.py - starting up at %s" % now)

if(options.lockfile):
  # Does the Lockfile exist?
  lockfile = "%s/%s" % (FitsStorageConfig.fits_lockfile_dir, options.lockfile)
  if(os.path.exists(lockfile)):
    logger.info("Lockfile %s already exists, bailing out" % lockfile)
    sys.exit()
  else:
    logger.info("Creating lockfile %s" % lockfile)
    open(lockfile, 'w').close()

session = sessionfactory()

# Loop forever.
loop = True
while(loop):
  try:
    # Request a queue entry
    iq = pop_ingestqueue(session)

    if(iq==None):
      logger.info("Didn't get anything to ingest, retrying")
      iq = pop_ingestqueue(session)
    if(iq==None):
      logger.info("Nothing on queue.")
      if options.empty:
        logger.info("--empty flag set, exiting")
        break
      else:
        logger.info("...Waiting")
      time.sleep(10)
    else:
      logger.info("Ingesting %s, (%d in queue)" % (iq.filename, ingestqueue_length(session)))
      if not fsc_localmode:
        session.begin_nested()
      else:
        session.begin(subtransactions=True)
      try:
        ingest_file(session, iq.filename, iq.path, options.force_crc, options.skip_fv, options.skip_wmd)
        session.commit()
      except:
        logger.info("Problem Ingesting File - Rolling back" )
        session.rollback()
        iq.inprogress=False
        session.commit()
        raise
      logger.debug("Deleteing ingestqueue id %d" % iq.id)
      session.delete(iq)
      session.commit()

  except KeyboardInterrupt:
    loop=False
    break

  except:
    string = traceback.format_tb(sys.exc_info()[2])
    if(iq):
      logger.error("File %s - Exception: %s : %s... %s" % (iq.filename, sys.exc_info()[0], sys.exc_info()[1], string))
    else:
      logger.error("Nothing on ingest queue - Exception: %s : %s... %s" % (sys.exc_info()[0], sys.exc_info()[1], string))
  

  finally:
    session.close()

session.close()
if(options.lockfile):
  logger.info("Deleting Lockfile %s" % lockfile)
  os.unlink(lockfile)
logger.info("*********  service_ingest_queue.py - exiting at %s" % datetime.datetime.now())

