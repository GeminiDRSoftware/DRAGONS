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
from sqlalchemy import or_


from optparse import OptionParser

parser = OptionParser()
parser.add_option("--inst", action="store", dest="inst", help="Instrument table to rebuild")
parser.add_option("--debug", action="store_true", dest="debug", help="Increase log level to debug")
parser.add_option("--demon", action="store_true", dest="demon", help="Run as a background demon, do not generate stdout")

(options, args) = parser.parse_args()

# Logging level to debug? Include stdio log?
setdebug(options.debug)
setdemon(options.demon)

# Define signal handler. This allows us to bail out neatly if we get a signal
def handler(signum, frame):
  logger.info("Received signal: %d " % signum)
  raise Exception('Signal', signum)

# Set handlers for the signals we want to handle
signal.signal(signal.SIGHUP, handler)
signal.signal(signal.SIGINT, handler)
signal.signal(signal.SIGQUIT, handler)
signal.signal(signal.SIGTERM, handler)

# Annouce startup
now = datetime.datetime.now()
logger.info("*********  rebuild_inst_table.py - starting up at %s" % now)

inst = options.inst

if(inst == 'gmos'):
  logger.info("Rebuilding gmos table")

  session = sessionfactory()

  try:
    # Get a list of header ids for which there is a present diskfile for this instrument
    query = session.query(Header.id).select_from(join(Header, DiskFile))
    query = query.filter(or_(Header.instrument == 'GMOS-N', Header.instrument == 'GMOS-S'))
    query = query.filter(DiskFile.present == True)
    headers = query.all()
    count = len(headers)
    logger.info("Found %s files to process" % count)

    i=1
    for id in headers:
      id=id[0]
      # Does it an instheader for this header id already esist?
      already = session.query(Gmos).filter(Gmos.header_id == id).count()
      if(already==0):
        # No, we should add it.
        query = session.query(Header).filter(Header.id == id)
        header = query.one() 
        #logger.info("Processing %s (%d/%d)" % (header.diskfile.file.filename, i, count))
        logger.info("Processing %d/%d" % (i, count))
        gmos = Gmos(header)
        session.add(gmos)
        session.commit()
      i+= 1
    

  except:
    logger.error("Exception: %s : %s" % (sys.exc_info()[0], sys.exc_info()[1]))
    traceback.print_tb(sys.exc_info()[2])

  finally:
    session.close()


if(inst == 'niri'):
  logger.info("Rebuilding NIRI table")

  session = sessionfactory()

  try:
    # Get a list of header ids for which there is a present diskfile for this instrument
    query = session.query(Header.id).select_from(join(Header, DiskFile))
    query = query.filter(Header.instrument == 'NIRI')
    query = query.filter(DiskFile.present == True)
    headers = query.all()
    count = len(headers)
    logger.info("Found %s files to process" % count)

    i=1
    for id in headers:
      id = id[0]
      # Does it an instheader for this header id already esist?
      already = session.query(Niri).filter(Niri.header_id == id).count()
      if(already==0):
        # No, we should add it.
        query = session.query(Header).filter(Header.id == id)
        header = query.one()
        logger.info("Processing %s (%d/%d)" % (header.diskfile.file.filename, i, count))
        niri = Niri(header)
        session.add(niri)
        session.commit()
      i+= 1


  except:
    logger.error("Exception: %s : %s" % (sys.exc_info()[0], sys.exc_info()[1]))
    traceback.print_tb(sys.exc_info()[2])

  finally:
    session.close()



logger.info("*********  rebuild_inst_table.py - exiting at %s" % datetime.datetime.now())
