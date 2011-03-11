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

# Option Parsing
from optparse import OptionParser
parser = OptionParser()
parser.add_option("--file", action="store", type="string", dest="filename", help="Standards text filename")
parser.add_option("--clean", action="store_true", dest="clean", help="Delete all rows in the table before adding")
parser.add_option("--debug", action="store_true", dest="debug", help="Increase log level to debug")

(options, args) = parser.parse_args()

# Logging level to debug?
setdebug(options.debug)

# Annouce startup
now = datetime.datetime.now()
logger.info("*********  ingest_standards.py - starting up at %s" % now)

session = sessionfactory()

if(options.clean):
  logger.info("Deleting all rows in standards table")
  session.execute("DELETE FROM standards")

ingest_standards(session, options.filename)

session.close()
now=datetime.datetime.now()
logger.info("*** ingest_standards exiting normally at %s" % now)

