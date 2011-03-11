import sys
sys.path=['/opt/sqlalchemy/lib/python2.5/site-packages', '/astro/iraf/x86_64/gempylocal/lib/stsci_python/lib/python2.5/site-packages']+sys.path

import FitsStorage
from FitsStorageConfig import *
from FitsStorageLogger import *
from FitsStorageUtils import *
from sqlalchemy import *
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
parser.add_option("--file-re", action="store", type="string", dest="filere", help="Regular expression used to select files to extract")
parser.add_option("--all", action="store_true", dest="all", help="When multiple versions of a file are on tape, get them all, not just the most recent")
parser.add_option("--dryrun", action="store_true", dest="dryrun", help="Dry Run - do not actually do anything")
parser.add_option("--debug", action="store_true", dest="debug", help="Increase log level to debug")
parser.add_option("--demon", action="store_true", dest="demon", help="Run as a background demon, do not generate stdout")
parser.add_option("--requester", action="store", type="string", dest="requester", help="name of whoever requested it")
(options, args) = parser.parse_args()

# Logging level to debug? Include stdio log?
setdebug(options.debug)
setdemon(options.demon)
requester = options.requester

# Annouce startup
logger.info("*********  request_from_tape.py - starting up at %s" % datetime.datetime.now())

if(not options.filere):
  logger.error("You must specify a file-re")
  sys.exit(1)

# Only populate one day at a time or a sequence of days separated by semicolons!!!
date = re.sub("\D", "", options.filere)
if(len(date)!=8):
  logger.info("You must specify one day at a time otherwise this script executes really slowly!")
  logger.info("Hit ctrl-c within 5 secs to abort")
  time.sleep(8)

# Query the DB to find a list of files to extract
# This is a little non trivial, given that there are multiple identical
# copies of the file on several tapes and also that there can be multiple
# non identical version of the file on tapes too.
session = sessionfactory()


# First, we want to query the Tape, TapeWrite and TapeFile tables to populate TapeRead
#query = session.query(TapeFile).select_from(join(TapeFile, join(TapeWrite, Tape)))
query = session.query(TapeFile).select_from(Tape, TapeWrite, TapeFile)
query = query.filter(Tape.id == TapeWrite.tape_id).filter(TapeWrite.id == TapeFile.tapewrite_id)
query = query.filter(TapeFile.filename.like('%'+options.filere+'%'))
query = query.filter(TapeWrite.suceeded == True)
query = query.filter(Tape.active == True)
query = query.order_by(TapeFile.filename, desc(TapeFile.lastmod))
query = query.distinct()
listed = query.all()

# Now, where the same filename occurs with multiple md5s, we should weed out
# the ones we don't want while populating the TapeRead columns
i=0
previous_file = ''
if(not options.all):
  for que in listed:
    i+=1
    this_file = que.filename
    if(previous_file != this_file):
      tr = TapeRead()
      tr.tapefile_id = que.id
      tr.filename = que.filename
      tr.md5 = que.md5
      tr.tape_id = que.tapewrite.tape.id
      tr.tape_label = que.tapewrite.tape.label
      tr.filenum = que.tapewrite.filenum
      tr.requester = options.requester
      session.add(tr)
      logger.info("adding file %s to taperead: (%d in queue)" % (que.filename, i))
      session.commit()
    previous_file = this_file
session.close()


