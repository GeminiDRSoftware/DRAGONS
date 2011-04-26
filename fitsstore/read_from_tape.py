import sys
sys.path=['/opt/sqlalchemy/lib/python2.5/site-packages', '/astro/iraf/x86_64/gempylocal/lib/stsci_python/lib/python2.5/site-packages']+sys.path

import FitsStorage
from FitsStorageConfig import *
from FitsStorageLogger import *
from FitsStorageUtils import *
from FitsStorageTape import TapeDrive
#from subprocess import call
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
parser.add_option("--tapedrive", action="store", type="string", default="/dev/nst0", dest="tapedrive", help="tapedrive to use.")
parser.add_option("--file-re", action="store", type="string", dest="filere", help="Regular expression used to select files to extract")
parser.add_option("--requester", action="store", type="string", dest="requester", help="filters the table for specific filenames")
parser.add_option("--list-tapes", action="store_true", dest="list_tapes", help="only lists the tapes in TapeRead")
parser.add_option("--all", action="store_true", dest="all", help="When multiple versions of a file are on tape, get them all, not just the most recent")
parser.add_option("--dryrun", action="store_true", dest="dryrun", help="Dry Run - do not actually do anything")
parser.add_option("--debug", action="store_true", dest="debug", help="Increase log level to debug")
parser.add_option("--demon", action="store_true", dest="demon", help="Run as a background demon, do not generate stdout")

(options, args) = parser.parse_args()

# Logging level to debug? Include stdio log?
setdebug(options.debug)
setdemon(options.demon)
requester = options.requester

# Query the DB to find a list of files to extract
# This is a little non trivial, given that there are multiple identical
# copies of the file on several tapes and also that there can be multiple
# non identical version of the file on tapes too.
session = sessionfactory()

# If --list-tapes is called then only list the tapes from the TapeRead table
if options.list_tapes:
  findlabels = session.query(TapeRead.tape_label).distinct().all()
  tapelabels = []
  for find in findlabels:
    label = find[0].encode()
    tapelabels.append(label)
  print "Choose a tape from these tape labels: %s" % tapelabels
  sys.exit(1)


# Annouce startup
logger.info("*********  read_from_tape.py - starting up at %s" % datetime.datetime.now())


try:
  # Make a FitsStorageTape object from class TapeDrive initializing the device and scratchdir
  td = TapeDrive(options.tapedrive, FitsStorageConfig.fits_tape_scratchdir)
  label = td.readlabel()
  print "You are reading from this tape: %s" % label

  # Make a working directory
  td.mkworkingdir()
  td.cdworkingdir()
  td.setblk0()

  # Choose the filenums from the tape in the tapedrive
  filenums = session.query(TapeRead.filenum).filter(TapeRead.tape_label==label).order_by(TapeRead.filenum).distinct().all()

  td.rewind()

  # Find the (TarFile) information for each 'track' on the tape
  for nums in filenums:
    # Go to the next 'track'
    td.skipto(filenum=nums[0])

    # Query the filenames at the filenum and make a list of filenames
    filename = session.query(TapeRead.filename).filter(TapeRead.tape_label==label).filter(TapeRead.filenum==nums[0]).all()
    filenames = []
    for name in filename:
      fn = name[0].encode()
      filenames.append(fn)

    # A function to yeild all tarinfo objects and 'delete' all the read files
    blksize = 64*1024
    def fits_files(members):
      for tarinfo in members:
        if tarinfo.name in filenames:
          session.query(TapeRead).filter(TapeRead.filename==tarinfo.name).delete()
          session.commit()
          logger.info("removing file %s from taperead" % tarinfo.name)
          yield tarinfo

    # Open the tarfiles on the specific tape and extract the tarinfo object
    tar = tarfile.open(name=options.tapedrive, mode='r|', bufsize=blksize)
    tar.extractall(members=fits_files(tar))
    tar.close()

  # Are there any more files in TapeRead?
  taperead = session.query(TapeRead).all()
  if(range(len(taperead))):
    findlabels = session.query(TapeRead.tape_label).distinct().all()
    tapelabels = []
    for find in findlabels:
      label = find[0].encode()
      tapelabels.append(label)
    print "There are still more values in taperead that have not been read on these tapes: %s." % tapelabels
  else:
    print "There are no more values in taperead that haven't been read."

  # Rebuild TapeRead without rows deleted

finally:
  td.cdback()
  session.close()

