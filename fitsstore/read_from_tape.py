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
import tarfile


# Option Parsing
from optparse import OptionParser
parser = OptionParser()
parser.add_option("--tapedrive", action="store", type="string", dest="tapedrive", help="tapedrive to use.")
parser.add_option("--file-re", action="store", type="string", dest="filere", help="Regular expression used to select files to extract")
parser.add_option("--all", action="store_true", dest="all", help="When multiple versions of a file are on tape, get them all, not just the most recent")
parser.add_option("--dryrun", action="store_true", dest="dryrun", help="Dry Run - do not actually do anything")
parser.add_option("--debug", action="store_true", dest="debug", help="Increase log level to debug")
parser.add_option("--demon", action="store_true", dest="demon", help="Run as a background demon, do not generate stdout")
parser.add_option("--requester", action="store", type="string", dest="requester", help="filters the table for specific filenames")

(options, args) = parser.parse_args()

# Logging level to debug? Include stdio log?
setdebug(options.debug)
setdemon(options.demon)
requester = options.requester

# Annouce startup
logger.info("*********  read_from_tape.py - starting up at %s" % datetime.datetime.now())
options.filere = '2009101'
options.tapedrive = 1

if(not options.filere):
  logger.error("You must specify a file-re")
  sys.exit(1)

if(not options.tapedrive):
  logger.error("You must specify a tape drive")
  sys.exit(1)

# Query the DB to find a list of files to extract
# This is a little non trivial, given that there are multiple identical
# copies of the file on several tapes and also that there can be multiple
# non identical version of the file on tapes too.
session = sessionfactory()


query = session.query(TapeRead).all()
while(range(len(query))):

  findlabels = session.query(TapeRead.tape_label).distinct().all()
  tapelabels = []
  for find in findlabels:
    label = find[0].encode()
    tapelabels.append(label)
  print "Choose a tape from these tape labels: %s" % tapelabels

  #FitsStorageTape.readlabel()

  it = raw_input("Which tape would you like to read? ")

  filenums = session.query(TapeRead.filenum).filter(TapeRead.tape_label==it).order_by(TapeRead.filenum).distinct().all()

  #FitsStorageTape.rewind()

  for nums in filenums:

    #FitsStorageTape.fastforward()

    filename = session.query(TapeRead.filename).filter(TapeRead.tape_label==it).filter(TapeRead.filenum==nums[0]).all()

    filenames = []
    for name in filename:
      fn = name[0].encode()
      filenames.append(fn)
      print "read file %s" % name[0]


#      tar = tarfile.open("file.tar")
#      def py_files(members):
#        for tarinfo in members:
#          print "tarinfo.name: %s" % tarinfo.name
#          yield tarinfo

#      print "tar: %s" % tar
#      tar.extractall(members=py_files(tar))

#      session.query(TapeRead).filter(TapeRead.filename==fn).delete()
#      session.flush()
#      logger.info("removing file %s from taperead" % fn)



    def py_files(members):
      for tarinfo in members:
        if tarinfo.name in filenames:
          print "tarinfo.name: %s" % tarinfo.name
          session.query(TapeRead).filter(TapeRead.filename==tarinfo.name).delete()
          session.flush()
          logger.info("removing file %s from taperead" % tarinfo.name)
          yield tarinfo

    tar = tarfile.open("file.tar")
    tar.extractall(members=py_files(tar))
    print "tar: %s" % tar

  query = session.query(TapeRead).all()



tar.close()
session.close()

