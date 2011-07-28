"""
This module drops and creates the tables, then refills them with 
"""
import sys
sys.path += ['/opt/sqlalchemy/lib/python2.5/site-packages']

from FitsStorageUtils.CreateTables import create_tables, drop_tables
from FitsStorageConfig import *
from FitsStorage import *
from FitsStorageLogger import *
from xml.dom.minidom import parseString

import datetime
import urllib

# Option Parsing
from optparse import OptionParser
parser = OptionParser()
parser.add_option("--tapeserver", action="store", type="string", dest="tapeserver", default="hbffitstape1", help="The Fits Storage Tape server to use to check the files are on tape")
parser.add_option("--debug", action="store_true", dest="debug", help="Increase log level to debug")
parser.add_option("--demon", action="store_true", dest="demon", help="Run as a background demon, do not generate stdout")

(options, args) = parser.parse_args()

# Logging level to debug? Include stdio log?
setdebug(options.debug)
setdemon(options.demon)


# Annouce startup
logger.info("*********  re_create_tables.py - starting up at %s" % datetime.datetime.now())

session = sessionfactory()

# Open the xmltape network object for reading
u = urllib.urlopen("http://%s/xmltape/" % options.tapeserver)
xml = u.read()
u.close()

dom = parseString(xml)


# Rebuild tables: Drop the tables
session.execute("DROP TABLE tape, tapewrite, tapefile CASCADE")
session.commit()

# Create the tables
File.metadata.create_all(bind=pg_db)

if fsc_localmode == False:
  # Now grant the apache user select on them for the www queries
  session.execute("GRANT SELECT ON tape, tape_id_seq, tapewrite, tapefile TO apache");
  session.execute("GRANT INSERT,UPDATE ON tape, tape_id_seq TO apache");
session.commit()


try:
  # Get tape tags
  tapeelements = dom.getElementsByTagName("tape")

  for te in tapeelements:
    print "in first for"
    label = te.getElementsByTagName("label")[0].childNodes[0].data
    firstwrite = te.getElementsByTagName("firstwrite")[0].childNodes[0].data
    lastwrite = te.getElementsByTagName("lastwrite")[0].childNodes[0].data
    lastverified = te.getElementsByTagName("lastwrite")[0].childNodes[0].data
    location = te.getElementsByTagName("location")[0].childNodes[0].data
    lastmoved = te.getElementsByTagName("lastmoved")[0].childNodes[0].data
    active = te.getElementsByTagName("active")[0].childNodes[0].data
    full = te.getElementsByTagName("full")[0].childNodes[0].data
    set = te.getElementsByTagName("set")[0].childNodes[0].data
    fate = te.getElementsByTagName("fate")[0].childNodes[0].data

    # Make tape object and add it to the database (fill tables)
    tape = Tape(label)
    session.add(tape)
    tape.firstwrite = firstwrite
    tape.lastwrite = lastwrite
    tape.lastverified = lastverified
    tape.location = location
    tape.lastmoved = lastmoved
    tape.active = active
    tape.full = full
    tape.set = set
    tape.fate = fate
    session.commit()

    # Get tapewrite tags
    tapewriteelements = te.getElementsByTagName("tapewrite")

    for twe in tapewriteelements:
      print "in second for"
      filenum = twe.getElementsByTagName("filenum")[0].childNodes[0].data
      startdate = twe.getElementsByTagName("startdate")[0].childNodes[0].data
      enddate = twe.getElementsByTagName("enddate")[0].childNodes[0].data
      suceeded = twe.getElementsByTagName("suceeded")[0].childNodes[0].data
      size = twe.getElementsByTagName("size")[0].childNodes[0].data
      beforestatus = twe.getElementsByTagName("beforestatus")[0].childNodes[0].data
      afterstatus = twe.getElementsByTagName("afterstatus")[0].childNodes[0].data
      hostname = twe.getElementsByTagName("hostname")[0].childNodes[0].data
      tapedrive = twe.getElementsByTagName("tapedrive")[0].childNodes[0].data
      notes = twe.getElementsByTagName("notes")[0].childNodes[0].data

      # Make tapewrite object and add it to the database (fill tables)
      tapewrite = TapeWrite()
      session.add(tapewrite)
      tapewrite.tape_id = tape.id
      tapewrite.filenum = filenum
      tapewrite.startdate = startdate
      tapewrite.enddate = enddate
      tapewrite.suceeded = suceeded
      tapewrite.size = size
      tapewrite.beforestatus = beforestatus
      tapewrite.afterstaus = afterstatus
      tapewrite.hostname = hostname
      tapewrite.tapedrive = tapedrive
      tapewrite.notes = notes
      session.commit()

      # Get tapefile tags
      tapefileelements = twe.getElementsByTagName("tapefile")

      for tfe in tapefileelements:
        print "in third for"
        filename = tfe.getElementsByTagName("filename")[0].childNodes[0].data
        size = tfe.getElementsByTagName("size")[0].childNodes[0].data
        ccrc = tfe.getElementsByTagName("ccrc")[0].childNodes[0].data
        md5 = tfe.getElementsByTagName("md5")[0].childNodes[0].data
        lastmod = tfe.getElementsByTagName("lastmod")[0].childNodes[0].data

        # Make tapefile object and add it to the database (fill tables)
        tapefile = TapeFile()
        session.add(tapefile)
        tapefile.tapewrite_id = tapewrite.id
        tapefile.filename = filename
        tapefile.size = size
        tapefile.ccrc = ccrc
        tapefile.md5 = md5
        tapefile.lastmod = lastmod
        session.commit()

except (TypeError):
  pass

finally:
  session.close()

  logger.info("**check_on_tape.py exiting normally")


