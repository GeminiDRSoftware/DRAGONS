from FitsStorageConfig import *
import datetime
import subprocess
import re
import os

from FitsStorageLogger import *

from optparse import OptionParser
parser = OptionParser()
parser.add_option("--dryrun", action="store_true", dest="dryrun", help="Don't actually delete anything, just say what would be deleted")
parser.add_option("--debug", action="store_true", dest="debug", help="Increase log level to debug")
parser.add_option("--demon", action="store_true", dest="demon", help="Run as a background demon, do not generate stdout")

(options, args) = parser.parse_args()

# Logging level to debug? Include stdio log?
setdebug(options.debug)
setdemon(options.demon)

# Annouce startup
split_date = datetime.datetime.now().isoformat().split('T')[0].split('-')
logger.info("*********  database_cleaner.py - starting up at %s" % split_date)

# Strip date
year = int(split_date[0])
month = int(split_date[1])
day = int(split_date[2])
now = datetime.date(year, month, day)

# Directory name in FitsStorageUtils = fits_db_backup_dir
db_backup = os.listdir(fits_db_backup_dir)

# The filename looks like:
#filenames = ["fitsdata.2011-04-08T07:30:01.886249.pg_dump_c"]
#filenames = ["fitsdata.2011-04-08T12:56:22.886036.pg_dump_c", "fitsdata.2011-04-07T12:56:22.886036.pg_dump_c", "fitsdata.2011-04-06T12:56:22.886036.pg_dump_c", "fitsdata.2011-04-05T12:56:22.886036.pg_dump_c", "fitsdata.2011-04-04T12:56:22.886036.pg_dump_c", "fitsdata.2011-04-03T12:56:22.886036.pg_dump_c", "fitsdata.2011-04-02T12:56:22.886036.pg_dump_c", "fitsdata.2011-04-01T12:56:22.886036.pg_dump_c", "fitsdata.2011-03-31T12:56:22.886036.pg_dump_c", "fitsdata.2011-03-30T12:56:22.886036.pg_dump_c", "fitsdata.2011-03-29T12:56:22.886036.pg_dump_c", "fitsdata.2011-03-28T12:56:22.886036.pg_dump_c", "fitsdata.2011-03-27T12:56:22.886036.pg_dump_c", "fitsdata.2011-03-26T12:56:22.886036.pg_dump_c", "fitsdata.2011-03-25T12:56:22.886036.pg_dump_c", "fitsdata.2011-03-24T12:56:22.886036.pg_dump_c", "fitsdata.2011-03-23T12:56:22.886036.pg_dump_c", "fitsdata.2011-03-22T12:56:22.886036.pg_dump_c", "fitsdata.2011-03-21T12:56:22.886036.pg_dump_c", "fitsdata.2011-03-20T12:56:22.886036.pg_dump_c", "fitsdata.2011-03-19T12:56:22.886036.pg_dump_c", "fitsdata.2011-03-18T12:56:22.886036.pg_dump_c", "fitsdata.2011-03-17T12:56:22.886036.pg_dump_c", "fitsdata.2011-03-16T12:56:22.886036.pg_dump_c", "fitsdata.2011-03-15T12:56:22.886036.pg_dump_c", "fitsdata.2011-03-14T12:56:22.886036.pg_dump_c", "fitsdata.2011-03-13T12:56:22.886036.pg_dump_c", "fitsdata.2011-03-12T12:56:22.886036.pg_dump_c", "fitsdata.2011-03-11T12:56:22.886036.pg_dump_c", "fitsdata.2011-03-10T12:56:22.886036.pg_dump_c", "fitsdata.2011-03-09T12:56:22.886036.pg_dump_c", "fitsdata.2011-03-08T12:56:22.886036.pg_dump_c", "fitsdata.2011-03-07T12:56:22.886036.pg_dump_c", "fitsdata.2011-03-06T12:56:22.886036.pg_dump_c", "fitsdata.2011-03-05T12:56:22.886036.pg_dump_c", "fitsdata.2011-03-04T12:56:22.886036.pg_dump_c", "fitsdata.2011-03-03T12:56:22.886036.pg_dump_c", "fitsdata.2011-03-02T12:56:22.886036.pg_dump_c", "fitsdata.2011-03-01T12:56:22.886036.pg_dump_c", "fitsdata.2011-02-28T12:56:22.886036.pg_dump_c", "fitsdata.2011-02-27T12:56:22.886036.pg_dump_c", "fitsdata.2011-02-26T12:56:22.886036.pg_dump_c", "fitsdata.2011-02-25T12:56:22.886036.pg_dump_c", "fitsdata.2011-02-24T12:56:22.886036.pg_dump_c", "fitsdata.2011-02-23T12:56:22.886036.pg_dump_c", "fitsdata.2011-02-22T12:56:22.886036.pg_dump_c", "fitsdata.2011-02-21T12:56:22.886036.pg_dump_c", "fitsdata.2011-02-20T12:56:22.886036.pg_dump_c", "fitsdata.2011-02-19T12:56:22.886036.pg_dump_c", "fitsdata.2011-02-18T12:56:22.886036.pg_dump_c", "fitsdata.2011-02-17T12:56:22.886036.pg_dump_c", "fitsdata.2011-02-16T12:56:22.886036.pg_dump_c", "fitsdata.2011-02-15T12:56:22.886036.pg_dump_c", "fitsdata.2011-02-14T12:56:22.886036.pg_dump_c", "fitsdata.2011-02-13T12:56:22.886036.pg_dump_c", "fitsdata.2011-02-12T12:56:22.886036.pg_dump_c", "fitsdata.2011-02-11T12:56:22.886036.pg_dump_c", "fitsdata.2011-02-10T12:56:22.886036.pg_dump_c", "fitsdata.2011-02-09T12:56:22.886036.pg_dump_c", "fitsdata.2011-02-08T12:56:22.886036.pg_dump_c", "fitsdata.2011-02-07T12:56:22.886036.pg_dump_c", "fitsdata.2011-02-06T12:56:22.886036.pg_dump_c", "fitsdata.2011-02-05T12:56:22.886036.pg_dump_c", "fitsdata.2011-02-04T12:56:22.886036.pg_dump_c", "fitsdata.2011-02-03T12:56:22.886036.pg_dump_c", "fitsdata.2011-02-02T12:56:22.886036.pg_dump_c", "fitsdata.2011-02-01T12:56:22.886036.pg_dump_c", "fitsdata.2011-01-31T12:56:22.886036.pg_dump_c", "fitsdata.2011-01-30T12:56:22.886036.pg_dump_c", "fitsdata.2011-01-29T12:56:22.886036.pg_dump_c", "fitsdata.2011-01-28T12:56:22.886036.pg_dump_c", "fitsdata.2011-01-27T12:56:22.886036.pg_dump_c", "fitsdata.2011-01-26T12:56:22.886036.pg_dump_c", "fitsdata.2011-01-25T12:56:22.886036.pg_dump_c", "fitsdata.2011-01-24T12:56:22.886036.pg_dump_c", "fitsdata.2011-01-23T12:56:22.886036.pg_dump_c", "fitsdata.2011-01-22T12:56:22.886036.pg_dump_c", "fitsdata.2011-01-21T12:56:22.886036.pg_dump_c", "fitsdata.2011-01-20T12:56:22.886036.pg_dump_c", "fitsdata.2011-01-19T12:56:22.886036.pg_dump_c", "fitsdata.2011-01-18T12:56:22.886036.pg_dump_c", "fitsdata.2011-01-17T12:56:22.886036.pg_dump_c", "fitsdata.2011-01-16T12:56:22.886036.pg_dump_c", "fitsdata.2011-01-15T12:56:22.886036.pg_dump_c", "fitsdata.2011-01-14T12:56:22.886036.pg_dump_c", "fitsdata.2011-01-13T12:56:22.886036.pg_dump_c", "fitsdata.2011-01-12T12:56:22.886036.pg_dump_c", "fitsdata.2011-01-11T12:56:22.886036.pg_dump_c", "fitsdata.2011-01-10T12:56:22.886036.pg_dump_c", "fitsdata.2011-01-09T12:56:22.886036.pg_dump_c", "fitsdata.2011-01-08T12:56:22.886036.pg_dump_c", "fitsdata.2011-01-07T12:56:22.886036.pg_dump_c", "fitsdata.2011-01-06T12:56:22.886036.pg_dump_c", "fitsdata.2011-01-05T12:56:22.886036.pg_dump_c", "fitsdata.2011-01-04T12:56:22.886036.pg_dump_c", "fitsdata.2011-01-03T12:56:22.886036.pg_dump_c", "fitsdata.2011-01-02T12:56:22.886036.pg_dump_c", "fitsdata.2011-01-01T12:56:22.886036.pg_dump_c", "fitsdata.2010-12-31T12:56:22.886036.pg_dump_c", "fitsdata.2010-12-30T12:56:22.886036.pg_dump_c", "fitsdata.2010-12-29T12:56:22.886036.pg_dump_c", "fitsdata.2010-12-28T12:56:22.886036.pg_dump_c", "fitsdata.2010-12-27T12:56:22.886036.pg_dump_c", "fitsdata.2010-12-26T12:56:22.886036.pg_dump_c", "fitsdata.2010-12-25T12:56:22.886036.pg_dump_c", "fitsdata.2010-12-24T12:56:22.886036.pg_dump_c", "fitsdata.2010-12-23T12:56:22.886036.pg_dump_c", "fitsdata.2010-12-22T12:56:22.886036.pg_dump_c", "fitsdata.2010-12-21T12:56:22.886036.pg_dump_c", "fitsdata.2010-12-20T12:56:22.886036.pg_dump_c", "fitsdata.2010-12-19T12:56:22.886036.pg_dump_c", "fitsdata.2010-12-18T12:56:22.886036.pg_dump_c", "fitsdata.2010-12-17T12:56:22.886036.pg_dump_c", "fitsdata.2010-12-16T12:56:22.886036.pg_dump_c", "fitsdata.2010-12-15T12:56:22.886036.pg_dump_c", "fitsdata.2010-12-14T12:56:22.886036.pg_dump_c", "fitsdata.2010-12-13T12:56:22.886036.pg_dump_c", "fitsdata.2010-12-12T12:56:22.886036.pg_dump_c", "fitsdata.2010-12-11T12:56:22.886036.pg_dump_c", "fitsdata.2010-12-10T12:56:22.886036.pg_dump_c", "fitsdata.2010-12-09T12:56:22.886036.pg_dump_c", "fitsdata.2010-12-08T12:56:22.886036.pg_dump_c", "fitsdata.2010-12-07T12:56:22.886036.pg_dump_c", "fitsdata.2010-12-06T12:56:22.886036.pg_dump_c", "fitsdata.2010-12-05T12:56:22.886036.pg_dump_c", "fitsdata.2010-12-04T12:56:22.886036.pg_dump_c", "fitsdata.2010-12-03T12:56:22.886036.pg_dump_c", "fitsdata.2010-12-02T12:56:22.886036.pg_dump_c", "fitsdata.2010-12-01T12:56:22.886036.pg_dump_c", "fitsdata.2010-11-30T12:56:22.886036.pg_dump_c", "fitsdata.2010-11-29T12:56:22.886036.pg_dump_c", "fitsdata.2010-11-28T12:56:22.886036.pg_dump_c", "fitsdata.2010-11-27T12:56:22.886036.pg_dump_c", "fitsdata.2010-11-26T12:56:22.886036.pg_dump_c", "fitsdata.2010-11-25T12:56:22.886036.pg_dump_c", "fitsdata.2010-11-24T12:56:22.886036.pg_dump_c", "fitsdata.2010-11-23T12:56:22.886036.pg_dump_c", "fitsdata.2010-11-22T12:56:22.886036.pg_dump_c", "fitsdata.2010-11-21T12:56:22.886036.pg_dump_c", "fitsdata.2010-11-20T12:56:22.886036.pg_dump_c", "fitsdata.2010-11-19T12:56:22.886036.pg_dump_c", "fitsdata.2010-11-18T12:56:22.886036.pg_dump_c", "fitsdata.2010-11-17T12:56:22.886036.pg_dump_c", "fitsdata.2010-11-16T12:56:22.886036.pg_dump_c", "fitsdata.2010-11-15T12:56:22.886036.pg_dump_c", "fitsdata.2010-11-14T12:56:22.886036.pg_dump_c", "fitsdata.2010-11-13T12:56:22.886036.pg_dump_c", "fitsdata.2010-11-12T12:56:22.886036.pg_dump_c", "fitsdata.2010-11-11T12:56:22.886036.pg_dump_c", "fitsdata.2010-11-10T12:56:22.886036.pg_dump_c", "fitsdata.2010-11-09T12:56:22.886036.pg_dump_c", "fitsdata.2010-11-08T12:56:22.886036.pg_dump_c", "fitsdata.2010-11-07T12:56:22.886036.pg_dump_c", "fitsdata.2010-11-06T12:56:22.886036.pg_dump_c", "fitsdata.2010-11-05T12:56:22.886036.pg_dump_c", "fitsdata.2010-11-04T12:56:22.886036.pg_dump_c", "fitsdata.2010-11-03T12:56:22.886036.pg_dump_c", "fitsdata.2010-11-02T12:56:22.886036.pg_dump_c", "fitsdata.2010-11-01T12:56:22.886036.pg_dump_c", ]

for file in db_backup:
  # Strip filename
  match = re.match('fitsdata.(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2}).(\d{6}).pg_dump_c', file)
  #match.group(0)
  if(match):
    datematch = file.split('.')[1].split('T')[0].split('-')
    y = int(datematch[0])
    m = int(datematch[1])
    d = int(datematch[2])
    date = datetime.date(y, m, d)

    if(date>now):
      logger.error("The file %s has a date larger than today's date" % file)
      break

    time_difference = (now-date).days
    # Filter though files from the same year
    if(time_difference<=365):
      # Keep one file per day for the past 10 days
      if(time_difference<10):
        print "in last 10 days %s" % file
        pass
      # Keep one file every 10 days in the last 2 months
      elif(time_difference<60 and (d==11 or d==21)):
        print "within 2 months %s" % file
        pass
      # Keep one file per month for the past year
      elif(d==1):
        print "in year if %s" % file
        pass
      else:
        if(options.dryrun):
          print "Not actually deleting file from year range: %s" % file
        else:
          os.remove("/net/endor/Sci_ARCH/mkofits1_backup/%s" % file)
          print "deleting file from year range: %s" % file
    else:
      if(options.dryrun):
        print "Not actually delete file greater than 1 year old: %s" % file
      else:
        print "delete file greater than 1 year old %s" % file
        os.remove("/net/endor/Sci_ARCH/mkofits1_backup/%s" % file)

  else:
    logger.info("The file %s is not in the expected format." % file)
    break

