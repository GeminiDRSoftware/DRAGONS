from FitsStorageConfig import *
import os
import re
import datetime
import subprocess

from FitsStorageLogger import *

datestring = datetime.datetime.now().isoformat()

from optparse import OptionParser
parser = OptionParser()
parser.add_option("--dontdelete", action="store_true", dest="dontdelete", help="Don't actually delete anything, just say what would be deleted")
parser.add_option("--dontbackup", action="store_true", dest="dontbackup", help="Don't back up the database, just clean up")
parser.add_option("--debug", action="store_true", dest="debug", help="Increase log level to debug")
parser.add_option("--demon", action="store_true", dest="demon", help="Run as a background demon, do not generate stdout")

(options, args) = parser.parse_args()

# Logging level to debug? Include stdio log?
setdebug(options.debug)
setdemon(options.demon)

# Annouce startup
now = datetime.datetime.now()
logger.info("*********  database_backup.py - starting up at %s" % now)

# BACKUP STUFF
if(not options.dontbackup):
  # The backup filename
  filename = "%s.%s.pg_dump_c" % (fits_dbname, datestring)

  command = ["/usr/bin/pg_dump", "--format=c", "--file=%s/%s" % (fits_db_backup_dir, filename), fits_dbname]

  logger.info("Executing pg_dump")

  sp = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  (stdoutstring, stderrstring) = sp.communicate()

  logger.info(stderrstring)

  logger.info(stdoutstring)

  logger.info("-- Finished, Exiting")


# CLEANUP STUFF
split_date = now.isoformat().split('T')[0].split('-')

# Strip date
year = int(split_date[0])
month = int(split_date[1])
day = int(split_date[2])
today = datetime.date(year, month, day)

# Directory name in FitsStorageUtils = fits_db_backup_dir
db_backup = os.listdir(fits_db_backup_dir)

for file in db_backup:
  # Strip filename
  match = re.match('fitsdata.(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2}).(\d{6}).pg_dump_c', file)
  if(match):
    datematch = file.split('.')[1].split('T')[0].split('-')
    y = int(datematch[0])
    m = int(datematch[1])
    d = int(datematch[2])
    date = datetime.date(y, m, d)

    if(date>today):
      logger.error("The file %s has a date larger than today's date" % file)
      break

    time_difference = (today-date).days
    # Filter though files from the same year
    if(time_difference<=365):
      # Keep one file per day for the past 10 days
      if(time_difference<10):
        logger.info("This file is less than 10 days old: %s" % file)
      # Keep one file every 10 days in the last 2 months
      elif(time_difference<60 and (d==11 or d==21)):
        logger.info("This file is less than 2 months old and falls on 01, 11 or 21 of the month: %s" % file)
      # Keep one file per month for the past year
      elif(d==1):
        logger.info("This file is less than 1 year old and falls on 01 of the month: %s" % file)
      else:
        if(options.dontdelete):
          logger.info("This file would be deleted: %s" % file)
        else:
          os.remove("%s/%s" % (fits_db_backup_dir, file))
          logger.info("Deleting file: %s" % file)
    else:
      if(options.dontdelete):
        logger.info("This file would be deleted: %s" % file)
      else:
        logger.info("Deleting file: %s" % file)
        os.remove("%s/%s" % (fits_db_backup_dir, file))
  else:
    logger.info("The file %s is not in the expected format." % file)

