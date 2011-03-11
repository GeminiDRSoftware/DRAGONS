from FitsStorageConfig import *
import os
import datetime
import subprocess

from FitsStorageLogger import *

datestring = datetime.datetime.now().isoformat()

from optparse import OptionParser
parser = OptionParser()
parser.add_option("--debug", action="store_true", dest="debug", help="Increase log level to debug")
parser.add_option("--demon", action="store_true", dest="demon", help="Run as a background demon, do not generate stdout")

(options, args) = parser.parse_args()

# Logging level to debug? Include stdio log?
setdebug(options.debug)
setdemon(options.demon)

# Annouce startup
now = datetime.datetime.now()
logger.info("*********  database_backup.py - starting up at %s" % now)

# The backup filename
filename = "%s.%s.pg_dump_c" % (fits_dbname, datestring)

command = ["/usr/bin/pg_dump", "--format=c", "--file=%s/%s" % (fits_db_backup_dir, filename), fits_dbname]

logger.info("Executing pg_dump")

sp = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
(stdoutstring, stderrstring) = sp.communicate()

logger.info(stderrstring)

logger.info(stdoutstring)

logger.info("-- Finished, Exiting")
