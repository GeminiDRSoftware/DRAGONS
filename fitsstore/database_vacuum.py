from FitsStorageConfig import *
import os
import datetime
import subprocess

from FitsStorageLogger import *

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
logger.info("*********  database_vacuum.py - starting up at %s" % now)

command = ["/usr/bin/vacuumdb", "--verbose", "--analyze", "--dbname", fits_dbname]

logger.info("Executing vacuumdb")

sp = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
(stdoutstring, stderrstring) = sp.communicate()

logger.info(stderrstring)

logger.info(stdoutstring)

logger.info("-- Finished, Exiting")
