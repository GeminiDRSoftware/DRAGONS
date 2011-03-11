import sys
sys.path=['/opt/sqlalchemy/lib/python2.5/site-packages', '/astro/iraf/x86_64/gempylocal/lib/stsci_python/lib/python2.5/site-packages']+sys.path

from FitsStorage import *
from FitsStorageLogger import *
import datetime

from optparse import OptionParser

parser = OptionParser()
parser.add_option("--debug", action="store_true", dest="debug", help="Increase log level to debug")
parser.add_option("--demon", action="store_true", dest="demon", help="Run as a background demon, do not generate stdout")

(options, args) = parser.parse_args()

# Logging level to debug? Include stdio log?
setdebug(options.debug)
setdemon(options.demon)


# Annouce startup
logger.info("*********  data_rates.py - starting up at %s" % datetime.datetime.now())

# Get a database session
session = sessionfactory()

f = open("/data/logs/data_rates.py", "w")

ndays=1000

today = datetime.datetime.utcnow().date()
zerohour = datetime.time(0,0,0)
ddelta = datetime.timedelta(days=1)

start = datetime.datetime.combine(today, zerohour)
end = start + ddelta

for i in range(1, ndays):
  query = session.query(func.sum(DiskFile.size)).select_from(join(Header, DiskFile)).filter(DiskFile.present==True).filter(Header.ut_datetime > start).filter(Header.ut_datetime < end)
  bytes = query.one()[0]
  if(not bytes):
    bytes = 0
  f.write("%s, %f\n" % (str(start.date()), bytes/1.0E9))
  start -= ddelta
  end -= ddelta

f.close()

logger.info("*** data_rates.py exiting normally at %s" % datetime.datetime.now())
