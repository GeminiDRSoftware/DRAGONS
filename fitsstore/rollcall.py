import sys
sys.path=['/opt/sqlalchemy/lib/python2.5/site-packages', '/astro/iraf/x86_64/gempylocal/lib/stsci_python/lib/python2.5/site-packages']+sys.path

from FitsStorage import *
from FitsStorageLogger import *
import datetime

from optparse import OptionParser

parser = OptionParser()
parser.add_option("--limit", action="store", type="int", help="specify a limit on the number of files to examine. The list is sorted by lastmod time before the limit is applied")
parser.add_option("--file-pre", action="store", type="string", dest="filepre", help="File prefix to check (omit for all)")
parser.add_option("--debug", action="store_true", dest="debug", help="Increase log level to debug")
parser.add_option("--demon", action="store_true", dest="demon", help="Run as a background demon, do not generate stdout")

(options, args) = parser.parse_args()

# Logging level to debug? Include stdio log?
setdebug(options.debug)
setdemon(options.demon)


# Annouce startup
logger.info("*********  rollcall.py - starting up at %s" % datetime.datetime.now())

# Get a database session
session = sessionfactory()

# Get a list of all diskfile_ids marked as present
# If we try and really brute force a list of DiskFile objects, we run out of memory...
query = session.query(DiskFile.id).select_from(join(DiskFile, File)).filter(DiskFile.present == True).order_by(DiskFile.lastmod)

if(options.filepre):
  likestr = "%s%%" % options.filepre
  query = query.filter(File.filename.like(likestr))

# Did we get a limit option?
if(options.limit):
  query = query.limit(options.limit)

logger.info("evaluating number of rows...")
n = query.count()
logger.info("%d files to check" % n)

# Semi Brute force approach for now. 
# It might be better to find some way to retrieve the items from the DB layer one at a time...
# If we try and really brute force a list of DiskFile objects, we run out of memory...
logger.info("Getting list...")
list = query.all()
logger.info("Starting checking...")

i=0
j=0
for dfid in list:
  # Search for it by ID (is there a better way?)
  df=session.query(DiskFile).filter(DiskFile.id == dfid[0]).one()
  if(not df.file.exists()):
    # This one doesn't actually exist
    df.present=False
    j+=1
    logger.info("File %d/%d: Marking file %s (diskfile id %d) as not present" % (i, n, df.file.filename, df.id))
    session.commit()
  else:
    if ((i % 1000) == 0):
      logger.info("File %d/%d: present and correct" % (i, n))
  i+=1

logger.info("\nMarked %d files as no longer present\n" % j)

logger.info("*** rollcall.py exiting normally at %s" % datetime.datetime.now())
