import sys
sys.path += ['/opt/sqlalchemy/lib/python2.5/site-packages']

from FitsStorageUtils import *

# Option Parsing
from optparse import OptionParser
parser = OptionParser()
parser.add_option("--drop", action="store_true", dest="drop", help="Drop the tables first")
parser.add_option("--nocreate", action="store_true", dest="nocreate", help="Do not actually create the tables")

(options, args) = parser.parse_args()


session = sessionfactory()

if options.drop:
  print "Dropping database tables"
  drop_tables(session)

if not options.nocreate:
  print "Creating database tables"
  create_tables(session)

session.close()

