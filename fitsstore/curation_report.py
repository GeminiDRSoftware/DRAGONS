"""
This module retrieves and prints out the desired values from the list created in 
FitsStorageCuration.py.
"""
import sys
sys.path=['/opt/sqlalchemy/lib/python2.5/site-packages', '/astro/iraf/x86_64/gempylocal/lib/stsci_python/lib/python2.5/site-packages']+sys.path

from FitsStorageCuration import *
from optparse import OptionParser

parser = OptionParser()
parser.add_option("--debug", action="store_true", dest="debug", help="Increase log level to debug")
parser.add_option("--demon", action="store_true", dest="demon", help="Run as a background demon, do not generate stdout")
parser.add_option("--checkonly", action="store", type="string", dest="checkonly", help="Limits the search by identifying a specific substring")
parser.add_option("--exclude", action="store", type="string", dest="exclude" , help="Limits the search by excluding  data with a specific substring")
parser.add_option("--noeng", action="store_const", const="ENG", dest="exclude", help="Limits the search by excluding data with the ENG substring")

(options, args) = parser.parse_args()
checkonly = options.checkonly
exclude = options.exclude


# Get a database session
session = sessionfactory()


# Work for duplicate_datalabels
dupdata = duplicate_datalabels(session, checkonly, exclude)
previous_ans = ''
# This for loop changes the list of df_ids to the corresponding DiskFile ids
if dupdata != []:
  for val in dupdata:
    this_ans = val
    header = session.query(Header).filter(Header.diskfile_id == val).first()
    if header:
      if previous_ans == '':
        pass
      if previous_ans != this_ans:
        print "duplicate datalabel rows: DiskFile id = %s,  Filename = %s,  Datalabel = %s" %  (header.diskfile.id, header.diskfile.file.filename, header.data_label) 
      previous_ans = this_ans
else:
  print "No duplicate datalabel rows detected."


# Work for duplicate_canonicals
dupcanon = duplicate_canonicals(session)
# Iterates through the list of rows and returns the desired values
previous_file = ''
empty = 0
for val in dupcanon: 
  this_file = val.file_id
  if previous_file == '':
    pass
  elif previous_file == this_file:
    print "duplicate canonical rows: DiskFile id = %s,  File id = %s,  canonical = %s" %  (val.id, val.file.id, val.canonical)
    empty =+ 1 
  previous_file = this_file
if empty == 0:
  print "No duplicate canonical rows detected."


# Work for duplicate_present
duppres = duplicate_present(session)
# Iterates through the list of rows and returns the desired values
previous_file = ''
empty = 0
for val in duppres: 
  this_file = val.file_id
  if previous_file == '':
    pass
  elif previous_file == this_file:
    print "duplicate present rows: DiskFile id = %s,  File id = %s,  present = %s" %  (val.id,  val.file.id, val.present)
  previous_file = this_file
if empty == 0:
  print "No duplicate present rows detected." 


# Work for present_not_canonical
presnotcanon = present_not_canonical(session)
# Iterates through the list of rows and returns the desired values
if presnotcanon != []:
  for val in presnotcanon:   
    print "duplicate present not canonical rows: DiskFile id = %s,  present = %s,  canonical = %s" %  (val.id, val.present, val.canonical)
else:
  print "No rows with the conditions present=True and canonical=False."


session.close()
