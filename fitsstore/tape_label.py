import os
import sys
import subprocess
from FitsStorageConfig import *
from FitsStorageTape import TapeDrive

# Option Parsing
from optparse import OptionParser
parser = OptionParser()
parser.add_option("--read", action="store_true", dest="read", help="Read the label from the tape in the drive")
parser.add_option("--label", action="store", dest="label", help="Write the label to the tape in the drive. This will write to the start of the tape, making any other data on the tape inaccessible")
parser.add_option("--tapedrive", action="store", dest="tapedrive", help="The tapedrive device to use")
parser.add_option("--force", action="store_true", dest="force", help="Normally, --label will refuse to label a tape that allready contains a tapelabel. This option forces it to do so.")

(options, args) = parser.parse_args()

if((not options.read) and (not options.label)):
  print "You must supply either the --read or the --label option"
  sys.exit(1)

td = TapeDrive(options.tapedrive, fits_tape_scratchdir)

if(options.read):
  print td.readlabel(fail=False)
  sys.exit(0)


if(options.label):
  oldlabel = td.readlabel(fail=False)
  if(oldlabel):
    print "This tape already has a FitsStorage tape label"
    print "Current label is: %s" % oldlabel
    if(options.force):
      print "--force specified: will overwrite"
      print "Writing new tape label: %s" % options.label
      td.writelabel(options.label)
      sys.exit(0)
    else:
      print "If you really want to overwrite, use the --force option"
      sys.exit(1)
  else:
     print "Writing tape label: %s" % options.label
     td.writelabel(options.label)
     sys.exit(0)

