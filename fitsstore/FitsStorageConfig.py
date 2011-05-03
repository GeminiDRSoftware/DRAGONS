# These are config parameters that are imported into the FitsStorage namespace
# We put them in a separate file to ease install issues

# controls if the system operates in regular FITSSTORE mode or Local Mode, as used by
# the Astrodata Recipe System.
fsc_localmode = False

# Configure the path to the storage root here 
storage_root = '/net/wikiwiki/dataflow'

target_gb_free = 100
target_max_files = 125000

# This is the path in the storage root where processed calibrations
# uploaded through the http server get stored.
upload_staging_path = "/data/upload_staging"
processed_cals_path = "reduced_cals"

# The DAS calibration reduction path is used to find the last processing
# date for the gmoscal page's autodetect daterange feature
das_calproc_path = '/net/endor/export/home/dataproc/data/gmos/'
#das_calproc_path = '/net/josie/staging/dataproc/gmos'

# Configure the site and other misc stuff here
fits_servername = "mkofits1"
fits_system_status = "production"

email_errors_to = "phirst@gemini.edu"

# Configure the path the data postgres database here
fits_dbname = 'fitsdata'
fits_database = 'postgresql:///'+fits_dbname

# Configure the Backup Directory here
fits_db_backup_dir = "/net/endor/Sci_ARCH/mkofits1_backup/"

# Configure the LockFile Directory here
fits_lockfile_dir = "/data/logs"

# Configure the log directory here
fits_log_dir = "/data/logs/"

# Configure the tape scratch directory here
fits_tape_scratchdir = "/data/tapescratch"


# the following implements logic where a if a copy of FitsStorageConfig 
# a subdirectory of the current directory, .fitsstore, then this is also
# loaded (equiv of from ___ import *) in the current directory

import os
runconfig = ".fitsstore/FitsStorageConfig.py"

if os.path.exists(runconfig):
    print "Found local config files"
    rc = open(runconfig)
    exec(rc)
    rc.close()
    print "after run config set"

