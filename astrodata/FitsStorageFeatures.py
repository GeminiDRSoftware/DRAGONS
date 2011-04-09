import os
# DON'T PUT THIS HERE... SETUP WILL CHANGE IT!!!!! from fitsstore import FitsStorageConfig
        
class FitsStorageIngest(object):
    def __init__(self, directory):
        from fitsstore import FitsStorageConfig
        fscbin = os.path.dirname(FitsStorageConfig.__file__)
        print fscbin
        if_cmd = os.path.join(fscbin,"local_add_to_ingest_queue.py")
        os.system("python "+if_cmd)

        siq_cmd = os.path.join(fscbin,"service_ingest_queue.py")
        os.system("python "+siq_cmd+" --empty")
 
        

FS_DIRTOKEN = ".fitsstore"
FSC_FNAME = "FitsStorageConfig.py"
class FitsStorageSetup(object):
    fs_config = None
    
    def __init__(self):
        cwd = os.getcwd()
        self.startingdir = cwd
        self.fitsstoredir = os.path.join(self.startingdir, FS_DIRTOKEN)
        self.fscName = os.path.join(self.fitsstoredir, FSC_FNAME)
        self.fs_config = FSC_TEMPLATE % { "startingdir":cwd}
        self.fs_logdir = os.path.join(self.fitsstoredir, "logs")
        
        
    def setup(self):
        if not os.path.exists(self.fitsstoredir):
            os.mkdir(self.fitsstoredir)
        if os.path.exists(self.fscName):
            print "FitsStoreConfig.py already present in this directory, not touching it."
        else:
            fscout = open(self.fscName, "w")
            fscout.write(self.fs_config)
            fscout.close()
        if not os.path.exists(self.fs_logdir):
            os.mkdir(self.fs_logdir)
        
        # create tables
        from fitsstore import FitsStorageConfig
        if not os.path.exists(os.path.join(self.fitsstoredir, FitsStorageConfig.fits_dbname)):
            print "make the database tables"
            fscbin = os.path.dirname(FitsStorageConfig.__file__)
            ct_cmd = os.path.join(fscbin,"create_tables.py")
            os.system("python "+ct_cmd)

            
FSC_TEMPLATE = """
import os
# These are config parameters that are imported into the FitsStorage namespace
# We put them in a separate file to ease install issues

# this flag used to fascilitate local mode, as used by Astrodata Recipe System
fsc_localmode=True

# Configure the path to the storage root here 
#storage_root = '/net/wikiwiki/dataflow'
storage_root = os.path.join("%(startingdir)s",".fitsstore")

target_gb_free = 100
target_max_files = 125000

# This is the path in the storage root where processed calibrations
# uploaded through the http server get stored.
#upload_staging_path = "/data/upload_staging"
#processed_cals_path = "reduced_cals"

# The DAS calibration reduction path is used to find the last processing
# date for the gmoscal page's autodetect daterange feature
#das_calproc_path = '/net/endor/export/home/dataproc/data/gmos/'
#das_calproc_path = '/net/josie/staging/dataproc/gmos'
das_calproc_path = "%(startingdir)s"

    # Configure the site and other misc stuff here
fits_servername = "local"
fits_system_status = "production"

email_errors_to = "phirst@gemini.edu"

# Configure the path the data postgres database here
fits_dbname = 'fitsdata.sql'
fits_database = 'sqlite:///'+storage_root+"/"+fits_dbname
# fits_dbname = 'fitsdata'
# fits_database = 'postgresql:///'+fits_dbname

# Configure the Backup Directory here
fits_db_backup_dir = os.path.join("%(startingdir)s","backups")
#fits_db_backup_dir = "/net/endor/Sci_ARCH/mkofits1_backup/"

# Configure the LockFile Directory here
fits_lockfile_dir = os.path.join("%(startingdir)s","locks")

# Configure the log directory here
fits_log_dir = storage_root + "/logs/"

# Configure the tape scratch directory here
fits_tape_scratchdir = storage_root + "/tapescratch"
"""
