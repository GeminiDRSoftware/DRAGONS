from pyraf import iraf
import nicimkflats as mkf
from niciTools import getFileList

def _nicimkflats (flats_ls,inputdir, outputdir, clobber,\
                logfile, verbose):
    list = getFileList(flats_ls)
    clobber = (clobber == 'yes')
    verbose = (verbose == 'yes')

    mkf.nicimkflats (list, inputdir, outputdir, clobber,\
                logfile, verbose)
 
parfile = iraf.osfn('home$nici/nicimkflats.par')
t = iraf.IrafTaskFactory(taskname='nicimkflats', value=parfile, function=_nicimkflats)
