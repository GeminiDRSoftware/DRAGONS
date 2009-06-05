from pyraf import iraf
import niciscience as nicis
from niciTools import getFileList


def _niciscience (science_lis,inputdir,outputdir, flatsdir, 
                  central, savePA, dataSetName, clobber,
                  logfile, verbose):

    list = getFileList(science_lis)
    print 'n_iraf:',list
    clobber = (clobber == 'yes')
    central = (central == 'yes')
    savePA = (savePA == 'yes')
    verbose = (verbose == 'yes')
    nicis.niciscience (list,inputdir,outputdir, flatsdir, 
                  central, savePA, dataSetName, clobber,
                  logfile, verbose)
 
parfile = iraf.osfn('home$nici/niciscience.par')
t = iraf.IrafTaskFactory(taskname='niciscience', value=parfile, function=_niciscience)
