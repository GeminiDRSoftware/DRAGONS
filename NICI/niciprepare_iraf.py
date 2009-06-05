from pyraf import iraf
import niciprepare as nicip
from niciTools import getFileList


def _niciprepare(inputList,outprefix,inputdir,outputdir,
                clobber, logfile, sci_ext, var_ext,
                dq_ext, fl_var, fl_dq, verbose):
    list = getFileList(inputList)
    clobber = (clobber == 'yes')
    fl_var = (fl_var == 'yes')
    fl_dq = (fl_dq == 'yes')
    verbose = (verbose == 'yes')
    nicip.niciprepare(list,outprefix,inputdir, outputdir,clobber,\
                logfile, sci_ext, var_ext, dq_ext, fl_var, fl_dq,verbose)
 
parfile = iraf.osfn('home$nici/niciprepare.par')
t = iraf.IrafTaskFactory(taskname='niciprepare', value=parfile, function=_niciprepare)
