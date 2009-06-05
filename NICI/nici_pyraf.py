from pyraf import iraf
import nicimkflats as mkf
import niciprepare as nicip
import niciscience as nicis
import nici_das as nicid
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


def _niciscience (science_lis,inputdir,outputdir, flatsdir, 
                  central, savePA, dataSetName, clobber,
                  logfile, verbose):

    list = getFileList(science_lis)
    clobber = (clobber == 'yes')
    central = (central == 'yes')
    savePA = (savePA == 'yes')
    verbose = (verbose == 'yes')
    nicis.niciscience (list,inputdir,outputdir, flatsdir, 
                  central, savePA, dataSetName, clobber,
                  logfile, verbose)
 
parfile = iraf.osfn('home$nici/niciscience.par')
t = iraf.IrafTaskFactory(taskname='niciscience', value=parfile, function=_niciscience)

def _nicidas (date, help,log,lists,saturate,display):
    """ nici_das date='20081116',saturate=2000 display=False  
       ---- parses the 20081116 data with a saturation limit at 2000 
       and does not display (for speed)
       keyword saturate. If not present, saturation >3500 
    """
    help = False
    date = date
    saturate = 3500
    log = True
    lists = True
    display = (display == 'yes')
    nicid.nici_das (date, help, log,lists,saturate, display)
 
parfile = iraf.osfn('home$nici/nici_das.par')
t = iraf.IrafTaskFactory(taskname='nici_das', value=parfile, function=_nicidas)
