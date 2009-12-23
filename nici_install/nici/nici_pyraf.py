from pyraf import iraf
import nici as nc
from nici.niciTools import getFileList
import imp

def _ncmkflats (inputs, idir, odir, sigma, clobber, suffix, logfile):

    list = getFileList(inputs)
    clobber = (clobber == 'yes')

    nc.ncmkflats (list, idir, odir, sigma, clobber, suffix, logfile)
 
parfile = iraf.osfn('nicipath$ncmkflats.par')
t = iraf.IrafTaskFactory(taskname='ncmkflats', value=parfile, function=_ncmkflats)

def _ncprepare(inputs,oprefix,idir,odir, clobber, fl_var, fl_dq):
    list = getFileList(inputs)
    clobber = (clobber == 'yes')
    fl_var = (fl_var == 'yes')
    fl_dq = (fl_dq == 'yes')
    nc.ncprepare(list,oprefix,idir, odir,clobber, fl_var, fl_dq)
 
parfile = iraf.osfn('nicipath$ncprepare.par')
t = iraf.IrafTaskFactory(taskname='ncprepare', value=parfile, function=_ncprepare)


def _ncscience (inputs,idir,odir, fdir, froot, central, suffix, 
                clobber, dobadpix, pad, logfile):

    list = getFileList(inputs)
    clobber = (clobber == 'yes')
    central = (central == 'yes')
    dobadpix = (dobadpix == 'yes')
    pad = (pad == 'no')
    nc.ncscience (list,idir,odir, fdir, froot, central, rootname, 
                  clobber,dobadpix, pad, logfile)
 
parfile = iraf.osfn('nicipath$ncscience.par')
t = iraf.IrafTaskFactory(taskname='ncscience', value=parfile, function=_ncscience)

def _ncqlook (inputs, idir, odir, log,lists,saturate,display,c2h):

    saturate = 3500
    log = True
    lists = True
    display = not (display == 'yes')
    c2h = (c2h == 'yes')
    nc.ncqlook (inputs, idir, odir, log,lists,saturate, display, c2h)
 
parfile = iraf.osfn('nicipath$ncqlook.par')
t = iraf.IrafTaskFactory(taskname='ncqlook', value=parfile, function=_ncqlook)
