#!/usr/bin/env python

import nici as nc   # Load the nici package
import glob, os, sys

def run_nici(datadir=None, outputdir=None):
    """
    Run the nici script. 

    datadir:    Directory pathname holding the raw Flats and 
                object FITS files.
    outputdir:  Directory to put logs and output files.


    -- Getting data files.
       mkdir /tmp/ncdata         # Raw data directory
       cd /tmp/ncdata

       ftp ftp.gemini.edu        # The test data in here
       cd pub
       get nctestdata.tar.gz
       quit

       tar xvf nctestdata.tar.gz  # Extract test data directory
     
       the run.self (datadir='/tmp/ncdata/raw', outputdir='/tmp')
    """

    if not os.access(datadir,os.R_OK):
       print 'Directory '+datadir+' does not exist.'
       sys.exit(0)
    elif datadir[-1] != '/': datadir += '/'
    if not os.access(outputdir,os.R_OK):
       print 'Directory '+outputdir+' does not exist.'
       sys.exit(0)
    elif outputdir[-1] != '/': outputdir += '/'
    
    #- Quick look

    #nc.ncqlook('@/tmp/ncdata/raw/object/lis') 
       
    #   The 'lis' file contains:
    #        /tmp/ncdata/raw/object/obj0.fits
    #        /tmp/ncdata/raw/object/obj1.fits
    #        /tmp/ncdata/raw/object/obj2.fits
    #        /tmp/ncdata/raw/object/obj3.fits
    #        /tmp/ncdata/raw/object/obj4.fits
    #        /tmp/ncdata/raw/object/obj5.fits
    #        /tmp/ncdata/raw/object/obj6.fits
    #        /tmp/ncdata/raw/object/obj7.fits
    #        /tmp/ncdata/raw/object/obj8.fits
    #        /tmp/ncdata/raw/object/obj9.fits
    #
    #   Instead of a list file you can write:
       
    opath = os.path.realpath(datadir+'object/')
    print '\n ##### RUNNING QUICK LOOK (ncqlook) in: ',opath,'\n'
    slis=glob.glob(opath+'/*.fits')
    if len(slis) == 0:
       print 'WARNING: No FITS files in directory:', opath,
    else:
       nc.ncqlook(slis,display=False) 

    #- Prepare science files
    print "\n ##### RUNNING ncprepare in: ",opath,'\n'
    if len(slis) == 0:
        print 'WARNING: No FITS files in directory:', opath
    else:
        if not os.access(outputdir+'science',os.R_OK):
            os.mkdir(outputdir+'science')
        nc.ncprepare(slis, outputdir=outputdir+'science')

    #- Generate calibration files
    #- Create the file list
    cal_path = os.path.realpath(datadir+'flats/')
    print "\n ##### MAKING CALIBRATION FILES (ncmkflats) in: ",cal_path,'\n'
    flis=glob.glob(cal_path+'/*.fits')
    if len(flis) == 0:
        print 'WARNING: No FITS files in directory:',cal_path
    else:
        nc.ncmkflats(flis, outputdir=outputdir)

    #- Finally run the science
    spath = outputdir+'science/'
    print "\n ##### RUNNING SCIENCE scripts (ncscience) in: ",spath,'\n'
    if not os.access(outputdir+'results',os.R_OK):
        os.mkdir(outputdir+'results')
    #- Create a list file:
    slis=glob.glob(spath+'/*.fits')
    if len(slis) == 0:
       print 'WARNING: No FITS files in directory:',spath 
    else:
       nc.ncscience(slis,outputdir=outputdir+'results', \
                   flatsdir=outputdir, dataSetName='Test')
# TEST
# nc.ncscience("@/data1/data/etienne_demo_py/science.lis",inputdir='/data1/data/etienne_demo_py/science',outputdir='/data1/data/etienne_demo_py/results',flatsdir='/data1/data/etienne_demo_py',dataSetName='Jul27')

    return
if __name__ == "__main__":

    import optparse

    # Parse input arguments
    usage = 'usage: %prog data_pathname'

    p = optparse.OptionParser(usage=usage)

    p.add_option('--datadir', action='store', type='string',
                 default='', help='Test Data directory pathname')

    p.add_option('--outputdir', action='store', type='string',
                 default='', help='Output directory pathname')

    (par, args) = p.parse_args()


    run_nici (datadir=par.datadir, outputdir=par.outputdir)


