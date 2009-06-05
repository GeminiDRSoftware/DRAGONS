#!/usr/bin/env python
#
import optparse

import os, sys
import pyfits as pf
import time
import re
import ndimage as nd
from niciTools import getFileList
from niciscience import shift_function
from numpy import zeros,where,dtype,int32,size,asarray,float32,isnan
from niciCntrd import nici_cntrd

def niciprepare (inputList, outprefix='n', inputdir='', outputdir='',
                clobber=True, logfile='', sci_ext='SCI', var_ext='VAR',
                dq_ext='DQ', fl_var=False, fl_dq=False, verbose=True):

    """
        #
 
        inputList =               Input Fits files on a list
        (inputdir = '')           Path for input raw files
        (outputdir = '')          Output directory name
        (outprefix = 'n')         Prefix for output image(s)
        clobber  = True           Replace output file if exits
        (logfile = "")            Logfile
    nyi (sci_ext = "SCI")         Name or number of science extension
    nyi (var_ext = "VAR")         Name or number of variance extension
    nyi (dq_ext = "DQ")           Name or number of data quality extension
        (fl_var = False)           Create variance frame?
        (fl_dq = False)            Create data quality frame?
    nyi (key_ron = "RDNOISE")     New header keyword for read noise (e-)
    nyi (key_gain = "GAIN")       New header keyword for gain (e-/ADU)
    nyi (key_sat = "SATURATI")    New header keyword for saturation (ADU)
    nyi (key_nonlinea="NONLINEA") New header keyword for non-linear regime (ADU)
        (verbose = True)          Verbose

     NOTE: 'nyi' above means 'not yet implemented'

     DESCRIPTION:

     Procedure to take raw NICI SCIENCE data, with two FITS extensions,
     and create the preliminary VAR and DQ frames.
    
     Data is "fixed" to account for low-noise reads, digital averages,
     and coadds as follows:
        noise = (value from nprepare.dat) * sqrt(n coadds)
        exptime = input * (n coadds)
        gain = 11.5 e-
        saturation = x * (n coadds) where x=200,000 e- for bias=-0.6,
                                            280,000 e- for bias=-0.9
        non-linear for > 0.7*saturation
        (these are nominal values; the data file must contain entries with
        the names readnoise, gain, shallowbias, shallowwell, deepbias, deepwell
        which are used in the appropriate places in the calculations above).
    
     The variance frame is generated as:
           var = (read noise/gain)^2 + max(data,0.0)/gain
    
     The preliminary DQ frame is constructed by using the bad pixel
     mask to set the 1 bit and saturation level to set the 4 bit.
     EXAMPLES:
        PYRAF:
           - Make sure you have the NICI scripts in a designated 
             directory, e.g. 'nici'.
           - Your UNIX PYTHONPATH shell variable should have an
             entry for the directory 'nici', like: /home/myname/nici
           - Start Pyraf
           --> #define the task
           --> pyexecute('/home/myname/nici/niciprepare.py')
           --> #execute
           --> niciprepare *.fits outputdir=/tmp

        Python_shell: 
           - Your UNIX PYTHONPATH shell variable should have an
             entry for the directory 'nici', like: /home/myname/nici
           >>> import niciprepare as nicip
           >>> # Make the input list, assuming you are in the data-directory
           >>> import glob
           >>> flis=glob.glob('*.fits')
           >>> nicip.niciprepare(flis,inputdir='/data1/data/tmp',outputdir='/tmp')
      
        Unix_shell:
           % cd /home/myname/nici
           % # to get help try: niciprepare.py -h
           % niciprepare.py /data1/data/tmp/*.fits --odir='/tmp'

    

    """


    #print inputList,outprefix,inputdir,'ODIR:',outputdir,\
    #            'cl:',clobber, 'log:',logfile, 'sci:',sci_ext, 'var:',var_ext,\
    #            'dqx:',dq_ext, 'flv:',fl_var, 'flq:',fl_dq, 'verbose:',verbose

    # set up logfile name niciprepare.log (default is nici.log)
    # 

    # Open log file

    # do we need data base with array dependent characteristics?
    # if so set up the database.dat

    # Check raw path names

    # check input list. Are we supported @ syntax

    if (len(outprefix) > 10):
        print "input prefix is too long, returning"
        return
    
    # Make sure inputdir and outputdir ends in '/'.
    if (inputdir != '' and inputdir[-1] != '/'):
        inputdir += '/'
    if (len(inputdir) > 0 and not os.access(inputdir, os.R_OK)):
        print "*** Warning: Inputdir:"+inputdir+' not readable.'
        return
    if (outputdir != '' and outputdir[-1] != '/'):
        outputdir += '/'
    if (len(outputdir) > 0 and not os.access(outputdir, os.W_OK)):
        print "*** Warning: Outputdir: "+outputdir+' not writeable.'
        return

    ninlen=size(inputList) 
    for fname in inputList:
     
        fpath = inputdir+fname.strip()
        #- gimverify. Do we need to verify that files have 3 extensions?
        #- check for PREPARE keyword   
        fits = pf.open(fpath)
        if (fits[0].header).get("INSTRUME") != 'NICI':
            print 'File:',fname,'is not for NICI. Ignored.' 
            fits.close()
            continue

        if (fits[0].header).get("PREPARE"):
            print fpath," already fixed using niciprepare. -skipped"
            continue

        if (len(fits) != 3):
            print 'File:',fname,'does not have 2 image extensions. Ignored' 
            fits.close()
            continue

        if (fits[0].header['obsclass']).lower() != 'science':
            print fname,' is not SCIENCE frame. -skipped'
            fits.close()
            continue

        print fname,fits[0].header['object'],fits[0].header['obstype'],\
                    fits[0].header['obsclass']
        # Get the root name
        tmp = re.sub('.gz','',fpath)
        root = (tmp.split('.')[-2].split('/')).pop()
        ext = tmp.split('.')[-1]

        outfile = outputdir+outprefix+root+'.'+ext
        if (os.access(outfile, os.R_OK)):
            if clobber:
               os.remove(outfile)
            else:
	       print 'File: ',outfile, 'already exists.'
	       return

        out = pf.open(outfile,mode='append')
        
        out.append(fits[0]) 
        # Copy input header
        ophu = out[0].header
        

        gmt = time.gmtime()
        time.asctime(gmt)
        fits_utc = '%d-%02d-%02dT%02d:%02d:%02d' % gmt[:6]
    
        # Time Stamps
        ophu.update("GEM-TLM",fits_utc, "UT Last modification with GEMINI")
        ophu.update("PREPARE",fits_utc, "UT Time stamp for NPREPARE")

        # Create a DHU
        #hdu = pf.PrimaryHDU()
        
        out.flush()             # write to disk, leave 'out' open  
        

        out.append(fits[1])
        oh = out[1].header       
        im = out[1].data       

        # See which channel is in Xtension 1
        if oh.get("MODE_R"):
           channel = 'red'
        elif  oh.get("MODE_B"):
           channel = 'blue'
        else:
           print 'ERROR: Keyword MODE_<R or B> not present in Extension 1'
           continue
        
        update,xcen,ycen,im = nici_cntrd(im,oh,center_im=False)
        print outfile, '[red]  xcen: %.2f, ycen: %.2f' % (xcen,ycen)  
        if (xcen < 0 or ycen <0 ):
            # Try to center again, now interactively
            update,xcen,ycen,im = nici_cntrd(im,oh,center_im=False)
            print '          [red again] xcen: %.2f, ycen: %.2f' % (xcen,ycen)  
        order_wcs(oh,channel,1)
        out.flush()
       
        out.append(fits[2])
        oh = out[2].header       
        im = out[2].data       

        # See which channel is in Xtension 2
        if oh.get("MODE_R"):
           channel = 'red'
        elif  oh.get("MODE_B"):
           channel = 'blue'
        else:
           print 'ERROR: Keyword MODE_<R or B> not present in Extension 1'
           continue

        # Get xcen,ycen by mapping this extension to ext[1] coordinate
        # system.

	if 'float' not in str(im.dtype):
            im = asarray(im,dtype=float32)

        imc = nd.geometric_transform (where(isnan(im),1,im),shift_function)

        update,xcen,ycen,imc = nici_cntrd(imc,oh,center_im=False)
        print outfile, '[blue] xcen: %.2f, ycen: %.2f' % (xcen,ycen)  
        if (xcen < 0 or ycen <0 ):
            # Try to center again, now interactively
            update,xcen,ycen,im = nici_cntrd(im,oh,center_im=False)
            print '          [blue again] xcen: %.2f, ycen: %.2f' % (xcen,ycen)  
        order_wcs(oh,channel,2)
        out.flush()
        
        if (fl_var):
            # Write VAR 1,2
            oh = out[1].header
            oh.update("EXTNAME",'VAR', "VAR extension")
            oh.update("EXTVER",1, "VAR ext version")
            #hdu= pf.ImageHDU(header=oh, data=out[1].data)
            hdu= pf.ImageHDU(header=oh)
            out.append(hdu)
            out.flush()

            #out.append(fits[2])
            oh = out[2].header
            oh.update("EXTNAME",'VAR', "VAR extension")
            oh.update("EXTVER",2, "VAR ext version")
            #hdu= pf.ImageHDU(header=oh, data=out[2].data)
            hdu= pf.ImageHDU(header=oh)
            out.append(hdu)
            out.flush()

        if (fl_dq):
            # Write DQ 1,2
            oh = out[1].header
            oh.update("EXTNAME",'DQ', "DQ extension")
            oh.update("EXTVER",1, "DQ ext version")
            im = out[1].data
            mask = where((im<100) | ((im<10) & (im>-10)))
            bpm = zeros([1024,1024],dtype=int32)
            bpm[mask]=-999   
            hdu= pf.ImageHDU(header=oh, data=bpm)
            out.append(hdu)
            out.flush()


            oh = out[2].header
            oh.update("EXTNAME",'DQ', "DQ extension")
            oh.update("EXTVER",2, "VAR ext version")
            im = out[2].data
            mask = where((im<100) | ((im<10) & (im>-10)))
            bpm = zeros([1024,1024],dtype=int32)
            bpm[mask]=-999   
            hdu= pf.ImageHDU(header=oh, data=bpm)
            out.append(hdu)
            out.flush()

        out.close()

    #o# OUTPUT Files
    # output filename can contain @ symbol
    # number of input and output names must be the same
    


    # BMP for NICI?
    # The steps to produce a mask are:
    # mask = where((im<100) | ((im<10) & (im>-10)))
    # bpm = zeros([1024,1024],dtype=<type of infile>)
    # mark the bad pixels
    # bpm[mask]=-999   


    ## LOOP over the number of input files and create VAR and DQ

    # create the DQ frame, if it doesn't already exist
    # The preliminary DQ frame is constructed by using the bad pixel
    # mask to set bad pixels to 1, pixels in the non-linear regime to 2,
    # and saturated pixels to 4.


    #
    # Get WCS info 
    # Change EXTVER to 1 on all extension.
    # Add NEXTEND to [0]
    # Add EXPTIME
    # Add CHANNEL keyword with 'Red' or 'Blue' value
    # Write GEM-TLM  /UT Last modification with GEMINI
    # WRite PREPARE / UT Time stamp for NPREPARE
    # BPMFILE ?

def order_wcs(header, channel, ext):
    """
      Arrange the wcs cards in the default order at 
      the end of the header.
    """
    # TODO: Need to put copy header to temp one in case of error

    nwcs = []
    hcc = header.ascard
    for h in ['RADECSYS','CTYPE1','CTYPE2','CRPIX1','CRPIX2','CRVAL1',
         'CRVAL2','CD1_1','CD1_2','CD2_1','CD2_2']:
        card = hcc[h] 
        nwcs.append(card)
        del  header[h]

    for cc in nwcs:
        hcc.append(cc)

    header.update("CHANNEL",channel, "")
    header.update("EXTNAME",'SCI', "SCIENCE extension",after='GCOUNT')
    del header['EXTVER']
    header.update("EXTVER",ext, "SCIENCE ext version",after='EXTNAME')

    return header

if __name__ == '__main__':

    # Parse input arguments
    usage = 'usage: %prog file_list [options]'

    p = optparse.OptionParser(usage=usage, version='niciprepare_1.0')
    p.add_option('--prefix', action='store', type='string', default='n', 
                 help='prefixes to use for prepared files')
    p.add_option('--idir', action='store', type='string',
                 default='', help='Input directory pathname')
    p.add_option('--odir', action='store', type='string',
                 default='', help='Output directory pathname')
    p.add_option('--clobber', default=True, action='store_true', 
                 help='Clobber output files?')
    p.add_option('--logfile', action='store', type='string', default='',
                 help='logfile name ')
    p.add_option('--sci', action='store', type='string', default='SCI',
                 help='Name of science extension') 
    p.add_option('--var', action='store', type='string', default='VAR', 
                 help='Name of var extension') 
    p.add_option('--dq', action='store', type='string', default='DQ', 
                 help='Name of dq extension') 
    p.add_option('--fl_var', action='store',default=False, 
                 help='create var frame?')
    p.add_option('--fl_dq', action='store',default=False, 
                 help='create dq frame?')
    p.add_option('--debug','-d', action='store_true',default=False, 
                 help='toggle debug messages')
    p.add_option('--v','-v', action='store_true',default=False, 
                 help='toggle on verbose mode')

    (par, args) = p.parse_args() 

    if par.debug:
        par.verbose = True
        print 'options: ', par
        print 'args: ', args

    iList= args 
    if len(iList) == 0:
       print 'options: ', par
       p.print_help()
       sys.exit(0)
    # Generate an input list from inputdir plus an @ file or a unix template 
    # 
    print 'INPUTLL:',iList
    inputList = getFileList(iList)
    niciprepare (inputList, outprefix=par.prefix, inputdir=par.idir,
                 outputdir=par.odir, clobber=par.clobber, 
                 logfile=par.logfile, sci_ext=par.sci, var_ext=par.var,
                 dq_ext=par.dq, fl_var=par.fl_var, fl_dq=par.fl_dq,
                 verbose=par.v)

#def niciprepare(inputList,outprefix='n',inputdir='',outputdir='',
#                clobber=True, logfile='', sci_ext='SCI', var_ext='VAR',
#                dq_ext='DQ', fl_var=False, fl_dq=False, verbose=True):
