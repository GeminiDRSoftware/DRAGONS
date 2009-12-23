#!/usr/bin/env python
#
import optparse

import os, sys
from os.path import join
import pyfits as pf
import re
import scipy.ndimage as nd
from niciTools import getFileList, fits_utc, order_wcs
from ncscience import shift_function
from numpy import zeros,where,dtype,int32,size,asarray,float32,isnan, nan_to_num
from niciCntrd import nici_cntrd

def ncprepare (inputs, oprefix='n', idir='', odir='', clobber=False, 
              fl_var=False, fl_dq=False):

    """
        inputs =               Input Fits files on a list
        (oprefix = 'n')        Prefix for output image(s)
        (idir = '')            Path for input raw files
        (odir = '')            Output directory name
        (clobber = False)       Replace output file if exits
     nyi (fl_var = False)       Create variance frame?
     nyi (fl_dq = False)        Create data quality frame?

     NOTE: 'nyi' above means 'not yet implemented'

     DESCRIPTION:
     Procedure update the extension header of NICI science data with
     the center of the mask.
    
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
           --> niciprepare *.fits odir=/tmp

        Python_shell: 
           - Your UNIX PYTHONPATH shell variable should have an
             entry for the directory 'nici', like: /home/myname/nici
           >>> import niciprepare as nicip
           >>> # Make the input list, assuming you are in the data-directory
           >>> import glob
           >>> flis=glob.glob('*.fits')
           >>> nicip.niciprepare(flis,idir='/data1/data/tmp',odir='/tmp')
      
        Unix_shell:
           % cd /home/myname/nici
           % # to get help try: niciprepare.py -h
           % niciprepare.py /data1/data/tmp/*.fits --odir='/tmp'

    """
    
    odir = join(odir,'')     # Make sure we have a trailing '/'

    if (len(oprefix) > 10):
        print "input prefix is too long, returning"
        return
    
    if (len(idir) > 0 and not os.access(idir, os.R_OK)):
        print "*** Warning: Inputdir:"+idir+' not readable.'
        return
    if (len(odir) > 0 and not os.access(odir, os.W_OK)):
        print "*** Warning: Outputdir: "+odir+' not writeable.'
        return

    # Expand 'inputs' to a list of files
    inputs = getFileList(inputs)

    ninlen=size(inputs) 
    for fname in inputs:
     
        fpath = join(idir,fname.strip())
        hdus = pf.open(fpath)
        if (hdus[0].header).get("INSTRUME") != 'NICI':
            print 'File:',fname,'is not for NICI. Ignored.' 
            hdus.close()
            continue

        if (hdus[0].header).get("PREPARE"):
            print fpath," already fixed using ncprepare. -skipped"
            continue

        if (len(hdus) != 3):
            print 'File:',fname,'does not have 2 image extensions. Ignored' 
            hdus.close()
            continue

        if (hdus[0].header['obsclass']).lower() != 'science':
            print fname,' is not SCIENCE frame. -skipped'
            hdus.close()
            continue

        if '-ENG' in hdus[0].header['obsid']:
            print fname,'Exposure is ENG data . -skipped'
            hdus.close()
            continue

        print fname,hdus[0].header['object'],hdus[0].header['obstype'],\
                    hdus[0].header['obsclass']

        # Get the root name and extension
        root,ext = os.path.basename(fpath).split('.')[0:2]

        outfile = odir+oprefix+root+'.'+ext
        if (os.access(outfile, os.R_OK)):
            if clobber:
               os.remove(outfile)
            else:
	       print 'File: ',outfile, 'already exists.'
	       return

        # Create empty output file
        out = pf.open(outfile,mode='append')
        
        out.append(hdus[0]) 
        # Copy input header
        ophu = out[0].header
        
        date_time = fits_utc()
    
        # Time Stamps
        ophu.update("GEM-TLM",date_time, "UT Last modification with GEMINI")
        ophu.update("PREPARE",date_time, "UT Time stamp for NPREPARE")

        out.flush()             # write to disk, leave 'out' open  

        # If the exposure is an ADI (100% flux on red or blue frame) then set.
        dichr = str(hdus[0].header['dichroic']).lower()
        adi = ''
        if 'mirror' in dichr:
            adi = 'blue'
        if 'open' in dichr:
            adi = 'red'

        for ext in [1,2]:       # Do red and Blue
            out.append(hdus[ext])
            oh = out[ext].header       
            im = out[ext].data       

            # Skip over the frames that do not have flux
            if ext == 1 and adi == 'blue': 
                print outfile,'ADI mode [no red flux ]' 
                continue
            if ext == 2 and adi == 'red':
                print outfile,'ADI mode [no blue flux ]' 
                continue

            # Get the center of the mask automatically. If not possible
            # nici_cntrd will display the image for the user to click
            # on the mask.
            if ext == 2:
                # register blue frame to red's frame coordinate system
                im = asarray(im,dtype=float32)
                im = nd.geometric_transform (nan_to_num(im),shift_function)

            updated,xcen,ycen,im = nici_cntrd(im,oh,center_im=False)
            print outfile,['red','blue'][ext-1] +': xcen: %.2f,ycen: %.2f' % (xcen,ycen)  

            # Order wcs -in place.
            order_wcs(oh)
            oh.update("EXTNAME",'SCI', "SCIENCE extension",after='GCOUNT')
            oh.__delitem__("EXTVER")
            oh.update("EXTVER",ext, "SCIENCE ext version",after='EXTNAME')
            out.flush()
           
        """
        # THIS CODE IS COMMENTS NOW until we decided if these extensions
        # are really needed.
            if (fl_var):
                # Write VAR 1,2
                oh = out[1].header
                oh.update("EXTNAME",'VAR', "VAR extension")
                oh.update("EXTVER",1, "VAR ext version")
                #hdu= pf.ImageHDU(header=oh, data=out[1].data)
                hdu= pf.ImageHDU(header=oh)
                out.append(hdu)
                out.flush()

                oh = out[2].header
                oh.update("EXTNAME",'VAR', "VAR extension")
                oh.update("EXTVER",2, "VAR ext version")
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
        """
        out.close()

    return 

if __name__ == '__main__':

    # Parse input arguments
    usage = 'usage: %prog file_list [options]'

    p = optparse.OptionParser(usage=usage, version='ncprepare_1.0')
    p.add_option('--oprefix', action='store', type='string', default='n', 
                 help='prefixes to use for prepared files')
    p.add_option('--idir', action='store', type='string',
                 default='', help='Input directory pathname')
    p.add_option('--odir', action='store', type='string',
                 default='', help='Output directory pathname')
    p.add_option('--clobber', default=False, action='store_true', 
                 help='Clobber output files?')
    p.add_option('--fl_var', action='store',default=False, 
                 help='create var frame?')
    p.add_option('--fl_dq', action='store',default=False, 
                 help='create dq frame?')

    (par, args) = p.parse_args() 

    iList= args 
    if len(iList) == 0:
       print 'options: ', par
       p.print_help()
       sys.exit(0)
    # Generate an input list from idir plus an @ file or a unix template 
    # 
    inputs = getFileList(iList)
    ncprepare (inputs, oprefix=par.oprefix, idir=par.idir,
                 odir=par.odir, clobber=par.clobber, 
                 fl_var=par.fl_var, fl_dq=par.fl_dq)

