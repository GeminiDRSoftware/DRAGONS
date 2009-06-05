#!/usr/bin/env python
#
import optparse
from niciTools import getFileList

import os
import numpy as np
import pyfits as pf

def nicimkflats(flats_ls,inputdir='',outputdir='',clobber=True,\
                logfile='',verbose=True):
    """
         flats_ls = ""              Input NICI flats files
         (inputdir = "")             Path for input directory
         (outputdir = "")             Path for output directory
         (clobber = yes)            Clobber output file?
     nyi (logfile = "")             Logfile
     nyi (verbose = yes)            Verbose


     Produces flats given a list of fits files with flats

     EXAMPLES: 
        PYRAF:
           - Make sure you have the NICI scripts in a designated 
             directory, e.g. 'nici'.
           - Make sure the UNIX PYTHONPATH shell variable has an
             entry for the directory 'nici', like: /home/myname/nici
           - Start Pyraf
           --> #define the task
           --> pyexecute('/home/myname/nici/nicimkflats_iraf.py')
           --> #execute
           --> nicimkflats *.fits outputdir=/tmp
 
        Python_shell:     
           - Make sure your UNIX PYTHONPATH shell variable has an
             entry for the directory 'nici', like: /home/myname/nici
           >>> import nicimkflats as nmf
           >>> # Make the input list, assuming you are in the data-directory
           >>> import glob
           >>> flis=glob.glob('*.fits')
           >>> nmf.nicimkflats(flis,inputdir='/data1/data/tmp',outputdir='/tmp')
      
        Unix_shell:
           % cd /home/myname/nici
           % # to get help try: nicimkflats.py -h
           % nicimkflats.py /data1/data/tmp/*.fits --odir='/tmp'
    """

    # Make sure paths end with a '/'
    if (len(outputdir)>0 and outputdir[-1] != '/'):
        outputdir += '/'
    if (len(inputdir)>0 and inputdir[-1] != '/'):
        inputdir += '/'

    flat_r=[]
    flat_b=[]
    dark_r=[]
    dark_b=[]
    

    hlis={}
    # We need to find which extension has the correct value
    #for a in flats_fits:
    if len(flats_ls) == 0:
        print "\n WARNING: Input list is empty.\n"
        return

    for file in flats_ls:

        a = pf.open(inputdir+file)

        ah0 = a[0].header
        if (a[0].header['INSTRUME'] != 'NICI'):
            print file, 'is not a NICI file, skipped.'
            a.close()
            continue
        obstype = a[0].header['OBSTYPE']
        if (obstype != 'FLAT' and obstype != 'DARK'):
            print file, 'is not a FLAT frame, skipped.'
            a.close()
            continue
        ss=ah0.get('GCALSHUT')
        if (ss == None and obstype == 'FLAT'):
            print 'PHU: ',a._HDUList__file.name,'[0], does not have GCALSHUT.'
            a.close()
            continue
        
        ah1 = a[1].header
        ah2 = a[2].header
        print file,ah0['object'],ah0['obstype'],ah0['obsclass'],ss
        if (obstype == 'DARK'):
	    if ah1.get('FILTER_R'): 
                linearzR = ah1['ITIME_R']*ah1['NCOADD_R']*ah1['NDR_R']
                linearzB = ah2['ITIME_B']*ah2['NCOADD_B']*ah2['NDR_B']
		dark_r.append(a[1].data/linearzR)
		dark_b.append(a[2].data/linearzB)
		exr=1; exb=2
	    else:
                linearzR = ah2['ITIME_R']*ah2['NCOADD_R']*ah2['NDR_R']
                linearzB = ah1['ITIME_B']*ah1['NCOADD_B']*ah1['NDR_B']
		dark_r.append(a[2].data/linearzR)
		dark_b.append(a[1].data/linearzB)
		exr=2; exb=1
            if ~hlis.has_key('dark_rb'):
                hlis['dark_rb'] = file
        else:     #obstype is FLAT and there is GCALSHUT
            if ('OPEN' in ss):
                # Shutter is open.
                if ah1.get('FILTER_R'): 
                    linearzR = ah1['ITIME_R']*ah1['NCOADD_R']*ah1['NDR_R']
                    linearzB = ah2['ITIME_B']*ah2['NCOADD_B']*ah2['NDR_B']
                    flat_r.append(a[1].data/linearzR)
                    flat_b.append(a[2].data/linearzB)
                    exr=1; exb=2
                else:
                    linearzR = ah2['ITIME_R']*ah2['NCOADD_R']*ah2['NDR_R']
                    linearzB = ah1['ITIME_B']*ah1['NCOADD_B']*ah1['NDR_B']
                    flat_r.append(a[2].data/linearzR)
                    flat_b.append(a[1].data/linearzB)
                    exr=2; exb=1
                if ~hlis.has_key('flat_rb'):
                    hlis['flat_rb'] = file
            else:
                # Shutter is closed.
                if ah1.get('FILTER_R'): 
                    linearzR = ah1['ITIME_R']*ah1['NCOADD_R']*ah1['NDR_R']
                    linearzB = ah2['ITIME_B']*ah2['NCOADD_B']*ah2['NDR_B']
                    dark_r.append(a[1].data/linearzR)
                    dark_b.append(a[2].data/linearzB)
                    exr=1; exb=2
                else:
                    linearzR = ah2['ITIME_R']*ah2['NCOADD_R']*ah2['NDR_R']
                    linearzB = ah1['ITIME_B']*ah1['NCOADD_B']*ah1['NDR_B']
                    dark_r.append(a[2].data/linearzR)
                    dark_b.append(a[1].data/linearzB)
                    exr=2; exb=1
                if ~hlis.has_key('dark_rb'):
                    hlis['dark_rb'] = file

	a.close()

    #print 'len:',(len(flat_r),)+np.shape(flat_r[0])
    ndarkr= len(dark_r)
    ndarkb= len(dark_b)
    print 'Number of flats_red:  ',len(flat_r)
    print 'Number of flats_blue: ',len(flat_b)
    print 'Number of darks_red: ',ndarkr
    print 'Number of darks_blue:',ndarkb

    # To save memory do R, del, B, del, write.
    cube_dark_r = np.zeros((len(dark_r),)+np.shape(dark_r[0]))

    ndarkr = len(dark_r)
    for a in range(len(dark_r)):
	cube_dark_r[a,:,:] = dark_r[a]

    # Save memory
    del dark_r

    dark_r_median = np.median(cube_dark_r,axis=0)

    print outputdir+"sky_red.fits"
    fh = pf.getheader(inputdir+hlis['dark_rb'])
    pf.writeto(outputdir + 'sky_red.fits', \
          dark_r_median.astype(np.float32),header=fh,clobber=clobber)
    print outputdir+"skycube_red.fits"
    pf.writeto(outputdir + 'skycube_red.fits', \
          cube_dark_r.astype(np.float32),header=fh,clobber=clobber)
    del cube_dark_r

    # BBBBBBBBB
    cube_dark_b = np.zeros((len(dark_b),)+np.shape(dark_b[0]))

    for a in range(len(dark_b)):
	cube_dark_b[a,:,:] = dark_b[a]
    del dark_b
	
    dark_b_median = np.median(cube_dark_b,axis=0)

    print outputdir+"sky_blue.fits cubes."
    pf.writeto(outputdir + 'sky_blue.fits', \
           dark_b_median.astype(np.float32),header=fh,clobber=clobber)
    print outputdir+"skycube_blue.fits"
    pf.writeto(outputdir + 'skycube_blue.fits', \
          cube_dark_b.astype(np.float32),header=fh,clobber=clobber)
    del cube_dark_b

    print ' Preparing flats..'
    # FLAT RRRRRR
    cube_flat_r = np.zeros((len(flat_r),)+np.shape(flat_r[0]))

    nflatsr = len(flat_r)
    for a in range(nflatsr):
	cube_flat_r[a,:,:] = flat_r[a]
    del flat_r

    for a in range(nflatsr):
        cube_flat_r[a,:,:] -= dark_r_median

    for a in range(nflatsr):
        cube_flat_r[a,:,:] /= np.median(cube_flat_r[a,:,:])

    if (ndarkr > 1):
       flat_r_median = np.median(cube_flat_r,axis=0)
    else:
       flat_r_median = cube_flat_r
    del cube_flat_r

    fhon = pf.getheader(inputdir+hlis['flat_rb'])
    file_red = outputdir + 'flat_red.fits'
    print file_red
    pf.writeto(file_red,flat_r_median.astype(np.float32),
         header=fhon,clobber=clobber)

    #FLAT BBBBBB
    cube_flat_b = np.zeros((len(flat_b),)+np.shape(flat_b[0]))

    nflatsb = len (flat_b)
    for a in range(nflatsb):
	cube_flat_b[a,:,:] = flat_b[a]
    del flat_b
	
    for a in range(nflatsb):
        cube_flat_b[a,:,:] -= dark_b_median

    for a in range(nflatsb):
        cube_flat_b[a,:,:] /= np.median(cube_flat_b[a,:,:])

    if (ndarkb > 1):
       flat_b_median = np.median(cube_flat_b,axis=0)
    else:
       flat_b_median = cube_flat_b
    del cube_flat_b
    
    file_blue = outputdir + 'flat_blue.fits'
    print file_blue
    pf.writeto(file_blue,flat_b_median.astype(np.float32),
         header=fhon,clobber=clobber)

    return 'Make flats done'

if __name__ == '__main__':

    # Parse input arguments
    usage = 'usage: %prog file_list [options]'

    p = optparse.OptionParser(usage=usage, version='nicimkflats Mar 2009')
    p.add_option('--idir', action='store', type='string',
                 default='', help='Input directory pathname')
    p.add_option('--odir', action='store', type='string',
                 default='', help='Output directory pathname')
    p.add_option('--clobber', default=True, action='store_true',
                 help='Clobber output files?')
    p.add_option('--logfile', action='store', type='string', default='',
                 help='logfile name ')
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
       exit(0)
    # Generate an input list from inputdir plus an @ file or a unix template
    #
    inputList = getFileList(iList)
    nicimkflats(inputList,inputdir=par.idir, outputdir=par.odir, 
                logfile=par.logfile, clobber=par.clobber, verbose=par.v)

