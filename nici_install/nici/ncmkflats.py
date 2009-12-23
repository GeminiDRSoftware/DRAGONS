#!/usr/bin/env python
#
import optparse
from niciTools import getFileList,nici_noise
from scipy.signal import medfilt2d
from robustsigma import robust_sigma

import os
from os.path import join
import numpy as np
import pyfits as pf
import time

def ncmkflats(inputs, idir='',odir='',sigma=6, clobber=True,\
                suffix='default', logfile=''):
    """
         inputs = ""         NICI flats and darks file list
         (idir = "")         Path for input directory
         (odir = "")         Path for output directory
         (sigma=6)           Set to Nan all pixel above this value from the median
         (clobber = yes)     Clobber output file?
         (suffix='default')  If 'default' it will take the rootname of 
                             the first element in the input list.
     nyi (logfile = "")      Logfile

     nyi: Not Yet Implemented

     PARAMETERS:
        Sigma=6  -- All pixels in the flat off (dark)
                 that are way above the median value
                 are set to NaN *** The selection is
                 done in the following way :
                 subtract a low-pass to the flat_off
                 to keep only single 'hot' pixels,
                 and select all pixels more than 6
                 sigma (estimated through
                 robust_sigma) of the median-filtered image.


     DESCRIPTION:
        Produces calibration  given a list of fits files with flats,
        darks and sky exposures

     OUTPUT FILES:
        flat_red_<suffix>.fits
        flat_blue_<suffix>.fits
        sky_red_<suffix>.fits
        sky_blue_<suffix>.fits
        skycube_red_<suffix>.fits
        skycube_blue_<suffix>.fits


     EXAMPLES: 
        PYRAF:
           - Make sure you have the NICI scripts in a designated 
             directory, e.g. 'nici'.
           - Make sure the UNIX PYTHONPATH shell variable has an
             entry for the directory 'nici', like: /home/myname/nici
           - Start Pyraf
           --> #Load the NICI package
           --> set nicid=/home/zarate/nici/     # Notice ending '/'
           --> task nici = nicid$nici.cl        # Define the package
           --> nici                             # load all nici scripts
           --> ncmkflats *.fits odir=/tmp
 
        Python_shell:     
           - Make sure your UNIX PYTHONPATH shell variable has an
             entry for the directory 'nici', like: /home/myname/nici
           >>> import nici as nc
           >>> nc.ncmkflats('*.fits',idir='/data1/data/tmp',odir='/tmp')
      
        Unix_shell:
           % cd /home/myname/nici
           % # to get help try: ncmkflats.py -h
           % ncmkflats.py /data1/data/tmp/*.fits --odir='/tmp'
    """

    if (len(idir) > 0 and not os.access(idir, os.R_OK)):
        print "*** Warning: Inputdir:"+idir+' not readable.'
        return
    if (len(odir) > 0 and not os.access(odir, os.W_OK)):
        print "*** Warning: Outputdir: "+odir+' not writeable.'
        return

    t1=time.time()
    flat_r=[]
    flat_b=[]
    dark_r=[]
    dark_b=[]
    
    hlis={}
    ilist = getFileList(inputs)

    # We need to find which extension has the correct value
    #for a in flats_fits:
    if len(ilist) == 0:
        print "\n WARNING: Input list is empty.\n"
        return

    # Find rootname from the first element of 'inputs' if
    # 'suffix' is not 'default'

    if suffix == 'default':
       root = os.path.basename(ilist[0]).split('.')[0]
       # If root is of the form 20yymmddSnnnn ==> get 20yymmdd
       if len(root) == 14 and root[:3] == 'S20': root = root[:9]
       suffix = root

    # Preread the list to sky files (same as flats_off or darks)
    sky_ls=[]
    for file in ilist:
        print file
        ff = pf.open(join(idir,file))
        if ff[0].header['OBSTYPE'] == 'DARK':
            sky_ls.append(file) 
        if (ff[0].header['OBSTYPE'] == 'FLAT') and \
           (ff[0].header['GCALSHUT'] == 'CLOSED'):
            sky_ls.append(file) 

    for file in ilist:

        hdus = pf.open(join(idir,file))

        ah0 = hdus[0].header
        if (ah0['INSTRUME'] != 'NICI'):
            print file, 'is not a NICI file, skipped.'
            hdus.close()
            continue
        obstype = ah0['OBSTYPE']
        if (obstype != 'FLAT' and obstype != 'DARK'):
            print file, 'is not a FLAT frame, skipped.'
            hdus.close()
            continue
        ss = ah0.get('GCALSHUT')
        if (ss == None and obstype == 'FLAT'):
            print 'PHU: ',hdus._HDUList__file.name,'[0], does not have GCALSHUT.'
            hdus.close()
            continue
        
        # EXTENSION 1 is RED !!
        # EXTENSION 2 is BLUE !!
        ah1 = hdus[1].header      # ah1.get('FILTER_R') needs to be TRUE
        ah2 = hdus[2].header

        print file,ah0['object'],ah0['obstype'],ah0['obsclass'],ss
        if (obstype == 'DARK'):  # or (Flats OFF, off)
            linearzR = ah1['ITIME_R']*ah1['NCOADD_R']
            linearzB = ah2['ITIME_B']*ah2['NCOADD_B']
	    dark_r.append(hdus[1].data/linearzR)
	    dark_b.append(hdus[2].data/linearzB)
            if ~hlis.has_key('dark_rb'):
                hlis['dark_rb'] = file
        else:     #obstype is FLAT and there is GCALSHUT
            if ('OPEN' in ss):
                # Shutter is open.
                linearzR = ah1['ITIME_R']*ah1['NCOADD_R']
                linearzB = ah2['ITIME_B']*ah2['NCOADD_B']
                flat_r.append(hdus[1].data/linearzR)
                flat_b.append(hdus[2].data/linearzB)
                if ~hlis.has_key('flat_rb'):
                    hlis['flat_rb'] = file
            else:
                # Shutter is closed.
                linearzR = ah1['ITIME_R']*ah1['NCOADD_R']
                linearzB = ah2['ITIME_B']*ah2['NCOADD_B']
                dark_r.append(hdus[1].data/linearzR)
                dark_b.append(hdus[2].data/linearzB)
                if ~hlis.has_key('dark_rb'):
                    hlis['dark_rb'] = file

	hdus.close()

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
    for i in range(len(dark_r)):
	cube_dark_r[i,:,:] = dark_r[i]

    # Save memory
    del dark_r

    dark_r_median = np.median(cube_dark_r,axis=0)
    del cube_dark_r

    cube_dark_b = np.zeros((len(dark_b),)+np.shape(dark_b[0]))

    for i in range(len(dark_b)):
	cube_dark_b[i,:,:] = dark_b[i]
    del dark_b
	
    dark_b_median = np.median(cube_dark_b,axis=0)
    del cube_dark_b

                              # *** here, all pixels in the flat off (dark)
                              # that are way above the median value
                              # are set to NaN *** The selection is
                              # done in the following way :
                              # subtract a low-pass to the flat_off
                              # to keep only single 'hot' pixels,
                              # and select all pixels more than 6
                              # sigma (estimated through
                              # robust_sigma) of the median-filtered image.


    dev_red = dark_r_median - medfilt2d(dark_r_median,7)
    dev_red /= robust_sigma(dev_red)

    dev_blue = dark_b_median - medfilt2d(dark_b_median,7)
    dev_blue /= robust_sigma(dev_blue)


    dark_r_median[np.where(np.abs(dev_red) > sigma)] = np.NaN
    dark_b_median[np.where(np.abs(dev_blue) > sigma)] = np.NaN

    print ' Preparing flats..'
    # FLAT RRRRRR
    cube_flat_r = np.zeros((len(flat_r),)+np.shape(flat_r[0]))

    nflatsr = len(flat_r)
    for i in range(nflatsr):
	cube_flat_r[i,:,:] = flat_r[i]
    del flat_r

    for i in range(nflatsr):
        cube_flat_r[i,:,:] -= dark_r_median

    for i in range(nflatsr):
        cube_flat_r[i,:,:] /= np.median(cube_flat_r[i,:,:])

    if (ndarkr > 1):
       flat_r_median = np.median(cube_flat_r,axis=0)
    else:
       flat_r_median = cube_flat_r
    del cube_flat_r

    # set very low or very high values of the flat to 
    # NaN (<5% or >150% of the median value)
    flat_r_median[np.where((flat_r_median > 1.5)|(flat_r_median < 0.05))] = np.nan

    fhon = pf.getheader(join(idir,hlis['flat_rb']))
    file_red = join(odir,'flat_red_'+suffix+'.fits')
    print file_red
    pf.writeto(file_red,flat_r_median.astype(np.float32),
         header=fhon,clobber=clobber)

    #FLAT BBBBBB
    cube_flat_b = np.zeros((len(flat_b),)+np.shape(flat_b[0]))

    nflatsb = len(flat_b)
    for i in range(nflatsb):
	cube_flat_b[i,:,:] = flat_b[i]
    del flat_b
	
    for i in range(nflatsb):
        cube_flat_b[i,:,:] -= dark_b_median

    for i in range(nflatsb):
        cube_flat_b[i,:,:] /= np.median(cube_flat_b[i,:,:])

    if (ndarkb > 1):
       flat_b_median = np.median(cube_flat_b,axis=0)
    else:
       flat_b_median = cube_flat_b
    del cube_flat_b
    
    flat_b_median[np.where((flat_b_median > 1.5)|(flat_b_median < 0.05))] = np.nan
    file_blue = join(odir,'flat_blue_'+suffix+'.fits')
    print file_blue
    pf.writeto(file_blue,flat_b_median.astype(np.float32),
         header=fhon,clobber=clobber)

    cube_sky_r = np.zeros((len(sky_ls),)+(1024,1024))
    cube_sky_b = np.zeros((len(sky_ls),)+(1024,1024))

    k = 0
    for file in sky_ls:

        hdus = pf.open(join(idir,file))

        ah0 = hdus[0].header
        if (hdus[0].header['INSTRUME'] != 'NICI'):
            print file, 'is not a NICI file, skipped.'
            hdus.close()
            continue
        
        # EXTENSION 1 is RED !!
        # EXTENSION 2 is BLUE !!
        ahr = hdus[1].header      # ah1.get('FILTER_R') needs to be TRUE
        ahb = hdus[2].header

        print 'SKY::',file,ah0['object'],ah0['obstype'],ah0['obsclass']
        linearzR = ahr['ITIME_R']*ahr['NCOADD_R']
        linearzB = ahb['ITIME_B']*ahb['NCOADD_B']
	cube_sky_r[k,:,:] = nici_noise(hdus[1].data/linearzR)
	cube_sky_b[k,:,:] = nici_noise(hdus[2].data/linearzR)
        k += 1
        if ~hlis.has_key('dark_rb'):
           hlis['dark_rb'] = file

	hdus.close()

    fh = pf.getheader(join(idir,hlis['dark_rb']))
    ofile = join(odir,'sky_red_'+suffix+'.fits')
    print ofile
    pf.writeto(ofile, np.median(cube_sky_r.astype(np.float32),axis=0),
               header=fh,clobber=clobber)
    ofile = join(odir,'sky_blue_'+suffix+'.fits')
    print ofile
    pf.writeto(ofile, np.median(cube_sky_b.astype(np.float32),axis=0),
              header=fh,clobber=clobber)

    ofile = join(odir,'skycube_red_'+suffix+'.fits')
    print ofile
    pf.writeto(ofile, cube_sky_b.astype(np.float32),header=fh,clobber=clobber)
    ofile = join(odir,'skycube_blue_'+suffix+'.fits')
    print ofile
    pf.writeto(ofile, cube_sky_b.astype(np.float32),header=fh,clobber=clobber)

    print 'TIME:',time.time()-t1
    return 'Make flats done'

if __name__ == '__main__':

    # Parse input arguments
    usage = 'usage: %prog file_list [options]'

    p = optparse.OptionParser(usage=usage, version='ncmkflats Sep 2009')
    p.add_option('--idir', action='store', type='string',
                 default='', help='Input directory pathname')
    p.add_option('--odir', action='store', type='string',
                 default='', help='Output directory pathname')
    p.add_option('--clobber', default=False, action='store_true',
                 help='Clobber output files?')
    p.add_option('--sigma', default=6, type='float', action='store',
                 help='Sigmas above median')
    p.add_option('--suffix', default='default', type='string',
                 help='postfix suffix to outfiles')
    p.add_option('--logfile', action='store', type='string', default='',
                 help='logfile name ')


    (par, args) = p.parse_args()

    iList= args
    if len(iList) == 0:
       print 'options: ', par
       p.print_help()
       exit(0)
    # Generate an input list from idir plus an @ file or a unix template
    #
    inputList = getFileList(iList)
    ncmkflats(inputList,idir=par.idir, odir=par.odir, logfile=par.logfile, 
             clobber=par.clobber, suffix=par.suffix, sigma=par.sigma)

