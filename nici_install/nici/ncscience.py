#!/usr/bin/env python
#
import optparse
import sys
from niciTools import getFileList,parangle,dmstod,nici_noise, rebin

import pyfits as pf
import scipy.ndimage as nd
from niciCntrd import nici_cntrd
import scipy.signal
import scipy.interpolate as scint
import os
from os.path import join
from dist_circle import dist_circle
import numpy as np
from numpy import pi
from numpy import size
#import gspline as gspline
from convolve import boxcar
import time

def ncscience (inputs,idir='',odir='', fdir='',fsuffix='default',central=False, 
               suffix='default', clobber=False, dobadpix=True,
               pad=False, logfile=''):
    """

        inputs =                  Input science Fits files
        (idir = '')               Directory for inputs 
        (odir = '')               Output directory name
        (fdir = '')               Directory where calibration files resides
        (fsuffix = 'default')       Calibration files suffix. If default, uses rootname's.
        (central = False)         Reduce the output image to 512x512
        (suffix='default')      Dataset name.  If 'default' it will take
                                  the rootname of the first element in the input list.
        (dobadpix = True)         Correct bad pixels the best we can
        (pad = False)              Use an extended area for interpolation
        (clobber  = False)         Replace output file if exits
    nyi (logfile = '')            Logfile
   
        nyi: Not Yet Implemented

        EXAMPLES:
        PYRAF:
           - Make sure you have the NICI scripts in a designated
             directory, e.g. 'nici'.
           - Your UNIX PYTHONPATH shell variable should have an
             entry for the directory 'nici', like: /home/myname/nici
           - Start Pyraf
           --> #define the task
           --> pyexecute('/home/myname/nici/niciscience_iraf.py')
           --> #execute
           --> niciscience *.fits odir=/tmp

        Python_shell:
           - Your UNIX PYTHONPATH shell variable should have an
             entry for the directory 'nici', like: /home/myname/nici
           >>> import niciscience as nicis
           >>> # Make the input list, assuming you are in the data-directory
           >>> import glob
           >>> flis=glob.glob('*.fits')
           >>> nicis.niciscience(flis,idir='/data1/data/tmp',\
                odir='/tmp'i,flats='/data/nici/200901/flats')

        Unix_shell:
           % cd /home/myname/nici
           % # to get help try: niciscience.py -h
           % niciscience.py /data/nici/tmp/*.fits --odir='/tmp'

    """


    # Make sure directories have ending '/'
    idir = join(idir,'')
    odir = join(odir,'')
    fdir = join(fdir,'')

    if (not os.access(odir,os.W_OK)):
       print 'Warning: Cannot write into output directory (odir):',odir
       return
    if (not os.access(fdir,os.R_OK)):
        print '\n     ERROR: Cannot read flats directory (fdir):',fdir
        return

    if (not central):
       imsize = (1024,1024)
    else:
       imsize = (512,512)

    science_lis = getFileList(inputs)
    if (len(science_lis) == 0):
        print 'ERROR: No input files selected.'
        return

    if suffix == 'default':
       # Takes the first entry in the input list and
       # grab its rootname.
       root = os.path.basename(science_lis[0]).split('.')[0]
       # If root is of the form S20yymmddSnnnn ==> get S20yymmdd
       if len(root[1:])==14 and root[1:4]=='S20': root = root[1:10]
       suffix = root

    # All the output files are of the form: <function>_suffix.fits
    # so prepend '_'
    if suffix[0] != '_':
        suffix = '_'+suffix
 
    if fsuffix == 'default':
       fsuffix = suffix
    else:
       fsuffix = fsuffix.strip()         

    
    try:
	flat_red  = pf.getdata(fdir + 'flat_red'+fsuffix+'.fits')
	flat_blue = pf.getdata(fdir + 'flat_blue'+fsuffix+'.fits')
	sky_red   = pf.getdata(fdir + 'sky_red'+fsuffix+'.fits')
	sky_blue  = pf.getdata(fdir + 'sky_blue'+fsuffix+'.fits')
    except IOError:
        dn = suffix
        print 'ERROR opening flat_red'+dn+'.fits,flat_blue'+dn+'.fits,'
        print '      sky_red'+dn+'.fits or sky_blue'+dn+'.fits in dir:',fdir
        return

    
    # Get the PHU from the first element of the input list. It will be
    # used as a PHU for the output products
    gphu = pf.getheader(idir+science_lis[0],0)

    science_lis = np.sort(science_lis)
    pa = _parangle_list (science_lis, idir, '')

    # List of filters
    hdr1 = pf.getheader(idir+science_lis[0],1) 
    filter_red = hdr1['filter_r'].split('_')[1]
    hdr2 = pf.getheader(idir+science_lis[0],2)
    filter_blue = hdr2['filter_b'].split('_')[1]
    # l1 list
    fr = {'G0724': 1.587,
        'G0728': 1.603,
        'G0720': 1.628,
        'G0735': 1.628,
        'G0742': 1.578,
        'G0740': 1.652,
        'G0714': 1.701,
        'G0710': 2.2718,
        'G0708': 4.68,
        'G0707': 3.78,
        'G0706': 2.12,
        'G0705': 2.15,
        'G0704': 2.20,
        'Block': (-1),
        }
    l1 = fr[filter_red]
    # l2 list
    fb = {'G0722': 1.587,
        'G0726': 1.603,
        'G0732': 1.628,
        'G0743': 1.578,
        'G0737': 1.652,
        'G0702': 1.25,
        'G0703': 1.65,
        'G0712': 1.644,
        'G0709': 2.1239,
        'G0711': 2.1686,
        'G0713': 1.596,
        'Block': (-1),
        }
    l2 = fb[filter_blue]

    print "################Generating Cube_"
    infiles= _cube (science_lis, idir, odir, suffix, central, \
         flat_red, flat_blue, sky_red, sky_blue, gphu,dobadpix,pad, clobber)
     
    print "################Generating Cube_median crunch"
    _cube_medcrunch (pa,infiles,suffix,gphu, clobber)

    print "\n############################# Generating  _medfilter"
    medfiles = _medfilter (infiles,central,suffix,gphu, clobber)

    print "\n######################### Generating _medfilter_medcrunch_rotate"
    _medfilter_medcrunch_rotate (pa,medfiles, suffix,gphu, clobber)

    shift_files = _shift_medfilter (medfiles,central, suffix,gphu,l1,l2, clobber)

    print "\n############################# Generating _sdi"
    cube_sdi_file = _sdi (odir,suffix,gphu,pa,l1,l2, clobber)

    print "\n############################# Generating _loci_sdi"
    _loci_sdi (pa,cube_sdi_file, suffix,gphu, clobber)

    print "\n############################# Generating _loci"
    _loci_medfilter (pa,medfiles, suffix,gphu, clobber)

    print "\n############################# Generating _asdi"
    _asdi (pa,odir,suffix,gphu,l1,l2, clobber)

    return 

def _cube (science_lis, idir, odir, suffix, central, flat_red, \
         flat_blue, sky_red, sky_blue, gphu,dobadpix,pad, clobber):
    """
       Produces a cube_red.fits and cube_flat.fits with all the
       input science data. Also:
       medcrunch_(red,blue).fits: median(cube,axis=0)       # Cube is cube_(red,blue)
       sumcrum_(red,blue).fits:  sum(cube,axis=0)/sum(isfinite(cube))
    """

    if (not central):
       imsize = ([1024,1600][pad], [1024,1600][pad])
    else:
       imsize = (512,512)
    nfiles = len (science_lis)
    tmp = np.zeros((nfiles,)+imsize,dtype=np.float32)

    for ext in [1,2]:

        i = 0
        for f in science_lis:
       
            file = idir+f
            fits = pf.open(file,mode='update')
            fext = fits[ext]
            hdr = fext.header
            imc = fext.data

            # Verify that input data has been 'prepared'
            if not (fits[0].header).get("PREPARE"):
                print "File has not been ncprepared. -skipped: ",file
                continue

            # Make a local copy of the data
            im = imc.copy()
  
            # Array need to be float (required by geometric_transform)
            # and in nici_cntrd
            if 'float' not in str(im.dtype):
                im = np.asarray(im,dtype=np.float32)
	    if hdr.has_key('FILTER_R'): 
                linearzR = hdr['ITIME_R']*hdr['NCOADD_R']
                im = im/linearzR
                im = nici_noise(im)
		im = (im - sky_red) / flat_red
	    else:
                linearzB = hdr['ITIME_B']*hdr['NCOADD_B']
                im = im/linearzB
                im = nici_noise(im)
		im = (im - sky_blue) / flat_blue


            if dobadpix:               # Aggresive bad pixel removal
                im = badpix(im)

            if ext == 2:
		# Map blue image to the coordinates of red
		# Need to read 'database' and 'xy.coo' from some
		# default directory
		# iraf.gregister(f+'[2]','/tmp/gregister_blue.fits',\
                #           'database','xy.coo') 

                # we need to rotate by the rotation 'angle' as well.
                # Maybe add the 'angle' to angle in the database value

                im = np.asarray(im,dtype=np.float32)

                im = nd.geometric_transform (np.nan_to_num(im),\
                                      shift_function)
               

            update,xcen,ycen,im = nici_cntrd (im,hdr)

            print 'Cntrd shift (x,y):',\
                 f+'['+str(ext)+']%.2f,%.2f'%(512-xcen,512-ycen)
            if (update):
                fits.flush()

            # Stack on a cube
            if (not central):
                tmp[i,:,:] = im
            else:
                tmp[i,:,:] = im[512-256:512+256,512-256:512+256]
 
            i += 1
            fits.close()


        fname = odir + 'cube'+ suffix +'_'+(['red','blue'])[ext-1]+'.fits'
        print 'Writing :',fname
        pf.writeto(fname,tmp,header=gphu,clobber=clobber)

        fname = odir + 'sumcrunch'+ suffix +\
                '_'+ (['red','blue'])[ext-1]+'.fits'
        tdiv = np.sum(tmp,axis=0) / np.sum(np.isfinite(tmp),axis=0)
        pf.writeto(fname, np.asfarray(tdiv,dtype=np.float32), \
             header=gphu,clobber=clobber)

        fname = odir + 'medcrunch'+ suffix +\
                '_'+ (['red','blue'])[ext-1]+'.fits'
        pf.writeto(fname,np.median(tmp,axis=0),header=gphu,clobber=clobber)
        
    return [odir+'cube'+suffix+'_'+ext+'.fits' for ext in ('red','blue')]

def _cube_medcrunch(pa,cube,suffix,gphu, clobber):
    """
       Rotate each slice of the cube to a common angle and
       produces a cube_rotate_red.fits and cube_rotate_flat.fits
       Also medcrunch_rotate_(red,blue).fits: median(cube,axis=0)       # Cube is cube_(red,blue)
       sumcrum_rotate_(red,blue).fits:  sum(cube,axis=0)/sum(isfinite(cube))
    """
    c = cube[0]    # Full pathname of first file
    odir = c[:-len(c.split('/').pop())]     # Get the directory portion
    iter=0
    for file in cube:
        tmp = pf.getdata(file)
        sz = np.shape(tmp)
        
        for p in range(sz[0]):
            t1 = time.time()
            tmp[p,:,:] = nd.rotate (tmp[p,:,:], -pa[p], reshape=False) 
            t2 = time.time()
            line = 'rotating '+file+'['+str(p)+'/'+str(sz[0])+']'
            print line,'%.2f'%(-pa[p]),'elapse:%0.2f'%(t2-t1)

        chan=['red','blue'][iter]
        fname = odir+'cube_rotate'+ suffix + '_'+ chan + '.fits'
        pf.writeto(fname,tmp,header=gphu,clobber=clobber)

        fname = odir+'sumcrunch_rotate'+ suffix + '_'+ chan +'.fits'
        tdiv = np.sum(tmp,axis=0) / np.sum(np.isfinite(tmp),axis=0)
        pf.writeto(fname, np.asfarray(tdiv,dtype=np.float32), \
             header=gphu,clobber=clobber)

        fname = odir+'medcrunch_rotate'+ suffix + '_'+ chan +'.fits'
        pf.writeto(fname,np.median(tmp,axis=0),header=gphu,clobber=clobber)
        iter += 1
    return

def _medfilter(cube,central, suffix,gphu, clobber):
    """
       Median filtering of each slice in the cube
    """
    c = cube[0]
    odir = c[:-len(c.split('/').pop())]
    for file in cube:
        if 'blue' in file:
            chan = 'blue'
        else:
            chan = 'red'
        tmp = pf.getdata(file)
        sz= np.shape(tmp)
        if not central:
            r = dist_circle([1024,1024])
        else:
            r = dist_circle([512,512])
        rad = np.zeros(round(np.amax(r))-10)

        # to accelerate the code, we bin the image and the 'r' 
        # image down to 256x256 and do the radial profile on the binned versions

        rbin = rebin(r,1024/4,1024/4)
     
        X = np.arange(size(rad))
        line = 'subtracting median profile: '+file
        for i in range(sz[0]):
            imbin = rebin(tmp[i,:,:], 1024/4,1024/4)   # bin down to 256x256
            for rr in range(8,size(rad)):
                # determine the median profile beyond r=8 
                # (irrelevant inside this because of focal plane mask)
                if rr < 50:
                    rad[rr]=np.median( (tmp[i,:,:])[np.where( abs(r-rr) < .5)]) 
                else:
                    rad[rr]=np.median( imbin[np.where( abs(rbin-rr) < 3)]) 
            # smooth the radial profile beyond r=22
            rad[22:] = boxcar(rad,(5,))[22:] 
            rad[40:] = boxcar(rad,(5,))[40:] # smooth even more
            print line+'['+str(i)+'/'+str(sz[0])+'] ['+chan+']'

            #tmp[i,:,:] -= nicispline(X,rad,r)
            tmp[i,:,:] -= ginterpolate(rad,r)

            data = np.nan_to_num(tmp[i,:,:])
            tmp[i,:,:] -= scipy.signal.medfilt2d(data,11)

        fname = odir+'cube_medfilter'+ suffix + '_'+ chan +'.fits'
        pf.writeto(fname,tmp,header=gphu,clobber=clobber)

    return [odir+'cube_medfilter'+suffix+'_'+chan+'.fits' \
            for chan in ('red','blue')]

def _medfilter_medcrunch_rotate(pa,cube_medf, suffix,gphu, clobber):
    """
       Rotate each slice of the  cube_medfilter
    """

    c = cube_medf[0]
    odir = c[:-len(c.split('/').pop())]
    for file in cube_medf:
        if 'blue' in file:
            chan = 'blue'
        else:
            chan = 'red'
        tmp = pf.getdata(file)
        sz = np.shape(tmp)
        for i in range(sz[0]):
            print 'rotating ',file+'['+str(i)+'/'+str(sz[0])+']','%.2f'%-pa[i] 
            #      '-- here we process the cube that has been flat'+\
            #      ' fielded and high-passed'

            data = np.nan_to_num(tmp[i,:,:])
            tmp[i,:,:] = nd.rotate (data,-pa[i], reshape=False)
         
        fname = odir+'medfilter_medcrunch_rotate'+ suffix + '_'+ chan +'.fits'
        pf.writeto(fname,np.median(tmp,axis=0),header=gphu,clobber=clobber)
    return    

def _shift_medfilter(medfiles,central, suffix,gphu,l1,l2, clobber):
    """
      Scaled each sliced of the medfilter_blue file to the scale
      of the corresponding red.
    """ 
       
    c = medfiles[0]
    odir = c[:-len(c.split('/').pop())]
    for file in medfiles:
        if 'blue' in file:
            chan = 'blue'
            scale = min(l1,l2)/l2
        else:
            chan = 'red'
            scale = min(l1,l2)/l1

        tmp = pf.getdata(file)
        sz = np.shape(tmp)

        #scale:  this is a parameter that depends on 
        #        the exact set of filters used

        line = 'Shift_medfilter: '+ os.path.split(file)[1]
        if scale != 1.0:
            for i in range(sz[0]):
                t1 = time.time()
                data = np.asarray(tmp[i,:,:],dtype=np.float32)
                tmp[i,:,:]=data
                data=np.nan_to_num(data)
                if (central):
                    data = nd.geometric_transform(data,scale_512,extra_arguments=(scale,))   
                else:
                    data = nd.geometric_transform(data,scale_1024,extra_arguments=(scale,))   
                t2 = time.time()
                tmp[i,:,:]=data
                print line+'['+str(i)+'/'+str(sz[0])+'] elapse:%0.2f'%(t2-t1)

        fname = odir+'cube_shift_medfilter'+ suffix + '_'+ chan +'.fits'
        pf.writeto(fname,tmp,header=gphu,clobber=clobber)

    return [odir+'cube_shift_medfilter'+suffix+'_'+ext+'.fits'\
             for ext in ('red','blue')]

def _sdi(odir,suffix,gphu,pa,l1,l2, clobber):
    """
      Computes the difference of the red and blue cubes.
    """

    red_file = odir+'cube_shift_medfilter'+ suffix + '_red.fits'
    blue_file = odir+'cube_shift_medfilter'+ suffix + '_blue.fits'
    redcube = pf.getdata(red_file) 
    bluecube = pf.getdata(blue_file)

    sz = np.shape(bluecube)
         # as above, r is the distance to the center of the image
    r = dist_circle([sz[1],sz[2]])
         # find pixels between 25 and 50 pixels of the center of the image.
         #  These values should be modified when using a mask other than 0.32''
    g = np.where((r > 25) & (r < 50))


    for i in range(sz[0]):
        slice1 = redcube[i,:,:]
        slice2 = bluecube[i,:,:]
        bluecube[i,:,:] /= np.sum(slice1[g]*slice2[g])/np.sum(slice1[g]**2)
                            # we project (scalar product)
                            # the pixels of slice1 onto
                            # slice2 inside the annulus. The amplitude gives the
                            # optimal (in a least-square sense)
                            # amplitude of slice2 to fit
                            # slice1. The slice2 is scaled to this
                            # optimal' amplitude

    if l2 < l1:
        diff = bluecube - redcube
    else:
        diff = redcube - bluecube
   
    for i in range(sz[0]):
        print 'rotating diff cube ['+str(i+1)+'/'+str(sz[0])+']','%.2f'%-pa[i]
        tmp = nd.rotate (np.nan_to_num(diff[i,:,:]),-pa[i], reshape=False)
        diff[i,:,:] = tmp
    
    fname = odir + 'cube_sdi' + suffix + '.fits'
    print 'Writing: ','cube_shift_medfilter'+ suffix + '_red.fits (-)'+\
                     'cube_shift_medfilter'+ suffix + '_blue.fits'
    pf.writeto(fname,diff,header=gphu,clobber=clobber)

    fname = odir + 'sdi_medcrunch' + suffix + '.fits'
    pf.writeto(fname,np.median(diff,axis=0),header=gphu,clobber=clobber)
    print 'Writing the median of the above difference:','sdi_medcrunch_' +\
           suffix + '.fits'

    return odir+'cube_sdi'+suffix+'.fits'

def _loci_sdi(pa, cube_sdi_file, suffix,gphu, clobber):
    """
      Use the LOCI (Locally Optimized Combination of images) algorithm
      to reduce the speckle noise on the sdi_cube.
    """
    
    c = cube_sdi_file
    odir = c[:-len(c.split('/').pop())]
    ##############################
    # Lets do the ADI operation

    tmp = pf.getdata(cube_sdi_file)
    sz = np.shape(tmp)

    tmp = np.nan_to_num(tmp)
    tmp = loci_subtract(tmp, pa)

    for i in range(sz[0]):
        print 'rotating '+cube_sdi_file+'['+str(i+1)+'/'+str(sz[0])+']','%.2f'%-pa[i]
        tmp[i,:,:] = nd.rotate (tmp[i,:,:],-pa[i], reshape=False)

    fname = odir+'cube_loci_sdi' + suffix + '.fits'
    pf.writeto(fname,tmp,header=gphu,clobber=clobber)

    fname = odir+'loci_sdi_medcrunch' + suffix + '.fits'
    pf.writeto(fname,np.median(tmp,axis=0),header=gphu,clobber=clobber)

    fname = odir+'loci_sdi_sumcrunch' + suffix + '.fits'
    tdiv = np.sum(tmp,axis=0) / np.sum(np.isfinite(tmp),axis=0)
    pf.writeto(fname, np.asfarray(tdiv,dtype=np.float32), \
             header=gphu,clobber=clobber)

    return

def _loci_medfilter(pa,medfiles, suffix,gphu, clobber):
    """
      Apply the LOCI algorithm to the medfilter cube.
      files: 'medfilter_'+suffix+'_'+chan+'.fits'
    """

    c = medfiles[0]
    odir = c[:-len(c.split('/').pop())]
    ##############################
    # ADI_MEDFILTER
    for file in medfiles:
        if 'blue' in file:
            chan = 'blue'
        else:
            chan = 'red'

        tmp = pf.getdata(file)
        sz = np.shape(tmp)
        
        tmp = loci_subtract(tmp, pa)

        for i in range(sz[0]):
            print 'rotating ',file+'['+str(i+1)+'/'+str(sz[0])+']','%.2f'%-pa[i]
            tmp[i,:,:] = nd.rotate (np.nan_to_num(tmp[i,:,:]),-pa[i], reshape=False)

        fname = odir+'cube_loci_medfilter'+suffix +'_'+chan+'.fits'
        pf.writeto(fname,tmp,header=gphu,clobber=clobber)

        fname = odir+'loci_medfilter_medcrunch'+suffix+'_'+chan+'.fits'
        pf.writeto(fname,np.median(tmp,axis=0),header=gphu,clobber=clobber)

        fname = odir+'loci_medfilter_sumcrunch'+suffix+'_'+chan+'.fits'
        tdiv = np.sum(tmp,axis=0) / np.sum(np.isfinite(tmp),axis=0)
        pf.writeto(fname, np.asfarray(tdiv,dtype=np.float32), \
             header=gphu,clobber=clobber)

    return

def _asdi(pa,odir,suffix,gphu,l1,l2, clobber):
    """
      Combine sdi and adi method on the shift_medfiltered cubes
    """
    
    # ASDI Method
    if l1 < l2:
        red_file = odir+'cube_shift_medfilter'+suffix+'_red.fits'
        blue_file = odir+'cube_shift_medfilter'+suffix+'_blue.fits'
    else:
        red_file = odir+'cube_shift_medfilter'+suffix+'_blue.fits'
        blue_file = odir+'cube_shift_medfilter'+suffix+'_red.fits'

    cube_red = pf.getdata(red_file)
    cube_blue = pf.getdata(blue_file)

    cube_red = loci_subtract(cube_red, pa, cube_blue)

    file = odir+'cube_asdi' + suffix + '.fits'
    pf.writeto(file, cube_red,header=gphu,clobber=clobber)

    cube_red2 = cube_red.copy()
    sz = np.shape(cube_red2)
    line = 'rotating: shift_medfilter'+suffix+'(_red,_blue)'
    for i in range(sz[0]):
        print line+'['+str(i+1)+'/'+str(sz[0])+']'
        data = cube_red[i,:,:]
        data=np.nan_to_num(data)
        cube_red[i,:,:] = nd.rotate (data, -pa[i], reshape=False)
        data = cube_red2[i,:,:]
        data=np.nan_to_num(data)
        cube_red2[i,:,:] = nd.rotate (data, pa[i], reshape=False)


    fname = odir+'asdi_medcrunch' + suffix + '.fits'
    pf.writeto(fname, np.median(cube_red,axis=0),header=gphu,clobber=clobber)

    fname = odir+'asdi_counter_medcrunch' + suffix + '.fits'
    pf.writeto(fname, np.median(cube_red2,axis=0),header=gphu,clobber=clobber)

    #nici_counter .... # to be done

    print 'total angular range explored : '+str(max(pa)-min(pa))+ \
         '   ('+str(min(pa))+' to '+str(max(pa))+')'

    return 'Science calibration done....'
        
def shift_function(out):
      """
      This function is used as callback in ndimage.geometric_transform.
      The transformation below is taken from iraf.images.immatch.geomap
      'database' file.

     deg = 57.295779513082323
     xshift = 990.5897
     yshift = 37.82109
     xmag = 0.9986863
     ymag = 0.9993059
     xrot = 178.9731/deg
     yrot = 358.9603/deg

     sin(xrot)=
     sin(yrot)= 
     cos(xrot) =
     cos(yrot) =
               xout = a + b * xref + c * yref
               yout = d + e * xref + f * yref
               b = xmag * cos (xrot)
               c = -ymag * sin (yrot)
               e = xmag * sin (xrot)
               f = ymag * cos (yrot)
               a = xo - b * xref0 - c * yref0 = xshift
               d = yo - e * xref0 - f * yref0 = yshift
       b,c,e,f=(
              -0.9985259      -0.0178984
              -0.0181331      0.9991414
               )
                
      """
      xref = out[1] - 990.5897
      yref = out[0] - 37.82109
      x = -0.9985259*xref - 0.0178984*yref
      y = -0.0181331*xref + 0.9991414*yref
      
      return y,x

def scale_1024(out,scale):
    x = (out[1]-512.)/scale + 512.
    y = (out[0]-512.)/scale + 512.
    return y,x
def scale_1600(out,scale):
    x = (out[1]-800.)/scale + 800.
    y = (out[0]-800.)/scale + 800.
    return y,x

def scale_512(out, scale):
    """ 
    Callback function to be used in ndimage.geometric transform.
    The input image size is 512x512
    """
    x = (out[1]-256.)/scale + 256.
    y = (out[0]-256.)/scale + 256.
    return y,x
    
def _parangle_list (science_lis, idir, pa_file):
    """
      @science_lis: Input science list
      @pa_file: If not empty, filename to contains science filenames
                with PA angle.
    """
    # Get information from the PHU to calculate the Parallactic angle
    #
    radeg = 180/pi
    pa = []
    for fi in science_lis:
        fi = idir+fi
        hdr = pf.getheader(fi,1)
        #ha = dmstod(hdr['ha'])        # convert to decimal hours.
        #dec = hdr['dec']
        #crpa = hdr['crpa']
        #dd = -parangle (ha,dec,-dmstod('30:14:24')) - crpa
        cd11=hdr['cd1_1']
        cd12=hdr['cd1_2']
        cd21=hdr['cd2_1']
        cd22=hdr['cd2_2']
        dd = 180 - np.arctan2(sum([cd11,cd22]),sum([cd21,-cd12]))*radeg
        pa.append(dd)  


    if (pa_file != ''):
        # Open and rewrite
        fd = open(pa_file, mode= 'w')
        for k  in range(len(science_lis)):
            s = science_lis[k]+' '+str(pa[k])+'\n'
            fd.write(s)
        fd.close()

    return pa

def badpix(im):
    """
    Aggresive bad pixel removal
    """

    y,x = np.where(~np.isfinite(im))  # find bad pixels (NaNs)

    # we will not attempt to correct badpixels at the edge of the image
    g = np.where((x==0)|(y==0)|(x==1023)|(y==1023))
    y,x = remove(g[0], y, x)

    im_corr=im.copy()   # im_corr is the 'corrected' image

    for nbad in range(np.size(x)):
        neighbours = im[y[nbad]-1:y[nbad]+1,x[nbad]-1:x[nbad]+1]
                        # if 3 or more of the neighbouring
                        # pixels are NOT NaNs, we assume that
                        # the value for the NaN pixel is the
                        # mean of the non-NaNs. If not, leave
                        # it as a NaN
        if np.sum(np.isfinite(neighbours)) >= 3:
            mm = np.mean(neighbours[np.where(np.isfinite(neighbours))])
            im_corr[y[nbad],x[nbad]] = mm
                        # Just for the fun (yeah right) of it,
                        # print the fraction of NaNs before
                        # and after this correction
    print 'Fraction of bad pixels before correction : '+ '%6.3f' % \
       (np.sum(np.isfinite(im)==0)/(1024.*1024.)*100) +'%'
    print 'Fraction of bad pixels after correction  : '+ '%6.3f' % \
       (np.sum(np.isfinite(im_corr)==0)/(1024.*1024.)*100) +'%'
    return im_corr   # replace the image by the corrected one


def remove(idx,x,y):
   """
     Remove from vectors x and y 
     those elements indexed by idx.
   """
   
   nv = np.size(x)
   ni = np.size(idx)
   
   xx=np.zeros(np.size(x)-np.size(idx),dtype=np.int)
   yy=np.zeros(np.size(xx),dtype=np.int)
   j=0
   for i in range(ni-1):
       ii = idx[i] 
       ie = idx[i+1]

       for k in range(ii+1,ie):
           xx[j] = x[k]
           yy[j] = y[k]
           j+=1
   return xx,yy    


def ginterpolate (im, X, Y=None, output_type=None, order=1, 
                  mode='constant', cval=0.0, q=False) :
    """
      Emulate the IDL interface to the intepolate
      function using the ndimage.map_coordinates call
      
      @type im:   2d image buffer
      @type X:    2d index array with the x-location for which 
                  interpolation is desired. 
      @type Y:    2d index array with the y-location for which 
                  interpolation is desired. 
      @type order: Default:1. Order of interpolation. The default for
                   map_coordinates is 3
      @type q:     Default False. If True, there is only X and
                   we use only the upper left quarter of the symmetric X
                   (w/r to the array center). NOTE: Currently it is slower
                   to use this option. 

      For more information see help ndimage.map_coordinates

    """
    dim = np.size(np.shape(im))
    szx = np.size(np.shape(X))
    X = np.array(X)
    if Y != None: Y = np.array(Y)

    if dim == 1:
       if szx == 1:
           X = X.reshape([1,np.size(X)])
           z = nd.map_coordinates(im,X, output_type=output_type, order=order,
                     mode=mode, cval=cval)
       elif szx == 2:
           lim = np.size(im)
           limy,limx = lim,lim
           if Y == None: 
               Y = X
               coords = np.array([X,Y])
               if q:
                   nx,ny = np.shape(X)
                   nx,ny=nx/2,ny/2
                   coords = np.array([X[:ny,:nx],X[:ny,:nx]])
                   limy,limx = lim/2,lim
           else:
               coords = np.array([X,Y])
           z = nd.map_coordinates(np.resize(im,[limy,limx]),coords,
                     output_type=output_type, order=order, mode=mode, cval=cval)
    elif dim == 2:
       if szx == 1 and Y != None:
           z = nd.map_coordinates(im,[X,Y],output_type=output_type, order=order,
                     mode=mode, cval=cval)
       elif szx == 2:
           if Y == None: Y = X
           z = nd.map_coordinates(im,[X,Y],output_type=output_type, order=order,
                     mode=mode, cval=cval)
    else:
       print 'ERROR, dimension of input buffer not supported: ',len
       return 

    return z

    
def nicispline(x,y,t):
    """
    Interpolate a curve (x,y) at points xnew using a spline fit.
    """
    #try: b
    #except NameError:
    #    print 'b does not exits'
    # The idea is to calculate the splines coeff only
    # once for a given set of x,y. But how to tell
    # whether x,y have changed.?
    b,c,d=gspline.spline(x,y)

    z = np.zeros(np.shape(t))
    zz = np.ravel(z)
    tt = np.ravel(t)
    for j in xrange(np.size(t)):
        zz[j] = gspline.seval(x,y,b,c,d,tt[j])

    return z
    
import time
import os
import copy

def loci_subtract(cube,PA,more_images=None):
   """
	PA: in degrees. Paralactic angle  
 
        Locally Optimized Combination of Images algorithm.
        See Lafreniere et all (2007b) APJ
   """

   PA -= np.amin(PA)
   XtraImages = (more_images is not None)
   cube2 = cube.copy()             # cube2 will contain reference PSF as 
                            # built from the ADI technique
   radeg = 180./np.pi
   sz = np.shape(cube)
   Nslice = sz[0]         # Dimension order is (z,y,x)  
   Nslice_total=Nslice
   if XtraImages:
       Nslice_more_images=(np.shape(more_images))[0]
       PA = np.concatenate((PA,np.zeros(Nslice_more_images)))
       Nslice_total += Nslice_more_images

   Npix_included = 5        # The PSF must move by more than Npix_included 
	                    # to be included in the ADI matrix
                            
                            # Generates an x/y matrix with zero at the center
   # this can be done with meshgrid
   nx = sz[1]
   ny = sz[2]
   y = np.arange(nx*ny).reshape(ny,nx)/nx - nx/2. + 0.5
   x = np.rot90(y) 

   theta = np.arange(nx*ny,dtype=float).reshape(nx,ny)
   theta = np.arctan2(x,y)*radeg

   r = np.sqrt(x**2 + y**2)          # Distance from the center of PSF
   theta += 180                   # just to start at 0

   NRINGS = 11
   NSECTORS = 6
   RINGWD = 30
   for RR in range(NRINGS):          # loop on radial distance of arcs
       r1 = 8 + RR*RINGWD
       r2 = 8 + RINGWD + RR*RINGWD
       Ntheta = np.pi*(r2**2 - r1**2)/Nslice_total
       Ntheta = np.ceil(Ntheta/100.) # we want at least 100 times more pixels 
                                   # than degrees of freedom,
       Ntheta = min(Ntheta,6)      # but no more than 6 sections

       for TT in range(Ntheta):     # loop on angle position of the arcs
          
          theta1 = (360./Ntheta)*TT   # defining radius and angle boundaries 
        			      # of arcs
          theta2 = (360./Ntheta)*(TT+1)

                     # g is the index of pixel within the arc
          g = np.where((r>=r1) & (r<r2) & (theta>=theta1) & (theta<theta2)) 
          ng = np.size(g)/2    # is an (y,x) array of indexes (2)
                     # subsection is a 2D matrix containing in one axis 
                     # the pixels of the arc and second axis the i
                     # Nth slice of the cube
          if ng == 0: break
          subsection = np.empty((Nslice_total,ng),dtype=float)
                     # filling the subsection matrix
          for k in range(Nslice):
              subsection[k,:] = cube[k,:,:][g]
          if XtraImages:
              for i in range(Nslice_more_images):
                  subsection[i+Nslice,:] = more_images[i,:,:][g]

		      # defining a MM, the matrix containing the linear 
                      # projection of Nth slice versus Mth slice of the cube
          MM = np.mat(np.empty((Nslice_total,Nslice_total),dtype=float))
          for ii in range(Nslice_total):
                      # no need to loop from 0 as the matrix is symetrical
              for jj in range(ii,Nslice_total):
                  MM[ii,jj] = np.nansum(subsection[ii,:]*subsection[jj,:])
                  MM[jj,ii] = MM[ii,jj]    
                      
                      # loop on the slices of the cube --NOT the slice 
                      # of the cube with the slices added in more_images
          for nth in range(Nslice):
	      pdiff = abs((PA[nth]-PA)/radeg * (r1+r2)/2 )
                      # defining which slices are included in the 
                      # building of the reference PSF of the Nth slice
              included = np.where( (pdiff >= Npix_included) | (PA == -1))[0]  
                                               # [0] because where
                                               # will produce an array[1,:] 
                                               # rather than [:,]
	      Nincluded = np.size(included)
              if Nincluded == 0: 
                 print 'size included is zero for nth slice:',nth
                 continue
                      # (MM[included,Nth]) ;amplitude vector of the 
                      # projection of all included slices on the Nth slice
	      amps = (MM[:,included][included,:]).I *(MM[included,nth])
              amps=np.array(amps).flatten()

		      # a 2D image where the psf estimate we are
                      # building will be inserted with the pixel ordered
                      # back to their original position
              delta  = np.empty((nx,ny),dtype=float)
              cdel = delta[g].flatten()
	      for i in range(Nincluded):
	          cdel += amps[i]*subsection[included[i],:]
	          #delta[g] += amps[i]*subsection[included[i],:]
              delta[g] = cdel
              
              cube2[nth,:,:] -= delta
       print 'adi subtraction R=',r1,'to',r2,\
                   'angle=',theta1,'to',theta2

   #-- this's not really ADI ---
   med = np.median(cube,axis=0)   # Median over all the slices
   far = np.where(r >= r2) # in the region outside of the largest arc, 
                   # we put in the cube the median of all slices
   for i in range(Nslice):
       tmp = copy.deepcopy(cube2[i,:,:])
       tmp[far] -= med[far]
       cube2[i,:,:] = tmp

   #----------------------------
   cube = cube2
   return cube

if __name__ == '__main__':
    # Parse input arguments
    usage = 'usage: %prog science_list [options]'

    p = optparse.OptionParser(usage=usage, version='ncscience.0')
    p.add_option('--idir', action='store', type='string',
                 default='', help='Input directory pathname')
    p.add_option('--odir', action='store', type='string',
                 default='', help='Output directory pathname')
    p.add_option('--fdir', action='store', type='string',
                 default='', help='Flats directory pathname')
    p.add_option('--fsuffix', action='store', type='string',
                 default='default', help='Calibration files rootname')
    p.add_option('--central','-c', action='store_true',default=False,
                 help='Choose 512x512 output images')
    p.add_option('--suffix', action='store', type='string',
                 default='default', help='suffix')
    p.add_option('--dobadpix', default=True, action='store_true',
                 help='Correct bad pixels the best we can')
    p.add_option('--pad', default=False, action='store_true',
                 help='Use an extended area for interpolation')
    p.add_option('--clobber', default=False, action='store_true',
                 help='Clobber output files?')
    p.add_option('--logfile', action='store', type='string', default='',
                 help='logfile name ')
    p.add_option('--debug','-d', action='store_true',default=False,
                 help='toggle debug messages')

    (par, args) = p.parse_args()

    if par.debug:
        print 'options: ', par
        print 'args: ', args

    iList= args
    if len(iList) == 0:
       print 'options: ', par
       p.print_help()
       sys.exit(0)
    # Generate an input list from idir plus an @ file or a unix template
    #
    science_lis = getFileList(iList)
    ncscience (science_lis, idir=par.idir,odir=par.odir,
                  fdir=par.fdir, fsuffix=par.fsuffix, central=par.central,
                  dobadpix=par.dobadpix, pad=par.pad,
                  suffix=par.suffix, clobber=par.clobber, 
                  logfile=par.logfile)

