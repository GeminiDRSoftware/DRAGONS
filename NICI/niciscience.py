#!/usr/bin/env python
#
import optparse
import sys
from niciTools import getFileList,parangle,dmstod

import pyfits as pf
import ndimage as nd
from niciCntrd import nici_cntrd
import scipy.signal
import scipy.interpolate as scint
import os
from dist_circle import dist_circle
import numpy as np
from numpy import pi
from numpy import size
#from adisub import loci_subtract
#import adisub
import gspline as gspline
import time

def niciscience (science_lis,inputdir='',outputdir='',\
                  flatsdir='',central=False, savePA=True,\
                  dataSetName='Object', clobber=True, 
                  logfile='', verbose=True):
    """
        #

        science_lis =             Input Fits files on a list
        (inputdir = '')           Path for input raw files
        (outputdir = '')          Output directory name
        (central = False)         Reduce the output image to 512x512
        (savePA = True)           Save the Paralactic angles in a file? 
        (dataSetName = 'Object')  Dataset name
        (clobber  = True)         Replace output file if exits
        (logfile = '')            Logfile
        (verbose = True)          Verbose

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
           --> niciscience *.fits outputdir=/tmp

        Python_shell:
           - Your UNIX PYTHONPATH shell variable should have an
             entry for the directory 'nici', like: /home/myname/nici
           >>> import niciscience as nicis
           >>> # Make the input list, assuming you are in the data-directory
           >>> import glob
           >>> flis=glob.glob('*.fits')
           >>> nicis.niciscience(flis,inputdir='/data1/data/tmp',\
                outputdir='/tmp'i,flats='/data/nici/200901/flats')

        Unix_shell:
           % cd /home/myname/nici
           % # to get help try: niciscience.py -h
           % niciscience.py /data/nici/tmp/*.fits --odir='/tmp'




     Reduce Science frames given as list of fits files.
     science_lis: filenames in a python list
    """

    dataname = dataSetName
    if dataname[0] != '_':
       dataname = '_' + dataname

    if (inputdir!= '' and inputdir[-1] != '/'):
       inputdir += '/'
    if (outputdir!= '' and outputdir[-1] != '/'):
       outputdir += '/'
    if (outputdir == ''):
        outputdir = './' 
    if (not os.access(outputdir,os.W_OK)):
       print "Warning: Cannot write into",outputdir
       return
    if (flatsdir == ''):
        print '\n     ERROR: Please enter a directory name for the flats.'
        return
    if (flatsdir!= '' and flatsdir[-1] != '/'):
       flatsdir += '/'

    if (not central):
       imsize = (1024,1024)
    else:
       imsize = (512,512)
    nfiles = len (science_lis)
    if (nfiles == 0):
        print 'ERROR: No input files selected.'
        return
    
#def _calib(inlis):

    try:
	#flat_red  = pf.getdata('etienne/flat_red_EpsEridani.fits')
	#flat_blue = pf.getdata('etienne/flat_blue_EpsEridani.fits')
	#sky_red   = pf.getdata('etienne/sky_red_EpsEridani.fits')
	#sky_blue  = pf.getdata('etienne/sky_blue_EpsEridani.fits')
	flat_red  = pf.getdata(flatsdir + 'flat_red.fits')
	flat_blue = pf.getdata(flatsdir + 'flat_blue.fits')
	sky_red   = pf.getdata(flatsdir + 'sky_red.fits')
	sky_blue  = pf.getdata(flatsdir + 'sky_blue.fits')
    except IOError:
        print 'ERROR opening flat_red.fits,flat_blue.fits,'
        print '      sky_red.fits or sky_blue.fits in dir:',flatsdir
        return

    
    # Calculate the paralactic angle and save the result in a file
    # if needed.
    pafile = ''
    if (savePA):
        pafile = dataname+'_pa.txt'

    # Get the PHU from the first element of the input list. It will be
    # used as a PHU for the output products

    gphu = pf.getheader(inputdir+science_lis[0],0)

    pa = _parangle_list (science_lis, inputdir, pafile)

    print "################Generating Cube_"
    infiles= _cube (science_lis, inputdir, outputdir, dataname, central, \
         flat_red, flat_blue, sky_red, sky_blue, gphu)

    print "################Generating Cube_median crunch"
    _cube_medcrunch (pa,infiles,dataname,gphu)

    print "\n############################# Generating  _medfilter"
    medfiles = _medfilter (infiles,central,dataname,gphu)

    print "\n########################### Generating _medfilter_medcrunch_rotate"
    _medfilter_medcrunch_rotate (pa,medfiles, dataname,gphu)

    print "\n############################# Generating _shift_medfilter"
    shift_files = _shift_medfilter (medfiles,central, dataname,gphu)

    print "\n############################# Generating _sdi"
    sdi_files = _sdi (outputdir,dataname,gphu)

    print "\n############################# Generating _loci_sdi"
    _loci_sdi (pa,sdi_files, dataname,gphu)

    print "\n############################# Generating _loci"
    _loci_medfilter (pa,medfiles, dataname,gphu)

    print "\n############################# Generating _asdi"
    _asdi (pa,outputdir,dataname,gphu)

    return 

def _cube (science_lis, inputdir, outputdir, dataname, central, flat_red, \
         flat_blue, sky_red, sky_blue, gphu):

    if (not central):
       imsize = (1024,1024)
    else:
       imsize = (512,512)
    nfiles = len (science_lis)
    tmp = np.zeros((nfiles,)+imsize,dtype=np.float32)

    for ext in [1,2]:

        i = 0
        for f in science_lis:
       
            file = inputdir+f
            fits = pf.open(file,mode='update')
            fext = fits[ext]
            hdr = fext.header
            imc = fext.data
            # Make a local copy of the data
            im = imc.copy()
  
            # Array need to be float (required by geometric_transform)
            # and in nici_cntrd
            if 'float' not in str(im.dtype):
                im = np.asarray(im,dtype=np.float32)

	    if hdr.has_key('FILTER_R'): 
                linearzR = hdr['ITIME_R']*hdr['NCOADD_R']*hdr['NDR_R']
		im = (im - sky_red) / flat_red / linearzR
	    else:
                linearzB = hdr['ITIME_B']*hdr['NCOADD_B']*hdr['NDR_B']
		im = (im - sky_blue) / flat_blue / linearzB

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
               

            # we need to rotate when ext==1, maybe we can rotate in
            # nici_cntrd
            # PUT pa[]
        
            # Shift to the center
            medval = np.median(im[where(np.isfinite(im))],axis=None)
            im=np.nan_to_num(im)

            update,xcen,ycen,im = nici_cntrd (im,hdr)

            print 'Cntrd shift (x,y):',\
                 f+'['+str(ext)+']%.2f,%.2f'%(512-xcen,512-ycen)
            if (update):
                fits.flush()

            # Substract median
            
            im -= medval

            # Stack on a cube
            if (not central):
                tmp[i,:,:] = im
            else:
                tmp[i,:,:] = im[512-256:512+256,512-256:512+256]
 
            i += 1
            fits.close()


        file = outputdir + 'cube'+ dataname +'_'+(['red','blue'])[ext-1]+'.fits'
        print 'Writing :',file
        pf.writeto(file,tmp,header=gphu,clobber=True)

        file = outputdir + 'sumcrunch'+ dataname +\
                '_'+ (['red','blue'])[ext-1]+'.fits'
        tdiv = np.sum(tmp,axis=0) / np.sum(np.isfinite(tmp),axis=0)
        pf.writeto(file, np.asfarray(tdiv,dtype=np.float32), \
             header=gphu,clobber=True)

        file = outputdir + 'medcrunch'+ dataname +\
                '_'+ (['red','blue'])[ext-1]+'.fits'
        pf.writeto(file,np.median(tmp,axis=0),header=gphu,clobber=True)
        
    return [outputdir+'cube'+dataname+'_'+ext+'.fits' \
                for ext in ('red','blue')]

def _cube_medcrunch(pa,cube,dataname,gphu):
    # ROTATE here for now
    c = cube[0]
    outputdir = c[:-len(c.split('/').pop())]
    iter=0
    for file in cube:
    #for chan in ['red','blue']:
        #file = 'cube'+ dataname + '_'+ chan + '.fits'
        #tmp = pf.getdata(outputdir+file)
        tmp = pf.getdata(file)
        sz = np.shape(tmp)
        
        for p in range(sz[0]):
            t1 = time.time()
            tmp[p,:,:] = nd.rotate (tmp[p,:,:], -pa[p], reshape=False) 
            t2 = time.time()
            line = 'rotating '+file+'['+str(p)+'/'+str(sz[0])+']'
            print line,'%.2f'%(-pa[p]),'elapse:%0.2f'%(t2-t1)

        chan=['red','blue'][iter]
        file = outputdir+'cube_rotate'+ dataname + '_'+ chan + '.fits'
        pf.writeto(file,tmp,header=gphu,clobber=True)

        file = outputdir+'sumcrunch_rotate'+ dataname + '_'+ chan +'.fits'
        tdiv = np.sum(tmp,axis=0) / np.sum(np.isfinite(tmp),axis=0)
        pf.writeto(file, np.asfarray(tdiv,dtype=np.float32), \
             header=gphu,clobber=True)

        file = outputdir+'medcrunch_rotate'+ dataname + '_'+ chan +'.fits'
        pf.writeto(file,np.median(tmp,axis=0),header=gphu,clobber=True)
        iter += 1
    return

def _medfilter(cube,central, dataname,gphu):

    c = cube[0]
    outputdir = c[:-len(c.split('/').pop())]
    # file needs to have 'blue' or 'red' in string
    for file in cube:
    #for chan in ['red','blue']:
        if 'blue' in file:
            chan = 'blue'
        else:
            chan = 'red'
        #file = 'cube'+ dataname + '_'+ chan + '.fits'
        #tmp = pf.getdata(outputdir+file)
        tmp = pf.getdata(file)
        sz= np.shape(tmp)
        if not central:
            r = dist_circle([1024,1024])
        else:
            r = dist_circle([512,512])
        rad = np.zeros(round(np.amax(r)))
     
        X = np.arange(size(rad))
        line = 'subtracting median profile: '+file
        for i in range(sz[0]):
            for rr in range(8,200):
                rad[rr]=np.median( (tmp[i,:,:])[np.where( abs(r-rr) < .5)]) 
            rad[0:8]=rad[8]
            rad[201:] = np.median((tmp[i,:,:])[np.where( r > 200)])
            print line+'['+str(i)+'/'+str(sz[0])+'] ['+chan+']'
            #tmp[i,:,:] -= scint.spline(X,rad,r)
            # Need to test spline against the results from 
            # the IDL interpolate(rad,r,cubic=-.6)

            tmp[i,:,:] -= nicispline(X,rad,r)
            #tmp[i,:,:] = np.nan_to_num(tmp[i,:,:])

            data = np.nan_to_num(tmp[i,:,:])
            tmp[i,:,:] -= scipy.signal.medfilt2d(data,11)

        file = outputdir+'cube_medfilter'+ dataname + '_'+ chan +'.fits'
        pf.writeto(file,tmp,header=gphu,clobber=True)

    return [outputdir+'cube_medfilter'+dataname+'_'+chan+'.fits' \
            for chan in ('red','blue')]

def _medfilter_medcrunch_rotate(pa,files, dataname,gphu):

    c = files[0]
    outputdir = c[:-len(c.split('/').pop())]
    datar=np.zeros([10,512,512])
    for file in files:
        if 'blue' in file:
            chan = 'blue'
        else:
            chan = 'red'
        # Rotate the file created above
        #file = cube_'medfilter_'+ dataname + '_'+ chan +'.fits' 
        tmp = pf.getdata(file)
        sz = np.shape(tmp)
        for i in range(sz[0]):
            print 'rotating ',file+'['+str(i)+'/'+str(sz[0])+']','%.2f'%-pa[i] 
            #      '-- here we process the cube that has been flat'+\
            #      ' fielded and high-passed'
            #data = tmp[i,:,:]
            #data = np.asarray(data,dtype=np.float32)

            data = np.nan_to_num(tmp[i,:,:])
            tmp[i,:,:] = nd.rotate (data,-pa[i], reshape=False)
         
        out = outputdir+'medfilter_medcrunch_rotate'+ dataname + '_'+ chan +'.fits'
        pf.writeto(out,np.median(tmp,axis=0),header=gphu,clobber=True)
    return    

def _shift_medfilter(files,central, dataname,gphu):

    c = files[0]
    outputdir = c[:-len(c.split('/').pop())]
    for file in files:
        if 'blue' in file:
            chan = 'blue'
        else:
            chan = 'red'

        root = file.split('/').pop()
        # Shift_medfilter the file created above
        #file = 'medfilter_'+ dataname + '_'+ chan + '.fits'
        #tmp = pf.getdata(outputdir+file)
        tmp = pf.getdata(file)
        sz = np.shape(tmp)

        scale = 1/1.043    # this is a parameter that depends on 
                         # the exact set of filters used
        amp = 1/0.81       # this depends on the filters used, 
                         # the channels and the spectral type of 
                         # the star observed
        angle=0          # should be zero if we did a good job with 
                         # the distortion grid
        line = 'Shift_medfilter: '+ root
        for i in range(sz[0]):
            #if chan == 'blue':
            if 'blue' in file:
                t1 = time.time()
                data = np.asarray(tmp[i,:,:],dtype=np.float32)
                tmp[i,:,:]=data
                data=np.nan_to_num(data)
                if (central):
                    data = nd.geometric_transform (data,rot_scale_512)   
                else:
                    data = nd.geometric_transform (data,rot_scale_1024)   
                data = data*amp
                t2 = time.time()
                tmp[i,:,:]=data
                print line+'['+str(i)+'/'+str(sz[0])+'] elapse:%0.2f'%(t2-t1)

        file = outputdir+'cube_shift_medfilter'+ dataname + '_'+ chan +'.fits'
        pf.writeto(file,tmp,header=gphu,clobber=True)

    return [outputdir+'cube_shift_medfilter'+dataname+'_'+ext+'.fits'\
             for ext in ('red','blue')]

def _sdi(outputdir,dataname,gphu):

    # lets write a single 'cube' after SDI'

    red_file = outputdir+'cube_shift_medfilter'+ dataname + '_red.fits'
    blue_file = outputdir+'cube_shift_medfilter'+ dataname + '_blue.fits'
    diff = pf.getdata(red_file) - pf.getdata(blue_file)

    file = outputdir + 'cube_sdi' + dataname + '.fits'
    print 'Writing: ','cube_shift_medfilter'+ dataname + '_red.fits (-)'+\
                     'cube_shift_medfilter'+ dataname + '_blue.fits'
    pf.writeto(file,diff,header=gphu,clobber=True)

    file = outputdir + 'sdi_medcrunch' + dataname + '.fits'
    pf.writeto(file,np.median(diff,axis=0),header=gphu,clobber=True)
    print 'Writing the median of the above difference:','sdi_medcrunch_' +\
           dataname + '.fits'

    return outputdir+'cube_sdi'+dataname+'.fits'

def _loci_sdi(pa,file, dataname,gphu):
    
    c = file
    outputdir = c[:-len(c.split('/').pop())]
    ##############################
    # Lets do the ADI operation

    #file = 'sdi_' + dataname + '.fits'
    tmp = pf.getdata(file)
    sz = np.shape(tmp)

    tmp=np.nan_to_num(tmp)
    tmp = loci_subtract(tmp, pa)

    for i in range(sz[0]):
        print 'rotating '+file+'['+str(i+1)+'/'+str(sz[0])+']','%.2f'%-pa[i]
        tmp[i,:,:] = nd.rotate (tmp[i,:,:],-pa[i], reshape=False)

    file = outputdir+'cube_loci_sdi' + dataname + '.fits'
    pf.writeto(file,tmp,header=gphu,clobber=True)

    file = outputdir+'loci_sdi_medcrunch' + dataname + '.fits'
    pf.writeto(file,np.median(tmp,axis=0),header=gphu,clobber=True)

    file = outputdir+'loci_sdi_sumcrunch' + dataname + '.fits'
    tdiv = np.sum(tmp,axis=0) / np.sum(np.isfinite(tmp),axis=0)
    pf.writeto(file, np.asfarray(tdiv,dtype=np.float32), \
             header=gphu,clobber=True)

    return

def _loci_medfilter(pa,files, dataname,gphu):

    c = files[0]
    outputdir = c[:-len(c.split('/').pop())]
    ##############################
    # ADI_MEDFILTER
    for file in files:
        if 'blue' in file:
            chan = 'blue'
        else:
            chan = 'red'

        #file = 'medfilter_' + dataname +'_'+ chan + '.fits'
        #tmp = pf.getdata(outputdir+file)
        tmp = pf.getdata(file)
        sz = np.shape(tmp)
        
        #tmp=np.nan_to_num(tmp)
        tmp = loci_subtract(tmp, pa)

        for i in range(sz[0]):
            print 'rotating ',file+'['+str(i+1)+'/'+str(sz[0])+']','%.2f'%-pa[i]
            tmp[i,:,:] = nd.rotate (np.nan_to_num(tmp[i,:,:]),-pa[i], reshape=False)

        file = outputdir+'cube_loci_medfilter'+dataname +'_'+chan+'.fits'
        pf.writeto(file,tmp,header=gphu,clobber=True)

        file = outputdir+'loci_medfilter_medcrunch'+dataname+'_'+chan+'.fits'
        pf.writeto(file,np.median(tmp,axis=0),header=gphu,clobber=True)

        file = outputdir+'loci_medfilter_sumcrunch'+dataname+'_'+chan+'.fits'
        tdiv = np.sum(tmp,axis=0) / np.sum(np.isfinite(tmp),axis=0)
        pf.writeto(file, np.asfarray(tdiv,dtype=np.float32), \
             header=gphu,clobber=True)

    return

def _asdi(pa,outputdir,dataname,gphu):
    
    # ASDI Method
    red_file = outputdir+'cube_shift_medfilter'+dataname+'_red.fits'
    blue_file = outputdir+'cube_shift_medfilter'+dataname+'_blue.fits'
    cube_red = pf.getdata(red_file)
    cube_blue = pf.getdata(blue_file)

    cube_red = loci_subtract(cube_red, pa, cube_blue)

    cube_red2 = cube_red.copy()
    sz = np.shape(cube_red2)
    line = 'rotating: shift_medfilter'+dataname+'(_red,_blue)'
    for i in range(sz[0]):
        print line+'['+str(i+1)+'/'+str(sz[0])+']'
        data = cube_red[i,:,:]
        data=np.nan_to_num(data)
        cube_red[i,:,:] = nd.rotate (data, -pa[i], reshape=False)
        data = cube_red2[i,:,:]
        data=np.nan_to_num(data)
        cube_red2[i,:,:] = nd.rotate (data, pa[i], reshape=False)

    file = outputdir+'cube_asdi' + dataname + '.fits'
    pf.writeto(file, cube_red,header=gphu,clobber=True)

    file = outputdir+'asdi_medcrunch' + dataname + '.fits'
    pf.writeto(file, np.median(cube_red,axis=0),header=gphu,clobber=True)

    file = outputdir+'asdi_counter_medcrunch' + dataname + '.fits'
    pf.writeto(file, np.median(cube_red2,axis=0),header=gphu,clobber=True)

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

def rot_scale_512(out):
    """ 
    Callback function to be used in ndimage.geometric transform.
    The input image size is 512x512
    """
    x = (out[1]-256.)/1.043 + 256.
    y = (out[0]-256.)/1.043 + 256.
    return y,x

def rot_scale_1024(out):
    """
    For an input image size of 1024x1024.
    Callback function to be used in ndimage.geometric transform
    for rotate, scale and magnify.
    See the doc for the shift_function above.

    xrot = 0.0
    yrot = 0.0
    xmag = 1/1.043  #this is a parameter that depends on the exact 
                    #set of filters used
    ymag = 1/1.043
    """ 

    #x = (1.043*out[1]-512) + 512
    #y = (1.043*out[0]-512) + 512
    x = (out[1]-512.)/1.043 + 512.
    y = (out[0]-512.)/1.043 + 512.

    return y,x

def parangle(ha,dec,lat):
    """
    Return the parallactic angle of a source in degrees.

    HA - the hour angle of the source in decimal hours; a scalar or vector.
    DEC - the declination of the source in decimal degrees; a scalar or
            vector.
    LAT - The latitude of the telescope; a scalar.
    """
    from numpy import pi,sin,cos,tan 

    d2r = pi/180.
    r2d = 180./pi
    had = 15.0*ha

    ac2 = np.arctan2( -sin(d2r*had), \
                 cos(d2r*dec)*tan(d2r*lat)-sin(d2r*dec)*cos(d2r*had))

    return -r2d*ac2

    
def _parangle_list (science_lis, inputdir, pa_file):
    """
      @science_lis: Input science list
      @pa_file: If not empty, filename to contains science filenames
                with PA angle.
    """
    # Get information from the PHU to calculate the Parallactic angle
    #
    pa = []
    for fi in science_lis:
        fi = inputdir+fi
        hdr = pf.getheader(fi)
        if (hdr.get('PREPARE') == None):
            print 'Science file: ',fi,' has not been Prepare.'
            return
        ha = dmstod(hdr['ha'])        # convert to decimal hours.
        #dec = dmstod(hdr['dec'])
        dec = hdr['dec']
        crpa = hdr['crpa']
        dd = -parangle (ha,dec,-dmstod('30:14:24')) - crpa
        pa.append(dd)  


    if (pa_file != ''):
        # Open and rewrite
        fd = open(pa_file, mode= 'w')
        for k  in range(len(science_lis)):
            s = science_lis[k]+' '+str(pa[k])+'\n'
            fd.write(s)
        fd.close()

    return pa

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
from numpy import empty,where,size,concatenate,arange,ceil,nansum

def loci_subtract(cube,PA,more_images=None):
   """
	PA: in degrees  (0..60) for testing
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
                            
              # Generates an x/y matrix with its center at 511/511
   #y = arange(1024*1024).reshape(1024,1024)/1024-511
   nx = sz[1]
   ny = sz[2]
   y = np.arange(nx*ny).reshape(ny,nx)/nx - (nx/2 -1)
   x = np.rot90(y) 
   #gen_matrix(1024,1024,x,y,/center)

   theta = np.arange(nx*ny,dtype=float).reshape(nx,ny)
   theta = np.arctan2(x,y)*radeg

   r = np.sqrt(x**2 + y**2)          # Distance from the center of PSF
   theta += 180                   # just to start at 0

   NRINGS = 11
   NSECTORS = 6
   for RR in range(NRINGS):          # loop on radial distance of arcs
       #r1 = 20 + RR*20
       #r2 = 40 + RR*20
       r1 = 8 + RR*30
       r2 = 8+30 + RR*30
       Ntheta = np.pi*(r2**2 - r1**2)/Nslice_total
       Ntheta = ceil(Ntheta/100.) # we want at least 100 times more pixels 
                                   # than degrees of freedom,
       Ntheta = min(Ntheta,6)      # but no more than 6 sections
       t1 = time.time()
       for TT in range(Ntheta):     # loop on angle position of the arcs
          
          theta1 = (360./Ntheta)*TT   # defining radius and angle boundaries 
        			      # of arcs
          theta2 = (360./Ntheta)*(TT+1)

                     # g is the index of pixel within the arc
          g = where((r>=r1) & (r<r2) & (theta>=theta1) & (theta<theta2)) 
          ng = size(g)/2    # is an (y,x) array of indexes (2)
                     # subsection is a 2D matrix containing in one axis 
                     # the pixels of the arc and second axis the i
                     # Nth slice of the cube
          if ng == 0: break
          subsection = empty((Nslice_total,ng),dtype=float)
                     # filling the subsection matrix
          for k in range(Nslice):
              subsection[k,:] = cube[k,:,:][g]
          if XtraImages:
              for i in range(Nslice_more_images):
                  subsection[i+Nslice,:]=more_images[i,:,:][g]

		      # defining a MM, the matrix containing the linear 
                      # projection of Nth slice versus Mth slice of the cube
          MM = np.mat(empty((Nslice_total,Nslice_total),dtype=float))
          for ii in range(Nslice_total):
                      # no need to loop from 0 as the matrix is symetrical
              for jj in range(ii,Nslice_total):
                  MM[ii,jj] = nansum(subsection[ii,:]*subsection[jj,:])
                  MM[jj,ii] = MM[ii,jj]    
                      
                      # loop on the slices of the cube --NOT the slice 
                      # of the cube with the slices added in more_images
          for nth in range(Nslice):
	      pdiff = abs((PA[nth]-PA)/radeg * (r1+r2)/2 )
                      # defining which slices are included in the 
                      # building of the reference PSF of the Nth slice
              included = where( (pdiff >= Npix_included) | (PA == -1))[0]  
                                               # [0] because where
                                               # will produce an array[1,:] 
                                               # rather than [:,]
	      Nincluded = size(included)
              if Nincluded == 0: 
                 print 'size included is zero for nth slice:',nth
                 continue
                      # (MM[included,Nth]) ;amplitude vector of the 
                      # projection of all included slices on the Nth slice
	      amps = (MM[:,included][included,:]).I *(MM[included,nth])
              amps=np.array(amps).flatten()

		      # a 2D image where the psf estimate we are
                      # building will be insterted with the pixel ordered
                      # back to their original position
              delta  = empty((nx,ny),dtype=float)
              cdel = delta[g].flatten()
	      for i in range(Nincluded):
	          cdel += amps[i]*subsection[included[i],:]
	          #delta[g] += amps[i]*subsection[included[i],:]
              delta[g] = cdel
              cube2[nth,:,:] -= delta
       t2 = time.time()
       print "%d %0.3f" % (RR,t2-t1)
             

   #-- this's not really ADI ---
   med = np.median(cube,axis=0)   # Median over all the slices
   far = where(r >= r2) # in the region outside of the largest arc, 
                   # we put in the cube the median of all slices
   for i in range(Nslice):
       tmp = cube2[i,:,:]
       tmp[far] -= med[far]
       cube2[i,:,:] = tmp

   #----------------------------
   cube = cube2
   return cube

if __name__ == '__main__':
    # Parse input arguments
    usage = 'usage: %prog science_list [options]'

    p = optparse.OptionParser(usage=usage, version='niciPrepare_1.0')
    p.add_option('--idir', action='store', type='string',
                 default='', help='Input directory pathname')
    p.add_option('--odir', action='store', type='string',
                 default='', help='Output directory pathname')
    p.add_option('--fdir', action='store', type='string',
                 default='', help='Flats directory pathname')
    p.add_option('--central','-c', action='store_true',default=False,
                 help='Choose 512x512 output images')
    p.add_option('--savePA', action='store_true',default=False,
                help='Save paralactic angles in file odir/<dataSetname>_pa.txt')
    p.add_option('--dname', action='store', type='string',
                 default='Object', help='Data set name')
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
       sys.exit(0)
    # Generate an input list from inputdir plus an @ file or a unix template
    #
    science_lis = getFileList(iList)
    niciscience (science_lis, inputdir=par.idir,outputdir=par.odir,\
                  flatsdir=par.fdir, central=par.central, savePA=par.savePA,\
                  dataSetName=par.dname, clobber=par.clobber, 
                  logfile=par.logfile, verbose=par.v)

#def nici_science (science_lis,inputdir='',outputdir='',\
#                  flatsdir='',central=False, savePA=True,\
#                  dataSetName='Object', clobber=True, 
#                  logfile='', verbose=True):
