#!/usr/bin/env python

import numpy as np
import datetime
import glob
import pyfits as pf
from robustsigma import robust_sigma
import gemfit as gf
import ndimage as nd
import math
import niciTools as nt
import pylab as py
import numdisplay as ndis
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import figure, mlab, cm
from niciTools import getFileList
import os

import time

def nici_das(date=None,help=None,log=True,lists=True,saturate=None,display=True):

    if ~help:
        help = None
    if (help != None):
        print 'calling line : '
        print 'nici_das   ---- parses today''s data'
        print 'nici_das date=''20081116'',saturate=2000 display=False  \
             ---- parses the 20081116 data with a saturation limit at 2000 \
                  and does not display (for speed)'
        print 'keyword saturate. If not present, saturation >3500 \
             otherwise give an input value'
        return

    if date == '':
       date = None
    if (date == None):
        today = datetime.date.today()
        dd = today.timetuple()
        date = str(dd[0]) + '%.2d'%dd[1] + '%.2d'%dd[2]


    path = '/net/petrohue/dataflow/'
    #path = '/data2/tmp/'
    pathS = path+'S'
    if (saturate == None): saturate=3500

    bad_b = pf.getdata('/home/nzarate/nici/cube/bad_b.fits')
            #loads the badpixels list for the blue and red channel
    bad_r = pf.getdata('/home/nzarate/nici/cube/bad_r.fits')

    # Chech that we can write in the current directory
    # the output FITS file is written here.
    if not os.access('./',os.W_OK):
        print "\nERROR: You do not have write access to \
                this directory (output FITS file)\n"
        return

    date = str(date)
    if ('@' in date):
        fits_files = getFileList(date)
    else:
        print 'Looking for files' +pathS+date+'S*.fits'
        fits_files = glob.glob(pathS+date+'S*.fits')
            #finds the list of science frames
    fits_files = np.sort(fits_files)
    if len(fits_files)==0:
        print 'couldn''t find files for :'+pathS \
              +date+'S*.fits'
        return

    print 'found '+str(len(fits_files))+\
          ' matching the following condition :'+ \
          pathS+date+'S*.fits'

    previous_angle = -999 #define a few dummy variables
    previous_time = -999
    previous_ra   = -999
    previous_dec  = -999
    previous_mode = 'NONE'

    #w,h=np.shape(out)
    dpi = py.rcParams['figure.dpi']
    W,H=(256./dpi,512./dpi)
    if (display):
        py.axis('off')
    xtxt=5./256;ytxt=5./512

    nfiles = 0
    nff = len(fits_files)
    cube = np.zeros((nff,256,512),dtype=np.float32)

    if nff > 0 and (log or lists):              # Open outputfiles
       if log:
           lg = open(date+'.log','w')
       if lists:
           flis_1 = open(date+'.1_flats','w')
           flis_2 = open(date+'.2_flats','w')
           slis_adi = open(date+'.adi','w')
           slis_sdi = open(date+'.sdi','w')
           slis_asdi = open(date+'.asdi','w')
           slis_other = open(date+'.other','w')

    hh = 'MIN-MAX rms 8x8 boxes: (min - max) MEDIAN: [ext1,ext2],(OBJECT),'\
           '(OBSCLASS),(OBSTYPE),MODE,(ITIME),(NCOADD)' 
    print hh
    for ff in fits_files:

        t1=time.time()
        fd = pf.open(ff)              # open the Fits file
        phu = fd[0].header
        inst = phu['instrume']       # instrument name  for this image

        if inst != 'NICI':             #if not NICI, get the next filename
            print ff+' .. '+inst
            fd.close()
            continue

        if len(fd) != 3:
           line = ' Nici file '+ff+' does not have 2 extensions.'
           print line
           if log:
                lg.write(line+'\n')
           fd.close()
           continue

        nfiles += 1

        im_ext1 = fd[1].data
        hd1 = fd[1].header    # reads extension header 1
        im_ext2 = fd[2].data
        hd2 = fd[2].header    # reads extension header 1
            
        shift_status1=-1 
        shift_status2=-1
                              # extension 1 and blue channel in
                              # extension 2.
                              # -1 if we do not have red channel
                              # 0 is all good
                              # 1 is shifted rows

        if hd1.get('filter_b') == None and hd1.get('filter_r') != None:
            shift_status1 = check_missing_rows (im_ext1,bad_r,log,lg)
        if hd2.get('filter_b') != None and hd1.get('filter_r') == None:
            shift_status2 = check_missing_rows (im_ext2,bad_b,log,lg)
 
        devs = np.zeros(8, dtype=np.float32)     
             # Extracting the RMS of the
             # 8x8-folded image sample. A high value may 
             # indicate readout noise problems
        for sample in range(4):
            x1 = 200+600*(sample % 2)
            x2 = 263+600*(sample % 2)
            y1 = 200+600*(sample/2)
            y2 = 263+600*(sample/2)
            devs[sample] = np.std(fold_88(im_ext1[x1:x2,y1:y2]))

        for sample in range(4):
            x1 = 200+600*(sample % 2)
            x2 = 263+600*(sample % 2)
            y1 = 200+600*(sample/2)
            y2 = 263+600*(sample/2)
            devs[sample+4] = np.std(fold_88(im_ext2[x1:x2,y1:y2]))

        ncoadd_r = hd1['ncoadd_r']
        ncoadd_b = hd2['ncoadd_b']
        itime_r = hd1['itime_r']
        crmode = phu.get('crmode')

        if crmode == None:
            line = 'Header keyword CRMODE not found in '+ff+' -skipped.'
            print line
            if log:
                lg.write(line+'\n')
            fd.close()
            continue

        #print the output
        obstype = str(phu.get('obstype'))
        oclass = str(phu.get('obsclass'))
        dichroic = phu.get('dichroic')
        mask = phu.get('fpmw')
        if 'clear' in mask.lower():
            mask = '-NoMASK'
        else:
            mask = ''
        tmp = obstype
        # 1-Channel (ADI)  (Dichroic is Mirror*)
        # 2-Channel (SDI or ASDI)  (Dichroic is H-50/50*)
        mode = "OTHER"
        if (obstype == 'FLAT' or obstype == 'DARK'):
            if obstype == 'FLAT':
                tmp = 'FLAT['+phu.get('gcalshut')+']'
            if 'mirror' in dichroic.lower():  
                mode = 'ADI'
            else:
                mode = 'SDI'
        elif (oclass == 'science'):
            if (crmode == 'FIXED'):
                if 'mirror' in dichroic.lower():  
                    mode = 'ADI'+mask
                else:
                    mode = 'ASDI'+mask
            elif (crmode == 'FOLLOW' and ('mirror' not in dichroic.lower())):  
                mode = 'SDI'+mask
            else:
                mode = '***WARNING***: Not ADI, SDI nor ASDI-- '+crmode+' '+dichroic
        line = str(phu.get('object'))+','+ oclass+','+tmp+', '+mode
        ti_coad = str(itime_r)+','+str(ncoadd_r)
        longl =ff+': '+'%.2f'%min(devs) + '-' + '%.2f'%max(devs) + ' [' + \
               '%.2f'%np.median(im_ext1) + ', '\
               '%.2f'%np.median(im_ext2) +'] '+line+', ' + ti_coad


      #hh = 'MIN-MAX rms 8x8 boxes: (min - max) MEDIAN: [ext1,ext2],(OBJECT),'\
      #     '(OBSCLASS),(OBSTYPE),(CRMODE),(DICHROIC),(FPMW),(ITIME),(NCOADD)' 
        print longl
        
        if log:
            lg.write(longl+'\n')
        if lists:
            dichr = str(phu.get('dichroic'))
            if (obstype == 'FLAT' or obstype == 'DARK'):
                if 'mirror' in dichr.lower():
                    flis_1.write(ff+'\n')
                else:
                    flis_2.write(ff+'\n')
            elif oclass == 'science':
                md = mode.lower()
                if 'mask' in md:
                    slis_other.write(ff+'\n')
                elif 'adi' in md:
                    slis_adi.write(ff+'\n')
                elif 'asdi' in md:
                    slis_asdi.write(ff+'\n')
                elif 'sdi' in md:
                    slis_sdi.write(ff+'\n')
                else:
                    slis_other.write(ff+'\n')
                
        #extast,hd1,astr ;extract the astrometric structure (astr) from the first extension header
        #cd=astr.cd ;extract the CD matrix from the astrometric structure, this matrix defines scale/rotation/shear


             # read the cassrotator mode from the header
        exp_time = hd1['itime_r']*ncoadd_r/60. 
             # exposure time ... in minutes!

        x = hd1['CD2_1'] - hd1['CD1_2']
        y = hd1['CD1_1'] - hd1['CD2_2']
        sky_angle = np.degrees(math.atan2(x/2, y/2))
             # detector angle on sky
        ra_pointing = hd1['crval1']                # pointing RA from astrometry
        dec_pointing = hd1['crval1']               # pointing DEC
        st = np.asfarray(phu['st'].split(':'))
        st = nt.dmstod(phu['st'])                  # sidereal time
        HA = st - ra_pointing/15.                  # defining the hour angle
        harr = HA + np.asfarray([0,-1/60.])
        para_angle = nt.parangle(harr, dec_pointing, -30.240750) 
              # computing the paralactic angle. -30.240750 is the 
              # latitude of Cerro Pachon

        obs_time = (nt.dmstod(phu['LT'])*60.+720) % 1440
              # here time is expressed in minutes after noon

        angular_rate = para_angle[0] - para_angle[1] 
              # computing the rate of change of the paralactic angle
        current_angle = para_angle[0]
        astrom_angular_rate = (current_angle - previous_angle)/ \
                              (obs_time - previous_time)
              # angle change as found by taking the difference in 
              # angle between this frame and the previous one

        telescope_offset = np.sqrt( ((ra_pointing - previous_ra)*\
              math.cos(np.radians(dec_pointing)))**2+\
              (dec_pointing - previous_dec)**2 ) 

        if (crmode == 'FIXED') and (previous_mode == 'FIXED'):
            # we are in ADI mode 
            if (abs(angular_rate - astrom_angular_rate) > 1/10.) \
                and (telescope_offset < 1E-3) and \
                ((obs_time - previous_time) < 3*exp_time): 
                    print '---------- !!!!!!!! ---------------'
                    print 'Expected angular rate : '+str(angular_rate)+\
                          ' deg/minute'
                    print 'Astrometric angular rate : '+\
                          str(astrom_angular_rate)+' deg/minute'

        tmp = nt.rebin(im_ext1,32,32) 
              # binning the image to find the bright region
        Ndev = (np.amax(tmp) - np.median(tmp))/robust_sigma(tmp) 
              # the peak is Ndev times higher than the sigma of 
              # the median-binned image
        crop1 = 0 
        crop2 = 0 #logical flag to keep track whether we cropped or not 
        if Ndev > 50: 
            crop1 = 1
            A,nfev = gf.gauss2dfit(tmp) 
                   # 2D gaussfit on the peak of the binned image. 
                   # We crop around that peak
            if (nfev > 1550):
                data=nt.medbin(im_ext1,32,32)
                g = data.argmax()
                A[4] = g % 32
                A[5] = g / 32
               
            x = A[4] ; y = A[5]
            x*=32 ; y*=32
            x+=16 ; y+=16

            if x-128 < 0 or x+128 > 1024 or y-128 < 0 or y+128 > 1024:
                out_ext1 = nt.rebin(im_ext1,256,256)
                crop1=0
            else:
                out_ext1 = im_ext1[y-128:y+128,x-128:x+128]

        else:
            out_ext1 = nt.rebin(im_ext1,256,256) 
                # if not cropping, we just bin the image down to 256x256

        tmp = nt.rebin(im_ext2,32,32)    #same thing for extension 2
        Ndev = (np.amax(tmp) - np.median(tmp))/robust_sigma(tmp) 
                   # the peak is Ndev times higher than the sigma 
                   # of the median-binned image
        if Ndev > 50:
            crop2=1
            A,nfev = gf.gauss2dfit(tmp)
            if (nfev > 1550):
                data=nt.medbin(im_ext2,32,32)
                g = data.argmax()
                A[4] = g % 32
                A[5] = g / 32
               
            x=A[4] ; y=A[5]
            x*=32 ; y*=32
            x+=16 ; y+=16
            
            if x-128 < 0 or x+128 > 1024 or y-128 < 0 or y+128 > 1024:
                out_ext2 = nt.rebin(im_ext2,256,256)
                crop2=0
            else:
                out_ext2 = im_ext2[y-128:y+128,x-128:x+128]

        else:
            out_ext2 = nt.rebin(im_ext2,256,256)

        a = out_ext1/ncoadd_r
        saturated_1 = np.clip(a, np.amin(a), saturate)
        saturated_1[0:140,0:20] = 0
        a = out_ext2/ncoadd_b
        saturated_2 = nd.rotate(np.clip(a,saturate,np.amax(a)),5,reshape=False)
        saturated_2[0:140,0:20] = 0

        #out_ext1>=out_ext1[ (sort(out_ext1))[.01*n_elements(out_ext1)]]
        szx1 = np.size(out_ext1)
        sout = np.sort(out_ext1.ravel())
        limit = sout[.01*szx1]
        xmin = sout[0]; xmax= sout[np.size(sout)-1]
        out_ext1 = np.clip(out_ext1, limit, xmax)
                # we put lower and upper limits and the 1% and 
                # 99.5% (OR NCOADD*3500) of the image
        #out_ext1<=( out_ext1[ (sort(out_ext1))[.995*n_elements(out_ext1)]]
        #           < (sxpar(hd1,'ncoadd_r')*saturate) )
        szx1 = np.size(out_ext1)
        if sout[.995*szx1] < ncoadd_r*saturate:
            limit = ncoadd_r*saturate
        else: 
            limit = sout[.995*szx1] 
         
        out_ext1 = np.clip(out_ext1, xmin, limit)


        #out_ext2>=out_ext2[ (sort(out_ext2))[.01*n_elements(out_ext2)]]
        #    ;same for channel 2
        
        szx2 = np.size(out_ext2)
        sout = np.sort(out_ext2.ravel())
        limit = sout[.01*szx2]
        xmin = sout[0]; xmax= sout[np.size(sout)-1]
        out_ext2 = np.clip(out_ext2, limit, xmax)
                # we put lower and upper limits and the 1% and
                # 99.5% (OR NCOADD*3500) of the image
        #out_ext2<=( out_ext2[ (sort(out_ext2))[.995*n_elements(out_ext2)]]
        #           < (sxpar(hd2,'ncoadd_b')*saturate) )
        szx2 = np.size(out_ext2)
        if sout[.995*szx2] < ncoadd_b*saturate:
            limit = ncoadd_b*saturate
        else:
            limit = sout[.995*szx2]

        out_ext2 = np.clip(out_ext2, xmin, limit)

        out = np.zeros((256,512),dtype=np.float32)
              #create the image that will receive the frames to display
        out[:,:256] = (out_ext1 - np.amin(out_ext1)) / \
                       (np.amax(out_ext1) - np.amin(out_ext1))
        a = out_ext2 - np.amin(out_ext2)
        b = np.amax(out_ext2) - np.amin(out_ext2)
        out[:,256:] = nd.rotate( (a/b), 5, reshape=False)

        name_out=ff     #extract the name of the file without the full path
        name_out = name_out.split('/')[-1:]  # Get the last element
        name_out = name_out[0][:-5]    # take '.fits' out
        name_out+='['+(['full','crop'])[crop1]+'/'+(['full','crop'])[crop2]+']'
       
        dpi = py.rcParams['figure.dpi']
        fig = figure.Figure(figsize=(H,W), frameon=False)
        canvas = FigureCanvas(fig)
        fig.figimage(out[::-1,:])
        fig.text(xtxt, ytxt,name_out,backgroundcolor='white')
        canvas.draw()
        buf = canvas.buffer_rgba(0,0)
        l, b, w, h = fig.bbox.get_bounds()
        I = np.fromstring(buf, np.uint8)
        I.shape = h,w,4
        
        if (display):
            py.imshow(I[:,:,2],cmap=cm.gray)

        cube[nfiles-1,:,:]=I[:,:,2]

        previous_angle=current_angle 
             #keep track of some variables for next iteration
        previous_time =obs_time
        previous_ra   =ra_pointing
        previous_dec  =dec_pointing
        previous_mode =crmode
        fd.close()
    if log:
       lg.close()
    if lists:
       flis_1.close()
       flis_2.close()
       slis_adi.close()
       slis_sdi.close()
       slis_asdi.close()
       slis_other.close()

    if (nfiles > 0): 
        pf.writeto(date+'_cube.fits',cube[:nfiles,::-1,:], clobber=True)
          # write the output cube with cropped/binned images
    else:
        print 'No NICI files found for this date:',date
    return


def  check_missing_rows(im,bad,log,lg):
    imrv = im.ravel()
    if abs( np.median(imrv[bad])-np.median(imrv[bad+1]) ) < 50:
        line = 'image is affected by the infamous shifting rows'
        print line
        if log:
            lg.write(line+'\n')
        
        bad_shift=bad.copy()
        top   = np.where(bad > 512*1024.)
        bottom = np.where(bad <= 512*1024.)
        bad_shift[top   ]-=2048
        bad_shift[bottom]+=2048

        if abs( np.median(imrv[bad_shift])-np.median(imrv[bad_shift+1]) ) > 50:
            return 1 #shifted
        line =  'something strange with the file'
        print line
        if log:
            lg.write(line+'\n')
        #return -1
    else:
        return 0 #normal

def  fold_88(im):
     box = np.zeros((8,8), dtype=np.float32)
     sz=np.shape(im)
     y,x = xygen(sz[0],sz[1])
     for i in range(8):
         for j in range(8): 
             box[i,j] = np.median(im[np.where(((x%8) == i) & (y%8 == j))])
     return box

def xygen(ny,nx):
     y = np.arange(nx*ny).reshape(ny,nx)/ny - (ny/2 - 1)
     x = np.rot90 (y)
     return y,x

if __name__ == "__main__":

    import optparse

    # Parse input arguments
    usage = 'usage: %prog date [options]'

    p = optparse.OptionParser(usage=usage, version='niciprepare_1.0')
    p.add_option('--saturate', action='store', type='int', default=5000,
                 help='saturate level')
    p.add_option('--nodisplay', action='store_false', default=True, 
                 help="Don't see running images")

    (par, args) = p.parse_args()

    if len(args) == 0:
        date=None
    else:
        date = args[0]

    nici_das(date=date, help=None,saturate=par.saturate,display=par.nodisplay)
