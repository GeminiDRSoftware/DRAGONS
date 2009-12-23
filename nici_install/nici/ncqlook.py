#!/usr/bin/env python

import nici as nc
import numpy as np
import datetime
import glob
import pyfits as pf
from nici.robustsigma import robust_sigma
import nici.gemfit as gf
import nici.niciTools as nt
import ndimage as nd
import math
import pysao as sao
import matplotlib.mathtext as mathtext
import os, sys, imp

import time

def ncqlook (inputs, idir='', odir='./', log=True,lists=True,saturate=None,display=True,crop=False):

    """
        nqlook
             Get NICI data from SBF: /net/petrohue/dataflow repository
             for last night date. Produces a quick display of each
             frame plus log file and lists that can be used with
             niciprepare, nicimkflats and niciscience scripts.
             The output lists are written in the working directory.

        nqlook inputs='20081116' saturate=2000 display=False 
             Get data from 20081116 with a saturation limit at 2000 
             and does not display (for speed)'

        nqlook @inlis idir='/data/myfiles' saturate=2000 display=False 
             Get data  from 'inlis' which is a list of input FITS files
             pathnames -one per line. The actual files are in the directory
             'idir'
             Use a saturation limit at 2000 and does not display (for speed)'
             Even though you have 'inputs' parameter in here, it is not
             taken into account since all the FITS files from the input
             directory will be read.

        keyword saturate. If not present, saturation >3500 

        NOTE: The files bad_r.fits ans bad_b.fits are read from 
              the nici directory wherever is it installed.
    """ 
    if (inputs == None):
        today = datetime.date.today()
        dd = today.timetuple()
        inputs = str(dd[0]) + '%.2d'%dd[1] + '%.2d'%dd[2]


    if (saturate == None): saturate=3500

    if display:
         ds9 = sao.ds9() 
         #try:
         #   ndis.open()
         #except:
         #   print "\n ******************  WARNING:  Please start ds9."
         #   return
         #else:
         #   ndis.close()

    # We need to know were we are. 'nici' is the package name
    fp, pathname, description = imp.find_module('nici')
    pathname += '/'

            #loads the badpixels list for the blue and red channel
    bad_b = pf.getdata(pathname+'bad_b.fits')
    bad_r = pf.getdata(pathname+'bad_r.fits')

    # Chech that we can write in the current directory
    # the output FITS file is written here.
    odir = os.path.join(odir,'')         # Make sure odir has '/'
    if not os.access(odir,os.W_OK):
        print "\nERROR: You do not have write access to \
                this directory (output FITS file)\n"
        return

    if idir != '':
        idir = os.path.join(idir,'')         # Make sure odir has '/'
        if not os.access(idir,os.R_OK):
            print "\nERROR: No access to input directory:",idir
            return

    pathS = '/net/petrohue/dataflow/S'
    froot = 'ncqlook'
    if 'list' not in str(type(inputs)):
        date = str(inputs)
        fits_files = nt.getFileList(date)
        # see if it date format  20yymmdd. Will not work for 20?.fits
        if len(fits_files) == 1:
           dd = fits_files[0] 
           if len(dd) == 8 and dd[:2] == '20':
               print 'Looking for files: ' +pathS+date+'S*.fits'
               fits_files = glob.glob(pathS+date+'S*.fits')
               froot = date
    else:
        # it is already a list:
        fits_files = inputs

    command = 'ncqlook ('+str(inputs)+',idir='+idir+',odir='+odir+',log=' \
             +str(log)+',lists='+str(lists)+',saturate='+str(saturate)\
             +',display='+str(display)+',crop='+str(crop)+')'

    fits_files = np.sort(fits_files)
    if len(fits_files)==0:
        print 'couldn''t find files for :'+pathS +inputs+'S*.fits'
        return

    print '\nfound '+str(len(fits_files))+ 'files.'

    previous_angle = -999 #define a few dummy variables
    previous_time = -999
    previous_ra   = -999
    previous_dec  = -999
    previous_mode = 'NONE'

    nfiles = 0
    nff = len(fits_files)
    cube = np.zeros((nff,256,512),dtype=np.float32)

    hh = 'MIN-MAX rms 8x8 boxes: (min - max) MEDIAN: [ext1,ext2],(OBJECT),'\
           '(OBSCLASS),(OBSTYPE),MODE,(ITIME),(NCOADD),(CORE2HALO)' 
    print command
    print hh
    lg = 0
    if nff > 0 and (log or lists):              # Open outputfiles
       if log:
           lg = open(odir+froot+'.log','w+')
           lg.write('COMMAND: '+command+'\n')
           lg.write(hh+'\n')
       if lists:
           flis_1 = open(odir+froot+'.1_flats','w+')
           flis_2 = open(odir+froot+'.2_flats','w+')
           slis_adi = open(odir+froot+'.adi','w+')
           slis_sdi = open(odir+froot+'.sdi','w+')
           slis_asdi = open(odir+froot+'.asdi','w+')
           slis_other = open(odir+froot+'.other','w+')


    parser = mathtext.MathTextParser("Bitmap")
    for ff in fits_files:

        t1=time.time()
        fd = pf.open(idir+ff)              # open the Fits file
        if len(fd) == 0:
            print ff," is EMPTY."
            continue
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
        dichr = str(phu.get('dichroic')).lower()
        adi = ''
        if 'mirror' in dichr:
            adi = 'blue'
        if 'open' in dichr:
            adi = 'red'

        mask = phu.get('fpmw')
        if 'clear' in mask.lower():
            mask = '-NoMASK'
        else:
            mask = ''
        tmp = obstype
        # 1-Channel (ADI)  (Dichroic is Mirror*(100% blue) or Open (100% red))
        # 2-Channel (SDI or ASDI)  (Dichroic is H-50/50*)
        mode = "OTHER"
        if (obstype == 'FLAT' or obstype == 'DARK'):
            if obstype == 'FLAT':
                tmp = 'FLAT['+phu.get('gcalshut')+']'
            if adi:
                mode = 'ADI-'+adi[0].upper()
            else:
                mode = 'SDI'
        elif (oclass == 'science'):
            if (crmode == 'FIXED'):
                if adi:
                    mode = 'ADI-'+adi[0].upper()+mask
                else:
                    mode = 'ASDI'+mask
            elif (crmode == 'FOLLOW' and adi):  
                mode = 'SDI'+mask
            elif '-ENG' in  phu.get('OBSID'):
                mode = '* ENGINEERING DATA *'
            else:
                mode = '***WARNING***: Not ADI, SDI nor ASDI-- '+crmode+' '+dichr
        line = str(phu.get('object'))+','+ oclass+','+tmp+', '+mode
        ti_coad = str(itime_r)+','+str(ncoadd_r)
        longl =os.path.basename(ff)+': '+'%5.2f'%min(devs) + '-' + '%5.2f'%max(devs) + ' [' + \
               '%6.2f'%np.median(im_ext1) + ', '\
               '%6.2f'%np.median(im_ext2) +'] '+line+', ' + ti_coad
        if 'ENGINE' in mode:
            print longl
            if log:
                lg.write(longl+'\n')
            continue

      #hh = 'MIN-MAX rms 8x8 boxes: (min - max) MEDIAN: [ext1,ext2],(OBJECT),'\
      #     '(OBSCLASS),(OBSTYPE),(CRMODE),(DICHROIC),(FPMW),(ITIME),(NCOADD)' 
        # Don't print until core2halo value is defined below.
        #print longl
        
        if lists:
            if (obstype == 'FLAT' or obstype == 'DARK'):
                if adi:
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


        # CROP THE FRAMES and get core2halo ratios

        out_ext1,out_ext2,c2h_1,c2h_2 = set_exts(im_ext1,im_ext2,crop,oclass=='science',adi)

        if not adi == 'red':    # Do not process if there is no red flux
            a = out_ext1/ncoadd_r
            saturated_1 = np.clip(a, np.amin(a), saturate)
            saturated_1[0:140,0:20] = 0

            #IDL out_ext1>=out_ext1[ (sort(out_ext1))[.01*n_elements(out_ext1)]]
            szx1 = np.size(out_ext1)
            sout = np.sort(out_ext1.ravel())
            limit = sout[.01*szx1]
            xmin = sout[0]; xmax= sout[np.size(sout)-1]

            out_ext1 = np.clip(out_ext1, limit, xmax)
                    # we put lower and upper limits and the 1% and 
                    # 99.5% (OR NCOADD*3500) of the image
            #IDL out_ext1<=( out_ext1[ (sort(out_ext1))[.995*n_elements(out_ext1)]]
            #           < (sxpar(hd1,'ncoadd_r')*saturate) )
            szx1 = np.size(out_ext1)
            if sout[.995*szx1] < ncoadd_r*saturate:
                limit = ncoadd_r*saturate
            else: 
                limit = sout[.995*szx1] 
         
            out_ext1 = np.clip(out_ext1, xmin, limit)

        if not adi == 'blue':    # Do not process if there is no blue flux 
            a = out_ext2/ncoadd_b
            saturated_2 = nd.rotate(np.clip(a,saturate,np.amax(a)),5,reshape=False)
            saturated_2[0:140,0:20] = 0

            szx2 = np.size(out_ext2)
            sout = np.sort(out_ext2.ravel())
            limit = sout[.01*szx2]
            xmin = sout[0]; xmax= sout[np.size(sout)-1]

            out_ext2 = np.clip(out_ext2, limit, xmax)
                    # we put lower and upper limits and the 1% and
                    # 99.5% (OR NCOADD*3500) of the image
            #IDL: out_ext2<=( out_ext2[ (sort(out_ext2))[.995*n_elements(out_ext2)]]
            #           < (sxpar(hd2,'ncoadd_b')*saturate) )
            szx2 = np.size(out_ext2)
            if sout[.995*szx2] < ncoadd_b*saturate:
                limit = ncoadd_b*saturate
            else:
                limit = sout[.995*szx2]

            out_ext2 = np.clip(out_ext2, xmin, limit)

        out = np.zeros((256,512),dtype=np.float32)
              #create the image that will receive the frames to display
        out[:,:256] = ((out_ext1 - np.amin(out_ext1)) / \
                       (np.amax(out_ext1) - np.amin(out_ext1)))[::-1,:]
        a = out_ext2 - np.amin(out_ext2)
        b = np.amax(out_ext2) - np.amin(out_ext2)
        out[:,256:] = nd.rotate( (a/b), 5, reshape=False)[::-1,::-1]

        name_out = ff     #extract the name of the file without the full path
        name_out = name_out.split('/')[-1:]  # Get the last element
        name_out = name_out[0][:-5]    # take '.fits' out
        if c2h_1 >= 0:
           name_out += '['+(['full','crop'])[crop]+']'
        else:
           name_out += '[full]'
       
        # Form the text to overlay onto the lower left of the output array.
        rgba, depth1 = parser.to_rgba(name_out, color='gray', fontsize=10, dpi=200)
        c = rgba[::2,::2,3]
        far = np.asarray(c,dtype=np.float32)
        stx = np.shape(far)
        out[:stx[0],:stx[1]] = far[::-1,:]/256
 
        if (display):
            #ndis.display(out,z1=-1,z2=1)
            ds9.view(out)           


        cube[nfiles-1,:,:] = out[::-1,:]

        # Now print 
        if oclass == 'science':
            longl += ',['+'%.2f,%.2f' % (c2h_1,c2h_2)+']'
        print longl
        if log:
            lg.write(longl+'\n')

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
        pf.writeto(odir+froot+'_cube.fits',cube[:nfiles,::-1,:], clobber=True)
          # write the output cube with cropped/binned images
    else:
        print 'No NICI files found for the specified input: ',inputs
    return


def set_exts(ext1,ext2,crop,science,adi):
        """
           See if we need to crop both frames
        """

        def _set_output(ext,crop):
            """
               Get center of mask and core2halo value plus
               recenter if crop is set.
            """
            from  peak2halo import peak2halo

            core2halo = -1
            if crop == 60:
                out_ext = nt.rebin(ext,256,256) 
                return out_ext, core2halo
            else:
                try:
                    core2halo,x,y = peak2halo('',image=ext)
                    x = round(x); y = round(y)
                    if x<0: crop=False
                except:
                    crop = False

                if  not crop:
                    out_ext = nt.rebin(ext,256,256) 
                    return out_ext, core2halo

                if x-128 < 0 or x+128 > 1024 or y-128 < 0 or y+128 > 1024:
                    out_ext = nt.rebin(ext,256,256)
                    core2halo = 0
                    return out_ext, core2halo
                
                #shift frame to center
                out_ext = ext[y-128:y+128,x-128:x+128]
                return out_ext, core2halo
       
        if not science:
            out_ext1 = nt.rebin(ext1,256,256) 
            out_ext2 = nt.rebin(ext2,256,256) 
            return out_ext1,out_ext2,-1,-1

        Ndev = lambda im : (np.amax(im) - np.median(im))/robust_sigma(im) 

        if (Ndev(ext1) >50) and (Ndev(ext2) >50):
            # the peak is Ndev times higher than the sigma of 
            # the median-binned image. We crop both extension
            if adi == 'red':
                # No flux in blue frame
                out_ext2, core2halo2 = _set_output(ext2,60)
                out_ext1, core2halo1 = _set_output(ext1,crop)
            elif adi == 'blue':
                # No flux in red frame
                out_ext1, core2halo1 = _set_output(ext1,60)
                out_ext2, core2halo2 = _set_output(ext2,crop)
            else:
                out_ext1, core2halo1 = _set_output(ext1,crop)
                out_ext2, core2halo2 = _set_output(ext2,crop)
        else:
            crop = 60   # Set the value of Ndev
            out_ext1, core2halo1 = _set_output(ext1,crop)
            out_ext2, core2halo2 = _set_output(ext2,crop)

        return out_ext1,out_ext2,core2halo1,core2halo2
            
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
    usage = 'usage: %prog wild_card_or_date(yyyymmdd) [options]'

    p = optparse.OptionParser(usage=usage, version='niciprepare_1.0')
    p.add_option('--idir', action='store', type='string',
                 default='', help='Input directory pathname')
    p.add_option('--odir', action='store', type='string',
                 default='./', help='Output directory pathname')
    p.add_option('--saturate', action='store', type='int', default=5000,
                 help='saturate level')
    p.add_option('--nodisplay', action='store_false', default=True, 
                 help="Do not display running images")
    p.add_option('--crop', action='store_true', dest="crop", default=False, 
                 help="Crop frames around the mask center")

    (par, args) = p.parse_args()

    #inputs= args
    if len(args) == 0:
        inputs=None
    else:
        inputs = args[0]

    ncqlook (inputs, idir=par.idir, odir=par.odir, saturate=par.saturate,display=par.nodisplay,crop=par.crop)

