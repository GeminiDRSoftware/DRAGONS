
import numpy as np
from numpy import size
from niciTools import medbin
from scipy.interpolate import interp2d
from dist_circle import dist_circle
import congrid as cg
import robustsigma as rs
from c_correlate import c_correlate
from gcntrd import gcentroid
import math as mt
import pyfits as pf

#pro peak2halo,fic,ratio,xcen,ycen,exten=exten,image=image
def peak2halo(files,exten=None,image=None):

    if (image is not None): 
        im = image
        xcen = 0.0
        ycen = 0.0
        ratio = 0.0
        nfiles = 1 
    else:   
        if isinstance(files,(tuple,list)):
           nfiles = size(files)
        else:
           nfiles = 1   
           files = [files]
        xcen = np.empty(nfiles,dtype=float)
        ycen = np.empty(nfiles,dtype=float)
        ratio = np.empty(nfiles,dtype=float)
        if exten is None: exten = 1
        if exten != 1 and exten != 2 :
            print 'the EXTEN keyword must be equal to 1 OR 2.'
            return

    k = 0
    for nfic in range(nfiles):

        #print 'peak2halo ['+str(nfic+1)+'/'+str(nfiles)+']'

        if (image is None):
            hdr = pf.getheader(files[nfic])
            if hdr['instrume']!='NICI':
                print files[nfic],' is not NICI file. Skip it'
                return
                #continue
            elif hdr['obsclass']!= 'science':
                print files[nfic],' is not NICI science file. Skip it'
                return
                #continue
                
            im = pf.getdata(files[nfic],exten)
                 # reads the fits image. We will measure the peak/halo 
                 # ratio only in the extension=exten
        bin = medbin(im,32,32)
        # # calculates de std inside
        #dev = bin[5:27,5:27].std()
        #bin = bin[5:27,5:27] - np.median(bin[5:27,5:27])
        
        ## see if we have a mask in the frame
        #nelem_mask = np.size(np.where(abs(bin)>dev*5))/2
        #if nelem_mask == 0:
        #    return -99,-1,-1

        im = cg.congrid( (bin > 3*rs.robust_sigma(bin))*1 ,(1024,1024))*im

        px = np.zeros(1024,dtype=float)
        py = np.zeros(1024,dtype=float)

        for i in range(64):
            py = np.clip(py,\
                np.nan_to_num(np.median(im[:,i*16:i*16+16],axis=1)),np.amax(py))
        for i in range(64):
            px = np.clip(px, \
                np.nan_to_num(np.median(im[i*16:i*16+16,:],axis=0)),np.amax(px))
                 # extract the maximum value seen in the image for a 
                 # 16 pixel-wide stripe- in the x and y direction 
        px = np.clip(px,min(px),max(px)*.25)
        py = np.clip(py,min(py),max(py)*.25)


        offset = 1023 - np.arange(2047.) 
                # cross-correlate this profile with a 'flipped' version 
                # of itself. A peak at pixel 512 will have a '0' offset
        xrr = size(px) - np.arange(size(px))-1
        cx = c_correlate(px,px[xrr],offset) 
                # a peak at pixel N will have a 2*(512-N) offset
        cy = c_correlate(py,py[size(py)-np.arange(size(py))-1],offset)
       
        ix = cx.argmax()
        xcen_tmp0 = 512-offset[ix]/2.
        iy = cy.argmax()
        ycen_tmp0 = 512-offset[iy]/2.    # derive the position of the peakmax
        xcen_tmp,ycen_tmp = gcentroid(im,xcen_tmp0,ycen_tmp0,9) 
               # get a fine centroid at this position
        if xcen_tmp < 0:
            ratio=0
            break 
        if mt.sqrt((xcen_tmp0-xcen_tmp)**2+(ycen_tmp0-ycen_tmp)**2) > 9: 
           break

        xc = round(xcen_tmp)
        yc = round(ycen_tmp)
        x = np.arange(xc-3,xc+4)
        y = np.arange(yc-3,yc+4)
        z = im[yc-3:yc+4,xc-3:xc+4]

        pxy = interp2d(x,y,z,kind='cubic')
        peak = float(pxy(xcen_tmp,ycen_tmp))
            # get an accurate estimate of the peak value
        r = dist_circle([1024,1024],xcen=xcen_tmp,ycen=ycen_tmp)
        g = np.where((r > 20) & (r < 30)) 
             # get the median value between 20 and 30 pixels of this peak
        halo = np.median(im[g])
       
         
        if nfiles > 1:
            ratio[k] = peak/halo
            xcen[k] = xcen_tmp
            ycen[k] = ycen_tmp
        else:
           if xcen_tmp < 0:
               ratio = 0.0
           else: ratio = peak/halo
           xcen = xcen_tmp
           ycen = ycen_tmp
           return ratio,np.float(xcen_tmp),np.float(ycen_tmp)
        
        k += 1
    
    if (nfiles == 1):
       if xcen_tmp < 0:
           ratio = 0.0
           xcen = -1; ycen = -1
       else:
           ratio = peak/halo
           xcen = np.float(xcen_tmp)
           ycen = np.float(ycen_tmp)

    return ratio,xcen,ycen
