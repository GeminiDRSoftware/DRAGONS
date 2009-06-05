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
            im = pf.getdata(files[nfic],exten)
                 # reads the fits image. We will measure the peak/halo 
                 # ratio only in the extension=exten

        bin = medbin(im,32,32)
        #im = rebin(bin gt 3*robust_sigma(bin),1024,1024)*im
        #pixs = bin > 3*rs.robust_sigma(bin)
        pixs = np.clip(np.nan_to_num(bin), 3*rs.robust_sigma(bin),\
               np.amax(bin))
        im = cg.congrid(pixs,(1024,1024))*im

        px = np.empty(1024,dtype=float)
        py = np.empty(1024,dtype=float)

        for i in range(64):
            #py >= killnan(median(im[i*16:i*16+15,*],dim=1)) 
            py = np.clip(py,\
                np.nan_to_num(np.median(im[:,i*16:i*16+16],axis=1)),np.amax(py))
        for i in range(64):
            #px >= killnan(np.median(im[*,i*16:i*16+15],dim=2)) 
            px = np.clip(px, \
                np.nan_to_num(np.median(im[i*16:i*16+16,:],axis=0)),np.amax(px))
                 # extract the maximum value seen in the image for a 
                 # 16 pixel-wide stripe- in the x and y direction 
        #px=px<max(px)*.25
        #py=py<max(py)*.25
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
        ycen_tmp0 = 512-offset[iy]/2.    # derive the position of the peak
        xcen_tmp,ycen_tmp = gcentroid(im,xcen_tmp0,ycen_tmp0,9) 
               # get a fine centroid at this position
        if mt.sqrt((xcen_tmp0-xcen_tmp)**2+(ycen_tmp0-ycen_tmp)**2) > 9: 
           break

        xc = round(xcen_tmp)
        yc = round(ycen_tmp)
        x = np.arange(xc-3,xc+4)
        y = np.arange(yc-3,yc+4)
        z = im[yc-3:yc+4,xc-3:xc+4]
        #pxy = interp2d(x,y,z,kind='cubic')
        #peak = float(pxy(xcen_tmp,ycen_tmp))
            # get an accurate estimate of the peak value
        # peak = interpolate(im,xcen_tmp,ycen_tmp,cubic=-.6) 
        #dist_circle,r,[1024,1024],xcen_tmp,ycen_tmp
        peak = 0.0
        r = dist_circle([1024,1024])
        g = np.where((r > 20) & (r < 30)) 
             # get the median value between 20 and 30 pixels of this peak
        halo = np.median(im[g])
        
        if nfiles > 1:
            #ratio[k] = peak/halo
            xcen[k] = xcen_tmp
            ycen[k] = ycen_tmp
        
        k += 1
    
    if (nfiles == 1):
       #ratio = peak/halo
       xcen = xcen_tmp
       ycen = ycen_tmp
    return xcen,ycen
