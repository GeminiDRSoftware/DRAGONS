import numpy as np
import ndimage as nd
from peak2halo import peak2halo
import scipy.signal.signaltools
import robustsigma as rs
import numdisplay as ndis
from gcntrd import gcentroid
import congrid as cg
import sys


def nici_cntrd(im,hdr,center_im=True,force=False):
    """
    Read xcen,ycen and update header if necessary
    If the automatic finding of the center mask fails, then
    the interactive session will start. The SAOIMAGE/ds9
    display must be up. If the default port is busy, then it will
    use port 5199, so make sure you start "ds9 -port 5199".
    """

    xcen = hdr.get('xcen')
    ycen = hdr.get('ycen')
    update = False
     
    if (xcen == None):
        xc,yc=peak2halo('',image=im)
        xcen = xc[0]
        ycen = yc[0]
        hdr.update("XCEN",xcen,"Start mask x-center")
        hdr.update("YCEN",ycen,"Start mask y-center")
      
        update = True 
    elif (xcen < 0 or ycen < 0):
        ndis.display(im)
        #if not ndis.checkDisplay():
        #   print "\n*******ERROR: cannot open display, trying port 5199"
        #   ndis.open(imtdev='inet:5199') 
            
        #ndis.display(im,name=file+'[1]')
        print " Mark center with left button, then use 'q' to continue."
        cursor=ndis.readcursor(sample=0)
        cursor = cursor.split()
        x1 = float(cursor[0])
        y1 = float(cursor[1])


        box = im[y1-64:y1+64,x1-64:x1+64].copy()
        box -= scipy.signal.signaltools.medfilt2d(box,11)
        box = box[32:32+64,32:32+64]

        bbx = box * ((box>(-rs.robust_sigma(box)*5)) & \
                     (box <(15*rs.robust_sigma(box))))

        imbb = cg.congrid(bbx,(1024,1024))
        ndis.display(imbb, name='bbx')
        del imbb

        cursor = ndis.readcursor(sample=0)
        cursor = cursor.split()
        x2 = float(cursor[0])
        y2 = float(cursor[1])

        xcen,ycen = gcentroid(box, x2/16., y2/16., 4)

        #xcen = xcen+x1-32
        #ycen = ycen+y1-32
        xcen = xcen+x1
        ycen = ycen+y1
        hdr.update("XCEN",xcen[0], "Start mask x-center")
        hdr.update("YCEN",ycen[0], "Start mask y-center")

    else:

        if center_im:
            # Shift the image. Use ndimage shift function. Make sure
            # the array is float.
            im = np.asarray(np.nan_to_num(im),dtype=np.float32)
            im = nd.shift (im,(512-ycen,512-xcen))     

    return update,xcen,ycen,im
