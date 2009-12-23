import pyfits as pf
import numdisplay as ndis
import numpy as np
import gcntrd as gc
import overlay as ov
import geomap as gm
import scipy.ndimage as nd

class XYtran:
    """
      Class to handle the coordinate transformation necessary to
      turn the Blue frame into the same coordinate system as the
      Red.

            xr = a + b*xb + c*yb
            yr = d + e*xb + f*yb
           
      The input is the FITS file with the NICI pinholes exposure.
      To start do:  (create the Python object)

            cp = xyt.Ctran('S20090114S0038.fits')


      (where 'xyt' comes from:   'import XYtran as xyt'

           
      where the argument is the FITS file name with the pinholes.
      DS9 must be up before you create 'cp'. Frame 1 and 2 are
      loaded with the red and blue frames of the pinhole images.

      OR    cp = xyt.XYtran()    # If you do not need to load the frames

      Next you need to get at least 6 control points from these
      frames by using:

            cp.getCpoints()

      Now to find the parameters (a..f) do:

            cp.doCfit()

      a listing of the residuals is giving together with a 
      line number. To remove a point from the list that
      shows a large residual use:

            cp.delxy(K)

      Where K is the number in the list you want to remove.
      Now repeat 'cp.doCfit()'.
      A text file 'xycoeff' is created in your working directory
      with the coefficients (a..f).

      Now to transform a Blue frame into the coordinate system of the
      Red do:

            im_blue_tran = cp.transform(im_blue)
      
      where im_blue is the variable containing the blue frame pixels.

    """
             
    def __init__(self,pfile=None):
        if pfile != None:
            self.xyr = np.zeros([20,2])
            self.xyb = np.zeros([20,2])
            self.npin = 0
            self.imr = pf.getdata(pfile,1)
            self.imb = pf.getdata(pfile,2)
            #if not ndis.checkDisplay():
            #    print "\n ****ERROR: ds9 is not running."
            #    return
            ndis.display(self.imr,frame=1)
            ndis.display(self.imb,frame=2)
        

    def getCpoints(self):
        """
          Get control points from a pair of Red and Blue pinholes
          frames from the NICI camera.
          You already have frames 1 and 2 (red and blue)
          from a pinhole image up in DS9.
          Please select at least 6 corresponding points. 

          Once you click 'r' in the Red frame, a red circle is drawn
          around the pinhole and a Blue circle is drawn 'near' the
          blue pinhole; click 'b' there to record the position.
          Press 'q' to finish.
        """
 
        i=0; j=0
        xkey = ''
        print "*** Please get at least 6 corresponding pinholes."
        while xkey != 'q':
            cursor = ndis.readcursor(sample=0)
            ss = cursor.split()
            print ss
            xkey = ss.pop()
            if xkey not in ('r','b','q'):
               print "Please choose a control point on the 'r' or 'b' frame,"
               print " then hit 'r' for red frame or 'b' for 'blue' frame"
               continue
            im = {'r':self.imr, 'b':self.imb, 'q':0}[xkey]
            if xkey == 'q': break
            
            x,y = np.asfarray(ss[:2])
            xc,yc = gc.gcentroid(im[y-10:y+10,x-10:x+10],10,10)
            xc,yc = x+xc[0]-10,y+yc[0]-10

            if xkey == 'r':
                #print "xc,yc: ",xc,yc,x+xc-10,y+yc-10
                self.xyr[i] = xc,yc; i+=1
                ov.circle(x=xc,y=yc,radius=10,color=ov.C_RED,frame=1)
                xn = 990.5897 - 0.9985259*x - 0.0178984*y
                yn = 37.82109 - 0.0181331*x + 0.9991414*y
                #xc,yc = gc.gcentroid(self.imb[yn-13:y+13,xn-13:x+13],13,13)
                #xc,yc = xn+xc[0]-10,yn+yc[0]-10
                ov.circle(x=xn,y=yn,radius=10,color=ov.C_BLUE, frame=2)
                print self.xyr[i-1]
            if xkey == 'b':
                self.xyb[j] = xc,yc; j+=1
                print self.xyb[j-1]
            
        self.npin = j
        self.xyr = np.resize(self.xyr,[j,2])
        self.xyb = np.resize(self.xyb,[j,2])

    def savecp(self):
        """
          Saves your control point you got in getCpoint() 
          into files 'xyr' and 'xyb' in your working 
          directory. You can retrieve them later on using 
          loadcp().
        """ 
        self.xyr.tofile('xyr',sep=' ')
        self.xyb.tofile('xyb',sep=' ')

    def loadcp(self,overlay=False):
        """
            Read text files 'xyr' and 'xyb' from your working 
            directory and arrange them into xyr[np,2] and xyb[np,2].
            These are the x,y positions of the control points 
            obtained when calling 'getCpoint' method from this
            XYtran class.

            @overlay: Default value is False. When True, it will
                      overlay the xyr positions into the Frame 1
                      on ds9 and xyb positions into Frame 2.
        """
            
        xrr = np.fromfile('xyr',sep=' ')
        ncp = np.size(xrr)/2
        self.xyr=xrr.reshape([ncp,2])
        xrr = np.fromfile('xyb',sep=' ')
        self.xyb=xrr.reshape([ncp,2])
        self.npin = ncp
        if overlay:
            for i in range(ncp):
                ov.circle(x=self.xyr[i][0],y=self.xyr[i][1],radius=10,\
                         color=ov.C_RED,frame=1)
            for i in range(ncp):
                ov.circle(x=self.xyb[i][0],y=self.xyb[i][1],radius=10,\
                         color=ov.C_YELLOW,frame=2)


    def doCfit(self):
        """
          Solves the linear system to find a..f.
          A text file 'xycoeff'  with the coefficients in the order a thru f 
          is created in your current working directory for reference.

          A listing of the residuals is printed to see
          how good is the fit. You can remove points from 
          here 'delxy(num)' and repeat doCfit to improve
          the fit.
        """
       
        gmo = gm.CoordTransform(self.xyr, self.xyb)
        self.gmo = gmo
        self.func = gmo.doFit()
        for i in range(self.npin):
            print gmo.fitObj.residuals[i],': ',i
        coeff = np.array([gmo.fitObj.a,gmo.fitObj.b,gmo.fitObj.c,\
                      gmo.fitObj.d,gmo.fitObj.e,gmo.fitObj.f])
        # Save to coeff
        coeff.tofile('xycoeff',sep=' ')

    def delxy(self,k):
        """
          Remove a point from the red and blue lists to improve
          the residuals after fitting the points to the
          linear model.

          @k: line number in the listing of residuals.

        """

        k = int(k)
        n = self.npin
        xyr = self.xyr
        xyb = self.xyb
        xyr[k:n-1] = xyr[k+1:n]
        xyb[k:n-1] = xyb[k+1:n]
        self.xyr = np.resize(xyr,[n-1,2])
        self.xyb = np.resize(xyb,[n-1,2])
        self.npin = n-1
        #print 'len(xyr):',len(xyr)
        #return xyr, xyb

    def transform(self,im):
        """
          Return the transform of the blue frame to the reference
          coordinate system of the red.
        """
        cref = self.gmo.fitObj
        def gfunc(out):
            xref = out[1]
            yref = out[0]
            x = cref.a + cref.b*xref + cref.c*yref
            y = cref.d + cref.e*xref + cref.f*yref
            return y,x 

        return nd.geometric_transform(im, gfunc)
        #return self.func(im)

    def gtransform(self,im):
        """
        #xn = a + b*x + c*y
        #yn = d + e*x + f*y
        #print 'out wOld'
        #def gfunc(out):
        #    xref=out[1]
        #    yref=out[0]
        #    
        #    # this is the cannonical solution (iraf.geomap)
        #    x = 990.5897 - 0.9985259*xref - 0.0178984*yref
        #    y = 37.82109 - 0.0181331*xref + 0.9991414*yref
        #    # newer iraf.geomap
        #    #x = 990.2734 - 0.9979063*xref - 0.0186165*yref
        #    #y = 37.68222 - 0.01741749*xref + 0.9985356*yref
        #
        #    return y,x 
        """ 
        def gfunc(out):
            xref = out[1]
            yref = out[0]
            x = 990.5897-0.9985259*xref - 0.0178984*yref
            y = 37.82109 -0.0181331*xref + 0.9991414*yref
            return y,x 


        return nd.geometric_transform(im, gfunc)

