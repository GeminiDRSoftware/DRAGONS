import numpy as np

def gcentroid(img,x,y,fwhm=5,maxgood=None,keepcenter=None):
    """
    Function from IDL routine gcntrd (astrolib):
    http://idlastro.gsfc.nasa.gov/contents.html
    http://idlastro.gsfc.nasa.gov/ftp/pro/idlphot/gcntrd.pro

    Need to run unit test to check if IDL output is the same for a given input data...!!!!!!!!!! 

;+
;  NAME: 
;       GCNTRD
;  PURPOSE:
;       Compute the stellar centroid by Gaussian fits to marginal X,Y, sums 
; EXPLANATION:
;       GCNTRD uses the DAOPHOT "FIND" centroid algorithm by fitting Gaussians
;       to the marginal X,Y distributions.     User can specify bad pixels 
;       (either by using the MAXGOOD keyword or setting them to NaN) to be
;       ignored in the fit.    Pixel values are weighted toward the center to
;       avoid contamination by neighboring stars. 
;
;  CALLING SEQUENCE: 
;       GCNTRD, img, x, y, xcen, ycen, [ fwhm , /SILENT, /DEBUG, MAXGOOD = ,
;                            /KEEPCENTER ]
;
;  INPUTS:     
;       IMG - Two dimensional image array
;       X,Y - Scalar or vector integers giving approximate stellar center
;
;  OPTIONAL INPUT:
;       FWHM - floating scalar; Centroid is computed using a box of half
;               width equal to 1.5 sigma = 0.637* FWHM.  GCNTRD will prompt
;               for FWHM if not supplied
;
;  OUTPUTS:   
;       XCEN - the computed X centroid position, same number of points as X
;       YCEN - computed Y centroid position, same number of points as Y
;
;       Values for XCEN and YCEN will not be computed if the computed
;       centroid falls outside of the box, or if there are too many bad pixels,
;       or if the best-fit Gaussian has a negative height.   If the centroid 
;       cannot be computed, then a  message is displayed (unless /SILENT is 
;       set) and XCEN and YCEN are set to -1.
;
;  OPTIONAL OUTPUT KEYWORDS:
;       MAXGOOD=  Only pixels with values less than MAXGOOD are used to in
;               Gaussian fits to determine the centroid.    For non-integer
;               data, one can also flag bad pixels using NaN values.
;       /SILENT - Normally GCNTRD prints an error message if it is unable
;               to compute the centroid.   Set /SILENT to suppress this.
;       /DEBUG - If this keyword is set, then GCNTRD will display the subarray
;               it is using to compute the centroid.
;       /KeepCenter  By default, GCNTRD finds the maximum pixel in a box 
;              centered on the input X,Y coordinates, and then extracts a new
;              box about this maximum pixel.   Set the /KeepCenter keyword  
;              to skip the step of finding the maximum pixel, and instead use
;              a box centered on the input X,Y coordinates.                          
;  PROCEDURE: 
;       Maximum pixel within distance from input pixel X, Y  determined 
;       from FHWM is found and used as the center of a square, within 
;       which the centroid is computed as the Gaussian least-squares fit
;       to the  marginal sums in the X and Y directions. 
;
;  EXAMPLE:
;       Find the centroid of a star in an image im, with approximate center
;       631, 48.    Assume that bad (saturated) pixels have a value of 4096 or
;       or higher, and that the approximate FWHM is 3 pixels.
;
;       IDL> GCNTRD, IM, 631, 48, XCEN, YCEN, 3, MAXGOOD = 4096       
;  MODIFICATION HISTORY:
;       Written June 2004, W. Landsman  following algorithm used by P. Stetson 
;             in DAOPHOT2.
;-      
       Translated to Python by Sergio Fernandez.
       Modified by NZ Gemini July 2008
    
    """
    
    #Need to run unit test to check if IDL output is the same for a given input data...!!!!!!!!!!
    
    from numpy import fix,round,zeros,arange,exp,transpose,where,mod,float
    from numpy import float64,array,isnan,size,around, isfinite, nansum, nanmax
    from numpy import ones, ravel
    import numpy as np
     
    #print 'Star..ting Gcentroid..'
    sz_image=img.shape
    if len(sz_image) != 2:
        error=True
        print 'Image array (first parameter) must be 2 dimensional!'
        return -1,-1
    xsize=sz_image[0]  # x, y means dimension 0 and dimension 1 for python,
		       # not standard axes!!
    ysize=sz_image[1]
    npts=size(x)
    maxbox=13
    radius=max(0.637*fwhm, 2.001)
    radsq=radius**2
    sigsq=(fwhm/2.35482)**2
    nhalf=int(min(fix(radius),(maxbox - 1)/2))
    nbox=2*nhalf+1     # number of pixels in side of convolution box

    # Turn into a float array of min size 1
    xcen=array(x,dtype=float,ndmin=1)
    ycen=array(y,dtype=float,ndmin=1)

    # Create arrays of size npts
    ix=zeros(npts); iy=zeros(npts)
    if (npts == 1):
        ix[0]=int(round(x))      # central x pixel
        iy[0]=int(round(y))      # central y pixel
    else:
        ix=[int(i) for i in around(x)]
        iy=[int(i) for i in around(y)]

    #Create the Gaussian convolution kernel in variable "g"
    g=zeros((nbox,nbox),dtype=float)
    row2=(arange(float(nbox)) - nhalf)**2
    g[nhalf] = row2
    for i in range(1,nhalf+1):
        temp=row2+i**2
        g[nhalf-i]=temp
        g[nhalf+i]=temp

    g = exp(-0.5*g/sigsq)	#Make c into a Gaussian kernel


# In fitting Gaussians to the marginal sums, pixels will arbitrarily be 
# assigned weights ranging from unity at the corners of the box to 
# NHALF^2 at the center (e.g. if NBOX = 5 or 7, the weights will be
#
#                                 1   2   3   4   3   2   1
#      1   2   3   2   1          2   4   6   8   6   4   2
#      2   4   6   4   2          3   6   9  12   9   6   3
#      3   6   9   6   3          4   8  12  16  12   8   4
#      2   4   6   4   2          3   6   9  12   9   6   3
#      1   2   3   2   1          2   4   6   8   6   4   2
#                                 1   2   3   4   3   2   1
#
# respectively).  This is done to desensitize the derived parameters to 
# possible neighboring, brighter stars.

    #print nbox,arange(10-nhalf,10+nhalf).shape
    x_wt = zeros((nbox,nbox),dtype=float)
    wt = nhalf - abs(arange(float(nbox)) - nhalf) + 1
    for i in range(nbox): 
        x_wt[i] = wt
    y_wt = transpose(x_wt)
    pos= str(x) + ' ' + str(y)

    for i in range(npts):
        if (keepcenter is None):
            if ((ix[i] < nhalf) or ((ix[i]+nhalf) > xsize-1)) or \
               ((iy[i] < nhalf) or ((ix[i]+nhalf) > xsize-1)):
                    print 'WARNING:: position'+pos+' too near edge of image'
                    xcen[i] = -1 ; ycen[i] = -1
		    break

            #OJO!!!!  [y,x]
            d=img[iy[i]-nhalf:iy[i]+nhalf +1,ix[i]-nhalf:ix[i]+nhalf + 1] 
            if  (maxgood is not None):
                if array(maxgood, copy=0).size > 0:
                   ig = where(ravel(d < maxgood))[0]
                   mx = nanmax(d[ig])
            mx = nanmax(ravel(d))       #Maximum pixel value in BIGBOX            
            #How many pixels have maximum value?
            mx_pos = where(ravel(d == mx))[0] # [0] because python returns
                                              # (array([...]),)
            idx= mx_pos % nbox      # X coordinate of Max pixel
            idy= mx_pos / nbox      # Y coordinate of Max pixel

            Nmax=size(mx_pos)
            if Nmax > 1:             # More than 1 pixel at maximum?
                idx=round(idx.sum()/Nmax)
                idy=round(idy.sum()/Nmax)
            else:
                idx=idx[0]
                idy=idy[0]

            xmax = ix[i] - (nhalf) + idx  # X coordinate in original image array
            ymax = iy[i] - (nhalf) + idy  # Y coordinate in original image array
        else:
            xmax = ix[i]
            ymax = iy[i]

        #---------------------------------------------------------------------
        # check *new* center location for range
        # added by Hogg

        if (xmax < nhalf) or ((xmax + nhalf) > (xsize-1)) \
                or (ymax < nhalf) or ((ymax + nhalf) > ysize-1):
                print xmax,nhalf,xsize,ymax,ysize
                print 'position moved too near edge of image'
                #xcen[i] = -1 ; ycen[i] = -1
                return  -1 , -1              

        # Extract  subimage centered on maximum pixel
        ym = round(ymax)
        xm = round(xmax)

        d = img[ym-nhalf : ym+nhalf+1 , xm-nhalf : xm+nhalf+1]

        if maxgood:
            mask=(d<maxgood)*1.     
        else:
            stype= str(img.dtype)
            if ('int' in stype) or ('float' in stype):
                mask = isfinite(d)*1
                mask[-isnan(mask)]=1.
                mask[isnan(mask)]=0.
            else:
                mask = ones([nbox, nbox],dtype=int)

        maskx = (mask.sum(1) > 0)*1
        masky = (mask.sum(0) > 0)*1

        # At least 3 points are needed in the partial sum 
        # to compute the Gaussian

        if maskx.sum() <3 or masky.sum() < 3:
            print 'position has insufficient good points'
            xcen[i]=-1; ycen[i]=-1
            return xcen, ycen

        ywt = y_wt*mask
        xwt = x_wt*mask
        wt1 = wt*maskx
        wt2 = wt*masky

        sd = nansum(d*ywt,0)
        sg = (g*ywt).sum(0)
        sumg = (wt1*sg).sum()
        sumgsq = (wt1*sg*sg).sum()

        sumgd = (wt1*sg*sd).sum()
        sumgx = (wt1*sg).sum()
        sumd = (wt1*sd).sum()
        p = wt1.sum()
        xvec = nhalf - arange(float(nbox)) 
        dgdx = sg*xvec
        sdgdxs = (wt1*dgdx**2).sum()
        sdgdx = (wt1*dgdx).sum()
        sddgdx = (wt1*sd*dgdx).sum()
        sgdgdx = (wt1*sg*dgdx).sum()

        hx = (sumgd - sumg*sumd/p) / (sumgsq - sumg**2/p)

        # HX is the height of the best-fitting marginal Gaussian.
        # If this is not positive then the centroid does not 
        # make sense

        if hx <= 0:
            print '*** Warning: (hx <=0)position cannot be fit by a gaussian',hx
            xcen[i]=-1; ycen[i]=-1
            return xcen,ycen

        skylvl = (sumd - hx*sumg)/p
        dx = (sgdgdx - (sddgdx-sdgdx*(hx*sumg + skylvl*p))) \
                             /(hx*sdgdxs/sigsq)

        #X centroid in original array
        xcen[i] = xmax + dx/(1+abs(dx)) 

        # Now repeat computation for Y centroid

        #sd = (d*xwt).sum(1)
        sd = nansum(d*xwt,1)
        sg = (g*xwt).sum(1)
        sumg = (wt2*sg).sum()
        sumgsq = (wt2*sg*sg).sum()
                               
        sumgd = (wt2*sg*sd).sum()
        sumd = (wt2*sd).sum()
        p = (wt2).sum()
                                
        yvec = nhalf - arange(float(nbox))
        dgdy = sg*yvec
        sdgdys = (wt2*dgdy**2).sum()
        sdgdy = (wt2*dgdy).sum()
        sddgdy = (wt2*sd*dgdy).sum()
        sgdgdy = (wt2*sg*dgdy).sum()
         
        hy = (sumgd - sumg*sumd/p) / (sumgsq - sumg**2/p)

        if (hy <= 0):
            print '*** Warning (hy <=0) position cannot be fit by a gaussian',hy
            xcen[i]=-1; ycen[i]=-1
            return xcen,ycen

        skylvl = (sumd - hy*sumg)/p
        dy = (sgdgdy-(sddgdy-sdgdy*(hy*sumg + skylvl*p)))/(hy*sdgdys/sigsq)
        ycen[i] = ymax + dy/(1+abs(dy))    #X centroid in original array

         #DONE
         #endfor
    return xcen,ycen
