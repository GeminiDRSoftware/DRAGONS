from numpy import arange,empty,sqrt,size

#pro dist_circle ,im, n, xcen ,ycen, DOUBLE = double 
#+
# NAME: 
#      DIST_CIRCLE
# PURPOSE:      
#      Form a square array where each value is its distance to a given center.
# EXPLANATION:
#      Returns a square array in which the value of each element is its 
#      distance to a specified center. Useful for circular aperture photometry.
#
# CALLING SEQUENCE:
#      IM = DIST_CIRCLE, N, [ XCEN, YCEN ]
#
# INPUTS:
#      N = either  a scalar specifying the size of the N x N square output
#               array, or a 2 element vector specifying the size of the
#               N x M rectangular output array.
#
# OPTIONAL INPUTS:
#      XCEN,YCEN = Scalars designating the X,Y pixel center.  These need
#               not be integers, and need not be located within the
#               output image.   If not supplied then the center of the output
#               image is used (XCEN = YCEN = (N-1)/2.).
#
# OUTPUTS:
#       IM  - N by N (or M x N) floating array in which the value of each 
#               pixel is equal to its distance to XCEN,YCEN
#
# OPTIONAL INPUT KEYWORD:
#       /DOUBLE - If this keyword is set and nonzero, the output array will
#               be of type DOUBLE rather than floating point.
#
# EXAMPLE:
#       Total the flux in a circular aperture within 3' of a specified RA
#       and DEC on an 512 x 512 image IM, with a header H.
#
#       IDL> adxy, H, RA, DEC, x, y       ;Convert RA and DEC to X,Y
#       IDL> getrot, H, rot, cdelt        ;CDELT gives plate scale deg/pixel
#       IDL> cdelt = cdelt*3600.          ;Convert to arc sec/pixel
#       IDL> dist_circle, circle, 512, x, y  ;Create a distance circle image
#       IDL> circle = circle*abs(cdelt[0])   ;Distances now given in arcseconds
#       IDL> good = where(circle LT 180)  ;Within 3 arc minutes
#       IDL> print,total( IM[good] )      ;Total pixel values within 3'
#
# RESTRICTIONS:
#       The speed of DIST_CIRCLE decreases and the the demands on virtual
#       increase as the square of the output dimensions.   Users should
#       dimension the output array as small as possible, and re-use the
#       array rather than re-calling DIST_CIRCLE
#
# MODIFICATION HISTORY:
#       Adapted from DIST    W. Landsman            March 1991
#       Allow a rectangular output array   W. Landsman     June 1994
#       Converted to IDL V5.0   W. Landsman   September 1997
#       Add /DOUBLE keyword, make XCEN,YCEN optional  W. Landsman Jun 1998
#-
def dist_circle(n, xcen=-99 ,ycen=-99):

    """
       DIST_CIRCLE, n,[ xcen, ycen, /DOUBLE ]' 
       'IM - output image array'
       'N - size of the output image array, scalar or 2 element vector'
       'XCEN,YCEN - position from which to specify distances'
       Returns the array of distances.
    """

#if N_elements(N) EQ 2 then begin
    sz= size(n)
    if (sz == 2):
       nx = n[0]
       ny = n[1] 
#endif else if N_elements(N) EQ 1 then begin
    elif (sz == 1):
       ny = n
       nx = n                    #Make a row
#endif else message, $
    else:
       print \
       'ERROR - Output size parameter N must contain 1 or 2 elements'
       return


#if N_params() LT 4 then begin
    if (xcen == -99):       
       #xcen = (nx-1)/2.    # this was incorrect
       xcen = nx/2. - 1 
    if (ycen == -99):       
       ycen = ny/2. - 1


    x_2 = (arange(nx) - xcen) ** 2     #X distances (squared)
    y_2 = (arange(ny) - ycen) ** 2     #Y distances (squared)  
    im = empty((nx, ny))      #Make uninitialized output array

    for i in xrange(ny): 
       im[i,:] = sqrt(x_2 + y_2[i])     #Euclidian distance

    return im
