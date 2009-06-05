import numpy as np

def robust_sigma(y,zero=None):
    """
    ;FUNCTION  ROBUST_SIGMA,Y, ZERO=REF
    ;
    ;
    ;+
    ; NAME:
    ;	ROBUST_SIGMA  
    ;
    ; PURPOSE:
    ;	Calculate a resistant estimate of the dispersion of a distribution.
    ; EXPLANATION:
    ;	For an uncontaminated distribution, this is identical to the standard
    ;	deviation.
    ;
    ; CALLING SEQUENCE:
    ;	result = ROBUST_SIGMA( Y, [ /ZERO ] )
    ;
    ; INPUT: 
    ;	Y = Vector of quantity for which the dispersion is to be calculated
    ;
    ; OPTIONAL INPUT KEYWORD:
    ;	/ZERO - if set, the dispersion is calculated w.r.t. 0.0 rather than the
    ;		central value of the vector. If Y is a vector of residuals, this
    ;		should be set.
    ;
    ; OUTPUT:
    ;	ROBUST_SIGMA returns the dispersion. In case of failure, returns 
    ;	value of -1.0
    ;
    ; PROCEDURE:
    ;	Use the median absolute deviation as the initial estimate, then weight 
    ;	points using Tukey's Biweight. See, for example, "Understanding Robust
    ;	and Exploratory Data Analysis," by Hoaglin, Mosteller and Tukey, John
    ;	Wiley & Sons, 1983.
    ;
    ; REVSION HISTORY: 
    ;	H. Freudenreich, STX, 8/90
    ;       Replace MED() call with MEDIAN(/EVEN)  W. Landsman   December 2001
      To Python  NZ: 9,2008
    ;
    ;-
    """

    eps = 1.0E-20
    if  (zero != None):
        y0= 0.0
    else:
        y0 = np.median(y,axis=None)

    # First, the median absolute deviation MAD about the median:
    
    mad = np.median( abs(y-y0),axis=None)/0.6745

    # If the MAD=0, try the MEAN absolute deviation:
    if mad < eps: mad = np.average( abs(y-y0) )/.80
    if mad < eps: 
       return 0.0
 

    # Now the biweighted value:
    u   = (y-y0)/(6.*mad)
    uu  = u*u
    q   = np.where(uu <= 1.0)

    if np.size(q) < 3:
       print 'robust_sigma: tHIS DISTRIBUTION IS too weird! rETURNING -1'
       siggma = -1.
       return siggma

    uq=(1-uu[q])
    yq=(y[q]-y0)
    
    arg = yq*yq * uq*uq*uq*uq
    numerator = np.sum(arg)
    n     = np.size(y)

    uuq=uu[q]
    arg=uq * (1.0 - 5*uuq)
    den1  = np.sum( arg )
    siggma = n*numerator/(den1*(den1-1.))
 
    if siggma > 0.0: return np.sqrt(siggma) 
    else: return 0.0

