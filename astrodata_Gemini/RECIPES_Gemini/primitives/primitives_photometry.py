import os
import sys
import re
import subprocess
import numpy as np
import pyfits as pf
import pywcs
from astrodata import AstroData
from astrodata import Errors
from astrodata import Lookups
from astrodata.adutils import gemLog
from astrodata.ConfigSpace import lookup_path
from gempy import astrotools as at
from gempy import geminiTools as gt
from primitives_GENERAL import GENERALPrimitives

# Define the earliest acceptable SExtractor version
# Currently: 2.8.6
SEXTRACTOR_VERSION = [2,8,6]

class PhotometryPrimitives(GENERALPrimitives):
    """
    This is the class containing all of the photometry primitives for
    the GEMINI level of the type hierarchy tree. It inherits all the
    primitives from the level above, 'GENERALPrimitives'.
    """
    astrotype = "GEMINI"
    
    def init(self, rc):
        GENERALPrimitives.init(self, rc)
        return rc
    init.pt_hide = True
    
    def addReferenceCatalog(self, rc):
        """
        Currently, the only supported source catalog is sdss7.

        Query a vizier server hosting an sdss7 catalog to get a
        catalog of all the SDSS DR7 sources within a given radius
        of the pointing center.

        Append the catalog as a FITS table with extenstion name
        'REFCAT', containing the following columns:

        - 'Id'       : Unique ID. Simple running number
        - 'Name'     : SDSS catalog source name
        - 'RAJ2000'  : RA as J2000 decimal degrees
        - 'DEJ2000'  : Dec as J2000 decimal degrees
        - 'umag'     : SDSS u band magnitude
        - 'e_umag'   : SDSS u band magnitude error estimage
        - 'gmag'     : SDSS g band magnitude
        - 'e_gmag'   : SDSS g band magnitude error estimage
        - 'rmag'     : SDSS r band magnitude
        - 'e_rmag'   : SDSS r band magnitude error estimage
        - 'imag'     : SDSS i band magnitude
        - 'e_imag'   : SDSS i band magnitude error estimage
        - 'zmag'     : SDSS z band magnitude
        - 'e_zmag'   : SDSS z band magnitude error estimage

        :param source: Source catalog to query. This used as the catalog
                       name on the vizier server
        :type source: string

        :param radius: The radius of the cone to query in the catalog, 
                       in degrees. Default is 4 arcmin
        :type radius: float
        """

        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])

        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "addReferenceCatalog", "starting"))

        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["addReferenceCatalog"]

        # Initialize the list of output AstroData objects
        adoutput_list = []

        # Get the necessary parameters from the RC
        source = rc["source"]
        radius = rc["radius"]

        # Get the Vizier server URL
        url = Lookups.get_lookup_table(
            "Gemini/refcat_dict", "refcat_dict")['VIZIER']

        # Add the source catalog specifier to the URL
        url += "-source=%s&" % source

        # Loop over each input AstroData object in the input list
        problem = False
        adinput = rc.get_inputs_as_astrodata()
        for ad in adinput:

            # Loop through the science extensions
            for sciext in ad['SCI']:
                extver = sciext.extver()

                ra = sciext.ra().as_pytype()
                dec = sciext.dec().as_pytype()

                # Query the vizier server, get the votable
                # Catch and ignore the warning about DEFINITIONS element being deprecated in VOTable 1.1
                log.fullinfo("Calling Vizier at %s" % url)
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        import vo.conesearch
                        import vo.table
                        table = vo.conesearch.conesearch(catalog_db=url, ra=ra, dec=dec, sr=radius, pedantic=False, verb=2, verbose=False)
                    except:
                        log.critical("Problem importing the vo module. This isn't going to work")
                        adoutput_list = adinput
                        problem = True
                        break
                        
                if len(table.array)==0:
                    log.stdinfo("No reference catalog sources found "\
                                "for %s['SCI',%d]" % (ad.filename, extver))
                    continue
                else:
                    log.stdinfo("Found %d reference catalog sources for %s['SCI',%d]" % (len(table.array), ad.filename, extver))

                # Did we get anything?
                if(len(table.array)):
                    # Parse the votable that we got back, into arrays for each column.
                    sdssname = table.array['SDSS']
                    umag = table.array['umag']
                    e_umag = table.array['e_umag']
                    gmag = table.array['gmag']
                    e_gmag = table.array['e_gmag']
                    rmag = table.array['rmag']
                    e_rmag = table.array['e_rmag']
                    imag = table.array['imag']
                    e_imag = table.array['e_imag']
                    zmag = table.array['zmag']
                    e_zmag = table.array['e_zmag']
                    ra = table.array['RAJ2000']
                    dec = table.array['DEJ2000']

                    # Create a running id number
                    refid=range(1, len(sdssname)+1)

                    # Make the pyfits columns and table
                    c1 = pf.Column(name="Id",format="J",array=refid)
                    c2 = pf.Column(name="Name", format="24A", array=sdssname)
                    c3 = pf.Column(name="RAJ2000",format="D",unit="deg",array=ra)
                    c4 = pf.Column(name="DEJ2000",format="D",unit="deg",array=dec)
                    c5 = pf.Column(name="umag",format="E",array=umag)
                    c6 = pf.Column(name="e_umag",format="E",array=e_umag)
                    c7 = pf.Column(name="gmag",format="E",array=gmag)
                    c8 = pf.Column(name="e_gmag",format="E",array=e_gmag)
                    c9 = pf.Column(name="rmag",format="E",array=rmag)
                    c10 = pf.Column(name="e_rmag",format="E",array=e_rmag)
                    c11 = pf.Column(name="imag",format="E",array=imag)
                    c12 = pf.Column(name="e_imag",format="E",array=e_imag)
                    c13 = pf.Column(name="zmag",format="E",array=zmag)
                    c14 = pf.Column(name="e_zmag",format="E",array=e_zmag)
                    col_def = pf.ColDefs([c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14])
                    tb_hdu = pf.new_table(col_def)

                    # Add comments to the REFCAT header to describe it.
                    tb_hdu.header.add_comment('Source catalog derived from the %s catalog on vizier' % table.name)
                    tb_hdu.header.add_comment('Vizier Server queried: %s' % url)
                    for fieldname in ('RAJ2000', 'DEJ2000', 'umag', 'e_umag', 'gmag', 'e_gmag', 'rmag', 'e_rmag', 'imag', 'e_imag', 'zmag', 'e_zmag'):
                        tb_hdu.header.add_comment('UCD for field %s is: %s' % (fieldname, table.get_field_by_id(fieldname).ucd))

                    tb_ad = AstroData(tb_hdu)
                    tb_ad.rename_ext('REFCAT', extver)

                    if(ad['REFCAT',extver]):
                        log.fullinfo("Replacing existing REFCAT in %s" % ad.filename)
                        ad.remove(('REFCAT', extver))
                    else:
                        log.fullinfo("Adding REFCAT to %s" % ad.filename)
                    ad.append(tb_ad)

            # If there was a problem found (ie. vo module couldn't
            # be imported), break the outer loop
            if problem:
                break

            # Match the object catalog against the reference catalog
            # Update the refid and refmag columns in the object catalog
            ad = _match_objcat_refcat(adinput=ad)[0]

            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, keyword=timestamp_key)

            # Change the filename
            ad.filename = gt.fileNameUpdater(adIn=ad, suffix=rc["suffix"],
                                             strip=True)

            # Append the output AstroData object to the list 
            # of output AstroData objects
            adoutput_list.append(ad)

        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)

        yield rc

    def detectSources(self, rc):
        """
        Find x,y positions of all the objects in the input image. Append 
        a FITS table extension with position information plus columns for
        standard objects to be updated with position from addReferenceCatalog
        (if any are found for the field).
    
        :param method: source detection algorithm to use
        :type method: string; options are 'daofind','sextractor'

        :param centroid_function: Function for centroid fitting with daofind
        :type centroid_function: string, can be: 'moffat','gauss'
                                 Default: 'moffat'

        :param sigma: The mean of the background value for daofind. If nothing
                      is passed, it will be automatically determined
        :type sigma: float
        
        :param threshold: Threshold intensity for a point source for daofind;
                      should generally be at least 3 or 4 sigma above
                      background RMS.
        :type threshold: float
        
        :param fwhm: FWHM to be used in the convolve filter for daofind. This
                     ends up playing a factor in determining the size of the
                     kernel put through the gaussian convolve.
        :type fwhm: float
        """
 
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "detectSources", "starting"))
        
        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["detectSources"]

        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():

            # Get the necessary parameters from the RC
            sigma = rc["sigma"]
            threshold = rc["threshold"]
            fwhm = rc["fwhm"]
            max_sources = rc["max_sources"]
            centroid_function = rc["centroid_function"]
            method = rc["method"]

            seeing_est = ad.phu_get_key_value("MEANFWHM")
            for sciext in ad["SCI"]:
                
                extver = sciext.extver()
                
                # Check source detection method
                if method not in ["sextractor","daofind"]:
                    raise Errors.InputError("Source detection method "+
                                            method+" is unsupported.")
                
                if method=="sextractor":
                    dqext = ad["DQ",extver]

                    # Check sextractor version, go to daofind if task not
                    # found or wrong version
                    right_version = _test_sextractor_version()

                    if not right_version:
                        log.warning("SExtractor version %d.%d.%d or later "\
                                    "not found. Setting method=daofind" %
                                    tuple(SEXTRACTOR_VERSION))
                        method="daofind"
                    else:
                        try:
                            columns,seeing_est = _sextractor(
                                sciext=sciext, dqext=dqext,
                                seeing_estimate=seeing_est)
                        except Errors.ScienceError:
                            log.warning("SExtractor failed. "\
                                        "Setting method=daofind")
                            method="daofind"
                        else:
                            nobj = len(columns["NUMBER"].array)
                            if nobj==0:
                                log.stdinfo("No sources found in "\
                                            "%s['SCI',%d]" %
                                            (ad.filename,extver))
                                continue
                            else:
                                log.stdinfo("Found %d sources in "\
                                            "%s['SCI',%d]" %
                                            (nobj,ad.filename,extver))

                if method=="daofind":
                    pixscale = sciext.pixel_scale()
                    if pixscale is None:
                        log.warning("%s does not have a pixel scale, " \
                                    "using 1.0 arcsec/pix" % ad.filename)
                        pixscale = 1.0

                    if fwhm is None:
                        if seeing_est is not None:
                            fwhm = seeing_est / pixscale
                        else:
                            fwhm = 0.8 / pixscale

                    obj_list = _daofind(sciext=sciext, sigma=sigma,
                                        threshold=threshold, fwhm=fwhm)

                    nobj = len(obj_list)
                    if nobj==0:
                        log.stdinfo("No sources found in %s['SCI',%d]" %
                                    (ad.filename,extver))
                        continue
                    else:
                        log.stdinfo("Found %d sources in %s['SCI',%d]" %
                                    (nobj,ad.filename,extver))

                    # Separate pixel coordinates into x, y lists
                    obj_x,obj_y = [np.asarray(obj_list)[:,k] for k in [0,1]]
                
                    # Use WCS to convert pixel coordinates to RA/Dec
                    wcs = pywcs.WCS(sciext.header)
                    obj_ra, obj_dec = wcs.wcs_pix2sky(obj_x,obj_y,1)
                
                    # Define pyfits columns to pass to add_objcat
                    columns = {
                        "X_IMAGE":pf.Column(name="X_IMAGE",format="E",
                                            array=obj_x),
                        "Y_IMAGE":pf.Column(name="Y_IMAGE",format="E",
                                            array=obj_y),
                        "X_WORLD":pf.Column(name="X_WORLD",format="E",
                                            array=obj_ra),
                        "Y_WORLD":pf.Column(name="Y_WORLD",format="E",
                                            array=obj_dec),
                        }

                
                # For either method, add OBJCAT
                ad = gt.add_objcat(adinput=ad, extver=extver, 
                                   replace=True, columns=columns)[0]

            # In daofind case, do some simple photometry on all
            # extensions to get fwhm, ellipticity
            if method=="daofind":
                log.stdinfo("Fitting sources for simple photometry")
                if seeing_est is None:
                    # Run the fit once to get a rough seeing estimate 
                    if max_sources>20:
                        tmp_max=20
                    else:
                        tmp_max=max_sources
                    junk,seeing_est = _fit_sources(
                        ad,ext=1,max_sources=tmp_max,threshold=threshold,
                        centroid_function=centroid_function,
                        seeing_estimate=None)
                ad,seeing_est = _fit_sources(
                    ad,max_sources=max_sources,threshold=threshold,
                    centroid_function=centroid_function,
                    seeing_estimate=seeing_est)

            
            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, keyword=timestamp_key)

            # Change the filename
            ad.filename = gt.fileNameUpdater(adIn=ad, suffix=rc["suffix"], 
                                             strip=True)
            
            # Append the output AstroData object to the list 
            # of output AstroData objects
            adoutput_list.append(ad)
        
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc

##############################################################################
# Below are the helper functions for the user level functions in this module #
##############################################################################

def _match_objcat_refcat(adinput=None):
    """
    Match the sources in the objcats against those in the corresponding
    refcat. Update the refid column in the objcat with the Id of the 
    catalog entry in the refcat. Update the refmag column in the objcat
    with the magnitude of the corresponding source in the refcat in the
    band that the image is taken in.

    :param adinput: AD object(s) to match catalogs in
    :type adinput: AstroData objects, either a single instance or a list
    """

    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = gemLog.getGeminiLog()

    # The validate_input function ensures that adinput is not None and returns
    # a list containing one or more AstroData objects
    adinput = gt.validate_input(adinput=adinput)

    # Initialize the list of output AstroData objects
    adoutput_list = []
    try:

        # Loop over each input AstroData object in the input list
        for ad in adinput:
            filter_name = ad.filter_name(pretty=True).as_pytype()
            if filter_name in ['u', 'g', 'r', 'i', 'z']:
                magcolname = filter_name+'mag'
                magerrcolname = 'e_'+filter_name+'mag'
            else:
                log.warning("Filter %s is not in SDSS - will not be able to flux calibrate" % filter_name)
                magcolname = None
                magerrcolname = None

            # Loop through the objcat extensions
            if ad['OBJCAT'] is None:
                raise Errors.InputError("Missing OBJCAT in %s" % (ad.filename))
            for objcat in ad['OBJCAT']:
                extver = objcat.extver()

                # Check that a refcat exists for this objcat extver
                refcat = ad['REFCAT',extver]
                if(not(refcat)):
                    log.warning("Missing [REFCAT,%d] in %s" % (extver,ad.filename))
                    log.warning("Cannot match objcat against missing refcat")
                else:
                    # Get the x and y position lists from both catalogs
                    xx = objcat.data['X_WORLD']
                    yy = objcat.data['Y_WORLD']
                    sx = refcat.data['RAJ2000']
                    sy = refcat.data['DEJ2000']

                    # FIXME - need to address the wraparound problem here
                    # if we straddle ra = 360.00 = 0.00

                    initial = 15.0/3600.0 # 15 arcseconds in degrees
                    final = 0.5/3600.0 # 0.5 arcseconds in degrees

                    (oi, ri) = at.match_cxy(xx,sx,yy,sy, firstPass=initial, delta=final, log=log)

                    log.stdinfo("Matched %d objects in ['OBJCAT',%d] against ['REFCAT',%d]" % (len(oi), extver, extver))

                    # Loop through the reference list updating the refid in the objcat
                    # and the refmag, if we can
                    for i in range(len(oi)):
                        objcat.data['REF_NUMBER'][oi[i]] = refcat.data['Id'][ri[i]]
                        if(magcolname):
                            objcat.data['REF_MAG'][oi[i]] = refcat.data[magcolname][ri[i]]
                            objcat.data['REF_MAG_ERR'][oi[i]] = refcat.data[magerrcolname][ri[i]]


            adoutput_list.append(ad)

        return adoutput_list
    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise



def _daofind(sciext=None, sigma=None, threshold=2.5, fwhm=5.5, 
             sharplim=[0.2,1.0], roundlim=[-1.0,1.0], window=None,
             grid=False, rejection=None, ratio=None):
    """
    Performs similar to the source detecting algorithm 
    'http://idlastro.gsfc.nasa.gov/ftp/pro/idlphot/find.pro'.
    
    References:
        This code is heavily influenced by 
        'http://idlastro.gsfc.nasa.gov/ftp/pro/idlphot/find.pro'.
        'find.pro' was written by W. Landsman, STX February, 1987.
        
        This code was converted to Python with areas re-written for 
        optimization by:
        River Allen, Gemini Observatory, December 2009. riverallen@gmail.com
        
        Updated by N. Zarate and M. Clarke for incorporation into gempy
        package, February 2011 and June 2011.
    """
    
    # import a few things only required by this helper function
    import time
    from convolve import convolve2d
    
    log = gemLog.getGeminiLog()
    
    if not sciext:
        raise Errors.InputError("_daofind requires a science extension.")
    else:
        sciData = sciext.data
        
    if window is not None:
        if type(window) == tuple:
            window = [window]
        elif type(window) == list:
            pass
        else:
            raise Errors.InputError("'window' must be a tuple of length 4, " +
                                    "or a list of tuples length 4.")
            
        for wind in window:
            if type(wind) == tuple:
                if len(wind) == 4:
                    continue
                else:
                    raise Errors.InputError("A window tuple has incorrect " +
                                            "information, %s, require x,y," +
                                            "width,height" %(str(wind)))
            else:
                raise Errors.InputError("The window list contains a " +
                                        "non-tuple. %s" %(str(wind)))
            
    if len(sharplim) < 2:
        raise Errors.InputError("Sharplim parameter requires 2 num elements. "+
                                "(i.e. [0.2,1.0])")
    if len(roundlim) < 2:
        raise Errors.InputError("Roundlim parameter requires 2 num elements. "+
                                "(i.e. [-1.0,1.0])")
             
    # Setup
    # -----
    
    ost = time.time()
    #Maximum size of convolution box in pixels 
    maxConvSize = 13
    
    #Radius is 1.5 sigma
    radius = np.maximum(0.637 * fwhm, 2.001)
    radiusSQ = radius ** 2
    kernelHalfDimension = np.minimum(np.array(radius, copy=0).astype(np.int32), 
                                  (maxConvSize - 1) / 2)
    # Dimension of the kernel or "convolution box"
    kernelDimension = 2 * kernelHalfDimension + 1 
    
    sigSQ = (fwhm / 2.35482) ** 2
    
    # Mask identifies valid pixels in convolution box 
    mask = np.zeros([kernelDimension, kernelDimension], np.int8)
    # g will contain Gaussian convolution kernel
    gauss = np.zeros([kernelDimension, kernelDimension], np.float32)
    
    row2 = (np.arange(kernelDimension) - kernelHalfDimension) ** 2
    
    for i in np.arange(0, (kernelHalfDimension)+(1)):
        temp = row2 + i ** 2
        gauss[kernelHalfDimension - i] = temp
        gauss[kernelHalfDimension + i] = temp
    
    #MASK is complementary to SKIP in Stetson's Fortran
    mask = np.array(gauss <= radiusSQ, copy=0).astype(np.int32)
    #Value of c are now equal to distance to center
    good = np.where(np.ravel(mask))[0]
    pixels = good.size
    
    # Compute quantities for centroid computations that can be used
    # for all stars
    gauss = np.exp(-0.5 * gauss / sigSQ)
    
    """
     In fitting Gaussians to the marginal sums, pixels will arbitrarily be
     assigned weights ranging from unity at the corners of the box to
     kernelHalfDimension^2 at the center (e.g. if kernelDimension = 5 or 7,
     the weights will be
    
                                     1   2   3   4   3   2   1
          1   2   3   2   1          2   4   6   8   6   4   2
          2   4   6   4   2          3   6   9  12   9   6   3
          3   6   9   6   3          4   8  12  16  12   8   4
          2   4   6   4   2          3   6   9  12   9   6   3
          1   2   3   2   1          2   4   6   8   6   4   2
                                     1   2   3   4   3   2   1
    
     respectively). This is done to desensitize the derived parameters to
     possible neighboring, brighter stars.[1]
    """
    
    xwt = np.zeros([kernelDimension, kernelDimension], np.float32)
    wt = kernelHalfDimension - abs(np.arange(kernelDimension).astype(np.float32)
                                   - kernelHalfDimension) + 1
    for i in np.arange(0, kernelDimension):
        xwt[i] = wt
    
    ywt = np.transpose(xwt)
    sgx = np.sum(gauss * xwt, 1)
    sumOfWt = np.sum(wt)
    
    sgy = np.sum(gauss * ywt, 0)
    sumgx = np.sum(wt * sgy)
    sumgy = np.sum(wt * sgx)
    sumgsqy = np.sum(wt * sgy * sgy)
    sumgsqx = np.sum(wt * sgx * sgx)
    vec = kernelHalfDimension - np.arange(kernelDimension).astype(np.float32)
    
    dgdx = sgy * vec
    dgdy = sgx * vec
    sdgdxs = np.sum(wt * dgdx ** 2)
    sdgdx = np.sum(wt * dgdx)
    sdgdys = np.sum(wt * dgdy ** 2)
    sdgdy = np.sum(wt * dgdy)
    sgdgdx = np.sum(wt * sgy * dgdx)
    sgdgdy = np.sum(wt * sgx * dgdy)
    
    kernel = gauss * mask          #Convolution kernel now in c
    sumc = np.sum(kernel)
    sumcsq = np.sum(kernel ** 2) - (sumc ** 2 / pixels)
    sumc = sumc / pixels
    
    # The reason for the flatten is because IDL and numpy treat 
    # statements like arr[index], where index 
    # is an array, differently. For example, arr.shape = (100,100), 
    # in IDL index=[400], arr[index]
    # would work. In numpy you need to flatten in order to get the 
    # arr[4][0] you want.
    kshape = kernel.shape
    kernel = kernel.flatten()
    kernel[good] = (kernel[good] - sumc) / sumcsq
    kernel.shape = kshape
    
    # Using row2 here is pretty confusing (From IDL code)
    # row2 will be something like: [1   2   3   2   1]
    c1 = np.exp(-.5 * row2 / sigSQ)
    sumc1 = np.sum(c1) / kernelDimension
    sumc1sq = np.sum(c1 ** 2) - sumc1
    c1 = (c1 - sumc1) / sumc1sq
    
    # From now on we exclude the central pixel
    mask[kernelHalfDimension,kernelHalfDimension] = 0
    
    # Reduce the number of valid pixels by 1
    pixels = pixels - 1
    # What this operation looks like:
    # ravel(mask) = [0 0 1 1 1 0 0 0 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 1 ...]
    # where(ravel(mask)) = (array([ 2,  3,  4,  8,  9, 10, 11, 12, 14, ...]),)
    # ("good" identifies position of valid pixels)
    good = np.where(np.ravel(mask))[0]
    
    # x and y coordinate of valid pixels 
    xx = (good % kernelDimension) - kernelHalfDimension
    
    # relative to the center
    yy = np.array(good / kernelDimension, 
                  copy=0).astype(np.int32) - kernelHalfDimension
    
    # Extension and Window / Grid
    # ---------------------------
    xyArray = []
    outputLines = []
    
    # Estimate the background if none provided
    if sigma is None:
        sigma = _estimate_sigma(sciData)
        log.fullinfo("Estimated Background: %.3f" % sigma)

    hmin = sigma * threshold
    
    if window is None:
        # Make the window the entire image
        window = [(0,0,sciData.shape[1],sciData.shape[0])]
    
    if grid:
        ySciDim, xSciDim = sciData.shape
        xgridsize = int(xSciDim / ratio) 
        ygridsize = int(ySciDim / ratio)
        window = []
        for ypos in range(ratio):
            for xpos in range(ratio):
                window.append( (xpos * xgridsize, ypos * ygridsize, 
                                xgridsize, ygridsize) )
    
    if rejection is None:
        rejection = []
        
    windName = 0
    for wind in window:
        windName += 1
        subXYArray = []
        
        ##@@TODO check for negative values, check that dimensions
        #        don't violate overall dimensions.
        yoffset, xoffset, yDimension, xDimension = wind
        
        sciSection = sciData[xoffset:xoffset+xDimension,
                             yoffset:yoffset+yDimension]
        
        # Quickly determine if a window is worth processing
        rejFlag = False
        
        for rejFunc in rejection:
            if rejFunc(sciSection, sigma, threshold):
                rejFlag = True
                break
        
        if rejFlag:
            # Reject
            continue
        
        # Convolve image with kernel
        log.debug("Beginning convolution of image")
        st = time.time()
        h = convolve2d( sciSection, kernel )
        
        et = time.time()
        log.debug("Convolve Time: %.3f" % (et-st))
        
        if not grid:
            h[0:kernelHalfDimension,:] = 0
            h[xDimension - kernelHalfDimension:xDimension,:] = 0
            h[:,0:kernelHalfDimension] = 0
            h[:,yDimension - kernelHalfDimension:yDimension] = 0
        
        log.debug("Finished convolution of image")
        
        # Filter
        offset = yy * xDimension + xx
        
        # Valid image pixels are greater than hmin
        index = np.where(np.ravel(h >= hmin))[0]
        nfound = index.size
        
        # Any maxima found?
        if nfound > 0:
            h = h.flatten()
            for i in np.arange(pixels):
                # Needs to be changed
                try:
                    stars = np.where(np.ravel(h[index] 
                                              >= h[index+ offset[i]]))[0]
                except:
                    break
                nfound = stars.size
                # Do valid local maxima exist?
                if nfound == 0:
                    log.debug("No objects found.")
                    break
                index = index[stars]
            h.shape = (xDimension, yDimension)
            
            ix = index % yDimension               # X index of local maxima
            iy = index / yDimension               # Y index of local maxima
            ngood = index.size
        else:
            log.debug("No objects above hmin (%s) were found." %(str(hmin)))
            continue
        
        # Loop over star positions; compute statistics
        
        st = time.time()
        for i in np.arange(ngood):
            temp = np.array(sciSection[iy[i]-kernelHalfDimension :
                                           (iy[i] + kernelHalfDimension)+1,
                                       ix[i] - kernelHalfDimension :
                                           (ix[i] + kernelHalfDimension)+1])
            
            # pixel intensity
            pixIntensity = h[iy[i],ix[i]]
            
            # Compute Sharpness statistic
            #@@FIXME: This should do proper checking...the issue
            # is an out of range index with kernelhalf and temp
            # IndexError: index (3) out of range (0<=index<=0) in dimension 0
            try:
                sharp1 = (temp[kernelHalfDimension,kernelHalfDimension] - 
                          (np.sum(mask * temp)) / pixels) / pixIntensity
            except:
                continue
            
            if (sharp1 < sharplim[0]) or (sharp1 > sharplim[1]):
                # Reject
                # not sharp enough?
                continue
            
            dx = np.sum(np.sum(temp, 1) * c1)
            dy = np.sum(np.sum(temp, 0) * c1)
            
            if (dx <= 0) or (dy <= 0):
                # Reject
                continue
            
            # Roundness statistic
            around = 2 * (dx - dy) / (dx + dy)
            
            # Reject if not within specified roundness boundaries.
            if (around < roundlim[0]) or (around > roundlim[1]):
                # Reject
                continue
            
            """
             Centroid computation: The centroid computation was modified
             in Mar 2008 and now differs from DAOPHOT which multiplies the
             correction dx by 1/(1+abs(dx)). The DAOPHOT method is more robust
             (e.g. two different sources will not merge)
             especially in a package where the centroid will be subsequently be
             redetermined using PSF fitting. However, it is less accurate,
             and introduces biases in the centroid histogram. The change here
             is the same made in the IRAF DAOFIND routine (see
             http://iraf.net/article.php?story=7211&query=daofind ) [1]
            """
            
            sd = np.sum(temp * ywt, 0)
            
            sumgd = np.sum(wt * sgy * sd)
            sumd = np.sum(wt * sd)
            sddgdx = np.sum(wt * sd * dgdx)
            
            hx = (sumgd - sumgx * sumd / sumOfWt) / (sumgsqy - 
                                                     sumgx ** 2 / sumOfWt)
            
            # HX is the height of the best-fitting marginal Gaussian. If
            # this is not positive then the centroid does not make sense. [1]
            if (hx <= 0):
                # Reject
                continue
            
            skylvl = (sumd - hx * sumgx) / sumOfWt
            dx = (sgdgdx -(sddgdx - sdgdx*(hx*sumgx + 
                                           skylvl*sumOfWt))) /(hx*sdgdxs /
                                                               sigSQ)
            
            if abs(dx) >= kernelHalfDimension:
                # Reject
                continue
            
            #X centroid in original array
            xcen = ix[i] + dx
            
            # Find Y centroid
            sd = np.sum(temp * xwt, 1)
            
            sumgd = np.sum(wt * sgx * sd)
            sumd = np.sum(wt * sd)
            
            sddgdy = np.sum(wt * sd * dgdy)
            
            hy = (sumgd - sumgy*sumd/sumOfWt) / (sumgsqx - sumgy**2/sumOfWt)
            
            if (hy <= 0):
                # Reject
                continue
            
            skylvl = (sumd - hy*sumgy) / sumOfWt
            dy = (sgdgdy - (sddgdy - 
                            sdgdy*(hy*sumgy + skylvl*sumOfWt))) / (hy*sdgdys /
                                                                   sigSQ)
            if abs(dy) >= kernelHalfDimension:
                # Reject 
                continue
            
            ycen = iy[i] + dy    #Y centroid in original array
            
            subXYArray.append( [xcen, ycen, pixIntensity] )
            
        et = time.time()
        log.debug("Looping over Stars time: %.3f" % (et-st))
        
        subXYArray = _average_each_cluster( subXYArray, 10 )
        xySize = len(subXYArray)
        
        
        for i in range( xySize ):
            subXYArray[i] = subXYArray[i].tolist()
            # I have no idea why the positions are slightly modified.
            # Was done originally in iqTool, perhaps for minute correcting.
            subXYArray[i][0] += 1
            subXYArray[i][1] += 1
            
            subXYArray[i][0] += yoffset
            subXYArray[i][1] += xoffset
        
        xyArray.extend(subXYArray)
            
    oet = time.time()
    overall_time = (oet-ost)
    log.debug("No. of objects detected: %i" % len(xyArray))
    log.debug("Overall time:%.3f seconds." % overall_time)
    
    return xyArray


def _estimate_sigma(scidata):
    fim = np.copy(scidata)
    stars = np.where(fim > (scidata.mean() + scidata.std()))
    fim[stars] = scidata.mean()
        
    outside = np.where(fim < (scidata.mean() - scidata.std()))
    fim[outside] = scidata.mean()

    sigma = fim.std()

    return sigma

def _average_each_cluster( xyArray, pixApart=10.0 ):
    """
    daofind can produce multiple centers for an object. This
    algorithm corrects that.
    For Example: 
    626.645599527 179.495974369
    626.652254706 179.012831637
    626.664059364 178.930738423
    626.676504143 178.804093054
    626.694643376 178.242374891
    
    This function will try to cluster these close points together, and
    produce a single center by taking the mean of the cluster. This
    function is based off the removeNeighbors function in iqUtil.py
    
    :param xyArray: The list of centers of found stars.
    :type xyArray: List
    
    :param pixApart: The max pixels apart for a star to be considered
                     part of a cluster. 
    :type pixApart: Number
    
    :return: The centroids of the stars sorted by the X dimension.
    :rtype: List
    """

    newXYArray = []
    xyArray.sort()
    xyArray = np.array( xyArray )
    xyArrayForMean = []
    xyClusterFlag = False
    j = 0
    while j < (xyArray.shape[0]):
        i = j + 1
        while i < xyArray.shape[0]:
            diffx = xyArray[j][0] - xyArray[i][0]
            if abs(diffx) < pixApart:
                diffy = xyArray[j][1] - xyArray[i][1]
                if abs(diffy) < pixApart:
                    if not xyClusterFlag:
                        xyClusterFlag = True
                        xyArrayForMean.append(j)
                    xyArrayForMean.append(i)
                    
                i = i + 1
            else:
                break
        
        if xyClusterFlag:
            xyMean = [np.mean( xyArray[xyArrayForMean], axis=0 ),
                      np.mean( xyArray[xyArrayForMean], axis=1 )]
            newXYArray.append( xyMean[0] )
            # Almost equivalent to reverse, except for numpy
            xyArrayForMean.reverse()
            for removeIndex in xyArrayForMean:
                xyArray = np.delete( xyArray, removeIndex, 0 )
            xyArrayForMean = []
            xyClusterFlag = False
            j = j - 1
        else:
            newXYArray.append( xyArray[j] )
        
        
        j = j + 1
    
    return newXYArray


def _sextractor(sciext=None,dqext=None,seeing_estimate=None):

    # Get the log
    log = gemLog.getGeminiLog()

    # Get path to default sextractor parameter files
    default_dict = Lookups.get_lookup_table(
                             "Gemini/source_detection/sextractor_default_dict",
                             "sextractor_default_dict")
    
    # Write the science extension to a temporary file on disk
    scitmpfn = "tmp%ssx%s%s%s" % (str(os.getpid()),sciext.extname(),
                                  sciext.extver(),
                                  os.path.basename(sciext.filename))
    log.fullinfo("Writing temporary file %s to disk" % scitmpfn)
    sciext.write(scitmpfn,rename=False,clobber=True)

    # If DQ extension is given, do the same for it
    if dqext is not None:
        # Make sure DQ data is 16-bit; flagging doesn't work
        # properly if it is 32-bit
        dqext.data = dqext.data.astype(np.int16)
        
        dqtmpfn = "tmp%ssx%s%s%s" % (str(os.getpid()),dqext.extname(),
                                   dqext.extver(),
                                   os.path.basename(dqext.filename))
        log.fullinfo("Writing temporary file %s to disk" % dqtmpfn)
        dqext.write(dqtmpfn,rename=False,clobber=True)

        # Get correct default files for this mode
        default_dict = default_dict['dq']
        for key in default_dict:
            default_file = lookup_path(default_dict[key]).rstrip(".py")
            default_dict[key] = default_file

    else:
        # Dummy temporary DQ file name
        dqtmpfn = ""

        # Get correct default files for this mode
        default_dict = default_dict['no_dq']
        for key in default_dict:
            default_file = lookup_path(default_dict[key]).rstrip(".py")
            default_dict[key] = default_file

    outtmpfn = "tmp%ssxOUT%s%s%s" % (str(os.getpid()),sciext.extname(),
                                     sciext.extver(),
                                     os.path.basename(sciext.filename))

    # if no seeing estimate provided, run sextractor once with
    # default, then re-run to get proper stellar classification
    if seeing_estimate is None:
        iter = [0,1]
    else:
        iter = [0]
    
    log.fullinfo("Calling SExtractor")
    for i in iter:

        if seeing_estimate is None:
            # use default seeing estimate for a first pass
            sx_cmd = ["sex",
                      "%s[0]" % scitmpfn,
                      "-c","%s" % default_dict["sex"],
                      "-FLAG_IMAGE","%s[0]" % dqtmpfn,
                      "-CATALOG_NAME","%s" % outtmpfn,
                      "-PARAMETERS_NAME","%s" % default_dict["param"],
                      "-FILTER_NAME","%s" % default_dict["conv"],
                      "-STARNNW_NAME","%s" % default_dict["nnw"],]
        else:
            # run with provided seeing estimate
            sx_cmd = ["sex",
                      "%s[0]" % scitmpfn,
                      "-c","%s" % default_dict["sex"],
                      "-FLAG_IMAGE","%s[0]" % dqtmpfn,
                      "-CATALOG_NAME","%s" % outtmpfn,
                      "-PARAMETERS_NAME","%s" % default_dict["param"],
                      "-FILTER_NAME","%s" % default_dict["conv"],
                      "-STARNNW_NAME","%s" % default_dict["nnw"],
                      "-SEEING_FWHM","%f" % seeing_estimate,
                      ]

        try:
            pipe_out = subprocess.Popen(sx_cmd,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT)
        except OSError:
            os.remove(scitmpfn)
            if dqext is not None:
                os.remove(dqtmpfn)
            raise Errors.ScienceError("SExtractor failed")

        # Sextractor output is full of non-ascii characters, send it
        # only to debug for now
        stdoutdata = pipe_out.communicate()[0]
        log.debug(stdoutdata)

        hdulist = pf.open(outtmpfn)
        tdata = hdulist[1].data
        tcols = hdulist[1].columns

        # Convert FWHM_WORLD to arcsec
        fwhm = tdata["FWHM_WORLD"]
        fwhm *= 3600.0
        
        # Mask out the bottom 3 bits of the sextractor flags
        # These are used for purposes we don't need here
        sxflag = tdata["FLAGS"]
        sxflag &= 65528

        # Get some extra flags to get point sources only
        # for seeing estimate
        if dqext is not None:
            dqflag = tdata["IMAFLAGS_ISO"]
        else:
            dqflag = np.zeros_like(sxflag)
        aflag = np.where(tdata["ISOAREA_IMAGE"]<100,1,0)
        eflag = np.where(tdata["ELLIPTICITY"]>0.5,1,0)
        sflag = np.where(tdata["CLASS_STAR"]<0.6,1,0)

        # Bitwise-or all the flags
        flags = sxflag | dqflag | aflag | eflag | sflag
        good_fwhm = fwhm[flags==0]
        if len(good_fwhm)>2:
            seeing_estimate,sigma = at.clipped_mean(good_fwhm)
            if np.isnan(seeing_estimate) or seeing_estimate==0:
                seeing_estimate = None
                break
        else:
            seeing_estimate = None
            break
        
    log.fullinfo("Removing temporary files from disk:\n%s\n%s" %
                 (scitmpfn,dqtmpfn))
    os.remove(scitmpfn)
    os.remove(outtmpfn)
    if dqext is not None:
        os.remove(dqtmpfn)

    columns = {}
    for col in tcols:
        columns[col.name] = col
    
    return columns,seeing_estimate

def _test_sextractor_version():
    """
    Returns True if sextractor is runnable and version is 
    SEXTRACTOR_VERSION or later
    """

    # Compile a regular expression for matching the version
    # number from sextractor output
    versioncre = re.compile("^.*version (?P<v1>\d+)(\.(?P<v2>\d+))?"\
                            "(\.(?P<v3>\d+))?.*$")

    # Get acceptable version from global variable
    std_version = SEXTRACTOR_VERSION

    right_version = False

    sx_cmd = ["sex", "--version"]
    try:
        pipe_out = subprocess.Popen(sx_cmd,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT)
    except OSError:
        return right_version

    stdoutdata = pipe_out.communicate()[0]
    m = versioncre.match(stdoutdata)
    if m is not None:
        # Get three version fields from match object
        v1 = int(m.group("v1"))
        if m.group("v2") is None:
            v2 = 0
        else:
            v2 = int(m.group("v2"))
        if m.group("v3") is None:
            v3 = 0
        else:
            v3 = int(m.group("v3"))

        if v1>std_version[0]:
            right_version = True
        elif v1==std_version[0]:
            if v2>std_version[1]:
                right_version = True
            elif v2==std_version[1]:
                if v3>=std_version[2]:
                    right_version = True

    return right_version


def _fit_sources(ad, ext=None, max_sources=50, threshold=5.0,
                 seeing_estimate=None,
                 centroid_function="moffat"):
    """
    This function takes a list of identified sources in an image, fits
    a Gaussian to each one, and stores the fit FWHM and ellipticity to
    the OBJCAT.  Bad fits are marked with a 1 in the 'FLAGS' column.
    If a DQ plane is provided, and a source has a non-zero DQ value,
    it will also receive a 1 in the 'FLAGS' column.
    
    :param ad: input image
    :type ad: AstroData instance with OBJCAT attached

    :param max_sources: Maximum number of sources to fit on each science
                        extension. Will start at the center of the 
                        extension and move outward. If None,
                        will fit all sources.
    :type max_sources: integer

    :param threshold: Number of sigmas above background level to fit source
    :type threshold: float

    :param centroid_function: Function for centroid fitting with daofind
    :type centroid_function: string, can be: 'moffat','gauss'
                    Default: 'moffat'
    """
    
    import scipy.optimize
    from gempy import astrotools as at

    if ext is None:
        sciexts = ad["SCI"]
    else:
        sciexts = ad["SCI",ext]

    good_source = []
    for sciext in sciexts:
        extver = sciext.extver()
        #print 'sci',extver

        objcat = ad["OBJCAT",extver]
        if objcat is None:
            continue
        if objcat.data is None:
            continue

        img_data = sciext.data

        dqext = ad["DQ",extver]

        if dqext is not None:
            # estimate background from non-flagged data
            good_data = img_data[dqext.data==0]
            default_bg = np.median(good_data)
            sigma = _estimate_sigma(good_data)
        else:
            # estimate background from whole image
            default_bg = np.median(img_data)
            sigma = _estimate_sigma(img_data)

        # first guess at fwhm is .8 arcsec
        pixscale = float(sciext.pixel_scale())
        if seeing_estimate is None:
            seeing_estimate = .8
        default_fwhm = seeing_estimate / pixscale

        # stamp is 10*2 times this size on a side (16")
        aperture = 10*default_fwhm
    
        img_objx = objcat.data.field("X_IMAGE")
        img_objy = objcat.data.field("Y_IMAGE")
        img_obji = range(len(img_objx))

        # Calculate source's distance from the center of the image
        ctr_x = (img_data.shape[1]-1)/2.0
        ctr_y = (img_data.shape[0]-1)/2.0
        r2 = (img_objx-ctr_x)**2 + (img_objy-ctr_y)**2
        
        obj = np.array(np.rec.fromarrays([img_objx,img_objy,r2,img_obji],
                                         names=["x","y","r2","i"]))
        obj.sort(order="r2")

        count = 0
        for objx,objy,objr2,obji in obj:
        
            # array coords start with 0
            objx-=1
            objy-=1
        
            xlow, xhigh = int(round(objx-aperture)), int(round(objx+aperture))
            ylow, yhigh = int(round(objy-aperture)), int(round(objy+aperture))
        
            if (xlow>0 and xhigh<img_data.shape[1] and 
                ylow>0 and yhigh<img_data.shape[0]):
                stamp_data = img_data[ylow:yhigh,xlow:xhigh]
                if dqext is not None:

                    # Don't fit source if there is a bad pixel within
                    # 2*default_fwhm
                    dxlow, dxhigh = (int(round(objx-default_fwhm*2)),
                                     int(round(objx+default_fwhm*2)))
                    dylow, dyhigh = (int(round(objy-default_fwhm*2)), 
                                     int(round(objy+default_fwhm*2)))
                    stamp_dq = dqext.data[dylow:dyhigh,dxlow:dxhigh]
                    if np.any(stamp_dq):
                        objcat.data.field("FLAGS")[obji] = 1
                        #print 'dq',obji
                        continue
            else:
                # source is too near the edge, skip it
                objcat.data.field("FLAGS")[obji] = 1
                #print 'edge',obji
                continue

            # after flagging for DQ/edge reasons, don't continue
            # with fit if max_sources was reached
            if max_sources is not None and count >= max_sources:
                continue

            # Check for too-near neighbors, don't fit source if found
            too_near = np.any((abs(obj['x']-objx)<default_fwhm) &
                              (abs(obj['y']-objy)<default_fwhm) &
                              (obj['i']!=obji))
            if too_near:
                objcat.data.field("FLAGS")[obji] = 1
                #print 'neighbor',obji
                continue


            # starting values for model fit
            bg = default_bg
            peak = stamp_data.max()-bg
            x_ctr = (stamp_data.shape[1]-1)/2.0
            y_ctr = (stamp_data.shape[0]-1)/2.0
            x_width = default_fwhm
            y_width = default_fwhm
            theta = 0.
            beta = 1.
        
            if peak<threshold*sigma:
                # source is too faint, skip it
                objcat.data.field("FLAGS")[obji] = 1
                #print 'faint',obji
                continue
            
            
            # instantiate model fit object and initial parameters
            if centroid_function=="gauss":
                pars = (bg, peak, x_ctr, y_ctr, x_width, y_width, theta)
                mf = at.GaussFit(stamp_data)
            elif centroid_function=="moffat":
                pars = (bg, peak, x_ctr, y_ctr, x_width, y_width, theta, beta)
                mf = at.MoffatFit(stamp_data)
            else:
                raise Errors.InputError("Centroid function %s not supported" %
                                        centroid_function)
                

            # least squares fit of model to data
            try:
                # for scipy versions < 0.9
                new_pars, success = scipy.optimize.leastsq(mf.calc_diff, pars,
                                                           maxfev=100, 
                                                           warning=False)
            except:
                # for scipy versions >= 0.9
                import warnings
                warnings.simplefilter("ignore")
                new_pars, success = scipy.optimize.leastsq(mf.calc_diff, pars,
                                                           maxfev=100)

            # track number of fits performed
            count += 1
            #print count

            # Set default flag for any fit source to 0
            objcat.data.field("FLAGS")[obji] = 0

            if success>3:
                # fit failed, move on
                objcat.data.field("FLAGS")[obji] = 1
                #print 'fit failed',obji
                continue
        
            if centroid_function=="gauss":
                (bg,peak,x_ctr,y_ctr,x_width,y_width,theta) = new_pars
            else: # Moffat
                (bg,peak,x_ctr,y_ctr,x_width,y_width,theta,beta) = new_pars
                
                # convert width to Gaussian-type sigma
                x_width = x_width*np.sqrt(((2**(1/beta)-1)/(2*np.log(2))))
                y_width = y_width*np.sqrt(((2**(1/beta)-1)/(2*np.log(2))))


            # convert fit parameters to FWHM, ellipticity
            fwhmx = abs(2*np.sqrt(2*np.log(2))*x_width)
            fwhmy = abs(2*np.sqrt(2*np.log(2))*y_width)
            pa = (theta*(180/np.pi))
            pa = pa%360
                
            if fwhmy < fwhmx:
                ellip = 1 - fwhmy/fwhmx
            elif fwhmx < fwhmy:
                ellip = 1 - fwhmx/fwhmy
                pa = pa-90 
            else:
                ellip = 0
                
            # FWHM is geometric mean of x and y FWHM
            fwhm = np.sqrt(fwhmx*fwhmy)


            # Shift PA to 0-180
            if pa > 180:
                pa -= 180
            if pa < 0:
                pa += 180

            # Check fit
            if peak<0.0:
                # source inverted, skip it
                objcat.data.field("FLAGS")[obji] = 1
                #print 'inverted',obji
                continue
            if bg<0.0:
                # bad fit, skip it
                objcat.data.field("FLAGS")[obji] = 1
                #print 'bg<0',obji
                continue
            if peak<threshold*sigma:
                # S/N too low, skip it
                objcat.data.field("FLAGS")[obji] = 1
                #print 's/n low',obji
                continue
                

            # update the position from the fit center
            newx = xlow + x_ctr + 1
            newy = ylow + y_ctr + 1
        

            # update the OBJCAT
            objcat.data.field("X_IMAGE")[obji] = newx
            objcat.data.field("Y_IMAGE")[obji] = newy
            objcat.data.field("FWHM_IMAGE")[obji] = fwhm
            objcat.data.field("FWHM_WORLD")[obji] = fwhm * pixscale
            objcat.data.field("ELLIPTICITY")[obji] = ellip

            # flag low ellipticity, reasonable fwhm sources as likely stars
            if ellip<0.1:
                objcat.data.field("CLASS_STAR")[obji] = 0.9
            elif ellip<0.3:
                objcat.data.field("CLASS_STAR")[obji] = 0.7
            elif ellip<0.5:
                objcat.data.field("CLASS_STAR")[obji] = 0.5
            else:
                objcat.data.field("CLASS_STAR")[obji] = 0.2

            if fwhm<1.0:
                # likely cosmic ray
                objcat.data.field("CLASS_STAR")[obji] *= 0.2
            elif fwhm<2*default_fwhm:
                # potential star
                objcat.data.field("CLASS_STAR")[obji] *= 0.9
            else:
                # likely extended source or bad fit
                objcat.data.field("CLASS_STAR")[obji] *= 0.2
                
            #print newx,newy,fwhm,ellip,peak,bg
            #print 'flag',objcat.data.field("FLAGS")[obji]
            #print 'class',objcat.data.field("CLASS_STAR")[obji]

        flags = (objcat.data.field("FLAGS")==0) & \
                (objcat.data.field("CLASS_STAR")>0.6)
        good_fwhm = objcat.data.field("FWHM_WORLD")[flags]

        #print good_fwhm
        if len(good_fwhm)>2:
            new_fwhm,sigma = at.clipped_mean(good_fwhm)
            if not(np.isnan(new_fwhm) or new_fwhm==0):
                seeing_estimate = new_fwhm

    return ad, seeing_estimate


