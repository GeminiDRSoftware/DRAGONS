import os
import sys
import re
import subprocess
import numpy as np
import pyfits as pf
import pywcs

#from itertools import compress
from copy import deepcopy

from astrodata import AstroData
from astrodata.utils import Errors
from astrodata.utils import logutils
from astrodata.utils.ConfigSpace import lookup_path

from astrodata_Gemini.ADCONFIG_Gemini.lookups import ColorCorrections
    
from gempy.library import astrotools as at
from gempy.gemini import gemini_tools as gt
from gempy.gemini.gemini_catalog_client import get_fits_table

from datetime import datetime

from primitives_GENERAL import GENERALPrimitives
from pifgemini import mask as pifmask

# Define the earliest acceptable SExtractor version, currently: 2.8.6
SEXTRACTOR_VERSION = [2,8,6]

# ------------------------------------------------------------------------------
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
        This primitive calls the gemini_catalog_client module to
        query a catalog server and construct a fits table containing
        the catalog data.

        That module will query either gemini catalog servers or
        vizier. Currently, sdss9 and 2mass (point source catalog)
        are supported.

        For example, with sdss9, the FITS table has the following columns:

        - 'Id'       : Unique ID. Simple running number
        - 'Cat-id'   : SDSS catalog source name
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

        With 2mass, the first 4 columns are the same, but the photometry
        columns reflect the J H and K bands.

        This primitive then adds the fits table catalog to the Astrodata
        object as 'REFCAT'

        :param source: Source catalog to query, as defined in the
                       gemini_catalog_client module
        :type source: string

        :param radius: The radius of the cone to query in the catalog, 
                       in degrees. Default is 4 arcmin
        :type radius: float
        """

        # Instantiate the log
        log = logutils.get_logger(__name__)

        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "addReferenceCatalog", "starting"))

        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["addReferenceCatalog"]

        # Initialize the list of output AstroData objects
        adoutput_list = []

        # Get the necessary parameters from the RC
        source = rc["source"]
        radius = rc["radius"]

        # Loop over each input AstroData object in the input list
        adinput = rc.get_inputs_as_astrodata()
        for ad in adinput:

            # Fetch the reference catalog here. Really, we should fetch a separate one for each
            # extension, but it turns out that you can only actually do a cone search (ie a 
            # circular area), so there's a huge overlap between the results anyway, and in fact the separate
            # ad[SCI]s don't report separate RA,Dec anyway, unless we decode the WCS directly.
            # More to the point, with 6 amp mode, doing 6 queries to an external server is annoyingly slow,
            # so for now at least I'm moving it to do one query and store the same refcat for each extver.
            # PH 20111202
            try:
                ra = ad.wcs_ra().as_pytype()
                dec = ad.wcs_dec().as_pytype()
                if type(ra) is not float:
                    raise ValueError("wcs_ra descriptor did not return a float.")
                if type(ra) is not float:
                    raise ValueError("wcs_dec descriptor did not return a float.")
            except:
                if "qa" in rc.context:
                    log.warning("No RA/Dec in header of %s; cannot find "\
                                "reference sources" % ad.filename)
                    adoutput_list.append(ad)
                    continue
                else:
                    raise

            log.fullinfo("Querying %s for reference catalog" % source)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Get the fits table HDU
                tb_hdu = get_fits_table(source, ra, dec, radius)

            # Loop through the science extensions
            for sciext in ad['SCI']:
                extver = sciext.extver()

                # See the note above - the actual fetch is moved this to outside the loop, at least for now
                if(tb_hdu is None):
                    log.stdinfo("No reference catalog sources found "\
                                "for %s['SCI',%d]" % (ad.filename, extver))
                    continue
                else:
                    log.stdinfo("Found %d reference catalog sources for %s['SCI',%d]" % (len(tb_hdu.data), ad.filename, extver))

                if(tb_hdu):
                    tb_ad = deepcopy(AstroData(tb_hdu))
                    tb_ad.rename_ext('REFCAT', extver)

                    if(ad['REFCAT', extver]):
                        log.fullinfo("Replacing existing REFCAT in %s" % ad.filename)
                        ad.remove(('REFCAT', extver))
                    else:
                        log.fullinfo("Adding REFCAT to %s" % ad.filename)

                    ad.append(tb_ad)

            # Match the object catalog against the reference catalog
            # Update the refid and refmag columns in the object catalog
            if ad.count_exts("OBJCAT")>0:
                ad = _match_objcat_refcat(adinput=ad)[0]
            else:
                log.warning("No OBJCAT found; not matching OBJCAT to REFCAT")

            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, primname=self.myself(), keyword=timestamp_key)

            # Change the filename
            ad.filename = gt.filename_updater(adinput=ad, suffix=rc["suffix"],
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
        
        :param mask: Whether to apply the DQ plane as a mask before detecting
                     the sources.
        :type sigma: bool
        """
 
        # Instantiate the log
        log = logutils.get_logger(__name__)

        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "detectSources", "starting"))
        
        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["detectSources"]

        # Get the necessary parameters from the RC
        sigma = rc["sigma"]
        threshold = rc["threshold"]
        fwhm = rc["fwhm"]
        max_sources = rc["max_sources"]
        centroid_function = rc["centroid_function"]
        method = rc["method"]
        set_saturation = rc["set_saturation"]
        mask = rc["mask"]
        
        # Check source detection method
        if method not in ["sextractor","daofind"]:
            raise Errors.InputError("Source detection method "+
                                    method+" is unsupported.")
                
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
                                
            # Get a seeing estimate from the header, if available
            seeing_est = ad.phu_get_key_value("MEANFWHM")
            
            # If masking is required, make a deepcopy of the astrodata 
            # instance and perform the masking to create the image from
            # which to do the source detection. 
            if mask:
                ad_for_source = deepcopy(ad)
                ad_for_source = pifmask.apply_dq_plane(ad_for_source)
            else:
                ad_for_source = ad

                
            if method=="sextractor":
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
                        ad_for_source = _sextractor(ad_for_source, seeing_est,
                                                sxdict=self.sx_default_dict,
                                                set_saturation=set_saturation)
                    except Errors.ScienceError:
                        log.warning("SExtractor failed. "\
                                    "Setting method=daofind")
                        method="daofind"
                
            if method=="daofind":
                try:
                    pixscale = ad_for_source.pixel_scale()
                except:
                    pixscale = None
                if pixscale is None:
                    log.warning("%s does not have a pixel scale, " \
                                "cannot fit sources" % ad_for_source.filename)
                    continue

                for sciext in ad_for_source["SCI"]:
                
                    extver = sciext.extver()
                
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
                                    (ad_for_source.filename,extver))
                        continue
                    else:
                        log.stdinfo("Found %d sources in %s['SCI',%d]" %
                                    (nobj,ad_for_source.filename,extver))

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

                
                    # Add OBJCAT
                    ad_for_source = gt.add_objcat(adinput=ad_for_source, 
                                                  extver=extver, 
                                                  replace=True, 
                                                  columns=columns,
                                                  sxdict=self.sx_default_dict)[0]
            
                # Do some simple photometry on all extensions to get fwhm, 
                # ellipticity
                log.stdinfo("Fitting sources for simple photometry")

                # Divide the max_sources by the number of extensions
                max_sources = int(max_sources/ad_for_source.count_exts("SCI"))

                if seeing_est is None:
                    # Run the fit once to get a rough seeing estimate 
                    if max_sources>20:
                        tmp_max=20
                    else:
                        tmp_max=max_sources
                    junk,seeing_est = _fit_sources(
                        ad_for_source,ext=1,max_sources=tmp_max, 
                        threshold=threshold, 
                        centroid_function=centroid_function,
                        seeing_estimate=None)
                ad_for_source,seeing_est = _fit_sources(
                    ad_for_source,max_sources=max_sources,threshold=threshold,
                    centroid_function=centroid_function,
                    seeing_estimate=seeing_est)

            # Run some profiling code on the best sources to produce
            # a more IRAF-like FWHM number
            # This will fill in a couple more columns in the OBJCAT
            # (PROFILE_FWHM, PROFILE_EE50)
            ad_for_source = _profile_sources(ad_for_source)
            
            # After running the source extraction code, if masking was 
            # done it is necessary to retrieve the original data and copy 
            # the OBJCAT and OBJMASK extensions to the original data
            if mask:
                for sciext in ad["SCI"]:
                    ext = sciext.extver()
                    if ad["OBJCAT", ext]:
                        ad.remove(("OBJCAT", ext))
                    if ad["OBJMASK", ext]:
                        ad.remove(("OBJMASK", ext))
                    try:                        
                        catcopy = deepcopy(ad_for_source["OBJCAT", ext])
                        ad.append(catcopy)          
                        maskcopy = deepcopy(ad_for_source["OBJMASK", ext])
                        ad.append(maskcopy)                                    
                    except:
                        log.stdinfo("No OBJCAT and OBJMASK extensions "
                                    "available for %s, source detection has " 
                                    "failed" % (ad.filename))

            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, primname=self.myself(), keyword=timestamp_key)

            # Change the filename
            ad.filename = gt.filename_updater(adinput=ad, suffix=rc["suffix"], 
                                              strip=True)
            
            # Append the output AstroData object to the list 
            # of output AstroData objects
            adoutput_list.append(ad)
        
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc

    def measureCCAndAstrometry(self, rc):

        # Instantiate the log
        log = logutils.get_logger(__name__)


        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "measureCCAndAstrometry",
                                 "starting"))

        # Test for OBJCATs first; if not present, there's no point in
        # continuing
        adinput = rc.get_inputs_as_astrodata()
        found_objcat = False
        for ad in adinput:
            for ext in ad["SCI"]:
                objcat = ad["OBJCAT",ext.extver()]
                if objcat is None:
                    continue
                elif objcat.data is None:
                    continue
                elif len(objcat.data)==0:
                    continue
                else:
                    found_objcat = True
                    break

        # If at least one non-empty OBJCAT is present, add reference
        # catalogs to the input
        if not found_objcat:
            log.stdinfo("No OBJCAT found in input, so no comparison to "\
                        "reference sources will be performed")
            rc.report_output(adinput)
        else:
            rc.run("addReferenceCatalog")
            
            # Test for reference catalogs; if not present, don't continue
            adinput = rc.get_inputs_as_astrodata()
            found_refcat = False
            for ad in adinput:
                for ext in ad["SCI"]:
                    refcat = ad["REFCAT",ext.extver()]
                    if refcat is None:
                        continue
                    elif refcat.data is None:
                        continue
                    elif len(refcat.data)==0:
                        continue
                    else:
                        found_refcat = True
                        break
            
            # If at least one non-empty REFCAT is present, measure
            # the zeropoint and the astrometric offset
            if not found_refcat:
                log.stdinfo("No reference sources found; no comparison "\
                            "will be performed")
                rc.report_output(adinput)
            else:
                rc.run("measureCC")
                rc.run("determineAstrometricSolution")

                # Check to see whether we should update the WCS
                correct_wcs = rc["correct_wcs"]
                if correct_wcs:
                    rc.run("updateWCS")
        
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

    import matplotlib.pyplot as plt

    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = logutils.get_logger(__name__)

    # The validate_input function ensures that the input is not None and
    # returns a list containing one or more inputs
    adinput_list = gt.validate_input(input=adinput)

    debug = False

    # Initialize the list of output AstroData objects
    adoutput_list = []
    try:

        # Loop over each input AstroData object in the input list
        for ad in adinput_list:
            filter_name = ad.filter_name(pretty=True).as_pytype()
            colterm_dict = ColorCorrections.colorTerms
            if filter_name in colterm_dict:
                formulae = colterm_dict[filter_name]
            else:
                log.warning("Filter %s is not in catalogs - will not be able to flux calibrate" % filter_name)
                formulae = []

            # If there are no refcats, don't try to go through them.
            if ad['REFCAT'] is None:
                log.warning("No Reference Catalogs present - cannot match to objcat")
            else:
                # Loop through the objcat extensions
                if ad['OBJCAT'] is None:
                    raise Errors.InputError("Missing OBJCAT in %s" % (ad.filename))
                # Plotting for debugging purposes
                if debug:
                    num_ext = len(ad['OBJCAT'])
                    if 'GSAOI' in ad.types:
                        plotcols = 2
                        plotrows = 2
                    else:
                        plotcols = num_ext
                        plotrows = 1
                    fig, axarr = plt.subplots(plotrows, plotcols, sharex=True,
                                sharey=True, figsize=(10,10), squeeze=False)
                    axarr[0,0].set_xlim(0,ad['SCI',1].data.shape[1])
                    axarr[0,0].set_ylim(0,ad['SCI',1].data.shape[0])

                # Try to be clever here, and work on the extension with the
                # highest number of matches first, as this will give the
                # most reliable offsets, which can then be used to constrain
                # the other extensions. The problem is we don't know how many
                # matches we'll get until we do it, and that's slow, so use
                # OBJCAT length as a proxy.
                objcat_lengths = {}
                for objcat in ad['OBJCAT']:
                    extver = objcat.extver()
                    objcat_lengths[extver] = len(objcat.data)
                objcat_order = sorted(objcat_lengths.items(),
                                      key=lambda x:x[1], reverse=True)

                initial = 10.0/ad.pixel_scale() # Search box size
                final = 1.0/ad.pixel_scale() # Matching radius
                xoffsets = []
                yoffsets = []

                for extver,objcat_len in objcat_order:
                    objcat = ad['OBJCAT', extver]
    
                    # Check that a refcat exists for this objcat extver
                    refcat = ad['REFCAT',extver]
                    if not refcat:
                        log.warning("Missing [REFCAT,%d] in %s - "
                                    "Cannot match objcat against missing refcat"
                                    % (extver,ad.filename))
                    else:
                        # Not needed as objects already culled
                        #xx = np.where(objcat.data['ISOAREA_IMAGE'] >= 20, 
                        #              objcat.data['X_IMAGE'], -999)
                        #yy = np.where(objcat.data['ISOAREA_IMAGE'] >= 20, 
                        #              objcat.data['Y_IMAGE'], -999)
                        xx = objcat.data['X_IMAGE']
                        yy = objcat.data['Y_IMAGE']
                        
                        # The coordinates of the reference sources are 
                        # corrected to pixel positions using the WCS of the 
                        # object frame
                        sra = refcat.data['RAJ2000']
                        sdec = refcat.data['DEJ2000']
                        wcsobj = pywcs.WCS(ad["SCI",extver].header)
                        sx, sy = wcsobj.wcs_sky2pix(sra,sdec,1)
                        sx_orig = np.copy(sx)
                        sy_orig = np.copy(sy)
                        # Reduce the search radius if we've found a match
                        # on a previous extension
                        if xoffsets:
                            sx += np.mean(xoffsets)
                            sy += np.mean(yoffsets)
                            initial = 2.5/ad.pixel_scale()
                        
                        # Here's CJS at work.
                        # First: estimate number of reference sources in field
                        # Do better using actual size of illuminated field
                        num_ref_sources = np.sum(np.all((sx>-initial,
                            sx<ad['SCI',extver].data.shape[1]+initial,
                            sy>-initial,
                            sy<ad['SCI',extver].data.shape[0]+initial),
                            axis=0))
                        # How many objects do we want to try to match?
                        if objcat_len > 2*num_ref_sources:
                            keep_num = max(int(1.5*num_ref_sources),
                                           min(10,objcat_len))
                        else:
                            keep_num = objcat_len
                        # Now sort the object catalogue -- MUST NOT alter order
                        sorted_indices = np.argsort(np.where(
                                objcat.data['NIMAFLAGS_ISO']>
                                0.5*objcat.data['ISOAREA_IMAGE'],
                                999,objcat.data['MAG_AUTO']))[:keep_num]
    
                        # FIXME - need to address the wraparound problem here
                        # if we straddle ra = 360.00 = 0.00
                        # CJS: is this a problem? Everything's in pixels so it should be OK
                        
                        (oi, ri) = at.match_cxy(xx[sorted_indices],sx,
                                    yy[sorted_indices],sy,
                                    firstPass=initial, delta=final, log=log)
                        log.stdinfo("Matched %d objects in ['OBJCAT',%d] against ['REFCAT',%d]"
                                    % (len(oi), extver, extver))
                        # If this is a "good" match, save it
                        if len(oi) > 2:
                            xoffsets.append(np.median(xx[sorted_indices[oi]] - sx_orig[ri]))
                            yoffsets.append(np.median(yy[sorted_indices[oi]] - sy_orig[ri]))

                        # Loop through the reference list updating the refid in the objcat
                        # and the refmag, if we can
                        for i in range(len(oi)):
                            real_index = sorted_indices[oi[i]]
                            objcat.data['REF_NUMBER'][real_index] = refcat.data['Id'][ri[i]]
                            # CJS: I'm not 100% sure about assigning the wcs.sky2pix
                            # values here, in case the WCS is poor
                            tempra = refcat.data['RAJ2000'][ri[i]]
                            tempdec = refcat.data['DEJ2000'][ri[i]]
                            tempx, tempy = wcsobj.wcs_sky2pix(tempra,tempdec,1)
#                            objcat.data['REF_X'][real_index] = tempx
#                            objcat.data['REF_Y'][real_index] = tempy

                            # Assign the magnitude
                            if formulae:
                                mag, mag_err = _calculate_magnitude(formulae, refcat, ri[i])
                                objcat.data['REF_MAG'][real_index] = mag
                                objcat.data['REF_MAG_ERR'][real_index] = mag_err

                        if debug:
                            # Show the fit
                            if 'GSAOI' in ad.types:
                                plotpos = ((4-extver) // 2, (extver % 3) % 2)
                            else:
                                plotpos = (0,extver-1)
                            if len(oi):
                                dx = np.median(xx[sorted_indices[oi]] - sx[ri])
                                dy = np.median(yy[sorted_indices[oi]] - sy[ri])
                            else:
                                dx = 0
                                dy = 0
                            # bright OBJCAT sources
                            axarr[plotpos].scatter(xx[sorted_indices], yy[sorted_indices], s=50, c='w')
                            # all REFCAT sources, original positions
                            axarr[plotpos].scatter(sx_orig, sy_orig, s=30, c='k', marker='+')
                            # all REFCAT sources shifted
                            axarr[plotpos].scatter(sx+dx, sy+dy, s=30, c='r', marker='+')
                            # matched REFCAT sources shifted
                            axarr[plotpos].scatter(sx[ri]+dx, sy[ri]+dy, s=30, c='r', marker='s')
                            for i in range(len(sx)):
                                if i in ri:
                                    axarr[plotpos].plot([sx_orig[i],sx[i]+dx],[sy_orig[i],sy[i]+dy], 'r')
                                else:
                                    axarr[plotpos].plot([sx_orig[i],sx[i]+dx],[sy_orig[i],sy[i]+dy], 'k--')

            if debug:
                plt.show()
            
            adoutput_list.append(ad)

        return adoutput_list
    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise

def _calculate_magnitude(formulae, refcat, indx):
    
    # This is a bit ugly: we want to iterate over formulae so we must
    # nest a single formula into a list
    if type(formulae[0]) is not list:
        formulae = [formulae]

    mags = []
    mag_errs = []
    for formula in formulae:
        mag = 0.0
        mag_err_sq = 0.0
        for term in formula:
            # single filter
            if type(term) is str:
                if term+'mag' in refcat.data.columns.names:
                    mag += refcat.data[term+'mag'][indx]
                    mag_err_sq += refcat.data[term+'mag_err'][indx]**2
                else:
                    # Will ensure this magnitude is not used
                    mag = np.nan
            # constant (with uncertainty)
            elif len(term) == 2:
                mag += float(term[0])
                mag_err_sq += float(term[1])**2
            # color term (factor, uncertainty, color)
            elif len(term) == 3:
                filters = term[2].split('-')
                if len(filters)==2 and np.all([f+'mag' in refcat.data.columns.names for f in filters]):
                    col = refcat.data[filters[0]+'mag'][indx] - \
                        refcat.data[filters[1]+'mag'][indx]
                    mag += float(term[0])*col
                    dmagsq = refcat.data[filters[0]+'mag_err'][indx]**2 + \
                        refcat.data[filters[1]+'mag_err'][indx]**2
                    # When adding a (H-K) color term, often H is a 95% upper limit
                    # If so, we can only return an upper limit, but we need to
                    # account for the uncertainty in K-band 
                    if np.isnan(dmagsq):
                        mag -= 1.645*np.sqrt(mag_err_sq)
                    mag_err_sq += ((term[1]/term[0])**2 + dmagsq/col**2) * \
                        (float(term[0])*col)**2
                else:
                    mag = np.nan        # Only consider this if values are sensible
        if not np.isnan(mag):
            mags.append(mag)
            mag_errs.append(np.sqrt(mag_err_sq))
    
    # Take the value with the smallest uncertainty (NaN = large uncertainty)
    if mags:
        lowest = np.argmin(np.where(np.isnan(mag_errs),999,mag_errs))
        return mags[lowest], mag_errs[lowest]
    else:
        return -999, -999

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
    try:
        from stsci.convolve import convolve2d
    except ImportError:
        from convolve import convolve2d
    
    log = logutils.get_logger(__name__)
    
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
            xyMean = [np.mean( xyArray[xyArrayForMean], axis=0 , dtype=np.float64),
                      np.mean( xyArray[xyArrayForMean], axis=1 , dtype=np.float64)]
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


def _sextractor(ad=None, seeing_estimate=None, sxdict=None, set_saturation=False):
    
    # Get the log
    log = logutils.get_logger(__name__)

    # Get path to default sextractor parameter files
    default_dict = sxdict
    
    # Write the AD instance to a temporary FITS file on disk
    tmpfn = "tmp%ssx%s" % (str(os.getpid()),
                           os.path.basename(ad.filename))
    log.fullinfo("Writing temporary file %s to disk" % tmpfn)
    ad.write(tmpfn,rename=False,clobber=True)

    result = {}

    for sciext in ad["SCI"]:
        extver = sciext.extver()
        dict_key = ("SCI",extver)

        # Get the integer extension number, relative to the
        # AstroData numbering convention (0=first extension
        # after PHU). This is the same convension sextractor
        # uses.
        extnum = ad.ext_index(("SCI",extver))
        scitmpfn = "%s[%d]" % (tmpfn,extnum)

        dqext = ad["DQ",extver]
        if dqext is not None:
            # Make sure DQ data is 16-bit; flagging doesn't work
            # properly if it is 32-bit
            dqext.data = dqext.data.astype(np.int16)
        
            extnum = ad.ext_index(("DQ",extver))
            dqtmpfn = "%s[%d]" % (tmpfn,extnum)
            dq_type = "dq"

        else:
            # Dummy temporary DQ file name
            dqtmpfn = ""
            dq_type = "no_dq"

        # Check if there are config files for this instrument
        inst = ad.instrument().as_pytype().lower()        
        if (inst + '_' + dq_type) in default_dict:
            dd = default_dict[inst + '_' + dq_type].copy()
        # Otherwise use default config files
        else:
            dd = default_dict[dq_type].copy()

        # Each file in the dictionary needs the correct path added                
        for key in dd:
            dict_file = lookup_path(dd[key])
            # lookup_path adds .py on automatically, so needs to be removed.
            # Not ideal!
            if dict_file.endswith(".py"):
                dict_file = dict_file[:-3]
            dd[key] = dict_file
                        
        # Temporary output name for this extension
        if tmpfn.endswith(".fits"):
            basename = tmpfn[:-5]
        else:
            basename = tmpfn
        outtmpfn = "%sSCI%dtab.fits" % (basename, extver)
        objtmpfn  ="%sSCI%dimg.fits" % (basename, extver)

        # Should the saturation level be dynamically set? Setting the 
        # saturation level according to the image (necessary for coadded 
        # NIRI images). If the keyword BUNIT is not present, assume the 
        # image is in ADU. Note that this saturation level assumes that any 
        # stacked images are averaged rather than added, and at some point 
        # this will need to be addressed (probably in the descriptor).
        if set_saturation:
            if sciext.get_key_value('BUNIT') == 'electron':
                satur_level = sciext.saturation_level().as_pytype() * \
                    sciext.gain().as_pytype()
            else:
                satur_level = sciext.saturation_level().as_pytype()

        # if no seeing estimate provided, run sextractor once with
        # default, then re-run to get proper stellar classification
        if seeing_estimate is None:
            iter = [0,1]
            #seeing_estimate = 0.15
        else:
            iter = [0]
                        
        problem = False
        for i in iter:
            
            sx_cmd = ["sex", scitmpfn,
                      "-c",dd["sex"],
                      "-FLAG_IMAGE",dqtmpfn,
                      "-CATALOG_NAME",outtmpfn,
                      "-PARAMETERS_NAME",dd["param"],
                      "-FILTER_NAME",dd["conv"],
                      "-STARNNW_NAME",dd["nnw"],
                      "-CHECKIMAGE_NAME",objtmpfn,
                      "-CHECKIMAGE_TYPE","OBJECTS",
                      ]

            # Setting the saturation level dynamically is required
            if set_saturation:
                extend_line = ("-SATUR_LEVEL,", str(satur_level))
                sx_cmd.extend(extend_line)
                
            # If there is a seeing estimate available, use it
            if seeing_estimate is not None:
                extend_line = ("-SEEING_FWHM", str(seeing_estimate))
                sx_cmd.extend(extend_line)
            # Otherwise, if the observation is AO, then set a static low 
            # starting value
            elif ad.is_ao().as_pytype():
                #ao_seeing_est = ad.pixel_scale().as_pytype() * 3.0
                #extend_line = ("-SEEING_FWHM", str(ao_seeing_est))                
                seeing_estimate = 0.15
                extend_line = ("-SEEING_FWHM", str(seeing_estimate))
                sx_cmd.extend(extend_line)       
               
            log.fullinfo("Calling SExtractor on [SCI,%d] with "\
                         "seeing estimate %s" % (extver,str(seeing_estimate)))
            try:
                pipe_out = subprocess.Popen(sx_cmd,
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT)
            except OSError:
                os.remove(tmpfn)
                raise Errors.ScienceError("SExtractor failed")

            # Sextractor output is full of non-ascii characters, throw
            # it away for now
            stdoutdata = pipe_out.communicate()[0]

            try:
                hdulist = pf.open(outtmpfn)
            except IOError:
                problem = True
                log.stdinfo("No sources found in %s[SCI,%d] "
                            "(SExtractor output not available)" %
                            (ad.filename,extver))
                break

            # If sextractor returned no data, don't bother with the
            # next iteration
            if len(hdulist) <= 1:
                problem = True
                log.stdinfo("No sources found in %s[SCI,%d] " 
                            "(SExtractor returned no data)" %
                            (ad.filename,extver))
                break
            else:
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
                dqpix = tdata["NIMAFLAGS_ISO"]
            else:
                dqflag = np.zeros_like(sxflag)
                dqpix = dqflag
            aflag = np.where(tdata["ISOAREA_IMAGE"]<20,1,0)
            tflag = np.where(tdata["B_IMAGE"]<1.1,1,0)
            eflag = np.where(tdata["ELLIPTICITY"]>0.5,1,0)
            sflag = np.where(tdata["CLASS_STAR"]<0.8,1,0)
            iflag = np.where(dqpix>=0.9*tdata["ISOAREA_IMAGE"],1,0)
            snflag = np.where(tdata["FLUX_AUTO"] < 
                              25*tdata["FLUXERR_AUTO"], 1, 0)

            # Bitwise-or all the flags
            flags = sxflag | dqflag | aflag | eflag | sflag | snflag | tflag
            #log.fullinfo ("Aflags  are "+str(aflag))
            #log.fullinfo ("Eflags  are "+str(eflag))
            #log.fullinfo ("Sflags  are "+str(sflag))
            #log.fullinfo ("SNflags are "+str(snflag))
            #log.fullinfo ("SXflags are "+str(sxflag))
            #log.fullinfo ("DQflags are "+str(dqflag))

            flagged_fwhm = fwhm[flags==0]

            # Throw out FWHM values of zero which means SExtractor has 
            # not found a FWHM for these objects
            good_fwhm = flagged_fwhm[flagged_fwhm!=0]
            #log.fullinfo ("I've got %d sources" % len(good_fwhm))
            if len(good_fwhm)>3:
                # Clip outliers in FWHM - single 1-sigma clip if 
                # more than 3 sources.
                mean = good_fwhm.mean()
                sigma = good_fwhm.std()
                good_fwhm = good_fwhm[(good_fwhm<mean+sigma) & 
                                      (good_fwhm>mean-sigma)]
                seeing_estimate = good_fwhm.mean()

                if np.isnan(seeing_estimate) or seeing_estimate==0:
                    seeing_estimate = None
                    break
            elif len(good_fwhm)>=1:
                seeing_estimate = good_fwhm.mean()
                if np.isnan(seeing_estimate) or seeing_estimate==0:
                    seeing_estimate = None
                    break
            else:
                seeing_estimate = None
                break

        try:
            os.remove(outtmpfn)
        except:
            pass
        if not problem:
            columns = {}
            for col in tcols:
                columns[col.name] = col
                # Cull things <20 pixels in area or <1.1 pixels wide
                columns[col.name].array = \
                    columns[col.name].array[aflag+tflag+iflag==0]
            result[dict_key] = columns

            nobj = len(columns["NUMBER"].array)
            log.stdinfo("Found %d sources in %s[SCI,%d]" %
                        (nobj,ad.filename,extver))
            # Reset NUMBER to be a consecutive list
            columns["NUMBER"].array = np.array(range(1,nobj+1))

            # Add OBJCAT
            ad = gt.add_objcat(adinput=ad, extver=extver, 
                               replace=True, columns=columns,
                               sxdict=sxdict)[0]

            # Read in object mask
            # >>>> *** WARNING:: The FITS  file  objtmpfn does not
            # This works OK for pyfits < 3.0, but NOT for newer version.

            # have a PHU (first card is XTENSION not SIMPLE as it
            # should). Fortunately AstroData can take care of this.
            #mask_hdu = pf.open(objtmpfn)[0]
            #mask_hdu.header.add_comment("Object mask created by SExtractor.")
            #mask_hdu.header.add_comment("0 indicates no object, 1 "\
            #                            "indicates object")
            #print 'MHDU00:',mask_hdu.header.items()
            #mask_ad = AstroData(mask_hdu)
            ## By default the newly AD object has 'SCI' as EXTNAME value
            #print '\nMAK00:',mask_ad.info()
            #print '\nMAK01:',mask_ad.header.items(),extver,len(mask_ad)

            
            # Solution is to open the XTENSION fits file with AstroData. It
            # creates a PHU.
            mask_ad = AstroData(objtmpfn)
            mask_ad.rename_ext("OBJMASK",extver)
            mask_hd = mask_ad['OBJMASK'].header
            mask_hd.add_comment("Object mask created by SExtractor.")
            mask_hd.add_comment("0 indicates no object, 1 indicates object")

            mask_ad.data = np.where(mask_ad.data!=0,1,0).astype(np.int16)

            # Remove old object mask if it exists 
            # (ie. this is a re-run of detectSources)
            old_mask = ad["OBJMASK",extver]
            if old_mask is not None:
                ad.remove(("OBJMASK",extver))
            ad.append(mask_ad)
            os.remove(objtmpfn)

    log.fullinfo("Removing temporary file from disk: %s" % tmpfn)
    os.remove(tmpfn)
    return ad

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

def _profile_sources(ad):
    
    for sciext in ad["SCI"]:
        extver = sciext.extver()
        objcat = ad["OBJCAT",extver]
        if objcat is None:
            continue

        catx = objcat.data.field("X_IMAGE")
        caty = objcat.data.field("Y_IMAGE")
        catfwhm = objcat.data.field("FWHM_IMAGE")
        catbg = objcat.data.field("BACKGROUND")
        cattotalflux = objcat.data.field("FLUX_AUTO")
        catmaxflux = objcat.data.field("FLUX_MAX")
        data = sciext.data
        stamp_size = max(10,int(0.5/sciext.pixel_scale()))
        # Make a default grid to use for distance measurements
        dist = np.mgrid[-stamp_size:stamp_size,-stamp_size:stamp_size]+0.5

        fwhm_list = []
        e50d_list = []
        newmax_list = []
        for i in range(0,len(objcat.data)):
            xc = catx[i]
            yc = caty[i]
            bg = catbg[i]
            tf = cattotalflux[i]
            mf = catmaxflux[i]
            
            xc -= 0.5
            yc -= 0.5

            # Check that there's enough room for a stamp
            sz = stamp_size
            if (int(yc)-sz<0 or int(xc)-sz<0 or
                int(yc)+sz>=data.shape[0] or int(xc)+sz>=data.shape[1]):
                fwhm_list.append(-999)
                e50d_list.append(-999)
                newmax_list.append(mf)
                continue

            # Estimate new FLUX_MAX from pixels around peak
            mf = np.max(data[int(yc)-2:int(yc)+3,int(xc)-2:int(xc)+3]) - bg
            # Bright sources in IR images can "volcano", so revert to
            # catalog value if these pixels are negative
            if mf < 0:
                mf = catmaxflux[i]

            # Get image stamp around center point
            stamp=data[int(yc)-sz:int(yc)+sz,int(xc)-sz:int(xc)+sz]

            # Reset grid to correct center coordinates
            shift_dist = dist.copy()
            shift_dist[0] += int(yc)-yc
            shift_dist[1] += int(xc)-xc
    
            # Square root of the sum of the squares of the distances
            rdist = np.sqrt(np.sum(shift_dist**2,axis=0))

            # Radius and flux arrays for the radial profile
            rpr = rdist.flatten()
            rpv = stamp.flatten() - bg
    
            # Sort by the radius
            sort_order = np.argsort(rpr) 
            radius = rpr[sort_order]
            flux = rpv[sort_order]

            # Find the first 10 points below the half-flux
            # Average the radius of the first point below and
            # the last point above the half-flux
            halfflux = mf / 2.0
            below = np.where(flux<=halfflux)[0]
            if below.size>0:
                if len(below)>=10:
                    first_below = below[0:10]
                else:
                    first_below = below
                inner = radius[first_below[0]]
                if first_below[0]>0:
                    min = first_below[0]-1
                else:
                    min = first_below[0]
                nearest_r = radius[min:first_below[-1]]
                nearest_f = flux[min:first_below[-1]]
                possible_outer = nearest_r[nearest_f>=halfflux]
                if possible_outer.size>0:
                    outer = np.max(possible_outer)
                    hwhm = 0.5 * (inner + outer)
                else:
                    hwhm = None
            else:
                hwhm = None

            # Resort by radius [CJS: Not needed!]
            #sort_order = np.argsort(rpr) 
            #radius = rpr[sort_order]
            #flux = rpv[sort_order]

            # Find the first radius that encircles half the total flux
            sumflux = np.cumsum(flux)
            halfflux = 0.5 * tf
            first_50pflux = np.where(sumflux>=halfflux)[0]
            if first_50pflux.size>0:
                e50r = radius[first_50pflux[0]]
            else:
                e50r = None

            if hwhm is not None:
                fwhm_list.append(hwhm*2.0)
            else:
                fwhm_list.append(-999)
            if e50r is not None:
                e50d_list.append(e50r*2.0)
            else:
                e50d_list.append(-999)
            newmax_list.append(mf)

        objcat.data.field("PROFILE_FWHM")[:] = np.array(fwhm_list)
        objcat.data.field("PROFILE_EE50")[:] = np.array(e50d_list)
        objcat.data.field("FLUX_MAX")[:] = np.array(newmax_list)

    return ad


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
    from gempy.library import astrotools as at

    if ext is None:
        sciexts = ad["SCI"]
    else:
        sciexts = ad["SCI",ext]

    good_source = []
    for sciext in sciexts:
        extver = sciext.extver()

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
                objcat.data.field("CLASS_STAR")[obji] = 0.96
            elif ellip<0.3:
                objcat.data.field("CLASS_STAR")[obji] = 0.91
            elif ellip<0.5:
                objcat.data.field("CLASS_STAR")[obji] = 0.2
            else:
                objcat.data.field("CLASS_STAR")[obji] = 0.2

            if fwhm<1.0:
                # likely cosmic ray
                objcat.data.field("CLASS_STAR")[obji] *= 0.2
            elif fwhm<2*default_fwhm:
                # potential star
                objcat.data.field("CLASS_STAR")[obji] *= 1.0
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