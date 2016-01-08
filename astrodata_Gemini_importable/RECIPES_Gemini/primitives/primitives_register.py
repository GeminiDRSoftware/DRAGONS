import numpy as np
import pywcs
import math
from astrodata import AstroData
from astrodata.utils import Errors
from astrodata.utils import logutils
from astrodata.utils import Lookups
from astrodata.utils.ConfigSpace  import lookup_path
from gempy.library import astrotools as at
from gempy.gemini import gemini_tools as gt
from primitives_GENERAL import GENERALPrimitives

# Load the standard comments for header keywords that will be updated
# in these functions
keyword_comments = Lookups.get_lookup_table("Gemini/keyword_comments",
                                            "keyword_comments")

class RegisterPrimitives(GENERALPrimitives):
    """
    This is the class containing all of the registration primitives for the
    GEMINI level of the type hierarchy tree. It inherits all the primitives
    from the level above, 'GENERALPrimitives'.
    """
    astrotype = "GEMINI"
    
    def init(self, rc):
        GENERALPrimitives.init(self, rc)
        return rc
    init.pt_hide = True
    
    def correctWCSToReferenceFrame(self, rc):
        """ 
        This primitive registers images to a reference image by correcting
        the relative error in their world coordinate systems. The function
        uses points of reference common to the reference image and the
        input images to fit the input WCS to the reference one. The fit
        is done by a least-squares minimization of the difference between
        the reference points in the input image pixel coordinate system.
        This function is intended to be followed by the
        align_to_reference_image function, which applies the relative
        transformation encoded in the WCS to transform input images into the
        reference image pixel coordinate system.
        
        The primary registration method is intended to be by direct mapping
        of sources in the image frame to correlated sources in the reference
        frame. This method fails when there are no correlated sources in the
        field, or when the WCSs are very far off to begin with. As a back-up
        method, the user can try correcting the WCS by the shifts indicated 
        in the POFFSET and QOFFSET header keywords (option fallback='header'), 
        By default, only the direct method is
        attempted, as it is expected that the relative WCS will generally be
        more correct than either indirect method. If the user prefers not to
        attempt direct mapping at all, they may set method to 'header'.
        
        In order to use the direct mapping method, sources must have been
        detected in the frame and attached to the AstroData instance in an 
        OBJCAT extension. This can be accomplished via the detectSources
        primitive. Running time is optimal, and sometimes the solution is 
        more robust, when there are not too many sources in the OBJCAT. Try
        running detectSources with threshold=20. The solution may also be
        more robust if sub-optimal sources are rejected from the set of 
        correlated sources (use option cull_sources=True). This option may
        substantially increase the running time if there are many sources in
        the OBJCAT.
        
        It is expected that the relative difference between the WCSs of 
        images to be combined should be quite small, so it may not be necessary
        to allow rotation and scaling degrees of freedom when fitting the image
        WCS to the reference WCS. However, if it is desired, the options 
        rotate and scale can be used to allow these degrees of freedom. Note
        that these options refer to rotation/scaling of the WCS itself, not the
        images. Significant rotation and scaling of the images themselves 
        will generally already be encoded in the WCS, and will be corrected for
        when the images are aligned.
        
        The WCS keywords in the headers of the output images are updated
        to contain the optimal registration solution.
        
        :param method: method to use to generate reference points. Options
                       are 'sources' to directly map sources from the input
                       image to the reference image,
                       or 'header' to generate reference points from the 
                       POFFSET and QOFFSET keywords in the image headers.
        :type method: string, either 'sources' or 'header'
        
        :param fallback: back-up method for generating reference points.
                         if the primary method fails. The 'sources' option
                         cannot be used as the fallback.
        :type fallback: string, either 'header' or None.
        
        :param use_wcs: the alignment method will use the encoded WCS in the 
                        header for the initial adjustment of the frames. The
                        alternative is that the shifts and rotation from the
                        header will be used.
        :type use_wcs: bool

        :param first_pass: estimated maximum distance between correlated
                           sources. This distance represents the expected
                           mismatch between the WCSs or header shifts of 
                           the input images.
        :type first_pass: float
        
        :param min_sources: minimum number of sources to use for cross-
                            correlation, depending on the instrument used.
        :type min_sources: int                    
                            
        :param cull_sources: flag to indicate whether sub-optimal sources 
                             should be rejected before attempting a direct
                             mapping. If True, sources that are saturated,
                             or otherwise unlikely to be point sources
                             will be eliminated from the list of reference
                             points.
        :type cull_sources: bool
        
        :param rotate: flag to indicate whether the input image WCSs should
                       be allowed to rotate with respect to the reference image
                       WCS
        :type rotate: bool
        
        :param scale: flag to indicate whether the input image WCSs should
                      be allowed to scale with respect to the reference image
                      WCS. The same scale factor is applied to all dimensions.
        :type scale: bool
        """
        
        # Instantiate the log
        log = logutils.get_logger(__name__)

        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "correctWCSToReferenceFrame",
                                 "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["correctWCSToReferenceFrame"]

        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Check whether two or more input AstroData objects were provided
        adinput = rc.get_inputs_as_astrodata()
        correcting = True
        if len(adinput) <= 1:
            log.warning("No correction will be performed, since at least " \
                        "two input AstroData objects are required for " \
                        "correctWCSToReferenceFrame")
            # Set the input AstroData object list equal to the output AstroData
            # objects list without further processing
            adoutput_list = adinput
            correcting = False

        # Check that method is sensibly defined
        if correcting:

            # Get the necessary parameters from the RC
            method = rc["method"]
            fallback = rc["fallback"]
            use_wcs = rc["use_wcs"]
            first_pass = rc["first_pass"]
            min_sources = rc["min_sources"]
            cull_sources = rc["cull_sources"]
            rotate = rc["rotate"]
            scale = rc["scale"]

            if method=="None":
                method=None
            if fallback=="None":
                fallback=None
            
            if method is None:
                if fallback is None:
                    log.warning("No correction will be performed, since both " \
                                "method and fallback are None")
                    adoutput_list = adinput
                    correcting = False
                else:
                    method = fallback
                
        # Check that images have one SCI extension, and if necessary,
        # sources defined in an OBJCAT extension
        if correcting:
            n_test = []
            for ad in adinput:

                # Make sure all images have one science extension
                if len(ad["SCI"])!=1:
                    raise Errors.InputError("Input images must have only one "
                                            "SCI extension.")
                
                # Get number of objects from OBJCAT
                objcat = ad["OBJCAT"]
                if objcat is None:
                    num_cat = 0
                else:
                    num_cat = len(objcat)
                if num_cat==0:
                    n_obj = 0
                elif num_cat>1:
                    raise Errors.InputError("Input images must have only one " +
                                            "OBJCAT extension.")
                else:
                    n_obj = len(objcat.data)
                
                n_test.append(n_obj)
            
            if n_test[0]==0 and method=="sources":
                log.warning("No objects found in reference image.")
                if fallback is not None:
                    log.warning("Only attempting indirect WCS alignment, " +
                                "via " + fallback + " mapping")
                    method=fallback
                
                else:
                    log.warning("WCS can only be corrected indirectly " +
                                "and fallback method is set to None. Not " +
                                "attempting WCS correction.")
                    adoutput_list = adinput
                    correcting = False

        # If input passed all checks, apply the selected method
        if correcting:
            
            # Reference image is first one supplied
            # (won't be modified)
            reference = adinput[0]
            adoutput_list.append(reference)
            log.stdinfo("Reference image: "+reference.filename)
            
            # If no OBJCAT/no sources in reference image, or user choice,
            # use indirect alignment for all images at once
            if method=="header":
                log.stdinfo("Using WCS specified in header for alignment")
                reg_ad = _header_align(reference, adinput[1:])
                adoutput_list.extend(reg_ad)
            elif method!="sources":
                raise Errors.InputError("Did not recognize method " + method)
            
            # otherwise try to do direct alignment for each image by correlating
            # sources in the reference and input images
            else:
                
                for i in range(1,len(adinput)):
                
                    ad = adinput[i]
                    
                    if n_test[i] == 0:
                        log.warning("No objects found in "+ ad.filename)
                        if fallback is not None:
                            log.warning(
                                "Only attempting indirect WCS alignment, " +
                                "via " + fallback + " mapping")
                            if fallback=="header":
                                adoutput = _header_align(reference, ad)
                            else:
                                raise Errors.InputError(
                                    "Did not recognize fallback method " + 
                                    fallback)
                        
                        else:
                            log.warning(
                                "WCS can only be corrected indirectly "+
                                "and fallback=None. Not attempting WCS " +
                                "correction for " + ad.filename)
                            adoutput_list.append(ad)
                            continue
                    else:
                        
                        log.fullinfo("Number of objects in image %s: %d" %
                                     (ad.filename, n_test[i]))
                        
                        log.fullinfo("Cross-correlating sources in %s, %s" %
                                   (reference.filename, ad.filename))
                        firstpasspix = first_pass / ad.pixel_scale()
                        if use_wcs is True:
                            obj_list = _correlate_sources(reference, ad, 
                                                          firstPass=firstpasspix,
                                                          min_sources=min_sources,
                                                          cull_sources=cull_sources)
                        else:
                            obj_list = _correlate_sources_offsets(reference, ad, 
                                                          firstPass=firstpasspix,
                                                          min_sources=min_sources,
                                                          cull_sources=cull_sources)
                            
                        n_corr = len(obj_list[0])
                        if n_corr==0:
                            log.warning("No correlated sources found.")
                            if fallback is not None:
                                log.warning("Only attempting indirect WCS " +
                                            "alignment, via " + fallback + 
                                            " mapping")
                                
                                if fallback=="header":
                                    adoutput = _header_align(reference, ad)
                                else:
                                    raise Errors.InputError(
                                        "Did not recognize " +
                                        "fallback method " + fallback)
                            
                            else:
                                log.warning(
                                    "WCS can only be corrected indirectly "+
                                    "and fallback=None. Not attempting " +
                                    "WCS correction for " + ad.filename)
                                adoutput_list.append(ad)
                                continue
                        else:
                            log.fullinfo("Number of correlated sources: %d" % 
                                         n_corr)
                            log.stdinfo("%s: Using source correlation for "
                                        "alignment" % ad.filename)

                            # Check the fit geometry depending on the 
                            # number of objects
                            if n_corr == 1:
                                log.warning("Too few objects. Setting " +
                                            "rotate=False, " +
                                            "scale=False")
                                rotate=False
                                scale=False
                            
                            log.fullinfo("\nSources used to align frames:")
                            log.fullinfo("  %7s %7s %7s %7s\n%s" % 
                                         (" Ref. x","Ref. y",
                                          "Img. x","Img. y",
                                          "  "+"-"*31))
                            output_obj = zip(obj_list[0],obj_list[1])
                            for obj in output_obj:
                                obj_string = ("  %7.2f %7.2f %7.2f %7.2f" % 
                                              (obj[0][0],obj[0][1],
                                               obj[1][0],obj[1][1]))
                                log.fullinfo(obj_string)
                            log.fullinfo("")
                            
                            adoutput = _align_wcs(reference, ad, [obj_list], 
                                                  rotate=rotate, scale=scale)
                    
                    adoutput_list.extend(adoutput)

            # Change the filenames and add the appropriate timestamps
            for ad in adoutput_list:
                gt.mark_history(adinput=ad, keyword=timestamp_key)

                ad.filename = gt.filename_updater(adinput=ad, 
                                                  suffix=rc["suffix"], 
                                                  strip=True)
        
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc
    
    def determineAstrometricSolution(self, rc):
        """
        This primitive calculates the average astrometric offset between
        the positions of sources in the reference catalog, and their
        corresponding object in the object catalog.
        It then reports the astrometric correction vector.
        For now, this is limited to a translational offset only.
        
        The solution is stored in a WCS object in the RC.  It can
        be applied to the image headers by calling the updateWCS
        primitive.
        """

        # Instantiate the log
        log = logutils.get_logger(__name__)


        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "determineAstrometricSolution",
                                 "starting"))

        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["determineAstrometricSolution"]

        # Initialize the list of output AstroData objects
        adoutput_list = []

        # Loop over each input AstroData object in the input list
        wcs_dict = {}
        for ad in rc.get_inputs_as_astrodata():
            wcs_dict[ad] = {}

            # Objcats to process
            objcats = ad['OBJCAT']
            if objcats is None:
                objcats = []
            # If there are *no* reference catalogs, don't even bother
            if ad['REFCAT'] is None:
                objcats = []

            # Initialise lists to keep the total offsets in
            all_delta_ra = []
            all_delta_dec = []

            # Make an empty dictionary, to be used for reporting values to
            # the fits store
            info_dict = {}

            # Loop over the OBJCAT extensions
            for objcat in objcats:
                extver = objcat.extver()

                # Check that a refcat exists for this objcat extver
                refcat = ad['REFCAT',extver]
                if(not(refcat)):
                    log.warning("Missing [REFCAT,%d] in %s - cannot calculate astrometry" % (extver, ad.filename))
                else:
                    # Initialise lists to keep the offsets in
                    delta_ra = []
                    delta_dec = []

                    # Need to build an index array that lets us quickly find the
                    # row in the refcat, given the id from the objcat
                    m = max(refcat.data['Id'])
                    m = m+1
                    refindex=([None] * m)
                    # Loop through the refcat, populating the index
                    for i in range(len(refcat.data)):
                        ref=refcat.data[i]
                        refindex[ref['Id']] = i
                    # so now you can find a refid with refcat.data[refindex[refid]]


                    # Loop through the objcat, 
                    for obj in objcat.data:
                        if(obj['REF_NUMBER'] != -999):
                            refid = obj['REF_NUMBER']
                            obj_ra = obj['X_WORLD']
                            obj_dec = obj['Y_WORLD']
                            ref_ra = None
                            ref_dec = None
                            # Find the reference catalog line.
                            ref = refcat.data[refindex[refid]]
                            if(ref['Id'] == refid):
                                ref_ra = ref['RAJ2000']
                                ref_dec = ref['DEJ2000']
                            dra = ref_ra - obj_ra
                            ddec = ref_dec - obj_dec
                            if(np.isfinite(dra) and np.isfinite(ddec)):
                                delta_ra.append(dra)
                                delta_dec.append(ddec)
                                all_delta_ra.append(dra)
                                all_delta_dec.append(ddec)

                    if not delta_ra or not delta_dec:
                        log.fullinfo("No reference sources in %s[SCI,extver]" % ad.filename)
                        continue

                    # Report the mean and standard deviation of the offsets:
                    ra_mean = np.mean(delta_ra) * 3600.0
                    ra_sigma = np.std(delta_ra) * 3600.0
                    dec_mean = np.mean(delta_dec) * 3600.0
                    dec_sigma = np.std(delta_dec) * 3600.0
                    ra_median_degs = np.median(delta_ra)
                    dec_median_degs = np.median(delta_dec)
                    ra_median = ra_median_degs * 3600.0
                    dec_median = dec_median_degs * 3600.0

                    log.fullinfo("Astrometric Offset between [OBJCAT, %d] and [REFCAT, %d] is:" % (extver, extver))
                    log.fullinfo("RA_mean +- RA_sigma: %.2f +- %.2f arcsec" % (ra_mean, ra_sigma))
                    log.fullinfo("Dec_mean +- Dec_sigma: %.2f +- %.2f arcsec" % (dec_mean, dec_sigma))
                    log.fullinfo("Median Offset is: %.2f, %.2f arcsec" % (ra_median, dec_median))

                    # Store it in the fitsstore info_dict
                    info_dict[("SCI",extver)] = {"dra":ra_mean,
                                                 "dra_std":ra_sigma,
                                                 "ddec":dec_mean,
                                                 "ddec_std":dec_sigma,
                                                 "nsamples":len(delta_ra)}

                    # Store the changes in a pywcs.WCS object so they
                    # can be applied by the updateWCS primitive if desired
                    sci = ad["SCI",extver]
                    wcs = pywcs.WCS(sci.header)
                    wcs.wcs.crval = np.array([wcs.wcs.crval[0]+ra_median_degs,
                                              wcs.wcs.crval[1]+dec_median_degs])
                    wcs_dict[ad][extver] = wcs

            if(objcats and all_delta_ra and all_delta_dec):
                # Report the mean and standard deviation of all the offsets over all the sci extensions:
                ra_mean = np.mean(all_delta_ra) * 3600.0
                ra_sigma = np.std(all_delta_ra) * 3600.0
                dec_mean = np.mean(all_delta_dec) * 3600.0
                dec_sigma = np.std(all_delta_dec) * 3600.0

                log.stdinfo("Mean Astrometric Offset between OBJCAT and REFCAT for %s:" % ad.filename)
                log.stdinfo("     RA: %.2f +- %.2f    Dec: %.2f +- %.2f   arcsec" % (ra_mean, ra_sigma, dec_mean, dec_sigma))

            else:
                log.stdinfo("Could not determine astrometric offset for %s" %
                            ad.filename)
                
            # Report the measurement to the fitsstore
            fitsdict = gt.fitsstore_report(ad,rc,"pe",info_dict)

            adoutput_list.append(ad)

        # Store the WCS solution in the RC
        rc["wcs"] = wcs_dict

        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)

        yield rc

    def updateWCS(self, rc):
        """
        This primitive applies a previously calculated WCS correction.
        The solution should be stored in the RC as a dictionary, with
        astrodata instances as the keys and pywcs.WCS objects as the
        values.
        """

        # Instantiate the log
        log = logutils.get_logger(__name__)


        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "updateWCS",
                                 "starting"))

        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["updateWCS"]

        # Initialize the list of output AstroData objects
        adoutput_list = []

        # Get the necessary parameters from the RC
        wcs = rc["wcs"]
        if wcs is None:
            log.warning("No new WCS supplied; no correction will be "\
                        "performed.")
        else:

            # Loop over each input AstroData object in the input list
            for ad in rc.get_inputs_as_astrodata():

                ad_wcs = None
                if isinstance(wcs,dict):
                    try:
                        ad_wcs = wcs[ad]
                    except KeyError:
                        ad_wcs = wcs
                elif isinstance(wcs,pywcs.WCS):
                    ad_wcs = wcs

                if ad_wcs is None:
                    log.warning("No new WCS supplied for %s; "\
                                "no correction will be performed" %
                                ad.filename)
                    adoutput_list.append(ad)
                    continue

                for ext in ad:
                    extname = ext.extname()
                    extver = ext.extver()
                    
                    ext_wcs = None
                    if isinstance(ad_wcs,dict):
                        try:
                            ext_wcs = ad_wcs[extver]
                        except KeyError:
                            pass
                    elif isinstance(ad_wcs,pywcs.WCS):
                        ext_wcs = wcs

                    if ext_wcs is None:
                        log.warning("No new WCS supplied for %s[%s,%d]; "\
                                    "no correction will be performed" %
                                    (ad.filename,extname,extver))
                        continue
                    elif not isinstance(ext_wcs, pywcs.WCS):
                        raise Errors.InputError("Parameter wcs must be "\
                                                "either a pywcs.WCS object "\
                                                "or a dictionary of pywcs.WCS "\
                                                "objects")
                  
                    # If image extension, correct the header values
                    if extname in ["SCI","VAR","DQ"]:
                        log.fullinfo("Correcting CRVAL, CRPIX, and CD in "\
                                     "image extension headers for %s[%s,%d]" %
                                     (ad.filename,extname,extver))
                        log.fullinfo("CRVAL: "+repr(ext_wcs.wcs.crval))
                        log.fullinfo("CRPIX: "+repr(ext_wcs.wcs.crpix))
                        log.fullinfo("CD: "+repr(ext_wcs.wcs.cd))

                        ext.set_key_value("CRVAL1", ext_wcs.wcs.crval[0],
                                          comment=keyword_comments["CRVAL1"])
                        ext.set_key_value("CRVAL2", ext_wcs.wcs.crval[1],
                                          comment=keyword_comments["CRVAL2"])
                        ext.set_key_value("CRPIX1", ext_wcs.wcs.crpix[0],
                                          comment=keyword_comments["CRPIX1"])
                        ext.set_key_value("CRPIX2", ext_wcs.wcs.crpix[1],
                                          comment=keyword_comments["CRPIX2"])
                        ext.set_key_value("CD1_1", ext_wcs.wcs.cd[0,0],
                                          comment=keyword_comments["CD1_1"])
                        ext.set_key_value("CD1_2", ext_wcs.wcs.cd[0,1],
                                          comment=keyword_comments["CD1_2"])
                        ext.set_key_value("CD2_1", ext_wcs.wcs.cd[1,0],
                                          comment=keyword_comments["CD2_1"])
                        ext.set_key_value("CD2_2", ext_wcs.wcs.cd[1,1],
                                          comment=keyword_comments["CD2_2"])

                    # If objcat, fix the RA/Dec columns
                    elif extname=="OBJCAT":
                        log.fullinfo("Correcting RA, Dec columns in OBJCAT "\
                                     "extension for %s[%s,%d]" %
                                     (ad.filename,extname,extver))
                        for row in ext.data:
                            xy = np.array([row['X_IMAGE'], row['Y_IMAGE']])
                            radec = ext_wcs.wcs_pix2sky([xy], 1)[0]
                            # FIXME - is it correct to set oring to 1 here?
                            # Also we should be setting ra_dec_order=True, but
                            # that breaks with the wcs missing the lattype
                            # property
                            row['X_WORLD'] = radec[0]
                            row['Y_WORLD'] = radec[1]

            

                # Add the appropriate time stamps to the PHU
                gt.mark_history(adinput=ad, keyword=timestamp_key)

                # Change the filename
                ad.filename = gt.filename_updater(adinput=ad, 
                                                  suffix=rc["suffix"],
                                                  strip=True)

                adoutput_list.append(ad)
        
            # Report the list of output AstroData objects to the reduction
            # context
            rc.report_output(adoutput_list)

        yield rc

##############################################################################
# Below are the helper functions for the user level functions in this module #
##############################################################################

def _correlate_sources(ad1, ad2, delta=None, firstPass=10, min_sources=1, cull_sources=False):
    """
    This function takes sources from the OBJCAT extensions in two
    images and attempts to correlate them, using the WCS encoded in the
    headers as a first approximation. It returns a list of reference source 
    positions and their correlated image source positions.
    
    :param ad1: reference image
    :type ad1: AstroData instance
    
    :param ad2: input image
    :type ad2: AstroData instance
    
    :param delta: maximum distance in pixels to allow a match. If
                  left as None, it will attempt to find an appropriate
                  number (recommended).
    :type delta: float
    
    :param firstPass: estimated maximum distance between correlated
                      sources. This distance represents the expected
                      mismatch between the WCSs of the input images.
    :type firstPass: float
    
    :param min_sources: minimum number of sources to use for cross-
                        correlation, depending on the instrument used.
    :type min_sources: int                    

    :param cull_sources: flag to indicate whether to reject sources that
                         are insufficiently star-like
    :type cull_sources: bool
    """
    
    log = logutils.get_logger(__name__)

    # If desired, clip out the most star-like sources in the OBJCAT
    if cull_sources:
        good_src_1 = gt.clip_sources(ad1)[("SCI",1)]
        good_src_2 = gt.clip_sources(ad2)[("SCI",1)]
        if len(good_src_1) < min_sources or len(good_src_2) < min_sources:
            log.warning("Too few sources in culled list, using full set "\
                        "of sources")
            x1 = ad1["OBJCAT"].data.field("X_IMAGE")
            y1 = ad1["OBJCAT"].data.field("Y_IMAGE")
            x2 = ad2["OBJCAT"].data.field("X_IMAGE")
            y2 = ad2["OBJCAT"].data.field("Y_IMAGE")
        else:
            x1 = good_src_1["x"]
            y1 = good_src_1["y"]
            x2 = good_src_2["x"]
            y2 = good_src_2["y"]
    else:
        # Otherwise, just get all sources
        x1 = ad1["OBJCAT"].data.field("X_IMAGE")
        y1 = ad1["OBJCAT"].data.field("Y_IMAGE")
        x2 = ad2["OBJCAT"].data.field("X_IMAGE")
        y2 = ad2["OBJCAT"].data.field("Y_IMAGE")

    # get WCS from both images
    wcs1 = pywcs.WCS(ad1["SCI"].header)
    wcs2 = pywcs.WCS(ad2["SCI"].header)
    
    # convert image 2 data to sky coordinates
    ra2, dec2 = wcs2.wcs_pix2sky(x2,y2,1)
    
    # convert image 2 sky data to image 1 pixel coordinates
    conv_x2, conv_y2 = wcs1.wcs_sky2pix(ra2,dec2,1)
    
    # find matches
    ind1,ind2 = at.match_cxy(x1,conv_x2,y1,conv_y2,
                             delta=delta, firstPass=firstPass, log=log)

    if len(ind1)!=len(ind2):
        raise Errors.ScienceError("Mismatched arrays returned from match_cxy")
    
    if len(ind1)<1 or len(ind2)<1:
        return [[],[]]
    else:
        obj_list = [zip(x1[ind1], y1[ind1]),
                    zip(x2[ind2], y2[ind2])]
        return obj_list

def _correlate_sources_offsets(ad1, ad2, delta=None, firstPass=10, min_sources=1, cull_sources=False):
    """
    This function takes sources from the OBJCAT extensions in two
    images and attempts to correlate them, using the offsets provided 
    in the header as a first approximation. It returns a list of 
    reference source positions and their correlated image source 
    positions. This is currently GNIRS-specific.
    
    :param ad1: reference image
    :type ad1: AstroData instance
    
    :param ad2: input image
    :type ad2: AstroData instance
    
    :param delta: maximum distance in pixels to allow a match. If
                  left as None, it will attempt to find an appropriate
                  number (recommended).
    :type delta: float
    
    :param firstPass: estimated maximum distance between correlated
                      sources. This distance represents the expected
                      mismatch between the header shifts of the input 
                      images.
    :type firstPass: float
    
    :param min_sources: minimum number of sources to use for cross-
                        correlation, depending on the instrument used.
    :type min_sources: int                    

    :param cull_sources: flag to indicate whether to reject sources that
                         are insufficiently star-like
    :type cull_sources: bool
    """
    
    log = logutils.get_logger(__name__)

    # If desired, clip out the most star-like sources in the OBJCAT
    if cull_sources:
        good_src_1 = gt.clip_sources(ad1)[("SCI",1)]
        good_src_2 = gt.clip_sources(ad2)[("SCI",1)]
        if len(good_src_1) < min_sources or len(good_src_2) < min_sources:
            log.warning("Too few sources in culled list, using full set "\
                        "of sources")
            x1 = ad1["OBJCAT"].data.field("X_IMAGE")
            y1 = ad1["OBJCAT"].data.field("Y_IMAGE")
            x2 = ad2["OBJCAT"].data.field("X_IMAGE")
            y2 = ad2["OBJCAT"].data.field("Y_IMAGE")
        else:
            x1 = good_src_1["x"]
            y1 = good_src_1["y"]
            x2 = good_src_2["x"]
            y2 = good_src_2["y"]
    else:
        # Otherwise, just get all sources
        x1 = ad1["OBJCAT"].data.field("X_IMAGE")
        y1 = ad1["OBJCAT"].data.field("Y_IMAGE")
        x2 = ad2["OBJCAT"].data.field("X_IMAGE")
        y2 = ad2["OBJCAT"].data.field("Y_IMAGE")    

    # Shift the catalog of image to be aligned using the header offsets.
    # The instrument alignment angle should be used to check how to 
    # use the offsets.
    poffset1 = ad1.phu_get_key_value("POFFSET")
    qoffset1 = ad1.phu_get_key_value("QOFFSET")
    poffset2 = ad2.phu_get_key_value("POFFSET")
    qoffset2 = ad2.phu_get_key_value("QOFFSET")
    pixscale = ad1.pixel_scale()
    xdiff = -1.0 * (qoffset1 - qoffset2) / pixscale
    ydiff = (poffset1 - poffset2) / pixscale
    
    pa1 = ad1.phu_get_key_value("PA")
    pa2 = ad2.phu_get_key_value("PA")
    if (abs(pa1 - pa2) < 1.0):
        conv_x2 = [(item - xdiff) for item in x2]
        conv_y2 = [(item - ydiff) for item in y2]
        log.fullinfo("Less than 1 degree of rotation between the frames, "
                    "no rotation applied.")
        log.fullinfo("dx = {} px, dy = {} px applied from the headers"
                    "".format(xdiff, ydiff, pa1 - pa2))
    else:
        theta = math.radians(pa1 - pa2)
    
        # Fetch the center of the frame
        inst = ad1.instrument()
        instlow = ad1.instrument().as_pytype().lower()
        lookup_dir = "Gemini/" + inst + "/" + inst + "CenterDict"
        lookup_name = instlow + "CenterDict"
        center_dict = Lookups.get_lookup_table(lookup_dir,
                                               lookup_name)
        
        key = "IMAGE"
        if key in center_dict:
            centerx, centery = center_dict[key]
            x2temp = [(item - xdiff - centerx) for item in x2]
            y2temp = [(item - ydiff - centery) for item in y2]
            x2trans = [((xt * math.cos(theta)) - (yt * math.sin(theta))) 
                       for xt, yt in zip(x2temp, y2temp)]
            y2trans = [((xt * math.sin(theta)) + (yt * math.cos(theta))) 
                       for xt, yt in zip(x2temp, y2temp)]
            conv_x2 = [(xt + centerx) for xt in x2trans]
            conv_y2 = [(yt + centery) for yt in y2trans]
            log.fullinfo("dx = {} px, dy = {} px, rotation = {} degrees "
                        "applied from the headers".format(xdiff, ydiff, pa1 - pa2))
        else:
            log.warning("No frame center found for {}, no rotation can "
                        "be applied.".format(ad2.filename))
            conv_x2 = [(item - xdiff) for item in x2]
            conv_y2 = [(item - ydiff) for item in y2]
            log.fullinfo("dx = {} px, dy = {} px applied from the headers"
                        "".format(xdiff, ydiff, pa1 - pa2))
   
    # find matches
    ind1,ind2 = at.match_cxy(x1,conv_x2,y1,conv_y2,
                             delta=delta, firstPass=firstPass, log=log)

    if len(ind1)!=len(ind2):
        raise Errors.ScienceError("Mismatched arrays returned from match_cxy")
    
    if len(ind1)<1 or len(ind2)<1:
        return [[],[]]
    else:
        obj_list = [zip(x1[ind1], y1[ind1]),
                    zip(x2[ind2], y2[ind2])]
        return obj_list

def _align_wcs(reference, adinput, objIns, rotate=False, scale=False):
    """
    This function fits an input image's WCS to a reference image's WCS
    by minimizing the difference in the input image frame between
    reference points present in both images.
    
    :param reference: reference image to register other images to. Must
                      have only one SCI extension.
    :type reference: AstroData object
    
    :param adinput: images to register to reference image. Must have
                  only one SCI extension.
    :type adinput: AstroData objects, either a single instance or a list
    
    :param objIns: list of object lists, one for each input image
    :type objIns: list of output lists from _correlate_sources
    
    :param rotate: flag to indicate whether the input image WCSs should
                   be allowed to rotate with respect to the reference image
                   WCS
    :type rotate: bool
    
    :param scale: flag to indicate whether the input image WCSs should
                  be allowed to scale with respect to the reference image
                  WCS. The same scale factor is applied to all dimensions.
    :type scale: bool
    """
    
    log = logutils.get_logger(__name__)
    
    if not isinstance(adinput,list):
        adinput = [adinput]
    if not isinstance(objIns,list):
        objIns = [objIns]
    if len(objIns) != len(adinput):
        raise Errors.InputError("Argument objIns should have the same " +
                                "number of elements as adinput")
    
    import scipy.optimize
    
    adoutput_list = []
    for i in range(len(adinput)):
        
        # copy input ad and rename
        ad = adinput[i]
        
        log.fullinfo("Adjusting WCS for " + ad.filename )
        
        ref_xy, inp_xy = objIns[i]
        ref_xy = np.array(ref_xy)
        inp_xy = np.array(inp_xy)
        
        ref_wcs = pywcs.WCS(reference["SCI"].header)
        inp_wcs = pywcs.WCS(ad["SCI"].header)
        
        # convert the reference coordinates to RA/Dec
        ref_radec = ref_wcs.wcs_pix2sky(ref_xy,1)
        
        # instantiate the alignment object used to fit input
        # WCS to reference WCS
        wcstweak = at.WCSTweak(inp_wcs, inp_xy, ref_radec, 
                               rotate=rotate, scale=scale)
        
        # find optimum WCS shift and rotation with
        # starting parameters: dRA, dDec = 0
        # (and dTheta=0 if rotate=True, dMag=1 if scale=True)
        
        update = False
        if rotate and scale:
            pars = [0,0,0,1]
        elif rotate:
            pars = [0,0,0]
        elif scale:
            pars = [0,0,1]
        else:
            pars = [0,0]
        
        try:
            # for scipy versions < 0.9
            new_pars,success = scipy.optimize.leastsq(wcstweak.calc_diff, pars,
                                                      warning=False, 
                                                      maxfev=1000)
        except:
            # for scipy versions >= 0.9
            import warnings
            warnings.simplefilter("ignore")
            new_pars,success = scipy.optimize.leastsq(wcstweak.calc_diff, pars,
                                                      maxfev=1000)
        
        if success<4:
            update = True
            if rotate and scale:
                (dRA, dDec, dTheta, dMag) = new_pars
                log.fullinfo("Best fit dRA, dDec, dTheta, dMag: " +
                             "%.5f %.5f %.5f %.5f" %
                             (dRA, dDec, dTheta, dMag))
            elif rotate:
                (dRA, dDec, dTheta) = new_pars
                log.fullinfo("Best fit dRA, dDec, dTheta: %.5f %.5f %.5f" %
                             (dRA, dDec, dTheta))
            elif scale:
                (dRA, dDec, dMag) = new_pars
                log.fullinfo("Best fit dRA, dDec, dMag: %.5f %.5f %.5f" %
                             (dRA, dDec, dMag))
            else:
                (dRA, dDec) = new_pars
                log.fullinfo("Best fit dRA, dDec: %.5f %.5f" % (dRA, dDec))
        else:
            log.warning("WCS alignment did not converge. Not updating WCS.")
        
        # update WCS in ad
        if update:
            log.fullinfo("Updating WCS in header")
            ad.phu_set_key_value("CRVAL1", wcstweak.wcs.wcs.crval[0],
                                 comment=keyword_comments["CRVAL1"])
            ad.phu_set_key_value("CRVAL2", wcstweak.wcs.wcs.crval[1],
                                 comment=keyword_comments["CRVAL2"])
            ad.phu_set_key_value("CD1_1", wcstweak.wcs.wcs.cd[0,0],
                                 comment=keyword_comments["CD1_1"])
            ad.phu_set_key_value("CD1_2", wcstweak.wcs.wcs.cd[0,1],
                                 comment=keyword_comments["CD1_2"])
            ad.phu_set_key_value("CD2_1", wcstweak.wcs.wcs.cd[1,0],
                                 comment=keyword_comments["CD2_1"])
            ad.phu_set_key_value("CD2_2", wcstweak.wcs.wcs.cd[1,1],
                                 comment=keyword_comments["CD2_2"])
            
            for ext in ad:
                if ext.extname() in ["SCI","VAR","DQ"]:
                    ext.set_key_value("CRVAL1", wcstweak.wcs.wcs.crval[0],
                                      comment=keyword_comments["CRVAL1"])
                    ext.set_key_value("CRVAL2", wcstweak.wcs.wcs.crval[1],
                                      comment=keyword_comments["CRVAL2"])
                    ext.set_key_value("CD1_1", wcstweak.wcs.wcs.cd[0,0],
                                      comment=keyword_comments["CD1_1"])
                    ext.set_key_value("CD1_2", wcstweak.wcs.wcs.cd[0,1],
                                      comment=keyword_comments["CD1_2"])
                    ext.set_key_value("CD2_1", wcstweak.wcs.wcs.cd[1,0],
                                      comment=keyword_comments["CD2_1"])
                    ext.set_key_value("CD2_2", wcstweak.wcs.wcs.cd[1,1],
                                      comment=keyword_comments["CD2_2"])
        
        adoutput_list.append(ad)
    
    return adoutput_list

def _header_align(reference, adinput):
    """
    This function uses the POFFSET, QOFFSET and PA header keywords 
    to get reference points to use in correcting an input WCS to
    a reference WCS. This function allows for relative shifts between 
    the images. Rotation and scaling will not be handled properly. 
    
    :param reference: reference image to register other images to. Must
                      have only one SCI extension.
    :type reference: AstroData object
    
    :param adinput: images to register to reference image. Must have
                  only one SCI extension.
    :type adinput: AstroData objects, either a single instance or a list
    """
    
    log = logutils.get_logger(__name__)
    
    if not isinstance(adinput,list):
        adinput = [adinput]
    
    # get starting offsets from reference image (first one given)
    ref_pixscale = float(reference.pixel_scale())
    ref_poff = reference.phu_get_key_value("POFFSET")/ref_pixscale
    ref_qoff = reference.phu_get_key_value("QOFFSET")/ref_pixscale
    ref_pa = reference.phu_get_key_value("PA")
    ref_theta = math.radians(ref_pa)
    ref_xoff = (ref_poff * math.cos(ref_theta)) - (ref_qoff * math.sin(ref_theta))
    ref_yoff =  (ref_poff * math.sin(ref_theta)) + (ref_qoff * math.cos(ref_theta))                 
    log.fullinfo("Pixel scale: %.4f" % ref_pixscale)

    # Reference position is the center of the reference frame
    data_shape = reference["SCI"].data.shape
    ref_coord = [data_shape[1]/2,data_shape[0]/2]
        
    objIns = []
    for i in range(len(adinput)):
        ad = adinput[i]
        pixscale = float(ad.pixel_scale())
        poff = ad.phu_get_key_value("POFFSET")/pixscale
        qoff = ad.phu_get_key_value("QOFFSET")/pixscale
        pa = ad.phu_get_key_value("PA")
        theta = math.radians(pa - ref_pa)
        pdiff = poff - ref_poff
        qdiff = qoff - ref_qoff
        xoff = (pdiff * math.cos(theta)) + (qdiff * math.sin(theta))
        yoff = (pdiff * math.sin(theta)) - (qdiff * math.cos(theta))
        
        img_x = ref_coord[0] - yoff
        img_y = ref_coord[1] - xoff
        
        log.fullinfo("Reference coordinates: %.1f %.1f" % 
                     (ref_coord[0], ref_coord[1]))
        log.fullinfo("For image " + ad.filename + ":")
        log.fullinfo("   Relative image offsets: %.4f %.4f" % (xoff, yoff))
        log.fullinfo("   Coordinates to transform: %.4f %.4f" % (img_x, img_y))
        objIns.append(np.array([[ref_coord],[[img_x,img_y]]]))

    adoutput_list = _align_wcs(reference, adinput, objIns, 
                               rotate=False, scale=False)
    
    return adoutput_list

