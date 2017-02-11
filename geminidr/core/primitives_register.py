#
#                                                                  gemini_python
#
#                                                          primitives_register.py
# ------------------------------------------------------------------------------
import math
import numpy as np
import scipy.optimize
from astropy.wcs import WCS
from astropy import table

from gempy.gemini import gemini_tools as gt
from gempy.gemini import qap_tools as qap
from gempy.library import astrotools as at
from gempy.utils import logutils

from geminidr import PrimitivesBASE
from .parameters_register import ParametersRegister

from recipe_system.utils.decorators import parameter_override
# ------------------------------------------------------------------------------
@parameter_override
class Register(PrimitivesBASE):
    """
    This is the class containing all of the primitives for registration.
    """
    tagset = None

    def __init__(self, adinputs, **kwargs):
        super(Register, self).__init__(adinputs, **kwargs)
        self.parameters = ParametersRegister

    def correctWCSToReferenceFrame(self, adinputs=None, **params):
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
        
        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        method: str ['sources' | 'header']
            method to use to generate reference points. 'sources' uses sources
            to align images, 'header' uses POFFSET and QOFFSET keywords
        fallback: 'header' or None
            backup method, if the primary one fails
        use_wcs: bool
            use the header's WCS for initial alignment guess, rather than
            shifts and rotation information in the header
        first_pass: float
            search radius (UNITS???????) for the initial alignment matching
        min_sources: int
            minimum number of matched sources required to apply a WCS shift
        cull_sources: bool
            remove sub-optimal (saturated and/or non-stellar) sources before
            alignment?
        rotate: bool
            allow image rotation to align to reference image?
        scale: bool
            allow image scaling to align to reference image?
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        if len(adinputs) <= 1:
            log.warning("No correction will be performed, since at least "
                        "two input AstroData objects are required for "
                        "correctWCSToReferenceFrame")
            return adinputs

        if not all(len(ad)==1 for ad in adinputs):
            raise IOError("All input images must have only one extension.")

        method = params["method"]
        fallback = params["fallback"]
        use_wcs = params["use_wcs"]
        first_pass = params["first_pass"]
        min_sources = params["min_sources"]
        cull_sources = params["cull_sources"]
        rotate = params["rotate"]
        scale = params["scale"]

        assert method=='header' or method=='sources', \
            "Invalid method specified: {}".format(method)

        assert fallback=='header' or fallback is None, \
            "Invalid fallback method specified: {}".format(fallback)

        # Use first image in list as reference
        ref_image = adinputs[0]
        log.stdinfo("Reference image: "+ref_image.filename)
        adoutputs = [ref_image]

        if len(ref_image[0].OBJCAT) < min_sources and method=='sources':
            log.warning("Insufficient objects found in reference image.")
            if fallback is None:
                log.warning("WCS can only be corrected indirectly and "
                            "fallback method is set to None. Not "
                            "attempting WCS correction.")
                return adinputs
            else:
                log.warning("Only attempting indirect WCS alignment, "
                            "via {} mapping".format(fallback))
                method = fallback

        # If no OBJCAT/no sources in reference image, or user choice,
        # use indirect alignment for all images at once
        if method == "header":
            log.stdinfo("Using WCS specified in header for alignment")
            reg_ad = _header_align(ref_image, adinputs[1:],
                                   self.keyword_comments)

            # Not sure this is needed as everything has been done in-place
            adoutputs.extend(reg_ad)

        # otherwise try to do direct alignment for each image by correlating
        # sources in the reference and input images
        else:
            for ad in adinputs[1:]:
                nobj = len(ad[0].OBJCAT)
                if nobj == 0:
                    log.warning("No objects found in {}".format(ad.filename))
                    if fallback == 'header':
                        log.warning("Only attempting indirect WCS alignment, "
                                    "via {} mapping".format(fallback))
                        adoutput = _header_align(ref_image, [ad],
                                                 self.keyword_comments)
                    else:
                        log.warning("WCS can only be corrected indirectly "
                            "and fallback=None. Not attempting WCS correction "
                            "for ".format(ad.filename))
                        adoutputs.append(ad)
                        continue
                else:
                    log.fullinfo("Number of objects in image {}: {}".format(
                                 ad.filename, nobj))
                    log.fullinfo("Cross-correlating sources in {}, {}".
                                 format(ref_image.filename, ad.filename))

                    firstpasspix = first_pass / ad.pixel_scale()
                    is_gnirs = 'GNIRS' in ad.tags
                    if is_gnirs and not use_wcs:
                        try:
                            obj_list = _correlate_sources_offsets(ref_image, ad,
                                                         firstPass=firstpasspix,
                                                         min_sources=min_sources,
                                                         cull_sources=cull_sources)
                        except ImportError as msg:
                            log.warning("{}: No object list.".format(msg))
                            obj_list = [[],[]]
                    else:
                        if not use_wcs:
                            log.warning("Parameter 'use_wcs' is False.")
                            log.warning("Using source correlation anyway.")
                        obj_list = _correlate_sources(ref_image, ad,
                                                      firstPass=firstpasspix,
                                                      min_sources=min_sources,
                                                      cull_sources=cull_sources)

                    n_corr = len(obj_list[0])
                    if n_corr==0:
                        log.warning("No correlated sources found.")
                        if fallback=='header':
                            log.warning("Only attempting indirect WCS "
                                "alignment, via {} mapping".format(fallback))
                            adoutput = _header_align(ref_image, ad,
                                                     self.keyword_comments)
                        else:
                            log.warning("WCS can only be corrected indirectly "
                                "and fallback=None. Not attempting WCS "
                                "correction for ".format(ad.filename))
                            adoutputs.append(ad)
                            continue
                    else:
                        log.fullinfo("Number of correlated sources: {}".
                                     format(n_corr))
                        log.stdinfo("{}: Using source correlation for "
                                    "alignment".format(ad.filename))

                        # Check the fit geometry depending on the
                        # number of objects
                        if n_corr == 1:
                            log.warning("Too few objects. Setting "
                                        "rotate=False, scale=False")
                            rotate=False
                            scale=False

                        log.fullinfo("\nSources used to align frames:")
                        log.fullinfo("   Ref. x Ref. y  Img. x  Img. y\n  {}".
                                     format("-"*31))
                        for ref, img in zip(*obj_list):
                            log.fullinfo("  {:7.2f} {:7.2f} {:7.2f} {:7.2f}".
                                          format(ref[0], ref[1], *img))
                        log.fullinfo("")

                        adoutput = _align_wcs(ref_image, [ad], [obj_list],
                                        rotate=rotate, scale=scale,
                                        keyword_comments=self.keyword_comments)
                adoutputs.extend(adoutput)

        # Timestamp and update filenames
        for ad in adoutputs:
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.filename = gt.filename_updater(adinput=ad, suffix=params["suffix"],
                                            strip=True)
        return adoutputs
    
    def determineAstrometricSolution(self, adinputs=None, **params):
        """
        This primitive calculates the average astrometric offset between
        the positions of sources in the reference catalog, and their
        corresponding object in the object catalog.
        It then reports the astrometric correction vector.
        For now, this is limited to a translational offset only.
        
        The solution is stored in the wcs attribute of the primitivesClass. It
        can be applied to the image headers by calling the updateWCS primitive.

        Parameters
        ----------
        None
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        wcs_dict = {}
        for ad in adinputs:
            wcs_dict[ad] = {}

            # Can't do this if there's no REFCAT
            try:
                refcat = ad.REFCAT
            except AttributeError:
                log.warning("No REFCAT in {} - cannot calculate astrometry".
                            format(ad.filename))
                continue

            # Initialise lists to keep the total offsets in
            all_delta_ra = []
            all_delta_dec = []

            # List of values to report to FITSstore
            info_list = []

            # Have to rename column to enable Table match
            refcat.rename_column('Id', 'REF_NUMBER')

            for ext in ad:
                extver = ext.hdr['EXTVER']
                try:
                    objcat = ext.OBJCAT
                except AttributeError:
                    log.warning("No OBJCAT in {}:{} - cannot calculate "
                    "astrometry".format(ad.filename, extver))
                    info_list.append({})
                    continue

                merged = table.join(objcat, refcat, keys='REF_NUMBER',
                                    metadata_conflicts='silent')[
                    'X_WORLD', 'Y_WORLD', 'RAJ2000', 'DEJ2000']
                if len(merged) > 0:
                    delta_ra = 3600 * (merged['RAJ2000'] -
                                       merged['X_WORLD']).data
                    delta_dec = 3600 * (merged['DEJ2000'] -
                                        merged['Y_WORLD']).data
                else:
                    log.fullinfo("No reference sources in {}:{}".format(
                        ad.filename, extver))
                    info_list.append({})
                    continue

                # Report the mean and standard deviation of the offsets:
                ra_mean = np.mean(delta_ra)
                ra_sigma = np.std(delta_ra)
                dec_mean = np.mean(delta_dec)
                dec_sigma = np.std(delta_dec)
                ra_median = np.median(delta_ra)
                dec_median = np.median(delta_dec)
                all_delta_ra.extend(delta_ra)
                all_delta_dec.extend(delta_dec)

                # These aren't *real* arcseconds in RA of course!
                log.fullinfo("Astrometric Offset between for EXTVER {} is:".
                             format(extver))
                log.fullinfo("RA:  {:.2f} +/- {:.2f} arcsec".format(ra_mean,
                                                                    ra_sigma))
                log.fullinfo("Dec: {:.2f} +/- {:.2f} arcsec".format(dec_mean,
                                                                    dec_sigma))
                log.fullinfo("Median Offset is: {:.2f}, {:.2f} arcsec".
                             format(ra_median, dec_median))

                # Store it in the fitsstore info_dict
                info_list.append({"dra":ra_mean, "dra_std":ra_sigma,
                                  "ddec":dec_mean, "ddec_std":dec_sigma,
                                  "nsamples":len(delta_ra)})

                # Store the changes in a WCS object so they
                # can be applied by the updateWCS primitive if desired
                wcs = WCS(ext.header[1])
                wcs.wcs.crval = np.array([wcs.wcs.crval[0]+ra_median/3600.,
                                          wcs.wcs.crval[1]+dec_median/3600.])
                wcs_dict[ad][extver] = wcs

            if all_delta_ra:
                # Report the mean and standard deviation of all the offsets over all the sci extensions:
                ra_mean = np.mean(all_delta_ra)
                ra_sigma = np.std(all_delta_ra)
                dec_mean = np.mean(all_delta_dec)
                dec_sigma = np.std(all_delta_dec)

                log.stdinfo("Mean Astrometric Offset for {}:".
                            format(ad.filename))
                log.stdinfo("     RA: {:.2f} +/- {:.2f}    Dec: {:.2f} +/- "
                    "{:.2f}   arcsec".format(ra_mean, ra_sigma,
                                             dec_mean, dec_sigma))
            else:
                log.stdinfo("Could not determine astrometric offset for {}".
                            format(ad.filename))
                
            # Report the measurement to the fitsstore
            fitsdict = qap.fitsstore_report(ad, "pe", info_list,
                        self.calurl_dict, self.context, self.upload_metrics)

            # Re-rename the column back to its original name
            refcat.rename_column('REF_NUMBER', 'Id')

        # Store the WCS solution and return the (untouched) AD list
        self.wcs = wcs_dict
        return adinputs

    def updateWCS(self, adinputs=None, **params):
        """
        This primitive applies a previously calculated WCS correction.
        The solution should be stored as an attribute of the primitives
        class, with astrodata instances as the keys and WCS objects as the
        values.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        # Get the necessary parameters from the RC
        try:
            wcs = self.wcs
        except AttributeError:
            log.warning("No new WCS supplied; no correction will be "
                        "performed.")
            return adinputs

        for ad in adinputs:
            failed_extver = []
            ad_wcs = wcs if isinstance(wcs, WCS) else wcs.get(ad)

            if ad_wcs is None:
                log.warning("No new WCS supplied for {}; no correction will "
                            "be performed".format(ad.filename))
                continue

            for ext in ad:
                extver = ext.hdr['EXTVER']
                ext_wcs = ad_wcs if isinstance(wcs, WCS) else ad_wcs.get(extver)

                if ext_wcs is None:
                    log.fullinfo("No new WCS supplied for {}:{}; no correction "
                                 "will be performed".format(ad.filename,extver))
                    failed_extver.append(extver)
                    continue

                # If image extension, correct the header values
                log.fullinfo("Correcting CRVAL, CRPIX, and CD in image "
                             "extension headers {}:{}".format(ad.filename,
                                                              extver))
                log.fullinfo("CRVAL: "+repr(ext_wcs.wcs.crval))
                log.fullinfo("CRPIX: "+repr(ext_wcs.wcs.crpix))
                log.fullinfo("CD: "+repr(ext_wcs.wcs.cd))

                for ax in range(1,3):
                    ad.hdr.set('CRPIX{}'.format(ax), ext_wcs.wcs.crpix[ax-1],
                        comment=self.keyword_comments["CRPIX{}".format(ax)])
                    ad.hdr.set('CRVAL{}'.format(ax), ext_wcs.wcs.crval[ax-1],
                        comment=self.keyword_comments["CRVAL{}".format(ax)])
                    for ax2 in range(1,3):
                        ad.hdr.set('CD{}_{}'.format(ax,ax2),
                                   ext_wcs.wcs.cd[ax-1,ax2-1],
                                    comment=self.keyword_comments["CD{}_{}".
                                   format(ax,ax2)])

                # If objcat, fix the RA/Dec columns
                try:
                    objcat = ext.OBJCAT
                except:
                    pass
                else:
                    log.fullinfo("Correcting RA, Dec columns in OBJCAT "
                                 "extension for {}:{}".format(ad.filename,
                                                              extver))
                    objcat['X_WORLD'], objcat['Y_WORLD'] = ext_wcs.all_pix2world(
                        objcat['X_IMAGE'], objcat['Y_WORLD'], 1)

            if failed_extver:
                ok_extver = [extver for extver in ad.hdr['EXTVER']
                             if extver not in failed_extver]
                log.stdinfo("Updated WCS for {} extver {}. Extver {} failed."
                            .format(ad.filename, ok_extver, failed_extver))
            else:
                log.stdinfo("Updated WCS for all extvers in {}.".
                            format(ad.filename))

            # Timestamp and update the filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.filename = gt.filename_updater(adinput=ad, suffix=params["suffix"],
                                              strip=True)
        return adinputs

##############################################################################
# Below are the helper functions for the user level functions in this module #
##############################################################################

def _correlate_sources(ad1, ad2, delta=None, firstPass=10, min_sources=1,
                       cull_sources=False):
    """
    This function takes sources from the OBJCAT extensions in two
    images and attempts to correlate them, using the WCS encoded in the
    headers as a first approximation. It returns a list of reference source 
    positions and their correlated image source positions.
    
    :param ad1: reference image (one extension only)
    :type ad1: AstroData instance
    
    :param ad2: input image (one extension only)
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
    # If desired, select only the most star-like sources in the OBJCAT
    log = logutils.get_logger(__name__)
    if cull_sources:
        good_src1 = gt.clip_sources(ad1)[0]
        good_src2 = gt.clip_sources(ad2)[0]
        if len(good_src1) < min_sources or len(good_src2) < min_sources:
            log.warning("Too few sources in culled list, using full set "
                        "of sources")
            x1, y1 = ad1[0].OBJCAT['X_IMAGE'], ad1[0].OBJCAT['Y_IMAGE']
            x2, y2 = ad2[0].OBJCAT['X_IMAGE'], ad2[0].OBJCAT['Y_IMAGE']
        else:
            x1, y1 = good_src1["x"], good_src1["y"]
            x2, y2 = good_src2["x"], good_src2["y"]
    else:
        x1, y1 = ad1[0].OBJCAT['X_IMAGE'], ad1[0].OBJCAT['Y_IMAGE']
        x2, y2 = ad2[0].OBJCAT['X_IMAGE'], ad2[0].OBJCAT['Y_IMAGE']

    # convert image 2 data to sky coordinates and then to image 1 pixels
    ra2, dec2 = WCS(ad2.header[1]).all_pix2world(x2,y2, 1)
    conv_x2, conv_y2 = WCS(ad1.header[1]).all_world2pix(ra2, dec2,1)
    
    # find matches
    ind1,ind2 = at.match_cxy(x1,conv_x2, y1,conv_y2,
                             first_pass=firstPass, delta=delta, log=log)

    obj_list = [[],[]] if len(ind1)<1 else [list(zip(x1[ind1], y1[ind1])),
                                            list(zip(x2[ind2], y2[ind2]))]
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
    # If desired, select only the most star-like sources in the OBJCAT
    log = logutils.get_logger(__name__)
    if cull_sources:
        good_src1 = gt.clip_sources(ad1)[0]
        good_src2 = gt.clip_sources(ad2)[0]
        if len(good_src1) < min_sources or len(good_src2) < min_sources:
            log.warning("Too few sources in culled list, using full set "
                        "of sources")
            x1, y1 = ad1[0].OBJCAT['X_IMAGE'], ad1[0].OBJCAT['Y_IMAGE']
            x2, y2 = ad2[0].OBJCAT['X_IMAGE'], ad2[0].OBJCAT['Y_IMAGE']
        else:
            x1, y1 = good_src1["x"], good_src1["y"]
            x2, y2 = good_src2["x"], good_src2["y"]
    else:
        x1, y1 = ad1[0].OBJCAT['X_IMAGE'], ad1[0].OBJCAT['Y_IMAGE']
        x2, y2 = ad2[0].OBJCAT['X_IMAGE'], ad2[0].OBJCAT['Y_IMAGE']

    # Shift the catalog of image to be aligned using the header offsets.
    # The instrument alignment angle should be used to check how to 
    # use the offsets.
    poffset1 = ad1.phu['POFFSET']
    qoffset1 = ad1.phu['QOFFSET']
    poffset2 = ad2.phu['POFFSET']
    qoffset2 = ad2.phu['QOFFSET']
    pixscale = ad1.pixel_scale()
    xdiff = -1.0 * (qoffset1 - qoffset2) / pixscale
    ydiff = (poffset1 - poffset2) / pixscale
    
    pa1 = ad1.phu['PA']
    pa2 = ad2.phu['PA']
    # Can only deal with rotations for GNIRS at the moment. This code
    # will need to be totally rewritten to generalize it, so I'm going
    # to do a quick and ugly refactor.
    if abs(pa1 - pa2)<1.0 or ad1.instrument()!='GNIRS':
        conv_x2 = [(item - xdiff) for item in x2]
        conv_y2 = [(item - ydiff) for item in y2]
        if abs(pa1 - pa2)<1.0:
            log.fullinfo("Less than 1 degree of rotation between the frames, "
                         "no rotation applied.")
        else:
            log.fullinfo("No frame center found for {}, no rotation can "
                        "be applied.".format(ad2.filename))
        log.fullinfo("dx = {} px, dy = {} px applied from the headers".
                    format(xdiff, ydiff))
    else:
        theta = math.radians(pa1 - pa2)
        centerx, centery = 630.0, 520.0 # grabbed from gnirsCenterDict
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

    # find matches
    ind1,ind2 = at.match_cxy(x1,conv_x2,y1,conv_y2,
                             delta=delta, first_pass=firstPass, log=log)

    obj_list = [[],[]] if len(ind1)<1 else [list(zip(x1[ind1], y1[ind1])),
                                            list(zip(x2[ind2], y2[ind2]))]
    return obj_list

def _align_wcs(ref_ad, adinput, objIns, rotate=False, scale=False,
               keyword_comments=None):
    """
    This function fits an input image's WCS to a reference image's WCS
    by minimizing the difference in the input image frame between
    reference points present in both images.
    
    :param ref_ad: reference image to register other images to. Must
                      have only one SCI extension.
    :type ref_ad: AstroData object
    
    :param adinput: images to register to reference image. Must have
                  only one SCI extension.
    :type adinput: list of AstroData objects
    
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
    if len(objIns) != len(adinput):
        raise IOError("Argument objIns should have the same number of "
                      "elements as adinput")

    ref_wcs = WCS(ref_ad.header[1])
    for ad, objIn in zip(adinput, objIns):
        log.fullinfo("Adjusting WCS for {}".format(ad.filename))
        ref_xy = np.array(objIn[0])
        inp_xy = np.array(objIn[1])
        inp_wcs = WCS(ad.header[1])
        
        # convert the reference coordinates to RA/Dec
        ref_radec = ref_wcs.all_pix2world(ref_xy,1)

        #TODO: redo this with astropy.modeling
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
        
        import warnings
        warnings.simplefilter("ignore")
        new_pars, success = scipy.optimize.leastsq(wcstweak.calc_diff, pars,
                                                   maxfev=1000)

        if success <= 4:
            update = True
            if rotate and scale:
                log.fullinfo("Best fit dRA, dDec, dTheta, dMag: {:.5f} {:5.f}"
                             " {:.5f} {:.5f}".format(*new_pars))
            elif rotate:
                log.fullinfo("Best fit dRA, dDec, dTheta: {:.5f} {:.5f} "
                             "{:.5f}".format(*new_pars))
            elif scale:
                log.fullinfo("Best fit dRA, dDec, dMag: {:.5f} {:.5f} "
                             "{:.5f}".format(*new_pars))
            else:
                log.fullinfo("Best fit dRA, dDec: {:.5f} {:.5f}".format(
                    *new_pars))
        else:
            log.warning("WCS alignment did not converge. Not updating WCS.")
        
        # update WCS in ad
        if update:
            log.fullinfo("Updating WCS in header")
            for ax in range(1, 3):
                ad.hdr.set('CRVAL{}'.format(ax), wcstweak.wcs.wcs.crval[ax-1],
                           comment=keyword_comments["CRVAL{}".format(ax)])
                for ax2 in range(1, 3):
                    ad.hdr.set('CD{}_{}'.format(ax, ax2), wcstweak.wcs.wcs.cd[ax-1, ax2-1],
                               comment=keyword_comments["CD{}_{}".format(ax, ax2)])
    return adinput

def _header_align(ref_ad, adinput, keyword_comments):
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
    # get starting offsets from reference image (first one given)
    log = logutils.get_logger(__name__)
    ref_pixscale = ref_ad.pixel_scale()
    ref_poff = ref_ad.phu['POFFSET'] / ref_pixscale
    ref_qoff = ref_ad.phu['QOFFSET'] / ref_pixscale
    ref_pa = ref_ad.phu['PA']
    ref_theta = math.radians(ref_pa)
    log.fullinfo("Pixel scale: {:.4}".format(ref_pixscale))

    # Reference position is the center of the reference frame
    data_shape = ref_ad[0].data.shape
    ref_coord = [data_shape[1]/2,data_shape[0]/2]
        
    objIns = []
    for ad in adinput:
        pixscale = ad.pixel_scale()
        poff = ad.phu['POFFSET'] / pixscale
        qoff = ad.phu['QOFFSET'] / pixscale
        pa = ad.phu['PA']
        theta = math.radians(pa - ref_pa)
        pdiff = poff - ref_poff
        qdiff = qoff - ref_qoff
        xoff = (pdiff * math.cos(theta)) + (qdiff * math.sin(theta))
        yoff = (pdiff * math.sin(theta)) - (qdiff * math.cos(theta))
        
        img_x = ref_coord[0] - yoff
        img_y = ref_coord[1] - xoff
        
        log.fullinfo("Reference coordinates: {:.1f} {:.1f}".format(*ref_coord))
        log.fullinfo("For image {}:".format(ad.filename))
        log.fullinfo("   Relative image offsets: {:.4f} {:.4f}".
                     format(xoff, yoff))
        log.fullinfo("   Coordinates to transform: {:.4f} {:.4f}".
                     format(img_x, img_y))
        objIns.append(np.array([[ref_coord],[[img_x,img_y]]]))

    adoutput_list = _align_wcs(ref_ad, adinput, objIns,
                rotate=False, scale=False, keyword_comments=keyword_comments)
    return adoutput_list