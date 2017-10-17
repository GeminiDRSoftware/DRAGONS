#
#                                                                  gemini_python
#
#                                                          primitives_register.py
# ------------------------------------------------------------------------------
import math
import numpy as np
from astropy.wcs import WCS
from astropy.modeling import models

from gempy.gemini import gemini_tools as gt
from gempy.gemini import qap_tools as qap
from gempy.utils import logutils

from gempy.library.matching import match_catalogs, align_images_from_wcs, Pix2Sky

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

        if (not hasattr(ref_image[0], 'OBJCAT') or len(ref_image[0].OBJCAT)
                                        < min_sources) and method=='sources':
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

        adoutputs = [ref_image]
        # If no OBJCAT/no sources in reference image, or user choice,
        # use indirect alignment for all images at once
        if method == "header":
            log.stdinfo("Using WCS specified in header for alignment")
            for ad in adinputs[1:]:
                adoutputs.append(_create_wcs_from_offsets(ad, ref_image))

        # otherwise try to do direct alignment for each image by correlating
        # sources in the reference and input images
        else:
            for ad in adinputs[1:]:
                try:
                    nobj = len(ad[0].OBJCAT)
                except AttributeError:
                    nobj = 0
                if nobj == 0:
                    log.warning("No objects found in {}".format(ad.filename))
                    if fallback == 'header':
                        log.warning("Only attempting indirect WCS alignment, "
                                    "via {} mapping".format(fallback))
                        ad = _create_wcs_from_offsets(ad, ref_image)
                        #adoutput = _header_align(ref_image, [ad],
                        #                         self.keyword_comments)
                    else:
                        log.warning("WCS can only be corrected indirectly "
                            "and fallback=None. Not attempting WCS correction "
                            "for ".format(ad.filename))
                else:
                    log.fullinfo("Number of objects in image {}: {}".format(
                                 ad.filename, nobj))
                    log.fullinfo("Cross-correlating sources in {}, {}".
                                 format(ref_image.filename, ad.filename))

                    # GNIRS WCS is dubious, so update WCS by using the ref
                    # image's WCS and the telescope offsets
                    if ad.instrument() == 'GNIRS' and not use_wcs:
                        ad = _create_wcs_from_offsets(ad, ref_image)

                    if not use_wcs:
                        log.warning("Parameter 'use_wcs' is False.")
                        log.warning("Using source correlation anyway.")
                    firstpasspix = first_pass / ad.pixel_scale()

                    # Calculate the offsets quickly using only a translation
                    obj_list, transform = align_images_from_wcs(ad, ref_image,
                            first_pass=firstpasspix, min_sources=min_sources,
                            cull_sources=cull_sources, full_wcs=False,
                            rotate=False, scale=False, tolerance=0.1,
                            return_matches=True)

                    n_corr = len(obj_list[0])
                    if n_corr==0:
                        log.warning("No correlated sources found.")
                        if fallback=='header':
                            log.warning("Only attempting indirect WCS "
                                "alignment, via {} mapping".format(fallback))
                            _create_wcs_from_offsets(ad, ref_image)
                            #adoutput = _header_align(ref_image, ad,
                            #                         self.keyword_comments)
                        else:
                            log.warning("WCS can only be corrected indirectly "
                                "and fallback=None. Not attempting WCS "
                                "correction for ".format(ad.filename))
                    else:
                        log.fullinfo("Number of correlated sources: {}".
                                     format(n_corr))
                        log.stdinfo("{}: Using source correlation for "
                                    "alignment".format(ad.filename))
                        x_offset = transform.x_offset.value
                        y_offset = transform.y_offset.value

                        # Check the fit geometry depending on the
                        # number of objects
                        if n_corr < 5:
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

                        # Determine a more accurate fit, and get the WCS
                        wcs = align_images_from_wcs(ad, ref_image,
                                    initial_shift=(x_offset, y_offset),
                                    first_pass=0.2*firstpasspix, refine=True,
                                    cull_sources=cull_sources, full_wcs=True,
                                    rotate=rotate, scale=scale, tolerance=0.01,
                                    return_matches=False).wcs
                        _write_wcs_keywords(ad, wcs, self.keyword_comments)
                adoutputs.append(ad)

        # Timestamp and update filenames
        for ad in adoutputs:
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.filename = gt.filename_updater(adinput=ad, suffix=params["suffix"],
                                            strip=True)
        return adoutputs
    
    def determineAstrometricSolution(self, adinputs=None, **params):
        """
        This primitive determines how to modify the WCS of each image to
        produce the best positional match between its sources (OBJCAT) and
        the REFCAT.

        Parameters
        ----------
        full_wcs: bool (or None)
            use an updated WCS for each matching iteration, rather than simply
            applying pixel-based corrections to the initial mapping?
            (None => not ('qa' in context))
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        full_wcs = params["full_wcs"]

        for ad in adinputs:
            # Check we have a REFCAT and at least one OBJCAT to match
            try:
                refcat = ad.REFCAT
            except AttributeError:
                log.warning("No REFCAT in {} - cannot calculate astrometry".
                            format(ad.filename))
                continue
            if not any(hasattr(ext, 'OBJCAT') for ext in ad):
                log.warning("No OBJCATs in {} - cannot match to REFCAT".
                            format(ad.filename))
                continue

            # List of values to report to FITSstore
            info_list = []
            all_delta_ra = []
            all_delta_dec = []

            # Try to be clever here, and work on the extension with the
            # highest number of matches first, as this will give the most
            # reliable offsets, which can then be used to constrain the other
            # extensions. The problem is we don't know how many matches we'll
            # get until we do it, and that's slow, so use len(OBJCAT) as a proxy.
            objcat_lengths = [len(ext.OBJCAT) if hasattr(ext, 'OBJCAT') else 0
                              for ext in ad]
            objcat_order = np.argsort(objcat_lengths)[::-1]

            pixscale = ad.pixel_scale()
            initial = params["initial"] / pixscale  # Search box size
            final = params["final"] / pixscale  # Matching radius
            max_ref_sources = 100 if 'qa' in self.context else None  # Don't need more than this many
            if full_wcs is None:
                full_wcs = not ('qa' in self.context)

            best_model = (0, None)

            for index in objcat_order:
                ext = ad[index]
                extver = ext.hdr['EXTVER']
                try:
                    objcat = ad[index].OBJCAT
                except AttributeError:
                    log.stdinfo('No OBJCAT in {}:{} -- cannot perform '
                                'astrometry'.format(ad.filename, extver))
                    info_list.append({})
                    continue
                objcat_len = len(objcat)

                # The reference coordinates are always (x,y) pixels in the OBJCAT
                # Set up the input coordinates
                wcs = WCS(ad.header[index + 1])
                xref, yref = refcat['RAJ2000'], refcat['DEJ2000']
                if not full_wcs:
                    xref, yref = wcs.all_world2pix(xref, yref, 1)

                # Now set up the initial model
                if full_wcs:
                    m_init = Pix2Sky(wcs, direction=-1)
                    m_init.factor.fixed = True
                    m_init.angle.fixed = True
                    if best_model[1] is None:
                        m_init.x_offset.bounds = (-initial, initial)
                        m_init.y_offset.bounds = (-initial, initial)
                    else:
                        # Copy parameters from best model to this model
                        # TODO: if rotation/scaling are used, the factor_ will need to
                        # be copied
                        for p in best_model[1].param_names:
                            setattr(m_init, p, getattr(best_model[1], p))
                else:
                    m_init = best_model[1]

                # Reduce the search space if we've previously found a match
                # TODO: This code is more generic than it needs to be now (the model
                # only has unfixed offsets) but less generic than it will need to be
                # if a rotation or magnification is added)
                if best_model[1] is not None:
                    initial = 2.5 / pixscale
                    for param in [getattr(m_init, p) for p in m_init.param_names]:
                        if 'offset' in param.name and not param.fixed:
                            param.bounds = (param.value - initial,
                                            param.value + initial)

                # First: estimate number of reference sources in field
                # Inverse map ref coords->image plane and see how many are in field
                xx, yy = m_init(xref, yref) if m_init else (xref, yref)
                x1, y1 = 0, 0
                y2, x2 = ad[index].data.shape
                # Could tweak y1, y2 here for GNIRS
                in_field = np.all((xx > x1 - initial, xx < x2 + initial,
                                   yy > y1 - initial, yy < y2 + initial), axis=0)
                num_ref_sources = np.sum(in_field)

                # We probably don't need zillions of REFCAT sources
                if max_ref_sources and num_ref_sources > max_ref_sources:
                    ref_mags = None
                    try:
                        ref_mags = refcat['filtermag']
                        if np.all(np.where(np.isnan(ref_mags), -999,
                                           ref_mags) < -99):
                            log.stdinfo('The REFCAT magnitude column has no '
                                        'valid values')
                            ref_mags = None
                    except KeyError:
                        log.stdinfo('Cannot find a magnitude column to cull REFCAT')
                    if ref_mags is None:
                        for filt in 'rhikgjzu':
                            try:
                                ref_mags = refcat[filt+'mag']
                            except KeyError:
                                pass
                            else:
                                if not np.all(np.where(np.isnan(ref_mags), -999,
                                                       ref_mags) < -99):
                                    log.stdinfo('Using {} magnitude instead'.
                                                format(filt))
                                    break

                    if ref_mags is not None:
                        in_field &= (ref_mags > -99)
                        num_ref_sources = np.sum(in_field)
                        if num_ref_sources > max_ref_sources:
                            sorted_args = np.argsort(ref_mags)
                            in_field = sorted_args[in_field[sorted_args]][:max_ref_sources]
                            log.stdinfo('Using only {} brightest REFCAT sources '
                                        'for speed'.format(max_ref_sources))
                            # in_field is now a list of indices, not a boolean array
                            num_ref_sources = len(in_field)

                # How many objects do we want to try to match? Keep brightest ones only
                if objcat_len > 2 * num_ref_sources:
                    keep_num = max(2 * num_ref_sources, min(10, objcat_len))
                else:
                    keep_num = objcat_len
                sorted_idx = np.argsort(objcat['MAG_AUTO'])[:keep_num]

                # Send all sources to the alignment/matching engine, indicating the ones to
                # use for the alignment
                if num_ref_sources > 0:
                    log.stdinfo('Aligning extver {} with {} REFCAT and {} OBJCAT sources'.
                                format(extver, num_ref_sources, keep_num))
                    matched, m_final = match_catalogs(xref, yref, objcat['X_IMAGE'], objcat['Y_IMAGE'],
                                                      use_in=in_field, use_ref=sorted_idx,
                                                      model_guess=m_init, translation_range=initial,
                                                      tolerance=0.05, match_radius=final)
                else:
                    log.stdinfo('No REFCAT sources in field of extver {}'.format(extver))
                    continue

                num_matched = np.sum(matched >= 0)
                log.stdinfo("Matched {} objects in OBJCAT:{} against REFCAT".
                            format(num_matched, extver))
                # If this is a "better" match, save it
                # TODO? Some sort of averaging of models?
                if num_matched > max(best_model[0], 2):
                    best_model = (num_matched, m_final)

                if num_matched > 0:
                    # Update WCS in the header and OBJCAT (X_WORLD, Y_WORLD)
                    if full_wcs:
                        new_wcs = m_final.wcs
                    else:
                        kwargs = dict(zip(m_final.param_names, m_final.parameters))
                        new_wcs = Pix2Sky(wcs, **kwargs).wcs
                    _write_wcs_keywords(ext, new_wcs, self.keyword_comments)
                    objcat['X_WORLD'], objcat['Y_WORLD'] = new_wcs.all_pix2world(
                        objcat['X_IMAGE'], objcat['Y_IMAGE'], 1)

                    # Sky coordinates of original CRPIX location with old
                    # and new WCS (easier than using the transform)
                    ra0, dec0 = wcs.all_pix2world([wcs.wcs.crpix], 1)[0]
                    ra1, dec1 = new_wcs.all_pix2world([wcs.wcs.crpix], 1)[0]
                    cosdec = math.cos(math.radians(dec0))
                    delta_ra = 3600 * (ra1-ra0) * cosdec
                    delta_dec = 3600 * (dec1-dec0)
                    all_delta_ra.append(delta_ra)
                    all_delta_dec.append(delta_dec)

                    # Associate REFCAT properties with their OBJCAT
                    # counterparts. Remember! matched is the reference
                    # (OBJCAT) source for the input (REFCAT) source
                    dra = []
                    ddec = []
                    for i, m in enumerate(matched):
                        if m >= 0:
                            objcat['REF_NUMBER'][m] = refcat['Id'][i]
                            try:
                                objcat['REF_MAG'][m] = refcat['filtermag'][i]
                                objcat['REF_MAG_ERR'][m] = refcat['filtermag_err'][i]
                            except KeyError:  # no such columns in REFCAT
                                pass
                            dra.append(3600*(objcat['X_WORLD'][m] -
                                             refcat['RAJ2000'][i]) * cosdec)
                            ddec.append(2600*(objcat['Y_WORLD'][m] -
                                              refcat['DEJ2000'][i]))
                    dra_std = np.std(dra)
                    ddec_std = np.std(ddec)
                    log.fullinfo("WCS Updated for extver {}. Astrometric "
                                 "offset is:".format(extver))
                    log.fullinfo("RA:  {:.2f} +/- {:.2f} arcsec".
                                 format(delta_ra, dra_std))
                    log.fullinfo("Dec: {:.2f} +/- {:.2f} arcsec".
                                 format(delta_dec, ddec_std))
                    info_list.append({"dra": delta_ra, "dra_std": dra_std,
                                      "ddec": delta_dec, "ddec_std": ddec_std,
                                      "nsamples": num_matched})
                else:
                    log.stdinfo("Could not determine astrometric offset for "
                                "{}:{}".format(ad.filename, extver))
                    info_list.append({})

            # Report the measurement to the fitsstore
            fitsdict = qap.fitsstore_report(ad, "pe", info_list,
                        self.calurl_dict, self.context, self.upload_metrics)

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.filename = gt.filename_updater(adinput=ad, suffix=params["suffix"],
                                            strip=True)

        return adinputs

##############################################################################
# Below are the helper functions for the user level functions in this module #
##############################################################################

def _create_wcs_from_offsets(adinput, adref, center_of_rotation=None):
    """
    This function uses the POFFSET, QOFFSET, and PA header keywords to create
    a new WCS for an image. Its primary role is for GNIRS. For ease, it works
    out the (RA,DEC) of the centre of rotation in the reference image and
    determines where in the input image this is.

    Parameters
    ----------
    adinput: AstroData
        The input image whose WCS needs to be rewritten
    adreference: AstroData
        The reference image with a trustworthy WCS
    center_of_rotation: 2-tuple
        Location of rotation center (x, y)
    """
    log = logutils.get_logger(__name__)
    if len(adinput) != len(adref):
        log.warning("Number of extensions in input files are different. "
                    "Cannot correct WCS.")
        return adinput

    log.stdinfo("Updating WCS of {} based on {}".format(adinput.filename,
                                                        adref.filename))
    try:
        xdiff = adinput.detector_x_offset() - adref.detector_x_offset()
        ydiff = adinput.detector_y_offset() - adref.detector_y_offset()
        pa1 = adref.phu['PA']
        pa2 = adinput.phu['PA']
    except (KeyError, TypeError):  # TypeError if offset is None
        log.warning("Cannot obtain necessary offsets from headers "
                    "so no change will be made")
        return adinput

    # We expect mosaicked inputs but there's no reason why this couldn't
    # work for all extensions in an image
    for extin, extref in zip(adinput, adref):
        # Will need to have some sort of LUT here eventually. But for now...
        if center_of_rotation is None:
            center_of_rotation = (630.0, 520.0) if 'GNIRS' in adref.tags \
                else tuple(0.5*x for x in extref.data.shape[::-1])

        wcsref = WCS(extref.header[1])
        ra0, dec0 = wcsref.all_pix2world(center_of_rotation[0],
                                         center_of_rotation[1], 1)
        extin.hdr['CRVAL1'] = float(ra0)
        extin.hdr['CRVAL2'] = float(dec0)
        extin.hdr['CRPIX1'] = center_of_rotation[0] + xdiff
        extin.hdr['CRPIX2'] = center_of_rotation[1] + ydiff
        cd = models.Rotation2D(angle=pa1-pa2)(*wcsref.wcs.cd)
        extin.hdr['CD1_1'] = cd[0][0]
        extin.hdr['CD1_2'] = cd[0][1]
        extin.hdr['CD2_1'] = cd[1][0]
        extin.hdr['CD2_2'] = cd[1][1]
    return adinput

def _apply_model_to_wcs(adinput, transform=None, keyword_comments=None):
    """
    This function modifies the WCS of input images according to astropy
    Models that describe how to map the input image pixels to their
    correct location, i.e., an input pixel (x,y) should have the world
    coordinates WCS(m(x,y)), where m is the transformation model.

    Parameters
    ----------
    adinput: list of AD objects
        input images for WCS to be modified
    transform: list of Models
        transformations (in pixel space)
    keyword_comments: dict
        the comment for each FITS keyword

    Returns
    -------
    list of ADs: modified AD instances
    """
    if len(transform) != len(adinput):
        raise IOError("List of models have the same number of "
                      "elements as adinput")

    for ad, model in zip(adinput, transform):
        crpix1 = ad.hdr['CRPIX1'][0]
        crpix2 = ad.hdr['CRPIX2'][0]
        newcrpix = model.inverse(crpix1, crpix2)

        # Determine total magnification and rotation from all relevant
        # model components
        magnification = np.multiply.reduce([getattr(model, p).value
                            for p in model.param_names if 'factor' in p])
        rotation = np.add.reduce([getattr(model, p).value
                            for p in model.param_names if 'angle' in p])

        cdmatrix = [[ad.hdr['CD{}_{}'.format(i,j)][0] / magnification
                     for j in 1, 2] for i in 1, 2]
        m = models.Rotation2D(-rotation)
        newcdmatrix = m(*cdmatrix)
        for ax in 1, 2:
            ad.hdr.set('CRPIX{}'.format(ax), newcrpix[ax-1],
                       comment=keyword_comments["CRPIX{}".format(ax)])
            for ax2 in 1, 2:
                ad.hdr.set('CD{}_{}'.format(ax, ax2), newcdmatrix[ax-1][ax2-1],
                           comment=keyword_comments['CD{}_{}'.format(ax, ax2)])
    return adinput

def _write_wcs_keywords(ad, wcs, keyword_comments):
    """
    Updates the FITS header WCS keywords, with comments. Will at some point
    probably be superseded by the WCS.to_header() method

    Parameters
    ----------
    ad: AstroData
        the AD object (or slice) to have its WCS keywords modified in place
    wcs: WCS
        WCS object with the information
    keyword_comments:
        list of comments for the keywords
    """
    for ax in 1, 2:
        ad.hdr.set('CRPIX{}'.format(ax), wcs.wcs.crpix[ax-1],
                   comment=keyword_comments["CRPIX{}".format(ax)])
        ad.hdr.set('CRVAL{}'.format(ax), wcs.wcs.crval[ax-1],
                   comment=keyword_comments["CRVAL{}".format(ax)])
        for ax2 in 1, 2:
            ad.hdr.set('CD{}_{}'.format(ax, ax2), wcs.wcs.cd[ax-1, ax2-1],
                       comment=keyword_comments["CD{}_{}".format(ax, ax2)])
    return