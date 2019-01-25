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

from gempy.library.matching import align_images_from_wcs, align_catalogs, match_sources
from gempy.library.transform import Transform
from gempy.library.astromodels import Pix2Sky

from geminidr import PrimitivesBASE
from . import parameters_register

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
        self._param_update(parameters_register)

    def matchWCSToReference(self, adinputs=None, **params):
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
        method: str ['sources' | 'offsets']
            method to use to generate reference points. 'sources' uses sources
            to align images, 'offsets' uses POFFSET and QOFFSET keywords
        fallback: 'header' or None
            backup method, if the primary one fails
        first_pass: float
            search radius (arcsec) for the initial alignment matching
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
        warnings = {"offsets": "Using header offsets for alignment",
                    None: "No WCS correction being performed"}

        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        if len(adinputs) <= 1:
            log.warning("No correction will be performed, since at least two "
                        "input images are required for matchWCSToReference")
            return adinputs

        if not all(len(ad)==1 for ad in adinputs):
            raise IOError("All input images must have only one extension.")

        method = params["method"]
        fallback = params["fallback"]
        first_pass = params["first_pass"]
        min_sources = params["min_sources"]
        cull_sources = params["cull_sources"]
        rotate = params["rotate"]
        scale = params["scale"]

        # Use first image in list as reference
        ref_ad = adinputs[0]
        log.stdinfo("Reference image: {}".format(ref_ad.filename))

        if (not hasattr(ref_ad[0], 'OBJCAT') or len(ref_ad[0].OBJCAT)
                                        < min_sources) and method == 'sources':
            log.warning("Too few objects found in reference image. "
                        "{}.".format(warnings[fallback]))
            if fallback is None:
                return adinputs
            else:
                method = fallback

        adoutputs = [ref_ad]
        if method == "offsets":
            log.stdinfo("Using offsets specified in header for alignment.")

        for ad in adinputs[1:]:
            # If we're doing source alignment but fallback is "offsets",
            # update the WCS to give us a better starting point
            # TODO: Think about what we really really really want
            if method == "offsets" or fallback == "offsets":
                #_create_wcs_from_offsets(ad, ref_ad)
                if method == "offsets":
                    _create_wcs_from_offsets(ad, ref_ad)
                    adoutputs.append(ad)
                    continue

            try:
                nobj = len(ad[0].OBJCAT)
            except AttributeError:
                nobj = 0
            if nobj < min_sources:
                log.warning("Too few objects found in {}. "
                            "{}.".format(ad.filename, warnings[fallback]))
                adoutputs.append(ad)
                continue

            log.fullinfo("Number of objects in {}: {}".format(ad.filename, nobj))
            log.stdinfo("Cross-correlating sources in {}, {}".
                         format(ref_ad.filename, ad.filename))

            # Calculate the offsets quickly using only a translation
            firstpasspix = first_pass / ad.pixel_scale()
            obj_list, transform = align_images_from_wcs(ad, ref_ad,
                    search_radius=firstpasspix, min_sources=min_sources,
                    cull_sources=cull_sources, full_wcs=False,
                    rotate=False, scale=False, return_matches=True)

            n_corr = len(obj_list[0])
            log.fullinfo("Number of correlated sources: {}".format(n_corr))
            if n_corr < min_sources:
                log.warning("Too few correlated sources found. "
                            "{}".format(warnings[fallback]))
                adoutputs.append(ad)
                continue

            log.fullinfo("\nMatched sources:")
            log.fullinfo("   Ref. x Ref. y  Img. x  Img. y\n  {}".
                         format("-"*31))
            for ref, img in zip(*obj_list):
                log.fullinfo("  {:7.2f} {:7.2f} {:7.2f} {:7.2f}".
                            format(ref[0], ref[1], *img))
            log.fullinfo("")

            # Check the fit geometry depending on the number of objects
            if n_corr < min_sources + 2 and (rotate or scale):
                log.warning("Too few objects. Setting rotate=False, scale=False")
                rotate=False
                scale=False

            # Determine a more accurate fit, and get the WCS
            wcs = align_images_from_wcs(ad, ref_ad, transform=transform,
                        search_radius=0.2*firstpasspix,
                        cull_sources=cull_sources, full_wcs=True,
                        rotate=rotate, scale=scale, return_matches=False).wcs
            _write_wcs_keywords(ad, wcs, self.keyword_comments)
            #_apply_model_to_wcs(ad, transform, keyword_comments=self.keyword_comments)
            adoutputs.append(ad)

        # Timestamp and update filenames
        for ad in adoutputs:
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=params["suffix"], strip=True)
        return adoutputs
    
    def determineAstrometricSolution(self, adinputs=None, **params):
        """
        This primitive determines how to modify the WCS of each image to
        produce the best positional match between its sources (OBJCAT) and
        the REFCAT.

        Parameters
        ----------
        initial: float
            search radius for cross-correlation (arcsec)
        final: float
            search radius for object matching (arcsec)
        full_wcs: bool (or None)
            use an updated WCS for each matching iteration, rather than simply
            applying pixel-based corrections to the initial mapping?
            (None => not ('qa' in mode))
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

            # Try to be clever here, and work on the extension with the
            # highest number of matches first, as this will give the most
            # reliable offsets, which can then be used to constrain the other
            # extensions. The problem is we don't know how many matches we'll
            # get until we do it, and that's slow, so use len(OBJCAT) as a proxy.
            objcat_lengths = [len(ext.OBJCAT) if hasattr(ext, 'OBJCAT') else 0
                              for ext in ad]
            objcat_order = np.argsort(objcat_lengths)[::-1]

            pixscale = ad.pixel_scale()
            offset_range = params["initial"] / pixscale  # Search box size
            final = params["final"] / pixscale  # Matching radius
            max_ref_sources = 100 if 'qa' in self.mode else None  # No more than this
            if full_wcs is None:
                full_wcs = not ('qa' in self.mode)

            best_matches = 0
            best_model = None

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

                # We always fit in pixel space of the image
                wcs = WCS(ad[index].hdr)
                xref, yref = refcat['RAJ2000'], refcat['DEJ2000']
                if full_wcs:
                    # Transform REFCAT RA,dec to OBJCAT x,y
                    fit_transform = Transform(Pix2Sky(wcs).inverse.rename("WCS"))
                    fit_transform.factor.fixed = True
                    fit_transform.angle.fixed = True
                    if best_model is not None:
                        fit_transform.x_offset = best_model.x_offset
                        fit_transform.y_offset = best_model.y_offset
                else:
                    # Transform REFCAT x,y (via image WCS) to OBJCAT x,y
                    xref, yref = wcs.all_world2pix(xref, yref, 1)
                    fit_transform = (Transform.create2d(shape=ext.shape)
                                     if best_model is None else best_model.copy())

                fit_transform.add_bounds('x_offset', offset_range)
                fit_transform.add_bounds('y_offset', offset_range)

                # First: estimate number of reference sources in field
                xx, yy = fit_transform(xref, yref)
                # Could tweak y1, y2 here for GNIRS
                x1 = y1 = 0
                y2, x2 = ext.shape
                in_field = np.all((xx > x1-offset_range, xx < x2+offset_range,
                                   yy > y1-offset_range, yy < y2+offset_range), axis=0)
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
                    log.stdinfo('Aligning {}:{} with {} REFCAT and {} OBJCAT sources'.
                                format(ad.filename, extver, num_ref_sources, keep_num))
                    transform = align_catalogs((xref[in_field], yref[in_field]),
                                               (objcat['X_IMAGE'][sorted_idx],
                                                objcat['Y_IMAGE'][sorted_idx]),
                                               transform=fit_transform, tolerance=0.05)
                    matched = match_sources(transform(xref, yref),
                                            (objcat['X_IMAGE'], objcat['Y_IMAGE']),
                                            radius=final, priority=in_field)
                else:
                    log.stdinfo('No REFCAT sources in field of extver {}'.format(extver))
                    continue

                num_matched = np.sum(matched >= 0)
                log.stdinfo("Matched {} objects in OBJCAT:{} against REFCAT".
                            format(num_matched, extver))
                # If this is a "better" match, save it
                # TODO? Some sort of averaging of models?
                if num_matched > max(best_matches, 2):
                    best_matches = num_matched
                    best_model = transform
                    offset_range = 2.5 / pixscale

                if num_matched > 0:
                    # Update WCS in the header and OBJCAT (X_WORLD, Y_WORLD)
                    if full_wcs:
                        _write_wcs_keywords(ext, transform.wcs, self.keyword_comments)
                    else:
                        _apply_model_to_wcs(ext, fit_transform, keyword_comments=self.keyword_comments)
                    new_wcs = WCS(ext.hdr)
                    objcat['X_WORLD'], objcat['Y_WORLD'] = new_wcs.all_pix2world(
                        objcat['X_IMAGE'], objcat['Y_IMAGE'], 1)

                    # Sky coordinates of original CRPIX location with old
                    # and new WCS (easier than using the transform)
                    ra0, dec0 = wcs.all_pix2world([wcs.wcs.crpix], 1)[0]
                    ra1, dec1 = new_wcs.all_pix2world([wcs.wcs.crpix], 1)[0]
                    cosdec = math.cos(math.radians(dec0))
                    delta_ra = 3600 * (ra1-ra0) * cosdec
                    delta_dec = 3600 * (dec1-dec0)

                    # Associate REFCAT properties with their OBJCAT
                    # counterparts. Remember! matched is the reference
                    # (OBJCAT) source for the input (REFCAT) source
                    dra = ddec = []
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
                    log.fullinfo("RA: {:6.2f} +/- {:.2f} arcsec".
                                 format(delta_ra, dra_std))
                    log.fullinfo("Dec:{:6.2f} +/- {:.2f} arcsec".
                                 format(delta_dec, ddec_std))
                    info_list.append({"dra": delta_ra, "dra_std": dra_std,
                                      "ddec": delta_dec, "ddec_std": ddec_std,
                                      "nsamples": int(num_matched)})
                else:
                    log.stdinfo("Could not determine astrometric offset for "
                                "{}:{}".format(ad.filename, extver))
                    info_list.append({})

            # Report the measurement to the fitsstore
            if self.upload and "metrics" in self.upload:
                fitsdict = qap.fitsstore_report(ad, "pe", info_list,
                                                self.calurl_dict,
                                                self.mode, upload=True)

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=params["suffix"], strip=True)

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
        xdiff = adref.detector_x_offset() - adinput.detector_x_offset()
        ydiff = adref.detector_y_offset() - adinput.detector_y_offset()
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
                else tuple(0.5*(x-1) for x in extref.data.shape[::-1])

        wcsref = WCS(extref.hdr)
        ra0, dec0 = wcsref.all_pix2world(center_of_rotation[0],
                                         center_of_rotation[1], 1)
        extin.hdr['CRVAL1'] = float(ra0)
        extin.hdr['CRVAL2'] = float(dec0)
        extin.hdr['CRPIX1'] = center_of_rotation[0] - xdiff
        extin.hdr['CRPIX2'] = center_of_rotation[1] - ydiff
        cd = models.Rotation2D(angle=pa1-pa2)(*wcsref.wcs.cd)
        extin.hdr['CD1_1'] = cd[0][0]
        extin.hdr['CD1_2'] = cd[0][1]
        extin.hdr['CD2_1'] = cd[1][0]
        extin.hdr['CD2_2'] = cd[1][1]
    return adinput

def _apply_model_to_wcs(ad, transform=None, fix_crpix=False,
                        keyword_comments=None):
    """
    This function modifies the WCS of an input image according to a
    Transform that describes how to map the input image pixels to their
    correct location, i.e., an input pixel (x,y) should have the world
    coordinates WCS(m(x,y)), where m is the transformation model.

    Parameters
    ----------
    ad: AstroData
        input image for WCS to be modified
    transform: Transform
        transformation (in pixel space)
    fix_crpix: bool
        if True, keep CRPIXi values fixed; otherwise keep CRVALi fixed
    keyword_comments: dict
        the comment for each FITS keyword

    Returns
    -------
    AstroData: modified AD instance
    """
    if ad.is_single:
        ext = ad
    else:
        if len(ad) > 1:
            raise ValueError("Cannot modify WCS of multi-extension AstroData object")
        ext = ad[0]
    wcs = WCS(ext.hdr)
    affine = transform.affine_matrices(ext.shape)

    # Affine matrix and offset are in python order!
    if fix_crpix:
        wcs.wcs.crval = wcs.all_pix2world(*transform(*wcs.wcs.crpix), 1)
    else:
        wcs.wcs.crpix = np.dot(np.linalg.inv(affine.matrix),
                               wcs.wcs.crpix[::-1] - affine.offset)[::-1]
    # np.flip(axis=None) available in v1.15
    wcs.wcs.cd = np.dot(affine.matrix[::-1,::-1], wcs.wcs.cd)

    _write_wcs_keywords(ext, wcs, keyword_comments)
    return ad

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
    for ax in (1, 2):
        ad.hdr.set('CRPIX{}'.format(ax), wcs.wcs.crpix[ax-1],
                   comment=keyword_comments["CRPIX{}".format(ax)])
        ad.hdr.set('CRVAL{}'.format(ax), wcs.wcs.crval[ax-1],
                   comment=keyword_comments["CRVAL{}".format(ax)])
        for ax2 in 1, 2:
            ad.hdr.set('CD{}_{}'.format(ax, ax2), wcs.wcs.cd[ax-1, ax2-1],
                       comment=keyword_comments["CD{}_{}".format(ax, ax2)])
    return
