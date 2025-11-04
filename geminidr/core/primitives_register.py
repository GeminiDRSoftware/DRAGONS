#
#                                                                  gemini_python
#
#                                                          primitives_register.py
# ------------------------------------------------------------------------------
import math
import numpy as np
from astropy.modeling import models

from copy import deepcopy

from gwcs.wcs import WCS as gWCS
from gwcs import coordinate_frames as cf

from gempy.gemini import gemini_tools as gt
from gempy.gemini import qap_tools as qap
from gempy.library import astromodels as am
from gempy.utils import logutils

from gempy.library.matching import find_alignment_transform, fit_model, match_sources

from geminidr import PrimitivesBASE
from . import parameters_register

from recipe_system.utils.decorators import parameter_override, capture_provenance


# ------------------------------------------------------------------------------
@parameter_override
@capture_provenance
class Register(PrimitivesBASE):
    """
    This is the class containing all of the primitives for registration.
    """
    tagset = None

    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_register)

    def adjustWCSToReference(self, adinputs=None, **params):
        """
        This primitive registers images to a reference image by correcting
        the relative error in their world coordinate systems. This is
        preferably done via alignment of sources common to the reference
        and input images but a fallback method that uses the header offsets
        is also available, if there are too few sources to provide a robust
        alignment, or if the initial WCSs are incorrect.

        The alignment of sources is done via the KDTreeFitter, which does
        not require a direct one-to-one mapping of sources between the
        images. The alignment is performed in the pixel frame of each input
        image, with sources in the reference image being transformed into
        that frame via the existing WCS transforms. Therefore, whatever
        transformation is required can simply be placed at the start of each
        input image's WCS pipeline.

        In order to use the direct mapping method, sources must have been
        detected in the frame and attached to the AstroData instance in an
        OBJCAT extension. This can be accomplished via the detectSources
        primitive.

        It is expected that the relative difference between the WCSs of
        images to be combined should be quite small, so it may not be necessary
        to allow rotation and scaling degrees of freedom when fitting the image
        WCS to the reference WCS. However, if it is desired, the options
        rotate and scale can be used to allow these degrees of freedom. Note
        that these options refer to rotation/scaling of the WCS itself, not the
        images. Significant rotation and scaling of the images themselves
        will generally already be encoded in the WCS, and will be corrected for
        when the images are aligned.

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
        match_radius: float
            search radius (arcsec) for source-to-source correlation matching
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
                        "input images are required for adjustWCSToReference")
            return adinputs

        if not all(len(ad) == 1 for ad in adinputs):
            raise ValueError("All input images must have only one extension.")

        method = params["method"]
        fallback = params["fallback"]
        first_pass = params["first_pass"]
        match_radius = params["match_radius"]
        min_sources = params["min_sources"]
        cull_sources = params["cull_sources"]
        rotate = params["rotate"]
        scale = params["scale"]
        use_wcs = not params["debug_ignore_wcs"]

        # Use first image in list as reference
        adref = adinputs[0]
        log.stdinfo(f"Reference image: {adref.filename}")
        # Create a dummy WCS to facilitate future operations
        if adref[0].wcs is None:
            adref[0].wcs = gWCS([(cf.Frame2D(name="pixels"), models.Identity(len(adref[0].shape))),
                                  (cf.Frame2D(name="world"), None)])

        try:
            ref_objcat = adref[0].OBJCAT
        except AttributeError:
            log.warning(f"Reference image has no OBJCAT. {warnings[fallback]}")
            method = fallback
        else:
            if len(ref_objcat) < min_sources and method == "sources":
                log.warning(f"Too few objects found in reference image {warnings[fallback]}")
                method = fallback
        if method is None:
            return adinputs

        adoutputs = [adref]

        if method is None:
            return adinputs

        for ad in adinputs[1:]:
            msg = ""
            if method == "sources":
                try:
                    objcat = ad[0].OBJCAT
                except AttributeError:
                    msg = f"{ad.filename} image has no OBJCAT. "
                else:
                    if len(objcat) < min_sources:
                        msg = f"{ad.filename} has too few sources. "

            if msg or method == "offsets":
                msg += warnings[fallback]
                log.stdinfo(msg)
                _create_wcs_from_offsets(ad, adref)
                adoutputs.append(ad)
                continue

            # GNIRS WCS is dubious, so update WCS by using the ref
            # image's WCS and the telescope offsets before alignment
            #if ad.instrument() == 'GNIRS':
            #    log.stdinfo("Recomputing WCS for GNIRS from offsets")
            #    ad = _create_wcs_from_offsets(ad, adref)

            log.fullinfo(f"Number of objects in {ad.filename}: {len(objcat)}")
            log.stdinfo(f"Cross-correlating sources in {adref.filename}, {ad.filename}")

            pixscale = ad.pixel_scale()
            if pixscale is None:
                log.warning(f'Cannot determine pixel scale for {ad.filename}. '
                            f'Using a search radius of {first_pass} pixels.')
                firstpasspix = first_pass
                matchpix = match_radius
            else:
                firstpasspix = first_pass / pixscale
                matchpix = match_radius / pixscale

            incoords = (objcat['X_IMAGE'].data - 1, objcat['Y_IMAGE'].data - 1)
            refcoords = (ref_objcat['X_IMAGE'].data - 1, ref_objcat['Y_IMAGE'].data - 1)
            if cull_sources:
                good_src1 = gt.clip_sources(ad)[0]
                good_src2 = gt.clip_sources(adref)[0]
                if len(good_src1) < min_sources or len(good_src2) < min_sources:
                    log.warning("Too few sources in culled list, using full set "
                                "of sources")
                else:
                    incoords = (good_src1["x"] - 1, good_src1["y"] - 1)
                    refcoords = (good_src2["x"] - 1, good_src2["y"] - 1)

            # So what we're doing here is working out where on the input
            # image the sources on the reference image should be if both WCSs
            # were correct. We then work out how to move the sources on the
            # input image so they lie in those positions.
            if use_wcs:
                try:
                    t_init = adref[0].wcs.forward_transform | ad[0].wcs.backward_transform
                except AttributeError:  # for cases with no defined WCS
                    pass
                else:
                    refcoords = t_init(*refcoords)

            # The code used to start with a translation-only model, but this
            # isn't helpful if there's a sizeable rotation or scaling, so
            # let's just try to do the whole thing (if the user asks) and
            # see what happens.
            transform, obj_list = find_alignment_transform(
                incoords, refcoords, transform=None, shape=ad[0].shape,
                search_radius=firstpasspix, match_radius=matchpix,
                rotate=rotate, scale=scale, return_matches=True)

            n_corr = len(obj_list[0])
            if (n_corr < min_sources + rotate + scale) and (rotate or scale):
                log.warning(f"Too few correlated objects ({n_corr}). "
                            "Setting rotate=False, scale=False")
                transform, obj_list = find_alignment_transform(
                    incoords, refcoords, transform=None, shape=ad[0].shape,
                    search_radius=firstpasspix, match_radius=matchpix,
                    rotate=False, scale=False, return_matches=True)
                n_corr = len(obj_list[0])

            log.stdinfo(f"Number of correlated sources: {n_corr}")
            log.fullinfo("\nMatched sources:")
            log.fullinfo("   Ref. x Ref. y  Img. x  Img. y\n  {}".
                         format("-"*31))
            for img, ref in zip(*obj_list):
                log.fullinfo("  {:7.2f} {:7.2f} {:7.2f} {:7.2f}".
                            format(*ref, *img))
            log.stdinfo("")

            if n_corr < min_sources:
                log.warning("Too few correlated sources found. "
                            "{}".format(warnings[fallback]))
                if fallback == 'offsets':
                    _create_wcs_from_offsets(ad, adref)
                adoutputs.append(ad)
                continue

            # If we're not using the WCS, then we want the adjusted AD to
            # have a WCS that is the transform followed by the reference WCS
            if not use_wcs:
                ad[0].wcs = deepcopy(adref[0].wcs)
            try:
                ad[0].wcs.insert_transform(ad[0].wcs.input_frame,
                                           am.make_serializable(transform), after=True)
            except AttributeError:  # no WCS
                ad[0].wcs = gWCS([(cf.Frame2D(name="pixels"), am.make_serializable(transform)),
                                  (cf.Frame2D(name="world"), None)])

            # Update X_WORLD and Y_WORLD (RA and DEC) in OBJCAT
            try:
                x, y = ad[0].OBJCAT['X_IMAGE'].data-1, ad[0].OBJCAT['Y_IMAGE'].data-1
            except (AttributeError, KeyError):
                pass
            else:
                ra, dec = ad[0].wcs(x, y)
                ad[0].OBJCAT['X_WORLD'] = ra
                ad[0].OBJCAT['Y_WORLD'] = dec
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
        rotate: bool
            allow image rotation to align to reference catalog?
        scale: bool
            allow image scaling to align to reference catalog?
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        rotate = params["rotate"]
        scale = params["scale"]

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

            if not ('RAJ2000' in refcat.colnames and 'DEJ2000' in refcat.colnames):
                log.warning(f'REFCAT in {ad.filename} is missing RAJ2000 '
                            'and/or DEJ2000 columns - cannot calculate astrometry')
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
            max_ref_sources = 100 if 'sq' not in self.mode else None  # No more than this

            best_matches = 0
            best_model = None

            for index in objcat_order:
                ext = ad[index]
                if not isinstance(ext.wcs.output_frame, cf.CelestialFrame):
                    log.warning(f"WCS of {ad.filename} extension {ext.id} does"
                                " not transform to a CelestialFrame -- cannot "
                                "perform astrometry")
                    info_list.append({})
                    continue
                try:
                    objcat = ad[index].OBJCAT
                except AttributeError:
                    log.stdinfo(f'No OBJCAT in {ad.filename} extension '
                                f'{ext.id} -- cannot perform astrometry')
                    info_list.append({})
                    continue
                objcat_len = len(objcat)

                # We always fit in pixel space of the image
                xref, yref = ext.wcs.backward_transform(refcat['RAJ2000'],
                                                        refcat['DEJ2000'])
                try:
                    m_init = best_model.copy()
                except AttributeError:
                    m_init = (models.Shift(0) & models.Shift(0))
                    m_init.offset_0.bounds = (m_init.offset_0 - offset_range,
                                              m_init.offset_0 + offset_range)
                    m_init.offset_1.bounds = (m_init.offset_1 - offset_range,
                                              m_init.offset_1 + offset_range)
                    if rotate:
                        m_init = am.Rotate2D(0, bounds={'angle': (-5, 5)}) | m_init
                    if scale:
                        m_init = am.Scale2D(1, bounds={'factor': (0.95, 1.05)}) | m_init

                # First: estimate number of reference sources in field
                xx, yy = m_init.inverse(xref, yref)
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
                                    log.stdinfo(f'Using {filt} magnitude instead')

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
                    log.stdinfo(f'Aligning {ad.filename} extension {ext.id} '
                                f'with {num_ref_sources} REFCAT and '
                                f'{keep_num} OBJCAT sources')
                    transform = fit_model(m_init,
                                          (objcat['X_IMAGE'][sorted_idx]-1,
                                           objcat['Y_IMAGE'][sorted_idx]-1),
                                          (xref[in_field], yref[in_field]),
                                          sigma=10.0, tolerance=0.0001, brute=True)
                    matched = match_sources(transform.inverse(xref, yref),
                                            (objcat['X_IMAGE']-1, objcat['Y_IMAGE']-1),
                                            radius=final)
                else:
                    log.stdinfo(f'No REFCAT sources in field of extension {ext.id}')
                    continue

                num_matched = np.sum(matched >= 0)
                log.stdinfo(f"Matched {num_matched} objects in OBJCAT "
                            f"extension {ext.id} against REFCAT")
                # If this is a "better" match, save it
                # TODO? Some sort of averaging of models?
                if num_matched > max(best_matches, 2):
                    best_matches = num_matched
                    best_model = transform
                    offset_range = 2.5 / pixscale

                if num_matched > 0:
                    # Update OBJCAT (X_WORLD, Y_WORLD)
                    crpix = (0, 0)  # doesn't matter since we're only reporting a shift
                    ra0, dec0 = ext.wcs(*crpix)
                    ext.wcs.insert_transform(ext.wcs.input_frame,
                                             am.make_serializable(transform), after=True)
                    objcat['X_WORLD'], objcat['Y_WORLD'] = ext.wcs(objcat['X_IMAGE']-1,
                                                                   objcat['Y_IMAGE']-1)

                    # Sky coordinates of original CRPIX location with old
                    # and new WCS (easier than using the transform)
                    ra1, dec1 = ext.wcs(*crpix)
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
                            ddec.append(3600*(objcat['Y_WORLD'][m] -
                                              refcat['DEJ2000'][i]))
                    dra_std = np.std(dra)
                    ddec_std = np.std(ddec)
                    log.fullinfo(f"WCS Updated for extension {ext.id}. "
                                 "Astrometric offset is:")
                    log.fullinfo(f"RA: {delta_ra:6.2f} +/- {dra_std:.2f} arcsec")
                    log.fullinfo(f"Dec:{delta_dec:6.2f} +/- {ddec_std:.2f} arcsec")
                    info_list.append({"dra": delta_ra, "dra_std": dra_std,
                                      "ddec": delta_dec, "ddec_std": ddec_std,
                                      "nsamples": int(num_matched)})
                else:
                    log.stdinfo("Could not determine astrometric offset for "
                                f"{ad.filename} extension {ext.id}")
                    info_list.append({})

            # Report the measurement to the fitsstore
            if self.upload and "metrics" in self.upload:
                fitsdict = qap.fitsstore_report(ad, "pe", info_list,
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
    a transform between pixel coordinates. Its primary role is for GNIRS.
    For ease, it works out the (RA,DEC) of the centre of rotation in the
    reference image and determines where in the input image this is. The
    AstroData object's WCS is updated with a new WCS based on the WCS of the
    reference AD and the relative offsets/rotation from the headers.

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
        return

    log.stdinfo(f"Updating WCS of {adinput.filename} from {adref.filename}")
    try:
        # Coerce to float to raise TypeError if a descriptor returns None
        xoff_in = float(adinput.detector_x_offset())
        yoff_in = float(adinput.detector_y_offset())
        xoff_ref = float(adref.detector_x_offset())
        yoff_ref = float(adref.detector_y_offset())
        pa1 = adref.phu['PA']
        pa2 = adinput.phu['PA']
    except (KeyError, TypeError):  # TypeError if offset is None
        log.warning("Cannot obtain necessary offsets from headers "
                    "so no change will be made")
        return

    if center_of_rotation is None:
        if 'GNIRS' in adref.tags:
            center_of_rotation = (519.0, 629.0)  # (y, x; 0-indexed)
        else:
            try:
                for m in adref[0].wcs.forward_transform:
                    if isinstance(m, models.RotateNative2Celestial):
                        ra, dec = m.lon.value, m.lat.value
                        center_of_rotation = adref[0].wcs.backward_transform(ra, dec)
                        break
            except (AttributeError, IndexError, TypeError):
                if len(adref) == 1:
                    # Assume it's the center of the image
                    center_of_rotation = tuple(0.5 * (x - 1)
                                               for x in adref[0].shape[::-1])
                else:
                    log.warning("Cannot determine center of rotation so no "
                                "change will be made")
                    return

    try:
        t = ((models.Shift(-xoff_in - center_of_rotation[1]) & models.Shift(-yoff_in - center_of_rotation[0])) |
             models.Rotation2D(pa1 - pa2) |
             (models.Shift(xoff_ref + center_of_rotation[1]) & models.Shift(yoff_ref + center_of_rotation[0])))
    except TypeError:
        log.warning("Problem creating offset transform so no change will be made")
        return
    adinput[0].wcs = deepcopy(adref[0].wcs)
    adinput[0].wcs.insert_transform(adinput[0].wcs.input_frame, t, after=True)
