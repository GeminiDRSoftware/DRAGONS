#
#                                                                  gemini_python
#
#                                                      primitives_gsaoi_image.py
# ------------------------------------------------------------------------------
import re
import math
from functools import reduce
import numpy as np

from matplotlib import collections, pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from astropy.modeling import models, fitting, Model
from astropy.stats import sigma_clip
from astropy import table
from astropy import units as u

from gwcs.wcs import WCS as gWCS
from gwcs import coordinate_frames as cf

from astrodata import wcs as adwcs

from gempy.gemini import gemini_tools as gt
from gempy.library import astromodels as am, astrotools as at
from gempy.library.matching import find_alignment_transform, fit_model, match_sources
from gempy.gemini import qap_tools as qap

from geminidr.core import Image, Photometry
from recipe_system.utils.decorators import parameter_override, capture_provenance

from .primitives_gsaoi import GSAOI
from . import parameters_gsaoi_image
from .lookups import gsaoi_static_distortion_info as gsdi
from .lookups.geometry_conf import tile_gaps


@parameter_override
@capture_provenance
class GSAOIImage(GSAOI, Image, Photometry):
    """
    This is the class containing all of the preprocessing primitives
    for the GSAOIImage level of the type hierarchy tree. It inherits all
    the primitives from the level above
    """
    tagset = {"GEMINI", "GSAOI", "IMAGE"}

    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_gsaoi_image)

    def adjustWCSToReference(self, adinputs=None, **params):
        """
        This primitive registers images to a reference image by correcting
        the relative error in their world coordinate systems. The function
        uses points of reference common to the reference image and the
        input images to fit the input WCS to the reference one. The fit
        is done via the KDTreeFitter, which does not require a direct
        one-to-one mapping of sources between the images.

        The GSAOI version of the primitive works differently to the core
        version as it performs the matching in the "static" reference frame,
        where the static distortion has been corrected. And, after the normal
        shift/scale/rotate transform is calculated using the KDTreeFitter, a
        polynomial transform is computed and iteratively improved from
        one-to-one source matching (to keep it under control).

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        first_pass: float
            search radius (arcsec) for the initial alignment matching
        final: float
            search radius (arcsec) for final object matching
        min_sources: int
            minimum number of matched sources required to apply a WCS shift
        cull_sources: bool
            remove sub-optimal (saturated and/or non-stellar) sources before
            alignment?
        rotate: bool
            allow image rotation to align to reference image?
        scale: bool
            allow image scaling to align to reference image?
        max_iters: int
            maximum number of iterations for polynomial fit
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        if len(adinputs) < 2:
            log.warning("No correction will be performed, since at least two "
                        f"input images are required for {self.myself()}")
            return adinputs

        initial = params["first_pass"]
        final = params["final"]
        min_sources = params["min_sources"]
        cull_sources = params["cull_sources"]
        rotate = params["rotate"]
        scale = params["scale"]
        order = params["order"]
        max_iters = params["max_iters"]
        debug = params["debug"]
        debug_figs = []

        self._attach_static_distortion(adinputs)

        adref = adinputs[0]
        try:
            ref_objcat = merge_gsaoi_objcats(adref, cull_sources=cull_sources)
        except ValueError:
            log.warning(f"Cannot run {self.myself()} as there are no "
                        f"OBJCATs on reference image {adref.filename}")
            return adinputs

        refcoords = (ref_objcat['X_STATIC'], ref_objcat['Y_STATIC'])
        if ("variable" in adref[0].wcs.available_frames and
                self.timestamp_keys["determineAstrometricSolution"] in adref.phu):
            log.debug(f"Variable frame identified in reference {adref.filename}")
            var_transform = adref[0].wcs.get_transform("static", "variable")
            refcoords = var_transform(*refcoords)

        # This is the last transform in the pipeline, from the variable frame
        # (or static frame if there's no variable frame) to the world frame.
        # This should be the same for all extensions but we can't check easily
        ref_transform = adref[0].wcs.pipeline[-2].transform

        # This means a warning will be triggered if images have a rotation that
        # differs by an amount large enough to prevent an object-to-object match
        faux_shape = (4000 / (final / adref.pixel_scale()),) * 2

        for ad in adinputs[1:]:
            objcat = merge_gsaoi_objcats(ad, cull_sources=cull_sources)
            if objcat is None:
                log.warning(f"No sources to correlate in {ad.filename}. "
                            "Cannot adjust WCS.")
                continue

            incoords = (objcat['X_STATIC'], objcat['Y_STATIC'])
            # from this 'static' frame to the reference's variable (or static)
            transform = ad[0].wcs.get_transform(
                "static", ad[0].wcs.output_frame) | ref_transform.inverse

            if ("variable" in ad[0].wcs.available_frames and
                    self.timestamp_keys["determineAstrometricSolution"] in ad.phu):
                log.stdinfo(f"Matching sources between {ad.filename} and "
                            f"{adref.filename} using distortion determined "
                            "by determineAstrometricSolution.")
                # This order will assign the closest OBJCAT to each ref-OBJCAT source
                matched = match_sources(refcoords, transform(*incoords),
                                        radius=final)
            else:
                log.stdinfo("Cross-correlating sources between "
                            f"{ad.filename} and {adref.filename}")
                transform = find_alignment_transform(incoords, refcoords, transform=transform,
                                                     shape=faux_shape, search_radius=initial,
                                                     rotate=rotate, scale=scale,
                                                     sigma=0.2, factor=1/0.02, brute=True)
                # This order will assign the closest OBJCAT to each REFCAT source
                matched = match_sources(refcoords, transform(*incoords),
                                        radius=final)
                num_matched = np.sum(matched >= 0)
                if (rotate or scale) and num_matched < min_sources + rotate + scale:
                    log.warning(f"Too few correlated objects ({num_matched}). "
                                "Setting rotate=False, scale=False")
                    transform = find_alignment_transform(incoords, refcoords, transform=transform,
                                                         shape=faux_shape, rotate=False, scale=False,
                                                         sigma=0.2, factor=1/0.02, brute=True)
                    matched = match_sources(refcoords, transform(*incoords),
                                            radius=final)

            if debug:
                debug_figs.append(make_alignment_figure(
                    refcoords, transform(*incoords), matched,
                    adref.filename, ad.filename, final))

            num_matched = np.sum(matched >= 0)
            if num_matched < min_sources:
                log.warning(f"Too few correlated sources in {ad.filename} "
                            f"({num_matched}). Cannot adjust WCS.")
                continue

            if num_matched > 0:
                if num_matched > 2:
                    transform, matched = create_polynomial_transform(transform,
                                                incoords, refcoords, order=order,
                                                max_iters=max_iters,
                                                match_radius=final, log=self.log)
                    num_matched = np.sum(matched >= 0)
                    log.stdinfo("Number of correlated sources: {}".format(num_matched))
                    log.fullinfo("\nMatched sources:  Ref. ext Ref. x  Ref. y  "
                                 "Img. ext Img. x  Img. y\n  {}".format("-" * 31))
                    #xmatched = np.full((len(objcat),), -999, dtype=float)
                    #ymatched = np.full((len(objcat),), -999, dtype=float)
                    for i, m in enumerate(matched):
                        if m >= 0:
                            objext = objcat['ext_index'][m]
                            objx, objy = objcat['X_IMAGE'][m], objcat['Y_IMAGE'][m]
                            refext = ref_objcat['ext_index'][i]
                            refx, refy = ref_objcat['X_IMAGE'][i], ref_objcat['Y_IMAGE'][i]
                            log.fullinfo(f"  {objext:7d} {objx:7.2f} {objy:7.2f}"
                                         f"  {refext:7d} {refx:7.2f} {refy:7.2f}")
                            #xmatched[m] = refcoords[0][i]
                            #ymatched[m] = refcoords[1][i]
                    log.fullinfo("")
                    #objcat["X_MATCHED"], objcat["Y_MATCHED"] = xmatched, ymatched
                    #objcat["X_TRANS"], objcat['Y_TRANS'] = transform(*incoords)
                    #objcat.write(f'obj_{ad.filename}', overwrite=True)

                for ext in ad:
                    var_frame = cf.Frame2D(unit=(u.arcsec, u.arcsec), name="variable")
                    ext.wcs = gWCS(ext.wcs.pipeline[:1] +
                                   [(ext.wcs.pipeline[1].frame, am.make_serializable(transform)),
                                    (var_frame, ref_transform),
                                    (ext.wcs.output_frame, None)])
            else:
                log.stdinfo("Could not determine astrometric offset for "
                            f"{ad.filename}")

        if debug:
            with PdfPages(f"debug_{self.myself()}.pdf") as pdf:
                for fig in debug_figs:
                    pdf.savefig(fig)

        # Timestamp and update filename
        for ad in adinputs:
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=params["suffix"], strip=True)
        return adinputs

    def determineAstrometricSolution(self, adinputs=None, **params):
        """
        This primitive determines how to modify the WCS of each image to
        produce the best positional match between its sources (OBJCAT) and
        the REFCAT. This is a GSAOI-specific version that differs from the
        core primitive in three ways: (a) it combines all the extensions to
        provide a single offset model for the entire AD object; (b) it
        inserts this model after the static distortion correction, instead
        of at the start of the WCS forward_transform; (c) the model is two
        separate Polynomial2D objects rather than a scale/shift/rotate
        transform.

        Parameters
        ----------
        initial : float
            search radius for cross-correlation (arcsec)
        final : float
            search radius for object matching (arcsec)
        rotate : bool
            allow image rotation in initial alignment of reference catalog?
        scale : bool
            allow image scaling in initial alignment of reference catalog?
        order : int
            order of polynomial fit in each ordinate
        max_iters : int
            maximum number of iterations when performing polynomial fit
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        rotate = params["rotate"]
        scale = params["scale"]
        initial = params["initial"]
        final = params["final"]
        order = params["order"]
        max_iters = params["max_iters"]
        debug = params["debug"]
        debug_figs = []

        self._attach_static_distortion(adinputs)
        if any(self.timestamp_keys["adjustWCSToReference"] in ad.phu for ad in adinputs):
            log.warning("One or more inputs has been processed by "
                        f"adjustWCSToReference. {self.myself()} will not"
                        "preserve these alignments.")

        for ad in adinputs:
            if len(ad) == 1:
                raise OSError(f"{self.myself()} must be run on unmosaicked "
                              f"GSAOI data, but {ad.filename} has been "
                              "mosaicked/tiled.")
            # Check we have a REFCAT and at least one OBJCAT to match
            try:
                refcat = ad.REFCAT
            except AttributeError:
                log.warning(f"No REFCAT in {ad.filename} - cannot calculate astrometry")
                continue
            if not ('RAJ2000' in refcat.colnames and 'DEJ2000' in refcat.colnames):
                log.warning(f"REFCAT in {ad.filename} is missing RAJ2000 "
                            "and/or DEJ2000 columns - cannot calculate astrometry")
                continue
            try:
                objcat = merge_gsaoi_objcats(ad)
            except ValueError:
                log.warning(f"No OBJCATs in {ad.filename} - cannot match to REFCAT")
                continue

            if not all([isinstance(getattr(ext.wcs, 'output_frame'),
                                   cf.CelestialFrame) for ext in ad]):
                log.warning("Missing CelestialFrame in at least one extension"
                            f" of {ad.filename}")
                continue

            # We're going to fit in the static-corrected coordinate frame
            # Find the its boundaries so we can cull the REFCAT
            static_transforms = [ext.wcs.get_transform(ext.wcs.input_frame,
                                                       "static") for ext in ad]
            all_corners = np.concatenate([np.array(static(*np.array(at.get_corners(ext.shape)).T))
                                          for ext, static in zip(ad, static_transforms)], axis=1)
            xmin, ymin = all_corners.min(axis=1) - 2 * initial
            xmax, ymax = all_corners.max(axis=1) + 2 * initial

            # This will be the same for all extensions if the user hasn't
            # hacked it (which is something we can't really check)
            static_to_world_transform = ad[0].wcs.get_transform("static",
                                                                ad[0].wcs.output_frame)
            xref, yref = static_to_world_transform.inverse(refcat['RAJ2000'],
                                                           refcat['DEJ2000'])
            if debug:
                refcat["X_STATIC"], refcat["Y_STATIC"] = xref, yref
                refcat.write(f"ref_{ad.filename}", overwrite=True)
                objcat.write(f"obj_{ad.filename}", overwrite=True)
            in_field = np.all((xref > xmin, xref < xmax,
                               yref > ymin, yref < ymax), axis=0)
            num_ref_sources = in_field.sum()
            if num_ref_sources == 0:
                log.stdinfo(f"No REFCAT sources in field of {ad.filename}")
                continue

            m_init = (models.Shift(0, bounds={'offset': (-initial, initial)}) &
                      models.Shift(0, bounds={'offset': (-initial, initial)}))
            if rotate:
                m_init = am.Rotate2D(0, bounds={'angle': (-2, 2)}) | m_init
            if scale:
                m_init = am.Scale2D(1, bounds={'factor': (0.98, 1.02)}) | m_init

            # How many objects do we want to try to match? Keep brightest ones only
            objcat_len = len(objcat)
            if objcat_len > 3 * num_ref_sources:
                keep_num = max(3 * num_ref_sources, min(10, objcat_len))
            else:
                keep_num = objcat_len
            sorted_idx = np.argsort(objcat['MAG_AUTO'])[:keep_num]

            log.stdinfo(f"Aligning {ad.filename} with {num_ref_sources} REFCAT"
                        f" and {keep_num} OBJCAT sources")
            in_coords = (objcat['X_STATIC'][sorted_idx],
                         objcat['Y_STATIC'][sorted_idx])
            ref_coords = (xref[in_field], yref[in_field])

            transform = fit_model(m_init, in_coords, ref_coords,
                                  sigma=0.2, tolerance=0.001,
                                  brute=True, scale=1/0.02)

            # This order will assign the closest OBJCAT to each REFCAT source
            matched = match_sources((xref, yref),
                                    transform(objcat['X_STATIC'], objcat['Y_STATIC']),
                                    radius=final)
            num_matched = np.sum(matched >= 0)
            log.stdinfo(f"Initial match: {num_matched} objects")

            if num_matched > 0:
                if num_matched > 2:
                    transform, matched = create_polynomial_transform(transform,
                                                (objcat['X_STATIC'], objcat['Y_STATIC']),
                                                (xref, yref), order=order,
                                                max_iters=max_iters,
                                                match_radius=final, log=self.log)
                    num_matched = np.sum(matched >= 0)
                    log.stdinfo(f"Final match: {num_matched} objects")
                else:
                    log.warning("Insufficient matches to perform distortion "
                                "correction - performing simple alignment only."
                                " Perhaps try increasing the value of "
                                f"'final' (using {final})?")

                # Associate REFCAT properties with their OBJCAT
                # counterparts. Remember! matched is the reference
                # (OBJCAT) source for the input (REFCAT) source
                dx, dy = [], []
                xtrans, ytrans = transform(objcat['X_STATIC'], objcat['Y_STATIC'])
                ratrans, dectrans = static_to_world_transform(xtrans, ytrans)
                cospa, sinpa = math.cos(ad.phu['PA']), math.sin(ad.phu['PA'])
                log.fullinfo("   xpix   ypix   RA  (OBJCAT)  DEC   RA "
                             "(xformed)  DEC   RA  (REFCAT)  DEC")
                log.fullinfo("-" * 75)

                if debug:
                    debug_figs.append(make_alignment_figure(
                        transform.inverse(xref, yref),
                        (objcat['X_STATIC'], objcat['Y_STATIC']),
                        matched, "refcat", ad.filename, final))

                for i, m in enumerate(matched):
                    if m >= 0:
                        try:
                            objcat['REF_NUMBER'][m] = refcat['Id'][i]
                        except KeyError:  # no such columns in REFCAT
                            pass
                        try:
                            objcat['REF_MAG'][m] = refcat['filtermag'][i]
                            objcat['REF_MAG_ERR'][m] = refcat['filtermag_err'][i]
                        except KeyError:  # no such columns in REFCAT
                            pass
                        dx.append(xref[i] - xtrans[m])
                        dy.append(yref[i] - ytrans[m])
                        refra, refdec = refcat['RAJ2000', 'DEJ2000'][i]
                        extid, objx, objy, objra, objdec = objcat[
                            'ext_index', 'X_IMAGE', 'Y_IMAGE', 'X_WORLD', 'Y_WORLD'][m]
                        log.stdinfo(f"{extid:1d} {objx:6.1f} {objy:6.1f} "
                                    f"{objra:9.5f} {objdec:9.5f} {ratrans[m]:9.5f} "
                                    f"{dectrans[m]:9.5f} {refra:9.5f} {refdec:9.5f}")

                x0, y0 = transform(0, 0)
                delta_ra = x0 * cospa - y0 * sinpa
                delta_dec = x0 * sinpa + y0 * cospa
                dra = np.array(dx) * cospa - np.array(dy) * sinpa
                ddec = np.array(dx) * sinpa + np.array(dy) * cospa
                dra_std, ddec_std = dra.std(), ddec.std()
                log.stdinfo(f"\nWCS Updated for {ad.filename}. Astrometric "
                             "offset is:")
                log.stdinfo("RA: {:6.2f} +/- {:.2f} arcsec".
                             format(delta_ra, dra_std))
                log.stdinfo("Dec:{:6.2f} +/- {:.2f} arcsec".
                             format(delta_dec, ddec_std))
                info_list = [{"dra": delta_ra, "dra_std": dra_std,
                              "ddec": delta_dec, "ddec_std": ddec_std,
                              "nsamples": int(num_matched)}]
                # Report the measurement to the fitsstore
                if self.upload and "metrics" in self.upload:
                    qap.fitsstore_report(ad, "pe", info_list,
                                         self.mode, upload=True)

                # Update OBJCAT (X_WORLD, Y_WORLD)
                transform = am.make_serializable(transform)
                for index, ext in enumerate(ad):
                    # TODO: use insert_frame method
                    var_frame = cf.Frame2D(unit=(u.arcsec, u.arcsec), name="variable")
                    ext.wcs = gWCS(ext.wcs.pipeline[:1] +
                                   [(ext.wcs.pipeline[1].frame, transform),
                                    (var_frame, static_to_world_transform),
                                    (ext.wcs.output_frame, None)])
                    ext.OBJCAT = objcat[objcat['ext_index'] == index]
                    ext.OBJCAT['X_WORLD'], ext.OBJCAT['Y_WORLD'] = ext.wcs(ext.OBJCAT['X_IMAGE'].data-1,
                                                                           ext.OBJCAT['Y_IMAGE'].data-1)
                    ext.OBJCAT.remove_columns(['ext_index', 'X_STATIC', 'Y_STATIC'])
            else:
                log.stdinfo("Could not determine astrometric offset for "
                            f"{ad.filename}")

            if debug:
                with PdfPages(f"debug_{self.myself()}.pdf") as pdf:
                    for fig in debug_figs:
                        pdf.savefig(fig)

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=params["suffix"], strip=True)
        return adinputs

    def makeLampFlat(self, adinputs=None, **params):
        """
        This produces an appropriate stacked GSAOI imaging flat, based on
        the inputs, since one of two procedures must be followed.

        In the standard recipe, the inputs will have come from getList and
        so will all have the same filter and will all need the same recipe.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        # Leave now with empty list to avoid error when looking at adinputs[0]
        if not adinputs:
            return adinputs

        if adinputs[0].effective_wavelength(output_units='micrometers') < 1.4:
            log.stdinfo('Using stackFrames to make flatfield')
            params.update({'scale': False, 'zero': False})
            adinputs = self.stackFrames(adinputs, **params)
        else:
            log.stdinfo('Using standard makeLampFlat primitive to make flatfield')
            adinputs = super().makeLampFlat(adinputs, **params)

        return adinputs

    def _attach_static_distortion(self, adinputs=None):
        """
        This primitive modifies the WCS of its input AD objects to include the
        static distortion read from the lookup table. It will ignore any ADs
        which have CoordinateFrames named "static" on all the extensions, and
        raise an OSError if any ADs have "static" on some (but not all)
        extensions.
        """
        log = self.log

        # Check the inputs haven't been mosaicked or tiled
        mosaic_kws = {self.timestamp_keys[prim] for prim in ("mosaicDetectors",
                                                             "tileArrays")}
        if any(mosaic_kws.intersection(ad.phu) for ad in adinputs):
            raise ValueError(f"Inputs to {self.myself()} must not have been "
                             "mosaicked or tiled.")

        static_corrections = gsdi.STATIC_CORRECTIONS
        sdmodels = []
        sdsubmodels = {}
        for direction in ("forward", "backward"):
            sdsubmodels[direction] = []
            for component in static_corrections[direction]:
                model_type = getattr(models, component["model"])
                for arr in component["parameters"]:
                    for ordinate in "xy":
                        pars = arr[ordinate]
                        max_xpower = max([int(k[1]) for k in pars])
                        max_ypower = max([int(k[-1]) for k in pars])
                        if component["model"] == "Polynomial2D":
                            degree = {"degree": max(max_xpower, max_ypower)}
                        else:
                            degree = {"xdegree": max_xpower, "ydegree": max_ypower}
                        sdsubmodels[direction].append(model_type(**degree, **pars))
        for index, pixref in enumerate(static_corrections["pixel_references"]):
            sdmodel = (models.Mapping((0, 1, 0, 1)) |
                  (reduce(Model.__add__, sdsubmodels["forward"][index * 2::8]) &
                   reduce(Model.__add__, sdsubmodels["forward"][index * 2 + 1::8])))
            sdmodel.inverse = (models.Mapping((0, 1, 0, 1)) |
                          (reduce(Model.__add__, sdsubmodels["backward"][index * 2::8]) &
                           reduce(Model.__add__, sdsubmodels["backward"][index * 2 + 1::8])))
            xref, yref = sdmodel.inverse(0, 0)
            if 0 < xref < 2048 and 0 < yref < 2048:
                ref_location = (index, xref-1, yref-1)  # store 0-indexed pixel location
            sdmodels.append(sdmodel)

        for ad in adinputs:
            # TODO: make this work when a detector is missing
            # Ideally we should be able to handle data where some detectors
            # are missing. But we require that the reference extension exists
            # because we build the WCS from it, so we can't cater for *all*
            # missing detectors.
            if len(ad) != 4:
                raise ValueError(f"{ad.filename} does not have 4 extensions")
            applied_static = sum("static" in ext.wcs.available_frames for ext in ad)
            if applied_static not in (0, len(ad)):
                raise OSError(f"Some (but not all) extensions in {ad.filename}"
                              " have had the static disortion correction applied")
            # No-op silently
            if applied_static == len(ad):
                continue
            ref_wcs = ad[ref_location[0]].wcs
            ra, dec = ref_wcs(*ref_location[1:])
            for ext, arrsec, sdmodel in zip(ad, ad.array_section(), sdmodels):
                # Include ROI shift
                sdmodel = (models.Shift(arrsec.x1 + 1) &
                           models.Shift(arrsec.y1 + 1) | sdmodel)
                static_frame = cf.Frame2D(unit=(u.arcsec, u.arcsec), name="static")
                # the PA header keyword isn't good enough
                wcs_dict = adwcs.gwcs_to_fits(ad[ref_location[0]].nddata)
                pa = np.arctan2(wcs_dict["CD2_1"] - wcs_dict["CD1_2"],
                                wcs_dict["CD1_1"] + wcs_dict["CD2_2"])
                if abs(pa) < 0.00175:  # radians ~ 0.01 degrees
                    pa = 0
                # Rotation2D() breaks things because it explicitly converts
                # the Column objects to Quantity objects, which retain units
                # of "pix" that Pix2Sky objects to. The AffineTransformation2D
                # does some multiplications (like Scale) which are agnostic to
                # the presence or absence of units.
                sky_model = models.AffineTransformation2D(
                    matrix=np.array([[math.cos(pa), -math.sin(pa)],
                                     [math.sin(pa), math.cos(pa)]]) / 3600)
                sky_model |= models.Pix2Sky_TAN() | models.RotateNative2Celestial(ra, dec, 180)
                ext.wcs = gWCS([(ext.wcs.input_frame, sdmodel),
                                (static_frame, sky_model),
                                (ext.wcs.output_frame, None)])

        return adinputs

    def _fields_overlap(self, ad1, ad2, frac_FOV=1.0):
        """
        Checks whether the fields of view of two F2 images overlap
        sufficiently to be considerd part of a single ExposureGroup.
        GSAOIImage requires its own code since it has multiple detectors

        Parameters
        ----------
        ad1: AstroData
            one of the input AD objects
        ad2: AstroData
            the other input AD object
        frac_FOV: float (0 < frac_FOV <= 1)
            fraction of the field of view for an overlap to be considered. If
            frac_FOV=1, *any* overlap is considered to be OK

        Returns
        -------
        bool: do the fields overlap sufficiently?
        """
        # In case they've been mosaicked in some way
        if len(ad1) * len(ad2) == 1:
            return super()._fields_overlap(ad1, ad2, frac_FOV)
        elif len(ad1) == 1 or len(ad2) == 1:
            raise NotImplementedError("Cannot compute overlap for GSAOI images"
                                      "if only one has been mosaicked")

        # We try to deal with the possibility of only 3 working detectors
        # which means we can't make assumptions about their relative positions
        # Compute center of FOV in ad[0] pixel coords
        gaps = tile_gaps[ad1.detector_name()]
        centers = [[2048 + 0.5 * gap if p1 == 0 else -0.5 * gap
                    for p1, gap in zip(ad[0].detector_section()[::2], gaps)]
                   for ad in (ad1, ad2)]
        pos2 = ad2[0].wcs.invert(*ad1[0].wcs(*centers[0]))
        return all(abs(p1 - p2) < frac_FOV * (4096 + gap)
                   for p1, p2, gap in zip(centers[1], pos2, gaps))


def merge_gsaoi_objcats(ad, cull_sources=False):
    """
    This function takes an AstroDataGsaoi object and combines the OBJCATs on
    the 4 extensions into a single Table, while also adding the static-
    distortion-corrected coordinates based on the WCS on each extension.

    Parameters
    ----------
    ad : AstroData
        input AstroDataGsaoi object with OBJCATs
    cull_sources : bool
        include only non-saturated point-like sources?

    Returns
    -------
    Table : merged OBJCATs with static-distortion-corrected coords added
    """
    # If we're not going to clip the OBJCATs we still need an iterable
    # for the main loop
    clipped_objcats = gt.clip_sources(ad) if cull_sources else ad
    objcats = []
    for index, (ext, objcat) in enumerate(zip(ad, clipped_objcats)):
        if cull_sources and not objcat:
            continue
        elif not cull_sources:
            try:
                objcat = ext.OBJCAT.copy()
            except AttributeError:
                continue
        objcat['ext_index'] = index
        static = ext.wcs.get_transform(ext.wcs.input_frame, "static")
        objcat['X_STATIC'], objcat['Y_STATIC'] = static(objcat['X_IMAGE'] - 1,
                                                        objcat['Y_IMAGE'] - 1)
        objcats.append(objcat)
    if objcats:
        return table.vstack(objcats, metadata_conflicts='silent')
    return


def create_polynomial_transform(transform, in_coords, ref_coords, order=3,
                                max_iters=5, match_radius=0.1, clip=True,
                                log=None):
    """
    This function maps a set of 2D input coordinates to a set of 2D reference
    coordiantes using a pair of Polynomial2D object (one for each ordinate),
    given an initial transforming model.

    Parameters
    ----------
    transform : astropy.models.Model
        the initial guess of the transform between coordinate frames
    in_coords : 2-tuple of sequences
        input coordinates being mapped to reference
    ref_coords : 2-tuple of sequences
        reference coordinates
    order : int
        order of polynomial fit in each ordinate
    max_iters : int
        maximum number of iterations to perform
    match_radius : float
        matching radius for sources (in units of the reference coords)
    clip : bool
        sigma-clip sources after matching?
    log : logging object

    Returns
    -------
    transform : Model
        a model (and its inverse) to map in_coords to ref_coords
    matched: ndarray
        matched incoord for each refcoord
    """
    affine = adwcs.calculate_affine_matrices(transform, shape=(100, 100))
    num_params = [len(models.Polynomial2D(degree=i).parameters)
                  for i in range(1, order+1)]

    orig_order = last_order = order
    xref, yref = ref_coords
    xin, yin = in_coords
    if clip:
        fit_it = fitting.FittingWithOutlierRemoval(fitting.LinearLSQFitter(),
                                                   sigma_clip, sigma=3)
    else:
        fit_it = fitting.LinearLSQFitter()

    matched = match_sources((xref, yref), transform(xin, yin),
                            radius=match_radius)
    num_matched = np.sum(matched >= 0)
    if num_matched < 3:
        raise RuntimeError(f"Only {num_matched} matched sources. Must be >=3.")

    niter = 0
    while True:
        # No point trying to compute a more complex model if it will
        # be insufficiently constrained
        order = min(np.searchsorted(num_params, num_matched, side="right"),
                    orig_order)
        if order < last_order:
            log.warning(f"Downgrading fit to order {order} due to "
                        "limited number of matches.")
        elif order > last_order:
            log.stdinfo(f"Upgrading fit to order {order} due to "
                        "increased number of matches.")

        xmodel = models.Polynomial2D(degree=order, c0_0=affine.offset[1],
                                     c1_0=affine.matrix[1, 1],
                                     c0_1=affine.matrix[1, 0])
        ymodel = models.Polynomial2D(degree=order, c0_0=affine.offset[0],
                                     c1_0=affine.matrix[0, 1],
                                     c0_1=affine.matrix[0, 0])
        old_num_matched = num_matched
        xobj_matched, yobj_matched = [], []
        xref_matched, yref_matched = [], []
        for i, m in enumerate(matched):
            if m >= 0:
                xref_matched.append(xref[i])
                yref_matched.append(yref[i])
                xobj_matched.append(xin[m])
                yobj_matched.append(yin[m])
        xmodel = fit_it(xmodel, np.array(xobj_matched), np.array(yobj_matched), xref_matched)
        ymodel = fit_it(ymodel, np.array(xobj_matched), np.array(yobj_matched), yref_matched)
        if clip:
            xmodel, ymodel = xmodel[0], ymodel[0]
        transform = models.Mapping((0, 1, 0, 1)) | (xmodel & ymodel)
        matched = match_sources((xref, yref), transform(xin, yin),
                                radius=match_radius)
        num_matched = np.sum(matched >= 0)
        last_order = order
        niter += 1
        log.debug(f"Iteration {niter}: Matched {num_matched} objects")
        if num_matched == old_num_matched or niter > max_iters:
            break
    xmodel_inv = fit_it(xmodel, np.array(xref_matched), np.array(yref_matched), xobj_matched)
    ymodel_inv = fit_it(ymodel, np.array(xref_matched), np.array(yref_matched), yobj_matched)
    if clip:
        xmodel_inv, ymodel_inv = xmodel_inv[0], ymodel_inv[0]
    transform.inverse = models.Mapping((0, 1, 0, 1)) | (xmodel_inv & ymodel_inv)
    return transform, matched


def make_alignment_figure(coords1, coords2, matched, fname1, fname2, radius=1):
    """
    Creates and returns a simple labelled figure showing the locations of
    sources from two catalogues in the "static" plane, so the user can
    assess how well the alignment has worked. Matched sources are indicated
    by black circles, unmatched sources by blue circles.

    Note: this plot is created to include the initial alignment transform,
    but *not* the more complex polynomial transform. Therefore misalignments
    should get resolved by the polynomial and are nothing to worry about.

    Parameters
    ----------
    coords1: 2-tuple
        first set of coords (to be plotted in black)
    coords2: 2-tuple
        second set of coords (to be plotted in red)
    matched: sequence
        sources in coords2 which have been matched
    fname1: str
        filename (or other identifier) of first coords
    fname2: str
        filename (or other identifier) of second coords
    radius: float
        matching radius (arcsec): used for plotting circles

    Returns
    -------
    Figure object
    """
    def trim_filename(fname):
        return re.sub("_(.*)\.(.*)", "", fname)

    fig, ax = plt.subplots()
    ax.set_title(f"{trim_filename(fname1)}[open] / "
                 f"{trim_filename(fname2)}")
    ax.set_xlabel("X_STATIC (arcsec)")
    ax.set_ylabel("Y_STATIC (arcsec)")
    ax.set_xlim(-45, 45)
    ax.set_ylim(-45, 45)
    ax.set_aspect('equal')
    ax.grid(color='0.9', zorder=0)
    ax.plot(*coords2, 'r.', zorder=10)
    for x, y, m in zip(*coords1, matched):
        circle = plt.Circle((x, y), radius=radius, linewidth=1, fill=None,
                            edgecolor='k' if m > -1 else 'b', zorder=20)
        ax.add_patch(circle)
    fig.tight_layout()
    return fig
