#
#                                                                  gemini_python
#
#                                                      primitives_gsaoi_image.py
# ------------------------------------------------------------------------------
import math
from functools import reduce
import numpy as np

from astropy.modeling import models, fitting, Model
from astropy.stats import sigma_clip
from astropy import table
from astropy import units as u

from gwcs.wcs import WCS as gWCS
from gwcs import coordinate_frames as cf

from astrodata import wcs as adwcs

from gempy.gemini import gemini_tools as gt
from gempy.library import astromodels as am, astrotools as at
from gempy.library.matching import fit_model, match_sources
from gempy.gemini import qap_tools as qap

from geminidr.core import Image, Photometry
from recipe_system.utils.decorators import parameter_override

from .primitives_gsaoi import GSAOI
from . import parameters_gsaoi_image
from .lookups import gsaoi_static_distortion_info as gsdi


@parameter_override
class GSAOIImage(GSAOI, Image, Photometry):
    """
    This is the class containing all of the preprocessing primitives
    for the F2Image level of the type hierarchy tree. It inherits all
    the primitives from the level above
    """
    tagset = {"GEMINI", "GSAOI", "IMAGE"}

    def __init__(self, adinputs, **kwargs):
        super().__init__(adinputs, **kwargs)
        self._param_update(parameters_gsaoi_image)

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

        self._attach_static_distortion(adinputs)

        for ad in adinputs:
            # Check we have a REFCAT and at least one OBJCAT to match
            try:
                refcat = ad.REFCAT
            except AttributeError:
                log.warning("No REFCAT in {} - cannot calculate astrometry".
                            format(ad.filename))
                continue
            if not ('RAJ2000' in refcat.colnames and 'DEJ2000' in refcat.colnames):
                log.warning(f'REFCAT in {ad.filename} is missing RAJ2000 '
                            'and/or DEJ2000 columns - cannot calculate astrometry')
                continue
            if not any(hasattr(ext, 'OBJCAT') for ext in ad):
                log.warning("No OBJCATs in {} - cannot match to REFCAT".
                            format(ad.filename))
                continue

            if not all([isinstance(getattr(ext.wcs, 'output_frame'),
                                   cf.CelestialFrame) for ext in ad]):
                log.warning("Missing CelestialFrame in at least one "
                            f"extension of {ad.filename}")
                continue

            # We're going to fit in the static-corrected coordinate frame
            # Find the its boundaries so we can cull the REFCAT
            static_transforms = [ext.wcs.get_transform(ext.wcs.input_frame,
                                                       "static") for ext in ad]
            all_corners = np.concatenate([np.array(static(*np.array(at.get_corners(ext.shape)).T))
                                          for ext, static in zip(ad, static_transforms)], axis=1)
            xmin, ymin = all_corners.min(axis=1) - initial
            xmax, ymax = all_corners.max(axis=1) + initial

            # This will be the same for all extensions if the user hasn't
            # hacked it (which is something we can't really check)
            backward_transform = ad[0].wcs.get_transform(ad[0].wcs.output_frame,
                                                         "static")
            xref, yref = backward_transform(refcat['RAJ2000'],
                                            refcat['DEJ2000'])
            in_field = np.all((xref > xmin, xref < xmax,
                               yref > ymin, yref < ymax), axis=0)
            num_ref_sources = in_field.sum()
            if num_ref_sources == 0:
                log.stdinfo(f"No REFCAT sources in field of {ad.filename}")
                continue

            m_init = (models.Shift(0, bounds={'offset': (-initial, initial)}) &
                      models.Shift(0, bounds={'offset': (-initial, initial)}))
            if rotate:
                m_init = am.Rotate2D(0, bounds={'angle': (-5, 5)}) | m_init
            if scale:
                m_init = am.Scale2D(1, bounds={'factor': (0.95, 1.05)}) | m_init

            objcats = []
            for ext, static in zip(ad, static_transforms):
                try:
                    objcat = ext.OBJCAT.copy()
                except AttributeError:
                    continue
                objcat['EXTVER'] = ext.hdr['EXTVER']
                objcat['X_STATIC'], objcat['Y_STATIC'] = static(objcat['X_IMAGE'] - 1,
                                                                objcat['Y_IMAGE'] - 1)
                objcats.append(objcat)
            objcat = table.vstack(objcats, metadata_conflicts='silent')
            objcat_len = len(objcat)

            # How many objects do we want to try to match? Keep brightest ones only
            if objcat_len > 2 * num_ref_sources:
                keep_num = max(2 * num_ref_sources, min(10, objcat_len))
            else:
                keep_num = objcat_len
            sorted_idx = np.argsort(objcat['MAG_AUTO'])[:keep_num]

            log.stdinfo(f"Aligning {ad.filename} with {num_ref_sources} REFCAT"
                        f" and {objcat_len} OBJCAT sources")
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
            log.stdinfo(f"Matched {num_matched} objects")

            if num_matched > 0:
                # Shape doesn't matter here since the transform is perfectly affine
                # We use this function for ease of determining polynomial
                # coefficients irrespective of whether we have a rotation/scaling
                affine = adwcs.calculate_affine_matrices(transform, shape=(100, 100))
                xmodel = models.Polynomial2D(degree=order, c0_0=affine.offset[1],
                                             c1_0=affine.matrix[1,1],
                                             c0_1=affine.matrix[1,0])
                ymodel = models.Polynomial2D(degree=order, c0_0=affine.offset[0],
                                             c1_0=affine.matrix[0,1],
                                             c0_1=affine.matrix[0,0])

                fit_it = fitting.FittingWithOutlierRemoval(fitting.LinearLSQFitter(),
                                                           sigma_clip, sigma=3)
                niter = 0
                while True:
                    old_num_matched = num_matched
                    xobj_matched, yobj_matched = [], []
                    xref_matched, yref_matched = [], []
                    for i, m in enumerate(matched):
                        if m >= 0:
                            xref_matched.append(xref[i])
                            yref_matched.append(yref[i])
                            xobj_matched.append(objcat['X_STATIC'][m])
                            yobj_matched.append(objcat['Y_STATIC'][m])
                    xmodel, _ = fit_it(xmodel, np.array(xobj_matched), np.array(yobj_matched), xref_matched)
                    ymodel, _ = fit_it(ymodel, np.array(xobj_matched), np.array(yobj_matched), yref_matched)
                    transform = models.Mapping((0,1,0,1)) | (xmodel & ymodel)
                    matched = match_sources((xref, yref),
                                            transform(objcat['X_STATIC'], objcat['Y_STATIC']),
                                            radius=final)
                    num_matched = np.sum(matched >= 0)
                    log.stdinfo(f"Matched {num_matched} objects")
                    if num_matched == old_num_matched or niter > max_iters:
                        break
                    niter += 1
                xmodel_inv, _ = fit_it(xmodel, np.array(xref_matched), np.array(yref_matched), xobj_matched)
                ymodel_inv, _ = fit_it(ymodel, np.array(xref_matched), np.array(yref_matched), yobj_matched)
                transform.inverse = models.Mapping((0,1,0,1)) | (xmodel_inv & ymodel_inv)

                # Associate REFCAT properties with their OBJCAT
                # counterparts. Remember! matched is the reference
                # (OBJCAT) source for the input (REFCAT) source
                dra, ddec = [], []
                cospa, sinpa = math.cos(ad.phu['PA']), math.sin(ad.phu['PA'])
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
                        dx = xref[i] - objcat['X_STATIC'][m]
                        dy = yref[i] - objcat['Y_STATIC'][m]
                        dra.append(dx * cospa - dy * sinpa)
                        ddec.append(dx * sinpa + dy * cospa)

                delta_ra = np.mean(dra)
                delta_dec = np.mean(ddec)
                dra_std = np.std(dra)
                ddec_std = np.std(ddec)
                log.fullinfo(f"WCS Updated for {ad.filename}. Astrometric "
                             "offset is:")
                log.fullinfo("RA: {:6.2f} +/- {:.2f} arcsec".
                             format(delta_ra, dra_std))
                log.fullinfo("Dec:{:6.2f} +/- {:.2f} arcsec".
                             format(delta_dec, ddec_std))
                info_list = [{"dra": delta_ra, "dra_std": dra_std,
                              "ddec": delta_dec, "ddec_std": ddec_std,
                              "nsamples": int(num_matched)}]
                # Report the measurement to the fitsstore
                if self.upload and "metrics" in self.upload:
                    qap.fitsstore_report(ad, "pe", info_list, self.calurl_dict,
                                         self.mode, upload=True)

                # Update OBJCAT (X_WORLD, Y_WORLD)
                for ext in ad:
                    # TODO: use insert_frame method
                    var_frame = cf.Frame2D(unit=(u.arcsec, u.arcsec), name="variable")
                    ext.wcs = gWCS(ext.wcs.pipeline[:1] +
                                   [(ext.wcs.pipeline[1].frame, transform),
                                    (var_frame, ext.wcs.pipeline[1].transform)] +
                                   ext.wcs.pipeline[2:])
                    ext.objcat = objcat[objcat['EXTVER'] == ext.hdr['EXTVER']]
                    ext.objcat['X_WORLD'], ext.objcat['Y_WORLD'] = ext.wcs(ext.objcat['X_IMAGE']-1,
                                                                           ext.objcat['Y_IMAGE']-1)
                    ext.objcat.remove_columns(['EXTVER', 'X_STATIC', 'Y_STATIC'])
            else:
                log.stdinfo("Could not determine astrometric offset for "
                            f"{ad.filename}")

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
                sky_model = models.Scale(1 / 3600) & models.Scale(1 / 3600)
                pa = ad.phu['PA']
                if abs(pa) > 0.01:
                    sky_model |= models.Rotation2D(-pa)
                sky_model |= models.Pix2Sky_TAN() | models.RotateNative2Celestial(ra, dec, 180)
                ext.wcs = gWCS([(ext.wcs.input_frame, sdmodel),
                                (static_frame, sky_model),
                                (ext.wcs.output_frame, None)])

        return adinputs