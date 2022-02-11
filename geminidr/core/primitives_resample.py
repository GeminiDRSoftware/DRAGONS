#
#                                                                  gemini_python
#
#                                                         primitives_resample.py
# ------------------------------------------------------------------------------
import numpy as np
from functools import reduce
from copy import copy

from scipy.ndimage import affine_transform
from astropy.modeling import models, Model
from astropy import units as u
from astropy.coordinates import SkyCoord
from gwcs.wcs import WCS as gWCS

import astrodata, gemini_instruments
from astrodata import wcs as adwcs

from gempy.library import transform
from gempy.gemini import gemini_tools as gt

from geminidr.gemini.lookups import DQ_definitions as DQ

from geminidr import PrimitivesBASE
from . import parameters_resample

from recipe_system.utils.decorators import parameter_override, capture_provenance


# ------------------------------------------------------------------------------
@parameter_override
@capture_provenance
class Resample(PrimitivesBASE):
    """
    This is the class containing all of the primitives for resampling images.
    """
    tagset = None

    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_resample)

    def resampleToCommonFrame(self, adinputs=None, **params):
        """
        This primitive applies the transformation encoded in the input images
        WCSs to align them with a reference image, in reference image pixel
        coordinates. The reference image is taken to be the first image in
        the input list if not explicitly provided as a parameter.

        By default, the transformation into the reference frame is done via
        interpolation. The variance plane, if present, is transformed in
        the same way as the science data.

        The data quality plane, if present, is handled in a bitwise manner
        with each bit of each pixel in the output image being set it it has
        >1% influence from that bit of a bad pixel. The transformed masks are
        then added back together to generate the transformed DQ plane.

        The WCS objects of the output images are updated to reflect the
        transformation.

        Parameters
        ----------
        suffix : str
            suffix to be added to output files
        order : int (0-5)
            order of interpolation (0=nearest, 1=linear, etc.)
        trim_data : bool
            trim image to size of reference image?
        clean_data : bool
            replace bad pixels with a ring median of their values to avoid
            ringing if using a high-order interpolation?
        conserve : bool
            conserve flux when resampling to a different pixel scale?
        force_affine : bool
            convert the true resampling transformation to an affine
            approximation? This speeds up the calculation and has a negligible
            effect for instruments lacking significant distortion
        reference : str/AstroData/None
            reference image for resampling (if not provided, the first image
            in the list will be used)
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params.pop("suffix")
        reference = params.pop("reference")
        trim_data = params.pop("trim_data")
        force_affine = params.pop("force_affine")
        # These two parameters are only for GSAOI and will help to define
        # the output WCS if there's no reference image
        pixel_scale = params.pop("pixel_scale", None)
        position_angle = params.pop("pa", None)

        # TODO: Can we make it so that we don't need to mosaic detectors
        # before doing this? That would mean we only do one interpolation,
        # not two, and that's definitely better!
        if not all(len(ad) == 1 or ad.instrument() == "GSAOI" for ad in adinputs):
            raise OSError("All input images must have only one extension.")

        if isinstance(reference, str):
            reference = astrodata.open(reference)
        elif reference is None and pixel_scale is None:
            # Reference image will be the first AD, so we need 2+
            if len(adinputs) < 2:
                log.warning("No alignment will be performed, since at least "
                            "two input AstroData objects are required for "
                            "resampleToCommonFrame")
                return adinputs

        if reference is None and pixel_scale:
            # This must be GSAOI projecting to the requested geometry
            ad0 = adinputs[0]
            ra, dec = ad0.target_ra(), ad0.target_dec()
            # using SkyCoord facilitates formatting the log
            center = SkyCoord(ra * u.deg, dec * u.deg)
            ra_str = center.ra.to_string(u.hour, precision=3)
            dec_str = center.dec.to_string(u.deg, precision=2, alwayssign=True)
            log.stdinfo(f"Projecting with center {ra_str} {dec_str}\n"
                        f"at PA={position_angle} with pixel scale={pixel_scale}")
            pixel_scale /= 3600
            new_wcs = (models.Scale(-pixel_scale) & models.Scale(pixel_scale) |
                       models.Rotation2D(position_angle) |
                       models.Pix2Sky_TAN() |
                       models.RotateNative2Celestial(ra, dec, 180))
            ref_wcs = gWCS([(ad0[0].wcs.input_frame, new_wcs),
                            (ad0[0].wcs.output_frame, None)])
            if trim_data:
                log.warning("Setting trim_data=False as required when no "
                            "reference imagevis provided.")
                trim_data = False
        else:
            if reference is None:
                reference = adinputs[0]
            else:
                log.stdinfo(f"Using {reference.filename} as reference image")
                if not trim_data:
                    log.warning("Setting trim_data=True to trim to size of the "
                                "reference image.")
                    trim_data = True
            if len(reference) != 1:
                raise OSError("Reference image must have only one extension.")
            ref_wcs = reference[0].wcs

        if trim_data:
            params.update({'origin': (0,) * len(reference[0].shape),
                           'output_shape': reference[0].shape})

        # No transform for the reference AD
        for ad in adinputs:
            transforms = []
            if reference is ad:
                transforms.append(models.Identity(len(ad[0].shape)))
            else:
                for ext in ad:
                    t_align = ext.wcs.forward_transform | ref_wcs.backward_transform
                    if force_affine:
                        affine = adwcs.calculate_affine_matrices(t_align, ext.shape)
                        t_align = models.AffineTransformation2D(matrix=affine.matrix[::-1, ::-1],
                                                                translation=affine.offset[::-1])
                    transforms.append(t_align)

            for ext, t_align in zip(ad, transforms):
                resampled_frame = copy(ext.wcs.input_frame)
                resampled_frame.name = "resampled"
                ext.wcs = gWCS([(ext.wcs.input_frame, t_align),
                                (resampled_frame, ref_wcs.pipeline[0].transform)] +
                                 ref_wcs.pipeline[1:])

        adoutputs = self._resample_to_new_frame(adinputs, frame="resampled",
                                                process_objcat=False, **params)
        for ad in adoutputs:
            try:
                trans_data = ad.nddata[0].meta.pop('transform')
            except KeyError:
                pass
            else:
                corners = np.array(trans_data['corners'][0])
                ncorners = len(corners)
                ad.hdr["AREATYPE"] = (f"P{ncorners}",
                                      f"Region with {ncorners} vertices")
                for i, corner in enumerate(zip(*corners), start=1):
                    for axis, value in enumerate(reversed(corner), start=1):
                        key_name = f"AREA{i}_{axis}"
                        key_comment = f"Vertex {i}, dimension {axis}"
                        ad.hdr[key_name] = (value + 1, key_comment)
                jfactor = trans_data['jfactors'][0]
                ad.hdr["JFACTOR"] = (jfactor, "J-factor in resampling")

            ad.update_filename(suffix=sfx, strip=True)
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)

        return adoutputs

    def shiftImages(self, adinputs=None, **params):
        """
        This primitive will shift images by user-defined amounts along each
        axis.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        shifts: str
            either: (a) list of colon-separated xshift,yshift pairs, or
                    (b) filename containing shifts, one set per image
        order: int (0-5)
            order of interpolation (0=nearest, 1=linear, etc.)
        trim_data: bool
            trim image to size of reference image?
        clean_data: bool
            replace bad pixels with a ring median of their values to avoid
            ringing if using a high-order interpolation?
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        # pop the params so we can pass the rest of the dict to helper method
        shifts_param = params.pop("shifts")
        trim_data = params.pop("trim_data")
        sfx = params.pop("suffix")

        # TODO: Maybe remove this requirement
        if not all(len(ad) == 1 for ad in adinputs):
            raise OSError("All input images must have only one extension.")

        # Ill-defined behaviour for this situation so
        if len(adinputs) == 1 and not trim_data:
            log.warning("Setting trim_data=True since there is only one input frame")
            trim_data = True

        # Check we can get some numerical shifts and that the number is
        # compatible with the number of images
        try:
            shifts = [tuple(float(x) for x in s.split(','))
                      for s in shifts_param.split(":")]
        except (TypeError, ValueError):
            try:
                f = open(shifts_param)
            except OSError:
                raise ValueError("Cannot parse parameter 'shifts' as either a "
                                 "list of shifts or a filename.")
            else:
                try:
                    shifts = [tuple(float(x) for x in line.strip().split(' '))
                              for line in f.readlines()]
                except (TypeError, ValueError):
                    raise ValueError(f"Cannot parse shifts from file {shifts_param}")

        num_shifts, num_images = len(shifts), len(adinputs)
        if num_shifts == 1:
            shifts *= num_images
        elif num_shifts != num_images:
            raise ValueError(f"Number of shifts ({num_shifts}) incompatible "
                             f"with number of images ({num_images})")

        if trim_data:
            reference = adinputs[0]
            params.update({'origin': (0,) * len(reference[0].shape),
                           'output_shape': reference[0].shape})

        for ad, shift in zip(adinputs, shifts):
            if len(shift) != len(ad[0].shape):
                raise ValueError(f"Shift {shift} incompatible with "
                                 f"dimensionality of {ad.filename}")
            log.debug(f"Shift for {ad.filename} is {shift}")
            shift_model = reduce(Model.__and__,
                                 [models.Shift(offset) for offset in shift])
            wcs = ad[0].wcs
            shifted_frame = copy(wcs.input_frame)
            shifted_frame.name = "shifted"
            # TODO: use insert_frame method
            ad[0].wcs = gWCS([(wcs.input_frame, shift_model),
                              (shifted_frame, wcs.pipeline[0].transform)] +
                               wcs.pipeline[1:])
            ad[0].wcs.insert_transform(shifted_frame, shift_model.inverse,
                                       after=True)

        adoutputs = self._resample_to_new_frame(adinputs, frame="shifted",
                                                process_objcat=True, **params)
        for ad in adoutputs:
            ad.update_filename(suffix=sfx, strip=True)

        return adoutputs

    def _resample_to_new_frame(self, adinputs=None, frame=None, order=3,
                               conserve=True, output_shape=None, origin=None,
                               clean_data=False, process_objcat=False):
        """
        This private method resamples a number of AstroData objects to a
        CoordinateFrame they share. It is basically just a wrapper for the
        transform.resample_from_wcs() method that creates an
        appropriately-sized output based on the complete set of input
        AstroData objects.

        Parameters
        ----------
        frame: str
            name of CoordinateFrame to be resampled to
        order: int (0-5)
            order of interpolation (0=nearest, 1=linear, etc.)
        output_shape : tuple/None
            shape of output image (if None, calculate and use shape that
            contains all resampled inputs)
        origin: tuple/None
            location of origin in reampled output (i.e., data to the left of
            or below this will be cut)
        clean_data : bool
            replace bad pixels with a ring median of their values to avoid
            ringing if using a high-order interpolation?
        process_objcat : bool
            update (rather than delete) the OBJCAT?
        """
        log = self.log

        if clean_data:
            self.applyDQPlane(adinputs, replace_flags=DQ.not_signal ^ DQ.no_data,
                              replace_value="median", inner=3, outer=5)

        if output_shape is None or origin is None:
            all_corners = np.concatenate([transform.get_output_corners(
                ext.wcs.get_transform(ext.wcs.input_frame, frame),
                input_shape=ext.shape) for ad in adinputs for ext in ad], axis=1)
            if origin is None:
                origin = tuple(np.ceil(min(corners)) for corners in all_corners)
            if output_shape is None:
                output_shape = tuple(int(np.floor(max(corners)) - np.ceil(min(corners)) + 1)
                                     for corners in all_corners)

        log.stdinfo("Output image will have shape "+repr(output_shape[::-1]))
        adoutputs = []
        for ad in adinputs:
            log.stdinfo(f"Resampling {ad.filename}")
            ad_out = transform.resample_from_wcs(ad, frame, order=order, conserve=conserve,
                                                 output_shape=output_shape, origin=origin,
                                                 process_objcat=process_objcat)
            ad_out.filename = ad.filename
            adoutputs.append(ad_out)

        return adoutputs

    def applyStackedObjectMask(self, adinputs=None, **params):
        """
        This primitive takes an image with an OBJMASK and transforms that
        OBJMASK onto the pixel planes of the input images, using their WCS
        information. If the first image is a stack, this allows us to mask
        fainter objects than can be detected in the individual input images.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        source: str
            name of stream containing single stacked image
        order: int (0-5)
            order of interpolation
        threshold: float
            threshold above which an interpolated pixel should be flagged
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        source = params["source"]
        order = params["order"]
        threshold = params["threshold"]
        sfx = params["suffix"]
        force_affine = True

        try:
            source_stream = self.streams[source]
        except KeyError:
            try:
                ad_source = astrodata.open(source)
            except:
                log.warning(f"Cannot find stream or file named {source}. Continuing.")
                return adinputs
        else:
            if len(source_stream) != 1:
                log.warning(f"Stream {source} does not contain single "
                            "AstroData object. Continuing.")
                return adinputs
            ad_source = source_stream[0]

        # There's no reason why we can't handle multiple extensions
        if any(len(ad) != len(ad_source) for ad in adinputs):
            log.warning("At least one AstroData input has a different number "
                        "of extensions to the reference. Continuing.")
            return adinputs

        for ad in adinputs:
            for ext, source_ext in zip(ad, ad_source):
                if getattr(ext, 'OBJMASK') is not None:
                    t_align = source_ext.wcs.forward_transform | ext.wcs.backward_transform
                    if force_affine:
                        affine = adwcs.calculate_affine_matrices(t_align.inverse, ad[0].shape)
                        objmask = affine_transform(source_ext.OBJMASK.astype(np.float32),
                                                   affine.matrix, affine.offset,
                                                   output_shape=ext.shape, order=order,
                                                   cval=0)
                    else:
                        objmask = transform.Transform(t_align).apply(source_ext.OBJMASK.astype(np.float32),
                                                                     output_shape=ext.shape, order=order,
                                                                     cval=0)
                    ext.OBJMASK = np.where(abs(objmask) > threshold, 1, 0).astype(np.uint8)
                # We will deliberately keep the input image's OBJCAT (if it
                # exists) since this will be required for aligning the inputs.
            ad.update_filename(suffix=sfx, strip=True)

        return adinputs
