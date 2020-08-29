#
#                                                                  gemini_python
#
#                                                         primitives_resample.py
# ------------------------------------------------------------------------------
import numpy as np
from astropy.wcs import WCS
from astropy.modeling import models, Model
from gwcs.wcs import WCS as gWCS

from functools import reduce
from copy import copy

from gempy.library import transform
from gempy.library.matching import Pix2Sky
from gempy.gemini import gemini_tools as gt

from geminidr.gemini.lookups import DQ_definitions as DQ

from geminidr import PrimitivesBASE
from . import parameters_resample

from recipe_system.utils.decorators import parameter_override
# ------------------------------------------------------------------------------
interpolators = {"nearest": 0,
                 "linear": 1,
                 "spline2": 2,
                 "spline3": 3,
                 "spline4": 4,
                 "spline5": 5,
                 }
# ------------------------------------------------------------------------------
@parameter_override
class Resample(PrimitivesBASE):
    """
    This is the class containing all of the primitives for resampling.
    """
    tagset = None

    def __init__(self, adinputs, **kwargs):
        super().__init__(adinputs, **kwargs)
        self._param_update(parameters_resample)

    def resampleToCommonFrame(self, adinputs=None, **params):
        """
        This primitive applies the transformation encoded in the input images
        WCSs to align them with a reference image, in reference image pixel
        coordinates. The reference image is taken to be the first image in
        the input list.

        By default, the transformation into the reference frame is done via
        interpolation. The variance plane, if present, is transformed in
        the same way as the science data.

        The data quality plane, if present, is handled in a bitwise manner
        with each bit of each pixel in the output image being set it it has
        >1% influence from that bit of a bad pixel. The transformed masks are
        then added back together to generate the transformed DQ plane.

        The WCS keywords in the headers of the output images are updated
        to reflect the transformation.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
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
        order = params["order"]
        trim_data = params["trim_data"]
        clean_data = params["clean_data"]
        sfx = params["suffix"]

        if len(adinputs) < 2:
            log.warning("No alignment will be performed, since at least two "
                        "input AstroData objects are required for "
                        "resampleToCommonFrame")
            return adinputs

        # TODO: Can we make it so that we don't need to mosaic detectors
        # before doing this? That would mean we only do one interpolation,
        # not two, and that's definitely better!
        if not all(len(ad)==1 for ad in adinputs):
            raise OSError("All input images must have only one extension.")

        attributes = [attr for attr in ('data', 'mask', 'variance', 'OBJMASK')
                      if all(hasattr(ad[0], attr) for ad in adinputs)]

        # Clean data en masse (improves logging by showing a single call)
        if clean_data:
            adinputs = self.applyDQPlane(adinputs, replace_flags=DQ.not_signal ^ DQ.no_data,
                                   replace_value="median", inner=3, outer=5)

        ad_ref = adinputs[0]
        ref_wcs = WCS(ad_ref[0].hdr)
        ndim = len(ad_ref[0].shape)

        #for ad in adinputs[1:]:
        #    print(ad.filename)
        #    print(WCS(ad[0].hdr))

        # No transform for the reference AD
        transforms = [transform.Transform()] + [transform.Transform([Pix2Sky(WCS(ad[0].hdr)),
                                                 Pix2Sky(ref_wcs).inverse])
                                      for ad in adinputs[1:]]
        for t in transforms:
            t._affine = True  # We'll perform an approximation

        # Compute output frame. If we're trimming data, this is the frame of
        # the reference image; otherwise we have to calculate it as the
        # smallest rectangle (in 2D) including all transformed inputs
        if trim_data:
            log.fullinfo("Trimming data to size of reference image")
            output_shape = ad_ref[0].shape
            origin = (0,) * ndim
        else:
            log.fullinfo("Output image will contain all transformed images")
            # Calculate corners of all transformed images so we can compute
            # the size and relative location of output image. We do this by
            # making an AstroDataGroup containing all of them, although we're
            # not going to process it.
            adg = transform.AstroDataGroup(adinputs, transforms)
            adg.calculate_output_shape()
            output_shape = adg.output_shape
            origin = adg.origin
        log.stdinfo("Output image will have shape "+repr(output_shape[::-1]))

        adoutputs = []
        for ad, t in zip(adinputs, transforms):
            #print(ad.filename+" "+repr(transform.affine_matrices(shape=ad[0].shape)))
            adg = transform.AstroDataGroup(ad, [t])
            # Set the output shape and origin to keep coordinate frame
            adg.output_shape = output_shape
            adg.origin = origin
            ad_out = adg.transform(attributes=attributes, order=order)

            # Add a bunch of header keywords describing the transformed shape
            corners = adg.corners[0]
            ad_out[0].hdr["AREATYPE"] = ("P{}".format(len(corners)),
                                         "Region with {} vertices".format(len(corners)))
            for i, corner in enumerate(zip(*corners), start=1):
                for axis, value in enumerate(corner, start=1):
                    key_name = "AREA{}_{}".format(i, axis)
                    key_comment = "Vertex {}, dimension {}".format(i, axis)
                    ad_out[0].hdr[key_name] = (value+1, key_comment)

            # Timestamp and update filename
            ad_out.update_filename(suffix=sfx, strip=True)
            log.fullinfo("{} resampled with jfactor {:.2f}".
                         format(ad.filename, adg.jfactors[0]))
            adoutputs.append(ad_out)

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
        shifts_param = params.pop("shifts")
        sfx = params.pop("suffix")

        # TODO: Maybe remove this requirement
        if not all(len(ad) == 1 for ad in adinputs):
            raise OSError("All input images must have only one extension.")

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
        log.debug("Shifts are: "+repr(shifts))

        num_shifts, num_images = len(shifts), len(adinputs)
        if num_shifts == 1:
            shifts *= num_images
        elif num_shifts != num_images:
            raise ValueError(f"Number of shifts ({num_shifts}) incompatible "
                             f"with number of images ({num_images})")

        for ad, shift in zip(adinputs, shifts):
            if len(shift) != len(ad[0].shape):
                raise ValueError(f"Shift {shift} incompatible with "
                                 f"dimensionality of {ad.filename}")
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

        adoutputs = self._resample_to_new_frame(adinputs, frame="shifted", **params)
        for ad in adoutputs:
            ad.update_filename(suffix=sfx, strip=True)

        return adoutputs

    def _resample_to_new_frame(self, adinputs=None, frame=None, order=3,
                               trim_data=False, clean_data=False):
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
        trim_data: bool
            trim image to size of reference image?
        clean_data: bool
            replace bad pixels with a ring median of their values to avoid
            ringing if using a high-order interpolation?
        """

        if clean_data:
            self.applyDQPlane(adinputs, replace_flags=DQ.not_signal ^ DQ.no_data,
                              replace_value="median", inner=3, outer=5)

        if trim_data:
            output_shape = adinputs[0][0].shape
            origin = (0,) * len(output_shape)
        else:
            all_corners = np.concatenate([transform.get_output_corners(
                ad[0].wcs.get_transform(ad[0].wcs.input_frame, frame),
                input_shape=ad[0].shape) for ad in adinputs], axis=1)
            origin = tuple(np.ceil(min(corners)) for corners in all_corners)
            output_shape = tuple(int(np.floor(max(corners)) - np.ceil(min(corners)) + 1)
                                 for corners in all_corners)

        adoutputs = []
        for ad in adinputs:
            self.log.stdinfo(f"Resampling {ad.filename}")
            # We can process the OBJCAT because we're just shifting images
            ad_out = transform.resample_from_wcs(ad, frame, order=order,
                                                 output_shape=output_shape, origin=origin,
                                                 process_objcat=True)
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

        source_stream = self.streams.get(source, [])
        if len(source_stream) != 1:
            log.warning("Stream {} not found or does not contain single "
                        "AstroData object. Continuing.".format(source_stream))
            return adinputs

        ad_source = source_stream[0]
        # There's no reason why we can't handle multiple extensions
        if any(len(ad) != len(ad_source) for ad in adinputs):
            log.warning("At least one AstroData input has a different number "
                        "of extensions to the reference. Continuing.")
            return adinputs

        for ad in adinputs:
            for ext, source_ext in zip(ad, ad_source):
                t = transform.Transform([Pix2Sky(WCS(source_ext.hdr)),
                                         Pix2Sky(WCS(ext.hdr)).inverse])
                t._affine = True
                try:
                    # Convert OBJMASK to float or else uint8 will be returned
                    objmask = t.apply(source_ext.OBJMASK.astype(np.float32),
                                      output_shape=ext.shape, order=order, cval=0)
                    ext.OBJMASK = np.where(abs(objmask) > threshold, 1, 0).astype(np.uint8)
                except:  # source_ext.OBJMASK not present, or None
                    pass
                # We will deliberately keep the input image's OBJCAT (if it
                # exists) since this will be required for aligning the inputs.
            ad.update_filename(suffix=sfx, strip=True)

        return adinputs
