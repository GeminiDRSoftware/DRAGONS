#
#                                                                  gemini_python
#
#                                                         primitives_resample.py
# ------------------------------------------------------------------------------
import numpy as np
from astropy.wcs import WCS
from scipy.ndimage import affine_transform

from gempy.library import astrotools as at
from gempy.library.transform import Transform, AstroDataGroup
from gempy.library.matching import Pix2Sky
from gempy.gemini import gemini_tools as gt
from gempy.utils import logutils

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
        super(Resample, self).__init__(adinputs, **kwargs)
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
            raise IOError("All input images must have only one extension.")

        attributes = [attr for attr in ('data', 'mask', 'variance', 'OBJMASK')
                      if all(hasattr(ad, attr) for ad in adinputs)]

        # Clean data en masse (improves logging by showing a single call)
        if clean_data:
            adinputs = self.applyDQPlane(adinputs, replace_flags=65535 ^ (DQ.non_linear | DQ.saturated | DQ.no_data),
                                   replace_value="median", inner_radius=3, outer_radius=5)[0]

        ref_ad = adinputs[0]
        ref_wcs = WCS(ref_ad[0].hdr)
        ndim = len(ref_ad[0].shape)

        for ad in adinputs[1:]:
            print(ad.filename)
            print(WCS(ad[0].hdr))

        # No transform for the reference AD
        transforms = [Transform()] + [Transform([Pix2Sky(WCS(ad[0].hdr)),
                                                 Pix2Sky(ref_wcs).inverse])
                                      for ad in adinputs[1:]]
        for t in transforms:
            t._affine = True  # We'll perform an approximation

        # Compute output frame. If we're trimming data, this is the frame of
        # the reference image; otherwise we have to calculate it as the
        # smallest rectangle (in 2D) including all transformed inputs
        if trim_data:
            log.fullinfo("Trimming data to size of reference image")
            output_shape = ref_ad[0].shape
            origin = (0,) * ndim
        else:
            log.fullinfo("Output image will contain all transformed images")
            # Calculate corners of all transformed images so we can compute
            # the size and relative location of output image. We do this by
            # making an AstroDataGroup containing all of them, although we're
            # not going to process it.
            adg = AstroDataGroup(adinputs, transforms)
            adg.calculate_output_shape()
            output_shape = adg.output_shape
            origin = adg.origin
        log.stdinfo("Output image will have shape "+repr(output_shape[::-1]))

        adoutputs = []
        for ad, transform in zip(adinputs, transforms):
            print(ad.filename+" "+repr(transform.affine_matrices(shape=ad[0].shape)))
            adg = AstroDataGroup(ad, [transform])
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