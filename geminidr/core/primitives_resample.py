#
#                                                                  gemini_python
#
#                                                         primitives_resample.py
# ------------------------------------------------------------------------------
from functools import reduce
from copy import copy
import numpy as np

from astropy.modeling import models, Model
from gwcs.wcs import WCS as gWCS

import astrodata, gemini_instruments

from gempy.gemini import gemini_tools as gt
from gempy.library import transform

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
        dq_threshold : float
            The fraction of a pixel's contribution from a DQ-flagged pixel to
            be considered 'bad' and also flagged.
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
                    shifts = [tuple(float(x) for x in line.strip().split())
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
                               clean_data=False, process_objcat=False,
                               dq_threshold=0.001):
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
        dq_threshold : float
            The fraction of a pixel's contribution from a DQ-flagged pixel to
            be considered 'bad' and also flagged.
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
            subsample = int(max(abs(np.linalg.det(transform.Transform(ext.wcs.get_transform(
                frame, ext.wcs.input_frame)).affine_matrices().matrix)) for ext in ad) + 0.5)
            log.debug(f"{ad.filename}: Subsampling factor of {subsample}")
            ad_out = transform.resample_from_wcs(
                ad, frame, order=order, conserve=conserve,
                output_shape=output_shape, origin=origin,
                process_objcat=process_objcat, subsample=subsample,
                threshold=dq_threshold)
            ad_out.filename = ad.filename
            adoutputs.append(ad_out)

        return adoutputs
