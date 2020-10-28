#
#                                                                  gemini_python
#
#                                                      primitives_gsaoi_image.py
# ------------------------------------------------------------------------------
from functools import reduce
from astropy.modeling import models, Model
from astropy import units as u

from gwcs.wcs import WCS as gWCS
from gwcs.coordinate_frames import Frame2D

from gempy.gemini import gemini_tools as gt

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
        static distortion read from the lookup table.
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
            ref_wcs = ad[ref_location[0]].wcs
            ra, dec = ref_wcs(*ref_location[1:])
            for ext, arrsec, sdmodel in zip(ad, ad.array_section(), sdmodels):
                # Include ROI shift
                sdmodel = (models.Shift(arrsec.x1 + 1) &
                           models.Shift(arrsec.y1 + 1) | sdmodel)

                static_frame = Frame2D(unit=(u.arcsec, u.arcsec), name="static")
                sky_model = models.Scale(1 / 3600) & models.Scale(1 / 3600)
                pa = ad.phu['PA']
                if abs(pa) > 0.01:
                    sky_model |= models.Rotation2D(-pa)
                sky_model |= models.Pix2Sky_TAN() | models.RotateNative2Celestial(ra, dec, 180)
                ext.wcs = gWCS([(ext.wcs.input_frame, sdmodel),
                                (static_frame, sky_model),
                                (ext.wcs.output_frame, None)])

        return adinputs