#
#                                                                  gemini_python
#
#                                                         primitives_f2_image.py
# ------------------------------------------------------------------------------
from gempy.gemini import gemini_tools as gt

from geminidr.core import Image, Photometry
from .primitives_f2 import F2
from . import parameters_f2_image

from recipe_system.utils.decorators import parameter_override, capture_provenance
from astropy.coordinates import SkyCoord

# ------------------------------------------------------------------------------
@parameter_override
@capture_provenance
class F2Image(F2, Image, Photometry):
    """
    This is the class containing all of the preprocessing primitives
    for the F2Image level of the type hierarchy tree. It inherits all
    the primitives from the level above
    """
    tagset = {"GEMINI", "F2", "IMAGE"}

    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_f2_image)

    def makeLampFlat(self, adinputs=None, **params):
        """
        This produces an appropriate stacked F2 imaging flat, based on
        the inputs, since one of two procedures must be followed.

        In the standard recipe, the inputs will have come from getList and
        so will all have the same filter and will all need the same recipe.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        suffix = params["suffix"]

        # Since this primitive needs a reference, it must no-op without any
        if not adinputs:
            return adinputs

        if adinputs[0].wavelength_band() in ('K',):
            # This is basically the generic makeLampFlat code, but altered to
            # distinguish between FLATs and DARKs, not LAMPONs and LAMPOFFs
            flat_list = self.selectFromInputs(adinputs, tags='FLAT')
            dark_list = self.selectFromInputs(adinputs, tags='DARK')
            stack_params = self._inherit_params(params, "stackFrames")
            if dark_list:
                self.showInputs(dark_list, purpose='darks')
                # stackDarks will check that all darks match each other,
                # here just check that the exptime of first one matches flats.
                if dark_list[0].exposure_time() != flat_list[0].exposure_time():
                    raise ValueError("Darks exposure time not equal to flats exposure time")
                dark_list = self.stackDarks(dark_list, **stack_params)
            self.showInputs(flat_list, purpose='flats')
            stack_params.update({'zero': False, 'scale': False})
            flat_list = self.stackFrames(flat_list, **stack_params)

            if flat_list and dark_list:
                log.fullinfo("Subtracting stacked dark from stacked flat")
                flat = flat_list[0]
                flat.subtract(dark_list[0])
                flat.update_filename(suffix=suffix, strip=True)
                return [flat]
            elif flat_list:
                log.fullinfo("Only had flats to stack. Calling darkCorrect.")
                flat_list = self.darkCorrect(flat_list, suffix=suffix,
                                             dark=None, do_cal='procmode')
                return flat_list
            else:
                return []

        else:
            log.stdinfo('Using standard makeLampFlat primitive to make flatfield')
            adinputs = super().makeLampFlat(adinputs, **params)

        return adinputs

    def _fields_overlap(self, ad1, ad2, frac_FOV=1.0):
        """
        Checks whether the fields of view of two F2 images overlap
        sufficiently to be considerd part of a single ExposureGroup.
        F2Image requires its own code since it has a circular FOV.

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
        # GeMS truncates the FOV to 2' with 0.09" pixels
        diameter = 1300 if ad1.is_ao() else 2048
        c1 = SkyCoord(*ad1[0].wcs(1024, 1024), unit='deg')
        c2 = SkyCoord(*ad2[0].wcs(1024, 1024), unit='deg')
        return c1.separation(c2).arcsec < frac_FOV * diameter * ad1.pixel_scale()
