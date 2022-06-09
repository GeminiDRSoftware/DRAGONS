#
#                                                                 gemini_python
#
#                                                     primitives_f2_longslit.py
# -----------------------------------------------------------------------------

from . import parameters_f2_longslit
from .primitives_f2_spect import F2Spect

from recipe_system.utils.decorators import (parameter_override,
                                            capture_provenance)

# -----------------------------------------------------------------------------
@parameter_override
@capture_provenance
class F2Longslit(F2Spect):
    """This class contains all of the processing primitives for the F2Longslit
    level of the type hiearchy tree. It inherits all the primitives from the
    above level.
    """
    tagset = {'GEMINI', 'F2', 'SPECT', 'LS'}
    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_f2_longslit)

    def _fields_overlap(self, ad1, ad2, frac_FOV=1.0, max_perpendicular_offset=None):
        slit_length = (1300 if ad1.is_ao() else 1460) * ad1.pixel_scale()
        slit_width = int(ad1.focal_plane_mask()[0]) * ad1.pixel_scale()
        return super()._fields_overlap(
            ad1, ad2, frac_FOV=frac_FOV, slit_length=slit_length,
            slit_width=slit_width, max_perpendicular_offset=max_perpendicular_offset)