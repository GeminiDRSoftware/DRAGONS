#
#
#
#                                                        primitives_longslit.py
# -----------------------------------------------------------------------------

from recipe_system.utils.decorators import (parameter_override,
                                            capture_provenance)

from geminidr.core import Spect
from . import parameters_longslit


@parameter_override
@capture_provenance
class Longslit(Spect):
    """This is the class containing primitives specifically for longslit data.
    It inherits all the primitives from the level above.

    Currently this is a placeholder for moving longspit-specific code into in
    the future.

    """
    tagset = {'GEMINI', 'SPECT', 'LS'}
    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_longslit)

    def _make_tab_labels(self, ad):
        """
        Create tab labels for longslit data.

        Parameters
        ----------
        ad : `~astrodata.AstroData`
            The AstroData object to be processed.

        Returns
        -------
        list
            A list of tab labels for the given AstroData object.
        """
        # If we haven't extracted spectra, then all we have are separate
        # extensions, so we use the default tab labels.
        if self.timestamp_keys['extractSpectra'] in ad.phu:
            return super()._make_tab_labels(ad)

        apertures = ad.hdr.get('APERTURE')
        if apertures.count(None) == len(ad):
            tab_labels = [f"Aperture {i+1}" for i in range(len(ad))]
        else:
            tab_labels = [f"Aperture {apertures[i]}" for i in range(len(ad))]
        return tab_labels
