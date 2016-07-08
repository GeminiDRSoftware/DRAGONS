from astrodata import astro_data_tag
from ..gemini import AstroDataGemini
import re

class AstroDataGpi(AstroDataGemini):
    @staticmethod
    def _matches_data(data_provider):
        return data_provider.phu.get('INSTRUME').upper() == 'GPI'

    @astro_data_tag
    def _tag_instrument(self):
        return (set(['GPI']), ())

    @astro_data_tag
    def _tag_pol(self):
        if self.phu.get('DISPERSR', '').startswith('DISP_WOLLASTON'):
            return (set(['POL']), ())

    @astro_data_tag
    def _tag_spect(self):
        if self.phu.get('DISPERSR', '').startswith('DISP_PRISM'):
            return (set(['SPECT', 'IFU']), ())
