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
    def _tag_disperser(self):
        disp = self.phu.get('DISPERSR', '')
        if disp.startswith('DISP_WOLLASTON'):
            return (set(['POL']), ())
        elif disp.startswith('DISP_PRISM'):
            return (set(['SPECT', 'IFU']), ())
