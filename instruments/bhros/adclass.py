from astrodata import astro_data_tag
from ..gemini import AstroDataGemini
import re

class AstroDataBhros(AstroDataGemini):
    @staticmethod
    def _matches_data(data_provider):
        return data_provider.phu.get('INSTRUME', '').upper() == 'BHROS'

    @astro_data_tag
    def _tag_instrument(self):
        return (set(['BHROS', 'SPECT']), ())
