from astrodata import astro_data_tag
from ..gemini import AstroDataGemini
import re

class AstroDataMichelle(AstroDataGemini):
    @staticmethod
    def _matches_data(data_provider):
        return data_provider.phu.get('INSTRUME').upper() == 'MICHELLE'

    @astro_data_tag
    def _tag_instrument(self):
        return (set(['MICHELLE']), ())

    @astro_data_tag
    def _tag_mode(self):
        camera = self.phu.get('CAMERA')
        if camera == 'imaging':
            return (set(['IMAGE']), ())
        elif camera == 'spectroscopy':
            return (set(['SPECT', 'LS']), ())
