from astrodata import astro_data_tag
from ..gemini import AstroDataGemini
import re

class AstroDataGsaoi(AstroDataGemini):
    @staticmethod
    def _matches_data(data_provider):
        return data_provider.phu.get('INSTRUME').upper() == 'GSAOI'

    @astro_data_tag
    def _tag_instrument(self):
        return (set(['GSAOI']), ())

    @astro_data_tag
    def _tag_dark(self):
        if self.phu.get('OBSTYPE') == 'DARK':
            return (set(['DARK', 'CAL']), ())

    @astro_data_tag
    def _tag_image(self):
        tags = ['IMAGE']
        if self.phu.get('OBSTYPE') == 'FLAT':
            tags.extend(['FLAT', 'CAL'])
        if 'DOMEFLAT' in self.phu.get('OBJECT', '').upper():
            tags.extend(['DOMEFLAT', 'FLAT', 'CAL'])
        elif 'TWILIGHT' in self.phu.get('OBJECT', '').upper():
            tags.extend(['TWILIGHT', 'FLAT', 'CAL'])

        return (set(tags), ())

    # Kept separate from _tag_image, because some conditions defined
    # at a higher level conflict with this
    @astro_data_tag
    def _type_gcal_lamp(self):
        obj = self.phu.get('OBJECT', '').upper()
        if obj == 'DOMEFLAT':
            return (set(['LAMPON']), ())
        elif obj == 'DOMEFLAT OFF':
            return (set(['LAMPOFF']), ())

