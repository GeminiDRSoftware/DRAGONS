from astrodata import astro_data_tag, TagSet
from ..gemini import AstroDataGemini
import re

class AstroDataNiri(AstroDataGemini):
    @staticmethod
    def _matches_data(data_provider):
        return data_provider.phu.get('INSTRUME').upper() == 'NIRI'

    @astro_data_tag
    def _tag_instrument(self):
        return TagSet(['NIRI'])

    @astro_data_tag
    def _tag_dark(self):
        if self.phu.get('OBSTYPE') == 'DARK':
            return TagSet(['DARK', 'CAL'], blocks=['SPECT', 'IMAGE'])

    @astro_data_tag
    def _tag_arc(self):
        if self.phu.get('OBSTYPE') == 'ARC':
            return TagSet(['ARC'])

    @astro_data_tag
    def _tag_flat(self):
        if self.phu.get('OBSTYPE') == 'FLAT':
            return TagSet(['CAL', 'FLAT'])

    @astro_data_tag
    def _tag_image(self):
        if 'grism' not in self.phu.get('FILTER3', ''):
            tags = ['IMAGE']
            if self.phu.get('OBJECT', '').upper() == 'TWILIGHT':
                tags.extend(['CAL', 'FLAT', 'TWILIGHT'])

            return TagSet(tags)

    @astro_data_tag
    def _tag_spect(self):
        if 'grism' in self.phu.get('FILTER3', ''):
            return TagSet(['SPECT', 'LS'])
