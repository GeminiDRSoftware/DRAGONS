#
#                                                            Gemini Observatory
#
#                                                                        Dragons
#                                                             gemini_instruments
#                                                               hrwfs.adclass.py
# ------------------------------------------------------------------------------
__version__      = "0.1 (beta)"
# ------------------------------------------------------------------------------

from astrodata import astro_data_tag
from astrodata import astro_data_descriptor
from astrodata import returns_list
from astrodata import TagSet

from ..gemini import AstroDataGemini

# ------------------------------------------------------------------------------
class AstroDataHrwfs(AstroDataGemini):

    @staticmethod
    def _matches_data(source):
        return source[0].header.get('INSTRUME', '') == 'hrwfs'

    @astro_data_tag
    def _tag_instrument(self):
        return TagSet(['HRWFS'])

    @astro_data_tag
    def _tag_image(self):
        return TagSet(['IMAGE'])

    @astro_data_tag
    def _tag_dark(self):
        if 'dark' in self.phu.get('OBSTYPE').lower():
            return TagSet(['DARK', 'CAL'], blocks=['IMAGE'])

    @astro_data_tag
    def _tag_flat(self):
        if 'flat' in self.phu.get('OBSTYPE').lower():
            return TagSet(['FLAT', 'CAL'])

    @astro_data_tag
    def _tag_bias(self):
        if 'bias' in self.phu.get('OBSTYPE').lower():
            return TagSet(['BIAS', 'CAL'], blocks=['IMAGE'])
