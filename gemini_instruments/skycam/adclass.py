#
#                                                             Gemini Observatory
#
#                                                                        Dragons
#                                                             gemini_instruments
#                                                              skycam.adclass.py
# ------------------------------------------------------------------------------

from astrodata import astro_data_tag, astro_data_descriptor, TagSet

from ..gemini import AstroDataGemini

# ------------------------------------------------------------------------------
class AstroDataSkyCam(AstroDataGemini):

    @staticmethod
    def _matches_data(source):
        # remove "s because they added them, but this will also work if the fix it
        match = source[0].header.get('TELESCOP', '').strip('"') == 'GS_ALLSKYCAMERA'
        return match

    @astro_data_tag
    def _tag_instrument(self):
        return TagSet(['GS_ALLSKYCAMERA'])

    @astro_data_tag
    def _tag_site(self):
        return TagSet(['SOUTH'])

    @astro_data_descriptor
    def instrument(self):
        return self.phu['TELESCOP'].strip('"')

    @astro_data_descriptor
    def object(self):
        return 'ZENITH'
