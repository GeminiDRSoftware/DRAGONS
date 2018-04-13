from astrodata import astro_data_tag
from astrodata import astro_data_descriptor
from astrodata import TagSet

from ..gemini import AstroDataGemini
from .. import gmu

class AstroDataOscir(AstroDataGemini):

    @staticmethod
    def _matches_data(source):
        return source[0].header.get('INSTRUME', '').upper() == 'OSCIR'

    @astro_data_tag
    def _tag_instrument(self):
        return TagSet(['OSCIR'])

    @astro_data_descriptor
    def airmass(self):
        return float(self.phu.get('AIRMASS1'))

    @astro_data_descriptor
    def exposure_time(self):
        return float(self.phu.get('EXPTIME'))

    @astro_data_descriptor
    def dec(self):
        """
        Returns the Declination of the center of the field, in degrees.

        Returns
        -------
        float
            declination in degrees
        """
        return self.phu.get('DEC')

    @astro_data_descriptor
    def ra(self):
        """
        Returns the Right Ascension of the center of the field, in degrees.

        Returns
        -------
        float
            right ascension in degrees
        """
        return self.phu.get('RA')
