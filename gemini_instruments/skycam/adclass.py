from __future__ import print_function

#
#                                                             Gemini Observatory
#
#                                                                        Dragons
#                                                             gemini_instruments
#                                                              skycam.adclass.py
# ------------------------------------------------------------------------------
import datetime
import dateutil.parser

from astrodata import astro_data_tag
from astrodata import astro_data_descriptor
from astrodata import returns_list
from astrodata import TagSet

from astrodata.fits import FitsLoader
from astrodata.fits import FitsProvider

from ..gemini import AstroDataGemini

# ------------------------------------------------------------------------------
class AstroDataSkyCam(AstroDataGemini):

    @staticmethod
    def _matches_data(source):
        match = source[1].header.get('TELESCOP', '') == '"GS_ALLSKYCAMERA"'
        return match

    @astro_data_tag
    def _tag_instrument(self):
        return TagSet(['GS_ALLSKYCAMERA'])

    @astro_data_tag
    def _tag_site(self):
        return TagSet(['SOUTH'])

    @astro_data_descriptor
    def instrument(self):
        hdr = self.TABLE1.meta.get('header')
        return hdr.get('TELESCOP').strip('"')

    @astro_data_descriptor
    def object(self):
        hdr = self.TABLE1.meta.get('header')
        return 'ZENITH'

    @astro_data_descriptor
    def exposure_time(self):
        hdr = self.TABLE1.meta.get('header')
        return hdr.get('EXPTIME')

    @astro_data_descriptor
    def ra(self):
        hdr = self.TABLE1.meta.get('header')
        return hdr.get('RA')

    @astro_data_descriptor
    def dec(self):
        hdr = self.TABLE1.meta.get('header')
        return hdr.get('DEC')
    
    @astro_data_descriptor
    def ut_datetime(self):
        hdr = self.TABLE1.meta.get('header')
        return dateutil.parser.parse(hdr.get('DATE-OBS'))
        

    @astro_data_descriptor
    def ut_time(self):
        return self.ut_datetime().time()
