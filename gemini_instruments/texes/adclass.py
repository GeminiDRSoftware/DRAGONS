from __future__ import print_function

#
#                                                             Gemini Observatory
#
#                                                                        Dragons
#                                                             gemini_instruments
#                                                               texes.adclass.py
# ------------------------------------------------------------------------------
from astrodata import astro_data_tag
from astrodata import astro_data_descriptor
from astrodata import returns_list
from astrodata import TagSet

from astrodata.fits import FitsLoader
from astrodata.fits import FitsProvider

from ..gemini import AstroDataGemini

# ------------------------------------------------------------------------------
class AstroDataTexes(AstroDataGemini):
    __keyword_dict = dict(
        ra = 'RA',
        dec = 'DEC',
        target_ra = 'TARGRA',
        target_dec = 'TARGDEC',
        exposure_time = 'OBSTIME',
        observation_type = 'OBSTYPE',
        )

    @classmethod
    def load(cls, source):
        def texes_parser(hdu):
            xnam, xver = hdu.header.get('EXTNAME'), hdu.header.get('EXTVER')
            if 'RAWFRAME' in [xnam] and xver:
                hdu.header.set('EXTNAME0', xnam,
                               'EXTNAME Orig (AstroData)',before='EXTNAME')
                hdu.header.set('EXTNAME', 'SCI', 'Renamed by AstroData')
            elif 'SCAN-FRAME' in [xnam] and xver:
                hdu.header.set('EXTNAME0', xnam,
                               'EXTNAME Orig (AstroData)',before='EXTNAME')
                hdu.header.set('EXTNAME', 'SCI', 'Renamed by AstroData')
            elif xnam and not xver:
                hdu.header.set('EXTVER', 1, 'Versioned by AstroData',
                               after='EXTNAME')

        return cls(FitsLoader(FitsProvider).load(source, extname_parser=texes_parser))

    @staticmethod
    def _matches_data(source):
        return source[0].header.get('INSTRUME', '') == 'TEXES'

    @astro_data_tag
    def _tag_instrument(self):
        return TagSet(['TEXES'])

    @astro_data_tag
    def _tag_image(self):
        return TagSet(['SPECT'])

    @astro_data_tag
    def _tag_dark(self):
        if 'dark' in self.phu.get('OBSTYPE').lower():
            return TagSet(['DARK', 'CAL'], blocks=['SPECT'])

    @astro_data_tag
    def _tag_flat(self):
        if 'flat' in self.phu.get('OBSTYPE').lower():
            return TagSet(['FLAT', 'CAL'])

    @astro_data_tag
    def _tag_bias(self):
        if 'bias' in self.phu.get('OBSTYPE').lower():
            return TagSet(['BIAS', 'CAL'], blocks=['SPECT'])

    @astro_data_descriptor
    def exposure_time(self):
        """
        Returns
        -------
        exposure_time: <float>
            Exposure time.

        """
        return self.phu.get(self._keyword_for('exposure_time'))

    @astro_data_descriptor
    def observation_type(self):
        return self.phu.get('OBSTYPE').upper()
    
    @astro_data_descriptor
    def ra(self):
        return self.phu.get(self._keyword_for('ra'))

    @astro_data_descriptor
    def dec(self):
        return self.phu.get(self._keyword_for('dec'))

    
