from astrodata import astro_data_tag
from ..gemini import AstroDataGemini
import re

class AstroDataNici(AstroDataGemini):
    @staticmethod
    def _matches_data(data_provider):
        return data_provider.phu.get('INSTRUME', '').upper() == 'NICI'

    @astro_data_tag
    def _tag_instrument(self):
        return (set(['NICI', 'IMAGE']), ())

    @astro_data_tag
    def _tag_dark(self):
        if self.phu.get('OBSTYPE') == 'DARK':
            return (set(['DARK', 'CAL']), ())

    @astro_data_tag
    def _tag_flat(self):
        # NOTE: This will be set also for old type DARK frames... What should we do?
        if self.phu.get('OBSTYPE') == 'FLAT':
            return (set(['FLAT', 'CAL']), set(['SDI', 'ASDI']))

    @astro_data_tag
    def _tag_dichroic(self):
        dich = self.phu.get('DICHROIC', '')
        if 'Mirror' in dich:
            return (set(['ADI_B']), ())
        elif 'Open' in dich:
            return (set(['ADI_R']), ())
        elif '50/50' in dich:
            crmode = self.phu.get('CRMODE')
            if crmode == 'FOLLOW':
                return (set(['SDI']), ())
            elif crmode == 'FIXED':
                return (set(['ASDI']), ())
