from astrodata import astro_data_tag
from ..gemini import AstroDataGemini
import re

class AstroDataF2(AstroDataGemini):
    @staticmethod
    def _matches_data(data_provider):
        return data_provider.phu.get('INSTRUME').upper() in ('F2', 'FLAM')

    @astro_data_tag
    def _tag_instrument(self):
        return (set(['F2']), ())

    @astro_data_tag
    def _tag_dark(self):
        ot = self.phu.get('OBSTYPE')
        dkflt = False
        for f in (self.phu.get('FILTER1', ''), self.phu.get('FILTER2', '')):
            if re.match('DK.?', f):
                dkflt = True
                break

        if dkflt or ot == 'DARK':
            return (set(['DARK', 'CAL']), set(['IMAGE', 'SPECT']))

    @astro_data_tag
    def _tag_image(self):
        if self.phu.get('GRISM') == 'Open':
            return (set(['IMAGE']), ())

    def _tag_is_spect(self):
        grism = self.phu.get('GRISM', '')
        grpos = self.phu.get('GRISMPOS', '')

        for pattern in ("JH.?", "HK.?", "R3K.?"):
            if re.match(pattern, grism) or re.match(pattern, grpos):
                return True

        return False

    @astro_data_tag
    def _tag_is_ls(self):
        if not self._tag_is_spect():
            return

        decker = self.phu.get('DECKER') == 'Long_slit' or self.phu.get('DCKERPOS') == 'Long_slit'

        if decker or re.match(".?pix-slit", self.phu.get('MOSPOS', '')):
            return (set(['LS', 'SPECT']), ())

    @astro_data_tag
    def _tag_is_mos(self):
        if not self._tag_is_spect():
            return

        decker = self.phu.get('DECKER') == 'mos' or self.phu.get('DCKERPOS') == 'mos'

        if decker or re.match("mos.?", self.phu.get('MOSPOS', '')):
            return (set(['MOS', 'SPECT']), ())

    @astro_data_tag
    def _tag_arc(self):
        if self.phu.get('OBSTYPE') == 'ARC':
            return (set(['ARC', 'CAL']), ())

    @astro_data_tag
    def _tag_flat(self):
        if self.phu.get('OBSTYPE') == 'FLAT':
            return (set(['FLAT', 'CAL']), ())

    @astro_data_tag
    def _tag_twilight(self):
        if self.phu.get('OBJECT').upper() == 'TWILIGHT':
            rej = set(['FLAT']) if self.phu.get('GRISM') != 'Open' else set()
            return (set(['TWILIGHT', 'CAL']), rej)

    @astro_data_tag
    def _tag_disperser(self):
        disp = self.phu.get('DISPERSR', '')
        if disp.startswith('DISP_WOLLASTON'):
            return (set(['POL']), ())
        elif disp.startswith('DISP_PRISM'):
            return (set(['SPECT', 'IFU']), ())
