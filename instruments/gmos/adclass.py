from astrodata import astro_data_tag
from ..gemini import AstroDataGemini
import re

class AstroDataGmos(AstroDataGemini):
    @staticmethod
    def _matches_data(data_provider):
        return data_provider.phu.get('INSTRUME').upper() in ('GMOS-N', 'GMOS-S')

    @astro_data_tag
    def _tag_instrument(self):
        # tags = ['GMOS', self.instrument().upper().replace('-', '_')]
        return (set(['GMOS']), ())

    @astro_data_tag
    def _tag_dark(self):
        if self.phu.get('OBSTYPE') == 'DARK':
            return (set(['DARK']), ())

    @astro_data_tag
    def _tag_arc(self):
        if self.phu.get('OBSTYPE') == 'ARC':
            return (set(['ARC', 'CAL']), ())

    @astro_data_tag
    def _tag_bias(self):
        if self.phu.get('OBSTYPE') == 'BIAS':
            return (set(['BIAS', 'CAL']), set(['IMAGE', 'SPECT']))

    @astro_data_tag
    def _tag_flat(self):
        if self.phu.get('OBSTYPE') == 'FLAT':
            if self.phu.get('GRATING') == 'MIRROR':
                f1, f2 = self.phu.FILTER1, self.phu.FILTER2
                # This kind of filter prevents imaging to be classified as FLAT
                if any(('Hartmann' in f) for f in (f1, f2)):
                    return

            return (set(['FLAT', 'CAL']), ())

    @astro_data_tag
    def _tag_twilight(self):
        if self.phu.get('OBJECT').upper() == 'TWILIGHT':
            rej = set(['FLAT']) if self.phu.get('GRATING') != 'MIRROR' else set()
            return (set(['TWILIGHT', 'CAL']), rej)

    def _tag_is_spect(self):
        pairs = (
            ('MASKTYP', 0),
            ('MASKNAME', 'None'),
            ('GRATING', 'MIRROR')
        )

        matches = (self.phu.get(kw) == value for (kw, value) in pairs)
        if any(matches):
            return False
        return True

    @astro_data_tag
    def _tag_ifu(self):
        if not self._tag_is_spect():
            return

        mapping = {
            'IFU-B': 'BLUE',
            'IFU-B-NS': 'BLUE',
            'b': 'BLUE',
            'IFU-R': 'RED',
            'IFU-R-NS': 'RED',
            'r': 'RED',
            'IFU-2': 'TWO',
            'IFU-2-NS': 'TWO',
            's': 'TWO'
        }

        names = set(key for key in mapping.keys() if key.startswith('IFU'))

        mskt, mskn = self.phu.get('MASKTYP'), self.phu.get('MASKNAME')
        if mskt == -1 and (mskn in names or re.match('g[ns]ifu_slit[rbs]_mdf', mskn)):
            if mskn not in names:
                mskn = re.match('g.ifu_slit(.)_mdf', mskn).groups()[0]

            return (set(['SPECT', 'IFU', mapping[mskn]]), ())

    @astro_data_tag
    def _tag_image(self):
        if self.phu.get('GRATING') == 'MIRROR':
            return (set(['IMAGE']), ())

    @astro_data_tag
    def _tag_ls(self):
        if not self._tag_is_spect():
            return

        if self.phu.get('MASKTYP') == 1 and self.phu.get('MASKNAME', '').endswith('arcsec'):
            return (set(['SPECT', 'LS']), ())

    @astro_data_tag
    def _tag_mos(self):
        if not self._tag_is_spect():
            return

        mskt = self.phu.get('MASKTYP')
        mskn = self.phu.get('MASKNAME', '')
        if mskt == 1 and not (mskn.startswith('IFU') or mskn.startswith('focus') or mskn.endswith('arcsec')):
            return (set(['SPECT', 'MOS']), ())

    @astro_data_tag
    def _tag_nodandshuffle(self):
        if 'NODPIX' in self.phu:
            return (set(['SPECT', 'NODANDSHUFFLE']), ())
