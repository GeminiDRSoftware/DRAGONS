#
#                                                            Gemini Observatory
#
#                                                                        Dragons
#                                                             gemini_instruments
#                                                              igrins.adclass.py
# ------------------------------------------------------------------------------

from astrodata import astro_data_tag
from astrodata import astro_data_descriptor
from astrodata import returns_list
from astrodata import TagSet

from ..gemini import AstroDataGemini
#from .. import gmu
# ------------------------------------------------------------------------------
class AstroDataIgrins(AstroDataGemini):
    __keyword_dict = dict(
        airmass = 'AMSTART',
        wavelength_band = 'BAND',
        detector_name = 'DETECTOR',
        observation_class = 'OBJTYPE',
        observation_type = 'OBJTYPE',
        ra = 'OBJRA',
        dec = 'OBJDEC',
        slit_x_center = 'SLIT_CX',
        slit_y_center = 'SLIT_CY',
        slit_width = 'SLIT_WID',
        slit_length = 'SLIT_LEN',
        slit_angle = 'SLIT_ANG'
    )

    @staticmethod
    def _matches_data(source):
        grins = source[0].header.get('INSTRUME', '').upper() == 'IGRINS'
        if not grins:
            grins = source[1].header.get('INSTRUME', '').upper() == 'IGRINS'
        return grins

    @astro_data_tag
    def _tag_instrument(self):
        return TagSet(['IGRINS'])

    @astro_data_tag
    def _tag_image(self):
        if self.phu.get('BAND') == 'S' or self[0].hdr.get('BAND') == 'S':
            return TagSet(['IMAGE', 'ACQUISITION'])

    # @astro_data_tag
    # def _tag_spect(self):
    #     if self.phu.get('BAND') in ['K', 'H']:
    #         return TagSet(['SPECT'])

    @astro_data_tag
    def _tag_dark(self):
        if self.phu.get('OBJTYPE') == 'DARK' or self[0].hdr.get('OBJTYPE') == 'DARK':
            return TagSet(['DARK'], blocks=['IMAGE', 'SPECT'])

    @astro_data_tag
    def _tag_arc(self):
        if self.phu.get('OBJTYPE') == 'ARC' or self[0].hdr.get('OBJTYPE') == 'ARC':
            return TagSet(['ARC', 'CAL'])

    @astro_data_tag
    def _tag_flat(self):
        if self.phu.get('OBJTYPE') == 'FLAT' or self[0].hdr.get('OBJTYPE') == 'FLAT':
            return TagSet(['FLAT', 'CAL'])

    @astro_data_tag
    def _tag_standard(self):
        if self.phu.get('OBJTYPE') == 'STD' or self[0].hdr.get('OBJTYPE') == 'STD':
            return TagSet(['STANDARD', 'CAL'])

    @astro_data_tag
    def _tag_science(self):
        if self.phu.get('OBJTYPE') == 'TAR' or self[0].hdr.get('OBJTYPE') == 'TAR':
            return TagSet(['SCIENCE'])

    @astro_data_descriptor
    def airmass(self):
        aim = self.phu.get(self._keyword_for('airmass'))
        if not aim:
            aim = self[0].hdr.get(self._keyword_for('airmass'))
        return aim

    @astro_data_descriptor
    def detector_name(self, pretty=False):
        """
        Returns the name of the detector

        Returns
        -------
        <str>:
            Detector name

        """
        detnam = self.phu.get(self._keyword_for('detector_name'))
        if not detnam:
            detnam = self[0].hdr.get(self._keyword_for('detector_name'))
        return detnam

    @astro_data_descriptor
    def filter_name(self, pretty=False):
        """
        IGRINS has no filters or filter names. But this descriptor is
        used to display 'WaveBand' in the archive. So, IGRINS data
        needs to "fake it." wavelength_band() returns a string exactly
        equal to a call on filter_name(pretty=True).

        Parameters
        ----------
        pretty: <bool>
            This keyword parameter is present for API purposes.
            It has no effect for this descriptor.

        Returns
        -------
        <str>:
             wavelength band substituting for filter_name(pretty=True)

        """
        return self.wavelength_band()

    @astro_data_descriptor
    def instrument(self, generic=False):
        """
        Returns the name of the instrument making the observation

        Parameters
        ----------
        generic: <bool>
            Request the generic instrument name, if applicable.

        Returns
        -------
        <str>:
            instrument name

        """
        inst = self.phu.get(self._keyword_for('instrument'))
        if not inst:
            inst = self[0].hdr.get(self._keyword_for('instrument'))

        return inst

    @astro_data_descriptor
    def observation_class(self):
        """
        Returns 'class' the observation; one of,

            'science', 'acq', 'projCal', 'dayCal', 'partnerCal', 'acqCal'

        An 'acq' is defined by BAND == 'S', where 'S' indicates a slit image.

        Returns
        -------
        oclass: <str>
            One of the above enumerated names for observation class.

        """
        oclass = None

        otype = self.phu.get(self._keyword_for('observation_class'))
        if not otype:
            otype = self[0].hdr.get(self._keyword_for('observation_class'))

        if 'S' in self.wavelength_band():
            oclass = 'acq'

        if 'STD' in otype:
            oclass = 'partnerCal'
        elif 'TAR' in otype:
            oclass = 'science'

        return oclass

    @astro_data_descriptor
    def observation_type(self):
        """
        Returns 'type' the observation. For IGRINS, this will be one of,

            'OBJECT', 'DARK', 'FLAT', 'ARC'

        Returns
        -------
        otype: <str>
            Observation type.

        """
        otype = self.phu.get(self._keyword_for('observation_type'))
        if not otype:
            otype = self[0].hdr.get(self._keyword_for('observation_type'))

        if otype in ['STD', 'TAR']:
            otype = 'OBJECT'

        return otype

    @astro_data_descriptor
    def ra(self):
        """
        Returns the RA of the observation.

        Returns
        -------
        rad: <str>
            Right Ascension

        """
        rad = self.phu.get(self._keyword_for('ra'))
        if not rad:
            rad = self[0].hdr.get(self._keyword_for('ra'))
        return rad

    @astro_data_descriptor
    def dec(self):
        """
        Returns the declination of observation.

        Returns
        -------
        decd: <str>
            Declination

        """
        decd = self.phu.get(self._keyword_for('dec'))
        if not decd:
            decd = self[0].hdr.get(self._keyword_for('dec'))
        return decd

    @astro_data_descriptor
    def wavelength_band(self):
        """
        Returns the name of the bandpass of the observation.

        Returns
        -------
        <str>:
            Name of the bandpass.

        """
        waveb = self.phu.get(self._keyword_for('wavelength_band'))
        if not waveb:
            waveb = self[0].hdr.get(self._keyword_for('wavelength_band'))
        return waveb

    #@astro_data_descriptor
    def _slit_x_center(self):
        """
        Returns Center x position of slit in the SVC image

        Returns
        -------
        <int>:
            center x position in pixels

        """
        return self.phu.get(self._keyword_for('slit_x_center'))

    #@astro_data_descriptor
    def _slit_y_center(self):
        """
        Returns Center y position of slit in the SVC image

        Returns
        -------
        <int>:
            center y position in pixels

        """
        return self.phu.get(self._keyword_for('slit_y_center'))

    #@astro_data_descriptor
    def _slit_width(self):
        """
        Returns slit width in the SVC image

        Returns
        -------
        <int>:
            slit width in pixels

        """
        return self.phu.get(self._keyword_for('slit_width'))

    #@astro_data_descriptor
    def _slit_length(self):
        """
        Returns slit length in the SVC image

        Returns
        -------
        <int>:
            slit length in pixels

        """
        return self.phu.get(self._keyword_for('slit_length'))

    #@astro_data_descriptor
    def _slit_angle(self):
        """
        Returns slit length in the SVC image

        Returns
        -------
        <int>:
            slit length in pixels

        """
        return self.phu.get(self._keyword_for('slit_angle'))

    @astro_data_descriptor
    def target_ra(self):
        return self.wcs_ra()

    @astro_data_descriptor
    def target_dec(self):
        return self.wcs_dec()
