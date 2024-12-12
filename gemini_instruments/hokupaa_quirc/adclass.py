#
#                                                            Gemini Observatory
#
#                                                                        Dragons
#                                                             gemini_instruments
#                                                       hokupaa_QUIRC.adclass.py
# ------------------------------------------------------------------------------

from astrodata import astro_data_tag
from astrodata import astro_data_descriptor
from astrodata import returns_list
from astrodata import TagSet

from ..gemini import AstroDataGemini

# ------------------------------------------------------------------------------
class AstroDataHokupaaQUIRC(AstroDataGemini):
    __keyword_dict = dict(
        airmass = 'AMEND',
        wavelength_band = 'FILTER',
        observation_type = 'IMAGETYP',
    )

    @staticmethod
    def _matches_data(source):
        return source[0].header.get('INSTRUME', '') == 'Hokupaa+QUIRC'

    @astro_data_tag
    def _tag_instrument(self):
        return TagSet(['HOKUPAAQUIRC'])

    @astro_data_tag
    def _tag_image(self):
        return TagSet(['IMAGE'])

    @astro_data_tag
    def _tag_dark(self):
        if 'dark' in self.phu.get('OBJECT', '').lower():
            return TagSet(['DARK'], blocks=['IMAGE', 'SPECT'])

    @astro_data_tag
    def _tag_flat(self):
        if 'flat' in self.phu.get('OBJECT', '').lower():
            return TagSet(['FLAT', 'CAL'])

    @astro_data_descriptor
    def airmass(self):
        return self.phu.get(self._keyword_for('airmass'))

    @astro_data_descriptor
    def detector_name(self, pretty=False):
        """
        Returns the name of the detector. For HOKUPAA+QUIRC, this is always
        'QUIRC'

        Returns
        -------
        <str>:
            Detector name

        """
        return 'QUIRC'

    @astro_data_descriptor
    def filter_name(self, pretty=False):
        """
        This descriptor is used to display 'WaveBand' in the archive.

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
        return self.phu.get(self._keyword_for('instrument'))


    @astro_data_descriptor
    def observation_type(self):
        """
        Returns 'type' the observation.

        Returns
        -------
        <str>:
            observation type.

        """
        return self.phu.get(self._keyword_for('observation_type'))

    @astro_data_descriptor
    def ra(self):
        """
        Returns the name of the

        Returns
        -------
        <str>:
            right ascension

        """
        return self._ra()

    @astro_data_descriptor
    def dec(self):
        """
        Returns the name of the

        Returns
        -------
        <str>:
            declination

        """
        return self._dec()

    @astro_data_descriptor
    def wavelength_band(self):
        """
        Returns the name of the bandpass of the observation.

        Returns
        -------
        <str>:
            Name of the bandpass.

        """
        return self.phu.get(self._keyword_for('wavelength_band'))

    @astro_data_descriptor
    def target_ra(self):
        return self.wcs_ra()

    @astro_data_descriptor
    def target_dec(self):
        return self.wcs_dec()
