from astrodata import astro_data_tag, astro_data_descriptor, returns_list, TagSet
from ..gemini import AstroDataGemini
from .. import gmu

class AstroDataBhros(AstroDataGemini):

    __keyword_dict = dict(array_section = 'CCDSEC',
                          central_wavelength = 'WAVELENG',
                          overscan_section = 'BIASSEC')

    @staticmethod
    def _matches_data(source):
        return source[0].header.get('INSTRUME', '').upper() == 'BHROS'

    @astro_data_tag
    def _tag_instrument(self):
        return TagSet({'BHROS', 'SPECT'}, ())

    @astro_data_descriptor
    @gmu.return_requested_units(input_units="AA")
    def central_wavelength(self):
        """
        Returns the central wavelength

        Returns
        -------
        float
            The central wavelength setting
        """
        # The central_wavelength keyword is in Angstroms
        keyword = self._keyword_for('central_wavelength')
        wave_in_angstroms = self.phu.get(keyword, -1)
        if wave_in_angstroms < 0:
            return None
        return wave_in_angstroms

    @astro_data_descriptor
    def dec(self):
        """
        Returns the Declination of the center of field in degrees.  Since a
        fiber is used it coincides with the position of the target. For code
        re-used, use target_dec() if you really want the position of the target
        rather than the center of the field.

        Returns
        -------
        float
            right ascension in degrees
        """
        # using the target_dec descriptor since the WCS is not sky coords
        return self.target_dec(offset=True, icrs=True)

    @astro_data_descriptor
    def disperser(self, stripID=False, pretty=False):
        """
        Returns the name of the disperser. This is always 'bHROS'

        Parameters
        ----------
        stripID : bool
            Does nothing
        pretty : bool
            Does nothing

        Returns
        -------
        str
            The name of the disperser
        """
        return 'bHROS'

    @astro_data_descriptor
    def ra(self):
        """
        Returns the Right Ascension of the center of field in degrees.
        Since a fiber is used it coincides with the position of the target.
        For code re-used, use target_ra() if you really want the position of
        the target rather than the center of the field.

        Returns
        -------
        float
            right ascension in degrees
        """
        # using the target_ra descriptor since the WCS is not sky coords
        return self.target_ra(offset=True, icrs=True)
