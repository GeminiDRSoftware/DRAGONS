from astrodata import astro_data_tag, astro_data_descriptor, returns_list, TagSet
from ..gemini import AstroDataGemini
from .. import gmu

class AstroDataFlamingos(AstroDataGemini):

    __keyword_dict = dict(detector = 'DETECTOR',
                          filter_name = 'FILTER',
                          disperser = 'GRISM',
                          exposure_time = 'EXP_TIME',
                          )

    @staticmethod
    def _matches_data(source):
        return source[0].header.get('INSTRUME', '').upper() == 'FLAMINGOS'

    @astro_data_tag
    def _tag_instrument(self):
        return TagSet(['FLAMINGOS'])

    @astro_data_tag
    def _tag_spect(self):
        if self.phu.get('BIAS') == 1.0:
            return TagSet(['IMAGE'])
        else:
            return TagSet(['SPECT'])

    @astro_data_tag
    def _tag_flat(self):
        if 'flat' in self.phu.get('OBJECT', '').lower():
            return TagSet(['FLAT', 'CAL'])

    @astro_data_tag
    def _tag_twilight(self):
        if 'twilight' in self.phu.get('OBJECT', '').lower():
            return TagSet(['TWILIGHT', 'CAL'])

    @astro_data_tag
    def _tag_dark(self):
        if 'dark' in self.phu.get('OBJECT', '').lower():
            return TagSet(['DARK', 'CAL'])

    @astro_data_descriptor
    def central_wavelength(self, asMicrometers=False, asNanometers=False,
                           asAngstroms=False):
        """
        Returns the central wavelength in meters or the specified units

        Parameters
        ----------
        asMicrometers : bool
            If True, return the wavelength in microns
        asNanometers : bool
            If True, return the wavelength in nanometers
        asAngstroms : bool
            If True, return the wavelength in Angstroms

        Returns
        -------
        float
            The central wavelength setting

        """
        unit_arg_list = [asMicrometers, asNanometers, asAngstroms]
        if unit_arg_list.count(True) == 1:
            # Just one of the unit arguments was set to True. Return the
            # central wavelength in these units
            if asMicrometers:
                output_units = "micrometers"
            if asNanometers:
                output_units = "nanometers"
            if asAngstroms:
                output_units = "angstroms"
        else:
            # Either none of the unit arguments were set to True or more than
            # one of the unit arguments was set to True. In either case,
            # return the central wavelength in the default units of meters.
            output_units = "meters"

        return gmu.convert_units('micrometers', 1.5, output_units)

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
            declination in degrees
        """
        return self.target_dec()

    @astro_data_descriptor
    def detector(self):
        return self.phu.get(self._keyword_for('detector'))

    @astro_data_descriptor
    def disperser(self, stripID=False, pretty=False):
        """
        Returns the name of the disperser.

        Parameters
        ----------
        stripID : <bool>
            Does nothing.

        pretty : <bool>
            Also does nothing.

        Returns
        -------
        <str>:
            Name of the disperser.

        """
        dispr = self.phu.get(self._keyword_for('disperser'))
        if 'open' not in dispr and 'dark' not in dispr:
            return dispr
        else:
            return None
        return

    @astro_data_descriptor
    def exposure_time(self):
        """
        Returns the exposure time in seconds.

        Returns
        -------
        float
            Exposure time.
        """
        return self.phu.get(self._keyword_for('exposure_time'), None)

    @astro_data_descriptor
    def filter_name(self, stripID=False, pretty=False):
        return self.phu.get(self._keyword_for('filter_name'))

    @astro_data_descriptor
    def ra(self):
        """
        Returns the Right Ascension of the center of field in degrees.  Since a
        fiber is used it coincides with the position of the target. For code
        re-used, use target_ra() if you really want the position of the target
        rather than the center of the field.

        Returns
        -------
        float
            right ascension in degrees
        """
        return self.target_ra()
