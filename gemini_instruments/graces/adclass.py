from astrodata import astro_data_tag, astro_data_descriptor, returns_list, TagSet
from ..gemini import AstroDataGemini
from .. import gmu

class AstroDataGraces(AstroDataGemini):

    __keyword_dict = dict(detector = 'DETECTOR',
                          )

    @staticmethod
    def _matches_data(source):
        return source[0].header.get('INSTRUME', '').upper() == 'GRACES'

    @astro_data_tag
    def _tag_instrument(self):
        return TagSet(['GRACES'])

    @astro_data_tag
    def _tag_spect(self):
        return TagSet(['SPECT'])

    @astro_data_tag
    def _tag_arc(self):
        if self.phu.get('OBSTYPE') == 'ARC':
            return TagSet(['ARC', 'CAL'])

    @astro_data_tag
    def _tag_flat(self):
        if self.phu.get('OBSTYPE') == 'FLAT':
            return TagSet(['FLAT', 'CAL'])

    @astro_data_tag
    def _tag_bias(self):
        if self.phu.get('OBSTYPE') == 'BIAS':
            return TagSet(['BIAS', 'CAL'])

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

        return gmu.convert_units('micrometers', 0.7, output_units)

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
        Returns the name of the disperser.  For GRACES, this is always
        "GRACES".

        Parameters
        ----------
        stripID : bool
            Does nothing.
        pretty : bool
            Also does nothing.

        Returns
        -------
        str
            The name of the disperser, "GRACES".

        """
        return 'GRACES'

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
