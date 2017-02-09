from astrodata import astro_data_tag, astro_data_descriptor, returns_list, TagSet
from ..gemini import AstroDataGemini
from .. import gmu

class AstroDataTrecs(AstroDataGemini):

    __keyword_dict = dict(camera = 'OBSMODE',
                          disperser = 'GRATING',
                          exposure_time = 'OBJTIME',
                          focal_plane_mask = 'SLIT',
                          pupil_mask = 'PUPILIMA')

    @staticmethod
    def _matches_data(data_provider):
        return data_provider.phu.get('INSTRUME', '').upper() == 'TRECS'

    @astro_data_tag
    def _tag_instrument(self):
        return TagSet(['TRECS'])

    @astro_data_tag
    def _tag_image_spect(self):
        if 'MIRROR' in self.phu.get('GRATING', '').upper():
            return TagSet(['IMAGE'])
        else:
            return TagSet(['SPECT'])

    @astro_data_descriptor
    def central_wavelength(self, asMicrometers=False, asNanometers=False,
                           asAngstroms=False):
        """
        Returns the central wavelength in meters or specified units

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

        disperser = self.disperser()
        if disperser is None:
            return None
        if disperser == 'LowRes-10':
            wave_in_microns = 10.5
        elif disperser == 'LowRes-20':
            wave_in_microns = 20.0
        elif disperser.startswith('HighRes-10'):
            wave_in_microns = self.phu.get('HRCENWL')
        else:
            return None
        return gmu.convert_units('microns', wave_in_microns, output_units)

    @astro_data_descriptor
    def dispersion(self, asMicrometers=False, asNanometers=False,
                   asAngstroms=False):
        """
        Returns the dispersion (wavelength units per pixel) in meters
        or specified units, as a list (one value per extension) or a
        float if used on a single-extension slice, or if the keyword
        is in the PHU

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
        list/float
            The dispersion(s)
        """
        # Look for the relevant, which we assume is in meters per pixel
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

        disperser = self.disperser()
        if disperser == 'LowRes-10':
            dispersion = 0.022
        elif disperser == 'LowRes-20':
            dispersion = 0.033
        elif disperser.startswith('HighRes-10'):
            dispersion = 0.0019
        else:
            dispersion = None
        return gmu.convert_units('microns', dispersion, output_units)

    @returns_list
    @astro_data_descriptor
    def gain(self):
        """
        Returns the gain (electrons/ADU) for each extension

        Returns
        -------
        list of floats/float
            Gains used for the observation
        """
        bias_level = self.phu.get('BIASLEVL', '')
        if bias_level == '2':
            return 214.0
        elif bias_level == '1':
            return 718.0

    @astro_data_descriptor
    def pixel_scale(self):
        """
        Returns the image scale in arcseconds per pixel

        Returns
        -------
        float
            the pixel scale
        """
        return 0.089