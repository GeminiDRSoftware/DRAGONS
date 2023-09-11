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
    def _matches_data(source):
        return source[0].header.get('INSTRUME', '').upper() == 'TRECS'

    @astro_data_tag
    def _tag_instrument(self):
        return TagSet(['TRECS'])

    # override gemini-level "type_mode" because OBSMODE is used for something
    # else in TReCS.
    @astro_data_tag
    def _type_mode(self):
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
        return gmu.convert_units('micrometers', wave_in_microns, output_units)

    @astro_data_descriptor
    def detector_x_offset(self):
        """
        Returns the offset from the reference position in pixels along
        the positive x-direction of the detector

        Returns
        -------
        float
            The offset in pixels
        """
        try:
            return self.phu.get('POFFSET') / self.pixel_scale()
        except TypeError:  # either is None
            return None

    @astro_data_descriptor
    def detector_y_offset(self):
        """
        Returns the offset from the reference position in pixels along
        the positive y-direction of the detector

        Returns
        -------
        float
            The offset in pixels
        """
        try:
            return -self.phu.get('QOFFSET') / self.pixel_scale()
        except TypeError:  # either is None
            return None

    @astro_data_descriptor
    def dispersion(self, asMicrometers=False, asNanometers=False,
                   asAngstroms=False):
        """
        Returns the dispersion in meters per pixel as a list (one value per
        extension) or a float if used on a single-extension slice.  It is
        possible to control the units of wavelength using the input arguments.

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
        return gmu.convert_units('micrometers', dispersion, output_units)

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
