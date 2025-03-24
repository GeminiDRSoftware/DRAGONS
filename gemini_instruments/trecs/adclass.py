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
    @gmu.return_requested_units(input_units="um")
    def central_wavelength(self):
        """
        Returns the central wavelength in microns

        Returns
        -------
        float
            The central wavelength setting
        """
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
        return wave_in_microns

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
    @gmu.return_requested_units(input_units="um")
    def dispersion(self):
        """
        Returns the dispersion in microns per pixel as a list (one value per
        extension) or a float if used on a single-extension slice.  It is
        possible to control the units of wavelength using the input arguments.

        Returns
        -------
        list/float
            The dispersion(s)
        """
        disperser = self.disperser()
        if disperser == 'LowRes-10':
            dispersion = 0.022
        elif disperser == 'LowRes-20':
            dispersion = 0.033
        elif disperser.startswith('HighRes-10'):
            dispersion = 0.0019
        else:
            return None
        return dispersion

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
