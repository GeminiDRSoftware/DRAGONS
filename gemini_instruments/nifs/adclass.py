import math

from astrodata import astro_data_tag, astro_data_descriptor, returns_list, TagSet
from ..gemini import AstroDataGemini
from .lookup import constants_by_bias, config_dict, lnrs_mode_map

# NOTE: Temporary functions for test. gempy imports astrodata and
#       won't work with this implementation
from .. import gmu

class AstroDataNifs(AstroDataGemini):

    __keyword_dict = dict(array_section = 'DATASEC',
                          camera = 'INSTRUME',
                          central_wavelength = 'GRATWAVE',
                          detector_section = 'DATASEC',
                          disperser = 'GRATING',
                          focal_plane_mask = 'APERTURE')
    @staticmethod
    def _matches_data(data_provider):
        return data_provider.phu.get('INSTRUME', '').upper() == 'NIFS'

    @astro_data_tag
    def _tag_instrument(self):
        return TagSet(['NIFS'])

    @astro_data_tag
    def _tag_dark(self):
        if self.phu.OBSTYPE == 'DARK':
            return TagSet(['DARK', 'CAL'], blocks=['IMAGE', 'SPECT'])

    @astro_data_tag
    def _tag_image(self):
        if self.phu.FLIP == 'In':
            return TagSet(['IMAGE'])

    @astro_data_tag
    def _tag_arc(self):
        if self.phu.OBSTYPE == 'ARC':
            return TagSet(['ARC', 'CAL'])

    @astro_data_tag
    def _tag_ronchi(self):
        req = self.phu.OBSTYPE, self.phu.APERTURE
        if req == ('FLAT', 'Ronchi_Screen_G5615'):
            return TagSet(['RONCHI', 'CAL'])

    @astro_data_tag
    def _tag_spect(self):
        if self.phu.FLIP == 'Out':
            return TagSet(['SPECT', 'IFU'])

    @astro_data_descriptor
    def filter_name(self, stripID=False, pretty=False):
        """
        Returns the name of the filter(s) used.  The component ID can be
        removed with either 'stripID' or 'pretty'.  If 'pretty' is True,
        filter positions such as 'Open', 'Dark', 'blank', and others are
        removed leaving only the relevant filters in the string.

        Parameters
        ----------
        stripID : bool
            If True, removes the component ID and returns only the name of
            the filter.
        pretty : bool
            Same as for stripID.  Pretty here does not do anything more.

        Returns
        -------
        str
            The name of the filter with or without the component ID.

        """
        filt = str(self.phu.FILTER)
        if stripID or pretty:
            filt = gmu.removeComponentID(filt)

        if filt == "Blocked":
            return "blank"
        return filt

    def _from_biaspwr(self, constant_name):
        bias_volt = self.phu.BIASPWR

        for bias, constants in constants_by_bias.items():
            if abs(bias - bias_volt) < 0.1:
                return getattr(constants, constant_name)

        raise KeyError("The bias value for this image doesn't match any on the lookup table")

    @returns_list
    @astro_data_descriptor
    def gain(self):
        """
        Returns the gain used for the observation.  A lookup table is
        uses to compare the bias value in the headers to the bias values
        associate with the various gain settings.

        Returns
        -------
        float
            Gain used for the observation.

        """
        return self._from_biaspwr("gain")

    @astro_data_descriptor
    def non_linear_level(self):
        """
        Returns the level at which the array becomes non-linear.  The
        return units are ADUs.  A lookup table is used.  Whether the data
        has been corrected for non-linearity or not is taken into account.
         A list is returned unless called on a single-extension slice.

        Returns
        -------
        list/int
            Level in ADU at which the non-linear regime starts.

        """
        saturation_level = self.saturation_level()
        corrected = 'NONLINCR' in self.phu

        linearlimit = self._from_biaspwr("linearlimit" if corrected else "nonlinearlimit")
        try:
            return [int(linear_limit * s) for s in saturation_level]
        except TypeError:
            return int(saturation_level * linearlimit)

    @returns_list
    @astro_data_descriptor
    def pixel_scale(self):
        """
        Returns the pixel scale in arc seconds.  A lookup table indexed on
        focal_plane_mask, disperser, and filter_name is used.
         A list is returned unless called on a single-extension slice.

        Returns
        -------
        list/float
            Pixel scale in arcsec.

        """
        fpm = self.focal_plane_mask()
        disp = self.disperser()
        filt = self.filter_name()
        pixel_scale = (fpm, disp, filt)

        try:
            config = config_dict[pixel_scale]
        except KeyError:
            raise KeyError("Unknown configuration: {}".format(pixel_scale))

        # The value is always a float for the common config. We'll keep the
        # coercion just to make sure people don't mess with new entries
        # This will raise a ValueError for bogus pixscales
        return float(config.pixscale)

    @astro_data_descriptor
    def read_mode(self):
        """
        Returns the read mode for the observation.  The read mode is directly
        associated with the LNRS header keyword value.

        Returns
        -------
        str
            Read mode for the observation.

        """
        # NOTE: The original read_mode descriptor obtains the bias voltage
        #       value, but then it does NOTHING with it. I'll just skip it.

        return lnrs_mode_map.get(self.phu.LNRS, 'Invalid')

    @returns_list
    @astro_data_descriptor
    def read_noise(self):
        """
        Returns the detector read noise, in electrons, for the observation.
        A lookup table is used.  The read noise depends on the gain setting
        and is affected by the number of coadds and non-destructive pairs.
        A list is returned unless called on a single-extension slice.
        Returns
        -------
        list/float
            Detector read noise in electrons.

        """
        rn = self._from_biaspwr("readnoise")
        return float(rn * math.sqrt(self.coadds()) / math.sqrt(self.phu.LNRS))

    @returns_list
    @astro_data_descriptor
    def saturation_level(self):
        """
        Returns the saturation level for the observation, in ADUs
        A lookup table is used to get the full well value based on the
        gain. A list is returned unless called on a single-extension slice.

        Returns
        -------
        list/int
            Saturation level in ADUs.

        """
        well = self._from_biaspwr("well")
        return int(well * self.coadds())
