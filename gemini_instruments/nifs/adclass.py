import math

from astrodata import astro_data_tag, astro_data_descriptor, returns_list, TagSet
from ..gemini import AstroDataGemini, use_keyword_if_prepared
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
                          focal_plane_mask = 'APERTURE',
                          observation_epoch = 'EPOCH')
    @staticmethod
    def _matches_data(source):
        return source[0].header.get('INSTRUME', '').upper() == 'NIFS'

    @astro_data_tag
    def _tag_instrument(self):
        return TagSet(['NIFS'])

    @astro_data_tag
    def _tag_dark(self):
        if self.phu.get('OBSTYPE') == 'DARK':
            return TagSet(['DARK', 'CAL'], blocks=['IMAGE', 'SPECT'])

    @astro_data_tag
    def _tag_image(self):
        if self.phu.get('FLIP') == 'In':
            return TagSet(['IMAGE'])

    @astro_data_tag
    def _tag_arc(self):
        if self.phu.get('OBSTYPE') == 'ARC':
            return TagSet(['ARC', 'CAL'])

    @astro_data_tag
    def _tag_ronchi(self):
        req = self.phu.get('OBSTYPE'), self.phu.get('APERTURE')
        if req == ('FLAT', 'Ronchi_Screen_G5615'):
            return TagSet(['RONCHI', 'CAL'])

    @astro_data_tag
    def _tag_spect(self):
        if self.phu.get('FLIP') == 'Out':
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
        filt = self.phu.get('FILTER')
        if stripID or pretty:
            filt = gmu.removeComponentID(filt)
        return 'blank' if filt == 'Blocked' else filt

    def _from_biaspwr(self, constant_name):
        bias_volt = self.phu.get('BIASPWR')

        for bias, constants in constants_by_bias.items():
            if abs(bias - bias_volt) < 0.1:
                return getattr(constants, constant_name, None)

        raise KeyError("The bias value for this image doesn't match any on the lookup table")

    @returns_list
    @use_keyword_if_prepared
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
    def gcal_lamp(self):
        """
        Returns the name of the GCAL lamp being used, or "Off" if no lamp is
        in used.  This applies to flats and arc observations when a lamp is
        used.  For other types observation, None is returned.

        This overrides the gemini level descriptor, as NIFS has more lamp names
        than are accommodated by that descriptor function.

        Returns
        -------
        lamps: <str>
            Name of the GCAL lamp, or "Off"

        """
        lamps, shut = self.phu.get('GCALLAMP'), self.phu.get('GCALSHUT')
        if lamps is None:
            return None
        
        if shut and "CLOSED" in shut.upper():
            return 'Off'
        elif lamps and "OPEN" in shut.upper():
            return lamps

    @use_keyword_if_prepared
    @astro_data_descriptor
    def non_linear_level(self):
        """
        Returns the level at which the array becomes non-linear, in the same
        units as the data. A lookup table is used. Whether the data
        has been corrected for non-linearity or not is taken into account.
        A list is returned unless called on a single-extension slice.

        Returns
        -------
        int/list
            Level in ADU at which the non-linear regime starts.
        """
        saturation_level = self.saturation_level()
        corrected = 'NONLINCR' in self.phu

        linear_limit = self._from_biaspwr("linearlimit" if corrected
                                          else "nonlinearlimit")
        if self.is_single:
            try:
                return int(saturation_level * linear_limit)
            except TypeError:
                return None
        else:
            return [int(linear_limit * s) if linear_limit and s else None
                    for s in saturation_level]

    @astro_data_descriptor
    def pixel_scale(self):
        """
        Returns the pixel scale in arc seconds.  A lookup table indexed on
        focal_plane_mask, disperser, and filter_name is used.

        Returns
        -------
        lfloat
            Pixel scale in arcsec.
        """
        fpm = self.focal_plane_mask()
        disp = self.disperser()
        filt = self.filter_name()
        return getattr(config_dict.get((fpm, disp, filt)), 'pixscale', None)

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
        return lnrs_mode_map.get(self.phu.get('LNRS'), 'Unknown')

    @returns_list
    @astro_data_descriptor
    def read_noise(self):
        """
        Returns the detector read noise, in electrons.
        A lookup table is used.  The read noise depends on the gain setting
        and is affected by the number of coadds and non-destructive pairs.
        A list is returned unless called on a single-extension slice.

        Returns
        -------
        list/float
            Detector read noise in electrons.

        """
        rn = self._from_biaspwr("readnoise")
        try:
            return float(rn * math.sqrt(self.coadds()) / math.sqrt(self.phu.get('LNRS')))
        except TypeError:
            return None

    @returns_list
    @use_keyword_if_prepared
    @astro_data_descriptor
    def saturation_level(self):
        """
        Returns the saturation level for the observation, in the same units as
        the data. A lookup table is used to get the full well value based on the
        bias voltage.

        Returns
        -------
        int/list
            Saturation level

        """
        try:
            return int(self._from_biaspwr("well") *
                       self._from_biaspwr("gain") * self.coadds() / self.gain()[0])
        except TypeError:
            return None
