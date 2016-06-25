import math

from astrodata import astro_data_tag, simple_descriptor_mapping, keyword
from ..generic import AstroDataGemini
from .lookups import constants_by_bias, config_dict, lnrs_mode_map

# NOTE: Temporary functions for test. gempy imports astrodata and
#       won't work with this implementation
from ..gmu import *

@simple_descriptor_mapping(
    bias = keyword("BIASPWR"),
    camera = keyword("INSTRUME"),
    central_wavelength = keyword("GRATWAVE"),
    focal_plane_mask = keyword("APERTURE"),
    lnrs = keyword("LNRS"),
    observation_epoch = keyword("EPOCH")
)
class AstroDataNifs(AstroDataGemini):
    @staticmethod
    def _matches_data(data_provider):
        return data_provider.phu.get('INSTRUME').upper() == 'NIFS'

    @astro_data_tag
    def _tag_image(self):
        if self.phu.FLIP == 'In':
            return (set(['IMAGE']), ())

    @astro_data_tag
    def _tag_ronchi(self):
        req = self.phu.OBSTYPE, self.phu.APERTURE
        if req == ('FLAT', 'Ronchi_Screen_G5615'):
            return (set(['RONCHI']), ())

    @astro_data_tag
    def _tag_spect(self):
        if self.phu.FLIP == 'Out':
            return (set(['SPECT']), ())

    def disperser(self, stripID=False, pretty=False):
        disp = str(self.phu.GRATING)
        if stripID or pretty:
            return removeComponentID(disp)
        return disp

    def filter_name(self, stripID=False, pretty=False):
        filt = str(self.phu.FILTER)
        if stripID or pretty:
            filt = removeComponentID(filt)

        if filt == "Blocked":
            return "blank"
        return filt

    def _from_biaspwr(self, constant_name):
        bias_volt = self.bias()

        for bias, constants in constants_by_bias.items():
            if abs(bias - bias_volt) < 0.1:
                return getattr(constants, constant_name)

        raise KeyError("The bias value for this image doesn't match any on the lookup table")

    def gain(self):
        return self._from_biaspwr("gain")

    def non_linear_level(self):
        saturation_level = self.saturation_level()
        corrected = 'NONLINCR' in self.phu

        linearlimit = self._from_biaspwr("linearlimit" if corrected else "nonlinearlimit")
        return int(saturation_level * linearlimit)

    def pixel_scale(self):
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

    def read_mode(self):
        # NOTE: The original read_mode descriptor obtains the bias voltage
        #       value, but then it does NOTHING with it. I'll just skip it.

        return lnrs_mode_map.get(self.lnrs(), 'Invalid')

    def read_noise(self):
        rn = self._from_biaspwr("readnoise")
        return float(rn * math.sqrt(self.coadds()) / math.sqrt(self.lnrs()))

    def saturation_level(self):
        well = self._from_biaspwr("well")
        return int(well * self.coadds())
