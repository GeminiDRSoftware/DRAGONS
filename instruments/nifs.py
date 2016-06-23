import math
import datetime
import dateutil
import re

from astrodata import factory, simple_descriptor_mapping, keyword
from collections import namedtuple
from .generic import AstroDataGemini

# NOTE: Temporary functions for test. gempy imports astrodata and
#       won't work with this implementation
from .gmu import *

# Data structures used by this module

Constants = namedtuple("Constants", "readnoise gain well linearlimit coeff1 coeff2 coeff3 nonlinearlimit")
Config = namedtuple("Config", "mdf offsetsection pixscale mode")

# Taken from nifs$data/nifsarray.fits
# Dictionary key is the bias
# Dictionary values are in the following order:
# readnoise, gain, well, linearlimit, coeff1, coeff2, coeff3,
# nonlinearlimit

# The index is bias value
constants_by_bias = {
    3. : Constants(readnoise=6.3, gain=2.4, well=50000., linearlimit=0.9, nonlinearlimit=0.9,
                   coeff1=1.017475, coeff2=0.244937, coeff3=1.019483)
}

common_config = Config(mdf="nifs$data/nifs-mdf.fits" , offsetsection="[900:1024,*]" ,
                       pixscale=0.043 , mode="IFU")

config_dict = {
    # Dictionary keys are in the following order:
    # fpmask, grating, filter
    ( "3.0_Mask_G5610" , "Mirror_G5601" , "HK_G0603" ) : common_config,
    ( "Ronchi_Screen_G5615" , "Mirror_G5601" , "HK_G0603" ) : common_config,
    ( "0.1_Hole_G5611" , "Mirror_G5601" , "HK_G0603" ) : common_config,
    ( "0.2_Hole_G5612" , "Mirror_G5601" , "HK_G0603" ) : common_config,
    ( "0.2_Hole_Array_G5613" , "Mirror_G5601" , "HK_G0603" ) : common_config,
    ( "0.2_Slit_G5614" , "Mirror_G5601" , "HK_G0603" ) : common_config,
    ( "0.1_Occ_Disc_G5616" , "Mirror_G5601" , "HK_G0603" ) : common_config,
    ( "0.2_Occ_Disc_G5617" , "Mirror_G5601" , "HK_G0603" ) : common_config,
    ( "0.5_Occ_Disc_G5618" , "Mirror_G5601" , "HK_G0603" ) : common_config,
    ( "KG3_ND_Filter_G5619" , "Mirror_G5601" , "HK_G0603" ) : common_config,
    ( "KG5_ND_Filter_G5620" , "Mirror_G5601" , "HK_G0603" ) : common_config,
    ( "Blocked_G5621" , "Mirror_G5601" , "HK_G0603" ) : common_config,
    ( "3.0_Mask_G5610" , "H_G5604" , "HK_G0603" ) : common_config,
    ( "Ronchi_Screen_G5615" , "H_G5604" , "HK_G0603" ) : common_config,
    ( "0.1_Hole_G5611" , "H_G5604" , "HK_G0603" ) : common_config,
    ( "0.2_Hole_G5612" , "H_G5604" , "HK_G0603" ) : common_config,
    ( "0.2_Hole_Array_G5613" , "H_G5604" , "HK_G0603" ) : common_config,
    ( "0.2_Slit_G5614" , "H_G5604" , "HK_G0603" ) : common_config,
    ( "0.1_Occ_Disc_G5616" , "H_G5604" , "HK_G0603" ) : common_config,
    ( "0.2_Occ_Disc_G5617" , "H_G5604" , "HK_G0603" ) : common_config,
    ( "0.5_Occ_Disc_G5618" , "H_G5604" , "HK_G0603" ) : common_config,
    ( "KG3_ND_Filter_G5619" , "H_G5604" , "HK_G0603" ) : common_config,
    ( "KG5_ND_Filter_G5620" , "H_G5604" , "HK_G0603" ) : common_config,
    ( "Blocked_G5621" , "H_G5604" , "HK_G0603" ) : common_config,
    ( "3.0_Mask_G5610" , "K_G5605" , "HK_G0603" ) : common_config,
    ( "Ronchi_Screen_G5615" , "K_G5605" , "HK_G0603" ) : common_config,
    ( "0.1_Hole_G5611" , "K_G5605" , "HK_G0603" ) : common_config,
    ( "0.2_Hole_G5612" , "K_G5605" , "HK_G0603" ) : common_config,
    ( "0.2_Hole_Array_G5613" , "K_G5605" , "HK_G0603" ) : common_config,
    ( "0.2_Slit_G5614" , "K_G5605" , "HK_G0603" ) : common_config,
    ( "0.1_Occ_Disc_G5616" , "K_G5605" , "HK_G0603" ) : common_config,
    ( "0.2_Occ_Disc_G5617" , "K_G5605" , "HK_G0603" ) : common_config,
    ( "0.5_Occ_Disc_G5618" , "K_G5605" , "HK_G0603" ) : common_config,
    ( "KG3_ND_Filter_G5619" , "K_G5605" , "HK_G0603" ) : common_config,
    ( "KG5_ND_Filter_G5620" , "K_G5605" , "HK_G0603" ) : common_config,
    ( "Blocked_G5621" , "K_G5605" , "HK_G0603" ) : common_config,
    ( "3.0_Mask_G5610" , "K_Short_G5606" , "HK_G0603" ) : common_config,
    ( "Ronchi_Screen_G5615" , "K_Short_G5606" , "HK_G0603" ) : common_config,
    ( "0.1_Hole_G5611" , "K_Short_G5606" , "HK_G0603" ) : common_config,
    ( "0.2_Hole_G5612" , "K_Short_G5606" , "HK_G0603" ) : common_config,
    ( "0.2_Hole_Array_G5613" , "K_Short_G5606" , "HK_G0603" ) : common_config,
    ( "0.2_Slit_G5614" , "K_Short_G5606" , "HK_G0603" ) : common_config,
    ( "0.1_Occ_Disc_G5616" , "K_Short_G5606" , "HK_G0603" ) : common_config,
    ( "0.2_Occ_Disc_G5617" , "K_Short_G5606" , "HK_G0603" ) : common_config,
    ( "0.5_Occ_Disc_G5618" , "K_Short_G5606" , "HK_G0603" ) : common_config,
    ( "KG3_ND_Filter_G5619" , "K_Short_G5606" , "HK_G0603" ) : common_config,
    ( "KG5_ND_Filter_G5620" , "K_Short_G5606" , "HK_G0603" ) : common_config,
    ( "Blocked_G5621" , "K_Short_G5606" , "HK_G0603" ) : common_config,
    ( "3.0_Mask_G5610" , "K_Long_G5607" , "HK_G0603" ) : common_config,
    ( "Ronchi_Screen_G5615" , "K_Long_G5607" , "HK_G0603" ) : common_config,
    ( "0.1_Hole_G5611" , "K_Long_G5607" , "HK_G0603" ) : common_config,
    ( "0.2_Hole_G5612" , "K_Long_G5607" , "HK_G0603" ) : common_config,
    ( "0.2_Hole_Array_G5613" , "K_Long_G5607" , "HK_G0603" ) : common_config,
    ( "0.2_Slit_G5614" , "K_Long_G5607" , "HK_G0603" ) : common_config,
    ( "0.1_Occ_Disc_G5616" , "K_Long_G5607" , "HK_G0603" ) : common_config,
    ( "0.2_Occ_Disc_G5617" , "K_Long_G5607" , "HK_G0603" ) : common_config,
    ( "0.5_Occ_Disc_G5618" , "K_Long_G5607" , "HK_G0603" ) : common_config,
    ( "KG3_ND_Filter_G5619" , "K_Long_G5607" , "HK_G0603" ) : common_config,
    ( "KG5_ND_Filter_G5620" , "K_Long_G5607" , "HK_G0603" ) : common_config,
    ( "Blocked_G5621" , "K_Long_G5607" , "HK_G0603" ) : common_config,
    ( "3.0_Mask_G5610" , "Mirror_G5601" , "HK+WireGrid_G0604" ) : common_config,
    ( "Ronchi_Screen_G5615" , "Mirror_G5601" , "HK+WireGrid_G0604" ) : common_config,
    ( "0.1_Hole_G5611" , "Mirror_G5601" , "HK+WireGrid_G0604" ) : common_config,
    ( "0.2_Hole_G5612" , "Mirror_G5601" , "HK+WireGrid_G0604" ) : common_config,
    ( "0.2_Hole_Array_G5613" , "Mirror_G5601" , "HK+WireGrid_G0604" ) : common_config,
    ( "0.2_Slit_G5614" , "Mirror_G5601" , "HK+WireGrid_G0604" ) : common_config,
    ( "0.1_Occ_Disc_G5616" , "Mirror_G5601" , "HK+WireGrid_G0604" ) : common_config,
    ( "0.2_Occ_Disc_G5617" , "Mirror_G5601" , "HK+WireGrid_G0604" ) : common_config,
    ( "0.5_Occ_Disc_G5618" , "Mirror_G5601" , "HK+WireGrid_G0604" ) : common_config,
    ( "KG3_ND_Filter_G5619" , "Mirror_G5601" , "HK+WireGrid_G0604" ) : common_config,
    ( "KG5_ND_Filter_G5620" , "Mirror_G5601" , "HK+WireGrid_G0604" ) : common_config,
    ( "Blocked_G5621" , "Mirror_G5601" , "HK+WireGrid_G0604" ) : common_config,
    ( "3.0_Mask_G5610" , "K_G5605" , "HK+WireGrid_G0604" ) : common_config,
    ( "Ronchi_Screen_G5615" , "K_G5605" , "HK+WireGrid_G0604" ) : common_config,
    ( "0.1_Hole_G5611" , "K_G5605" , "HK+WireGrid_G0604" ) : common_config,
    ( "0.2_Hole_G5612" , "K_G5605" , "HK+WireGrid_G0604" ) : common_config,
    ( "0.2_Hole_Array_G5613" , "K_G5605" , "HK+WireGrid_G0604" ) : common_config,
    ( "0.2_Slit_G5614" , "K_G5605" , "HK+WireGrid_G0604" ) : common_config,
    ( "0.1_Occ_Disc_G5616" , "K_G5605" , "HK+WireGrid_G0604" ) : common_config,
    ( "0.2_Occ_Disc_G5617" , "K_G5605" , "HK+WireGrid_G0604" ) : common_config,
    ( "0.5_Occ_Disc_G5618" , "K_G5605" , "HK+WireGrid_G0604" ) : common_config,
    ( "KG3_ND_Filter_G5619" , "K_G5605" , "HK+WireGrid_G0604" ) : common_config,
    ( "KG5_ND_Filter_G5620" , "K_G5605" , "HK+WireGrid_G0604" ) : common_config,
    ( "Blocked_G5621" , "K_G5605" , "HK+WireGrid_G0604" ) : common_config,
    ( "3.0_Mask_G5610" , "K_Short_G5606" , "HK+WireGrid_G0604" ) : common_config,
    ( "Ronchi_Screen_G5615" , "K_Short_G5606" , "HK+WireGrid_G0604" ) : common_config,
    ( "0.1_Hole_G5611" , "K_Short_G5606" , "HK+WireGrid_G0604" ) : common_config,
    ( "0.2_Hole_G5612" , "K_Short_G5606" , "HK+WireGrid_G0604" ) : common_config,
    ( "0.2_Hole_Array_G5613" , "K_Short_G5606" , "HK+WireGrid_G0604" ) : common_config,
    ( "0.2_Slit_G5614" , "K_Short_G5606" , "HK+WireGrid_G0604" ) : common_config,
    ( "0.1_Occ_Disc_G5616" , "K_Short_G5606" , "HK+WireGrid_G0604" ) : common_config,
    ( "0.2_Occ_Disc_G5617" , "K_Short_G5606" , "HK+WireGrid_G0604" ) : common_config,
    ( "0.5_Occ_Disc_G5618" , "K_Short_G5606" , "HK+WireGrid_G0604" ) : common_config,
    ( "KG3_ND_Filter_G5619" , "K_Short_G5606" , "HK+WireGrid_G0604" ) : common_config,
    ( "KG5_ND_Filter_G5620" , "K_Short_G5606" , "HK+WireGrid_G0604" ) : common_config,
    ( "Blocked_G5621" , "K_Short_G5606" , "HK+WireGrid_G0604" ) : common_config,
    ( "3.0_Mask_G5610" , "K_Long_G5607" , "HK+WireGrid_G0604" ) : common_config,
    ( "Ronchi_Screen_G5615" , "K_Long_G5607" , "HK+WireGrid_G0604" ) : common_config,
    ( "0.1_Hole_G5611" , "K_Long_G5607" , "HK+WireGrid_G0604" ) : common_config,
    ( "0.2_Hole_G5612" , "K_Long_G5607" , "HK+WireGrid_G0604" ) : common_config,
    ( "0.2_Hole_Array_G5613" , "K_Long_G5607" , "HK+WireGrid_G0604" ) : common_config,
    ( "0.2_Slit_G5614" , "K_Long_G5607" , "HK+WireGrid_G0604" ) : common_config,
    ( "0.1_Occ_Disc_G5616" , "K_Long_G5607" , "HK+WireGrid_G0604" ) : common_config,
    ( "0.2_Occ_Disc_G5617" , "K_Long_G5607" , "HK+WireGrid_G0604" ) : common_config,
    ( "0.5_Occ_Disc_G5618" , "K_Long_G5607" , "HK+WireGrid_G0604" ) : common_config,
    ( "KG3_ND_Filter_G5619" , "K_Long_G5607" , "HK+WireGrid_G0604" ) : common_config,
    ( "KG5_ND_Filter_G5620" , "K_Long_G5607" , "HK+WireGrid_G0604" ) : common_config,
    ( "Blocked_G5621" , "K_Long_G5607" , "HK+WireGrid_G0604" ) : common_config,
    ( "3.0_Mask_G5610" , "Mirror_G5601" , "JH_G0602" ) : common_config,
    ( "Ronchi_Screen_G5615" , "Mirror_G5601" , "JH_G0602" ) : common_config,
    ( "0.1_Hole_G5611" , "Mirror_G5601" , "JH_G0602" ) : common_config,
    ( "0.2_Hole_G5612" , "Mirror_G5601" , "JH_G0602" ) : common_config,
    ( "0.2_Hole_Array_G5613" , "Mirror_G5601" , "JH_G0602" ) : common_config,
    ( "0.2_Slit_G5614" , "Mirror_G5601" , "JH_G0602" ) : common_config,
    ( "0.1_Occ_Disc_G5616" , "Mirror_G5601" , "JH_G0602" ) : common_config,
    ( "0.2_Occ_Disc_G5617" , "Mirror_G5601" , "JH_G0602" ) : common_config,
    ( "0.5_Occ_Disc_G5618" , "Mirror_G5601" , "JH_G0602" ) : common_config,
    ( "KG3_ND_Filter_G5619" , "Mirror_G5601" , "JH_G0602" ) : common_config,
    ( "KG5_ND_Filter_G5620" , "Mirror_G5601" , "JH_G0602" ) : common_config,
    ( "Blocked_G5621" , "Mirror_G5601" , "JH_G0602" ) : common_config,
    ( "3.0_Mask_G5610" , "J_G5603" , "JH_G0602" ) : common_config,
    ( "Ronchi_Screen_G5615" , "J_G5603" , "JH_G0602" ) : common_config,
    ( "0.1_Hole_G5611" , "J_G5603" , "JH_G0602" ) : common_config,
    ( "0.2_Hole_G5612" , "J_G5603" , "JH_G0602" ) : common_config,
    ( "0.2_Hole_Array_G5613" , "J_G5603" , "JH_G0602" ) : common_config,
    ( "0.2_Slit_G5614" , "J_G5603" , "JH_G0602" ) : common_config,
    ( "0.1_Occ_Disc_G5616" , "J_G5603" , "JH_G0602" ) : common_config,
    ( "0.2_Occ_Disc_G5617" , "J_G5603" , "JH_G0602" ) : common_config,
    ( "0.5_Occ_Disc_G5618" , "J_G5603" , "JH_G0602" ) : common_config,
    ( "KG3_ND_Filter_G5619" , "J_G5603" , "JH_G0602" ) : common_config,
    ( "KG5_ND_Filter_G5620" , "J_G5603" , "JH_G0602" ) : common_config,
    ( "Blocked_G5621" , "J_G5603" , "JH_G0602" ) : common_config,
    ( "3.0_Mask_G5610" , "H_G5604" , "JH_G0602" ) : common_config,
    ( "Ronchi_Screen_G5615" , "H_G5604" , "JH_G0602" ) : common_config,
    ( "0.1_Hole_G5611" , "H_G5604" , "JH_G0602" ) : common_config,
    ( "0.2_Hole_G5612" , "H_G5604" , "JH_G0602" ) : common_config,
    ( "0.2_Hole_Array_G5613" , "H_G5604" , "JH_G0602" ) : common_config,
    ( "0.2_Slit_G5614" , "H_G5604" , "JH_G0602" ) : common_config,
    ( "0.1_Occ_Disc_G5616" , "H_G5604" , "JH_G0602" ) : common_config,
    ( "0.2_Occ_Disc_G5617" , "H_G5604" , "JH_G0602" ) : common_config,
    ( "0.5_Occ_Disc_G5618" , "H_G5604" , "JH_G0602" ) : common_config,
    ( "KG3_ND_Filter_G5619" , "H_G5604" , "JH_G0602" ) : common_config,
    ( "KG5_ND_Filter_G5620" , "H_G5604" , "JH_G0602" ) : common_config,
    ( "Blocked_G5621" , "H_G5604" , "JH_G0602" ) : common_config,
    ( "3.0_Mask_G5610" , "Mirror_G5601" , "ZJ_G0601" ) : common_config,
    ( "Ronchi_Screen_G5615" , "Mirror_G5601" , "ZJ_G0601" ) : common_config,
    ( "0.1_Hole_G5611" , "Mirror_G5601" , "ZJ_G0601" ) : common_config,
    ( "0.2_Hole_G5612" , "Mirror_G5601" , "ZJ_G0601" ) : common_config,
    ( "0.2_Hole_Array_G5613" , "Mirror_G5601" , "ZJ_G0601" ) : common_config,
    ( "0.2_Slit_G5614" , "Mirror_G5601" , "ZJ_G0601" ) : common_config,
    ( "0.1_Occ_Disc_G5616" , "Mirror_G5601" , "ZJ_G0601" ) : common_config,
    ( "0.2_Occ_Disc_G5617" , "Mirror_G5601" , "ZJ_G0601" ) : common_config,
    ( "0.5_Occ_Disc_G5618" , "Mirror_G5601" , "ZJ_G0601" ) : common_config,
    ( "KG3_ND_Filter_G5619" , "Mirror_G5601" , "ZJ_G0601" ) : common_config,
    ( "KG5_ND_Filter_G5620" , "Mirror_G5601" , "ZJ_G0601" ) : common_config,
    ( "Blocked_G5621" , "Mirror_G5601" , "ZJ_G0601" ) : common_config,
    ( "3.0_Mask_G5610" , "J_G5603" , "ZJ_G0601" ) : common_config,
    ( "Ronchi_Screen_G5615" , "J_G5603" , "ZJ_G0601" ) : common_config,
    ( "0.1_Hole_G5611" , "J_G5603" , "ZJ_G0601" ) : common_config,
    ( "0.2_Hole_G5612" , "J_G5603" , "ZJ_G0601" ) : common_config,
    ( "0.2_Hole_Array_G5613" , "J_G5603" , "ZJ_G0601" ) : common_config,
    ( "0.2_Slit_G5614" , "J_G5603" , "ZJ_G0601" ) : common_config,
    ( "0.1_Occ_Disc_G5616" , "J_G5603" , "ZJ_G0601" ) : common_config,
    ( "0.2_Occ_Disc_G5617" , "J_G5603" , "ZJ_G0601" ) : common_config,
    ( "0.5_Occ_Disc_G5618" , "J_G5603" , "ZJ_G0601" ) : common_config,
    ( "KG3_ND_Filter_G5619" , "J_G5603" , "ZJ_G0601" ) : common_config,
    ( "KG5_ND_Filter_G5620" , "J_G5603" , "ZJ_G0601" ) : common_config,
    ( "Blocked_G5621" , "J_G5603" , "ZJ_G0601" ) : common_config,
    ( "3.0_Mask_G5610" , "Z_G5602" , "ZJ_G0601" ) : common_config,
    ( "Ronchi_Screen_G5615" , "Z_G5602" , "ZJ_G0601" ) : common_config,
    ( "0.1_Hole_G5611" , "Z_G5602" , "ZJ_G0601" ) : common_config,
    ( "0.2_Hole_G5612" , "Z_G5602" , "ZJ_G0601" ) : common_config,
    ( "0.2_Hole_Array_G5613" , "Z_G5602" , "ZJ_G0601" ) : common_config,
    ( "0.2_Slit_G5614" , "Z_G5602" , "ZJ_G0601" ) : common_config,
    ( "0.1_Occ_Disc_G5616" , "Z_G5602" , "ZJ_G0601" ) : common_config,
    ( "0.2_Occ_Disc_G5617" , "Z_G5602" , "ZJ_G0601" ) : common_config,
    ( "0.5_Occ_Disc_G5618" , "Z_G5602" , "ZJ_G0601" ) : common_config,
    ( "KG3_ND_Filter_G5619" , "Z_G5602" , "ZJ_G0601" ) : common_config,
    ( "KG5_ND_Filter_G5620" , "Z_G5602" , "ZJ_G0601" ) : common_config,
    ( "Blocked_G5621" , "Z_G5602" , "ZJ_G0601" ) : common_config
}

lnrs_mode_map = {
    1: "Bright Object",
    4: "Medium Object",
    16: "Faint Object"
}

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


factory.addClass(AstroDataNifs)
