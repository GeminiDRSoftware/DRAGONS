import re
import datetime
import dateutil

from astrodata import factory, simple_descriptor_mapping, keyword
from .generic import AstroDataGemini

# NOTE: Temporary functions for test. gempy imports astrodata and
#       won't work with this implementation
from .gmu import *

class AstroDataNifs(AstroDataGemini):
    @staticmethod
    def _matches_data(data_provider):
        return data_provider.phu.get('INSTRUME').upper() == 'NIFS'


factory.addClass(AstroDataNifs)
