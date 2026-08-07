__all__ = ['AstroDataGemini', 'addInstrumentFilterWavelengths',
           'use_keyword_if_prepared', 'get_specphot_name']

from astrodata import factory
from .adclass import AstroDataGemini, use_keyword_if_prepared, get_specphot_name
from .lookup import filter_wavelengths

def addInstrumentFilterWavelengths(instrument, wl):
    update_dict = {
            (instrument, fltname): value
            for (fltname, value)
            in wl.items()
    }
    filter_wavelengths.update(update_dict)

factory.add_class(AstroDataGemini)
