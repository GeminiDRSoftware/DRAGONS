__all__ = ['AstroDataIGRINS', 'AstroDataIGRINS']

from astrodata import factory
from gemini_instruments.gemini import addInstrumentFilterWavelengths
from .adclass import AstroDataIGRINS 
from .lookup import filter_wavelengths

factory.addClass(AstroDataIGRINS)

addInstrumentFilterWavelengths('fox', filter_wavelengths)


