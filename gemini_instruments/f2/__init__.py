__all__ = ['AstroDataF2']

from astrodata import factory
from ..gemini import addInstrumentFilterWavelengths
from .adclass import AstroDataF2
from .lookup import filter_wavelengths

factory.add_class(AstroDataF2)
addInstrumentFilterWavelengths('F2', filter_wavelengths)
