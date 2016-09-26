__all__ = ['AstroDataNiri']

from astrodata import factory
from ..gemini import addInstrumentFilterWavelengths
from .adclass import AstroDataNiri
from .lookup import filter_wavelengths

factory.addClass(AstroDataNiri)
addInstrumentFilterWavelengths('NIRI', filter_wavelengths)
