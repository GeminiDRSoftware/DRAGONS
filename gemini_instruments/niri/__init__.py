__all__ = ['AstroDataNiri']

from astrodata import factory
from ..gemini import addInstrumentFilterWavelengths
from .adclass import AstroDataNiri
from .lookup import filter_wavelengths

factory.add_class(AstroDataNiri)
addInstrumentFilterWavelengths('NIRI', filter_wavelengths)
