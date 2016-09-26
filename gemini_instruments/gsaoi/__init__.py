__all__ = ['AstroDataGsaoi']

from astrodata import factory
from ..gemini import addInstrumentFilterWavelengths
from .adclass import AstroDataGsaoi
from .lookup import filter_wavelengths

factory.addClass(AstroDataGsaoi)
addInstrumentFilterWavelengths('GSAOI', filter_wavelengths)
