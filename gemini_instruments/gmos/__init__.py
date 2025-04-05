__all__ = ['AstroDataGmos']

from astrodata import factory
from ..gemini import addInstrumentFilterWavelengths
from .adclass import AstroDataGmos
from .lookup import filter_wavelengths

factory.add_class(AstroDataGmos)
# Use the generic GMOS name for both GMOS-N and GMOS-S
addInstrumentFilterWavelengths('GMOS', filter_wavelengths)
