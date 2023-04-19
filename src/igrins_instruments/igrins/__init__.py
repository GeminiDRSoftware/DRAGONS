__all__ = ['AstroDataIGRINS', 'AstroDataIGRINS2']

from astrodata import factory
from .adclass import _AstroDataIGRINS, AstroDataIGRINS, AstroDataIGRINS2

factory.addClass(AstroDataIGRINS)
factory.addClass(AstroDataIGRINS2)



