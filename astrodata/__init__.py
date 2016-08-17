__all__ = ['AstroData', 'AstroDataError', 'astro_data_tag', 'TagSet', 'keyword', 'open', 'create']

from .core import *
from .mynddata.nduncertainty import VarUncertainty
from .fits import AstroDataFits, KeywordCallableWrapper

from .factory import AstroDataFactory

keyword = KeywordCallableWrapper

factory = AstroDataFactory()
# Let's make sure that there's at least one class that matches the data
# (if we're dealing with a FITS file)
factory.addClass(AstroDataFits)

open = factory.getAstroData
create = factory.createFromScratch
