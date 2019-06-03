"""
    This package add another abstraction layer to astronomical data by parsing
    the information contained in the headers as attributes. To do so,
    one must subclass :class:`astrodata.AstroData` and add parse methods
    accordingly to the :class:`~astrodata.core.TagSet` received.
"""

__all__ = ['AstroData', 'AstroDataError', 'TagSet', 'NDAstroData',
           'astro_data_descriptor', 'astro_data_tag', 'keyword',
           'open', 'create', '__version__']


from .core import *
# TODO: Remove 'write' when there's nothing else using it
from .fits import AstroDataFits, KeywordCallableWrapper
from .fits import add_header_to_table

from .factory import AstroDataFactory

from .nddata import NDAstroData

from .version import version

__version__ = version()

keyword = KeywordCallableWrapper

factory = AstroDataFactory()
# Let's make sure that there's at least one class that matches the data
# (if we're dealing with a FITS file)
factory.addClass(AstroDataFits)

open = factory.getAstroData
create = factory.createFromScratch
