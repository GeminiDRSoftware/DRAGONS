from astrodata import version
__version__ = version()

__all__ = ['reduce_data']

from .reduction.coreReduce import reduce_data
