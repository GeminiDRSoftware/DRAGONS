from geminidr import _version
__version__ = _version.version()

__all__ = ['reduce_data']

from .reduction.coreReduce import reduce_data
