__all__ = ['AstroData', 'AstroDataError', 'astro_data_tag', 'TagSet', 'factory', 'keyword']

from .core import *
from .fits import AstroDataFits, FitsLoader, KeywordCallableWrapper
from .mynddata.nduncertainty import VarUncertainty

keyword = KeywordCallableWrapper

class AstroDataFactory(object):
    def __init__(self):
        self._registry = set()

    def addClass(self, cls):
        """
        Add a new class to the AstroDataFactory registry. It will be used when
        instantiating an AstroData class for a FITS file.
        """
        if not hasattr(cls, '_matches_data'):
            raise AstroDataError("Class '{}' contains no '_matches_data' method".format(cls.__name__))
        self._registry.add(cls)

    def _getAstroData(self, data_provider):
        """
        Searches the internal registry for an AstroData derivative matching
        the metadata in the HDUList that we're passed.

        Returns an instantiated object, or raises AstroDataError if it was
        not possible to find a match.
        """
        candidates = [x for x in self._registry if x._matches_data(data_provider)]

        # For every candidate in the list, remove the ones that are base classes
        # for other candidates. That way we keep only the more specific ones.
        final_candidates = []
        for cnd in candidates:
            if any(cnd in x.mro() for x in candidates if x != cnd):
                continue
            final_candidates.append(cnd)

        if len(final_candidates) > 1:
            raise AstroDataError("More than one class is candidate for this dataset")
        elif not final_candidates:
            raise AstroDataError("No class matches this dataset")

        return final_candidates[0](data_provider)

    def getAstroData(self, source):
        """
        Takes either a string (with the path to a file) or an HDUList as input, and
        tries to return an AstroData instance.

        It will raise exceptions if the file is not found, or if there is no match
        for the HDUList, among the registered AstroData classes.
        """

        # NOTE: This is not Python3 ready, but don't worry about it now...
        if isinstance(source, StringTypes):
            return self._getAstroData(FitsLoader.from_path(source))
        else:
            # NOTE: This should be tested against the appropriate class.
            return self._getAstroData(FitsLoader.from_hdulist(source))

factory = AstroDataFactory()
# Let's make sure that there's at least one class that matches the data
# (if we're dealing with a FITS file)
factory.addClass(AstroDataFits)

open = factory.getAstroData
