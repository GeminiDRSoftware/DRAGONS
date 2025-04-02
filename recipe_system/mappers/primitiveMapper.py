#
#                                                                        DRAGONS
#
#                                                     mappers.primitiveMapper.py
# ------------------------------------------------------------------------------
import pkgutil

from importlib import import_module
from inspect import isclass

from gempy.utils import logutils

from .baseMapper import Mapper

from ..utils.mapper_utils import dotpath
from ..utils.errors import PrimitivesNotFound
# ------------------------------------------------------------------------------
log = logutils.get_logger(__name__)
# ------------------------------------------------------------------------------
class PrimitiveMapper(Mapper):
    """
    Retrieve the appropriate primitive class for a dataset, using all
    defined defaults:

    >>> ad = astrodata.from_file(<fitsfile>)
    >>> dtags = set(list(ad.tags)[:])
    >>> instpkg = ad.instrument(generic=True).lower()
    >>> pm = PrimitiveMapper(dtags, instpkg)
    >>> pclass = pm.get_applicable_primitives()
    >>> pclass
    <class 'primitives_IMAGE.PrimitivesIMAGE'>

    """
    def get_applicable_primitives(self):
        tag_match, primitive_class = self._retrieve_primitive_set()
        if primitive_class is None:
            raise PrimitivesNotFound("No qualified primitive set could be found")

        return primitive_class

    # --------------------------------------------------------------------------
    # Primtive search cascade
    def _retrieve_primitive_set(self):
        """
        Start of the primitive class search cascade.

        Parameters
        ----------
        <void>

        Returns
        -------
        <tuple> : (set, <class>)
                  Tuple including the best tag set match and the primitive class
                  that best matched.

        """
        matched_set = (set(), None)
        for pclass in self._get_tagged_primitives():
            if pclass is None:
                break
            if pclass.tagset is None:
                continue

            if self.tags.issuperset(pclass.tagset):
                isect = pclass.tagset
                l1 = len(isect)
                l2 = len(matched_set[0])
                matched_set = (isect, pclass) if l1 > l2 else matched_set
            else:
                continue

        return matched_set

    def _get_tagged_primitives(self):
        try:
            loaded_pkg = import_module(self.dotpackage)
        except ModuleNotFoundError:
            log.error("Uncaught exception", exc_info=True)
            yield None
            return
        for pkgpath, pkg in self._generate_primitive_modules(loaded_pkg):
            lmod = import_module(dotpath(self.dotpackage, pkg))
            for atrname in dir(lmod):
                if atrname.startswith('_'):        # no prive, no magic
                    continue

                atr = getattr(lmod, atrname)
                if isclass(atr) and hasattr(atr, 'tagset'):
                    yield atr

    def _generate_primitive_modules(self, pkg):
        ppath = pkg.__path__[0]
        #pkg_importer = pkgutil.ImpImporter(ppath)
        #for pkgname, ispkg in pkg_importer.iter_modules():
            # if ispkg:
            #     continue
            # else:
            #     yield (pkg_importer.path, pkgname)
        for modinfo in pkgutil.iter_modules([ppath]):
            if modinfo.ispkg:
                continue
            else:
                yield (modinfo.module_finder.path, modinfo.name)
