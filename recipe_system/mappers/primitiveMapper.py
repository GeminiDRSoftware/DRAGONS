#
#                                                     mappers.primitiveMapper.py
# ------------------------------------------------------------------------------
import imp
import sys
import pkgutil
import importlib

from inspect import isclass

from .baseMapper import Mapper
from ..utils.errors import PrimitivesNotFound
# ------------------------------------------------------------------------------
class PrimitiveMapper(Mapper):
    """
    Retrieve the appropriate primitive class for a dataset, using all
    defined defaults:

    >>> ad = astrodata.open(<fitsfile>)
    >>> adinputs = [ad]
    >>> pm = PrimitiveMapper(adinputs)
    >>> p = pm.get_applicable_primitives()
    >>> p.__class__
    <class 'primitives_IMAGE.PrimitivesIMAGE'>

    """
    def get_applicable_primitives(self):
        tag_match, primitive_actual = self._retrieve_primitive_set()
        if primitive_actual is None:
            raise PrimitivesNotFound("No qualified primitive set could be found")

        return primitive_actual(self.adinputs, self.context, ucals=self.usercals,
                            uparms=self.userparams, upmetrics=self.upload_metrics)

    # --------------------------------------------------------------------------
    # Primtive search cascade
    def _retrieve_primitive_set(self):
        """
        :returns: tuple including the best tag set match and the primitive class
                  that provided the match.
        :rtype:   <tuple>, (set, class)

        """
        matched_set = (set([]), None)
        for pclass in self._get_tagged_primitives():
            if pclass.tagset is None:
                continue

            if self.tags.issuperset(pclass.tagset):
                isect = pclass.tagset
                matched_set = (isect, pclass) if isect > matched_set[0] else matched_set
            else:
                continue

        return matched_set

    def _get_tagged_primitives(self):
        loaded_pkg = self._package_loader(self.pkg)
        for pkgpath, pkg in self._generate_primitive_modules(loaded_pkg.__path__[0]):
            fd, path, descr = imp.find_module(pkg, [pkgpath])
            sys.path.insert(0, path)
            mod = importlib.import_module(pkg)
            for atrname in dir(mod):
                if atrname.startswith('_'):        # no prive, no magic
                    continue
                
                atr = getattr(mod, atrname)
                if isclass(atr) and hasattr(atr, 'tagset'):
                    yield atr

    def _generate_primitive_modules(self, pkg):
        pkg_importer = pkgutil.ImpImporter(pkg)
        for pkgname, ispkg in pkg_importer.iter_modules():
            if ispkg:
                continue
            else:
                yield (pkg_importer.path, pkgname)
