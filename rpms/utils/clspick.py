import imp
import pkgutil

from inspect import isclass
from importlib import import_module


def _package_loader(pkgname):
    pfile, pkgpath, descr = imp.find_module(pkgname)
    loaded_pkg = imp.load_module(pkgname, pfile, pkgpath, descr)
    return loaded_pkg

def _generate_pkg_modules(pkg):
    pkg_importer = pkgutil.ImpImporter(pkg)
    for pkgname, ispkg in pkg_importer.iter_modules():
        if ispkg:
            continue
        else:
            yield (pkg_importer.path, pkgname)

def _get_tagged_classes(pkgname):
    loaded_pkg = package_loader(pkgname)
    for pkgpath, pkg in _generate_pkg_modules(loaded_pkg.__path__[0]):
        fd, path, descr = imp.find_module(pkg, [pkgpath])
        mod = imp.load_module(pkg, fd, path, descr)
        for atrname in dir(mod):
            if atrname.startswith('_'):        # no prive, no magic
                continue
                
            atr = getattr(mod, atrname)
            if isclass(atr) and hasattr(atr, 'tagset'):
                yield atr

def retrieve_primtive_set(adtags, pkgname):
    """
    Caller passes a set of AstroData tags and the instrument package name.

    :parameter adtags: set of AstroData tags on an 'ad' instance.
    :type adtags:      <type 'set'>
                       E.g., set(['GMOS', 'SIDEREAL', 'SPECT', 'GMOS_S', 'GEMINI'])

    :parameter pkgname: An instrument package under GeminiDR.
    :type pkgname:     <str>, E.g., "GMOS"

    :returns: tuple including the best tag set match and the primitive class
              that provided the match.
    :rtype: <tuple>, (set, class)

    """
    matched_set = (set([]), None)
    for pclass in _get_tagged_classes(pkgname):
        isection = adtags.intersection(pclass.tagset)
        matched_set = (isection, pclass) if isection > matched_set[0] else matched_set
    return matched_set
