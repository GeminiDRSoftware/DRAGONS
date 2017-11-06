"""
The geminidr package provides the base classes for all parameter and primitive
classes in the geminidr package.

This module provides the caches library to primitives, but currently, only
Bookkeeping uses the cache directly (addToList). 

This module now provides the Calibrations class, formerly part of cal_service.
Calibrations() also uses the caches functions, which are now directly available
here.

E.g.,
>>> from geminidr import ParametersBASE
>>> from geminidr import PrimitivesBASE

"""

# ------------------------------------------------------------------------------
import os
import pickle
import warnings
from inspect import stack

from astropy.io.fits.verify import VerifyWarning

from gempy.utils import logutils
# new system imports - 10-06-2016 kra
# NOTE: imports of these and other tables will be moving around ...
from .gemini.lookups import keyword_comments
from .gemini.lookups import timestamp_keywords
from .gemini.lookups.source_detection import sextractor_dict

from recipe_system.cal_service import calurl_dict
from recipe_system.utils.decorators import parameter_override

# ------------------------------ caches ----------------------------------------
# Formerly in cal_service/caches.py
#
# GLOBAL/CONSTANTS (could be exported to config file)
CALS = "calibrations"

# [caches]
caches = {
    'reducecache'  : '.reducecache',
    'calibrations' : CALS
    }

calindfile = os.path.join('.', caches['reducecache'], "calindex.pkl")
stkindfile = os.path.join('.', caches['reducecache'], "stkindex.pkl")

def set_caches():
    cachedict = {}
    for cachename, cachedir in caches.items():
        if not os.path.exists(cachedir):
            os.makedirs(cachedir)
        cachedict.update({cachename:cachedir})
    return cachedict

def load_cache(cachefile):
    if os.path.exists(cachefile):
        return pickle.load(open(cachefile, 'r'))
    else:
        return {}

def save_cache(object, cachefile):
    pickle.dump(object, open(cachefile, 'wb'))
    return
# ------------------------- END caches------------------------------------------
class Calibrations(dict):
    def __init__(self, calindfile, user_cals={}, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self._calindfile = calindfile
        self.update(load_cache(self._calindfile))
        self._usercals = user_cals or {}                 # Handle user_cals=None

    def __getitem__(self, key):
        return self._get_cal(*key)

    def __setitem__(self, key, val):
        self._add_cal(key, val)
        return

    def __delitem__(self, key):
        # Cope with malformed keys
        try:
            self.pop((key[0].calibration_key(), key[1]), None)
        except (TypeError, IndexError):
            pass

    def _add_cal(self, key, val):
        # Munge the key from (ad, caltype) to (ad.calibration_key, caltype)
        key = (key[0].calibration_key(), key[1])
        self.update({key: val})
        self.cache_to_disk()
        return

    def _get_cal(self, ad, caltype):
        key = (ad.calibration_key(), caltype)
        if key in self._usercals:
            return self._usercals[key]
        calfile = self.get(key)
        return calfile

    def cache_to_disk(self):
        save_cache(self, self._calindfile)
        return
# ------------------------------------------------------------------------------
class ParametersBASE(object):
    """
    Base class for all Gemini package parameter sets.

    Most other parameter classes will be separate from their
    matching primitives class. Here, we incorporate the parameter
    class, ParametersBASE, into the mod.

    """
    pass

# ------------------------------------------------------------------------------
@parameter_override
class PrimitivesBASE(object):
    """
    This is the base class for all of primitives classes for the geminidr
    primitive sets. __init__.py provides, or should provide, all attributes
    needed by subclasses.

    Parameters
    ----------
    adinputs: <list> A list of astrodata objects

    mode:     <str>  Operational Mode, one of 'sq', 'qa', 'ql'.

    upload:   <list> A list of products to upload to fitsstore.
                     QA metrics uploaded if 'metrics' in upload.
              E.g.,

                  upload = ['metrics', ['calibs', ... ]]

    """
    tagset = None

    def __init__(self, adinputs, mode='sq', ucals=None, uparms=None, upload=None):
        self.streams          = {'main': adinputs}
        self.mode             = mode
        self.parameters       = ParametersBASE
        self.log              = logutils.get_logger(__name__)
        self._upload          = upload
        self.user_params      = uparms if uparms else {}
        self.calurl_dict      = calurl_dict.calurl_dict
        self.timestamp_keys   = timestamp_keywords.timestamp_keys
        self.keyword_comments = keyword_comments.keyword_comments
        self.sx_dict          = sextractor_dict.sx_dict.copy()
        # Prepend paths to SExtractor input files now
        self.sx_dict.update({k:
                os.path.join(os.path.dirname(sextractor_dict.__file__), v)
                for k,v in self.sx_dict.items()})

        self.cachedict        = set_caches()
        self.calibrations     = Calibrations(calindfile, user_cals=ucals)
        self.stacks           = load_cache(stkindfile)

        # This lambda will return the name of the current caller.
        self.myself           = lambda: stack()[1][3]

        warnings.simplefilter('ignore', category=VerifyWarning)

    @property
    def upload(self):
        return self._upload

    @upload.setter
    def upload(self, upl):
        if upl is None:
            self._upload = None
        elif isinstance(upl, str):
            self._upload = [seg.lower().strip() for seg in upl.split(',')]
        elif isinstance(upl, list):
            self._upload = upl
        return
