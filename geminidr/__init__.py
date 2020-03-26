"""
The geminidr package provides the base classes for all parameter and primitive
classes in the geminidr package.

This module provides the caches library to primitives, but currently, only
Bookkeeping uses the cache directly (addToList).

This module now provides the Calibrations class, formerly part of cal_service.
Calibrations() also uses the caches functions, which are now directly available
here.

E.g.,
>>> from geminidr import PrimitivesBASE

"""

# ------------------------------------------------------------------------------
import os
import gc
import pickle
import warnings
import weakref

from copy import deepcopy
from inspect import stack, isclass

from gempy.eti_core.eti import ETISubprocess
from gempy.library import config
from gempy.utils import logutils
from gempy import display

from astropy.io.fits.verify import VerifyWarning

# new system imports - 10-06-2016 kra
# NOTE: imports of these and other tables will be moving around ...
from .gemini.lookups import keyword_comments
from .gemini.lookups import timestamp_keywords
from .gemini.lookups.source_detection import sextractor_dict

from recipe_system.cal_service import calurl_dict
from recipe_system.utils.decorators import parameter_override

import atexit
# ------------------------------ caches ---------------------------------------
# Formerly in cal_service/caches.py
#
# GLOBAL/CONSTANTS (could be exported to config file)
CALS = "calibrations"

# [caches]
caches = {
    'reducecache': '.reducecache',
    'calibrations': CALS
}

calindfile = os.path.join('.', caches['reducecache'], "calindex.pkl")
stkindfile = os.path.join('.', caches['reducecache'], "stkindex.pkl")


def set_caches():
    cachedict = {name: cachedir for name, cachedir in caches.items()}
    for cachedir in cachedict.values():
        os.makedirs(cachedir, exist_ok=True)
    return cachedict


def load_cache(cachefile):
    if os.path.exists(cachefile):
        with open(cachefile, 'rb') as fp:
            return pickle.load(fp)
    else:
        return {}


def save_cache(obj, cachefile):
    with open(cachefile, 'wb') as fp:
        pickle.dump(obj, fp, protocol=2)

# ------------------------- END caches-----------------------------------------


class Calibrations:
    def __init__(self, calindfile, user_cals={}, *args, **kwargs):
        self._calindfile = calindfile
        self._dict = {}
        self._dict.update(load_cache(self._calindfile))
        self._usercals = user_cals or {}  # Handle user_cals=None

    def __getitem__(self, key):
        return self._get_cal(*key)

    def __setitem__(self, key, val):
        self._add_cal(key, val)

    def __delitem__(self, key):
        # Cope with malformed keys
        try:
            self._dict.pop((key[0].calibration_key(), key[1]), None)
        except (TypeError, IndexError):
            pass

    def _add_cal(self, key, val):
        # Munge the key from (ad, caltype) to (ad.calibration_key, caltype)
        key = (key[0].calibration_key(), key[1])
        self._dict.update({key: val})
        self.cache_to_disk()

    def _get_cal(self, ad, caltype):
        key = (ad.calibration_key(), caltype)
        if key in self._usercals:
            return self._usercals[key]
        calfile = self._dict.get(key)
        return calfile

    def cache_to_disk(self):
        save_cache(self._dict, self._calindfile)

# ------------------------------------------------------------------------------


class dormantViewer:
    """
    An object that p.viewer can be assigned to, which only creates or connects
    to a display tool when required.
    """
    def __init__(self, parent=None, viewer_name=None, use_existing=True):
        """
        Create a dormantViewer() object.

        Parameters
        ----------
        parent: a PrimitivesBASE object
            the parent to attach the viewer to when it awakes
        viewer_name: str/None
            name of the viewer (passed to gempy.display.connect)
        use_existing: bool
            must connect to an existing viewer?
        """
        if not isinstance(parent, PrimitivesBASE):
            raise ValueError("dormantViewer must be instantiated with a "
                             "parent")
        self.parent = weakref.ref(parent)
        self.viewer_name = viewer_name
        self.use_existing = use_existing

    def __getattr__(self, name):
        """This is how the viewer will wake up"""
        if self.viewer_name is not None:
            try:
                self.parent().viewer = display.connect(
                    self.viewer_name, use_existing=self.use_existing,
                    quit_window=False
                )
            except NotImplementedError:
                self.parent().log.warning("Attempting to display to an unknown"
                                          " display ({}). Image display turned"
                                          " off".format(self.viewer_name))
                self.viewer_name = None
            except ValueError:
                self.viewer_name = None
            else:
                return getattr(self.parent().viewer, name)
        return self

    def __setattr__(self, name, value):
        # Ignore anything else
        if name in ('parent', 'viewer_name', 'use_existing'):
            object.__setattr__(self, name, value)

    def __call__(self, *args, **kwargs):
        pass

# ------------------------------------------------------------------------------


def cleanup(process):
    # Function for the atexit registry to kill the ETISubprocess
    process.terminate()


@parameter_override
class PrimitivesBASE:
    """
    This is the base class for all of primitives classes for the geminidr
    primitive sets. __init__.py provides, or should provide, all attributes
    needed by subclasses.

    Parameters
    ----------
    adinputs : list
        A list of astrodata objects.
    mode : str
        Operational Mode, one of 'sq', 'qa', 'ql'.
    upload : list
        A list of products to upload to fitsstore.
        QA metrics uploaded if 'metrics' in upload.  E.g.::

            upload = ['metrics', ['calibs', ... ]]

    """
    tagset = None

    def __init__(self, adinputs, mode='sq', ucals=None, uparms=None, upload=None):
        self.streams          = {'main': adinputs}
        self.mode             = mode
        self.params           = {}
        self.log              = logutils.get_logger(__name__)
        self._upload          = upload
        self.user_params      = dict(uparms) if uparms else {}
        self.calurl_dict      = calurl_dict.calurl_dict
        self.timestamp_keys   = timestamp_keywords.timestamp_keys
        self.keyword_comments = keyword_comments.keyword_comments
        self.sx_dict          = sextractor_dict.sx_dict.copy()

        # Prepend paths to SExtractor input files now
        self.sx_dict.update({
            k: os.path.join(os.path.dirname(sextractor_dict.__file__), v)
            for k, v in self.sx_dict.items()
        })

        self.cachedict        = set_caches()
        self.calibrations     = Calibrations(calindfile, user_cals=ucals)
        self.stacks           = load_cache(stkindfile)

        # This lambda will return the name of the current caller.
        self.myself           = lambda: stack()[1][3]

        warnings.simplefilter('ignore', category=VerifyWarning)

        # Create a parallel process to which we can send shell commands.
        # Spawning a shell command makes a copy of its parent process in RAM
        # so we need this process to have a small memory footprint and hence
        # create it now. Garbage collect too, in case stuff has happened
        # previously.
        gc.collect()
        self.eti_subprocess = ETISubprocess()
        atexit.register(cleanup, self.eti_subprocess)

        # Instantiate a dormantViewer(). Only ds9 for now.
        self.viewer = dormantViewer(self, 'ds9')

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

    def _param_update(self, module):
        """Create/update an entry in the primitivesClass's params dict
        using Config classes in the module provided"""
        for attr in dir(module):
            obj = getattr(module, attr)
            if isclass(obj) and issubclass(obj, config.Config):
                # Allow classes purely for inheritance purposes to be ignored
                # Wanted to use hasattr(self) but things like NearIR don't inherit
                if attr.endswith("Config"):
                    primname = attr.replace("Config", "")
                    self.params[primname] = obj()

        # Play a bit fast and loose with python's inheritance. We need to check
        # if we're redefining a Config that has already been inherited by
        # another Config and, if so, update the child Config
        # Do this in the correct inheritance order
        for k, v in sorted(self.params.items(),
                           key=lambda x: len(x[1].__class__.__bases__)):
            self.params[k] = v.__class__()

            for cls in reversed((v.__class__,) + v.__class__.__bases__):
                cls_name = cls.__name__
                if cls_name.find('Config') > 0:
                    # We may not have yet imported a Config from which we inherit.
                    # In fact, we may never do so, in which case what's already
                    # there from standard inheritance is fine and we move on.
                    cls_name = cls_name.replace("Config", "")
                    try:
                        new_cls = self.params[cls_name].__class__
                    except KeyError:
                        pass
                    else:
                        # Inherit Fields from the latest version of the Config
                        # Delete history from previous passes through this code
                        for field in new_cls():
                            if field in self.params[k]:
                                self.params[k]._history[field] = []
                            self.params[k]._fields[field] = deepcopy(new_cls._fields[field])
                        # Call inherited setDefaults from configs with the same name
                        # but simply copy parameter values from others
                        #if cls.__name__ == k+'Config':
                        #    cls.setDefaults.__func__(self.params[k])
                        #else:
                        #    new_cls.setDefaults.__func__(self.params[k])
                        if cls_name == k:
                            cls.setDefaults(self.params[k])
                        else:
                            self.params[k].update(**self._inherit_params(dict(self.params[cls_name].items()), k))
            # Actually set the defaults that have been changed after setDefaults()!
            for field in self.params[k]._fields.values():
                field.default = self.params[k]._storage[field.name]

    def _inherit_params(self, params, primname, pass_suffix=False):
        """Create a dict of params for a primitive from a larger dict,
        using only those that the primitive needs

        Parameters
        ----------
        params: dict
            parent parameter dictionary
        primname: str
            name of primitive to be called
        pass_suffix: bool
            pass "suffix" parameter?
        """
        return {k: v for k, v in params.items()
                if k in list(self.params[primname]) and
                not (k == "suffix" and not pass_suffix)}
