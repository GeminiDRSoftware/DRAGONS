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
import pickle
import warnings
from copy import deepcopy
from inspect import stack, isclass
from multiprocessing import Process, Queue
from subprocess import check_output, STDOUT, CalledProcessError

from queue import Empty
import atexit

from gempy.library import config

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
        return pickle.load(open(cachefile, 'rb'))
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
def cmdloop(inQueue, outQueue):
    # Run shell commands as they arrive and return the output
    while True:
        cmd = inQueue.get()
        try:
            result = check_output(cmd, stderr=STDOUT)
        except CalledProcessError as e:
            result = e
        outQueue.put(result)

def cleanup(process):
    # Ensure the child process is terminated with the parent
    process.terminate()

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
        self.params           = {}
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

        # Create a parallel process to which we can send shell commands.
        # Spawning a shell command makes a copy of its parent process in RAM
        # so we need this process to have a small memory footprint.
        self._inQueue = Queue()
        self._outQueue = Queue()
        self._cmd_host_process = Process(target=cmdloop, args=(self._inQueue,
                                                               self._outQueue))
        self._cmd_host_process.start()
        atexit.register(cleanup, self._cmd_host_process)

    def _kill_subprocess(self):
        self._cmd_host_process.terminate()

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
                primname = attr.replace("Config", "")
                # Allow classes purely for inheritance purposes to be ignored
                # Wanted to use hasattr(self) but things like NearIR don't inherit
                if attr.endswith("Config"):
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
        passed_params = {k: v for k, v in params.items()
                         if k in list(self.params[primname]) and
                         not (k == "suffix" and not pass_suffix)}
        return passed_params
