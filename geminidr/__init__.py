"""
The geminidr package provides the base classes for all parameter and primitive
classes in the geminidr package.

This module provides the caches library to primitives, but currently, only
Bookkeeping uses the cache directly (addToList).

E.g.,
>>> from geminidr import PrimitivesBASE

"""

# ------------------------------------------------------------------------------
import os
import gc
import pickle
import warnings
import weakref
import re

from copy import deepcopy
from inspect import isclass, currentframe
from itertools import chain

from gempy.eti_core.eti import ETISubprocess
from gempy.library import config
from gempy.utils import logutils
from gempy import display

from astropy.io.fits.verify import VerifyWarning

# new system imports - 10-06-2016 kra
# NOTE: imports of these and other tables will be moving around ...
from gempy.utils.soundex import soundex
from recipe_system.reduction.coreReduce import UnrecognizedParameterException
from .gemini.lookups import keyword_comments
from .gemini.lookups import timestamp_keywords
from .gemini.lookups.source_detection import sextractor_dict

from recipe_system.cal_service import init_calibration_databases
from recipe_system.utils.decorators import parameter_override, capture_provenance
from recipe_system.config import load_config
# ------------------------------ caches ---------------------------------------
# Formerly in cal_service/caches.py
#
# GLOBAL/CONSTANTS (could be exported to config file)

# [caches]
caches = {
    'reducecache': '.reducecache',
}

stkindfile = os.path.join('.', caches['reducecache'], "stkindex.pkl")


def set_caches():
    cachedict = {name: cachedir for name, cachedir in caches.items()}
    return cachedict


def load_cache(cachefile):
    if os.path.exists(cachefile):
        with open(cachefile, 'rb') as fp:
            return pickle.load(fp)
    else:
        return {}


def save_cache(obj, cachefile):
    cachedir = os.path.dirname(cachefile)
    if not os.path.exists(cachedir):
        os.makedirs(cachedir)
    with open(cachefile, 'wb') as fp:
        pickle.dump(obj, fp, protocol=2)

# ------------------------- END caches-----------------------------------------


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


@parameter_override
@capture_provenance
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
    ucals : dict
        user-defined cals, e.g., {"processed_bias": "mybias.fits"}
    uparms : dict
        user-defined parameters, e.g., {"stackFrames:reject_method": "sigclip"}
    upload : list
        A list of products to upload to fitsstore.
        QA metrics uploaded if 'metrics' in upload.  E.g.::

            upload = ['metrics', ['calibs', ... ]]

    config_file : str/None
        name of DRAGONS configuration file (None => default)
    """
    tagset = None

    def __init__(self, adinputs, mode='sq', ucals=None, uparms=None, upload=None,
                 config_file=None):
        if hasattr(self, "_in_init") and self._in_init:
            raise OverflowError("Caught recursive call to __init__.  "
                                "This is likely an error in the subclass definition.  Did you forget "
                                "to update the super() call when redefining _init() to _initialize()?")
        self._in_init = True

        self._initialize(adinputs, mode=mode, ucals=ucals, uparms=uparms, upload=upload, config_file=config_file)

        # Most logic is separated into initialize() so subclasses can do custom initialization
        # while leaving any final logic to this base class.  This avoids having to repeat
        # this validate call across all subclass definitions
        self._initialize_params()
        self._validate_user_parms()

        self._in_init = False

    def _initialize(self, adinputs, mode='sq', ucals=None, uparms=None, upload=None,
                    config_file=None):
        # This is a general config file so we should load it now. Some of its
        # information may be overridden by other parameters passed here.

        load_config(config_file)

        self.streams          = {'main': adinputs}
        self.mode             = mode
        self.params           = {}
        self.log              = logutils.get_logger(__name__)
        self._upload          = upload
        self.user_params      = uparms if isinstance(uparms, dict) else dict(uparms) if uparms else {}

        # remove quotes from string values.  This happens when quotes are used
        # in the @-file.  The shell removes the quotes automatically.
        quote_pattern = "^[\"\'](.+)[\"\']$"
        for key, value in self.user_params.items():
            if isinstance(value, str):
                self.user_params[key] = re.sub(quote_pattern, r"\1", value)

        self.timestamp_keys   = timestamp_keywords.timestamp_keys
        self.keyword_comments = keyword_comments.keyword_comments
        self.sx_dict          = sextractor_dict.sx_dict.copy()

        # Prepend paths to SExtractor input files now
        self.sx_dict.update({
            k: os.path.join(os.path.dirname(sextractor_dict.__file__), v)
            for k, v in self.sx_dict.items()
        })

        self.caldb            = init_calibration_databases(
            getattr(self, "inst_lookups", None), ucals=ucals, upload=upload,
            procmode=self.mode)
        self.cachedict        = set_caches()
        self.stacks           = load_cache(stkindfile)

        # This lambda will return the name of the current caller.
        self.myself = lambda: currentframe().f_back.f_code.co_name

        warnings.simplefilter('ignore', category=VerifyWarning)

        # Create a parallel process to which we can send shell commands.
        # Spawning a shell command makes a copy of its parent process in RAM
        # so we need this process to have a small memory footprint and hence
        # create it now. Garbage collect too, in case stuff has happened
        # previously.
        gc.collect()
        self.eti_subprocess = ETISubprocess()

        # Instantiate a dormantViewer(). Only ds9 for now.
        self.viewer = dormantViewer(self, 'ds9')

    def _validate_user_parms(self):
        """
        Validate the user parameters.

        This is an internal method that checks the user parameters for
        any obvious issues.  This lets us fail fast in case the user
        made an error.

        :raises: :class:~recipe_system.reduction.coreReduce.UnrecognizedParameterException: \
            when a user parameter uses an unrecognized primitive or parameter name
        """

        def format_exception(name, alternative_names, noun):
            msg = f"{noun} {name} not recognized"
            if len(alternative_names) == 1:
                msg += f", did you mean {alternative_names[0]}?"
            elif len(alternative_names) > 1:
                msg += f", did you mean one of {alternative_names}?"
            return msg

        def find_similar_names(name, valid_names):
            """Will return None if it's valid"""
            # moved outside _validate_user_parms for pytest use
            if name in valid_names:
                return None
            alternative_names = [n for n in valid_names if n.upper() == name.upper()]
            if alternative_names:
                return alternative_names
            alternative_names = [n for n in valid_names if soundex(n) == soundex(name)]
            return alternative_names

        for key in self.user_params.keys():
            primitive = None
            if ':' in key:
                split_key = key.split(':')
                if len(split_key) != 2:
                    raise UnrecognizedParameterException("Expecting parameter or primitive:parameter in "
                                                         "-p user parameters")
                primitive = split_key[0]
                parameter = split_key[1]
            else:
                parameter = key

            if primitive:
                alternative_primitives = find_similar_names(primitive, self.params.keys())
                if alternative_primitives is None:  # it's valid
                    if parameter in ('skip_primitive', 'write_outputs'):
                        alternative_parameters = None
                    else:
                        alternative_parameters = find_similar_names(parameter, self.params[primitive])
            else:
                alternative_parameters = find_similar_names(
                    parameter, chain(*[v.keys() for v in self.params.values()]))

            if primitive and alternative_primitives is not None:  # it's invalid
                raise UnrecognizedParameterException(format_exception(primitive, alternative_primitives, "Primitive"))
            if alternative_parameters is not None:  # it's invalid
                raise UnrecognizedParameterException(format_exception(parameter, alternative_parameters, "Parameter"))

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
        """Create/update an entry in the primitivesClass's params dict;
        this will be initialized later"""
        for attr in dir(module):
            obj = getattr(module, attr)
            if isclass(obj) and issubclass(obj, config.Config):
                # Allow classes purely for inheritance purposes to be ignored
                # Wanted to use hasattr(self) but things like NearIR don't inherit
                if attr.endswith("Config"):
                    primname = attr.replace("Config", "")
                    self.params[primname] = obj()

    def _initialize_params(self):
        """
        Instantiate all the Config instances that store the parameters for
        each of the primitives. We do this once after all the modules have
        been loaded.
        """
        # Play a bit fast and loose with python's inheritance. A Config that
        # inherits from a parent might not want that parent, but the
        # identically-named parent associated with this primitivesClass.
        # Do this in the correct inheritance order
        for k, v in sorted(self.params.items(),
                           key=lambda x: len(x[1].__class__.__bases__)):
            self.params[k] = v.__class__()

            for cls in reversed((v.__class__,) + v.__class__.__bases__):
                cls_name = cls.__name__
                if cls_name.find('Config') > 0:
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
                        # Call inherited setDefaults from configs with the
                        # same name but simply copy parameter values from
                        # others, since inherited Configs with other names
                        # will have already have had setDefaults() run
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
