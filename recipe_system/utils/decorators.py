"""
This module provides decorator functions for implementing the (somewhat TBD)
parameter override policy of the prototype primitive classes. This implementation
is subject to any change in policy.

Currently, the policy defines an order of parameter precedence:

1. *user parameters*-- as passed by -p on the reduce command line. These take
   full precedence over any other parameter setting. User parameters may be
   primitive-specific, as in `-p makeFringeFrame:reject_method=jilt` in which
   case, only that primitive's parameter will be overridden. Other primitives
   with the same parameter (e.g. `reject_method` is also a parameter for
   `stackFlats`) will remain unaffected. If a user passes an un-specified
   parameter, as in `reject_method=jilt`, then any primitive with that parameter
   will receive that parameter value.
2. *recipe parameters* -- as passed on a recipe function call, like
   `recipe_name(par1=val1)`. These will be overridden by same named user
   parameters.
3. *default parameters* -- The default parameter sets as defined in package
   parameter files. These will be overridden by recipe parameters and then
   user parameters when applicable.

This policy is implemented in the decorator function,

    parameter_override

This is the primitive decorator and should be the only one decorating a
primitive class.

This decorator is fully enhanced by the make_class_wrapper decorator, which
maps the parameter_override decorator function to all public methods on
the decorated class.

E.g.,::

    from pkg_utilities.decorators import parameter_override

    @parameter_override
    class PrimitivesIMAGE(PrimitivesGMOS):
        def __init__(self, adinputs, uparms={}):
            [ . . . ]

"""
import gc
import inspect
import traceback
from collections import Iterable
from datetime import datetime

import psutil
from functools import wraps
from copy import copy, deepcopy

import geminidr
from astrodata import AstroData, ProvenanceHistory

from gempy.utils import logutils
from recipe_system.utils.md5 import md5sum


def memusage():
    import psutil
    proc = psutil.Process()
    return '{:9.3f}'.format(float(proc.memory_info().rss) / 1e6)
# ------------------------------------------------------------------------------
LOGINDENT = 0
log = logutils.get_logger(__name__)

# ------------------------------------------------------------------------------
def userpar_override(pname, args, upars):
    """
    Implement user parameter overrides. In this implementation, user
    parameters *always* take precedence. Any user parameters passed to the
    Primitive class constructor, usually via -p on the 'reduce' command
    line, *must* override any specified recipe parameters.

    Note: user parameters may be primitive-specific, i.e. passed as

    -p makeFringeFrame:reject_method='jilt'

    in which case, the parameter will only be overridden for that primitive.
    Other primitives with the same parameter (e.g. stackFlats reject_method)
    will not be affected.

    This returns a dict of the overridden parameters and their values.
    """
    parset = {}
    for key, val in list(upars.items()):
        if ':' in key:
            prim, par = key.split(':')
            if prim == pname and par in args:
                parset.update({par: val})
        elif key in args:
            parset.update({key: val})
    return parset

def set_logging(pname):
    global LOGINDENT
    LOGINDENT += 1
    logutils.update_indent(LOGINDENT)
    #stat_msg = "{} PRIMITIVE: {}".format(memusage(), pname)
    stat_msg = "PRIMITIVE: {}".format(pname)
    log.status(stat_msg)
    log.status("-" * len(stat_msg))
    return

def unset_logging():
    global LOGINDENT
    log.status(".")
    LOGINDENT -= 1
    logutils.update_indent(LOGINDENT)
    return

def zeroset():
    global LOGINDENT
    LOGINDENT = 0
    logutils.update_indent(LOGINDENT)
    return


# -------------------------------- decorators ----------------------------------
def make_class_wrapper(wrapped):
    @wraps(wrapped)
    def class_wrapper(cls):
        for attr_name, attr_fn in list(cls.__dict__.items()):
            if attr_name.startswith("_") or not callable(attr_fn): # no prive,magic
                continue

            setattr(cls, attr_name, wrapped(attr_fn))
        return cls
    return class_wrapper


def _get_provenance_inputs(adinputs):
    """
    gets the input information for a future call to store provenance history.

    The AstroData inputs can change during the call to a primitive.  We use this
    helper function to extract the 'before' state of things so that we can accurately
    record provenance history.  After the primitive returns, we have the AstroData
    objects into which we'll want to record this information.


    Args
    -----
    adinputs : list of incoming `AstroData` objects
        We expect to be called before the primitive executes, since we want to capture the
        state of the adinputs before they may be modified.

    Returns
    --------
    list of dictionaries with the filename, md5, provenance and provenance_history data from the inputs
    """
    retval = list()
    for ad in adinputs:
        if ad.path:
            md5 = md5sum(ad.path) or ""
        else:
            md5 = ""
        retval.append({"filename": ad.filename,
                        "md5": md5,
                        "provenance": ad.provenance,
                        "provenance_history": ad.provenance_history})
    return retval


def _clone_provenance(provenance_input, ad):
    """
    For a single input's provenance, copy it into the output
    `AstroData` object as appropriate.

    This takes a dictionary with a source filename, md5 and both it's
    original provenance and provenance_history information.  It duplicates
    the provenance data into the outgoing `AstroData` ad object.

    Args
    -----
    provenance_input : dictionary with provenance data from a single input.
        We only care about the `provenance` element, which holds a list of `Provenance` data
    ad : outgoing `AstroData` object to add provenance data to

    Returns
    --------
    none

    """
    # set will be faster for checking contents
    existing_provenance = ad.provenance

    provenance = provenance_input["provenance"]

    for prov in provenance:
        if prov not in existing_provenance:
            ad.add_provenance(prov)
            existing_provenance.append(prov)


def _clone_history(provenance_input, ad):
    """
    For a single input's provenance history, copy it into the output
    `AstroData` object as appropriate.

    This takes a dictionary with a source filename, md5 and both it's
    original provenance and provenance_history information.  It duplicates
    the provenance data into the outgoing `AstroData` ad object.

    Args
    -----
    provenance_input : dictionary with provenance data from a single input.
        We only care about the `provenance_history` element, which holds a list of `ProvenanceHistory` data
    ad : outgoing `AstroData` object to add provenance data to

    Returns
    --------
    none
    """
    # set will be faster for checking contents
    existing_history = ad.provenance_history

    provenance_history = provenance_input["provenance_history"]
    for ph in provenance_history:
        if ph not in existing_history:
            ad.add_provenance_history(ph)
            existing_history.append(ph)


def __top_level_primitive__():
    """ Check if we are at the top-level, not being called from another primitive.
    
    We only want to capture provenance history when we are passing through the
    uppermost primitive calls.  These are the calls that get made from the recipe.
    """
    for trace in inspect.stack():
        if "self" in trace[0].f_locals:
            inst = trace[0].f_locals["self"]
            if isinstance(inst, geminidr.PrimitivesBASE):
                return False
    # if we encounter no primitives above this decorator, then this is a top level primitive call
    return True


def _capture_provenance(provenance_inputs, ret_value, timestamp_start, fn, args):
    """
    Add the given provenance data to the outgoing `AstroData` objects in ret_value,
    with an additional provenance entry for the current operation.

    This is a fairly specific function that does a couple of things.  First, it will
    iterate over collected provenance and history data in provenance_inputs and add
    them as appropriate to the outgoing `AstroData` in ret_value.  Second, it takes
    the current operation, expressed in timestamp_start, fn and args, and adds that
    to the outgoing ret_value objects' provenance as well.

    Args
    -----
    provenance_inputs : `ProvenanceHistory` to add
        This is an array of dictionaries.  There is one dictionary per original incoming
        `AstroData` object.  Each dictionary contains the filename, md5 and the
        provenance and provenance_history of that `AstroData` prior to execution of
        the primitive.
    ret_value : outgoing list of `AstroData` data
    fn : name of the function (primitive) being executed
    args : arguments that are being passed to the primitive

    Returns
    --------
    none
    """
    if not __top_level_primitive__():
        # We're inside a primitive being called from another, we want to omit this
        # from the provenance.  Only the recipe calls are relevant.
        return
    try:
        timestamp = datetime.now()
        if len(provenance_inputs) == len(ret_value):
            for provenance_input, ad in zip(provenance_inputs, ret_value):
                _clone_provenance(provenance_input, ad)
                if not ad.provenance_history:
                    # if our output has no history, it's fresh and we want
                    # to copy in the history from the input
                    _clone_history(provenance_input, ad)
        else:
            for ad in ret_value:
                # if our output has no history, it's fresh and we want
                # to copy in the history from the inputs
                if ad.provenance_history:
                    clone_history = False
                else:
                    clone_history = True
                for provenance_input in provenance_inputs:
                    _clone_provenance(provenance_input, ad)
                    if clone_history:
                        _clone_history(provenance_input, ad)

        for ad in ret_value:
            ad.add_provenance_history(ProvenanceHistory(timestamp_start, timestamp, fn.__name__, args))
    except Exception as e:
        # we don't want provenance failures to prevent data reduction
        log.warn("Unable to save provenance information, continuing on: %s" % e)
        traceback.print_exc()


@make_class_wrapper
def parameter_override(fn):
    @wraps(fn)
    def gn(pobj, *args, **kwargs):
        pname = fn.__name__

        # for provenance information
        stringified_args = "%s" % kwargs
        timestamp_start = datetime.now()

        # Start with the config file to get list of parameters
        # Copy to avoid permanent changes; shallow copy is OK
        config = copy(pobj.params[pname])
        # Find user parameter overrides
        params = userpar_override(pname, list(config), pobj.user_params)
        # Override with values in the function call
        params.update(kwargs)
        set_logging(pname)
        # config doesn't know about streams or adinputs
        instream = params.get('instream', params.get('stream', 'main'))
        outstream = params.get('outstream', params.get('stream', 'main'))
        adinputs = params.get('adinputs')
        for k in ('adinputs', 'stream', 'instream', 'outstream'):
            try:
                del params[k]
            except KeyError:
                pass
        # Can update config now it only has parameters it knows about
        config.update(**params)
        config.validate()

        if len(args) == 0 and adinputs is None:
            # Use appropriate stream input/output
            # Many primitives operate on AD instances in situ, so need to
            # copy inputs if they're going to a new output stream
            if instream != outstream:
                adinputs = [deepcopy(ad) for ad in pobj.streams[instream]]
            else:
                # Allow a non-existent stream to be passed
                adinputs = pobj.streams.get(instream, [])
            try:
                provenance_inputs = _get_provenance_inputs(adinputs)
                fnargs = dict(config.items())
                stringified_args = "%s" % fnargs
                ret_value = fn(pobj, adinputs=adinputs, **fnargs)
                _capture_provenance(provenance_inputs, ret_value, timestamp_start, fn, stringified_args)
            except Exception:
                zeroset()
                raise
            # And place the outputs in the appropriate stream
            pobj.streams[outstream] = ret_value
        else:
            if args:  # if not, adinputs has already been assigned from params
                adinputs = args[0]
            try:
                if isinstance(adinputs, AstroData):
                    raise TypeError("Single AstroData instance passed to primitive, should be a list")
                provenance_inputs = _get_provenance_inputs(adinputs)
                ret_value = fn(pobj, adinputs=adinputs, **dict(config.items()))
                _capture_provenance(provenance_inputs, ret_value, timestamp_start, fn, stringified_args)
            except Exception:
                zeroset()
                raise
        unset_logging()
        gc.collect()
        return ret_value
    return gn
