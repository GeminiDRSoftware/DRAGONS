"""
This module provides decorator functions for implementing the (somewhat TBD)
parameter override policy of the prototype primitive classes. This
implementation is subject to any change in policy.

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
import json
import traceback
from contextlib import suppress
from copy import copy, deepcopy
from datetime import datetime, timezone
from functools import wraps

import geminidr
from astrodata import AstroData
from astrodata.provenance import add_history, clone_provenance, clone_history
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
            if prim == pname:
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
    gets the input information for a future call to store provenance and
    history.

    The AstroData inputs can change during the call to a primitive.  We use this
    helper function to extract the 'before' state of things so that we can
    accurately record provenance and history.  After the primitive returns, we
    have the AstroData objects into which we'll want to record this information.


    Args
    -----
    adinputs : list of incoming `AstroData` objects
        We expect to be called before the primitive executes, since we want to
        capture the state of the adinputs before they may be modified.

    Returns
    --------
    `dict` by datalabel of dictionaries with the filename, md5, provenance and
        history data from the inputs
    """
    retval = dict()
    for ad in adinputs:
        if ad.path:
            md5 = md5sum(ad.path) or ""
        else:
            md5 = ""
        if hasattr(ad, 'PROVENANCE'):
            provenance = ad.PROVENANCE.copy()
        else:
            provenance = []
        if hasattr(ad, 'HISTORY'):
            history = ad.HISTORY.copy()
        elif hasattr(ad, 'PROVHISTORY'):
            # Old name for backwards compatability
            history = ad.PROVHISTORY.copy()
        else:
            history = []

        retval[ad.data_label()] = {
            "filename": ad.filename,
            "md5": md5,
            "provenance": provenance,
            "history": history
        }

    return retval


def _top_level_primitive():
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
    provenance_inputs : provenance and history information to add
        This is a dictionary keyed by datalabel of dictionaries with the relevant
        provenance for that particular input.  Each dictionary contains the filename,
        md5 and the provenance and history of that `AstroData` prior to execution of
        the primitive.
    ret_value : outgoing list of `AstroData` data
    fn : name of the function (primitive) being executed
    args : arguments that are being passed to the primitive

    Returns
    --------
    none
    """
    try:
        timestamp = datetime.now(timezone.utc).replace(tzinfo=None)
        for ad in ret_value:
            if ad.data_label() in provenance_inputs:
                # output corresponds to an input, we only need to copy from there
                clone_provenance(provenance_inputs[ad.data_label()]['provenance'], ad)
                clone_history(provenance_inputs[ad.data_label()]['history'], ad)
            else:
                if hasattr(ad, 'HISTORY') or hasattr(ad, 'PROVHISTORY'):
                    clone_hist = False
                else:
                    clone_hist = True
                for provenance_input in provenance_inputs.values():
                    clone_provenance(provenance_input['provenance'], ad)
                    if clone_hist:
                        clone_history(provenance_input['history'], ad)
        for ad in ret_value:
            add_history(ad, timestamp_start.isoformat(), timestamp.isoformat(), fn.__name__, args)
    except Exception as e:
        # we don't want provenance failures to prevent data reduction
        log.warn("Unable to save provenance information, continuing on: %s" % e)
        traceback.print_exc()


@make_class_wrapper
def capture_provenance(fn):
    """
    Decorator for carrying forward provenance data and updating the history
    """
    @wraps(fn)
    def gn(pobj, *args, **kwargs):
        # grab kwargs for provenance
        stringified_args = json.dumps({k: v for k, v in kwargs.items()
                                       if not k.startswith('debug_') and not k == 'adinputs'},
                                      default=lambda v: v.filename if hasattr(v, 'filename')
                                      else '<not serializable>')

        # Determine if this is a top-level primitive, by checking if the
        # calling function contains a self that is also a primitive
        toplevel = _top_level_primitive()
        timestamp_start = datetime.now(timezone.utc).replace(tzinfo=None)

        if toplevel:
            provenance_inputs = _get_provenance_inputs(kwargs["adinputs"])

        ret_value = fn(pobj, **kwargs)

        if toplevel:
            _capture_provenance(provenance_inputs, ret_value,
                                timestamp_start, fn, stringified_args)
        return ret_value
    return gn


@make_class_wrapper
def parameter_override(fn):
    """
    Decorator for handling primitive configuration and user supplied parameters.
    """
    @wraps(fn)
    def gn(pobj, *args, **kwargs):
        pname = fn.__name__

        # Determine if this is a top-level primitive, by checking if the
        # calling function contains a self that is also a primitive
        toplevel = _top_level_primitive()

        # The first thing on the stack is the decorator, so check the name
        # of the function calling that... don't indent or log if it's the
        # same as this primitive.
        try:
            caller = inspect.stack()[1].function
        except (AttributeError, IndexError):
            caller = None

        # Start with the config file to get list of parameters
        # Copy to avoid permanent changes; shallow copy is OK
        if pname not in pobj.params:
            err_msg = ('Could not find "{}Config" configuration in "{}" module'
                       ' or any parent module.')
            raise KeyError(err_msg.format(
                pname, pobj.__module__.replace("primitives", "parameters")))

        config = copy(pobj.params[pname])

        # Find user parameter overrides
        params = userpar_override(pname, list(config), pobj.user_params)
        if caller != pname:
            set_logging(pname)

        if toplevel:
            for k, v in kwargs.items():
                if k in params:
                    log.warning(f'Parameter {k}={params[k]} was set but will '
                                'be ignored because the recipe enforces a '
                                'different value (={v})')

        # Override with values in the function call
        params.update(kwargs)

        # config doesn't know about streams or adinputs
        stream = params.get('stream', 'main')
        instream = params.get('instream', stream)
        outstream = params.get('outstream', stream)
        adinputs = params.get('adinputs')
        skip = params.get('skip_primitive', False)
        write_after = params.get('write_outputs', False) and toplevel

        if skip and toplevel:
            log.stdinfo(f"Parameter skip_primitive has been set so {pname} "
                        "will not be run")
            if instream != outstream:
                log.warning("The input and output streams differ so skipping "
                            "this primitive may have unintended consequences")
            ret_value = pobj.streams[instream]
        else:
            for k in ('adinputs', 'stream', 'instream', 'outstream',
                      'skip_primitive', 'write_outputs'):
                if k not in config:
                    with suppress(KeyError):
                        del params[k]
            # Can update config now it only has parameters it knows about
            config.update(**params)
            config.validate()

            if len(args) == 0 and adinputs is None:
                # Use appropriate stream input/output
                # Many primitives operate on AD instances in situ, so need to
                # copy inputs if they're going to a new output stream
                if instream != outstream:
                    adinputs = [deepcopy(ad) for ad in pobj.streams.get(instream, [])]
                else:
                    # Allow a non-existent stream to be passed
                    adinputs = pobj.streams.get(instream, [])

                try:
                    fnargs = dict(config.items())
                    ret_value = fn(pobj, adinputs=adinputs, **fnargs)
                    assert_expected_dtypes(ret_value)
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
                        raise TypeError("Single AstroData instance passed to "
                                        "primitive, should be a list")
                    ret_value = fn(pobj, adinputs=adinputs, **dict(config.items()))
                    assert_expected_dtypes(ret_value)
                except Exception:
                    zeroset()
                    raise

        if write_after:
            pobj.writeOutputs(stream=outstream)
        if caller != pname:
            unset_logging()
        gc.collect()
        return ret_value
    return gn


# Enforce the expected dtypes between primitives, to catch NumPy 2 issues:
def assert_expected_dtypes(adinputs):
    msg = ""
    for ad in adinputs:
        for n, ext in enumerate(ad):
            emsg = f'  File {ad.filename}, AstroData ext {n}:\n'
            initlen = len(emsg)
            ndd = ext.nddata
            if ndd.data.dtype.itemsize > 4:  # int/float with max 32 bits
                emsg += f'    data:        {ndd.data.dtype}\n'
            if ndd.uncertainty is not None and (
                ndd.uncertainty.array.dtype.kind != 'f' or
                ndd.uncertainty.array.dtype.itemsize != 4  # expect float32
            ):
                emsg += f'    uncertainty: {ndd.data.dtype}\n'
            if ndd.mask is not None and (
                ndd.mask.dtype.kind != 'u' or
                ndd.mask.dtype.itemsize > 2  # uint16 for Gemini data; OK?
            ):
                emsg += f'    mask:        {ndd.mask.dtype}\n'
            # The other attribute that gets to >1MB in practice is OBJMASK:
            if hasattr(ext, 'OBJMASK') and ext.OBJMASK.dtype.itemsize > 1:
                mesg += f'    OBJMASK:     {ext.OBJMASK.dtype}\n'
            if len(emsg) > initlen:
                msg += emsg
    if msg:
        raise AssertionError(
            f'Produced unexpected output data type(s):\n\n{msg}'
        )
