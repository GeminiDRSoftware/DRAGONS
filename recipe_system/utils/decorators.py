"""
This module provides decorator functions for implementing the (somewhat TBD) 
parameter override policy of the prototype primitive classes. This implementation 
is subject to any change in policy.

Currently, the policy defines an order of parameter precedence:

1)  *user parameters*-- as passed by -p on the reduce command line. These take
    full precedence over any other parameter setting. User parameters may be
    primitive-specific, as in `-p makeFringeFrame:reject_method=jilt` in which
    case, only that primitive's parameter will be overridden. Other primitives
    with the same parameter (e.g. `reject_method` is also a parameter for
    `stackFlats`) will remain unaffected. If a user passes an un-specified
    parameter, as in `reject_method=jilt`, then any primitive with that parameter
    will receive that parameter value.

2)  *recipe parameters* -- as passed on a recipe function call, like 
   `recipe_name(par1=val1)`. These will be overridden by same named user 
    parameters.

3)  *default parameters* -- The default parameter sets as defined in package 
    parameter files. These will be overridden by recipe parameters and then 
    user parameters when applicable.

This policy is implemented in the decorator function,

    parameter_override 

This is the primitive decorator and should be the only one decorating a 
primitive class.

This decorator is fully enhanced by the make_class_wrapper decorator, which
maps the parameter_override decorator function to all public methods on 
the decorated class.

    E.g., 

        from pkg_utilities.decorators import parameter_override

        @parameter_override
        class PrimitivesIMAGE(PrimitivesGMOS):
            def __init__(self, adinputs, uparms={}):
                [ . . . ]

"""
from builtins import zip
from functools import wraps
from copy import deepcopy

from gempy.utils import logutils
import inspect

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

@make_class_wrapper
def parameter_override(fn):
    @wraps(fn)
    def gn(pobj, *args, **kwargs):
        pname = fn.__name__
        # Start with the config file to get list of parameters
        config = pobj.parameters[pname]
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

        if len(args) == 0 and adinputs is None:
            # Use appropriate stream input/output
            # Many primitives operate on AD instances in situ, so need to
            # copy inputs if they're going to a new output stream
            if instream != outstream:
                adinputs = [deepcopy(ad) for ad in pobj.streams[instream]]
            else:
                # Allow a non-existent stream to be passed
                adinputs = pobj.streams.get(instream, [])
            config.validate()
            ret_value = fn(pobj, adinputs=adinputs, **dict(config.items()))
            # And place the outputs in the appropriate stream
            pobj.streams[outstream] = ret_value
        else:
            config.validate()
            if args:  # if not, adinputs has already been assigned from params
                adinputs = args[0]
            ret_value = fn(pobj, adinputs=adinputs, **dict(config.items()))

        unset_logging()
        return ret_value
    # Make dict of default values (ignore args[0]='self', args[1]='adinputs')
    #argspec = inspect.getargspec(fn)
    #gn.parameters = dict(list(zip(argspec.args[2:], argspec.defaults[1:])))
    return gn
