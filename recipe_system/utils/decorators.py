# -*- coding: utf-8 -*-
"""
This module provides decorator functions for implementing the (somewhat TBD) 
parameter override policy of the prototype primitive classes. This implementation 
is subject to any change in policy.

Currently, the policy defines an order of parameter precedence:

1) ​*user parameters*​ -- as passed by -p on the ‘reduce’ command line. These take
    full precedence over any other parameter setting. User parameters ​_may_​ be
    primitive-specific, as in `-p makeFringeFrame:reject_method=jilt` in which
    case, ​_only_​ that primitive’s parameter will be overridden. Other primitives
    with the same parameter (e.g. `reject_method` is also a parameter for
    `stackFlats`) will remain unaffected. If a user passes an un-specified
    parameter, as in `reject_method=jilt`, then any primitive with that parameter
    will receive that parameter value.

2) ​*recipe parameters*​ -- as passed on a recipe function call, like 
   `recipe_name(par1=val1)`. These will be overridden by same named user 
    parameters.

3) ​*default parameters*​ -- The default parameter sets as defined in package 
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
            [ … ]

"""
from functools import wraps
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
    for key, val in upars.items():
        if ':' in key:
            prim, par = key.split(':')
            if prim == pname and par in args:
                parset.update({par: val})
        elif key in args:
            parset.update({par: val})
    return parset

# -------------------------------- decorators ----------------------------------
def make_class_wrapper(wrapped):
    @wraps(wrapped)
    def class_wrapper(cls):
        for attr_name in dir(cls):
            if attr_name.startswith("_"):        # no prive, no magic
                continue

            attr_fn = getattr(cls, attr_name)
            if callable(attr_fn):
                if attr_name not in attr_fn.im_class.__dict__:
                    continue
                else:
                    setattr(cls, attr_name, wrapped(attr_fn))
        return cls
    return class_wrapper

@make_class_wrapper
def parameter_override(fn):
    @wraps(fn)
    def gn(*args, **kwargs):
        global LOGINDENT
        LOGINDENT += 1
        logutils.update_indent(LOGINDENT)
        pobj = args[0]
        pname = fn.__name__
        params = getattr(getattr(pobj, pname), 'parameters').copy()
        params.update(getattr(pobj.parameters, pname, {}))
        params.update(userpar_override(pname, params.keys(),
                      pobj.user_params))
        params.update(kwargs)

        if len(args) == 1 and 'adinputs' not in params:
            instream = params.get('instream', params.get('stream', 'main'))
            params.update({'adinputs': pobj.streams[instream]})
            ret_value = fn(*args, **params)
            outstream = params.get('outstream', params.get('stream', 'main'))
            pobj.streams[outstream] = ret_value
        else:
            ret_value = fn(*args, **params)

        LOGINDENT -= 1
        logutils.update_indent(LOGINDENT)
        return ret_value
    # Make dict of default values (ignore args[0]='self', args[1]='adinputs')
    argspec = inspect.getargspec(fn)
    gn.parameters = dict(zip(argspec.args[2:], argspec.defaults[1:]))
    return gn
