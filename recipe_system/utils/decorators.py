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
import hashlib
import traceback
from collections import Iterable
from datetime import datetime
from time import strptime

import astropy
import psutil
from functools import wraps
from copy import copy, deepcopy

import astrodata
from astrodata import AstroData, AstroDataFits
from astropy.table import Table, Column
from gempy.utils import logutils
import numpy as np


def memusage():
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


# TODO pull these from FitsStorage?
def md5sum_size_fp(fobj):
    """
    Generates the md5sum and size of the data returned by the file-like object fobj, returns
    a tuple containing the hex string md5 and the size in bytes.
    f must be open. It will not be closed. We will read from it until we encounter EOF.
    No seeks will be done, fobj will be left at eof
    """
    # This is the block size by which we read chunks from the file, in bytes
    block = 1000000 # 1 MB

    hashobj = hashlib.md5()

    size = 0

    while True:
        data = fobj.read(block)
        if not data:
            break
        size += len(data)
        hashobj.update(data)

    return hashobj.hexdigest(), size


def md5sum(filename):
    """
    Generates the md5sum of the data in filename, returns the hex string.
    """

    with open(filename, 'rb') as filep:
        (md5, size) = md5sum_size_fp(filep)
        return md5


def ensure_provenance_extensions(ad):
    log.debug("Not a FITS AstroData, we don't store provenance")


def add_provenance(ad, timestamp, filename, md5, primitive):
    if isinstance(ad, AstroDataFits):
        if timestamp is None:
            timestamp_str = ""
        else:
            timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        if md5 is None:
            md5 = ""

        if 'GEM_PROVENANCE' not in ad:
            timestamp_data = np.array([timestamp_str])
            filename_data = np.array([filename])
            md5_data = np.array([md5])
            primitive_data = np.array([primitive])
            my_astropy_table = Table([timestamp_data, filename_data, md5_data, primitive_data],
                                     names=('timestamp', 'filename', 'md5', 'primitive'),
                                     dtype=('S20', 'S128', 'S128', 'S128'))
            ad.append(my_astropy_table, name='GEM_PROVENANCE')
            pass
        else:
            provenance = ad.GEM_PROVENANCE
            provenance.add_row((timestamp_str, filename, md5, primitive))
            pass
    else:
        log.warn("Not a FITS AstroData, add provenance does nothing")


def __stringify_list__(obj):
    if obj is None:
        return ""
    if isinstance(obj, AstroData):
        if 'filename' in obj:
            return obj.filename
        else:
            return "(internal data)"
    elif isinstance(obj, Iterable):
        # NOTE: generator will not work for string format, needs a list
        retval = list()
        for el in obj:
            retval.append(__stringify_list__(el))
        return "%s" % retval
    else:
        return obj


def add_provenance_history(ad, timestamp_start, timestamp_end, primitive, args):
    if isinstance(ad, AstroDataFits):
        timestamp_start_str = ""
        if timestamp_start is not None:
            timestamp_start_str = timestamp_start.strftime("%Y-%m-%d %H:%M:%S")
        timestamp_end_str = ""
        if timestamp_end is not None:
            timestamp_end_str = timestamp_end.strftime("%Y-%m-%d %H:%M:%S")
        if 'GEM_PROVENANCE_HISTORY' not in ad:
            timestamp_start_data = np.array([timestamp_start_str])
            timestamp_end_data = np.array([timestamp_end_str])
            primitive_data = np.array([primitive])
            args_data = np.array([args])

            my_astropy_table = Table([timestamp_start_data, timestamp_end_data, primitive_data, args_data],
                                     names=('timestamp_start', 'timestamp_end', 'primitive', 'args'),
                                     dtype=('S20', 'S20', 'S128', 'S128'))
            # astrodata.add_header_to_table(my_astropy_table)
            ad.append(my_astropy_table, name='GEM_PROVENANCE_HISTORY', header=astropy.io.fits.Header())
        else:
            history = ad.GEM_PROVENANCE_HISTORY
            history.add_row((timestamp_start_str, timestamp_end_str, primitive, args))
    else:
        log.warn("Not a FITS AstroData, add provenance history does nothing")


def get_provenance(ad):
    retval = list()
    if isinstance(ad, AstroDataFits):
        if 'GEM_PROVENANCE' in ad:
            provenance = ad.GEM_PROVENANCE
            pass
            for row in provenance:
                timestamp = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
                filename = row[1]
                md5 = row[2]
                primitive = row[3]
                retval.append({"timestamp": timestamp, "filename": filename, "md5": md5, "primitive": primitive})
    return retval


def get_provenance_history(ad):
    retval = list()
    if isinstance(ad, AstroDataFits):
        if 'GEM_PROVENANCE_HISTORY' in ad:
            provenance_history = ad.GEM_PROVENANCE_HISTORY
            for row in provenance_history:
                timestamp_start = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
                timestamp_end = datetime.strptime(row[1], "%Y-%m-%d %H:%M:%S")
                primitive = row[2]
                args = row[3]
                retval.append({"timestamp_start": timestamp_start, "timestamp_end": timestamp_end,
                               "primitive": primitive, "args": args})
    return retval


def clear_provenance_extension(ad):
    if isinstance(ad, AstroDataFits):
        if 'GEM_PROVENANCE' in ad:
            provenance = ad.GEM_PROVENANCE
            print("%d" % len(provenance))
            pass


def clear_provenance_history_extension(ad):
    if isinstance(ad, AstroDataFits):
        if 'GEM_PROVENANCE_HISTORY' in ad:
            pass


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


def _get_input_info(adinputs):
    input_filenames = list()
    for ad in adinputs:
        if ad.filename:
            origname = ad.filename
            if isinstance(ad, AstroDataFits):
                if "ORIGNAME" in ad.phu:
                    origname = ad.phu["ORIGNAME"]
                    log.warn("ENHANCING PROVENANCE using origname from header of %s" % origname)
            input_filenames.append((ad.filename, ad.path, origname,
                                    get_provenance(ad), get_provenance_history(ad)))
    return input_filenames


def _capture_provenance(input_filenames, ret_value, timestamp_start, fn, args):
    try:
        timestamp = datetime.now()
        for ad in ret_value:
            ensure_provenance_extensions(ad)
            # If we have matching inputs, or if we are a consolidated output with one matching input,
            # we want to pull in the source provenance
            if isinstance(ad, AstroDataFits) and 'GEM_PROVENANCE' not in ad:
                if len(ret_value) > 1:
                    for filename, path, orig_filename, provenance, provenance_history in input_filenames:
                        output_origname = ad.filename
                        if isinstance(ad, AstroDataFits):
                            if "ORIGNAME" in ad.phu:
                                output_origname = ad.phu["ORIGNAME"]
                        if orig_filename == output_origname:
                            # clear_provenance_extension(ad)
                            # matching input, copy provenance
                            log.warn("ENHANCING PROVENANCE saw a matching input: %s" % orig_filename)
                            for prov in provenance:
                                log.warn("ENHANCING PROVENANCE saw %s" % prov)
                                add_provenance(ad, prov['timestamp'], prov['filename'], prov['md5'],
                                               prov['primitive'])
                            for ph in provenance_history:
                                log.warn("ENHANCING PROVENANCE saw history %s" % ph)
                                add_provenance_history(ad, ph['timestamp_start'], ph['timestamp_end'],
                                                       ph['primitive'], ph['args'])
                else:
                    if len(input_filenames) > 0:
                        filename, path, orig_filename, provenance, provenance_history = input_filenames[0]
                        output_origname = ad.filename
                        if isinstance(ad, AstroDataFits):
                            if "ORIGNAME" in ad.phu:
                                output_origname = ad.phu["ORIGNAME"]
                        if orig_filename == output_origname:
                            # clear_provenance_history_extension(ad)
                            log.warn("ENHANCING PROVENANCE multi->single saw a matching input: %s"
                                     % orig_filename)
                            # if the first input of a M->1 primitive matches our output, pull in all
                            # the input provenance data
                            for filename, path, orig_filename, provenance, provenance_history in input_filenames:
                                for prov in provenance:
                                    log.warn("ENHANCING PROVENANCE saw %s" % prov)
                                    add_provenance(ad, prov['timestamp'], prov['filename'], prov['md5'],
                                                   prov['primitive'])
                                for ph in provenance_history:
                                    log.warn("ENHANCING PROVENANCE saw history %s" % ph)
                                    add_provenance_history(ad, ph['timestamp_start'], ph['timestamp_end'],
                                                           ph['primitive'], ph['args'])
            for filename, path, orig_filename, provenance, provenance_history in input_filenames:
                if path:
                    try:
                        md5 = md5sum(path)
                    except Exception as e:
                        log.warn("Unable to compute md5 for file, "
                                 "not storing md5 in provenance: %s" % e)
                        md5 = None
                else:
                    md5 = None
                output_origname = ad.filename
                if isinstance(ad, AstroDataFits):
                    if "ORIGNAME" in ad.phu:
                        output_origname = ad.phu["ORIGNAME"]
                if len(ret_value) == 1 or orig_filename == output_origname:
                    add_provenance(ad, timestamp, filename, md5, fn.__name__)
                    add_provenance_history(ad, timestamp_start, timestamp, fn.__name__, args)
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
                input_filenames = _get_input_info(adinputs)
                ret_value = fn(pobj, adinputs=adinputs, **dict(config.items()))
                _capture_provenance(input_filenames, ret_value, timestamp_start, fn, stringified_args)
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
                input_filenames = _get_input_info(adinputs)
                ret_value = fn(pobj, adinputs=adinputs, **dict(config.items()))
                _capture_provenance(input_filenames, ret_value, timestamp_start, fn, stringified_args)
            except Exception:
                zeroset()
                raise
        unset_logging()
        gc.collect()
        return ret_value
    return gn
