"""
class Reduce provides one (1) public method:

    runr()

which calls on the mapper classes and passes the received data to them.

"""
# ------------------------------------------------------------------------------
#                                                                        DRAGONS
#
#                                                                  coreReduce.py
# ------------------------------------------------------------------------------
import os

import sys
import inspect
#import traceback

from importlib import import_module

import astrodata
import gemini_instruments

from gempy.utils import logutils
from gempy.library import config

from astrodata import AstroDataError

from recipe_system import __version__

from recipe_system.utils.errors import ModeError
from recipe_system.utils.errors import RecipeNotFound
from recipe_system.utils.errors import PrimitivesNotFound

from recipe_system.utils.reduce_utils import buildParser
from recipe_system.utils.reduce_utils import normalize_ucals
from recipe_system.utils.reduce_utils import set_btypes

from recipe_system.mappers.recipeMapper import RecipeMapper
from recipe_system.mappers.primitiveMapper import PrimitiveMapper


class UnrecognizedParameterException(Exception):
    """ Exception for unrecognized user parameters. """
    pass


# ------------------------------------------------------------------------------
log = logutils.get_logger(__name__)
# ------------------------------------------------------------------------------
class Reduce:
    """
    The Reduce class encapsulates the core processing to be done by reduce.
    __init__ may receive one (1) parameter, nominally, an argparse Namespace
    instance. However, this object type is not required, but only that any
    passed object *must* present an equivalent interface to that of an
    <argparse.Namespace> instance, i.e. a duck type.

    The class provides one (1) public method, runr(), the only call needed to
    run reduce on the supplied argument set.

    Parameters
    ----------
    sys_args : :class:`argparse.Namespace` (optional) or <duck-type object>
            This object type is not required, per se, but only that any passed
            object *must* present an equivalent interface to that of an
            :class:`argparse.Namespace` instance.

    Attributes
    ----------
    adinputs: <list>
          attribute is a list of the input astrodata objects as made from
          the 'files' list (see 'files' below).

    output_filenames: <list>
          read-only property is a list of final output filenames.

    mode: <str>
          operational mode. Currently, only 'qa', 'sq' modes are supported.

    drpkg: <str>
          Data reduction package name. Default is 'geminidr', the Gemini
          Observatory data reduction package.

    files: <list>
          List of input filenames. Passed to Reduce.__init__(), these are
          converted to astrodata objects.

    suffix: <str>
          User supplied suffix to be applied as a final suffix to output
          filenames.

    ucals: <dict>
          Dictionary of calibration files passed by --user_cals flag.

    uparms: <dict>
          Dictionary of user parameters as passed by -p, --param flag.

    upload : <list>
          List of products to upload to fitsstore as passed by --upload.
          E.g.,
              --upload metrics calibs

         (==> upload == ['metrics', 'calibs'])
         will upload QA metrics to fitsstore and processing calibration
         files.

    recipename: <str> or callable
        The name of the recipe that will be run. If None, the 'default'
        recipe is used, as specified in the appropriate recipe library.

    """
    def __init__(self, sys_args=None):
        if sys_args:
            args = sys_args
        elif self._confirm_args():
            args = buildParser(__version__).parse_args()
        else:
            args = buildParser(__version__).parse_args([])

        # acquire any new astrodata classes.
        if args.adpkg:
            import_module(args.adpkg)

        self.mode = args.mode
        self.drpkg = args.drpkg
        self.files = args.files
        self.suffix = args.suffix
        self.ucals = normalize_ucals(args.user_cal)
        uparms = set_btypes(args.userparam)
        uparms = dict(uparms) if uparms else {}
        self.uparms = uparms
        self.config_file = args.config
        self._upload = args.upload
        self._output_filenames = None
        self.recipename = args.recipename if args.recipename else '_default'

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

    @property
    def output_filenames(self):
        return self._output_filenames

    def runr(self):
        """
        Map and run the requested or defaulted recipe.

        Parameters
        ----------
        <void>

        Returns
        -------
        <void>

        """
        self._output_filenames = reduce_data(files=self.files, mode=self.mode, drpkg=self.drpkg,
                                             recipename=self.recipename,
                                             uparms=self.uparms, ucals=self.ucals, upload=self.upload,
                                             config_file=self.config_file, suffix=self.suffix)


    # -------------------------------- prive -----------------------------------
    def _check_files(self, ffiles):
        """
        Sanity check on submitted files.

        Parameters
        --------
        ffiles: <list>
                list of passed FITS files.

        Return
        ------
        input_files: <list>
              list of 'good' input fits datasets.

        """
        try:
            assert ffiles
        except AssertionError:
            log.error("NO INPUT FILE(s) specified")
            log.stdinfo("type 'reduce -h' for usage information")
            raise OSError("NO INPUT FILE(s) specified")

        input_files = []
        bad_files = []

        for image in ffiles:
            if not os.access(image, os.R_OK):
                log.error('Cannot read file: '+str(image))
                bad_files.append(image)
            else:
                input_files.append(image)
        try:
            assert bad_files
            err = "\n\t".join(bad_files)
            log.warning("Files not found or cannot be loaded:\n\t%s" % err)
            if input_files:
                found = "\n\t".join(input_files)
                log.stdinfo("These datasets were loaded:\n\t%s" % found)
            else:
                log.error("Caller passed no valid input files")
                sys.exit(1)
        except AssertionError:
            log.stdinfo("All submitted files appear valid:")
            if len(input_files) > 1:
                filestr = input_files[0]
                filestr += " ... " + input_files[-1]
                filestr += ", {} files submitted.".format(len(input_files))
            else:
                filestr = input_files[0]

            log.stdinfo(filestr)
        return input_files

    def _convert_inputs(self, inputs):
        """
        Convert files into AstroData objects.

        Parameters
        ----------
        inputs: <list>, list of FITS file names

        Return
        ------
        allinputs: <list>, list of AstroData objects

        """
        allinputs = []
        for inp in inputs:
            try:
                ad = astrodata.open(inp)
            except AstroDataError as err:
                log.warning("Can't Load Dataset: %s" % inp)
                log.warning(err)
                continue
            except OSError as err:
                log.warning("Can't Load Dataset: %s" % inp)
                log.warning(err)
                continue

            if not len(ad):
                log.warning("%s contains no extensions." % ad.filename)
                continue

            allinputs.append(ad)

        return allinputs

    def _confirm_args(self):
        """
        Confirm that the first executable frame in the call stack is a reduce
        command line. This asserts that a nominal reduce parser, as returned by
        buildParser() function, is an equivalent Namespace object to that
        of an 'args' key in the stack's 'f_locals' namespace. If the Namespace
        objects are not equal, reduce is not calling this class.

        Parameters
        ----------
        <void>

        Returns
        -------
        is_reduce: <bool>,
            Did 'reduce' call this?

        """
        is_reduce = False
        exe_path = sys.argv[0]
        red_namespace = buildParser(__version__).parse_args([])
        if exe_path:
            cstack = inspect.stack()
            for local, value in list(cstack[-1][0].f_locals.items()):
                if local == 'args':
                    try:
                        assert(
                            list(value.__dict__.keys()) ==
                            list(red_namespace.__dict__.keys())
                        )
                        is_reduce = True
                    except AttributeError:
                        pass
                    except AssertionError:
                        log.stdinfo("A non-reduce command line was detected.")
                        pass

        return is_reduce


def _logheader(recipe, recipename):
    if recipename == '_default':
        r_actual = recipe.__name__
    else:
        r_actual = recipename

    logstring = "RECIPE: {}".format(r_actual)
    log.status("="*80)
    log.status(logstring)
    log.status("="*80)
    return


def _write_final(outputs, suffix):
    """
    Write final outputs. Write only if filename is not == orig_filename, or
    if there is a user suffix (self.suffix)

    Parameters
    ----------
    outputs: <list>, List of AstroData objects

    Return
    ------
    <void>

    """
    outstr = "\tWrote {} in output directory"
    def _sname(name):
        head, tail = os.path.splitext(name)
        ohead = head.split("_")[0]
        newname = ohead + suffix + tail
        return newname

    for ad in outputs:
        if suffix:
            username = _sname(ad.filename)
            ad.write(username, overwrite=True)
            log.stdinfo(outstr.format(username))
        elif ad.filename != ad.orig_filename:
            ad.write(ad.filename, overwrite=True)
            log.stdinfo(outstr.format(ad.filename))
    return


def _convert_inputs(inputs):
    """
    Convert files into AstroData objects.

    Parameters
    ----------
    inputs: <list>, list of FITS file names

    Return
    ------
    allinputs: <list>, list of AstroData objects

    """
    allinputs = []
    for inp in inputs:
        try:
            ad = astrodata.open(inp)
        except AstroDataError as err:
            log.warning("Can't Load Dataset: %s" % inp)
            log.warning(err)
            continue
        except OSError as err:
            log.warning("Can't Load Dataset: %s" % inp)
            log.warning(err)
            continue

        if not len(ad):
            log.warning("%s contains no extensions." % ad.filename)
            continue

        allinputs.append(ad)

    return allinputs


def _check_files(ffiles):
    """
    Sanity check on submitted files.

    Parameters
    --------
    ffiles: <list>
        list of passed FITS files.

    Return
    ------
    input_files: <list>
        list of 'good' input fits datasets.

    """
    try:
        assert ffiles
    except AssertionError:
        log.error("NO INPUT FILE(s) specified")
        log.stdinfo("type 'reduce -h' for usage information")
        raise OSError("NO INPUT FILE(s) specified")

    input_files = []
    bad_files = []

    for image in ffiles:
        if not os.access(image, os.R_OK):
            log.error('Cannot read file: '+str(image))
            bad_files.append(image)
        else:
            input_files.append(image)
    try:
        assert bad_files
        err = "\n\t".join(bad_files)
        log.warning("Files not found or cannot be loaded:\n\t%s" % err)
        try:
            assert input_files
            found = "\n\t".join(input_files)
            log.stdinfo("These datasets were loaded:\n\t%s" % found)
        except AssertionError:
            log.error("Caller passed no valid input files")
            raise OSError("No valid files passed.")
    except AssertionError:
        log.stdinfo("All submitted files appear valid:")
        if len(input_files) > 1:
            filestr = input_files[0]
            filestr += " ... " + input_files[-1]
            filestr += ", {} files submitted.".format(len(input_files))
        else:
            filestr = input_files[0]

        log.stdinfo(filestr)
    return input_files


def _log_reduce(files, mode, drpkg, recipename, uparms, ucals, upload, config_file, suffix):
    if files:
        log.debug("Files".ljust(33) + ":: " + files[0])
        for f in files[1:]:
            log.debug(" "*33 + ":: {}".format(f))
    log.debug("Mode".ljust(33) + ":: " + mode)
    log.debug("Data Reduction Package".ljust(33) + ":: " + drpkg)
    log.debug("Recipe Name".ljust(33) + ":: " + recipename)
    if uparms:
        log.debug("\nParameters")
        log.debug("----------")
        if isinstance(uparms, dict):
            for param in uparms.keys():
                log.debug(param.ljust(33) + ":: " + str(uparms[param]))
        elif isinstance(uparms, list):
            for param in uparms:
                if len(param) == 2:
                    log.debug(param[0].ljust(33) + ":: " + str(param[1]))
                else:
                    log.debug("Unrecognized parameter".ljust(33) + ":: " + param)
        else:
            log.debug("(Unrecognized parameters)")
    if ucals:
        log.debug("\nCalibrations")
        log.debug("------------")
        for cal in ucals.keys():
            log.debug(cal.ljust(33) + ":: " + ucals[cal])
    if upload:
        log.debug("%s:: %s" % ("Upload".ljust(33), upload))
    if config_file:
        log.debug("Config File".ljust(33) + ":: " + config_file)
    if suffix:
        log.debug("Suffix".ljust(33) + ":: " + suffix)
    log.debug("-"*65+"\n")


def reduce_data(files, mode='sq', drpkg='geminidr', recipename=None, uparms={}, ucals={},
                upload=None, config_file=None, suffix=None, logmode=None):
    """
    Map and run the requested or defaulted recipe.

    Parameters
    ----------
    files : <list> or str
        The set of files to reduce, if a string it is assumed to be a single file
    mode : <str>
        The mode of reduction: ``qa`` ``ql`` or ``sq``, defaults to ``sq``
    drpkg :<str>
        The data reduction package to map. Default is 'geminidr'.
        This package *must* be importable.
    recipename : <str>
        The name of the recipe or primitive to run, or None for default
    uparms : <dict>
        The parameters for the recipes, if any
    ucals : <dict>
        Calibration files to use, if any, as a dictionary mapping calibration type to file, ``None`` for default
    upload : <list>
        List of types to upload, default None
    config_file : str
        Configuration file to use, None for default
    suffix : str
        Suffix to add to output file(s), None for default
    logmode : str
        Mode of logging such as 'debug', or None for default

    Returns
    -------
    <list> : List of files produced by the reduction
    """
    if logmode is not None:
        # User requesting an override of the logging mode
        from gempy.utils import logutils
        logutils.config(mode=logmode)

    _log_reduce(files, mode, drpkg, recipename, uparms, ucals, upload, config_file, suffix)
    recipe = None
    if isinstance(files, str):
        files = [files,]
    try:
        ffiles = _check_files(files)
    except OSError as err:
        log.error(str(err))
        raise

    try:
        adinputs = _convert_inputs(ffiles)
    except OSError as err:
        log.error(str(err))
        raise

    upload = [seg.lower().strip() for seg in upload.split(',')] if isinstance(upload, str) else upload

    # build mapper inputs, pass no 'ad' objects.
    # mappers now receive tags and instr pkg name, e.g., 'gmos'
    datatags = set(list(adinputs[0].tags)[:])
    instpkg = adinputs[0].instrument(generic=True).lower()

    rm = RecipeMapper(datatags, instpkg, mode=mode, drpkg=drpkg,
                      recipename=recipename)
    try:
        recipe = rm.get_applicable_recipe()
    except ModeError as err:
        log.warning("WARNING: {}".format(err))
        pass
    except RecipeNotFound:
        log.warning("No recipe can be found in {} recipe libs.".format(instpkg))
        log.warning("Searching primitives ...")
        # If it's not a primitive, we'll crash later on, so assume it's a
        # primitive and update uparms to prepend the name to any parameters
        # without the primitive named explicitly
        uparms = [((k if ':' in k else f"{recipename}:{k}"), v)
                       for k, v in uparms.items()]

    # clear reference for GC
    rm = None

    # PrimitiveMapper now returns the primitive class, not an instance.
    pm = PrimitiveMapper(datatags, instpkg, mode=mode, drpkg=drpkg)
    try:
        pclass = pm.get_applicable_primitives()
    except PrimitivesNotFound as err:
        log.error(str(err))
        raise

    p = pclass(adinputs, mode=mode, ucals=ucals, uparms=uparms,
               upload=upload, config_file=config_file)

    # Clean references to avoid keeping adinputs objects in memory one
    # there are no more needed.
    adinputs = None

    # If the RecipeMapper was unable to find a specified user recipe,
    # it is possible that the recipe passed was a primitive name.
    # Here we examine the primitive set to see if this recipe is actually
    # a primitive name.
    norec_msg = "{} recipes do not define a '{}' recipe for these data."
    if recipe is None and recipename == '_default':
        raise RecipeNotFound(norec_msg.format(instpkg.upper(), mode))

    if recipe is None:
        try:
            primitive_as_recipe = getattr(p, recipename)
        except AttributeError as err:
            err = "Recipe {} Not Found".format(recipename)
            log.error(str(err))
            raise RecipeNotFound("No primitive named {}".format(recipename))

        pname = primitive_as_recipe.__name__
        log.stdinfo("Found '{}' as a primitive.".format(pname))
        _logheader(pname, recipename)
        try:
            primitive_as_recipe()
        except Exception as err:
            log.error("Reduce received an unhandled exception.", exc_info=True)
            raise
    else:
        _logheader(recipe, recipename)
        try:
            recipe(p)
        except Exception:
            log.error("Reduce received an unhandled exception. Aborting ...",
                      exc_info=True)
            log.stdinfo("Writing final outputs ...")
            try:
                for ad in p.streams['main']:
                    ad.update_filename(suffix="_crash")
            except:  # in case something has gone really wrong!
                log.stdinfo("Problem updating filenames")
            _write_final(p.streams['main'], None)
            _output_filenames = [ad.filename for ad in p.streams['main']]
            raise

    _write_final(p.streams['main'], suffix)
    _output_filenames = [ad.filename for ad in p.streams['main']]

    log.stdinfo("reduce completed successfully.")

    return _output_filenames
