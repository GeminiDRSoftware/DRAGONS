#!/usr/bin/env python
#
#                                                                  gemini_python
#
#                                                                reduce_alpha.py
# ------------------------------------------------------------------------------
# $Id$
# ------------------------------------------------------------------------------
__version__      = '$Rev$'[6:-1]
__version_date__ = '$Date$'[7:-3]
# ------------------------------------------------------------------------------
# reduce_alpha.py -- The new hope reduce prototype. Demonstrates command line
# interface with the RecipeMapper class. 

# ------------------------------------------------------------------------------
_version = '2.0 (new_hope_alpha)'
# ------------------------------------------------------------------------------
"""
Prototype reduce (New Hope).

"""
# ---------------------------- Package Import ----------------------------------
import os
import sys
import inspect
import signal

from astrodata import AstroData

from astrodata.utils import Errors
from astrodata.utils import logutils

from rpms.recipeMapper import RecipeMapper
from rpms.reduce_utils import set_btypes
from rpms.reduce_utils import buildParser
from rpms.reduce_utils import normalize_args
from rpms.reduce_utils import show_parser_options
# ------------------------------------------------------------------------------
class ReduceNH(object):
    """
    The Reduce class encapsulates the core processing to be done by reduce.
    The constructor may receive one (1) parameter, which will be an instance
    of a parse_args call on a reduce-defined ArgumentParser object. As with
    all constructors, an instance of this class is returned.

    parameters: <instance>, optional ArgumentParser.parse_args() instance
    return:     <instance>, Reduce instance

    The class provides one (1) public method, runr(), the only call needed to
    run reduce on the supplied argument set.
    """
    def __init__(self, sys_args=None):
        if sys_args:
            args = sys_args
        elif self._confirm_args():
            args = buildParser(__version__).parse_args()
        else:
            args = buildParser(__version__).parse_args([])

        self.adinputs = None
        self.files = args.files
        self.uparms = set_btypes(args.userparam)
        self.context = args.context if args.context else 'QA'
        self.urecipe = args.recipename if args.recipename else 'default'

    def runnh(self):
        xstat = 0
        try:
            ffiles = self._check_files(self.files)
        except IOError, err:
            xstat = signal.SIGIO
            log.error("_check_files() raised IOError exception.")
            log.error(str(err))
            return xstat

        try:
            self.adinputs = self._convert_inputs(ffiles)
        except IOError, err:
            xstat = signal.SIGIO
            log.error("_convert_inputs() raised IOError exception.")
            log.error(str(err))
            return xstat

        for ad in self.adinputs:
            rm = RecipeMapper(ad, recipename=self.urecipe, 
                              context=self.context, uparms=self.uparms)
            rm.set_recipe_library()
            try:
                recipe = rm.get_recipe_actual()
            except Errors.RecipeNotFoundError, err:
                xstat = signal.SIGIO
                log.error(str(err))
                return xstat

            p = rm.get_applicable_primitives()
            self._logheader(recipe)
            recipe(p)

        return xstat

# -------------------------------- prive ---------------------------------------
    def _check_files(self, ffiles):
        """
        Sanity check on submitted files.
        
        parameters ffiles: list of passed FITS files.
        return:     <list>, list of 'good' input fits datasets.

        """
        try:
            assert ffiles
        except AssertionError:
            log.error("NO INPUT FILE(s) specified")
            log.stdinfo("type 'reduce -h' for usage information")
            raise IOError("NO INPUT FILE(s) specified")

        input_files = []
        bad_files   = []

        for image in ffiles:
            if not os.access(image, os.R_OK):
                log.error('Cannot read file: '+str(image))
                bad_files.append(image)
            else:
                input_files.append(image)
        try:
            assert(bad_files)
            err = "\n\t".join(bad_files)
            log.warn("Files not found or cannot be loaded:\n\t%s" % err)
            try:
                assert(input_files)
                found = "\n\t".join(input_files)
                log.stdinfo("These datasets were found and loaded:\n\t%s" % found)
            except AssertionError:
                log.error("Caller passed no valid input files")
                raise IOError("No valid files passed.")
        except AssertionError:
            log.stdinfo("All submitted files appear valid")

        return input_files

    def _convert_inputs(self, inputs):
        """
        Convert files into AstroData objects.
        
        :parameter inputs: list of FITS file names
        :type inputs: <list>

        :return: list of AstroData objects
        :rtype: <list> 

        """
        allinputs = []
        for inp in inputs:
            try:
                ad = AstroData(inp)
                ad.filename = os.path.basename(ad.filename)
                ad.mode = "readonly"
            except Errors.AstroDataError, err:
                log.warning("Can't Load Dataset: %s" % inp)
                log.warning(err)
                continue
            except ValueError, err:
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

        :parameters: <void>
        :returns: Value of whether 'reduce' or some other executable is
                  instantiating this class.
        :rtype: <bool>
 
        """
        is_reduce = False
        exe_path = sys.argv[0]
        red_namespace = buildParser(_version).parse_args([])
        if exe_path:
            cstack = inspect.stack()
            for local, value in cstack[-1][0].f_locals.items():
                if local == 'args':
                    try:
                        assert value.__dict__.keys() == red_namespace.__dict__.keys()
                        is_reduce = True
                    except AssertionError:
                        log.stdinfo("A non-reduce command line was detected.")
                        pass

        return is_reduce

    def _logheader(self, recipe):
        if self.urecipe == 'default':
            r_actual = self._recipe_actual(recipe) + " ({})".format(self.urecipe)
        else:
            r_actual = self.urecipe

        logstring = "RECIPE: {}".format(r_actual)
        log.status("="*80)
        log.status(logstring)
        log.status("="*80)
        return

    def _recipe_actual(self, recipe):
        rname = inspect.getsourcelines(recipe)[0][1].strip().split('(')[0]
        return rname

# ------------------------------------------------------------------------------
def main(args):
    """
    See the module docstring on how to call main.

    parameters: <inst>, 'args' object
    return:     <int>,   exit code

    """
    global log
    estat = 0
    log = logutils.get_logger(__name__)
    try:
        assert log.root.handlers
        log.root.handlers = []
        logutils.config(mode=args.logmode, console_lvl=args.loglevel,
                        file_name=args.logfile)
        log = logutils.get_logger(__name__)
        log.info("Logging configured for application: reduce")
        log.info(" ")
    except AssertionError:
        pass


    log.stdinfo("\t\t\t--- reduce, v%s ---" % _version)
    r_reduce = ReduceNH(args)
    estat = r_reduce.runnh()
    if estat != 0:
        log.stdinfo("\n\nreduce exit status: %d\n" % estat)
    else:
        pass
    return estat

# --------------------------------------------------------------------------
if __name__ == "__main__":
    version_report = _version
    parser = buildParser(version_report)
    args = parser.parse_args()

    if args.displayflags:
        show_parser_options(parser, args)
        for item in ["Input fits file(s):\t%s" % inf for inf in args.files]:
            print item
        sys.exit()

    # Deal with argparse structures that are different than optparse 
    args = normalize_args(args)
    sys.exit(main(args))
