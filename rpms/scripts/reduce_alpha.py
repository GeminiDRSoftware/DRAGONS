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
_version = ' alpha (new_hope) '
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

from rpms.utils.reduce_utils import buildParser
from rpms.utils.reduce_utils import normalize_args
from rpms.utils.reduce_utils import set_btypes
from rpms.utils.reduce_utils import show_parser_options

# ------------------------------------------------------------------------------
class ReduceNH(object):
    """
    The ReduceNH class encapsulates the core processing to be done by reduce.
    __init__ may receive one (1) parameter, nominally, an argparse Namespace 
    instance. However, this object type is not required, but only that any 
    passed object *must* present an equivalent interface to that of an
    <argparse.Namespace> instance.

    The class provides one (1) public method, runnh(), the only call needed to
    run reduce on the supplied argument set.

    """
    def __init__(self, sys_args=None):
        """
        :parameter sys_args: optional argparse.Namespace instance
        :type sys_args: <Nameapace> 

        :return: ReduceNH instance
        :rtype: <ReduceNH>

        """

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
        """
        Map and run the requested or defaulted recipe.

        :parameters: <void>

        :returns: exit code 
        :rtype: <int>

        """
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

        rm = RecipeMapper(self.adinputs, recipename=self.urecipe, 
                              context=self.context, uparms=self.uparms)
        rm.set_recipe_library()
        try:
            recipe = rm.get_applicable_recipe()
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
        
        :parameter ffiles: list of passed FITS files.
        :type ffiles: <list>

        :return: list of 'good' input fits datasets.
        :rtype: <list>

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
            r_actual = recipe.__name__
        else:
            r_actual = self.urecipe

        logstring = "RECIPE: {}".format(r_actual)
        log.status("="*80)
        log.status(logstring)
        log.status("="*80)
        return

# ------------------------------------------------------------------------------
def main(args):
    """
    'main' is called with a Namespace 'args' parameter, or an object that 
    presents an equivalent interface.
    
    Eg.,
    
    Get "args' from the defined reduce parser:
    
    >>> args = buildParser(version).parse_args()
    >>> import reduce_alpha
    >>> reduce_alpha.main(args)
    
    In the above example, 'args' is

    -- argparse Namespace instance
    
    Use of the reduce_utils function buildParser will get the caller a fully defined 
    reduce Namespace instance, values for which can be then be adjusted as desired.
    
    Eg.,
    
    buildParser:
    -----------
    >>> args = buildParser(version).parse_args()
    >>> args.logfile
    'reduce.log'
    >>> args.files
    []
    >>> args.files.append('some_fits_file.fits')
    
    Once 'args' attributes have been appropriately set, the caller then simply 
    calls main():
    
    >>> reduce_alpha.main(args)

    :parameter args: argparse Namespace object
    :type args: <Namespace>

    :return: exit code
    :rtype:  <int>

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
