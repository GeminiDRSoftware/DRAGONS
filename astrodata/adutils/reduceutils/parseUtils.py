#
#                                                                  gemini_python
#
#                                                        astrodata.(reduceutils)
#                                                                  parseUtils.py
# ------------------------------------------------------------------------------
# $Id$
# ------------------------------------------------------------------------------
__version__      = '$Revision$'[11:-3]
__version_date__ = '$Date$'[7:-3]
# ------------------------------------------------------------------------------
# This cli parser employs argparse rather than the depracated optparse.
#
# kra 04-06-13
#
# This module provides upgraded reduce command line argument handling, employing
# the ArgumentParser "fromfile" facility, customized "action" classes, and a 
# customized help formatter, the ReduceHelpFormatter class.
# ------------------------------------------------------------------------------
""" This module provides command line parsing facilities tuned to reduce and 
GDPSG requirements on handling reduce arguments.
"""
import os
import re
import sys

from argparse import ArgumentParser
from argparse import HelpFormatter

from astrodata.RecipeManager import UserParam, UserParams
from astrodata.AstroDataType import get_classification_library

from astrodata.adutils import logutils
from astrodata.adutils import gemLog, strutil

#Do not know where 'reduceActions' is going to be yet ...
from reduceActions import PosArgAction
from reduceActions import BooleanAction 
from reduceActions import ParameterAction
from reduceActions import CalibrationAction
from reduceActions import UnitaryArgumentAction

# ------------------------------------------------------------------------------
log = logutils.get_logger(__name__)

# ------------------------------------------------------------------------------
class ReduceHelpFormatter(HelpFormatter):
    """ReduceHelpFormatter class overrides default help formatting on customized
    reduce actions.
    """
    def _format_args(self, action, default_metavar):
        get_metavar = self._metavar_formatter(action, default_metavar)
        if action.nargs is None:
            result = '%s' % get_metavar(1)
        elif isinstance(action, BooleanAction):
            result = ''
        elif isinstance(action, PosArgAction):
            result = '%s [%s ...]' % get_metavar(2)
        elif isinstance(action, UnitaryArgumentAction):
            result = '%s' % get_metavar(1)
        elif isinstance(action, ParameterAction):
            result = '%s [%s ...]' % get_metavar(2)
        elif isinstance(action, CalibrationAction):
            result = '%s [%s ...]' % get_metavar(2)
        else:
            formats = ['%s' for _ in range(action.nargs)]
            result = ' '.join(formats) % get_metavar(action.nargs)
        return result

# ------------------------------------------------------------------------------
class ReduceArgumentParser(ArgumentParser):
    """
    Converts an argument line from a user param file into an actual argument,
    yields to the calling parser.
    """
    def convert_arg_line_to_args(self, arg_line):
        if not arg_line.startswith("#"):
            for arg in arg_line.split():
                if not arg.strip():
                    continue
                if arg.strip().startswith("#"):
                    break
                yield arg

# ------------------------------------------------------------------------------
def buildParser(version):
    parser = ReduceArgumentParser(description="_"*11 + 
                                  " Gemini Observatory Recipe System Processor "
                                  " (v1.1 2013) " + "_"*10 + "\n" + "_"*30 +\
                                  " Written by GDPSG " + "_"*29, prog="reduce2",
                                  formatter_class=ReduceHelpFormatter,
                                  fromfile_prefix_chars='@')

    parser.add_argument("-v", "--version", action='version',
                        version='%(prog)s v'+ version)

    parser.add_argument("-d", "--displayflags", dest='displayflags',
                        default=False,
                        nargs='*', action=BooleanAction,
                        help="display all parsed option flags.")

    parser.add_argument('files', metavar='fitsfile', nargs = "*",
                        action=PosArgAction, default=[],
                        help="fitsfile [fitsfile ...] ")

    parser.add_argument("-i", "--intelligence", dest='intelligence', nargs="*",
                        default=False, action=BooleanAction, 
                        help="Endow recipe system with intelligence to perform "
                        "operations faster and smoother")

    parser.add_argument("-m", "--monitor", dest="bMonitor", default=False,
                        nargs="*", action=BooleanAction,
                        help="Open TkInter window to monitor progress of "
                        "execution (NOTE: One window will open per recipe run)")

    parser.add_argument("-p", "--param", dest="userparam", default=None,
                        nargs="*", action=ParameterAction,
                        help="Set a parameter from the command line. The form  "
                        "'-p par=val' sets parameter in the reduction context "
                        "such that all primitives will 'see' it.  The form: "
                        "'-p ASTROTYPE:primitivename:par=val', sets the "
                        "parameter such that it applies only when the current "
                        "reduction type (type of current reference image) "
                        "is 'ASTROTYPE' and the primitive is 'primitivename'. "
                        "Separate par/val pairs by whitespace: "
                        "(eg. '-p par1=val1 par2=val2')")

    parser.add_argument("--context", dest="running_contexts", default=None,
                        nargs="*", action=UnitaryArgumentAction,
                        help="provides general 'context name' for primitives"
                        " sensitive to context.")

    parser.add_argument("-r", "--recipe", dest="recipename", default=None,
                        nargs="*", action=UnitaryArgumentAction,
                        help="specify which recipe to run by name")

    parser.add_argument("-t", "--astrotype", dest="astrotype", default=None,
                        nargs="*", action=UnitaryArgumentAction,
                        help="Run a recipe based on astrotype (either overrides"
                        " default type, or begins without initial input. Eg. "
                        "recipes that begin with primitives that acquire data)")

    ##@@FIXME: This next option should not be put into the package
    parser.add_argument("--rtf-mode", dest="rtf", default=False,
                        nargs="*", action=BooleanAction,
                        help="Use rtf mode.")

    parser.add_argument("--throw_descriptor_exceptions", 
                        dest = "throwDescriptorExceptions", default=False,
                        nargs="*", action=BooleanAction,
                        help="Throw exceptions when Descriptors fail")

    parser.add_argument("--addprimset", dest="primsetname", default = None,
                        nargs="*", action=UnitaryArgumentAction,
                        help="add user supplied primitives to reduction."
                        "A primitives module or path to a primitives module.")

    parser.add_argument("--calmgr", dest="cal_mgr", default=None,
                        help="calibration manager url overides lookup table")

    parser.add_argument("--force-height", dest="forceHeight", default=None,
                        nargs="*", action=UnitaryArgumentAction,
                        help="force height of terminal output")

    parser.add_argument("--force-width", dest="forceWidth", default=None,
                        nargs="*", action=UnitaryArgumentAction,
                        help="force width of terminal output")

    parser.add_argument("--invoked", dest="invoked", default=False, 
                        nargs="*", action=BooleanAction,
                        help="tell user reduce invoked by adcc")

    parser.add_argument("--logmode", dest="logmode", default="standard",
                        nargs="*", action=UnitaryArgumentAction,
                        help="Set log mode (standard, console, debug, null)")

    parser.add_argument("--logfile", dest="logfile", default="reduce.log",
                        nargs="*", action=UnitaryArgumentAction,
                        help="name of log (default = 'reduce.log')") 

    parser.add_argument("--loglevel", dest="loglevel", default="stdinfo", 
                        nargs="*", action=UnitaryArgumentAction,
                        help="Set the verbose level for console "
                        "logging; (critical, error, warning, status, stdinfo, "
                        "fullinfo, debug)")

    parser.add_argument("--override_cal", dest="user_cals", default=None,
                        nargs="*", action=CalibrationAction,
                        help="Add calibration to User Calibration Service. "
                        "'--override_cal CALTYPE_1:CAL_PATH_1 CALTYPE_N:CAL_PATH_N' "
                        "Eg., --override_cal processed_arc:wcal/gsN20011112S064_arc.fits ")

    parser.add_argument("--writeInt", dest='writeInt', default=False,
                        nargs="*", action=BooleanAction,
                        help="Write intermediate outputs (UNDER CONSTRUCTION)")

    parser.add_argument("--suffix", dest='suffix', default=None,
                        nargs="*", action=UnitaryArgumentAction,
                        help="Add 'suffix' to filenames at end of reduction.")    
    return parser

# --------------------------- Emulation functions ------------------------------
# The functions below encapsulate ArgumentParser access to option strings and 
# matches them to 'dest' attributes and attribute values. There is no public 
# interface as with OptionParser.has_option() and OptionParser.get_option() for
# testing and getting option flags.

# The functions
#
#     parser_has_option()
#     get_option_flags()
#
# emulate those methods.
#
#     insert_option_value()  -- assigns an option value to matching 'dest' attr
#     show_parser_options()  -- pretty print options, 'dest' attrs, values.
# ------------------------------------------------------------------------------

def parser_has_option(parser, option):
    return parser.__dict__['_option_string_actions'].has_key(option)

def get_option_flags(parser, option):
    return parser.__dict__['_option_string_actions'][option].__dict__['option_strings']

def insert_option_value(parser, args, option, value):
    exec("args.%s=value" % \
         str(parser.__dict__['_option_string_actions'][option].__dict__['dest']))
    return

def show_parser_options(parser, args):
    all_opts = parser.__dict__['_option_string_actions'].keys()
    handled_flag_set = []
    print "\n\t"+"-"*25+" switches, vars, vals "+"-"*20+"\n"
    print "\t  Literals\t\t\tvar 'dest'\t\tValue"
    print "\t", "-"*60
    for opt in all_opts:
        all_option_flags = get_option_flags(parser, opt)
        if opt in handled_flag_set:
            continue
        elif "--help" in all_option_flags:
            continue
        elif "--version" in all_option_flags:
            continue
        else:
            handled_flag_set.extend(all_option_flags)
            dvar = parser.__dict__['_option_string_actions'][opt].__dict__['dest']
            val = args.__dict__[dvar]
            if len(all_option_flags) == 1 and len(dvar) == 3:
                print "\t", all_option_flags, "\t"*3,"::", dvar, "\t\t\t::", val
                continue
            if len(all_option_flags) == 1 and (12 < len(dvar) < 17):
                print "\t", all_option_flags, "\t"*3,"::", dvar, "\t::", val
                continue
            if len(all_option_flags) == 1 and len(all_option_flags[0]) > 24:
                print "\t", all_option_flags, "::", dvar, "\t::", val
                continue
            elif len(all_option_flags) == 1 and len(all_option_flags[0]) < 11:
                print "\t", all_option_flags, "\t"*3+"::", dvar, "\t\t::", val
                continue
            elif len(all_option_flags) == 2 and len(all_option_flags[1]) > 12:
                print "\t", all_option_flags, "\t"+"::", dvar, "\t::", val
                continue
            elif len(all_option_flags) == 2:
                print "\t", all_option_flags, "\t"*2+"::", dvar, "\t\t::", val
                continue
            else: 
                print "\t", all_option_flags, "\t"*2+"::", dvar, "\t\t::", val
    print "\t"+"-"*60+"\n"
    return

# ------------------------------------------------------------------------------
def abortBadParamfile(lines):
    for i in range(len(lines)):
        log.error("  %03d:%s" % (i, lines[i]))
    log.error("  %03d:<<stopped parsing due to error>>" % (i+1))
    sys.exit(1)
    return

# ------------------------------------------------------------------------------
def check_files(args):
    """
    Sanity check on submitted files.
    """
    try:
        assert(args.files or args.astrotype)
    except AssertionError:
        log.info("Either file(s) OR an astrotype is required;"
                 "-t or --astrotype.")
        log.error("NO INPUT FILE or ASTROTYPE specified")
        log.info("type 'reduce -h' for usage information")
        sys.exit(1)

    input_files = []
    bad_files   = []

    for image in args.files:
        if not os.access(image, os.R_OK):
            log.error('Cannot read file: '+str(image))   
            log.warning("Some files not found or cannot be opened:\n\t" + image)
            bad_files.append(image)
        else:
            input_files.append(image)

    try:
        assert(bad_files)
        print "Got a badList ... ", bad_files
        print "I.e. File not found or unreadable."
        err = "\n\t".join(bad_files)
        log.error("Some files not found or can't be loaded:\n\t" + err)
        log.error("Exiting due to missing datasets.")
        try:
            assert(input_files)
            found = "\n\t".join(input_files)
            log.info("These datasets were found and loaded:\n\t" + found)
        except AssertionError:
            print "Got no input files"
            pass
        sys.exit(1)
    except AssertionError:
            return input_files

# ------------------------------------------------------------------------------
def set_user_params(userparams):
    """
    Convert cli user parameters into UserParam objects. If a user parameter
    is typed, it must be fully typed, i.e. <adtype>:<primitive>:par=value.
    See the reduce usage for -p --param.

    parameters: <list>, <fileobj>, args.userparams, log file.
    return:     <tuple>, (UserParams obj, <dict>)
    """
    fups   = UserParams() # a set of UserParam objects
    gparms = {}
    ups    = []

    if userparams:
        for upar in userparams:
            tmp = upar.split("=")
            spec, val = tmp[0].strip(), tmp[1].strip()
            if val == 'None':
                val = None
            if ":" in spec:
                typ, prim, param = spec.split(":")
                up = UserParam(typ, prim, param, val)
                ups.append(up)
            else:
                up = UserParam(None, None, spec, val)
                ups.append(up)
                gparms.update({spec:val})
        [fups.add_user_param(up) for up in ups]
    else:
        pass

    # typed parameters and global (untyped) parameters.
    return fups, gparms


def normalize_args(args):
    """
    Convert argparse argument lists to single string values.

    parameters: <Namespace>, argparse Namespace object
    return:     <Namespace>, with converted types.
    """

    if isinstance(args.astrotype, list):
        args.astrotype = args.astrotype[0]
    if isinstance(args.recipename, list):
        args.recipename = args.recipename[0]
    if isinstance(args.running_contexts, list):
        args.running_contexts = args.running_contexts[0]
    if isinstance(args.loglevel, list):
        args.loglevel = args.loglevel[0]
    if isinstance(args.logmode, list):
        args.logmode = args.logmode[0]
    if isinstance(args.logfile, list):
        args.logfile = args.logfile[0]
    if isinstance(args.primsetname, list):
        args.primsetname = args.primsetname[0]
    if isinstance(args.cal_mgr, list):
        args.cal_mgr = args.cal_mgr[0]
    if isinstance(args.suffix, list):
        args.suffix = args.suffix[0]
    if isinstance(args.forceHeight, list):
        args.forceHeight = args.forceHeight[0]
    if isinstance(args.forceWidth, list):
        args.forceWidth = args.forceWidth[0]
    return args
