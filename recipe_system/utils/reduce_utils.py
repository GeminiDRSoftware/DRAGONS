#
#                                                                reduce_utils.py
# ------------------------------------------------------------------------------
# Utility function library for reduce and the Reduce class.
__version__ = "2.0.0 (beta)"
# ------------------------------------------------------------------------------
from argparse import ArgumentParser
from argparse import HelpFormatter

import astrodata
import gemini_instruments

from .reduceActions import PosArgAction
from .reduceActions import BooleanAction
from .reduceActions import ParameterAction
from .reduceActions import CalibrationAction
from .reduceActions import UnitaryArgumentAction

from ..cal_service import localmanager_available

# ------------------------------------------------------------------------------
class ReduceHelpFormatter(HelpFormatter):
    """
    ReduceHelpFormatter class overrides default help formatting on customized
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
    parser = ReduceArgumentParser(description="_"*29 + " Gemini Observatory " +
                                  "_"*28 + "\n" + "_"*20 +
                                  " Recipe Processing Management System " +
                                  "_"*20 + "\n" + "_"*22 +
                                  " Recipe System Release"+version+" "+"_"*22,
                                  prog="reduce",
                                  formatter_class=ReduceHelpFormatter,
                                  fromfile_prefix_chars='@')

    parser.add_argument("-v", "--version", action='version',
                        version='%(prog)s v'+ version)

    parser.add_argument("-d", "--displayflags", dest='displayflags',
                        default=False, nargs='*', action=BooleanAction,
                        help="display all parsed option flags and exit.")

    parser.add_argument('files', metavar='fitsfile', nargs = "*",
                        action=PosArgAction, default=[],
                        help="fitsfile [fitsfile ...] ")

    parser.add_argument("--context", dest="context", default=None,
                        nargs="*", action=UnitaryArgumentAction,
                        help="Use <context> for recipe selection and "
                        " primitives sensitive to context. Eg., --context QA")

    parser.add_argument("--drpkg", dest='drpkg', default='geminidr',
                        nargs="*", action=UnitaryArgumentAction,
                        help="Specify another data reduction (dr) package. "
                        "The package must be importable either through sys.path "
                        "or a user's PYTHONPATH. Recipe system default is "
                        "'geminidr'. E.g., --drpkg ghostdr ")

    parser.add_argument("--logfile", dest="logfile", default="reduce.log",
                        nargs="*", action=UnitaryArgumentAction,
                        help="name of log (default is 'reduce.log')")

    parser.add_argument("--loglevel", dest="loglevel", default="stdinfo",
                        nargs="*", action=UnitaryArgumentAction,
                        help="Set the verbose level for console "
                        "logging; (critical, error, warning, status, stdinfo, "
                        "fullinfo, debug)")

    parser.add_argument("--logmode", dest="logmode", default="standard",
                        nargs="*", action=UnitaryArgumentAction,
                        help="Set log mode: 'standard', 'console', 'quiet', "
                        "'debug', or 'null'.")

    parser.add_argument("-p", "--param", dest="userparam", default=None,
                        nargs="*", action=ParameterAction,
                        help="Set a parameter from the command line. The form "
                        "'-p par=val' sets a parameter such that all primitives "
                        "with that defined parameter will 'see' it.  The form: "
                        "'-p primitivename:par=val', sets the parameter only "
                        "for 'primitivename'. Separate par/val pairs by "
                        "whitespace: "
                        "(eg. '-p par1=val1 par2=val2')")

    parser.add_argument("-r", "--recipe", dest="recipename", default=None,
                        nargs="*", action=UnitaryArgumentAction,
                        help="Specify a recipe by name. Users can request "
                        "non-default system recipe functions by their simple "
                        "names, e.g., -r qaStack, OR may specify their own "
                        "recipe file and recipe function. A user defined "
                        "recipe function must be 'dotted' with the recipe file."
                        " E.g., "
                        " '-r /path/to/recipes/recipefile.recipe_function' "
                        "For a recipe file in the current working directory "
                        "(cwd), only the file name is needed, as in, "
                        "'-r recipefile.recipe_function' "
                        "The fact that the recipe function is dotted with the "
                        "recipe file name implies that multiple user defined "
                        "recipe functions can be defined in a single file. "
                        "Readers should understand that these recipe files "
                        "shall behave as python modules and should be named "
                        "accordingly. I.e., in the example above, 'recipefile'"
                        "is a python module named,  'recipefile.py' ")

    parser.add_argument("--suffix", dest='suffix', default=None,
                        nargs="*", action=UnitaryArgumentAction,
                        help="Add 'suffix' to filenames at end of reduction; "
                        "strip all other suffixes marked by '_'; ")

    parser.add_argument("--upload_metrics", dest='upmetrics', default=False,
                        action=BooleanAction, nargs="*",
                        help="Send QA metrics to fitsstore. Default is False."
                        "Eg., --upload_metrics")

    parser.add_argument("--user_cal", dest='user_cal', default=None,
                        nargs="*", action=CalibrationAction,
                        help="Specify user supplied calibrations for "
                        "calibration types. "
                        "Eg., --user_cal processed_arc:gsTest_arc.fits")

    if localmanager_available:
        parser.add_argument("--local_db_dir", dest='local_db_dir', default=None,
                            nargs="*", action=UnitaryArgumentAction,
                            help="Point to a directory where the local database "
                            "for calibration association can be found. Default "
                            "is to look it up in the config file (if any).")
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
    return option in parser._option_string_actions

def get_option_flags(parser, option):
    return parser._option_string_actions[option].option_strings

def insert_option_value(parser, args, option, value):
    dest = parser._option_string_actions[option].dest
    setattr(args, dest, value)
    return

def show_parser_options(parser, args):
    all_opts = parser.__dict__['_option_string_actions'].keys()
    handled_flag_set = []
    print "\n\t"+"-"*20+"   switches, vars, vals  "+"-"*20+"\n"
    print "\t  Literals\t\t\tvar 'dest'\t\tValue"
    print "\t", "-"*65
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
            fmt1 = "\t{}".format(all_option_flags)
            fmt2 = ":: {} ".format(dvar)
            fmt3 = ":: {}".format(val)
            fmtf = fmt1.ljust(33) + fmt2.ljust(24) + fmt3
            print fmtf
    print "\t"+"-"*65+"\n"
    return

# ------------------------------------------------------------------------------
def set_btypes(userparams):
    """
    All cmd line args are delivered as strings. Find any user parameters that
    should be other python types and set them to those actual corresponding types.

    I.e.,

        'None'  --> None
        'True'  --> True
        'False' --> False

    :parameters userparams: user parameters (if any) passed on the command line.
    :type userparms: <list>

    :returns: A tuple of same parameters with converted None and boolean types.
              preserved with any specified primitive name.
              E.g., [('foo','bar'), ('tileArrays:par1','val1')]
    :rtype:   <list> of tuples.

    """
    upars = []
    if userparams:
        for upar in userparams:
            tmp = upar.split("=")
            spec, val = tmp[0].strip(), tmp[1].strip()
            if val == 'None':
                val = None
            elif val == 'True':
                val = True
            elif val == 'False':
                val = False
            upars.append((spec,val))

    return upars

# ------------------------------------------------------------------------------
def normalize_args(args):
    """
    Convert argparse argument lists to single string values.

    :parameter args: argparse Namespace object or equivalent
    :type args: <Namespace>

    :return:  Same with converted types.
    :rtype: <Namespace>

    """

    if isinstance(args.drpkg, list):
        args.drpkg = args.drpkg[0]
    if isinstance(args.recipename, list):
        args.recipename = args.recipename[0]
    if localmanager_available:
        if isinstance(args.local_db_dir, list):
            args.local_db_dir = args.local_db_dir[0]
    if isinstance(args.loglevel, list):
        args.loglevel = args.loglevel[0]
    if isinstance(args.logmode, list):
        args.logmode = args.logmode[0]
    if isinstance(args.logfile, list):
        args.logfile = args.logfile[0]
    if isinstance(args.suffix, list):
        args.suffix = args.suffix[0]
    return args

def normalize_context(context):
    """
    For Recipe System v2.0, context shall now be a list of context values.
    E.g.,

    $ reduce --context QA upload    <file.fits> <file2.fits>
    $ reduce --context=QA,upload    <file.fits> <file2.fits>
    $ reduce --context="QA, upload" <file.fits> <file2.fits>

    all result in context == ['qa', 'upload']

    A passed None defaults to 'qa'.

    :parameter context: context argument received by the reduce command line.
    :type context: <list>

    :return: list of coerced or defaulted context values.
    :rtype: <list>

    """
    if context is None:
        context = ['qa']                   # Set default 'qa' [later, 'sq']
    else:
        try:
            assert isinstance(context, list)
        except AssertionError:
            raise TypeError("context must be a list")

    splitc = context if len(context) > 1 else context[0].split(',')
    return [c.lower() for c in splitc]

def normalize_ucals(files, cals):
    """
    When a user passes a --user_cal argument of the form,

    --user_cal processed_bias:/path/to/foo.fits

    The parser produces a user calibrations list like,

    ['processed_bias:/path/to/foo.fits']

    This list would pass to the Reduce __init__ as such, but, this function
    will translate and apply all user cals to all passed files.

    {(ad.data_label(), 'processed_bias'): '/path/to/foo.fits'}

    This dictionary is of the same form as the calibrations dictionary for
    retrieved and stored calibrations.

    User calibrations always take precedence over nominal calibration
    retrieval. User calibrations are not cached because they are not
    retrieved from fitsstore and are presumably on disk.

    Parameters:
    ----------
        cals: a list of strings like, 'caltype:calfilepath'
        type: <list>

    Returns:
    -------
        normalz: a dictionary of the cal types applied to input files.
        type:  <dict>

    E.g., a returned dict,

    {('GS-2017A-Q-32-7-029', 'processed_flat'): '/path/to/XXX_flat.fits'}

    """
    def _site(tags):
        site = None
        if "SOUTH" in tags:
            site = 'GS'
        elif "NORTH" in tags:
            site = 'GN'
        else:
            emsg = "Site cannot be determined."
            raise TypeError(emsg)
        return site

    normalz = {}
    if cals is None:
        return normalz

    for cal in cals:
        ctype, cpath = cal.split(":")
        scal, stype = ctype.split("_")
        caltags = set([scal.upper(), stype.upper()])
        cad = astrodata.open(cpath)
        try:
            assert caltags.issubset(cad.tags)
        except AssertionError:
            errmsg = "Calibration type {}\ndoes not match file {}"
            raise TypeError(errmsg.format(ctype, cpath))

        for f in files:
            ad = astrodata.open(f)
            if _site(cad.tags) != _site(ad.tags):
                emsg = "Calibration {} does not match site of observation {}"
                raise TypeError(emsg.format(cad.filename, ad.filename))

            normalz.update({(ad.data_label(), ctype): cpath})

    return normalz
