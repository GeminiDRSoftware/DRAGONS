#!/usr/bin/env python
#
#                                                                  gemini_python
#
#                                                                     swapper.py
# ------------------------------------------------------------------------------
# $Id$
# ------------------------------------------------------------------------------
__version__  = '$Rev$'[6:-1]
__version_date__ = '$Date$'[7:-2]
# ------------------------------------------------------------------------------
#
#    functional tasks
#

# -- switchable search trunk, either 'trunk' or a passed 'branch'
# -- sift through each module, searching for input 'string'
# -- report each found line, showing current current line
# -- report the matching line with it's new string insert.
# -- user approves line swap OR edits file. In auto mode, 
#    line swapping proceeds.
# -- report all swapped lines to log, indicate line #
#
# Faces will show highlights on terminal out, but will look a mess in log
# (escape chars, color codes). more the log to get highlighting text.
#
# $ swapper -h
# usage: swapper [-h] [-a] [-b BRANCH] [-c] [-d] [-l LOGNAME] [-m MODULE]
#                [-p PKG] [-r] [-u USERPATH]
#                ostring nstring
#
# positional arguments:
#   ostring      <old_string>
#   nstring      <new_string>
#
# optional arguments:
#   -h, --help   show this help message and exit
#   -a, --auto   Execute swaps without user confirmation. Default is False. User
#                must request auto execute
#   -b BRANCH    Execute swaps in <branch> of gemini_python. Default is 'trunk.'
#   -c           Switch on color high lighting. Default is Off.
#   -d           Document line changes w/ swapper comments.
#   -l LOGNAME   Set the logfile name. Default is 'swap.log'.
#   -m MODULE    Execute swaps in <module> only. Default is all.
#   -p PKG       Execute swaps in <package>. Default is 'Gemini'.
#   -r           Report potential swaps only. Default is 'False'.
#   -u USERPATH  Use this path to build search paths. Default is None. Without
#                -u, search under $GEM.
# ------------------------------------------------------------------------------
desc = """
Description:
  swapper replaces string literals that occur within predefined gemini_python 
  packages. By default, these packages are specifically, 

    astrodata/
    astrodata_FITS
    astrodata_Gemini/ 
    gempy/

  A user may specify that a different 'astrodata_X' package is searched rather 
  than the default 'astrodata_Gemini' (see -p option). The 'astrodata_FITS'
  package is fixed and present within gemini_python. It provides neither recipes
  nor primitives and provides only a limited set of generic descriptors.
  astrodata_FITS/ is searched if present, but is not presumed to exist.

  Search paths are based upon an environment variable, $GEM, OR on the path
  passed with the '-u USERPATH' option. $GEM defines a path to a user's 
  gemini_python installation as pulled from the GDPSG repository, and which
  nominally contains the 'branches' and 'trunk' directories as they appear 
  in the gemini_python repo. I.e.,

    export GEM=/user/path/to/gemini_python

  which shall contain

    branches/
    trunk/

  The critical paths are 'branches' and 'trunk'. 'branches' need only be 
  present if a branch is specified by the user (-b option). Other repo 
  directories need not be present, as they are not considered in the search. 
  If a user has a non-standard or partial gemini_python installation, or has 
  otherwise changed the above organisation, the -u option should be used to 
  pass the location of this code base to swapper. If -u is passed, search 
  packages should be directly under this path and any -b option will be ignored.

  Examples:

  -- a standard gemini_python repo checkout in ~ :

      $ export GEM=~/gemini_python  [or setenv ... for csh-ish]
      $ swapper -c -r "old string" "new string"

  -- astrodata, gempy, and other astrodata packages are in directory
     ~/foobar/ . I.e. there is no 'trunk' subdir. Use -u:

      $ swapper -r -c -u ~/foobar "old string" "new string"

"""
import os
import sys
import glob
import shlex
import fileinput
import subprocess

from time import strftime
from shutil import copyfile
from os.path import basename, exists, join
from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter

from gempy.utils import logutils

# ------------------------------------------------------------------------------
version = "1.0 (r" + __version__.strip() +")"
def handleCLArgs():
    parser = ArgumentParser(description=desc, prog='swapper',
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('ostring', help="<old_string>")
    parser.add_argument('nstring', help="<new_string>")

    parser.add_argument("-a", "--auto", dest="auto", action="store_true",
                        help="Execute swaps without user confirmation."
                        " Default is False. User must request auto execute")

    parser.add_argument("-b", dest="branch", default="trunk",
                        help="Execute swaps in <branch> of gemini_python."
                        " Default is 'trunk.'")

    parser.add_argument("-c", dest="color", action="store_true",
                        help="Switch on color high lighting."
                        " Default is Off.")

    parser.add_argument("-d", dest="doc", action="store_true",
                        help="Document line changes w/ swapper comments.")

    parser.add_argument("-l", dest="logname", default="swap.log",
                        help="Set the logfile name."
                        " Default is 'swap.log'.")

    parser.add_argument("-m", dest="module", default=None,
                        help="Execute swaps in <module> only."
                        " Default is all.")

    parser.add_argument("-p", dest="pkg", default="Gemini",
                        help="Execute swaps in <package>."
                        " Default is 'Gemini'.")

    parser.add_argument("-r", dest="report", action="store_true",
                        help="Report potential swaps only."
                        " Default is 'False'.")

    parser.add_argument("-u", dest="userpath", default=None,
                        help="Use this path to build search paths."
                        " Default is None. Without -u, search under $GEM. ")

    parser.add_argument("-v", '--version', action='version', 
                        version='%(prog)s ' + version)  

    args = parser.parse_args()
    return args

# ------------------------------------------------------------------------------
# Faces class
# -- set stdout face symbols
# Eg.,
#
# print Faces.BOLD + 'Hello World !' + Faces.END

class Faces(object):
    PURPLE    = '\033[95m'
    CYAN      = '\033[96m'
    DARKCYAN  = '\033[36m'
    BLUE      = '\033[94m'
    GREEN     = '\033[92m'
    YELLOW    = '\033[93m'
    RED       = '\033[91m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'
    END       = '\033[0m'

# ------------------------------------------------------------------------------
class Swap(object):

    def __init__(self, args):
        """ Instance definitions."""
        
        # Get the gemini_python location. Users should define $GEM
        # set up with report only flag. TBR
        try:
            self.GEM = os.path.abspath(os.environ['GEM'])
        except KeyError:
            self.GEM = None

        self.pif    = "PIF"
        self.fits   = "FITS"
        self.config = "ADCONFIG"
        self.recipe = "RECIPES"
        self.pymods = []
        self.pkg_paths  = []
        self.fits_paths = []
        self.full_paths = []
        self.swap_summary = ()
        
        self.doc      = args.doc
        self.package  = args.pkg
        self.auto_run = args.auto
        self.branch   = args.branch
        self.focus    = args.module
        self.colorize = args.color
        self.cur_str  = args.ostring
        self.new_str  = args.nstring
        if args.userpath:
            self.userpath = os.path.abspath(args.userpath)
        else:
            self.userpath = args.userpath

    # paths in a package
    def setup_search(self):
        ppif     = self.pif    + "_" + self.package
        adconfig = self.config + "_" + self.package
        recipes  = self.recipe + "_" + self.package
        adfits   = self.config + "_" + "FITS"
        
        self.search_set = {"astro_paths": ['astrodata', 'astrodata/adutils', 
                                           'astrodata/adutils/reduceutils',
                                           'astrodata/eti', 'astrodata/scripts'
                                       ],
                           "astro_new_paths": ['astrodata', 'astrodata/eti',
                                               'astrodata/interface',
                                               'astrodata/scripts', 
                                               'astrodata/utils',
                                               'recipe_system/adcc/servers',
                                               'recipe_system/apps',
                                               'recipe_system/cal_service',
                                               'recipe_system/reduction'
                                           ],
                           "gemp_paths": [ 'gempy/adlibrary', 'gempy/gemini',
                                           'gempy/gemini/eti', 'gempy/library',
                                           'gempy/scripts'
                                       ],
                           "fits_paths": [ join(adfits, 'descriptors')
                                       ],
                           "pkg_paths": [ join(recipes, 'primitives'),
                                          join(adconfig,'structures'),
                                          join(adconfig,'lookups'),
                                          join(adconfig,'lookups/F2'),
                                          join(adconfig,'lookups/GMOS'),
                                          join(adconfig,'lookups/GNIRS'),
                                          join(adconfig,'lookups/GSAOI'),
                                          join(adconfig,'lookups/NIFS'),
                                          join(adconfig,'lookups/NIRI'),
                                          join(adconfig,'descriptors'),
                                          join(adconfig,'descriptors/F2'),
                                          join(adconfig,'descriptors/GMOS'),
                                          join(adconfig,'descriptors/GNIRS'),
                                          join(adconfig,'descriptors/GSAOI'),
                                          join(adconfig,'descriptors/MICHELLE'),
                                          join(adconfig,'descriptors/NICI'),
                                          join(adconfig,'descriptors/NIFS'),
                                          join(adconfig,'descriptors/NIRI'),
                                          join(adconfig,'descriptors/PHOENIX'),
                                          join(adconfig,'descriptors/TRECS'),
                                          join(adconfig,'classifications/status'),
                                          join(adconfig,'classifications/types'),
                                          join(adconfig,'classifications/types/ABU'),
                                          join(adconfig,'classifications/types/BHROS'),
                                          join(adconfig,'classifications/types/CIRPASS'),
                                          join(adconfig,'classifications/types/F2'),
                                          join(adconfig,'classifications/types/GMOS'),
                                          join(adconfig,'classifications/types/GNIRS'),
                                          join(adconfig,'classifications/types/GPI'),
                                          join(adconfig,'classifications/types/GSAOI'),
                                          join(adconfig,'classifications/types/MICHELLE'),
                                          join(adconfig,'classifications/types/NICI'),
                                          join(adconfig,'classifications/types/NIFS'),
                                          join(adconfig,'classifications/types/NIRI'),
                                          join(adconfig,'classifications/types/OSCIR'),
                                          join(adconfig,'classifications/types/PHOENIX'),
                                          join(adconfig,'classifications/types/QUIRC'),
                                          join(adconfig,'classifications/types/TEXES'),
                                          join(adconfig,'classifications/types/TRECS'),
                                          join(ppif,'primdicts'),
                                          join(ppif,'pifgemini'),
                                          join(ppif,'pifgemini/bookkeeping'),
                                          join(ppif,'pifgemini/display'),
                                          join(ppif,'pifgemini/general'),
                                          join(ppif,'pifgemini/gmos'),
                                          join(ppif,'pifgemini/gmos_image'),
                                          join(ppif,'pifgemini/gmos_spect'),
                                          join(ppif,'pifgemini/mask'),
                                          join(ppif,'pifgemini/photometry'),
                                          join(ppif,'pifgemini/preprocess'),
                                          join(ppif,'pifgemini/qa'),
                                          join(ppif,'pifgemini/register'),
                                          join(ppif,'pifgemini/resample'),
                                          join(ppif,'pifgemini/stack'),
                                          join(ppif,'pifgemini/standardize')
                                      ]
                       }
        return

    def set_pkg_paths(self):
        """ Configure list of package paths for package name, pkg.

        Populates <list> instance variable,

            self.package_search.

        parameters: <void>
        return:     <void>
        
        Eg., 
        >>> self.set_pkg_paths()
        >>> paths[0]
        'astrodata_Gemini/RECIPES_Gemini/primitives'
        """
        package_paths = self.search_set['pkg_paths']
        for path in package_paths:
            self.pkg_paths.append(join('astrodata_' + self.package, path))
        return

    def set_fits_paths(self):
        """ Configure list of paths for the generic astrodata_FITS.
        Currently, there is only one code directory under astrodata_FITS,
        which is ADCONFIG_FITS/descriptors.

        parameters: <void>
        return:     <void>
        
        Eg., 
        >>> self.set_fits_paths()
        >>> paths[0]
        'astrodata_FITS/ADCONFIG_FITS/descriptors'
        """
        package_paths = self.search_set['fits_paths']
        for path in package_paths:
            self.fits_paths.append(join('astrodata_' + self.fits, path))
        return

    def set_full_paths(self):
        """ Sets the instance var 'full_paths' with the fulls paths
        for the search, as defined by .setup_search()

        parameters: <void>
        return:     <void>
        """
        gemp_paths  = self.search_set['gemp_paths']
        gem_path    = self._determine_gem_path()
        branch_path = self._determine_branch_path()

        try:
            assert exists(gem_path)
        except AssertionError:
            msg = "Supplied path '" + gem_path + "' cannot be found."
            raise SystemExit(msg)

        try:
            assert exists(join(gem_path, branch_path))
        except AssertionError:
            msg = "Branch '" + self.branch + "' cannot be found."
            raise SystemExit(msg)

        if exists(join(gem_path, branch_path, 'recipe_system')):
            astro_paths = self.search_set['astro_new_paths']
        else:
            astro_paths = self.search_set['astro_paths']

        for path in astro_paths:
            self.full_paths.append(join(gem_path, branch_path, path))
        for path in gemp_paths:
            self.full_paths.append(join(gem_path, branch_path, path))
        for path in self.pkg_paths:
            self.full_paths.append(join(gem_path, branch_path, path))
        for path in self.fits_paths:
            fpath = join(gem_path, branch_path, path)
            if exists(fpath):
                self.full_paths.append(fpath)
        return

    def set_searchable_mods(self):
        """ Builds the set of searchable modules in all paths."""
        for path in self.full_paths:
            py_modules = self._get_py_modules(path)
            self.pymods.extend(py_modules)
        return

    def report(self):
        """
        Report *only* matches found in modules. Called when user specifies
        -r switch.
        """
        new_head = ""
        self._echo_header()
        for mod in self.pymods:
            mod_test = basename(mod)
            if self.focus and not mod_test == self.focus:
                continue
            match_lines = []
            fpath, tail  = os.path.split(mod)
            if self.userpath:
                head = fpath.split(self.userpath)[-1]
            else:
                head = fpath.split(self.branch)[-1]
            match_lines = self._search_and_report(mod, self.cur_str)
            if match_lines:
                if head != new_head:
                    new_head = head
                    log.stdinfo(Faces.YELLOW + "\n------------" + Faces.END)
                    log.stdinfo("@ " + Faces.DARKCYAN + new_head + ":" + Faces.END)
                for line in match_lines:
                    log.stdinfo(Faces.BOLD + tail + Faces.END + line)
        return

    def report_and_execute(self):
        """
        Method will discover matched lines as in .report(), but will 
        display what the new line will look like upon substitution. 
        User confirmation is required to execute the swap unless 
        -a, --auto has been specified.
        """
        nmods  = 0
        nswaps = 0
        new_head = ""
        current_mod = None
        self._echo_header()
        for mod in self.pymods:
            mod_test = basename(mod)
            if self.focus and not mod_test == self.focus:
                continue
            match_lines = []
            fpath, tail = os.path.split(mod)

            if self.userpath:
                head = fpath.split(self.userpath)[-1]
            else:
                head = fpath.split(self.branch)[-1]

            match_lines = self._search_for_execute(mod, self.cur_str, self.new_str)
            if match_lines:
                current_mod = mod
                if head != new_head:
                    new_head = head
                    log.stdinfo(Faces.YELLOW + "\n------------" + Faces.END)
                    log.stdinfo("@ " + Faces.DARKCYAN + new_head + ":" + Faces.END)
                for line_set in match_lines:
                    log.stdinfo(Faces.BOLD + tail + Faces.END + line_set[1])
                    log.stdinfo(Faces.BOLD + tail + Faces.END + line_set[3])
                    if self.auto_run:
                        if current_mod == mod:
                            nmods  += 1
                        nswaps += 1
                        self._execute_swap(mod, line_set)
                    else:
                        if self._confirm_swap(mod, line_set[0] + 1):
                            if current_mod == mod:
                                pass
                            else:
                                current_mod = mod
                                nmods += 1
                            nswaps += 1
                            log.stdinfo("Swap confirmed.")
                            self._execute_swap(mod, line_set)
                        else:
                            continue
        if nswaps and nmods:
            self.swap_summary = (nswaps, nmods)
        return

    def summarize(self):
        if self.swap_summary:
            swaps, mods = self.swap_summary
            log.stdinfo(Faces.YELLOW + "------------" + Faces.END)
            log.stdinfo("\n%s swap(s) executed in %s module(s)" % 
                        (str(swaps), str(mods)))
            log.stdinfo("\tNote: User edits are not tallied")
        return

    # ------------------------------ prive -------------------------------------
    def _echo_header(self):
        astro_pkg = "astrodata_" + self.package
        log.stdinfo("\n" + basename(__file__) + " \tr" + __version__)

        if self.userpath:
            log.stdinfo("USERPATH\t" + Faces.BOLD + self.userpath + Faces.END)
            log.stdinfo("BRANCH: \t" + Faces.BOLD + "None" + Faces.END)
        elif self.GEM:
            log.stdinfo("Searching\t" + Faces.BOLD + "gemini_python ..." + Faces.END)
            log.stdinfo("BRANCH: \t" + Faces.BOLD + self.branch + Faces.END)
            log.stdinfo("PACKAGE:\t" + Faces.BOLD + astro_pkg + Faces.END + "\n")
        return

    def _determine_gem_path(self):
        """ Build the instance gem_path variable. """
        gem_path = None
        if not self.userpath and not self.GEM:
            msg = "Specify -u USERPATH or define $GEM. -h for help."
            raise SystemExit(msg)

        # Override gem_path if userpath has been specified.
        if self.userpath:
            gem_path = self.userpath
        else:
            gem_path = self.GEM
        return gem_path

    def _determine_branch_path(self):
        """ Build an appropriate branch path. """
        if self.userpath:
            branch_path = ""
        elif self.branch is 'trunk':
            branch_path = self.branch
        else:
            branch_path = join('branches', self.branch)
        return branch_path

    def _get_py_modules(self, path):
        """ Return a list of python modules in the passed path.

        parameters: <str>
        return:     <list>
        """
        return glob.glob(join(path, "*.py"))

    def _search_and_report(self, module, string):
        match_lines = []
        mod_lines = open(module).readlines()

        for i in range(len(mod_lines)):
            line = mod_lines[i].strip()
            sindex = line.find(string)
            eindex = sindex + len(string)
            if sindex > -1:
                if self.colorize:
                    matched_line = self._build_electric_line(line, i, 
                                                                   sindex, eindex)
                else:
                    matched_line = self._build_vanilla_line(line, i, 
                                                             sindex, eindex)
                match_lines.append(matched_line)
        return match_lines

    def _search_for_execute(self, module, ostring, nstring):
        swap_lines = []
        open_mod  = open(module)
        mod_lines = open_mod.readlines()
        open_mod.close()

        for i in range(len(mod_lines)):
            line = mod_lines[i].rstrip()
            sindex = line.find(ostring)
            eindex = sindex + len(ostring)
            if sindex > -1:
                if self.colorize:
                    matched_line = self._build_electric_line(line, i, sindex, eindex)
                    new_line, print_line = self._build_electric_new_line(line, i, 
                                                                            sindex, 
                                                                            ostring, 
                                                                            nstring)
                else:
                    matched_line = self._build_vanilla_line(line, i, sindex, eindex)
                    new_line, print_line  = self._build_vanilla_new_line(line, i, 
                                                                          sindex, 
                                                                          ostring, 
                                                                          nstring)

                swap_lines.append((i, matched_line, new_line, print_line))
        return swap_lines

    def _build_electric_line(self, line, i, sindex, eindex):
        """ Build electric line, high lighting various parts.

        parameters: <int>, <int>, <int>, indices: line, string match start, end
        return:     <str>, string with highlighting escape chars.
        """
        electric_line = (Faces.CYAN + "@L" + str(i + 1) + "::  " + Faces.END 
                         + line[:sindex] + Faces.RED + line[sindex:eindex] 
                         + Faces.END + line[eindex:])

        return electric_line

    def _build_vanilla_line(self, line, i, sindex, eindex):
        """ Build vanilla line, no high lighting.

        parameters: <int>, <int>, <int>, indices: line, start string match, end
        return:     <str>, string with highlighting escape chars.
        """
        vanilla_line = ("@L" + str(i + 1) + "::  " + line[:sindex] + 
                         line[sindex:eindex] + line[eindex:])
        return vanilla_line

    def _build_electric_new_line(self, line, i, sindex, ostring, nstring):
        """ Build electric line, high lighting various parts.

        parameters: <str>, <int>, <int>, <str>, <str> 
                    line, lineno, str index start, end, old str, new str
        return:     <str>, string with highlighting escape chars.
        """
        line = line.replace(ostring, nstring)
        eindex = sindex + len(nstring)
        electric_line = (Faces.CYAN + "@L" + str(i + 1) + "::  " + Faces.END 
                         + line[:sindex] + Faces.RED + line[sindex:eindex] 
                         + Faces.END + line[eindex:] + Faces.CYAN +"\t <== NEW"
                         + Faces.END)
        return line, electric_line

    def _build_vanilla_new_line(self, line, i, sindex, ostring, nstring):
        """ Build vanilla line, no high lighting.

        parameters: <str>, <int>, <int>, <str>, <str> 
                    line, string index start, end, old string, new string
        return:     <str>, string with highlighting escape chars.
        """
        line = line.replace(ostring, nstring)
        eindex = sindex + len(nstring)
        vanilla_line = ("@L" + str(i + 1) + "::  " + line[:sindex] + 
                         line[sindex:eindex] + line[eindex:] + "\t <== NEW")
        return line, vanilla_line

    def _confirm_swap(self, mod, lineno):
        confirm = False
        try:
            response = raw_input("\nConfirm swap (y/n/e): ")
            if response == "y":
                confirm = True
            elif response == "e":
                self._user_edit_mode(mod, lineno)
        except KeyboardInterrupt:
            sys.exit("\tExiting on ^C\n")
        return confirm

    def _execute_swap(self, mod, line_set):
        """ Execute approved line swap. line_set is a tuple comprising
        the match (current) line, the computed new line, and the built
        printable line (unused here).

        mod      --  module name, which is a full path
        line_set --  tuple of values, where index is the index of the line
                     in the list of lines of the module.

        parameters: <str>, <tuple>, file, index, old line, new line, printer
        return:     <void>
        """
        lineno   = line_set[0] + 1
        if self.doc:
            comment  = " # Changed by swapper, " + strftime("%d %b %Y") + "\n"
        else:
            comment = "\n"
        new_line = line_set[2] + comment

        log.stdinfo("Executing swap in module: " +  basename(mod))
        log.stdinfo(Faces.CYAN + "New line @L" + str(lineno) + ":: " + 
                    Faces.END + new_line)
        for line in fileinput.input(mod, inplace=1, backup=".bak"):
            if fileinput.filelineno() == lineno:
                line = new_line
            sys.stdout.write(line)
        fileinput.close()
        return

    def _user_edit_mode(self, mod, lineno):
        cmd = "emacs +" + "%s %s" % (str(lineno), mod)
        args = shlex.split(cmd)
        subprocess.call(args)
        return

# ------------------------------------------------------------------------------
def main(args):
    swap = Swap(args)
    swap.setup_search()
    swap.set_pkg_paths()
    swap.set_fits_paths()
    swap.set_full_paths()
    swap.set_searchable_mods()
    if args.report:
        swap.report()
    else:
        swap.report_and_execute()
        swap.summarize()
    return

# ____________________
if __name__ == "__main__":
    args = handleCLArgs()
    # Comfig the logger.
    logutils.config(file_name=args.logname)
    log = logutils.get_logger(__name__)
    sys.exit(main(args))
