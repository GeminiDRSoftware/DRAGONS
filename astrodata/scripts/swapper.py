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
# usage: swapper [-h] [-a] [-b BRANCH] [-c] [-m MODULE] [-p PKG] [-r]
#                ostring nstring

# positional arguments:
#   ostring     <old_string>
#   nstring     <new_string>

# optional arguments:
#   -h, --help  show this help message and exit
#   -a, --auto  Execute swaps without user confirmation. Default is False. User
#               must request auto execute
#   -b BRANCH   Execute swaps in <branch> of gemini_python. Default is 'trunk.'
#   -c          Switch on color high lighting. Default is Off.
#   -m MODULE   Execute swaps in <module> only. Default is all.
#   -p PKG      Execute swaps in <package>. Default is 'Gemini'.
#   -r          Report potential swaps only. Default is 'False'.
# ------------------------------------------------------------------------------
import os
import sys
import glob
import shlex
import fileinput
import subprocess

from time import strftime
from shutil import copyfile
from argparse import ArgumentParser
from os.path import basename, exists, join

from astrodata.adutils import logutils
# ------------------------------------------------------------------------------
# Comfig the logger. Logger is module level.
logutils.config(file_name="swap.log")
log = logutils.get_logger(__name__)

# ------------------------------------------------------------------------------
def handleCLArgs():
    parser = ArgumentParser()
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

    parser.add_argument("-m", dest="module", default=None,
                        help="Execute swaps in <module> only."
                        " Default is all.")

    parser.add_argument("-p", dest="pkg", default="Gemini",
                        help="Execute swaps in <package>."
                        " Default is 'Gemini'.")

    parser.add_argument("-r", dest="report", action="store_true",
                        help="Report potential swaps only."
                        " Default is 'False'.")    

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
        self.GEM    = os.environ['GEM']
        self.pif    = "PIF"
        self.config = "ADCONFIG"
        self.recipe = "RECIPES"
        self.pymods = []
        self.pkg_paths = []
        self.full_paths = []
        self.swap_summary = ()
        
        self.package  = args.pkg
        self.auto_run = args.auto
        self.branch   = args.branch
        self.focus    = args.module
        self.colorize = args.color
        self.cur_str  = args.ostring
        self.new_str  = args.nstring

    # paths in a package
    def setup_search(self):
        ppif     = self.pif    + "_" + self.package
        adconfig = self.config + "_" + self.package
        recipes  = self.recipe + "_" + self.package
        
        self.search_set = {"astro_paths": ['astrodata', 'astrodata/adutils', 
                                           'astrodata/adutils/reduceutils',
                                           'astrodata/eti', 'astrodata/scripts'
                                       ],
                           "gemp_paths": [ 'gempy/adlibrary', 'gempy/gemini',
                                           'gempy/gemini/eti', 'gempy/library',
                                           'gempy/scripts'
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
                                          join(ppif,'primdicts'),
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

    def set_full_paths(self):
        """ Sets the instance var 'full_paths' with the fulls paths
        for the search, as defined by .setup_search()

        parameters: <void>
        return:     <void>
        """
        astro_paths = self.search_set['astro_paths']
        gemp_paths  = self.search_set['gemp_paths']
        gem_path    = self.GEM
        if self.branch is 'trunk':
            branch_path = self.branch
        else:
            branch_path = join('branches', self.branch)

        if not exists(join(gem_path, branch_path)):
            raise SystemExit("Specified branch " + self.branch + " cannot be found.")

        for path in astro_paths:
            self.full_paths.append(join(gem_path, branch_path, path))
        for path in gemp_paths:
            self.full_paths.append(join(gem_path, branch_path, path))
        for path in self.pkg_paths:
            self.full_paths.append(join(gem_path, branch_path, path))
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
        self._echo_header()
        for mod in self.pymods:
            mod_test = basename(mod)
            if self.focus and not mod_test == self.focus:
                continue
            match_lines = []
            chunks = mod.split('/')
            pretty_path = join(chunks[-4], chunks[-3], chunks[-2], chunks[-1])
            match_lines = self._search_and_report(mod, self.cur_str)
            if match_lines:
                print Faces.YELLOW + "------------" + Faces.END
                for line in match_lines:
                    print Faces.BOLD + chunks[-1] + Faces.END, line
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
        self._echo_header()
        for mod in self.pymods:
            mod_test = basename(mod)
            if self.focus and not mod_test == self.focus:
                continue
            match_lines = []
            chunks = mod.split('/')
            pretty_path = join(chunks[-4], chunks[-3], chunks[-2], chunks[-1])
            match_lines = self._search_for_execute(mod, self.cur_str, self.new_str)
            nmods += 1
            if match_lines:
                print Faces.YELLOW + "------------" + Faces.END
                for line_set in match_lines:
                    print Faces.BOLD + chunks[-1] + Faces.END, line_set[1]
                    print Faces.BOLD + chunks[-1] + Faces.END, line_set[3]
                    if self.auto_run:
                        nswaps += 1
                        self._execute_swap(mod, line_set)
                    else:
                        if self._confirm_swap(mod, line_set[0] + 1):
                            nswaps += 1
                            print "Swap confirmed."
                            self._execute_swap(mod, line_set)
                        else:
                            continue
        if nswaps and nmods:
            self.swap_summary = (nswaps, nmods)
        return

    def summarize(self):
        if self.swap_summary:
            swaps. mods = self.swap_summary
            print Faces.YELLOW + "------------" + Faces.END
            print "\n%s swap(s) executed in %s module(s)" % (str(swaps), str(mods))
            print "\tNote: does not include user edits that may have occurred."
        return

    # ------------------------------ prive -------------------------------------
    def _echo_header(self):
        astro_pkg = "astrodata_" + self.package
        print "\n", basename(__file__) , "\tr" + __version__
        print "Searching\t",Faces.BOLD + "gemini_python ...\n" + Faces.END
        print "BRANCH: \t", Faces.BOLD + self.branch + Faces.END
        print "PACKAGE:\t", Faces.BOLD + astro_pkg + Faces.END
        print
        return

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
                    line, string index start, end, old string, new string
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
        comment  = "  # Changed by swapper, " + strftime("%d %b %Y") + "\n"
        new_line = line_set[2] + comment

        print "Executing swap in module: ", basename(mod)
        print Faces.CYAN + "New line @L" + str(lineno) + ":: " + Faces.END + new_line
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
    sys.exit(main(args))
