#!/usr/bin/env python
#
#                                                                  gemini_python
#
#                                                              astrodata/scripts
#                                                                     reduce2.py
# ------------------------------------------------------------------------------
# $Id$
# ------------------------------------------------------------------------------
__version__      = '$Rev$'[6:-1]
__version_date__ = '$Date$'[7:-3]
# ------------------------------------------------------------------------------
# reduce2.py -- refactored reduce, cl parsing exported, functionalized.
#               see parseUtils.py
# ------------------------------------------------------------------------------
_version = '2.0'
# ------------------------------------------------------------------------------
"""
This module (reduce2) provides a functionalized interface to running the core 
processing of the reduce pipeline. This core functionality is provided by the 
imported coreReduce module. 

It provides both a nominal 'reduce' command line interface and a 'main'
function that can be called with an 'args' parameter.

Eg.,

Get "args' from the defined reduce parser:

>>> args = parseUtils.buildParser(version).parse_args()
>>> import reduce2
>>> reduce2.main(args)

In the above example, 'args' is

-- argparse Namespace instance

Use of the parseUtils function buildParser will get the caller a fully defined 
reduce Namespace instance, values for which can be then be adjusted as desired.

Eg.,

parseUtils.buildParser:
----------------------
>>> args = parseUtils.buildParser(version).parse_args()
>>> args.logfile
'reduce.log'
>>> args.files
[]
>>> args.files.append('some_fits_file.fits')

Once 'args' attributes have been appropriately set, the caller then simply 
calls main():

>>> reduce2.main(args)
"""
# ---------------------------- Package Import ----------------------------------
import os
import sys
from   signal import SIGTERM

from astrodata import __version__ as ad_version
from astrodata.adutils import logutils

import parseUtils
from   coreReduce import Reduce
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

    try:
        log.stdinfo("\t\t\t--- reduce, v%s ---" % _version)
    except NameError:
        log.stdinfo("\t\t\t--- reduce, v%s ---" % __version__)
    log.stdinfo("\t\tRunning under astrodata Version "+ ad_version)
    r_reduce = Reduce(args)
    try:
        estat = r_reduce.runr()
    except Exception as err:
        estat = SIGTERM
        log.error("main() caught unhandled exception:")
        log.error(type(err))
        log.error(str(err))

    if estat != 0:
        log.stdinfo("\n\nreduce exit status: %d\n" % estat)
    else:
        pass
    return (estat)

# --------------------------------------------------------------------------
if __name__ == "__main__":
    version_report = _version + "(r" + __version__.strip() + ")"
    parser = parseUtils.buildParser(version_report)
    args   = parser.parse_args()

    if args.displayflags:
        parseUtils.show_parser_options(parser, args)
        for item in ["Input fits file(s):\t%s" % inf for inf in args.files]:
            print item
        sys.exit()

    # Deal with argparse structures that are different than optparse 
    args = parseUtils.normalize_args(args)
    sys.exit(main(args))
