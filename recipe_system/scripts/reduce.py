#!/usr/bin/env python
#
#                                                                        DRAGONS
#
#                                                                      reduce.py
# ------------------------------------------------------------------------------
from __future__ import print_function
# ---------------------------- Package Import ----------------------------------
import os
import sys
import signal
import traceback

from gempy.utils import logutils

from recipe_system import __version__ as rs_version
from recipe_system.reduction.coreReduce import Reduce

from recipe_system.utils.reduce_utils import buildParser
from recipe_system.utils.reduce_utils import normalize_args
from recipe_system.utils.reduce_utils import normalize_upload
from recipe_system.utils.reduce_utils import show_parser_options

from recipe_system.cal_service import set_calservice
from recipe_system.cal_service import localmanager_available
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
    
    Use of the reduce_utils function buildParser will get the caller a fully
    defined reduce Namespace instance, values for which can be then be adjusted
    as desired.
    
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
        logutils.config(mode=args.logmode, file_name=args.logfile)
        log = logutils.get_logger(__name__)
        log.info("Logging configured for application: reduce")
        log.info(" ")
    except AssertionError:
        pass

    # Config local calibration manager with passed args object
    if localmanager_available:
        set_calservice(local_db_dir=args.local_db_dir)

    log.stdinfo("\n\t\t\t--- reduce v{} ---".format(rs_version))
    log.stdinfo("\nRunning on Python {}".format(sys.version.split()[0]))
    r_reduce = Reduce(args)
    try:
        r_reduce.runr()
    except KeyboardInterrupt:
        log.error("Caught KeyboardInterrupt (^C) signal")
        estat = signal.SIGINT
    except Exception as err:
        log.error("reduce caught an unhandled exception.\n")
        log.error(err)
        log.error("\nReduce instance aborted.")
        estat = signal.SIGABRT

    if estat != 0:
        log.stdinfo("\n\nreduce exit status: %d\n" % estat)
    else:
        pass
    return estat
# --------------------------------------------------------------------------

if __name__ == "__main__":
    parser = buildParser(rs_version)
    args = parser.parse_args()

    # Deal with argparse structures that are different than optparse
    # Normalizing argument types should happen before 'args' is passed to Reduce.
    args = normalize_args(args)
    args.upload = normalize_upload(args.upload)

    if args.displayflags:
        show_parser_options(parser, args)
        for item in ["Input fits file(s):\t%s" % inf for inf in args.files]:
            print(item)
        sys.exit()

    sys.exit(main(args))
