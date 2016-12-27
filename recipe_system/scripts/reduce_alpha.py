#!/usr/bin/env python
#
#
#                                                                reduce_alpha.py
# ------------------------------------------------------------------------------
# reduce_alpha.py -- The new hope reduce. [pending name change status quo ante]
# ------------------------------------------------------------------------------
from __future__ import print_function
_version = ' alpha (new_hope) '
# ------------------------------------------------------------------------------
"""
Prototype reduce (New Hope).

"""
# ---------------------------- Package Import ----------------------------------
import sys

from gempy.utils import logutils

from recipe_system.reduction.coreReduce import Reduce

from recipe_system.utils.reduce_utils import buildParser
from recipe_system.utils.reduce_utils import normalize_args
from recipe_system.utils.reduce_utils import show_parser_options

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

    log.stdinfo("\t\t\t--- reduce, v{} ---".format(_version))
    r_reduce = Reduce(args)
    estat = r_reduce.runr()
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
            print(item)
        sys.exit()

    # Deal with argparse structures that are different than optparse 
    args = normalize_args(args)
    sys.exit(main(args))
