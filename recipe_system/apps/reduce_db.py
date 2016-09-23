#!/usr/bin/env python
#
#                                                                  gemini_python
#
#                                                                   reduce_db.py
# ------------------------------------------------------------------------------
# $Id$
# ------------------------------------------------------------------------------
__version__      = '$Rev$'[6:-1]
__version_date__ = '$Date$'[7:-3]
# ------------------------------------------------------------------------------
# reduce_db.py -- calibration database management tool
# ------------------------------------------------------------------------------
_version = '0.1'
# ------------------------------------------------------------------------------

from os.path import expanduser, isdir
from argparse import ArgumentParser
from functools import partial
import sys

from recipe_system.config import globalConf, STANDARD_REDUCTION_CONF
from recipe_system.cal_service import CONFIG_SECTION as CAL_CONFIG_SECTION
from recipe_system.cal_service.localmanager import LocalManager, LocalManagerError
from recipe_system.cal_service.localmanager import ERROR_CANT_WIPE, ERROR_CANT_CREATE

def buildArgumentParser():
    parser = ArgumentParser(description="Calibration Database Management Tool")
    sub = parser.add_subparsers(help="Sub-command help", dest='action')

    p_add = sub.add_parser('add', help="Add files to the calibration "
                           "database. One or more files or directories may "
                           "be specified.")
    p_add.add_argument('files', metavar='path', nargs='+',
                       help="FITS file or directory")
    p_add.add_argument('-k', '--walk', dest='walk', action='store_true',
                       help="If this option is active, directories will be "
                       "explored recursively. Otherwise, only the first "
                       "level will be searched for FITS files.")

    p_list = sub.add_parser('list', help="List calib files in the current database.")

    p_wipe = sub.add_parser('init', help="Create and initialize a new "
                            "database.")
    p_wipe.add_argument('-w', '--wipe', dest='wipe', action='store_true',
                        help="Force the initialization of an already "
                        "existing database.")

    for sp in (p_add, p_wipe, p_list):
        sp.add_argument('-d', '--database', dest='db_path',
                        help="Path to the directory where the database file "
                        "can be found. Optional if the path is defined in a "
                        "config file or environment variable.")
        sp.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                        help="Give information about the operations "
                        "being performed.")

    return parser, sub

def usage(parser, message=None, stream=sys.stderr):
    if message is not None:
        log(message, stream, bold=True, add_newlines=1)

    parser.print_help(file=stream)

def log(message, stream, bold=False, add_newlines=0):
    if stream is not None:
        if bold:
            message = "\x1b[1m{0}\x1b[0m".format(message)
        print >> stream, message
        if add_newlines > 0:
            print >> stream, '\n' * add_newlines,

class Dispatcher(object):
    def __init__(self, parser, manager, log):
        self._parser = parser
        self._mgr = manager
        self._log = log

    def apply(self, action, args):
        try:
            method = getattr(self, '_action_{0}'.format(action))
        except AttributeError:
            raise AttributeError("No such action '{0}'".format(action))

        return method(args)

    def usage(self, message):
        usage(self._parser, message)

    def _action_add(self, args):
        for path in args.files:
            try:
                if isdir(path):
                    # Do something about verbose...
                    if args.walk:
                        m = "Ingesting the files under {0}".format(path)
                    else:
                        m = "Ingesting the files at {0}".format(path)
                    self._log(m)
                    self._mgr.ingest_directory(path, walk=args.walk,
                                               log=self._log)
                else:
                    self._mgr.ingest_file(path)
                    self._log("Ingested {0}".format(path))
            except IOError as e:
                print e
                return -1

        return 0

    def _action_init(self, args):
        try:
            self._log("Initializing {}...".format(self._mgr.path))
            self._mgr.init_database(wipe=args.wipe)
        except LocalManagerError as e:
            if e.error_type == ERROR_CANT_WIPE:
                self.usage(message="Can't initialize an existing database. If "
                           "you're sure about this, either\nremove the file "
                           "first, or pass the -w option to confirm that you "
                           "want\nto wipe the contents.")
            elif e.error_type == ERROR_CANT_CREATE:
                log(e.message, sys.stderr, bold=True)
            return -1

        return 0

    def _action_list(self, args):
        for file_data in self._mgr.list_files():
            print "{:30} {}".format(file_data.name, file_data.path)

if __name__ == '__main__':
    argp, subp = buildArgumentParser()
    args = argp.parse_args(sys.argv[1:])

    globalConf.load(STANDARD_REDUCTION_CONF)

    if args.db_path is not None:
        globalConf.update(CAL_CONFIG_SECTION, dict(
            standalone=True,
            database_dir=args.db_path
        ))

    ret = -1
    conf = globalConf[CAL_CONFIG_SECTION]
    try:
        if not conf.standalone:
            usage(argp, message="The database is not configured as standalone.")
        else:
            act = args.action
            lm = LocalManager(expanduser(conf.database_dir))
            logstream = sys.stderr if args.verbose else None
            disp = Dispatcher(subp.choices[act], lm,
                              log=partial(log, stream=logstream))
            ret = disp.apply(act, args)
    except AttributeError as e:
        usage(argp, message="The database location is undefined.")

    sys.exit(ret)
