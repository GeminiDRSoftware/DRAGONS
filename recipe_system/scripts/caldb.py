#!/usr/bin/env python
#                                                                        DRAGONS
#
#                                                                       caldb.py
# ------------------------------------------------------------------------------
# caldb.py -- local calibration database management tool
# ------------------------------------------------------------------------------
from __future__ import print_function
# ------------------------------------------------------------------------------
import sys
import traceback

from os.path import expanduser, isdir, exists, basename
from argparse import ArgumentParser
from functools import partial

from recipe_system.config import globalConf, STANDARD_REDUCTION_CONF
from recipe_system.cal_service import load_calconf, update_calconf, get_calconf
from recipe_system.cal_service.localmanager import LocalManager, LocalManagerError
from recipe_system.cal_service.localmanager import ERROR_CANT_WIPE, ERROR_CANT_CREATE
from recipe_system.cal_service.localmanager import ERROR_CANT_READ, ERROR_DIDNT_FIND
# ------------------------------------------------------------------------------

def buildArgumentParser():

    parser = ArgumentParser(description="Calibration Database Management Tool")
    sub = parser.add_subparsers(help="Sub-command help", dest='action')

    p_config = sub.add_parser('config', help="Display configuration info")

    p_init = sub.add_parser('init', help="Create and initialize a new "
                            "database.")
    p_init.add_argument('-w', '--wipe', dest='wipe', action='store_true',
                        help="Force the initialization of an already "
                        "existing database.")

    p_list = sub.add_parser('list', help="List calib files in the current database.")

    p_add = sub.add_parser('add', help="Add files to the calibration "
                           "database. One or more files or directories may "
                           "be specified.")
    p_add.add_argument('files', metavar='path', nargs='+',
                       help="FITS file or directory")
    p_add.add_argument('-k', '--walk', dest='walk', action='store_true',
                       help="If this option is active, directories will be "
                       "explored recursively. Otherwise, only the first "
                       "level will be searched for FITS files.")

    p_remove = sub.add_parser('remove', help="Remove files from the "
                              "calibration database. One or more files "
                              "may be specified.")
    p_remove.add_argument('files', metavar='filenames', nargs='+',
                          help="FITS file names. Paths will be disregarded.")

    for sp in (p_config, p_add, p_init, p_list, p_remove):
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
        print(message, file=stream)
        if add_newlines > 0:
            print('\n' * add_newlines, file=stream)


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

    def _action_config(self, args):
        isactive = "The 'standalone' flag is \033[1mactive\033[0m, meaning that local "
        isactive += "calibrations will be used"
        inactive = "The 'standalone' flag is not active, meaning that remote "
        inactive += "calibrations will be downloaded."

        conf = get_calconf()
        print('')
        print("Using configuration file: \033[1m{}\033[0m".format(STANDARD_REDUCTION_CONF))
        #print()
        print("Active database directory: \033[1m{}\033[0m".format(conf.database_dir))
        path = self._mgr._db_path
        print("Database file: \033[1m{}\033[0m".format(path))
        if not exists(path):
            print("   NB: The database does not exist. Please initialize it.")
            print("       (Read the help message about 'init' command)")
        print()
        if conf.standalone:
            print(isactive)
        else:
            print(inactive)
        print('')

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
                traceback.print_last()
                print(e, file=sys.stderr)
                return -1

        return 0

    def _action_remove(self, args):
        for path in args.files:
            try:
                self._mgr.remove_file(basename(path))
                self._log("Removed {0}".format(path))
            except LocalManagerError as e:
                if e.error_type == ERROR_DIDNT_FIND:
                    log(e.message, sys.stderr)

    def _action_init(self, args):
        msg = "Can't initialize an existing database. If "
        msg += "you're sure about this, either\nremove the file "
        msg += "first, or pass the -w option to confirm that you "
        msg += "want\nto wipe the contents."
        try:
            self._log("Initializing {}...".format(self._mgr.path))
            self._mgr.init_database(wipe=args.wipe)
        except LocalManagerError as e:
            if e.error_type == ERROR_CANT_WIPE:
                self.usage(message=msg)
            elif e.error_type == ERROR_CANT_CREATE:
                log(e.message, sys.stderr, bold=True)
            return -1

        return 0

    def _action_list(self, args):
        msg = "Could not read information from the database. "
        msg += "Have you initialized it? (Use --help on the 'init' command)"
        try:
            total = 0
            for file_data in self._mgr.list_files():
                total += 1
                print("{:30} {}".format(file_data.name, file_data.path))
            if total == 0:
                print("There are no files in the database")
        except LocalManagerError as e:
            if e.error_type == ERROR_CANT_READ:
                self.usage(message=msg)
            else:
                log(e.message, sys.stderr, bold=True)
            return -1


if __name__ == '__main__':
    argp, subp = buildArgumentParser()
    args = argp.parse_args()            #(sys.argv[1:])
    if args.action is None:
        msg = "\n\tNo action specified for caldb"
        usage(argp, message=msg)
        sys.exit(-1)

    conf = load_calconf()

    # Override some options if the user has specified the path to
    # a database
    if args.db_path is not None:
        update_calconf(dict(
            standalone=True,
            database_dir=args.db_path
        ))
        conf = get_calconf()

    ret = -1
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
        raise
        usage(argp, message="The database location is undefined.")

    sys.exit(ret)
