#!/usr/bin/env python
#                                                                        DRAGONS
#
#                                                                       caldb.py
# ------------------------------------------------------------------------------
# caldb.py -- local calibration database management tool
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
import os
import sys
import traceback

from os.path import isdir, exists, expanduser
from argparse import ArgumentParser
from functools import partial

from recipe_system.config import load_config
from recipe_system.cal_service import get_db_path_from_config, LocalDB, RemoteDB, parse_databases
from recipe_system.cal_service.localmanager import LocalManagerError
from recipe_system.cal_service.localmanager import ERROR_CANT_WIPE, ERROR_CANT_CREATE
from recipe_system.cal_service.localmanager import ERROR_CANT_READ, ERROR_DIDNT_FIND
# ------------------------------------------------------------------------------

def buildArgumentParser():

    parser = ArgumentParser(description="Calibration Database Management Tool")
    sub = parser.add_subparsers(help="Sub-command help", dest='action')

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

    p_config = sub.add_parser('config', help="Show configuration for caldb")

    for sp in (p_add, p_init, p_list, p_remove, p_config):
        sp.add_argument('-c', '--config', dest='config',
                        help="Path to the config file, if not the default "
                             "location, or defined by the environment "
                             "variable.")
        sp.add_argument('-d', '--database', dest='db_path', default=None,
                        help="Path to the database file. Optional if the "
                             "config file defines a single database.")
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
            message = "\x1b[1m{}\x1b[0m".format(message)
        print(message, file=stream)
        if add_newlines > 0:
            print('\n' * add_newlines, file=stream)


class Dispatcher:
    def __init__(self, parser, db, log):
        self.db = db
        self._parser = parser
        self._log = log

    def apply(self, action, args):
        try:
            method = getattr(self, f"_action_{action}")
        except AttributeError:
            raise AttributeError(f"No such action '{action}'")

        return method(args)

    def usage(self, message):
        usage(self._parser, message)

    def _action_add(self, args):
        for path in args.files:
            try:
                if isdir(path):
                    # Do something about verbose...
                    if args.walk:
                        m = f"Ingesting the files under {path}"
                    else:
                        m = f"Ingesting the files at {path}"
                    self._log(m)
                    self.db.add_directory(path, walk=args.walk)
                else:
                    self.db.add_cal(path)
                    self._log(f"Ingested {path}")
            except OSError as e:
                try:  # reproduce previous code
                    traceback.print_last()
                except ValueError:
                    pass
                print(e, file=sys.stderr)
                return -1
            except ValueError as e:
                print(e, file=sys.stderr)
                return -1

        return 0

    def _action_remove(self, args):
        for path in args.files:
            try:
                self.db.remove_cal(path)
                self._log(f"Removed {path}")
            except LocalManagerError as e:
                if e.error_type == ERROR_DIDNT_FIND:
                    log(e.message, sys.stderr)

    def _action_init(self, args):
        try:
            if not args.wipe and os.path.exists(self.db.dbfile):
                msg = ("You are initializing a database in a pre-existing file "
                       "without wiping it first. \nThis will ensure the "
                       "database tables exist and are properly configured \nbut"
                       " will not erase database contents. If you mean to "
                       "start-over, re-run \nwith the -w flag to wipe the "
                       "existing database.")
                print(msg)
            self._log("Initializing {}...".format(self.db.dbfile))
            self.db.init(wipe=args.wipe)
        except LocalManagerError as e:
            if e.error_type == ERROR_CANT_WIPE:
                print(f"Error wiping existing database file "
                          f"{self.db.dbfile}")
            elif e.error_type == ERROR_CANT_CREATE:
                print(f"Error creating database file {self.db.dbfile}")
            return -1

        return 0

    def _action_list(self, args):
        msg = "Could not read information from the database. "
        msg += "Have you initialized it? (Use --help on the 'init' command)"
        print(f"Database: {self.db.name}")
        try:
            total = 0
            for file_data in self.db.list_files():
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

    def _action_config(self, args):
        from recipe_system.config import STANDARD_REDUCTION_CONF
        from recipe_system.config import expand_filenames

        # print(f"in caldb config {self.db.db_file}")
        isactive = "The calibration dbs are all local, meaning that remote " \
                   "calibrations will not be downloaded "
        inactive = "There are remote dbs configured, meaning that remote " \
                   "calibrations will be downloaded."

        # parse up here, just to keep warnings and errors from cluttering up the config output
        dbs = parse_databases()

        print('')

        dragonsrc_location = STANDARD_REDUCTION_CONF
        dragonsrc_override = os.getenv('DRAGONSRC', None)
        if dragonsrc_override:
            print(f"\033[1mDRAGONS rc file override from environment: {dragonsrc_override}\033[0m")
            print('')
            dragonsrc_location = dragonsrc_override
        if not exists(expanduser(dragonsrc_location)):
            print(f"\033[1mDRAGONS will fall back to legacy config, rc file {dragonsrc_location} not found\033[0m")
            print('')

        all_local = True
        filenames = expand_filenames(deprecation_warning=False)
        print(f"Using configuration files: \033[1m{filenames}\033[0m")
        print('')
        for db in dbs:
            if db[0] != LocalDB:
                all_local = False
            print(db[1])
            print(f"  Type:  {db[0].__name__}")
            print(f"  Get:   {db[2]['get_cal']}")
            print(f"  Store: {db[2]['store_cal']}")
            print('')
        print(f"\nDatabase file: \033[1m{self.db.name}\033[0m")
        if not exists(os.path.expanduser(self.db.name)):
            print("   NB: The database does not exist. Please initialize it.")
            print("       (Read the help message about 'init' command)")
        print()
        print(isactive if all_local else inactive)


if __name__ == '__main__':
    argp, subp = buildArgumentParser()
    args = argp.parse_args()            #(sys.argv[1:])
    if args.action is None:
        msg = "\n\tNo action specified for caldb"
        usage(argp, message=msg)
        sys.exit(-1)

    if args.db_path is None:
        load_config()
        db_path = get_db_path_from_config()
    else:
        db_path = args.db_path

    ret = -1
    try:
        act = args.action
        logstream = sys.stderr if args.verbose else None
        db = LocalDB(db_path, log=None)
        disp = Dispatcher(subp.choices[act], db,
                          partial(log, stream=logstream))
        ret = disp.apply(act, args)
    except AttributeError as e:
        raise
        usage(argp, message="The database location is undefined.")

    sys.exit(ret)
