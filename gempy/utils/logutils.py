#
#                                                                  gemini_python
#
#                                                                    gempy.utils
#                                                                    logutils.py
# ------------------------------------------------------------------------------
# $Id$
# ------------------------------------------------------------------------------
import types
import logging
from datetime import time
from logging import handlers

STDFMT = '%(asctime)s %(levelname)-8s - %(message)s'
DBGFMT = '%(asctime)s %(name)-40s - %(levelname)-8s - %(message)s'
SW = 3

# Turn off logging exception messages
logging.raiseExceptions = 0

# Logging levels
ll = {'CRITICAL':50, 'ERROR':40, 'WARNING' :30, 'STATUS':25,
      'STDINFO' :21, 'INFO' :20, 'FULLINFO':15, 'DEBUG' :10}

def add_filter(filt, log):
    """
    Add given filter to given logger, checking for duplicates first.

    Parameters
    ----------
    filt : class inheriting from `logging.Filter`
        The filter to be added.
    log : `logging.logger` object.
        The logger to which the filter is to be added.

    Returns
    -------
    None.

    """
    if sum([isinstance(f, filt) for f in log.filters]) < 1:
        log.addFilter(filt(log))


def customize_log(log=None):
    """
    Sets up custom attributes for logger

    Parameters
    ----------
    log : <logging.Logger>
          Logger object from logging.getLogger()

    Returns
    -------
    <void>

    """
    def arghandler(args=None, levelnum=None, prefix=None):
        largs = list(args)
        slargs = str(largs[0]).split('\n')
        for line in slargs:
            if prefix:
                line = prefix + line
            if len(line) == 0:
                log.log(levelnum, '')
            else:
                log.log(levelnum, line)

    def ccritical(*args):
        arghandler(args, ll['CRITICAL'], 'CRITICAL - ' )
    def cerror(*args):
        arghandler(args, ll['ERROR'], 'ERROR - ')
    def cwarning(*args):
        arghandler(args, ll['WARNING'], 'WARNING - ')
    def cstatus(*args):
        arghandler(args, ll['STATUS'])
    def cstdinfo(*args):
        arghandler(args, ll['STDINFO'])
    def cinfo(*args):
        arghandler(args, ll['INFO'])
    def cfullinfo(*args):
        arghandler(args, ll['FULLINFO'])
    def cdebug(*args):
        arghandler(args, ll['DEBUG'], 'DEBUG - ')

    setattr(log, 'critical', ccritical)
    setattr(log, 'error', cerror)
    setattr(log, 'warning', cwarning)
    setattr(log, 'status', cstatus)
    setattr(log, 'stdinfo', cstdinfo)
    setattr(log, 'info', cinfo)
    setattr(log, 'fullinfo', cfullinfo)
    setattr(log, 'debug', cdebug)

    add_filter(DuplicateWarningFilter, log)

    return

def get_logger(name=None):
    """
    Wraps logging.getLogger and returns a custom logging object

    Parameters
    ----------
    name: <str>
          Name of logger (usually __name__)

    Returns
    -------
    log : <logging.Logger>
          Logger with new levels and prefixes for some levels

    """
    log = logging.getLogger(name)
    try:
        assert log.root.handlers
        customize_log(log)
    except AssertionError:
        config(mode='standard')
        customize_log(log)
    return log

def config(mode='standard', file_name=None, file_lvl=15, stomp=False,
           additional_handlers=None):
    """
    Controls Dragons logging configuration.

    Parameters
    ----------
    mode : <str>
          logging mode: 'debug', 'standard', 'quiet'

    file_lvl : <int>
          file logging level

    file_name : <atr>
          filename of the logger

    stomp : <bool>
          Controls append to logfiles found with same name

    additional_handlers : `logging.Handler` or <list>
        An initialized handler or a list of initialized handlers to be added to the
        root logger.

    Returns
    -------
    <void>

    """
    logfmt = None
    lmodes = ['debug', 'standard', 'quiet', 'rotating']
    fm = 'w' if stomp else 'a'
    mode = mode.lower()
    if mode not in lmodes:
        raise NameError("Unknown mode")

    rootlog = logging.getLogger('')
    rootlog.handlers = []     # every call on config clears the handlers list.

    # Add the new levels
    logging.addLevelName(ll['STATUS'], 'STATUS')
    logging.addLevelName(ll['STDINFO'], 'STDINFO')
    logging.addLevelName(ll['FULLINFO'], 'FULLINFO')

    # Define rootlog handler(s) through basicConfig() according to mode
    customize_log(rootlog)
    if mode == 'quiet':
        logfmt = STDFMT
        logging.basicConfig(level=file_lvl, format=logfmt,
                            datefmt='%Y-%m-%d %H:%M:%S',
                            filename=file_name, filemode=fm)

    elif mode == 'standard':
        logfmt = STDFMT
        console_lvl = 21
        logging.basicConfig(level=file_lvl, format=logfmt,
                            datefmt='%Y-%m-%d %H:%M:%S',
                            filename=file_name, filemode=fm)

        # add console handler for rootlog through addHandler()
        console = logging.StreamHandler()
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        console.setLevel(console_lvl)
        rootlog.addHandler(console)

    elif mode == 'debug':
        logfmt = DBGFMT
        console_lvl = 10
        file_lvl = 10
        logging.basicConfig(level=file_lvl, format=logfmt,
                            datefmt='%Y-%m-%d %H:%M:%S',
                            filename=file_name, filemode=fm)

        # add console handler for rootlog through addHandler()
        console = logging.StreamHandler()
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        console.setLevel(console_lvl)
        rootlog.addHandler(console)

    elif mode == 'rotating':
        log_handler = handlers.TimedRotatingFileHandler(file_name, when='midnight')
        formatter = logging.Formatter(STDFMT, datefmt='%Y-%m-%d %H:%M:%S')
        log_handler.setFormatter(formatter)
        log_handler.setLevel(file_lvl)
        rootlog.addHandler(log_handler)

    # Attach additional handlers if provided.
    if additional_handlers is not None:
        if isinstance(additional_handlers, list):
            for handler in additional_handlers:
                rootlog.addHandler(handler)
        else:
            rootlog.addHandler(additional_handlers)

    return

def update_indent(li=0, mode=''):
    """
    Updates indents for reduce by changing the formatter

    Parameters
    ----------
    li : <int>
         log indentation

    mode: <str>
          logging mode

    Returns
    -------
    <void>

    """
    log = logging.getLogger('')

    # Handle the case if logger has not been configured
    if len(log.handlers) == 0:
        return

    for hndl in log.handlers:
        if isinstance(hndl, logging.StreamHandler):
            sf = logging.Formatter(' ' * (li * SW) + '%(message)s')
            hndl.setFormatter(sf)
        if isinstance(hndl, logging.FileHandler):
            if mode == 'debug':
                ff = logging.Formatter(DBGFMT[:-11] + ' ' * (li * SW) + \
                    DBGFMT[-11:],'%Y-%m-%d %H:%M:%S')
            else:
                ff = logging.Formatter(STDFMT[:-11] + ' ' * (li * SW) + \
                    STDFMT[-11:],'%Y-%m-%d %H:%M:%S')
            hndl.setFormatter(ff)
    return

def change_level(new_level=''):
    """
    Change the level of the console handler

    """
    log = logging.getLogger('')
    for hndl in log.handlers:
        if isinstance(hndl, logging.StreamHandler):
            if new_level:
                hndl.setLevel(ll[new_level.upper()])
    return

class DuplicateWarningFilter(logging.Filter):
    """
    This class contains a filter for log messages to suppress repeated instances
    of the same message. When a different message comes along, it prints a
    message summarizing how many duplicate messages were suppressed.

    Parameters
    ----------
    logger: a `logging.Logger` object
        The logger to which the filtering should be applied.

    """
    def __init__(self, logger):
        self.logger = logger
        self.counter = 1

    def filter(self, record):
        # Only operate on warnings, pass everything else through.
        if record.levelno != 30:  # Not a warning
            return True

        current_log = (record.module, record.levelno, record.msg)
        if current_log != getattr(self, "last_log", None):
            self.last_log = current_log
            if self.counter > 1:
                temp_log = current_log
                self.logger.warning(f"Last message repeated {self.counter} times")
                self.last_log = temp_log
                self.counter = 1
            return True
        self.counter += 1
        return False


if __name__ == "__main__":
    log = get_logger("testing_logutils")
    for i in range(3):
        log.warning("This is a test")
    log.warning("Some other message to generate the repeated message")
