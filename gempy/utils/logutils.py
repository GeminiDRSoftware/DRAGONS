"""
gempy.utils.logutils provides functions to support logging in DRAGONS. There are
two groups of functions:

Logging setup:

- customize_logger()
DRAGONS logging uses some extra non-standard logging levels. The
customize_logger() function provided here will add the extra levels to the
logging namespace, and will also add methods to a logger to log messages
with those levels - eg log.fullinfo() is the equivalent of log.info() but for
the custom FULLINFO level. It will also add a filter to avoid identical repeat
warning messages. It will *not* add or remove handlers to or from the logger.

3rd party applications calling dragons code that configure their own log
handlers should call customize_logger() to add the extra levels to the logger.

- config()
This is the traditional logging configuration method for DRAGONS. It will call
customize_logger() to set up additional log levels. It will also add file and
stream (console) handlers to the logger. The console handler gets a custom
formatter that includes the level name in the output only for non-info-like
levels.

Some other utility methods are provided to support logging in DRAGONS:

- get_logger() gets the root logger, and ensures that customize_logger has been
run on it.

- update_indent() updates the format strings, and the console log custom
formatter, to facilitate indenting log messages to reflect call stack depth in
recipes and primitives.
"""

#
#                                                                  gemini_python
#
#                                                                    gempy.utils
#                                                                    logutils.py
# ------------------------------------------------------------------------------

import logging
from logging import handlers
from collections.abc import Iterable

STDFMT = '%(asctime)s %(levelname)-8s - %(message)s'
DBGFMT = '%(asctime)s %(name)-40s - %(levelname)-8s - %(message)s'
SW = 3

# Turn off logging exception messages
logging.raiseExceptions = 0

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


def customize_logger(log=None):
    """
    Adds the DRAGONS custom log levels to the logging module.
    Adds methods to the given logger (or the root logger if logger is None)
    to facilitate logging messages with the custom log levels.

    The extra logging methods added are: logger.status(), logger.stdinfo() and
    logger.fullinfo()

    Parameters
    ----------
    log : <logging.Logger>
          Logger object from logging.getLogger(). Usually leave this as None
          to have the extra methods added to the root logger

    """
    if log is None:
        log = logging.getLogger()

    # If this logger has already been customized, this is a no-op
    if getattr(log, 'customized_for_dragons', None):
        return

    # Add the marker flag to the log object to record the fact that it has
    # been customized for DRAGONS.
    log.customized_for_dragons = True

    # Register the extra logging levels with the logging module. The native
    # levels can be found at
    # https://docs.python.org/3/library/logging.html#levels
    extra_levels = {25: 'STATUS', 21: 'STDINFO', 15: 'FULLINFO'}
    for level in extra_levels:
        levelname = extra_levels[level]
        logging.addLevelName(level, levelname)

    # Add the extra methods to the given logger to facilitate
    # logging messages with the custom log levels.

    # It may be possible to do this more "programmatically"
    # But here's a simple version for now
    def status(msg, *args, **kwargs):
        log._log(25, msg, args, **kwargs)

    def stdinfo(msg, *args, **kwargs):
        log._log(21, msg, args, **kwargs)

    def fullinfo(msg, *args, **kwargs):
        log._log(15, msg, args, **kwargs)

    setattr(log, 'status', status)
    setattr(log, 'stdinfo', stdinfo)
    setattr(log, 'fullinfo', fullinfo)

    # Add the filter for duplicate warning messages
    add_filter(DuplicateWarningFilter, log)


def get_logger(name=None):
    """
    Wraps logging.getLogger and ensures the logger returned has been
    customized. Note, this does not "configure" the logger - ie set
    the handlers and formatters that are usually used with DRAGONS. That
    should be done explicitly by the script that is going to use it, so that
    other modules calling dragons can use the custom log levels, but define
    their own log handlers etc.

    It could be argued the same should apply to adding the custom levels, but
    this check is left in this code for backwards compatability at least for
    now.

    Parameters
    ----------
    name: <str>
          Name of logger (usually __name__)

    Returns
    -------
    log : <logging.Logger>
          Logger with custom log levels

    """
    log = logging.getLogger(name)
    customize_logger(log)  # This is a no-op if it's already customized
    return log

class DragonsConsoleFormatter(logging.Formatter):
    """
    Dragons configures (at least) two handlers on the logger. One going to
    console, which is typically what the user sees when running DRAGONS
    interactively, and one going to a log file. The format string for the log
    file is verbose and includes the log level in every message.

    The format string for the console output is much more concise and
    traditionally only includes the log level for DEBUG and WARNINGS and above.
    There's no obvious trivial way to conditionally include the level name in
    the log message, this custom formatter class seems the best way, replacing the
    previous implementation using custom methods for all the logging calls
    (.info(), .debug(), etc.).

    In addition, DRAGONS uses indentation in log messages. This handler supports
    an update_indent() method to support this.

    """

    # We instantiate two logging.Formatters here, _short_formatter, whose format
    # string includes an indent and just the message, and_long_formatter, whose
    # format string includes the indent, the log level name, and the message.
    # When we format a log record, if record's level name is in our
    # short_levels list, we call the _short_formatter's format method, otherwise
    # we call the _long_formatter's format method.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initial indent parameters
        self._indent_level = 0
        self._indent_width = 3

        # Log level names for which to emit short form
        self.short_levels = ['STATUS', 'STDINFO', 'INFO', 'FULLINFO']

        self._init_formatters()

    def _indent_str(self):
        return ' ' * (self._indent_level * self._indent_width)

    def _init_formatters(self):
        self._short_fmt_str = self._indent_str() + '%(message)s'
        self._long_fmt_str = self._indent_str() + '%(levelname)s - %(message)s'
        self._short_formatter = logging.Formatter(self._short_fmt_str)
        self._long_formatter = logging.Formatter(self._long_fmt_str)

    def update_indent(self, indent_level):
        self._indent_level = indent_level
        self._init_formatters()

    def format(self, record):
        # Levels for which to emit short form
        if record.levelname in self.short_levels:
            return self._short_formatter.format(record)
        else:
            return self._long_formatter.format(record)


def config(mode='standard', file_name=None, file_lvl=15, stomp=False,
           additional_handlers=None, keep_handlers=False):
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

    keep_handlers : <bool>
        If True, do not remove all existing Handlers. Default is False and will
        remove all existing handlers.
    """

    logfmt = None
    lmodes = ['debug', 'standard', 'quiet', 'rotating', 'verbose']
    fm = 'w' if stomp else 'a'
    mode = mode.lower()
    if mode not in lmodes:
        raise NameError("Unknown logging config mode")

    # Get the root logger, and customize it
    rootlog = logging.getLogger()
    customize_logger(rootlog)

    if not keep_handlers:
        # Drop all existing handlers. We can't simply do a for h in
        # rootlog.handlers here because of list mutability, and the logger
        # documentation says we should treat log.handlers as readonly so we
        # also shouldn't just set it to [] or call handlers.clear()
        for h in range(len(rootlog.handlers)):
            rootlog.removeHandler(rootlog.handlers[0])

    # Add handlers depending on the logging configuration mode
    if file_name:
        filehandler = logging.FileHandler(file_name, mode=fm)
        filehandler.setLevel(file_lvl)
    stdformat = logging.Formatter(STDFMT, datefmt='%Y-%m-%d %H:%M:%S')
    dbgformat = logging.Formatter(DBGFMT, datefmt='%Y-%m-%d %H:%M:%S')
    consolehandler = logging.StreamHandler()
    consolehandler.setFormatter(DragonsConsoleFormatter())

    if mode == 'quiet':
        if file_name:
            filehandler.formatter = stdformat
            rootlog.addHandler(filehandler)

    elif mode == 'standard':
        if file_name:
            filehandler.formatter = stdformat
            rootlog.addHandler(filehandler)
        consolehandler.setLevel(21)
        rootlog.addHandler(consolehandler)

    elif mode == 'verbose':
        if file_name:
            filehandler.formatter = stdformat
            rootlog.addHandler(filehandler)
        consolehandler.setLevel(15)
        rootlog.addHandler(consolehandler)

    elif mode == 'debug':
        if file_name:
            filehandler.formatter = dbgformat
            filehandler.setLevel(10)
            rootlog.addHandler(filehandler)
        consolehandler.setLevel(10)
        rootlog.addHandler(consolehandler)

    elif mode == 'rotating':
        filehandler = logging.handlers.TimedRotatingFileHandler(file_name,
                                                                when='midnight')
        filehandler.formatter = stdformat
        if file_name:
            rootlog.addHandler(filehandler)

    # Attach additional handlers if provided.
    if additional_handlers is not None:
        if isinstance(additional_handlers, Iterable):
            for handler in additional_handlers:
                rootlog.addHandler(handler)
        else:
            rootlog.addHandler(additional_handlers)


def update_indent(li=0, mode=''):
    """
    Updates indents for reduce by changing the formatter

    Parameters
    ----------
    li : <int>
         log indentation

    mode: <str>
          logging mode

    """
    log = logging.getLogger('')

    for hndl in log.handlers:
        if isinstance(hndl.formatter, DragonsConsoleFormatter):
            hndl.formatter.update_indent(li)
        if isinstance(hndl, logging.FileHandler):
            if mode == 'debug':
                ff = logging.Formatter(DBGFMT[:-11] + ' ' * (li * SW) + \
                    DBGFMT[-11:],'%Y-%m-%d %H:%M:%S')
            else:
                ff = logging.Formatter(STDFMT[:-11] + ' ' * (li * SW) + \
                    STDFMT[-11:],'%Y-%m-%d %H:%M:%S')
            hndl.setFormatter(ff)
    return

def change_level(new_level=None):
    """
    Change the level of the console handler

    """
    log = logging.getLogger()
    if new_level:
        for handler in log.handlers:
            if isinstance(handler, logging.StreamHandler):
                log.setLevel(new_level)

class DuplicateWarningFilter(logging.Filter):
    """
    This class contains a filter for log messages to suppress repeated instances
    of the same message. When a different message comes along, it logs a
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
