import sys
import os

import logging
import textwrap
import traceback as tb

from astrodata.Errors import Error


_listOfLoggers = None

# Used to separate stderr stdout stream
class SingleLevelFilter(logging.Filter):
    def __init__(self, passlevel, reject):
        self.passlevel = passlevel
        self.reject = reject

    def filter(self, record):
        if self.reject:
            return (record.levelno == self.passlevel)
        else:
            return (record.levelno != self.passlevel)

class GeminiLogger(object):
    """
    This logger object is used througout the recipe system, associated 
    user level functions, and toolboxes.  It is based on the logging facility
    for python (http://docs.python.org/library/logging.html)
    -------------------------------------------------------------------------
    Log Level Descriptions:
    -------------------------------------------------------------------------
    DEBUG           Engineering, programmer debugging.
    FULLINFO        Details, input parameters, header changes.
    STDINFO         Science info (eg. background level).
    STATUS          Start/end processes, number of files, name of I/O files.
    WARNING         Something out of the ordinary happened but was overlooked. 
    ERROR           An error has occured during the reduction.
    CRITICAL        A serious error occured during the reduction.
             
    :param logName: Name of the file the log messages will be written to
    :type logName: string
    
    :param logLevel: verbosity setting for the lowest level of messages to 
                     print to the screen.
    :type logLevel: string
    
    :param logType: A string to indicate the type of usage of the log object.
    :type logType: string
    
    :param debug: Flag to log debug level messages to file. 
                  NOTE: this is independent of logLevel for the screen msgs.
    :type debug: boolean
    
    :param noLogFile: Flag for stopping a log file from being created
    :type noLogFile: boolean
    
    :param allOff: Flag to turn off all messages to the screen or file and 
                   to not make a log file. ie, completely ignore all messages.
    :type allOff: boolean
    
    """
    logger = None

    def __init__(self, logName=None, logLevel='critical', logType='main', debug=False, 
                 noLogFile=False, allOff=False, indentLevel=0):
        if logLevel==None:
            logLevel = 'critical'
        if allOff:
            logLevel='none'
            noLogFile=True
        self._logLevel = logLevel.lower()
        
        # save the type setting for the log to a private variable to allow for 
        # changing the value in future calls.
        if isinstance(logType,str):
            self._logType = logType.lower()
        else:
            raise Error('The value of logType must be a string.')
        self._debug = debug
        if noLogFile:
            self._logName = '/dev/null'
        else:
            if logName==None:
                self._logName = 'gemini.log'
            else:
                self._logName = logName
        
        # Adding logger levels not in default Python logger 
        # note: INFO level = 20 and it is not being used
        # Please note that the log_levels are numbered internally by 
        #  the built-in python logger system.
        #
        # log_number         log_level
        # --------------------------------
        #     50            CRITICAL
        #     40            ERROR
        #     30            WARNING
        #     25            STATUS
        #     21            STDINFO
        #     15            FULLINFO
        #     10            DEBUG
        self.FULLINFO    = 15
        self.STDINFO     = 21
        self.STATUS      = 25
        log_levels = {
                      self.FULLINFO  : 'FULLINFO',
                      self.STDINFO   : 'STDINFO',
                      self.STATUS    : 'STATUS'}
        for lvl in log_levels.keys():
            logging.addLevelName(lvl, log_levels[lvl])
        
        # Create logger object and its default level
        self.logger = logging.getLogger(self._logName)
        self.logger.setLevel(logging.DEBUG)
    
        setattr(self.logger, 'stdinfo', 
                lambda *args: self.logger.log(self.STDINFO, *args))
        setattr(self.logger, 'status', 
                lambda *args: self.logger.log(self.STATUS, *args))
        setattr(self.logger, 'fullinfo', 
                lambda *args: self.logger.log(self.FULLINFO, *args))

        ## Creating and setting up the levels of the handlers
        # initialize the console and file handlers
        self.initializeHandlers()
        # set the console handler level
        self.setConsoleLevel(logLevel) 
        # Remove any pre-existing handlers, set up formatters and add handlers 
        # to logger.         
        self.finalizeHandlers()      
                
        # Default category strings by order of importance
        self._criticalDefaultCategory = 'critical'
        self._errorDefaultCategory = 'error'
        self._warningDefaultCategory = 'warning'
        self._statusDefaultCategory = 'status'
        self._stdinfoDefaultCategory = 'stdinfo'
        self._infoDefaultCategory = 'info'
        self._fullinfoDefaultCategory = 'fullinfo'
        self._debugDefaultCategory = 'debug'
        
        # Set indentation parameter 
        # (so that output from nested recipes will be indented)
        self.indentLevel = indentLevel

        # Define a text wrapper for formatting output lines
        self.wrapper = textwrap.TextWrapper(width=80,break_long_words=False,
                                       initial_indent=self.indentLevel*'   ',
                                       subsequent_indent=self.indentLevel*'   ')

    def initializeHandlers(self):
        """
        A function to initialize and then set the file handler level depending
        on if the log has 'debug' set to True or False; if True, file handler
        level goes to highest value (DEBUG), else it goes to default
        of (FULLINFO).
        """
        # Create console and file handlers
        self.ch = logging.StreamHandler(sys.stdout)
        
        self.fh = logging.FileHandler(self._logName)
        
        ## Set level for the file handler 
        # Use file handler level FULLINFO unless debug is True
        #NOTE: Using private member variable _fhLogLevel that could possibly
        #      be made available to user in the future. But I can't see a reason
        #      to do so at the moment.
        if self._debug:
            self._fhLogLevel = logging.DEBUG 
        else:
            self._fhLogLevel = self.FULLINFO
        self.fh.setLevel(self._fhLogLevel)

    def finalizeHandlers(self):
        """
        This function will set up the console and file message formats, remove 
        any previous handlers to ensure they are not doubled, then add the 
        finalized handlers to the the logger.
        """
        # Create formatters for console and file messages
        ch_formatter = logging.Formatter('%(message)s')
        fh_formatter = logging.Formatter('%(asctime)s %(levelname)-8s '+
                                         '- %(message)s')
        
        # Add formatters to the handlers
        self.ch.setFormatter(ch_formatter)
        self.fh.setFormatter(fh_formatter)
        
        #add filter
        f1 = SingleLevelFilter(logging.ERROR, False)
        self.ch.addFilter(f1)

        # Add console and file handlers to the logger
        self.logger.addHandler(self.ch)
        self.logger.addHandler(self.fh)

    def setConsoleLevel(self, logLevel):
        """
        Set the level in the console handler.
        
        :param logLevel: verbosity setting for the lowest level of messages to 
                         print to the screen.
        :type logLevel: integer from 0-6, or 10 following above chart
        
        """
        if (logLevel == 'debug'):
            self.ch.setLevel(logging.DEBUG)
        elif (logLevel == 'fullinfo'):
            self.ch.setLevel(self.FULLINFO)
        elif (logLevel == 'stdinfo'):
            self.ch.setLevel(self.STDINFO)
        elif (logLevel == 'status'):
            self.ch.setLevel(self.STATUS)
        elif (logLevel == 'warning'):
            self.ch.setLevel(logging.WARNING)
        elif (logLevel == 'error'):
            self.ch.setLevel(logging.ERROR)
        elif (logLevel == 'critical'):
            self.ch.setLevel(logging.CRITICAL)
        elif (logLevel== 'none'):
            #ie. 'MAX' out the level so it is above all existing log levels
            # so no messages will ever have a high enough level to go to screen.
            self.ch.setLevel(100)
        else:
            # set to default, CRITICAL, if all else are false
            self.ch.setLevel(logging.CRITICAL)
    
    def changeIndent(self, indentLevel=None):
        self.indentLevel = indentLevel
        self.wrapper = textwrap.TextWrapper(width=80,break_long_words=False,
                                       initial_indent=self.indentLevel*'   ',
                                       subsequent_indent=self.indentLevel*'   ')
 
    def changeLevels(self, logLevel=None, debug=None):
        """
        A function to allow for changing the level of the console handler 
        (and file handler if debug is set to True or False).
        It re-initializes the handlers, re-sets their levels and then
        re-finalizes them.
        
        :param logLevel: The NEW verbosity setting for the lowest level of 
                         messages to print to the screen.
        :type logLevel: integer from 0-6, or 10 following above chart
        
        :param debug: Flag to log debug level messages to file. 
                      NOTE: this is independent of logLevel for the screen msgs.
        :type debug: python boolean (True/False). 
                     None indicates to not change file handler level. 
        """
        # updating private member variable _debug if debug not None
        # re-initializing the handlers 
        # setting console level of re-initialized console handler
        # re-finalizing the re-initialized handlers
        #return the log object with its handlers updated with new logLevel 
        if debug==True:
            self._debug = True
        elif debug==False:
            self._debug = False
        self.initializeHandlers()
        self.setConsoleLevel(logLevel)
        self.finalizeHandlers()
        return self
    
    def logname(self):
        """
        Just a function to return the 'private' member variable _logName
        to allow checking if logger with the same file name exists all ready
        
        """
        return self._logName
    
    def logtype(self, newLogType=None):
        """
        Just a function to return the 'private' member variable _logType
        to allow checking if logger with the same file type exists all ready,
        OR if the user wishes to change the logger's current logType to a new 
        one.
        
        :param newLogType: A new value to change the logger's logType to.
        :type newLogType: String in all lower case.
        """
        if newLogType!=None:
            if isinstance(newLogType, str):
                self._logType = newLogType.lower()
        return self._logType
    
    def loglevel(self):
        """
        Just a function to return the 'private' member variable _logLevel
        to allow checking what the current verbosity of this logger object is.        
        """     
        return self._logLevel
    
    def defaultCategory(self, level=None, category=None):
        """
         A function to access and set the 'private' default category variables.
         If category = None, then the current value will be returned.
         Else, the default value will be replaced with the newly provided 
         category string.
         For the case where the user would like ALL the levels to have the 
         same default category, then set level = 'ALL' in the call to 
         defaultCategory. 
         ie log.defaultCategory(level='ALL', category='tulips')
         
         :param level: level to edit the default category for. 
                       eg. fullinfo, stdinfo, status...
         :type level: string
         
         :param level: new default value for the levels category
         :type level: string
         
        """
        if level == 'ALL':
            self._criticalDefaultCategory = category
            self._errorDefaultCategory = category
            self._warningDefaultCategory = category
            self._statusDefaultCategory = category
            self._stdinfoDefaultCategory = category
            self._infoDefaultCategory = category
            self._fullinfoDefaultCategory = category
            self._debugDefaultCategory = category
            
        if level == 'critical':
            # If no category value was passed in for this level then 
            # just return the current value, else set the default category
            # to the string passed in
            if category is None:
                return self._criticalDefaultCategory
            else:
                self._criticalDefaultCategory = category
        elif level == 'error':
            if category is None:
                return self._errorDefaultCategory
            else:
                self._errorDefaultCategory = category
        elif level == 'warning':
            if category is None:
                return self._warningDefaultCategory
            else:
                self._warningDefaultCategory = category
        elif level == 'status':
            if category is None:
                return self._statusDefaultCategory
            else:
                self._statusDefaultCategory = category
        elif level == 'stdinfo':
            if category is None:
                return self._stdinfoDefaultCategory
            else:
                self._stdinfoDefaultCategory = category
        elif level == 'info':
            if category is None:
                return self._infoDefaultCategory
            else:
                self._infoDefaultCategory = category
        elif level == 'fullinfo':
            if category is None:
                return self._fullinfoDefaultCategory
            else:
                self._fullinfoDefaultCategory = category
        elif level == 'debug':
            if category is None:
                return self._debugDefaultCategory
            else:
                self._debugDefaultCategory = category
            
    def fullinfo(self, msg, category=None):
        """ The function to call for making 'fullinfo' level log messages
        """
        # Check if a category string was passed in, else set to default value
        # In case the message is multiple lines, break them up into a list
        if category is None:
            category = self._fullinfoDefaultCategory
        msgs = str(msg).split('\n')
        
        # Loop through the message lines in the list and log them
        # split long lines up, and print blank line if message whitespace
        for line in msgs:
            new_lines = self.wrapper.wrap(line)
            if len(new_lines)==0:
                self.logger.fullinfo('')
            else:
                for new_line in new_lines:
                    self.logger.fullinfo(new_line)

    def info(self, msg, category=None):
        """ The function to call for making 'info' level log messages
        """
        if category is None:
            category = self._infoDefaultCategory
        msgs = str(msg).split('\n')
        for line in msgs:
            new_lines = self.wrapper.wrap(line)
            if len(new_lines)==0:
                self.logger.info('')
            else:
                for new_line in new_lines:
                    self.logger.info(new_line)
     
    def stdinfo(self, msg, category=None ):
        """ The function to call for making 'stdinfo' level log messages
        """
        if category is None:
            category = self._stdinfoDefaultCategory
        msgs = str(msg).split('\n')
        for line in msgs:
            new_lines = self.wrapper.wrap(line)
            if len(new_lines)==0:
                self.logger.stdinfo('')
            else:
                for new_line in new_lines:
                    self.logger.stdinfo(new_line)

    def status(self, msg, category=None):
        """ The function to call for making 'status' level log messages
        """
        if category is None:
            category = self._statusDefaultCategory
        msgs = str(msg).split('\n')
        for line in msgs:
            new_lines = self.wrapper.wrap(line)
            if len(new_lines)==0:
                self.logger.status('')
            else:
                for new_line in new_lines:
                    self.logger.status(new_line)
    
    def debug(self, msg, category=None): 
        """ The function to call for making 'debug' level log messages"""
        # Check if a category string was passed in, else set to default value 
        # Retrieving the important parts of the call stack using callInfo()
        # In case the message is multiple lines, break them up into a list
        # Loop through the message lines in the list and log them
        # split long lines up and add indentation
        # print blank line if msg was only whitespace
        if category is None:
            category = self._debugDefaultCategory  
        b = callInfo()
        msgs = str(msg).split('\n')
        for line in msgs:
            new_lines = self.wrapper.wrap(category.ljust(10).upper()+'-' + 
                                          b[0].ljust(20)+' - '+
                                          b[2].ljust(20)+'-' +
                                          str(b[1]).ljust(3) +
                                          ' - '+line)
            if len(new_lines) == 0:
                self.logger.debug('')
            else:
                for new_line in new_lines:
                    self.logger.debug(new_line)
    
    def critical(self, msg, category=None):
        """ The function to call for making 'critical' level log messages"""
        if category is None:
            category = self._criticalDefaultCategory
        b = callInfo()
        msgs = str(msg).split('\n')
        for line in msgs:
            new_lines = self.wrapper.wrap(category.ljust(10).upper()+'-' + 
                                          b[0].ljust(20)+' - '+
                                          b[2].ljust(20)+'-' +
                                          str(b[1]).ljust(3) +
                                          ' - '+line)
            if len(new_lines)==0:
                self.logger.critical('')
            else:
                for new_line in new_lines:
                    self.logger.critical(new_line)
            
    def warning(self, msg, category=None):
        """ The function to call for making 'warning' level log messages
        """
        if category is None:
            category = self._warningDefaultCategory
        b = callInfo()
        msgs = str(msg).split('\n')
        for line in msgs:
            new_lines = self.wrapper.wrap('WARNING - '+line)
            if len(new_lines)==0:
                self.logger.warning('')
            else:
                for new_line in new_lines:
                    self.logger.warning(new_line)
            
    def error(self, msg, category=None):
        """ The function to call for making 'error' level log messages
        """
        if category is None:
            category = self._errorDefaultCategory
        b = callInfo()
        msgs = str(msg).split('\n')
        #file_handler = open(self._logName, w)
        for line in msgs:
            #The ERROR is a temp fix until a solution can be found
            #sys.stderr.write("ERROR    40- " + category.ljust(10) + '-' +
            #    b[0].ljust(20)+' - '+ b[2].ljust(20) + '-' +
            #    str(b[1]).ljust(3) + ' - ' + line + "\n")
            # split long lines up and add indentation
            new_lines = self.wrapper.wrap('ERROR - '+line)
            for new_line in new_lines:
                sys.stderr.write(new_line+'\n')
            self.logger.error(category.ljust(10) + '-' + 
                b[0].ljust(20)+' - '+ b[2].ljust(20) + '-' + 
                str(b[1]).ljust(3) + ' - ' + line)

#functions outside GeminiLogger class------------------------------------------

def callInfo():
    """ A function used by log levels debug, critical, warning and error 
    to allow for adding information about the call stack in the log messages.
    The information is returned in the format of a list of strings 
    ['module name', 'line number', 'function']
    """
    st = tb.extract_stack()
    return [os.path.basename(st[-3][0]), st[-3][1], st[-3][2]] 
    
def createGeminiLog(logName=None, logLevel=None, logType='main', debug=False, 
                    noLogFile=False, allOff=False):
    """This function is to create a new logger object.
    
    :param logName: Name of the file the log messages will be written to
    :type logName: string
    
    :param logLevel: Lowest log level of messages to print to the screen.
                     (i.e status will print itself, warning, error and
                      critical, but skip over stdinfo, fullinfo, and debug)
    :type logLevel: string 
    
    :param logType: Indicates the type of usage for the logger object.
    :type logType: string
    
    :param debug: Show debug level messages.
    :type debug: boolean
    
    :param noLogFile: Prevent a logfile from being created.
    :type noLogFile: boolean
    
    :param allOff: Turn off all log messaging (logfile not created as well).
    :type allOff: boolean
    
    """
    global _listOfLoggers
    _geminiLogger = None
    if not _listOfLoggers:
        _listOfLoggers = []
    try:
        _geminiLogger = GeminiLogger(logName=logName, logLevel=logLevel, 
                                     logType=logType, debug=debug, 
                                     noLogFile=noLogFile, allOff=allOff)
        _listOfLoggers.append(_geminiLogger)
    except:
        raise Error('An error occured while trying to create logger object \
                    named, '+str(logName)+', of type, '+str(logType)+'.')
    return _geminiLogger
        
def getGeminiLog(logLevel=None, logType='main'):
    """ The function called to retrieve the desired logger object based on the 
    parameter 'logType', and update its logLevel value if not None.   
    If the requested logger does not exist, then the 'main' logger is passed 
    back, and if that does not exist then a null logger is returned 
    (ie, no logfile, no messages to screen).  
    Thus, a logger will always be returned when getGeminiLog is called.
        
    :param logLevel: Lowest log level of messages to print to the screen.
                     (i.e status will print itself, warning, error and
                      critical, but skip over stdinfo, fullinfo, and debug)
    :type logLevel: string 
    
    :param logType: Indicates the type of usage for the logger object.
    :type logType: string
    """
    global _listOfLoggers
    _geminiLogger = None
    if not _listOfLoggers:
        _geminiLogger = createGeminiLog(allOff=True)
    else:     
        # variable to hold the 'main' log to pass back if non-main one requested
        # doesn't exist in list.
        # Loop through logs in list
        # The log of the type requested is found in list
        # Updating the value of the logLevel if passed in value was 
        # not None or the same as the current logger's logLevel.
        # Converting logLevel to its int value if needed
        # non-main requested, but main found, so store it for later 
        mainLog = None
        for log in _listOfLoggers:                
            if log.logtype()==logType:
                if logLevel!=None:
                    if logLevel!=log.loglevel():
                        #print '*******TRACEBACK:'+repr(tb.extract_stack()[-2])
                        _geminiLogger = log.changeLevels(logLevel=logLevel)
                        break
                    else:
                        _geminiLogger = log
                        break
                else:
                    _geminiLogger = log
                    break
            elif log.logtype()=='main':
                mainLog = log
                
        # The non-main log requested was not in the list, So pass back 'main' 
        # if it was found, else create an alloff=True log to be passed back 
        # (ie, no log file, no msgs to screen).
        if not _geminiLogger:
            if mainLog:
                _geminiLogger = mainLog
            else:
                _geminiLogger = createGeminiLog(allOff=True)
    return _geminiLogger
