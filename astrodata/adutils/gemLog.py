#Author: Kyle Mede, 2010.    
import sys, os

import logging
import traceback as tb
from astrodata.Errors import Error

_listOfLoggers = None


# Used to apply stderr stdout stream
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
    This is a logger object for use througout the Gemini recipe system and 
    associated User Level Functions and toolboxes.  
    It is based on the Python logging object.
    
    Logging levels chart:
    
    level#, logLevel,  level,        includes (current rough outline)

    10       10      debug           engineering, programmer debugging
    15       6       fullinfo        details, input parameters, header changes
    20               info
    21       5       stdinfo         science info eg. background level
    25       4       status          start processing/end, # of files, 
                                     name of inputs/output files
    30       3       warning    
    40       2       error    
    50       1       critical
             0       none            
             
    note: level 'info' exists, but it is not being used in the
    Recipe Systems logging standards.
    
    :param logName: Name of the file the log messages will be written to
    :type logName: string
    
    :param logLevel: verbosity setting for the lowest level of messages to 
                     print to the screen.
    :type logLevel: integer from 0-6, or 10 following above chart
    
    :param logType: A string to indicate the type of usage of the log object.
    :type logType: String in all lower case. Default: 'main'
    
    :param debug: Flag to log debug level messages to file. 
                  NOTE: this is independent of logLevel for the screen msgs.
    :type debug: python boolean (True/False)
    
    :param noLogFile: Flag for stopping a log file from being created
    :type noLogFile: python boolean (True/False)
    
    :param allOff: Flag to turn off all messages to the screen or file and 
                   to not make a log file. ie, completely ignore all messages.
    :type allOff: python boolean (True/False)
    
    """
    logger = None
    def __init__(self, logName=None, logLevel=1, logType='main', debug=False, 
                 noLogFile=False, allOff=False):
        # Converting None -> 1(default) for logLevel, shouldn't happen though.
        if logLevel==None:
            logLevel = 1
        
        # Setting logLevel and noFile accordingly if allOff is turned on
        if allOff:
            logLevel=0
            noLogFile=True
            
        # Save verbosity setting for log to a private variable to allow for 
        # changing the value in future calls. 
        self._logLevel = logLevel
        
        # Save the type setting for the log to a private variable to allow for 
        # changing the value in future calls. First check it is a string, 
        # then force it to lower just in case.
        if isinstance(logType,str):
            # Force it to be a lower case string just in case
            self._logType = logType.lower()
        else:
            raise Error('The value of logType must be a string.')
        
        # storing debug boolean in private variable for later use
        self._debug = debug
        
        # If noLogFile=True, then set the log file as 'null'
        if noLogFile:
            self._logName = '/dev/null'
        # Set the file name for the log, default is currently 'gemini.log'
        else:
            # Setting the logName to the default or the value passed in, 
            # if there was one.
            if logName==None:
                self._logName = 'gemini.log'
            else:
                self._logName = logName
        
        # Adding logger levels not in default Python logger 
        # note: INFO level = 20
        self.FULLINFO    = 15
        self.STDINFO     = 21
        self.STATUS      = 25
        log_levels = {
                      self.FULLINFO  : 'FULLINFO',
                      self.STDINFO   : 'STDINFO',
                      self.STATUS    : 'STATUS'
                      }
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
                                         '%(levelno)d- %(message)s')
        err_formatter = logging.Formatter('%(levelname)-8s '+
                                          '---- %(message)s')
        
        # Add formatters to the handlers
        self.ch.setFormatter(ch_formatter)
        self.fh.setFormatter(fh_formatter)
        # Neet to revisit this, not formatting correctly
        #self.fh.setFormatter(ch_formatter)
        
        # Check if log has handlers and if so, close them to alleviate double 
        # messaging from multiple handers to same file or console
        #self = checkHandlers(self, remove=True)
        
        #add filter
        f1 = SingleLevelFilter(logging.ERROR, False)
        self.ch.addFilter(f1)

        # Add console and file handlers to the logger
        self.logger.addHandler(self.ch)
        self.logger.addHandler(self.fh)

    def setConsoleLevel(self, logLevel):
        """
        A function to set the level in the console handler.
        
        :param logLevel: verbosity setting for the lowest level of messages to 
                         print to the screen.
        :type logLevel: integer from 0-6, or 10 following above chart
        
        """
        if (logLevel == 10):
            self.ch.setLevel(logging.DEBUG)
        elif (logLevel == 6):
            # set to new FULLINFO value (15)
            self.ch.setLevel(self.FULLINFO)
        elif (logLevel == 5):
            # set to new STDINFO value (21)
            self.ch.setLevel(self.STDINFO)
        elif (logLevel == 4):
            # set to new STATUS value (25)
            self.ch.setLevel(self.STATUS)
        elif (logLevel == 3):
            self.ch.setLevel(logging.WARNING)
        elif (logLevel == 2):
            self.ch.setLevel(logging.ERROR)
        elif (logLevel == 1):
            self.ch.setLevel(logging.CRITICAL)
        elif (logLevel== 0):
            #ie. 'MAX' out the level so it is above all existing log levels
            # so no messages will ever have a high enough level to go to screen.
            self.ch.setLevel(100)
        else:
            # set to default, CRITICAL, if all else are false
            self.ch.setLevel(logging.CRITICAL)
    
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
        if debug==True:
            self._debug = True
        elif debug==False:
            self._debug = False
            
        # re-initializing the handlers 
        self.initializeHandlers()
        # setting console level of re-initialized console handler
        self.setConsoleLevel(logLevel)
        # re-finalizing the re-initialized handlers
        self.finalizeHandlers()
        #return the log object with its handlers updated with new logLevel 
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
                # Setting the private member _logType value to the one passed in.
                # Forcing the string to be in all lower case
                self._logType = newLogType.lower()
            else:
                raise Error('The value of newLogType must be a string.')
        # Return the current, or updated value if one was passed into newLogType
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
            # ditto
            if category is None:
                return self._errorDefaultCategory
            else:
                self._errorDefaultCategory = category
        elif level == 'warning':
            # ditto
            if category is None:
                return self._warningDefaultCategory
            else:
                self._warningDefaultCategory = category
        elif level == 'status':
            # ditto
            if category is None:
                return self._statusDefaultCategory
            else:
                self._statusDefaultCategory = category
        elif level == 'stdinfo':
            # ditto
            if category is None:
                return self._stdinfoDefaultCategory
            else:
                self._stdinfoDefaultCategory = category
        elif level == 'info':
            # ditto
            if category is None:
                return self._infoDefaultCategory
            else:
                self._infoDefaultCategory = category
        elif level == 'fullinfo':
            # ditto
            if category is None:
                return self._fullinfoDefaultCategory
            else:
                self._fullinfoDefaultCategory = category
        elif level == 'debug':
            # ditto
            if category is None:
                return self._debugDefaultCategory
            else:
                self._debugDefaultCategory = category
            
    def fullinfo(self, msg, category=None):
        """ The function to call for making 'fullinfo' level log messages"""
        # Check if a category string was passed in, else set to default value
        if category is None:
            category = self._fullinfoDefaultCategory
        # In case the message is multiple lines, break them up into a list
        msgs = str(msg).split('\n')
        # Loop through the message lines in the list and log them
        for line in msgs:
            # Log the message with the category appended to the beginning 
            #self.logger.fullinfo(category.ljust(10)+'-'+line)
            self.logger.fullinfo(line)
    
    # Comments for this function are the same as that of fullinfo function above
    def info(self, msg, category=None):
        """ The function to call for making 'info' level log messages"""
        if category is None:
            category = self._infoDefaultCategory
        msgs = str(msg).split('\n')
        for line in msgs:
            #self.logger.info(category.ljust(10)+'-'+line)
            self.logger.info(line)
     
     # Comments for this function are the same as that of fullinfo function above       
    def stdinfo(self, msg, category=None ):
        """ The function to call for making 'stdinfo' level log messages"""
        if category is None:
            category = self._stdinfoDefaultCategory
        msgs = str(msg).split('\n')
        for line in msgs:
            #self.logger.stdinfo(category.ljust(10)+'-'+line)
            self.logger.stdinfo(line)
    
    # Comments for this function are the same as that of fullinfo function above        
    def status(self, msg, category=None):
        """ The function to call for making 'status' level log messages"""
        if category is None:
            category = self._statusDefaultCategory
        msgs = str(msg).split('\n')
        for line in msgs:
            #self.logger.status(category.ljust(10)+'-'+line)
            self.logger.status(line)
    
    def debug(self, msg, category=None): 
        """ The function to call for making 'debug' level log messages"""
        # Check if a category string was passed in, else set to default value 
        if category is None:
            category = self._debugDefaultCategory  
        # Retrieving the important parts of the call stack using callInfo()
        b = callInfo()
        # In case the message is multiple lines, break them up into a list
        msgs = str(msg).split('\n')
        # Loop through the message lines in the list and log them
        for line in msgs:
            # Log the message with a modified format to match recipe system 
            # standards
            self.logger.debug(category.ljust(10)+'-'+b[0].ljust(20)+' - '+
                              b[2].ljust(20)+'-'+str(b[1]).ljust(3)+' - '+line)
    
    # Comments for this function are the same as that of debug function above
    def critical(self, msg, category=None):
        """ The function to call for making 'critical' level log messages"""
        if category is None:
            category = self._criticalDefaultCategory
        b = callInfo()
        msgs = str(msg).split('\n')
        for line in msgs:
            self.logger.critical(category.ljust(10).upper()+'-'+b[0].ljust(20)+
                                 ' - '+b[2].ljust(20)+'-'+str(b[1]).ljust(3)+
                                 ' - '+line)
            
    # Comments for this function are the same as that of debug function above    
    def warning(self, msg, category=None):
        """ The function to call for making 'warning' level log messages"""
        if category is None:
            category = self._warningDefaultCategory
        b = callInfo()
        msgs = str(msg).split('\n')
        for line in msgs:
            #self.logger.warning(category.ljust(10)+'-'+b[0].ljust(20)+' - '+
            #                    b[2].ljust(20)+'-'+str(b[1]).ljust(3)+' - '+
            #                    line)
            self.logger.warning('WARNING - '+line)
            
    # Comments for this function are the same as that of debug function above    
    def error(self, msg, category=None):
        """ The function to call for making 'error' level log messages"""
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
            sys.stderr.write('ERROR - ' + line)
            self.logger.error(category.ljust(10) + '-' + 
                b[0].ljust(20)+' - '+ b[2].ljust(20) + '-' + 
                str(b[1]).ljust(3) + ' - ' + line)
    
def callInfo():
    """ A function used by log levels debug, critical, warning and error 
        to allow for adding information about the call stack in the
        log messages. 
        
        ie. The module, function with in it and the line number of the log call
        
        The information is returned in the format of a list of strings following
        [module name, line number, function]
        
        """
    st = tb.extract_stack()
    return [os.path.basename(st[-3][0]), st[-3][1], st[-3][2]] 
    
def checkHandlers(log, remove=True):
    """
    This function is to close the handlers of the log to 
    avoid multiple handlers sending messages to same console and or file 
    when the logger is used outside of the Recipe System.
    
    :param remove: If handlers are found, remove them?
    :type remove: Python boolean (True/False)
    
    """
    handlers = log.logger.handlers
    # If there are handlers in handlers the loop through them, otherwise
    # return the log as it doesn't have any handlers to remove or False
    # to let you know the log had none
    if len(handlers) > 0:
        # If the handlers are to be removed, set using remove=True, then do so
        # if False, then just return True to indicate the log has handlers
        if remove:
            # Loop through the handlers list and close then remove them
            for i in range(0, len(handlers)):
                # After once through loop, length of handlers is reduced by 1
                # thus, must remove zeroth handler every time
                handler = handlers[0]
                try:
                    # First try to close the handler 
                    handler.close()
                except:
                    pass
                finally:
                    # Now remove the handler from the log
                    log.logger.removeHandler(handler)
            # Return the log with the handlers removed
            return log
        else:
            return True
    else:
        if remove:
            return log
        else:
            return False    

def createGeminiLog(logName=None, logLevel=None, logType='main', debug=False, 
                    noLogFile=False, allOff=False):
    """
    This function is to create a new logger object.
    
    :param logName: Name of the file the log messages will be written to
    :type logName: String
    
    :param logLevel: verbosity setting for the lowest level of messages to 
                     print to the screen.
    :type logLevel: integer from 0-6, or 10 following above chart, or the message
                    level as a string (ie. 'critical', 'status', 'fullinfo'...).
                    None indicates to use default of 1/'critical' OR to leave 
                    the current value in requested logger object if it exists.
    
    :param logType: A string to indicate the type of usage of the log object.
    :type logType: String in all lower case. Default: 'main'
    
    :param debug: Flag for showing debug level messages
    :type debug: Python boolean (True/False)
    
    :param noLogFile: Flag for stopping a log file from being created
    :type noLogFile: Python boolean (True/False)
    
    :param allOff: Flag to turn off all messages to the screen or file and 
                   to not make a log file. ie, completely ignore all messages.
    :type allOff: Python boolean (True/False)
    
    """
    # Retrieve the list of loggers
    global _listOfLoggers
    
    _geminiLogger = None
    
    # No logger list (ie, not even one log object) exists, so create one
    if not _listOfLoggers:
        _listOfLoggers = []
    #print '\n'+'-'*100+'\n'
    #print 'GL486: creating new logger' #$$$$$$$$$$$$$$$$$$$$$
    # Converting logLevel to its int value if needed
    logLevel = logLevelConverter(logLevel=logLevel)
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
    """ 
    The function called to retrieve the desired logger object based on the 
    parameter 'logType', and update its logLevel value if not None.   
    If the requested logger does not exist, then the 'main' logger is passed 
    back, and if that does not exist then a null logger is returned 
    (ie, no log file, no messages to screen).  
    Thus, a logger will always be returned when getGeminiLog is called.
        
    :param logLevel: verbosity setting for the lowest level of messages to 
                     print to the screen.
    :type logLevel: integer from 0-6, or 10 following above chart, or the message
                    level as a string (ie. 'critical', 'status', 'fullinfo'...).
                    None indicates to use default of 1/'critical' OR to leave 
                    the current value in requested logger object if it exists.
                    
    :param logType: A string to indicate the type of usage of the log object.
    :type logType: String in all lower case. Default: 'main'
    """
 
    # Retrieve the list of loggers
    global _listOfLoggers
    
    _geminiLogger = None
    
    #print '\nGL547: logger list: '+repr(_listOfLoggers)##########
    # No logger list (ie, not even one log object) exists, so create an 
    # alloff=True log to be passed back (ie, no log file, no msgs to screen)
    if not _listOfLoggers:
        _geminiLogger = createGeminiLog(allOff=True)
       
    # At least one logger object exists, so loop through current loggers in the 
    # list and see if the one you are requesting exists.            
    else:     
        # variable to hold the 'main' log to pass back if non-main one requested
        # doesn't exist in list.
        mainLog = None
        # Loop through logs in list
        #print "gemLog 629 _listOfLoggers = ", _listOfLoggers
        for log in _listOfLoggers:                
            # The log of the type requested is found in list
            if log.logtype()==logType:
                #print 'GL565: using old logger of type: '+log.logtype()
                
                # Updating the value of the logLevel if passed in value was 
                # not None or the same as the current logger's logLevel.
                if logLevel!=None:
                    # Converting logLevel to its int value if needed
                    logLevel = logLevelConverter(logLevel=logLevel)
                    
                    if logLevel!=log.loglevel():
                        #print 'G578: logs current logLevel: '+str(log.loglevel())
                        #print '*******TRACEBACK:'+repr(tb.extract_stack()[-2])
                        #print 'GL574: updating verbosity of existing log to '+str(logLevel)
                        
                        # updating the console level and reloading the handlers
                        _geminiLogger = log.changeLevels(logLevel=logLevel)
                        break
                    else:
                        # levels matched, so just pass current log back
                        _geminiLogger = log
                        break
                else:
                    # no level change requested, so just pass current log back
                    _geminiLogger = log
                    break
            # non-main requested, but main found, so store it for later 
            elif log.logtype()=='main':
                mainLog = log
                
        # The non-main log requested was not in the list, So pass back 'main' 
        # if it was found, else create an alloff=True log to be passed back 
        # (ie, no log file, no msgs to screen).
        if not _geminiLogger:
            if mainLog:
                _geminiLogger = mainLog
            else:
                #print 'requested type, '+logType+', and no main log was found for : '+repr(tb.extract_stack()[-2])
                _geminiLogger = createGeminiLog(allOff=True)
    
    #print 'GL597: logger being returned: '+_geminiLogger.logname()      
                
    # return the log that was requested, or the 'main' if specific one not there,
    # or a null one if even the 'main' wasn't there. 
    return _geminiLogger

def logLevelConverter(logLevel=None):
    """
    A basic function to map log message types (ie. error, warning, status...)
    to the integer logLevel equivalents to allow the user to pass in either 
    for the logLevel parameters of gemLog.
    Simply pass in the string for the message type and the matching integer 
    will be returned.
    
    :param logLevel: The logLevel value to be converted to an integer, if not 
                     one all ready.
    :type logLevel: integer from 0-6, or 10 following above chart, or the message
                    level as a string (ie. 'critical', 'status', 'fullinfo'...)
    """
    # Set up dictionary
    levelDict=  {'none':0,
                 'quiet':0,
                 'critical':1,
                 'error':2,
                 'warning':3,
                 'status':4,
                 'stdinfo':5,
                 'fullinfo':6,
                 'debug':10
                 }
    # Take care of both cases, logLevel is a string or int and perform checks
    # then return the appropriate int value.
    try:
        if isinstance(logLevel,str):
            if logLevel.isdigit():
                logLevel=int(logLevel)
                if (logLevel>=0 and logLevel<=6) or logLevel==10:
                    return logLevel
            else:
                return levelDict[logLevel]
        elif isinstance(logLevel, int):
            if (logLevel>=0 and logLevel<=6) or logLevel==10:
                return logLevel
    except:       
        raise Error('logLevel= '+str(logLevel)+' was not a valid input. Please'+
                ' enter a logLevel value that is either an integer between '+
                '0-6, 10 for debug, or one of the strings: none, quiet, '+
                'critical, error, warning, status, stdinfo, fullinfo or debug.')
    
    
    
