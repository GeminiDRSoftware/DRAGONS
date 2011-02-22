#Author: Kyle Mede, 2010.    
import sys, os

import logging
import traceback as tb

_listOfLoggers = None

class GeminiLogger(object):
    """
    This is a logger object for use in the Gemini recipe system.  
    It is based on the Python logging object.
    
    Logging levels chart:
level#, verbose,  level,          includes (current rough outline)

 10       10      debug           engineering, programmer debugging
 15       6       fullinfo        details, input parameters, header changes
 20               info
 21       5       stdinfo         science info eg. background level
 25       4       status          start processing/end, # of files, 
                                  name of inputs/output files
 30       3       warning    
 40       2       error    
 50       1       critical
          0       (quiet)
    note: level 'info' exists, but it is not being used in the
    Recipe Systems logging standards
    
    @param logName: Name of the file the log messages will be written to
    @type logName: string
    
    @param verbose: verbosity setting for the lowest level of messages to 
                    print to the screen.
    @type verbose: integer from 0-10 following above chart
    
    @param debug: Flag for showing debug level messages
    @type debug: python boolean (True/False)
    
    @param noLogFile: Flag for stopping a log file from being created
    @type noLogFile: python boolean (True/False)
    
    @param allOff: Flag to turn off all messages to the screen or file
    @type allOff: python boolean (True/False)
    
    """
    logger = None
    def __init__(self, logName=None, verbose=1, debug=False, noLogFile=False, allOff=False):
        # Setting verbose and noFile accordingly if allOff is turned on
        if allOff:
            verbose=0
            noLogFile=True
        
        # Save verbosity setting for log to a private variable to allow for 
        # changing the value in future calls. 
        self._verbose=verbose
        
        # If noLogFile=True, then set the log file as 'null'
        if noLogFile:
            self._logName = '/dev/null'
        # Set the file name for the log, default is currently 'gemini.log'
        else:
            # Setting the logName to the default or the value passed in, 
            # if there was one
            if not logName:
                self._logName = 'gemini.log'
            else:
                self._logName = logName
            
        # Adding logger levels not in default Python logger 
        # note: INFO level = 20
        FULLINFO    = 15
        STDINFO     = 21
        STATUS      = 25
        log_levels = {
                      FULLINFO  : 'FULLINFO',
                      STDINFO   : 'STDINFO',
                      STATUS    : 'STATUS'
                      }
        for lvl in log_levels.keys():
            logging.addLevelName(lvl, log_levels[lvl])
        
        # Create logger object and its default level
        self.logger = logging.getLogger(self._logName)
        self.logger.setLevel(logging.DEBUG)
    
        setattr(self.logger, 'stdinfo', 
                lambda *args: self.logger.log(STDINFO, *args))
        setattr(self.logger, 'status', 
                lambda *args: self.logger.log(STATUS, *args))
        setattr(self.logger, 'fullinfo', 
                lambda *args: self.logger.log(FULLINFO, *args))

        # Create console and file handlers
        ch = logging.StreamHandler()
        fh = logging.FileHandler(self._logName)
        
        # Set levels for handlers 
        # Use file handler level FULLINFO unless debug is True
        fh.setLevel(FULLINFO)
        if debug:
            ch.setLevel(FULLINFO)
            fh.setLevel(logging.DEBUG)
            
        # Set console handler depending on value provided on verbose value
        else:
            if (verbose == 10):
                ch.setLevel(logging.DEBUG)
            elif (verbose == 6):
                ch.setLevel(FULLINFO)
            elif (verbose == 5):
                ch.setLevel(STDINFO)
            elif (verbose == 4):
                ch.setLevel(STATUS)
            elif (verbose == 3):
                ch.setLevel(logging.WARNING)
            elif (verbose == 2):
                ch.setLevel(logging.ERROR)
            elif (verbose == 1):
                ch.setLevel(logging.CRITICAL)
            elif (verbose==0):
                #ie. 'MAX' out the level so it is above all existing log levels
                ch.setLevel(100)
            else:
                ch.setLevel(FULLINFO)
                fh.setLevel(FULLINFO)
                
        # Create formatters for console and file messages
        ch_formatter = logging.Formatter('%(levelname)-8s '+
                                        '%(levelno)d- %(message)s')
        fh_formatter = logging.Formatter('%(asctime)s %(levelname)-8s '+
                                         '%(levelno)d- %(message)s')
        
        # Add formatters to the handlers
        ch.setFormatter(ch_formatter)
        fh.setFormatter(fh_formatter) 
        
        # Check if log has handlers and if so, close them to alleviate double 
        # messaging  from multiple handers to same file or console
        self = checkHandlers(self, remove=True)
           
        # Add console and file handlers to logger
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)
        
        # Default category strings by order of importance
        self._criticalDefaultCategory = 'critical'
        self._errorDefaultCategory = 'error'
        self._warningDefaultCategory = 'warning'
        self._statusDefaultCategory = 'status'
        self._stdinfoDefaultCategory = 'stdinfo'
        self._infoDefaultCategory = 'info'
        self._fullinfoDefaultCategory = 'fullinfo'
        self._debugDefaultCategory = 'debug'
        
    def logname(self):
        """Just a function to return the 'private' member variable _logName
        to allow checking if logger with the same file name exists all ready
        
        """
        return self._logName
    
    def verbosity(self):
        """Just a function to return the 'private' member variable _verbose
        to allow checking what the current verbosity of this logger object is.        
        """     
        return self._verbose
    
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
         
         @param level: level to edit the default category for. 
                       eg. fullinfo, stdinfo, status...
         @type level: string
         
         @param level: new default value for the levels category
         @type level: string
         
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
            self.logger.fullinfo(category.ljust(10)+'-'+line)
    
    # Comments for this function are the same as that of fullinfo function above
    def info(self, msg, category=None):
        """ The function to call for making 'info' level log messages"""
        if category is None:
            category = self._infoDefaultCategory
        msgs = str(msg).split('\n')
        for line in msgs:
            self.logger.info(category.ljust(10)+'-'+line)
     
     # Comments for this function are the same as that of fullinfo function above       
    def stdinfo(self, msg, category=None ):
        """ The function to call for making 'stdinfo' level log messages"""
        if category is None:
            category = self._stdinfoDefaultCategory
        msgs = str(msg).split('\n')
        for line in msgs:
            self.logger.stdinfo(category.ljust(10)+'-'+line)
    
    # Comments for this function are the same as that of fullinfo function above        
    def status(self, msg, category=None):
        """ The function to call for making 'status' level log messages"""
        if category is None:
            category = self._statusDefaultCategory
        msgs = str(msg).split('\n')
        for line in msgs:
            self.logger.status(category.ljust(10)+'-'+line)
    
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
            self.logger.critical(category.ljust(10)+'-'+b[0].ljust(20)+' - '+
                                 b[2].ljust(20)+'-'+str(b[1]).ljust(3)+' - '+
                                 line)
            
    # Comments for this function are the same as that of debug function above    
    def warning(self, msg, category=None):
        """ The function to call for making 'warning' level log messages"""
        if category is None:
            category = self._warningDefaultCategory
        b = callInfo()
        msgs = str(msg).split('\n')
        for line in msgs:
            self.logger.warning(category.ljust(10)+'-'+b[0].ljust(20)+' - '+
                                b[2].ljust(20)+'-'+str(b[1]).ljust(3)+' - '+
                                line)
            
    # Comments for this function are the same as that of debug function above    
    def error(self, msg, category=None):
        """ The function to call for making 'error' level log messages"""
        if category is None:
            category = self._errorDefaultCategory
        b = callInfo()
        msgs = str(msg).split('\n')
        for line in msgs:
            self.logger.error(category.ljust(10)+'-'+b[0].ljust(20)+' - '+
                              b[2].ljust(20)+'-'+str(b[1]).ljust(3)+' - '+line)
    
def callInfo():
    """ A function used by log levels debug, critical, warning and error 
        to allow for adding information about the call stack in the
        log messages. 
        
        ie. The module, function with in it and the line number of the log call
        
        The information is returned in the format of a list of strings following
        [module name, line number, function]
        
        """
    st = tb.extract_stack()
    #ran = range(len(st))
    #ran.reverse()
    #for i in ran:
    #    print i
    #    filenam=os.path.basename(st[i][0])
    #    print filenam
    #    linenum=st[i][1]
    #    print linenum
    #    funcnam=st[i][2]
    #    print funcnam
    #    print '------------------'
    #print 'callInfo using ('+os.path.basename(st[-3][0])+','+str(st[-3][1])+','+st[-3][2]+')'
    return [os.path.basename(st[-3][0]), st[-3][1], st[-3][2]] 
    
def checkHandlers(log, remove=True):
    """
    This function is to close the handlers of the log to 
    avoid multiple handlers sending messages to same console and or file 
    when the logger is used outside of the Recipe System.
    
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

def getGeminiLog(logName=None , verbose=1, debug=False, noLogFile=False, allOff=False):
    """ The function called to retrieve the desired logger object.
        This can be a new one, and thus getGeminiLog will create one to be 
        returned, else it will return the requested one based on the 
        parameter 'logName' in the call.
        
        """
    # Retrieve the list of loggers
    global _listOfLoggers
    
    _geminiLogger = None
    
    if verbose==None:
        verbose=1
        
    # No logger list (ie, not even one log object) exists, so create one
    if not _listOfLoggers:
        #print 'GL415: creating new logger' #$$$$$$$$$$$$$$$$$$$$$
        _listOfLoggers = []
        _geminiLogger = GeminiLogger(logName=logName, verbose=verbose, 
                                     debug=debug, noLogFile=noLogFile, 
                                     allOff=allOff)
        _listOfLoggers.append(_geminiLogger)
    # At least one logger object exists, so loop through current loggers in the 
    # in the list and see if the one you are requesting exists.
    else:
        # Since logName=None means use default 'gemini.log', just set it to that 
        # now to make searching the logger list easier.
        if logName==None:
            logName='gemini.log'
        # Loop through logs in list
        for log in _listOfLoggers:
            # The log you requested is found in list
            if log.logname() == logName:
                #print 'GL432: using old logger'
                if verbose!=log.verbosity():
                    #print 'GL445: updating verbosity of existing log to '+str(verbose)
                    _geminiLogger = GeminiLogger(logName=logName, 
                                         verbose=verbose, debug=debug, 
                                         noLogFile=noLogFile, allOff=allOff)
                    log = _geminiLogger
                else:
                    _geminiLogger=log
        # The log you requested is not there, so create it and add it to list. 
        if not _geminiLogger:
            #print 'GL436: creating new logger but list was there'
            _geminiLogger = GeminiLogger(logName=logName, verbose=verbose, 
                                         debug=debug, noLogFile=noLogFile, 
                                         allOff=allOff)
            _listOfLoggers.append(_geminiLogger)
                
    # return the log that was requested, whether it had to be created or was all 
    # ready there.             
    return _geminiLogger

    