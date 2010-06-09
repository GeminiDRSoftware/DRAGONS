import logging
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
GEMINI_FORMAT = logging.Formatter( "%(asctime)s %(name)-20s %(levelname)-8s %(message)s" )  
logName = None

def defaultInit( name="", logfile="", verbose=False, errorlogfile="", debug=True ):
    '''
    The default Gemini initialization for logging.
    
    @param name: Name for the root of the logger. By default, this will end up being something like 'root'.
    @type name: string
    
    @param logfile: URI of logfile
    @type logfile: string
    
    @param verbose: Whether or not to print to stream (Basically whether to print to console).
    @type verbose: boolean
    
    @param errorlogfile: URI of error logfile
    @type errorlogfile: string
    
    @param debug: If true, debug information will be printed to log and console if applicable.
    @type debug: boolean
    
    @return: A logger to start logging to!
    @rtype: logging.log
    '''
    global logName
    global logVerbose
    
    logName = name
    logVerbose = verbose
    
    if debug:
        minlevel = logging.DEBUG
    else:
        minlevel = logging.INFO
        
    gemLog = logging.getLogger( name )
    gemLog.setLevel( minlevel )
    gemLog = checkHandlers( gemLog )
    # Check if logFile acceptable ...etc
    if logfile == "":
        logfile = "gemini.log"
    fh = logging.FileHandler( logfile, "a" )
    fh.setFormatter( GEMINI_FORMAT )
    fh.setLevel( minlevel )
    gemLog.addHandler( fh )
    
    if errorlogfile == "":
        errorlogfile = "geminiError.log"
    eh = logging.FileHandler( errorlogfile, "a" )
    eh.setFormatter( GEMINI_FORMAT )
    eh.setLevel( logging.WARNING )
    gemLog.addHandler( eh )
    
    
    if verbose:
        ch = logging.StreamHandler()
        ch.setFormatter( GEMINI_FORMAT )
        ch.setLevel( minlevel )
        gemLog.addHandler( ch )
    else:
        ec = logging.StreamHandler()
        ec.setFormatter( GEMINI_FORMAT )
        ec.setLevel( logging.WARNING )
        gemLog.addHandler( ec )
    
    return gemLog

def getLogger( name="", logfile="", verbose=False, debug=True, errorlogfile="" ):
    '''
    This should be called when you want a gemini logger. For the most part, it should be called with a gemini
    logger already created (i.e. defaultInit was called.) Although, this is built to create a new log if
    defaultInit has not been created yet.
    - Get the main, already created logger.
    
    @param name: Name of the logger. (Not variable name, name that will be printed)
    @type name: string
    
    @param logfile: URI of logfile
    @type logfile: string
    
    @param verbose: Whether or not to print to stream (Basically whether to print to console).
    @type verbose: boolean
    
    @param debug: If true, debug information will be printed to log and console if applicable.
    @type debug: boolean
    
    @param errorlogfile: URI of error logfile
    @type errorlogfile: string
    
    @return: A logger to start logging to!
    @rtype: logging.log
    '''
    global logName
    global logVerbose
    
    if logName == None:
        return defaultInit( name, logfile, verbose, debug=debug, errorlogfile=errorlogfile )
    
    if name == "" or name == logName:
        getLog = logging.getLogger( logName )
    else:
        print 'gemlogger106: logName = ', logName
        getLog = logging.getLogger( logName + "." + name )
    
    minlevel = logging.INFO
    if debug:
        minlevel = logging.DEBUG
       
    if not checkHandlers( getLog, remove=False ):  
        if logVerbose:
            verbose = False
            # this could theoretically only have 'pass' in it
        elif verbose:
            ch = logging.StreamHandler()
            ch.setFormatter( GEMINI_FORMAT )
            ch.setLevel( minlevel )
            getLog.addHandler( ch )
        
        if logfile != "":
            fh = logging.FileHandler( logfile )
            fh.setFormatter( GEMINI_FORMAT )
            fh.setLevel( minlevel )
            getLog.addHandler( fh )
    
    return getLog


def checkHandlers( log, remove=True ):
    '''
    
    '''
    handlers = log.handlers
    if len( handlers ) > 0:
        if remove:
            for handler in handlers:
                try:
                    handler.close()
                except:
                    pass
                finally:
                    log.removeHandler( handler )
            return log
        else:
            return True
    else:
        if remove:
            return log
        else:
            return False
    
    
    
    