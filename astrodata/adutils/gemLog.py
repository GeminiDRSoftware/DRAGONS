#Author: Kyle Mede, 2010.    
import logging
import traceback as tb
import sys, os
_geminiLogger = None

class GeminiLogger(object):
    '''
    This is a logger object for use in the Gemini recipe system.  It is based on the Python logging object.
    '''
    logger = None
    def __init__(self,logName=None, verbose=1, debug=False):
        
        if not logName:
            self._logName="gemini.log"
        else:
            self._logName=logName
                
        # setting up additional logger levels
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
            logging.addLevelName(lvl,log_levels[lvl])
        
        # create logger
        self.logger = logging.getLogger(self._logName)
        self.logger.setLevel(logging.DEBUG)
    
        setattr(self.logger, 'stdinfo', lambda *args: self.logger.log(STDINFO, *args))
        setattr(self.logger, 'status', lambda *args: self.logger.log(STATUS, *args))
        setattr(self.logger, 'fullinfo', lambda *args: self.logger.log(FULLINFO, *args))

        # create console and file handler 
        ch = logging.StreamHandler()
        fh = logging.FileHandler(self._logName)
        
        # set levels according to flags
        fh.setLevel(FULLINFO)
        if debug:
            ch.setLevel(FULLINFO)
            fh.setLevel(logging.DEBUG)
            
        else:
            fh.setLevel(FULLINFO)
            if (verbose == 6):
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
            else:
                ch.setLevel(FULLINFO)
                fh.setLevel(FULLINFO)
        # create formatters
        ch_formatter = logging.Formatter("%(levelname)-8s %(levelno)d- %(message)s")
        fh_formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(levelno)d- %(message)s")
        
        # add formatter to ch and fh
        ch.setFormatter(ch_formatter)
        fh.setFormatter(fh_formatter) 
        
        # check if log has handlers and if so, close them
        self = checkHandlers(self,remove=True)
           
        # add console and file handlers to logger
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)
        
        # default category default strings by order of importance
        self._criticalDefaultCategory = 'critical'
        self._errorDefaultCategory = 'error'
        self._warningDefaultCategory = 'warning'
        self._statusDefaultCategory = 'status'
        self._stdinfoDefaultCategory = 'stdinfo'
        self._infoDefaultCategory = 'info'
        self._fullinfoDefaultCategory = 'fullinfo'
        self._debugDefaultCategory = 'debug'
        
    def logname(self):
        '''just a function to return the 'private' member variable _logName'''
        return self._logName
     
    def defaultCategory(self, level=None, category=None):
        '''
         A function to access and set the 'private' default category variables.
         If category = None, then the current value will be returned.
         Else, the default value will be replaced with the newly provided 
         category string.
         
         @param level: level to edit the default category for. eg. fullinfo, stdinfo, status...
         @type level: string
         
         @param level: new default value for the levels category
         @type level: string
        '''
        if level == 'critical':
            if category==None:
                return self._criticalDefaultCategory
            else:
                self._criticalDefaultCategory = category
        elif level == 'error':
            if category==None:
                return self._errorDefaultCategory
            else:
                self._errorDefaultCategory = category
        elif level == 'warning':
            if category==None:
                return self._warningDefaultCategory
            else:
                self._warningDefaultCategory = category
        elif level == 'status':
            if category==None:
                return self._statusDefaultCategory
            else:
                self._statusDefaultCategory = category
        elif level == 'stdinfo':
            if category==None:
                return self._stdinfoDefaultCategory
            else:
                self._stdinfoDefaultCategory = category
        elif level == 'info':
            if category==None:
                return self._infoDefaultCategory
            else:
                self._infoDefaultCategory = category
        elif level == 'fullinfo':
            if category==None:
                return self._fullinfoDefaultCategory
            else:
                self._fullinfoDefaultCategory = category
        elif level == 'debug':
            if category==None:
                return self._debugDefaultCategory
            else:
                self._debugDefaultCategory = category
        
    def debug(self,msg,category=None):  
        if category==None:
            category = self._debugDefaultCategory  
        b=callInfo()
        msgs = str(msg).split('\n')
        for line in msgs:
            self.logger.debug(category.ljust(10)+"-"+b[0].ljust(20)+" - "+b[2].ljust(20)+"-"+str(b[1]).ljust(3)+" - "+line)
            
    def fullinfo(self,msg,category=None):
        if category==None:
            category = self._fullinfoDefaultCategory
        msgs = str(msg).split('\n')
        for line in msgs:
            self.logger.fullinfo(category.ljust(10)+'-'+line)
    
    def info(self,msg,category=None):
        if category==None:
            category = self._infoDefaultCategory
        msgs = str(msg).split('\n')
        for line in msgs:
            self.logger.info(category.ljust(10)+'-'+line)
            
    def stdinfo(self,msg,category =None ):
        if category==None:
            category = self._stdinfoDefaultCategory
        msgs = str(msg).split('\n')
        for line in msgs:
            self.logger.stdinfo(category.ljust(10)+'-'+line)
            
    def status(self,msg,category = None):
        if category==None:
            category = self._statusDefaultCategory
        msgs = str(msg).split('\n')
        for line in msgs:
            self.logger.status(category.ljust(10)+'-'+line)
        
    def critical(self,msg,category=None):
        if category==None:
            category = self._criticalDefaultCategory
        b=callInfo()
        msgs = str(msg).split('\n')
        for line in msgs:
            self.logger.critical(category.ljust(10)+"-"+b[0].ljust(20)+" - "+b[2].ljust(20)+"-"+str(b[1]).ljust(3)+" - "+line)
        
    def warning(self,msg,category=None):
        if category==None:
            category = self._warningDefaultCategory
        b=callInfo()
        msgs = str(msg).split('\n')
        for line in msgs:
            self.logger.warning(category.ljust(10)+"-"+b[0].ljust(20)+" - "+b[2].ljust(20)+"-"+str(b[1]).ljust(3)+" - "+line)
        
    def error(self,msg,category=None):
        if category==None:
            category = self._errorDefaultCategory
        b=callInfo()
        msgs = str(msg).split('\n')
        for line in msgs:
            self.logger.error(category.ljust(10)+"-"+b[0].ljust(20)+" - "+b[2].ljust(20)+"-"+str(b[1]).ljust(3)+" - "+line)
    
def getGeminiLog(logName=None ,verbose = 0, debug = False):
    global _geminiLogger
    
    # no logger exists, so create one
    if not _geminiLogger:
        _geminiLogger=GeminiLogger(logName, verbose, debug)
        return _geminiLogger
    
    # you want a non-default logger, but there is a different one already
    # , so create new non-default logger
    elif _geminiLogger and (_geminiLogger.logname()!=logName) and (logName!=None):
        _geminiLogger=GeminiLogger(logName, verbose, debug)
        return _geminiLogger
    
    # a non-default logger exists, but you want a default one, so create it
    elif _geminiLogger and (logName!=None):
        _geminiLogger=GeminiLogger(logName, verbose, debug)
        return _geminiLogger
    
    else:
        return _geminiLogger
    
def checkHandlers(log, remove=True ):
    '''
    This function is to close the handlers of the log to 
    avoid multiple handlers sending messages to same console and or file 
    when the logger is used outside of the Recipe System.
    '''
    handlers = log.logger.handlers
    if len( handlers ) > 0:
        if remove:
            #for handler in handlers:
            for i in range(0,len(handlers)):
                handler=handlers[0]
                try:
                    handler.close()
                except:
                    pass
                finally:
                    log.logger.removeHandler( handler )
            return log
        else:
            return True
    else:
        if remove:
            return log
        else:
            return False    

def callInfo():
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
    #print 'callInfo using ('+os.path.basename(st[-3][0])+","+str(st[-3][1])+","+st[-3][2]+")"
    return [os.path.basename(st[-3][0]),st[-3][1],st[-3][2]]

    