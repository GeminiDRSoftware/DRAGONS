    
import logging
import inspect #$$$$$$$$$$$$$$$$$$
import traceback as tb
import sys, os
_geminiLogger = None

class GeminiLogger(object):
    logger = None
    def __init__(self,logName=None, verbose=1, debug=False):
        if not logName:
            logName="gemini.log"
            
        # setting up additional logger levels
        FULLINFO = 15
        STDINFO = 21
        STATUS = 25

        log_levels = {
        FULLINFO : 'FULLINFO',
        STDINFO : 'STDINFO',
        STATUS : 'STATUS'
        }
        for lvl in log_levels.keys():
            logging.addLevelName(lvl,log_levels[lvl])
        
        #create logger
        self.logger = logging.getLogger(logName)
        self.logger.setLevel(logging.DEBUG)
    
        setattr(self.logger, 'stdinfo', lambda *args: self.logger.log(STDINFO, *args))
        setattr(self.logger, 'status', lambda *args: self.logger.log(STATUS, *args))
        setattr(self.logger, 'fullinfo', lambda *args: self.logger.log(FULLINFO, *args))

        #create console and file handler 
        ch = logging.StreamHandler()
        fh = logging.FileHandler(logName)
        
        #set levels according to flags
        fh.setLevel(FULLINFO)
        if debug:
            #ch.setLevel(logging.DEBUG)
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
        #create formatters
        ch_formatter = logging.Formatter("%(levelname)-8s %(levelno)d- %(message)s")
        fh_formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(levelno)d- %(message)s")
        
        #add formatter to ch and fh
        ch.setFormatter(ch_formatter)
        fh.setFormatter(fh_formatter) 
        
        #add ch and fh to logger
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)
    
    def debug(self,msg,cat='DefautCat'):    
        #a={'category':cat}
        b=callInfo()
        msgs = str(msg).split('\n')
        for line in msgs:
            self.logger.debug(cat.ljust(10)+"-"+b[0].ljust(20)+" - "+b[2].ljust(20)+"-"+str(b[1]).ljust(3)+" - "+line)
            
    def fullinfo(self,msg,cat ='DefautCat'):
        msgs = str(msg).split('\n')
        for line in msgs:
            self.logger.fullinfo(cat.ljust(10)+'-'+line)
    
    def info(self,msg,cat='DefautCat'):
       #a={'category':cat}
        msgs = str(msg).split('\n')
        for line in msgs:
            self.logger.info(cat.ljust(10)+'-'+line)
            #self.logger.info(line,extra=a)
            
    def stdinfo(self,msg,cat = 'DefautCat'):
        msgs = str(msg).split('\n')
        for line in msgs:
            self.logger.stdinfo(cat.ljust(10)+'-'+line)
            
    def status(self,msg,cat = 'DefautCat'):
        msgs = str(msg).split('\n')
        for line in msgs:
            self.logger.status(cat.ljust(10)+'-'+line)
        
    def critical(self,msg,cat='DefautCat'):
        #a={'category': cat}
        b=callInfo()
        msgs = str(msg).split('\n')
        for line in msgs:
            self.logger.critical(cat.ljust(10)+"-"+b[0].ljust(20)+" - "+b[2].ljust(20)+"-"+str(b[1]).ljust(3)+" - "+line)
            #self.logger.critical(line,extra=a)
        
    def warning(self,msg,cat='DefautCat'):
        #a={'category':cat}
        b=callInfo()
        msgs = str(msg).split('\n')
        for line in msgs:
            self.logger.warning(cat.ljust(10)+"-"+b[0].ljust(20)+" - "+b[2].ljust(20)+"-"+str(b[1]).ljust(3)+" - "+line)
            #self.logger.warning(line,extra=a)
        
    def error(self,msg,cat='DefautCat'):
        #a={'category': cat}
        b=callInfo()
        msgs = str(msg).split('\n')
        for line in msgs:
            self.logger.error(cat.ljust(10)+"-"+b[0].ljust(20)+" - "+b[2].ljust(20)+"-"+str(b[1]).ljust(3)+" - "+line)
            #self.logger.error(line,extra=a)
    
def getGeminiLog(logName=None ,verbose = 0, debug = False):
    global _geminiLogger
    
    if not _geminiLogger:
        _geminiLogger=GeminiLogger(logName, verbose, debug)
    return _geminiLogger
        
def checkHandlers(log, remove=True ):
    '''
    this function is to close the handlers of the log to avoid an error when used in pyraf
    $$$$$$$ THIS FUNCTION IS CURRENTLY NOT BEING USED, IT WILL BE INCORPERATED WHEN WE START USING PYRAF $$$$$$$$$$
    '''
    handlers = log.logger.handlers
    if len( handlers ) > 0:
        if remove:
            for handler in handlers:
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

    




