    
import logging
_geminiLogger = None

class GeminiLogger(object):
    logger = None
    def __init__(self,logName=None, verbose=1, debug=False):
        if not logName:
            logName="gemini.log"
            
        #create logger
        self.logger = logging.getLogger(logName)
        self.logger.setLevel(logging.DEBUG)
    
        #create console and file handler 
        ch = logging.StreamHandler()
        fh = logging.FileHandler(logName)
        
        #set levels according to flags
        fh.setLevel(logging.INFO)
        if debug:
            ch.setLevel(logging.DEBUG)
            fh.setLevel(logging.DEBUG)
        else:
            fh.setLevel(logging.INFO)
            if (verbose == 4):
                ch.setLevel(logging.INFO)
            elif (verbose == 3):
                ch.setLevel(logging.WARNING)
            elif (verbose == 2):
                ch.setLevel(logging.ERROR)
            elif (verbose == 1):
                ch.setLevel(logging.CRITICAL)
            else:
                ch.setLevel(logging.DEBUG)
                fh.setLevel(logging.DEBUG)
        #create formatters
        ch_formatter = logging.Formatter("%(category)-6s  %(levelno)d  %(levelname)s  %(message)s")
        fh_formatter = logging.Formatter("%(asctime)s %(category)-6s  %(levelno)d  %(levelname)s  %(message)s")
        
        #add formatter to ch and fh
        ch.setFormatter(ch_formatter)
        fh.setFormatter(fh_formatter)
        
        #add ch and fh to logger
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)
    
    def info(self,msg,cat='DEFAULT'):
        a={'category':cat}
        self.logger.info(msg, extra=a)
        
    def debug(self,msg,cat='DEFAULT'):     
        a={'category':cat}
        self.logger.debug(msg, extra=a)
        
    def critical(self,msg,cat='DEFAULT'):
        a={'category': cat}
        self.logger.critical(msg,extra=a)
        
    def warning(self,msg,cat='DEFAULT'):
        a={'category':cat}
        self.logger.warning(msg,extra=a)
        
    def error(self,msg,cat='DEFAULT'):
        a={'category': cat}
        self.logger.error(msg,extra=a)
        
def getGeminiLog(logName=None ,verbose = 0, debug = False):
    global _geminiLogger
    
    if not _geminiLogger:
        _geminiLogger=GeminiLogger(logName, verbose, debug)
    return _geminiLogger
        
    

    #"application" code
    #logger.debug("debug message")
    #logger.info("info message")
    #logger.warn("warn message")
    #logger.error("error message")
    #logger.critical("critical message")

