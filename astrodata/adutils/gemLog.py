    
import logging

def createGeminiLog(verbose = 0, debug = False)

    #create logger
    logger = logging.getLogger("gemini")
    logger.setLevel(logging.DEBUG)

    #create console and file handler 
    ch = logging.StreamHandler()
    fh = logging.FileHandler("gemini.log")
    
    #set levels according to flags
    fh.setLevel(logging.INFO)
    if debug:
        ch.setLevel(logging.DEBUG)
        fh.setLevel(logging.DEBUG)
    else:
        fh.setLevel(logging.INFO
        if (verbose == 3):
            ch.setLevel(logging.INFO)
        elif (verbose == 2):
            ch.setLevel(logging.WARNING)
        elif (verbose == 1):
            ch.setLevel(logging.ERROR)
        elif (verbose == 0):
            ch.setLevel(logging.CRITICAL)
        
    #create formatters
    ch_formatter = logging.Formatter("%(lineno)d -%(module)-10s - %(levelno)d - %(levelname)s - %(message)s")
    fh_formatter = logging.Formatter("%(lineno)d -%(module)-10s - %(levelno)d - %(levelname)s - %(message)s - %(name) - %(asctime)s")
    
    #add formatter to ch and fh
    ch.setFormatter(ch_formatter)
    fh.setFormatter(sh_formatter)
    #add ch and fh to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    #"application" code
    #logger.debug("debug message")
    #logger.info("info message")
    #logger.warn("warn message")
    #logger.error("error message")
    #logger.critical("critical message")

