import logging
import types

STDFMT = '%(asctime)s %(levelname)-8s - %(message)s' 
DBGFMT = '%(asctime)s %(name)-40s - %(levelname)-8s - %(message)s'
SW = 3

# Turn off logging exception messages 
logging.raiseExceptions = 0

# Logging levels
ll = {'CRITICAL':50, 'ERROR':40, 'WARNING':30, 'STATUS':25, 'STDINFO':21,
      'INFO':20, 'FULLINFO':15, 'DEBUG':10}

def customize_log(log=None):
    '''
    :param log: A fresh log from logging.getLogger()
    :type log: logging.Logger

    Sets up custom attributes for logger
    '''
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

    #return log

def get_logger(name=None):
    '''
    :param name: Name of logger (usually __name__)
    :type name: string

    :returns: Logger with new levels and prefixes for some levels
    :rtype: logging.Logger

    Wraps logging.getLogger and returns a custum logging object
    '''
    log = logging.getLogger(name)
    customize_log(log)
    return log
        
def config(mode='standard', console_lvl=None, file_lvl=None, \
           file_name='ad.log', stomp=False):
    '''
    :param mode: logging mode
    :type mode: string

    :param console_lvl: conole logging level
    :type console_lvl: string

    :param file_lvl: file logging level
    :type file_lvl: string

    :param file_name: filename of the logger
    :type file_name: string

    :param stomp: Controls append to logfiles found with same name
    :type stomp: boolean
    
    Controls the recipe system (rs) logging configuration using python logging.
    '''
    mode = mode.lower()
    rootlog = logging.getLogger('')
    
    # Set the console and file logging levels
    if mode == 'standard':
        if console_lvl:
            console_lvl_asint = ll[str(console_lvl).upper()]
        else:   
            console_lvl_asint = ll['STDINFO']
        if file_lvl:
            file_lvl_asint = ll[str(file_lvl).upper()]
        else:
            file_lvl_asint = ll['FULLINFO']
    elif mode == 'debug':
        if console_lvl:
            console_lvl_asint = ll[str(console_lvl).upper()]
        else:   
            console_lvl_asint = ll['DEBUG']
        if file_lvl:
            file_lvl_asint = ll[str(file_lvl).upper()]
        else:
            file_lvl_asint = ll['DEBUG']
        
    # If rootlog has handlers, flush and delete them
    if len(rootlog.handlers) > 0:
        for hand in rootlog.handlers:
            if isinstance(hand, logging.StreamHandler):
                hand.flush()
            rootlog.removeHandler(hand)

    # Add the new levels
    logging.addLevelName(ll['STATUS'], 'STATUS')
    logging.addLevelName(ll['STDINFO'], 'STDINFO')
    logging.addLevelName(ll['FULLINFO'], 'FULLINFO')

    # Assign new attributes to root logger
    #setattr(rootlog, 'stdinfo', 
    #        lambda *args: rootlog.log(ll['STDINFO'], *args))
    #setattr(rootlog, 'status', 
    #        lambda *args: rootlog.log(ll['STATUS'], *args))
    #setattr(rootlog, 'fullinfo', 
    #        lambda *args: rootlog.log(ll['FULLINFO'], *args))

    # Define rootlog handler(s) through basicConfig() according to mode
    customize_log(rootlog)
    
    
    if mode == 'null':
        pass
    elif mode == 'console':
        logging.basicConfig(level=console_lvl_asint, format='%(message)s')        
    else:
        if mode == 'standard':
            fmt = STDFMT
        elif mode == 'debug':
            fmt = DBGFMT
        else:
            raise NameError('unknown mode')
        if stomp:
            fm = 'w'
        else:
            fm = 'a'
        logging.basicConfig(level=file_lvl_asint, format=fmt,  datefmt= \
            '%Y-%m-%d %H:%M:%S', filename=file_name, filemode=fm)

        # add console handler for rootlog through addHandler()
        console = logging.StreamHandler()
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        console.setLevel(console_lvl_asint)
        rootlog.addHandler(console)

def update_indent(li=0, mode=''):
    '''
    :param li: log indent
    :type li: integer

    :param mode: logging mode
    :type mode: string

    Updates indents for reduce by changing the formatter
    '''
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

def change_level(new_level=''):
    '''
    Change the level of the console handler
    '''
    log = logging.getLogger('')
    for hndl in log.handlers:
        if isinstance(hndl, logging.StreamHandler):
            if new_level:
                hndl.setLevel(ll[new_level.upper()])
