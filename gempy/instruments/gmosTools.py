#Author: Kyle Mede, May 2010
#This module provides functions to be used by all GMOS primitives.

from astrodata.adutils import gemLog
from astrodata.AstroData import AstroData

def stdInstHdrs(ad, verbose=1):  
    """ A function used by StandardizeInstrumentHeaders in primitives_GMOS.
        
        It currently just adds the DISPAXIS header key to the SCI extensions.
    
    """
    # Adding the missing/needed keywords into the headers
    if not ad.isType('GMOS_IMAGE'):
    # Do the stuff to the headers that is for the MOS, those for IMAGE are 
    # taken care of with stdObsHdrs all ready 
    
        log=gemLog.getGeminiLog(verbose=verbose)
    
        # Formatting so logger looks organized for these messages
        log.fullinfo('****************************************************', 
                     'header') 
        log.fullinfo('file = '+ad.filename, 'header')
        log.fullinfo('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~', 
                     'header')
        for ext in ad['SCI']:
            ext.header.update(('SCI',ext.extver()),'DISPAXIS', \
                            ext.dispersion_axis() , 'Dispersion axis')
            # Updating logger with new header key values
            log.fullinfo('SCI extension number '+str(ext.header['EXTVER'])+
                         ' keywords updated/added:\n', 'header')       
            log.fullinfo('DISPAXIS = '+str(ext.header['DISPAXIS']), 'header' )
            log.fullinfo('---------------------------------------------------',
                         'header')

def valInstData(ad, verbose=1):  
    """A function used by validateInstrumentData in primitives_GMOS.
    
        It currently just checks if there are 1, 3, 6 or 12 SCI extensions 
        in the input. 
    
    """
    log=gemLog.getGeminiLog(verbose=verbose)
    length=ad.countExts('SCI')
    # If there are 1, 3, 6, or 12 extensions, all good, if not log a critical 
    # message and raise an exception
    if length==1 or length==3 or length==6 or length==12:
        pass
    else: 
        log.critical('There are NOT 1, 3, 6 or 12 extensions in file = '+
                     ad.filename)
        raise 'Error occurred in valInstData for input '+ad.filename
  
         
       