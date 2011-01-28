
from astrodata.adutils import gemLog
from astrodata.AstroData import AstroData
from gempy.instruments import geminiTools  as gemt

def ADUtoElectrons(adIns=None, outNames=None, postpend=None, logName='', verbose=1, noLogFile=False):
    """
    adIns must either be a single astrodata object or a 
    list of astrodata objects.
    
    adOuts must be a single string or a list of strings
    of equal length to the number of inputs (ie. adIns).
    Else, the 'postpend' value may be set to any string,
    which will then be postpended onto all the input filenames 
    for the outputs.  
    ie. Either 'adOuts' or 'postpend' must be set.
    
    A string representing the name of the log file to write all log messages to
    can be defined, or a default of 'gemini.log' will be used.  If the file
    all ready exists in the directory you are working in, then this file will 
    have the log messages during this function added to the end of it.
    
    @param adIns: Astrodata inputs to be converted to Electron pixel units
    @type adIns: Astrodata objects, either a single or a list of objects
    
    @param outNames: filenames of output(s)
    @type outNames: String, either a single or a list of strings
    
    @param postpend: string to postpend on the end of the input filenames
                    for the output filenames.
    @type postpend: string
    
    @param logName: Name of the log file, default is 'gemini.log'
    @type logName: string
    
    @param verbose: verbosity setting for the log messages to screen,
                    default is 'critical' messages only.
    @type verbose: integer from 0-6, 0=nothing to screen, 6=everything to screen
    
    @param noLogFile: A boolean to make it so no log file is created
    @type noLogFile: Python boolean (True/False)
    """
    
    if logName!='':
        log=gemLog.getGeminiLog(logName=logName, verbose=verbose, noLogFile=noLogFile)
    else:
        log=gemLog.getGeminiLog(verbose=verbose, noLogFile=noLogFile)
        
    log.status('**STARTING** the ADUtoElectrons function')
    
    if (adIns!=None) and (outNames!=None):
        if len(adIns)!= len(outNames):
            if postpend==None:
                   raise ('Then length of the inputs, '+str(len(adIns))+
                       ', did not match the length of the outputs, '+str(len(outNames))+
                       ' AND no value of "postpend" was passed in')
    
    try:
        count=0
        if adIns!=None:
            # Creating empty list of ad's to be returned that will be filled below
            if len(adIns)>1:
                adOuts=[]
            
            # Do the work on each ad in the inputs
            for ad in adIns:
                log.fullinfo('calling ad.mult on '+ad.filename)
                
                # mult in this primitive will multiply the SCI frames by the
                # frame's gain, VAR frames by gain^2 (if they exist) and leave
                # the DQ frames alone (if they exist).
                log.debug('Calling ad.mult to convert pixel units from '+
                          'ADU to electrons')

                adOut = ad.mult(ad['SCI'].gain(asDict=True))  
                
                log.status('ad.mult completed converting the pixel units'+
                           ' to electrons')              
                
                # Updating SCI headers
                for ext in adOut['SCI']:
                    # Retrieving this SCI extension's gain
                    gainorig = ext.gain()
                    # Updating this SCI extension's header keys
                    ext.header.update('GAINORIG', gainorig, 
                                       'Gain prior to unit conversion (e-/ADU)')
                    ext.header.update('GAIN', 1.0, 'Physical units is electrons') 
                    ext.header.update('BUNIT','electrons' , 'Physical units')
                    # Logging the changes to the header keys
                    log.fullinfo('SCI extension number '+str(ext.extver())+
                                 ' keywords updated/added:\n', 
                                 category='header')
                    log.fullinfo('GAINORIG = '+str(gainorig), 
                                 category='header' )
                    log.fullinfo('GAIN = '+str(1.0), category='header' )
                    log.fullinfo('BUNIT = '+'electrons', category='header' )
                    log.fullinfo('--------------------------------------------'
                                 ,category='header')
                # Updating VAR headers if they exist (not updating any 
                # DQ headers as no changes were made to them here)  
                for ext in adOut['VAR']:
                    # Ensure there are no GAIN and GAINORIG header keys for 
                    # the VAR extension. No errors are thrown if they aren't 
                    # there initially, so all good not to check ahead. 
                    del ext.header['GAINORIG']
                    del ext.header['GAIN']
                    
                    # Updating then logging the change to the BUNIT 
                    # key in the VAR header
                    ext.header.update('BUNIT','electrons squared' , 
                                       'Physical units')
                    # Logging the changes to the VAR extensions header keys
                    log.fullinfo('VAR extension number '+str(ext.extver())+
                                 ' keywords updated/added:\n',
                                  category='header')
                    log.fullinfo('BUNIT = '+'electrons squared', 
                                 category='header' )
                    log.fullinfo('--------------------------------------------'
                                 ,category='header')
                        
                # Adding GEM-TLM (automatic) and ADU2ELEC time stamps to PHU
                adOut.historyMark(key='ADU2ELEC',comment='Time the pixel units were converted to e-', stomp=False)
                
                # Updating the file name with the postpend/outsuffix for this
                # primitive and then reporting the new file to the reduction 
                # context.
                if postpend!=None:
                    log.debug('Calling gemt.fileNameUpdater on '+ad.filename)
                    if outNames!=None:
                        adOut.filename = gemt.fileNameUpdater(adIn=ad, 
                                                              infilename=outNames[count],
                                                          postpend=postpend, 
                                                          strip=False)
                    else:
                        adOut.filename = gemt.fileNameUpdater(adIn=ad, 
                                                          postpend=postpend, 
                                                          strip=False)
                elif postpend==None:
                    if len(outNames)>1: 
                        adOut.filename = outNames[count]
                    else:
                        adOut.filename = outNames
                log.status('File name updated to '+adOut.filename)
                
                # Updating logger with time stamps
                log.fullinfo('************************************************'
                             , category='header')
                log.fullinfo('File = '+adOut.filename, category='header')
                log.fullinfo('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
                             , category='header')
                log.fullinfo('PHU keywords updated/added:\n', category='header')
                log.fullinfo('GEM-TLM = '+adOut.phuGetKeyValue('GEM-TLM'), 
                             category='header')
                log.fullinfo('ADU2ELEC = '+adOut.phuGetKeyValue('ADU2ELEC'), 
                             category='header')
                log.fullinfo('------------------------------------------------'
                             , category='header')
                if len(adIns)>1:
                    adOuts.append(adOut)
                else:
                    adOuts = adOut

                count=count+1
        else:
            raise('The parameter "adIns" must not be None')
        log.status('**FINISHED** the ADUtoElectrons function')
        
        return adOuts
    except:
        raise('An error occurred while trying to run ADUtoElectrons')