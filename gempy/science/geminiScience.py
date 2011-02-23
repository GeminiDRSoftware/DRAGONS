
#Author: Kyle Mede, January 2011
#For now, this module is to hold the code which performs the actual work of the 
#primitives that is considered generic enough to be at the 'gemini' level of
#the hierarchy tree.

import os

import pyfits as pf
import numpy as np
from copy import deepcopy

from astrodata.adutils import gemLog
from astrodata.AstroData import AstroData
from gempy.instruments import geminiTools  as gemt
from astrodata.adutils.gemutil import pyrafLoader
from gempy.instruments.geminiCLParDicts import CLDefaultParamsDict

def adu_to_electrons(adIns=None, outNames=None, postpend=None, logName='', 
                                                    verbose=1, noLogFile=False):
    """
    This function will convert the inputs from having pixel values in ADU to 
    that of electrons by use of the arith 'toolbox'.
    
    A string representing the name of the log file to write all log messages to
    can be defined, or a default of 'gemini.log' will be used.  If the file
    all ready exists in the directory you are working in, then this file will 
    have the log messages during this function added to the end of it.

    Note: 
      the SCI extensions of the input AstroData objects must have 'GAIN'
      header key values available to multiply them by for conversion to 
      e- units.
          
    :param adIns: Astrodata inputs to be converted to Electron pixel units
    :type adIns: Astrodata objects, either a single or a list of objects
    
    :param outNames: filenames of output(s)
    :type outNames: 
        String, either a single or a list of strings of same length as adIns.
    
    :param postpend: 
        string to postpend on the end of the input filenames 
        (or outNames if not None) for the output filenames.
    :type postpend: string
    
    :param logName: Name of the log file, default is 'gemini.log'
    :type logName: string
    
    :param verbose: 
         verbosity setting for the log messages to screen,
         default is 'critical' messages only.
         Note: independent of verbose setting, all messages always go 
         to the logfile if it is not turned off.
    :type verbose: integer from 0-6, 0=nothing to screen, 6=everything to screen
    
    :param noLogFile: A boolean to make it so no log file is created
    :type noLogFile: Python boolean (True/False)
    """
    
    log=gemLog.getGeminiLog(logName=logName, verbose=verbose, noLogFile=noLogFile)
        
    log.status('**STARTING** the ADUtoElectrons function')
    
    if not isinstance(adIns,list):
        adIns=[adIns]
    
    if (adIns!=None) and (outNames!=None):
        if isinstance(outNames,list):
            if len(adIns)!= len(outNames):
                if postpend==None:
                   raise ('Then length of the inputs, '+str(len(adIns))+
                       ', did not match the length of the outputs, '+
                       str(len(outNames))+
                       ' AND no value of "postpend" was passed in')
    
    try:
        if adIns!=None:
            # Set up counter for looping through outNames list
            count=0
            
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
                    log.fullinfo('-'*50, category='header')
                    
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
                    log.fullinfo('-'*50, category='header')
                        
                # Adding GEM-TLM (automatic) and ADU2ELEC time stamps to PHU
                adOut.historyMark(key='ADU2ELEC', stomp=False)
                
                # Updating the file name with the postpend for this
                # function and then reporting the new file 
                if postpend!=None:
                    log.debug('Calling gemt.fileNameUpdater on '+adOut.filename)
                    if outNames!=None:
                        adOut.filename = gemt.fileNameUpdater(adIn=adOut, 
                                                              infilename=outNames[count],
                                                          postpend=postpend, 
                                                          strip=False, verbose=verbose)
                    else:
                        adOut.filename = gemt.fileNameUpdater(adIn=adOut, 
                                                          postpend=postpend, 
                                                          strip=False, verbose=verbose)
                elif postpend==None:
                    if outNames!=None:
                        if len(outNames)>1: 
                            adOut.filename = outNames[count]
                        else:
                            adOut.filename = outNames
                    else:
                        raise('outNames and postpend parameters can not BOTH\
                                                                    be None')
                        
                log.status('File name updated to '+adOut.filename)
                
                # Updating logger with time stamps
                log.fullinfo('*'*50, category='header')
                log.fullinfo('File = '+adOut.filename, category='header')
                log.fullinfo('~'*50, category='header')
                log.fullinfo('PHU keywords updated/added:\n', category='header')
                log.fullinfo('GEM-TLM = '+adOut.phuGetKeyValue('GEM-TLM'), 
                             category='header')
                log.fullinfo('ADU2ELEC = '+adOut.phuGetKeyValue('ADU2ELEC'), 
                             category='header')
                log.fullinfo('-'*50, category='header')
                
                if (isinstance(adIns,list)) and (len(adIns)>1):
                    adOuts.append(adOut)
                else:
                    adOuts = adOut

                count=count+1
        else:
            raise('The parameter "adIns" must not be None')
        log.status('**FINISHED** the ADUtoElectrons function')
        # Return the outputs (list or single, matching adIns)
        return adOuts
    except:
        raise('An error occurred while trying to run ADUtoElectrons')
    
    
def add_dq(adIns, fl_nonlinear=True, fl_saturated=True,outNames=None, postpend=None, 
                                    logName='', verbose=1, noLogFile=False):
    """
    This function will create a numpy array for the data quality 
    of each SCI frame of the input data. This will then have a 
    header created and append to the input using AstroData as a DQ 
    frame. The value of a pixel will be the sum of the following: 
    (0=good, 1=bad pixel (found in bad pixel mask), 
    2=value is non linear, 4=pixel is saturated)
    
    NOTE: For every SCI extension of the inputs, a matching BPM extension must
          also exist to be updated with the non-linear and saturated pixels from
          the SCI data array for creation of the DQ array.
          ie. for now, no BPM extensions=this function will crash
          
    NOTE:
        FIND A WAY TO TAKE CARE OF NO BPM EXTENSION EXISTS ISSUE, OR ALLOWING THEM
        TO PASS IT IN...
    
    A string representing the name of the log file to write all log messages to
    can be defined, or a default of 'gemini.log' will be used.  If the file
    all ready exists in the directory you are working in, then this file will 
    have the log messages during this function added to the end of it.
    
    :param adIns: Astrodata inputs to have DQ extensions added to
    :type adIns: Astrodata objects, either a single or a list of objects
    
    :param fl_nonlinear: Flag to turn checking for nonlinear pixels on/off
    :type fl_nonLinear: Python boolean (True/False), default is True
    
    :param fl_saturated: Flag to turn checking for saturated pixels on/off
    :type fl_saturated: Python boolean (True/False), default is True
    
    :param outNames: filenames of output(s)
    :type outNames: String, either a single or a list of strings of same length as adIns.
    
    :param postpend: 
       string to postpend on the end of the input filenames 
       (or outNames if not None) for the output filenames.
    :type postpend: string
    
    :param logName: Name of the log file, default is 'gemini.log'
    :type logName: string
    
    :param verbose: 
         verbosity setting for the log messages to screen,
         default is 'critical' messages only.
         Note: independent of verbose setting, all messages always go 
         to the logfile if it is not turned off.
    :type verbose: integer from 0-6, 0=nothing to screen, 6=everything to screen
    
    :param noLogFile: A boolean to make it so no log file is created
    :type noLogFile: Python boolean (True/False)
    """
    
    log=gemLog.getGeminiLog(logName=logName, verbose=verbose, noLogFile=noLogFile)
        
    log.status('**STARTING** the addDQ function')
    
    if not isinstance(adIns,list):
        adIns=[adIns]
    
    if (adIns!=None) and (outNames!=None):
        if isinstance(outNames,list):
            if len(adIns)!= len(outNames):
                if postpend==None:
                   raise ('Then length of the inputs, '+str(len(adIns))+
                       ', did not match the length of the outputs, '+
                       str(len(outNames))+
                       ' AND no value of "postpend" was passed in')
    
    try:
        if adIns!=None:
            # Set up counter for looping through outNames list
            count=0
            
            # Creating empty list of ad's to be returned that will be filled below
            if len(adIns)>1:
                adOuts=[]
            
            # Loop through the inputs to perform the non-linear and saturated
            # pixel searches of the SCI frames to update the BPM frames into
            # full DQ frames. 
            for ad in adIns:                
                # Check if DQ extensions all ready exist for this file
                if not ad['DQ']:
                    # Making a deepcopy of the input to work on
                    # (ie. a truly new+different object that is a complete copy of the input)
                    adOut = deepcopy(ad)
                    # moving the filename over as deepcopy doesn't do that
                    adOut.filename = ad.filename
                    
                    for sciExt in adOut['SCI']: 
                        # Retrieving BPM extension 
                        bpmAD = adOut[('BPM',sciExt.extver())]
                        
                        # Extracting the BPM data array for this extension
                        BPMArray = bpmAD.data
                        
                        # Extracting the BPM header for this extension to be 
                        # later converted to a DQ header
                        dqheader = bpmAD.header
                        
                        # Getting the data section from the header and 
                        # converting to an integer list
                        datasecStr = sciExt.data_section()
                        datasecList = gemt.secStrToIntList(datasecStr) 
                        dsl = datasecList
                        
                        # Preparing the non linear and saturated pixel arrays
                        # and their respective constants
                        nonLinArray = np.zeros(sciExt.data.shape, 
                                               dtype=np.int16)
                        saturatedArray = np.zeros(sciExt.data.shape, 
                                                  dtype=np.int16)
                        linear = sciExt.non_linear_level()
                        saturated = sciExt.saturation_level()
    
                        if (linear is not None) and (fl_nonlinear): 
                            log.debug('Performing an np.where to find '+
                                      'non-linear pixels for extension '+
                                      str(sciExt.extver())+' of '+adOut.filename)
                            nonLinArray = np.where(sciExt.data>linear,2,0)
                            log.status('Done calculating array of non-linear'+
                                       ' pixels')
                        if (saturated is not None) and (fl_saturated):
                            log.debug('Performing an np.where to find '+
                                      'saturated pixels for extension '+
                                      str(sciExt.extver())+' of '+adOut.filename)
                            saturatedArray = np.where(sciExt.data>saturated,4,0)
                            log.status('Done calculating array of saturated'+
                                       ' pixels') 
                        
                        # Creating one DQ array from the three
                        dqArray=np.add(BPMArray, nonLinArray, 
                                       saturatedArray) 
                        # Updating data array for the BPM array to be the 
                        # newly calculated DQ array
                        adOut[('BPM',sciExt.extver())].data = dqArray
                        
                        # Renaming the extension to DQ from BPM
                        dqheader.update('EXTNAME', 'DQ', 'Extension Name')
                        
                        # Using renameExt to correctly set the EXTVER and 
                        # EXTNAME values in the header   
                        bpmAD.renameExt('DQ', ver=sciExt.extver(), force=True)

                        # Logging that the name of the BPM extension was changed
                        log.fullinfo('BPM Extension '+str(sciExt.extver())+
                                     ' of '+adOut.filename+' had its EXTVER '+
                                     'changed to '+
                                     adOut[('DQ',sciExt.extver())].header['EXTNAME'])
                        
                # If DQ frames exist, send a critical message to the logger
                else:
                    log.critical('DQ frames all ready exist for '+adOut.filename+
                                 ', so addDQ will not calculate new ones')
                    
                # Adding GEM-TLM (automatic) and ADDDQ time stamps to the PHU
                adOut.historyMark(key='ADDDQ', stomp=False) 
                
                # updating logger with updated/added time stamps
                log.fullinfo('*'*50, category='header')
                log.fullinfo('PHU keywords updated/added:\n', 
                             category='header')
                log.fullinfo('GEM-TLM = '+adOut.phuGetKeyValue('GEM-TLM'), 
                             category='header')
                log.fullinfo('ADDDQ = '+adOut.phuGetKeyValue('ADDDQ'), 
                             category='header')
                log.fullinfo('-'*50, category='header')
                
                # Updating the file name with the postpend for this
                # function and then reporting the new file 
                if postpend!=None:
                    log.debug('Calling gemt.fileNameUpdater on '+adOut.filename)
                    if outNames!=None:
                        adOut.filename = gemt.fileNameUpdater(adIn=adOut, 
                                                              infilename=outNames[count],
                                                          postpend=postpend, 
                                                          strip=False, verbose=verbose)
                    else:
                        adOut.filename = gemt.fileNameUpdater(adIn=adOut, 
                                                          postpend=postpend, 
                                                          strip=False, verbose=verbose)
                elif postpend==None:
                    if outNames!=None:
                        if len(outNames)>1: 
                            adOut.filename = outNames[count]
                        else:
                            adOut.filename = outNames
                    else:
                        raise('outNames and postpend parameters can not BOTH\
                                                                    be None')
                        
                log.status('File name updated to '+adOut.filename)
            
                if (isinstance(adIns,list)) and (len(adIns)>1):
                    adOuts.append(adOut)
                else:
                    adOuts = adOut

                count=count+1
        else:
            raise('The parameter "adIns" must not be None')
        
        log.status('**FINISHED** the addDQ function')
        # Return the outputs (list or single, matching adIns)
        return adOuts
    except:
        raise ('An error occurred while trying to run addDQ')
    

def add_bpm(adIns=None, BPMs=None, matchSize=False, outNames=None, postpend=None, logName='', 
                                                    verbose=1, noLogFile=False):
    """
    This function will add the provided BPM (Bad Pixel Mask) to the inputs.  
    The BPM will be added as frames matching that of the SCI frames and ensure
    the BPM's data array is the same size as that of the SCI data array.  If the 
    SCI array is larger (say SCI's were overscan trimmed, but BPMs were not), the
    BPMs will have their arrays padded with zero's to match the sizes and use the 
    data_section descriptor on the SCI data arrays to ensure the match is
    a correct fit.  There must be a matching number of DQ extensions in the BPM 
    as the input the BPM frames are to be added to 
    (ie. if input has 3 SCI extensions, the BPM must have 3 DQ extensions).
    
    A string representing the name of the log file to write all log messages to
    can be defined, or a default of 'gemini.log' will be used.  If the file
    all ready exists in the directory you are working in, then this file will 
    have the log messages during this function added to the end of it.
          
    :param adIns: Astrodata inputs to be converted to Electron pixel units
    :type adIns: Astrodata objects, either a single or a list of objects
    
    :param BPMs: The BPM(s) to be added to the input(s).
    :type BPMs: 
       AstroData objects in a list, or a single instance.
       Note: If there is multiple inputs and one BPM provided, then the
       same BPM will be applied to all inputs; else the BPMs list  
       must match the length of the inputs.
           
    :param matchSize: A flag to use zeros and the key 'DETSEC' of the 'SCI'
                      extensions to match the size of the BPM arrays to those of
                       fo the 'SCI' data arrays.
    :type matchSize: Python boolean (True/False). Default: False.
                      
    :param outNames: filenames of output(s)
    :type outNames: String, either a single or a list of strings of same length as adIns.
    
    :param postpend: 
         string to postpend on the end of the input filenames 
         (or outNames if not None) for the output filenames.
    :type postpend: string
    
    :param logName: Name of the log file, default is 'gemini.log'
    :type logName: string
    
    :param verbose: 
          verbosity setting for the log messages to screen,
          default is 'critical' messages only.
          Note: independent of verbose setting, all messages always go 
          to the logfile if it is not turned off.
    :type verbose: integer from 0-6, 0=nothing to screen, 6=everything to screen
    
    :param noLogFile: A boolean to make it so no log file is created
    :type noLogFile: Python boolean (True/False)
    """
    
    log=gemLog.getGeminiLog(logName=logName, verbose=verbose, noLogFile=noLogFile)
        
    log.status('**STARTING** the addBPM function')
    
    if not isinstance(adIns,list):
        adIns=[adIns]
    
    if (adIns!=None) and (outNames!=None):
        if isinstance(outNames,list):
            if len(adIns)!= len(outNames):
                if postpend==None:
                   raise ('Then length of the inputs, '+str(len(adIns))+
                       ', did not match the length of the outputs, '+
                       str(len(outNames))+
                       ' AND no value of "postpend" was passed in')
                   
    if BPMs==None:
        raise('There must be at least one BPM provided, the "BPMs" parameter must not be None.')
                   
    try:
        # Ensure there are inputs to work on and BPMs to add to the inputs
        if (adIns!=None) and (BPMs!=None):
            # Set up counter for looping through outNames/BPMs lists
            count=0
            
            # Creating empty list of ad's to be returned that will be filled below
            if len(adIns)>1:
                adOuts=[]
            
            # Do the work on each ad in the inputs
            for ad in adIns:
                # Getting the right BPM for this input
                if isinstance(BPMs, list):
                    if len(BPMs)>1:
                        BPM = BPMs[count]
                    else:
                        BPM = BPMs[0]
                else:
                    BPM = BPMs
                
                # Check if this input all ready has a BPM extension
                if not ad['BPM']:
                    # Making a deepcopy of the input to work on
                    # (ie. a truly new+different object that is a complete copy of the input)
                    adOut = deepcopy(ad)
                    # moving the filename over as deepcopy doesn't do that
                    adOut.filename = ad.filename
                    
                    # Getting the filename for the BPM and removing any paths
                    BPMfilename = os.path.basename(BPM.filename)
                    
                    for sciExt in adOut['SCI']:
                        # Extracting the matching DQ extension from the BPM 
                        BPMArrayIn = BPM[('DQ',sciExt.extver())].data
                        
                        # logging the BPM file being used for this SCI extension
                        log.fullinfo('SCI extension number '+
                                     str(sciExt.extver())+', of file '+
                                     adOut.filename+ ' is matched to DQ extension '
                                     +str(sciExt.extver())+' of BPM file '+
                                     BPMfilename)
                        
                        # Matching size of BPM array to that of the SCI data array
                        if matchSize:
                            # Getting the data section from the header and 
                            # converting to an integer list
                            datasecStr = sciExt.data_section()
                            datasecList = gemt.secStrToIntList(datasecStr) 
                            dsl = datasecList
                            datasecShape = (dsl[3]-dsl[2]+1, dsl[1]-dsl[0]+1)
                            
                            # Creating a zeros array the same size as SCI array
                            # for this extension
                            BPMArrayOut = np.zeros(sciExt.data.shape, 
                                                   dtype=np.int16)
        
                            # Loading up zeros array with data from BPM array
                            # if the sizes match then there is no change, else
                            # output BPM array will be 'padded with zeros' or 
                            # 'not bad pixels' to match SCI's size.
                            if BPMArrayIn.shape==datasecShape:
                                BPMArrayOut[dsl[2]-1:dsl[3], dsl[0]-1:dsl[1]] = \
                                                                        BPMArrayIn
                            elif BPMArrayIn.shape==BPMArrayOut.shape:
                                BPMArrayOut[dsl[2]-1:dsl[3], dsl[0]-1:dsl[1]] = \
                                    BPMArrayIn[dsl[2]-1:dsl[3], dsl[0]-1:dsl[1]]
                        
                        # Don't match size
                        else:
                            BPMArrayOut = BPMArrayIn
                            
                        # Creating a header for the BPM array and updating
                        # further updating to this header will take place in 
                        # addDQ primitive
                        BPMheader = pf.Header() 
                        BPMheader.update('BITPIX', 16, 
                                        'number of bits per data pixel')
                        BPMheader.update('NAXIS', 2)
                        BPMheader.update('PCOUNT', 0, 
                                        'required keyword; must = 0')
                        BPMheader.update('GCOUNT', 1, 
                                        'required keyword; must = 1')
                        BPMheader.update('BUNIT', 'bit', 'Physical units')
                        BPMheader.update('BPMFILE', BPMfilename, 
                                            'Bad Pixel Mask file name')
                        BPMheader.update('EXTVER', sciExt.extver(), 
                                            'Extension Version')
                        # This extension will be renamed DQ in addDQ
                        BPMheader.update('EXTNAME', 'BPM', 'Extension Name')
                        
                        # Creating an astrodata instance from the 
                        # DQ array and header
                        bpmAD = AstroData(header=BPMheader, data=BPMArrayOut)
                        
                        # Using renameExt to correctly set the EXTVER and 
                        # EXTNAME values in the header   
                        bpmAD.renameExt('BPM', ver=sciExt.extver())
                        
                        # Appending BPM astrodata instance to the input one
                        log.debug('Appending new BPM HDU onto the file '+ 
                                  adOut.filename)
                        adOut.append(bpmAD)
                        log.status('Appending BPM complete for '+ adOut.filename)
            
                # If BPM frames exist, send a critical message to the logger
                else:
                    log.critical('BPM frames all ready exist for '+adOut.filename+
                                 ', so addBPM will add new ones')
                    
                # Updating GEM-TLM (automatic) and ADDBPM time stamps to the PHU
                adOut.historyMark(key='ADDBPM', stomp=False) 
                # Updating logger with updated/added time stamps
                log.fullinfo('*'*50, category='header')
                log.fullinfo('PHU keywords updated/added:\n', category='header')
                log.fullinfo('GEM-TLM = '+adOut.phuGetKeyValue('GEM-TLM'),
                             category='header')
                log.fullinfo('ADDBPM = '+adOut.phuGetKeyValue('ADDBPM'), 
                             category='header')
                log.fullinfo('-'*50, category='header')
                
                # Updating the file name with the postpend for this
                # function and then reporting the new file 
                if postpend!=None:
                    log.debug('Calling gemt.fileNameUpdater on '+adOut.filename)
                    if outNames!=None:
                        adOut.filename = gemt.fileNameUpdater(adIn=adOut, 
                                                              infilename=outNames[count],
                                                          postpend=postpend, 
                                                          strip=False, verbose=verbose)
                    else:
                        adOut.filename = gemt.fileNameUpdater(adIn=adOut, 
                                                          postpend=postpend, 
                                                          strip=False, verbose=verbose)
                elif postpend==None:
                    if outNames!=None:
                        if len(outNames)>1: 
                            adOut.filename = outNames[count]
                        else:
                            adOut.filename = outNames
                    else:
                        raise('outNames and postpend parameters can not BOTH\
                                                                    be None')
                        
                log.status('File name updated to '+adOut.filename)
            
                if (isinstance(adIns,list)) and (len(adIns)>1):
                    adOuts.append(adOut)
                else:
                    adOuts = adOut

                count=count+1
        else:
            raise('The parameter "adIns" must not be None')
        
        log.status('**FINISHED** the addBPM function')
        # Return the outputs (list or single, matching adIns)
        return adOuts
    except:
        raise ('An error occurred while trying to run addBPM')
    

def add_var(adIns, outNames=None, postpend=None, logName='', verbose=1, 
                                                            noLogFile=False):
    """
    This function uses numpy to calculate the variance of each SCI frame
    in the input files and appends it as a VAR frame using AstroData.
    
    The calculation will follow the formula:
    variance = (read noise/gain)2 + max(data,0.0)/gain
    
    A string representing the name of the log file to write all log messages to
    can be defined, or a default of 'gemini.log' will be used.  If the file
    all ready exists in the directory you are working in, then this file will 
    have the log messages during this function added to the end of it.
    
    :param adIns: Astrodata inputs to have DQ extensions added to
    :type adIns: Astrodata objects, either a single or a list of objects
    
    :param outNames: filenames of output(s)
    :type outNames: String, either a single or a list of strings of same length as adIns.
    
    :param postpend: 
        string to postpend on the end of the input filenames 
        (or outNames if not None) for the output filenames.
    :type postpend: string
    
    :param logName: Name of the log file, default is 'gemini.log'
    :type logName: string
    
    :param verbose: 
        verbosity setting for the log messages to screen,
        default is 'critical' messages only.
        Note: independent of verbose setting, all messages always go 
        to the logfile if it is not turned off.
    :type verbose: integer from 0-6, 0=nothing to screen, 6=everything to screen
    
    :param noLogFile: A boolean to make it so no log file is created
    :type noLogFile: Python boolean (True/False)
    """
    
    log=gemLog.getGeminiLog(logName=logName, verbose=verbose, noLogFile=noLogFile)
        
    log.status('**STARTING** the addVAR function')
    
    if not isinstance(adIns,list):
        adIns=[adIns]
    
    if (adIns!=None) and (outNames!=None):
        if isinstance(outNames,list):
            if len(adIns)!= len(outNames):
                if postpend==None:
                   raise ('Then length of the inputs, '+str(len(adIns))+
                       ', did not match the length of the outputs, '+
                       str(len(outNames))+
                       ' AND no value of "postpend" was passed in')
    
    try:
        if adIns!=None:
            # Set up counter for looping through outNames list
            count=0
            
            # Creating empty list of ad's to be returned that will be filled below
            if len(adIns)>1:
                adOuts=[]
            
            # Loop through the inputs to perform the non-linear and saturated
            # pixel searches of the SCI frames to update the BPM frames into
            # full DQ frames. 
            for ad in adIns:   
                # Making a deepcopy of the input to work on
                # (ie. a truly new+different object that is a complete copy of the input)
                adOut = deepcopy(ad)
                # moving the filename over as deepcopy doesn't do that
                adOut.filename = ad.filename
                
                # To clean up log and screen if multiple inputs
                log.fullinfo('+'*50, category='format')
                # Check if there VAR frames all ready exist
                if not adOut['VAR']:                 
                    # If VAR frames don't exist, loop through the SCI extensions 
                    # and calculate a corresponding VAR frame for it, then 
                    # append it
                    for sciExt in adOut['SCI']:
                        # var = (read noise/gain)2 + max(data,0.0)/gain
                        
                        # Retrieving necessary values (read noise, gain)
                        readNoise=sciExt.read_noise()
                        gain=sciExt.gain()
                        # Creating (read noise/gain) constant
                        rnOverG=readNoise/gain
                        # Convert negative numbers (if they exist) to zeros
                        maxArray=np.where(sciExt.data>0.0,0,sciExt.data)
                        # Creating max(data,0.0)/gain array
                        maxOverGain=np.divide(maxArray,gain)
                        # Putting it all together
                        varArray=np.add(maxOverGain,rnOverG*rnOverG)
                         
                        # Creating the variance frame's header and updating it     
                        varheader = pf.Header()
                        varheader.update('NAXIS', 2)
                        varheader.update('PCOUNT', 0, 
                                         'required keyword; must = 0 ')
                        varheader.update('GCOUNT', 1, 
                                         'required keyword; must = 1')
                        varheader.update('EXTNAME', 'VAR', 
                                         'Extension Name')
                        varheader.update('EXTVER', sciExt.extver(), 
                                         'Extension Version')
                        varheader.update('BITPIX', -32, 
                                         'number of bits per data pixel')
                        
                        # Turning individual variance header and data 
                        # into one astrodata instance
                        varAD = AstroData(header=varheader, data=varArray)
                        
                        # Appending variance astrodata instance onto input one
                        log.debug('Appending new VAR HDU onto the file '
                                     +adOut.filename)
                        adOut.append(varAD)
                        log.status('appending VAR frame '+str(sciExt.extver())+
                                   ' complete for '+adOut.filename)
                        
                # If VAR frames all ready existed, 
                # make a critical message in the logger
                else:
                    log.critical('VAR frames all ready exist for '+adOut.filename+
                                 ', so addVAR will not calculate new ones')
                
                # Adding GEM-TLM(automatic) and ADDVAR time stamps to the PHU     
                adOut.historyMark(key='ADDVAR', stomp=False)    
                
                log.fullinfo('*'*50, category='header')
                log.fullinfo('file = '+adOut.filename, category='header')
                log.fullinfo('~'*50, category='header')
                log.fullinfo('PHU keywords updated/added:\n', category='header')
                log.fullinfo('GEM-TLM = '+adOut.phuGetKeyValue('GEM-TLM'), 
                             category='header')
                log.fullinfo('ADDVAR = '+adOut.phuGetKeyValue('ADDVAR'), 
                             category='header')
                log.fullinfo('-'*50, category='header')
                
                # Updating the file name with the postpend for this
                # function and then reporting the new file 
                if postpend!=None:
                    log.debug('Calling gemt.fileNameUpdater on '+adOut.filename)
                    if outNames!=None:
                        adOut.filename = gemt.fileNameUpdater(adIn=adOut, 
                                                              infilename=outNames[count],
                                                          postpend=postpend, 
                                                          strip=False, verbose=verbose)
                    else:
                        adOut.filename = gemt.fileNameUpdater(adIn=adOut, 
                                                          postpend=postpend, 
                                                          strip=False, verbose=verbose)
                elif postpend==None:
                    if outNames!=None:
                        if len(outNames)>1: 
                            adOut.filename = outNames[count]
                        else:
                            adOut.filename = outNames
                    else:
                        raise('outNames and postpend parameters can not BOTH\
                                                                    be None')
                        
                log.status('File name updated to '+adOut.filename)
            
                if (isinstance(adIns,list)) and (len(adIns)>1):
                    adOuts.append(adOut)
                else:
                    adOuts = adOut

                count=count+1
        else:
            raise('The parameter "adIns" must not be None')
        
        log.status('**FINISHED** the addVAR function')
        # Return the outputs (list or single, matching adIns)
        return adOuts
    except:
        raise ('An error occurred while trying to run addVAR')
    
    
def flat_correct(adIns, flats=None, outNames=None, postpend=None, logName='', verbose=1, 
                                                            noLogFile=False):
    """
    This function performs a flat correction by dividing the inputs by  
    processed flats, similar to the way gireduce would perform this operation
    but written in pure python in the arith toolbox.
    
    A string representing the name of the log file to write all log messages to
    can be defined, or a default of 'gemini.log' will be used.  If the file
    all ready exists in the directory you are working in, then this file will 
    have the log messages during this function added to the end of it.
    
    :param adIns: Astrodata inputs to have DQ extensions added to
    :type adIns: Astrodata objects, either a single or a list of objects
    
    :param flats: The flat(s) to divide the input(s) by.
    :type flats: AstroData objects in a list, or a single instance.
                Note: If there is multiple inputs and one flat provided, then the
                same flat will be applied to all inputs; else the flats   
                list must match the length of the inputs.
    
    :param outNames: filenames of output(s)
    :type outNames: String, either a single or a list of strings of same length as adIns.
    
    :param postpend: string to postpend on the end of the input filenames 
                    (or outNames if not None) for the output filenames.
    :type postpend: string
    
    :param logName: Name of the log file, default is 'gemini.log'
    :type logName: string
    
    :param verbose: verbosity setting for the log messages to screen,
                    default is 'critical' messages only.
                    Note: independent of verbose setting, all messages always go 
                    to the logfile if it is not turned off.
    :type verbose: integer from 0-6, 0=nothing to screen, 6=everything to screen
    
    :param noLogFile: A boolean to make it so no log file is created
    :type noLogFile: Python boolean (True/False)
    """
    
    log=gemLog.getGeminiLog(logName=logName, verbose=verbose, noLogFile=noLogFile)
        
    log.status('**STARTING** the flatCorrect function')
    
    if not isinstance(adIns,list):
        adIns=[adIns]
    
    if (adIns!=None) and (outNames!=None):
        if isinstance(outNames,list):
            if len(adIns)!= len(outNames):
                if postpend==None:
                   raise ('Then length of the inputs, '+str(len(adIns))+
                       ', did not match the length of the outputs, '+
                       str(len(outNames))+
                       ' AND no value of "postpend" was passed in')
    
    if flats==None:
        raise('There must be at least one processed flat provided, the "flats" parameter must not be None.')
    
    try:
        if adIns!=None:
            # Set up counter for looping through outNames list
            count=0
            
            # Creating empty list of ad's to be returned that will be filled below
            if len(adIns)>1:
                adOuts=[]
            
            # Loop through the inputs to perform the non-linear and saturated
            # pixel searches of the SCI frames to update the BPM frames into
            # full DQ frames. 
            for ad in adIns:                   
                # To clean up log and screen if multiple inputs
                log.fullinfo('+'*50, category='format')    
    
                # Getting the right flat for this input
                if isinstance(flats, list):
                    if len(flats)>1:
                        processedFlat = flats[count]
                    else:
                        processedFlat = flats[0]
                else:
                    processedFlat = flats
    
                log.status('Input flat file being used for flat correction '
                           +processedFlat.filename)
                log.debug('Calling ad.div on '+ad.filename)
                
                adOut = ad.div(processedFlat)
                adOut.filename = ad.filename
                log.status('ad.div successfully flat corrected '+ad.filename)
                
                # Updating GEM-TLM (automatic) and FLATCORR time stamps to the 
                # PHU
                adOut.historyMark(key='FLATCORR', stomp=False)   
                
                # Updating logger with new GEM-TLM value
                log.fullinfo('*'*50, category='header')
                log.fullinfo('File = '+adOut.filename, category='header')
                log.fullinfo('~'*50, category='header')
                log.fullinfo('PHU keywords updated/added:\n', 'header')
                log.fullinfo('GEM-TLM = '+adOut.phuGetKeyValue('GEM-TLM'), 
                             category='header')
                log.fullinfo('FLATCORR = '+adOut.phuGetKeyValue('FLATCORR'), 
                             category='header')
                log.fullinfo('-'*50, category='header')  
                
                # Updating the file name with the postpend for this
                # function and then reporting the new file 
                if postpend!=None:
                    log.debug('Calling gemt.fileNameUpdater on '+adOut.filename)
                    if outNames!=None:
                        adOut.filename = gemt.fileNameUpdater(adIn=adOut, 
                                                              infilename=outNames[count],
                                                          postpend=postpend, 
                                                          strip=False, verbose=verbose)
                    else:
                        adOut.filename = gemt.fileNameUpdater(adIn=adOut, 
                                                          postpend=postpend, 
                                                          strip=False, verbose=verbose)
                elif postpend==None:
                    if outNames!=None:
                        if len(outNames)>1: 
                            adOut.filename = outNames[count]
                        else:
                            adOut.filename = outNames
                    else:
                        raise('outNames and postpend parameters can not BOTH\
                                                                    be None')
                        
                log.status('File name updated to '+adOut.filename)
            
                if (isinstance(adIns,list)) and (len(adIns)>1):
                    adOuts.append(adOut)
                else:
                    adOuts = adOut

                count=count+1
        else:
            raise('The parameter "adIns" must not be None')
        
        log.status('**FINISHED** the flatCorrect function')
        
        # Return the outputs (list or single, matching adIns)
        return adOuts
    except:
        raise ('An error occurred while trying to run flatCorrect')
    
def overscan_trim(adIns, outNames=None, postpend=None, logName='', verbose=1, 
                                                            noLogFile=False):
    """
    This function uses AstroData to trim the overscan region 
    from the input images and update their headers.
    
    A string representing the name of the log file to write all log messages to
    can be defined, or a default of 'gemini.log' will be used.  If the file
    all ready exists in the directory you are working in, then this file will 
    have the log messages during this function added to the end of it.
    
    :param adIns: Astrodata inputs to have DQ extensions added to
    :type adIns: Astrodata objects, either a single or a list of objects
    
    :param outNames: filenames of output(s)
    :type outNames: String, either a single or a list of strings of same length
                    as adIns.
    
    :param postpend: string to postpend on the end of the input filenames 
                    (or outNames if not None) for the output filenames.
    :type postpend: string
    
    :param logName: Name of the log file, default is 'gemini.log'
    :type logName: string
    
    :param verbose: verbosity setting for the log messages to screen,
                    default is 'critical' messages only.
                    Note: independent of verbose setting, all messages always go 
                          to the logfile if it is not turned off.
    :type verbose: integer from 0-6, 0=nothing to screen, 6=everything to screen
    
    :param noLogFile: A boolean to make it so no log file is created
    :type noLogFile: Python boolean (True/False)
    """
    
    if logName!='':
        log=gemLog.getGeminiLog(logName=logName, verbose=verbose, 
                                noLogFile=noLogFile)
    else:
        # Use default logName 'gemini.log'
        log=gemLog.getGeminiLog(verbose=verbose, noLogFile=noLogFile)
        
    log.status('**STARTING** the overscanTrim function')
    
    if not isinstance(adIns,list):
        adIns=[adIns]
    
    if (adIns!=None) and (outNames!=None):
        if isinstance(outNames,list):
            if len(adIns)!= len(outNames):
                if postpend==None:
                   raise ('Then length of the inputs, '+str(len(adIns))+
                       ', did not match the length of the outputs, '+
                       str(len(outNames))+
                       ' AND no value of "postpend" was passed in')
    
    try:
        if adIns!=None:
            # Set up counter for looping through outNames list
            count=0
            
            # Creating empty list of ad's to be returned that will be filled below
            if len(adIns)>1:
                adOuts=[]
            
            # Loop through the inputs to perform the non-linear and saturated
            # pixel searches of the SCI frames to update the BPM frames into
            # full DQ frames. 
            for ad in adIns:  
                # Making a deepcopy of the input to work on
                # (ie. a truly new+different object that is a complete copy of the input)
                adOut = deepcopy(ad)
                # moving the filename over as deepcopy doesn't do that
                adOut.filename = ad.filename
                                 
                # To clean up log and screen if multiple inputs
                log.fullinfo('+'*50, category='format')    
                
                for sciExt in adOut['SCI']:
                    # Converting data section string to an integer list
                    datasecStr=sciExt.data_section()
                    datasecList=gemt.secStrToIntList(datasecStr) 
                    dsl=datasecList
                    # Updating logger with the section being kept
                    log.stdinfo('\nfor '+adOut.filename+' extension '+
                                str(sciExt.extver())+
                                ', keeping the data from the section '+
                                datasecStr,'science')
                    # Trimming the data section from input SCI array
                    # and making it the new SCI data
                    sciExt.data=sciExt.data[dsl[2]-1:dsl[3],dsl[0]-1:dsl[1]]
                    # Updating header keys to match new dimensions
                    sciExt.header['NAXIS1'] = dsl[1]-dsl[0]+1
                    sciExt.header['NAXIS2'] = dsl[3]-dsl[2]+1
                    newDataSecStr = '[1:'+str(dsl[1]-dsl[0]+1)+',1:'+\
                                    str(dsl[3]-dsl[2]+1)+']' 
                    sciExt.header['DATASEC']=newDataSecStr
                    sciExt.header.update('TRIMSEC', datasecStr, 
                                       'Data section prior to trimming')
                    # Updating logger with updated/added keywords to each SCI frame
                    log.fullinfo('*'*50, category='header')
                    log.fullinfo('File = '+adOut.filename, category='header')
                    log.fullinfo('~'*50, category='header')
                    log.fullinfo('SCI extension number '+str(sciExt.extver())+
                                 ' keywords updated/added:\n', 'header')
                    log.fullinfo('NAXIS1= '+str(sciExt.header['NAXIS1']),
                                category='header')
                    log.fullinfo('NAXIS2= '+str(sciExt.header['NAXIS2']),
                                 category='header')
                    log.fullinfo('DATASEC= '+newDataSecStr, category='header')
                    log.fullinfo('TRIMSEC= '+datasecStr, category='header')
                    
                adOut.phuSetKeyValue('TRIMMED','yes','Overscan section trimmed')    
                # Updating the GEM-TLM value and reporting the output to the RC    
                adOut.historyMark(key='OVERTRIM', stomp=False)                
                
                # Updating logger with updated/added keywords to the PHU
                log.fullinfo('*'*50, category='header')
                log.fullinfo('file = '+adOut.filename, category='header')
                log.fullinfo('~'*50, category='header')
                log.fullinfo('PHU keywords updated/added:\n', 'header')
                log.fullinfo('GEM-TLM = '+adOut.phuGetKeyValue('GEM-TLM'), 
                             category='header') 
                log.fullinfo('OVERTRIM = '+adOut.phuGetKeyValue('OVERTRIM')+'\n', 
                             category='header') 
                
                # Updating the file name with the postpend for this
                # function and then reporting the new file 
                if postpend!=None:
                    log.debug('Calling gemt.fileNameUpdater on '+adOut.filename)
                    if outNames!=None:
                        adOut.filename = gemt.fileNameUpdater(adIn=adOut, 
                                                              infilename=outNames[count],
                                                          postpend=postpend, 
                                                          strip=False, verbose=verbose)
                    else:
                        adOut.filename = gemt.fileNameUpdater(adIn=adOut, 
                                                          postpend=postpend, 
                                                          strip=False, verbose=verbose)
                elif postpend==None:
                    if outNames!=None:
                        if len(outNames)>1: 
                            adOut.filename = outNames[count]
                        else:
                            adOut.filename = outNames
                    else:
                        raise('outNames and postpend parameters can not BOTH\
                                                                    be None')
                        
                log.status('File name updated to '+adOut.filename)
            
                if (isinstance(adIns,list)) and (len(adIns)>1):
                    adOuts.append(adOut)
                else:
                    adOuts = adOut

                count=count+1
        else:
            raise('The parameter "adIns" must not be None')
        
        log.status('**FINISHED** the overscanTrim function')
        
        # Return the outputs (list or single, matching adIns)
        return adOuts
    except:
        raise ('An error occurred while trying to run overscanTrim')
    
################################## CL Based functions #########################


def combine(adIns, fl_vardq=True, fl_dqprop=True, method='average', 
            outNames=None, postpend=None, logName='', verbose=1, noLogFile=False):
    """
    This function will average and combine the SCI extensions of the 
    inputs. It takes all the inputs and creates a list of them and 
    then combines each of their SCI extensions together to create 
    average combination file. New VAR frames are made from these 
    combined SCI frames and the DQ frames are propagated through 
    to the final file.
    NOTE: The inputs to this function MUST be prepared. 

    A string representing the name of the log file to write all log messages to
    can be defined, or a default of 'gemini.log' will be used.  If the file
    all ready exists in the directory you are working in, then this file will 
    have the log messages during this function added to the end of it.
    
    :param adIns: Astrodata inputs to be combined
    :type adIns: Astrodata objects, either a single or a list of objects
    
    :param fl_vardq: Create variance and data quality frames?
    :type fl_vardq: Python boolean (True/False), OR string 'AUTO' to do 
                    it automatically if there are VAR and DQ frames in the inputs.
                    NOTE: 'AUTO' uses the first input to determine if VAR and DQ frames exist, 
                    so, if the first does, then the rest MUST also have them as well.
    
    :param fl_dqprop: propogate the current DQ values?
    :type fl_dqprop: Python boolean (True/False)
    
    :param method: type of combining method to use.
    :type method: string, options: 'average', 'median'.
    
    :param outNames: filenames of output(s)
    :type outNames: String, either a single or a list of strings of same length
                    as adIns.
    
    :param postpend: string to postpend on the end of the input filenames 
                    (or outNames if not None) for the output filenames.
    :type postpend: string
    
    :param logName: Name of the log file, default is 'gemini.log'
    :type logName: string
    
    :param verbose: verbosity setting for the log messages to screen,
                    default is 'critical' messages only.
                    Note: independent of verbose setting, all messages always go 
                    to the logfile if it is not turned off.
    :type verbose: integer from 0-6, 0=nothing to screen, 6=everything to screen
    
    :param noLogFile: A boolean to make it so no log file is created
    :type noLogFile: Python boolean (True/False)
    """
    
    log=gemLog.getGeminiLog(logName=logName, verbose=verbose, noLogFile=noLogFile)

    log.status('**STARTING** the combine function')
    
    if not isinstance(adIns,list):
        adIns=[adIns]
    
    if (adIns!=None) and (outNames!=None):
        if isinstance(outNames,list):
            if len(adIns)!= len(outNames):
                if postpend==None:
                   raise ('Then length of the inputs, '+str(len(adIns))+
                       ', did not match the length of the outputs, '+
                       str(len(outNames))+
                       ' AND no value of "postpend" was passed in')
    
    try:
        if adIns!=None:
            # Set up counter for looping through outNames list
            count=0
            
            # Ensuring there is more than one input to combine
            if (isinstance(adIns,list)) and (len(adIns)>1):
                
                # loading and bringing the pyraf related modules into the name-space
                pyraf, gemini, yes, no = pyrafLoader()
                
                # Determining if gireduce should propigate the VAR and DQ frames, if 'AUTO' was chosen 
                if fl_vardq=='AUTO':
                    if isinstance(adIns,list):
                        if adIns[0].countExts('VAR')==adIns[0].countExts('DQ')==adIns[0].countExts('SCI'):
                            fl_vardq=yes
                        else:
                            fl_vardq=no
                    else:
                        if adIns.countExts('VAR')==adIns.countExts('DQ')==adIns.countExts('SCI'):
                            fl_vardq=yes
                        else:
                            fl_vardq=no
                else:
                    if fl_vardq:
                        fl_vardq=yes
                    elif fl_vardq==False:
                        fl_vardq=no
                
                # Preparing input files, lists, parameters... for input to 
                # the CL script
                clm=gemt.CLManager(adIns=adIns, outNames=outNames, postpend=postpend, 
                                   funcName='combine', logName=logName, verbose=verbose, 
                                   noLogFile=noLogFile)
                
                # Check the status of the CLManager object, True=continue, False= issue warning
                if clm.status:
                
                    # Creating a dictionary of the parameters set by the CLManager  
                    # or the definition of the primitive 
                    clPrimParams = {
                        # Retrieving the inputs as a list from the CLManager
                        'input'       :clm.inputList(),
                        # Maybe allow the user to override this in the future. 
                        'output'      :clm.combineOutname(), 
                        # This returns a unique/temp log file for IRAF  
                        'logfile'     :clm.logfile(),  
                        # This is actually in the default dict but wanted to 
                        # show it again       
                        'Stdout'      :gemt.IrafStdout(verbose=verbose), 
                        # This is actually in the default dict but wanted to 
                        # show it again    
                        'Stderr'      :gemt.IrafStdout(verbose=verbose),
                        # This is actually in the default dict but wanted to 
                        # show it again     
                        'verbose'     :yes,    
                        'reject'      :'none'                
                                  }
                    
                    # Creating a dictionary of the parameters from the Parameter 
                    # file adjustable by the user
                    clSoftcodedParams = {
                        'fl_vardq'      :fl_vardq,
                        # pyrafBoolean converts the python booleans to pyraf ones
                        'fl_dqprop'     :gemt.pyrafBoolean(fl_dqprop),
                        'combine'       :method,
                                        }
                    # Grabbing the default parameters dictionary and updating 
                    # it with the two above dictionaries
                    clParamsDict = CLDefaultParamsDict('gemcombine', verbose=verbose)
                    clParamsDict.update(clPrimParams)
                    clParamsDict.update(clSoftcodedParams)
                    
                    # Logging the parameters that were not defaults
                    log.fullinfo('\nParameters set automatically:', category='parameters')
                    # Loop through the parameters in the clPrimParams dictionary
                    # and log them
                    gemt.logDictParams(clPrimParams, verbose=verbose)
                    
                    log.fullinfo('\nParameters adjustable by the user:', category='parameters')
                    # Loop through the parameters in the clSoftcodedParams dictionary
                    # and log them
                    gemt.logDictParams(clSoftcodedParams)
                    
                    log.debug('Calling the gemcombine CL script for input list '+
                              clm.inputList())
                    
                    gemini.gemcombine(**clParamsDict)
                    
                    if gemini.gemcombine.status:
                        log.critical('gemcombine failed for inputs '+
                                     clm.inputsAsStr())
                        raise ('gemcombine failed')
                    else:
                        log.status('Exited the gemcombine CL script successfully')
                    
                    
                    # Renaming CL outputs and loading them back into memory 
                    # and cleaning up the intermediate temp files written to disk
                    adOuts = clm.finishCL(combine=True) 
                
                    # There is only one at this point so no need to perform a loop
                    # CLmanager outputs a list always, so take the 0th
                    adOut = adOuts[0]
                    
                    # Adding a GEM-TLM (automatic) and COMBINE time stamps 
                    # to the PHU
                    adOut.historyMark(key='COMBINE',stomp=False)
                    # Updating logger with updated/added time stamps
                    log.fullinfo('*'*50, category='header')
                    log.fullinfo('file = '+adOut.filename, category='header')
                    log.fullinfo('~'*50, category='header')
                    log.fullinfo('PHU keywords updated/added:\n', category='header')
                    log.fullinfo('GEM-TLM = '+adOut.phuGetKeyValue('GEM-TLM'), 
                                 category='header')
                    log.fullinfo('COMBINE = '+adOut.phuGetKeyValue('COMBINE'), 
                                 category='header')
                    log.fullinfo('-'*50, category='header')    
                else:
                    log.critical('One of the inputs has not been prepared,\
                    the combine function can only work on prepared data.')
                    raise('One of the inputs was not prepared')
        else:
            log.critical('The parameter "adIns" must not be None')
            raise('The parameter "adIns" must not be None')
        
        log.status('**FINISHED** the combine function')
        
        # Return the outputs (list or single, matching adIns)
        return adOut
    except:
        raise #('An error occurred while trying to run combine')
                
                
                
def bias_correct(adIns, biases=None,fl_vardq='AUTO', fl_trim=False, fl_over=False, 
                outNames=None, postpend=None, logName='', verbose=1, noLogFile=False):
    """
    This function will subtract the biases from the inputs using the 
    CL script gireduce.
    
    WARNING: The gireduce script used here replaces the previously 
    calculated DQ frames with its own versions.  This may be corrected 
    in the future by replacing the use of the gireduce
    with a Python routine to do the bias subtraction.
    
    NOTE: The inputs to this function MUST be prepared. 

    A string representing the name of the log file to write all log messages to
    can be defined, or a default of 'gemini.log' will be used.  If the file
    all ready exists in the directory you are working in, then this file will 
    have the log messages during this function added to the end of it.
    
    :param adIns: Astrodata inputs to be bias subtracted
    :type adIns: Astrodata objects, either a single or a list of objects
    
    :param biases: The bias(es) to divide the input(s) by.
    :type biases: 
        AstroData objects in a list, or a single instance.
        Note: If there is multiple inputs and one bias provided, then the
        same bias will be applied to all inputs; else the biases   
        list must match the length of the inputs.
    
    :param fl_vardq: Create variance and data quality frames?
    :type fl_vardq: 
         Python boolean (True/False), OR string 'AUTO' to do 
         it automatically if there are VAR and DQ frames in the input(s).
    
    :param fl_trim: Trim the overscan region from the frames?
    :type fl_trim: Python boolean (True/False)
    
    :param fl_over: Subtract the overscan level from the frames?
    :type fl_over: Python boolean (True/False)
    
    :param outNames: filenames of output(s)
    :type outNames: String, either a single or a list of strings of same length as adIns.
    
    :param postpend: 
           string to postpend on the end of the input filenames 
           (or outNames if not None) for the output filenames.
    :type postpend: string
    
    :param logName: Name of the log file, default is 'gemini.log'
    :type logName: string
    
    :param verbose: 
          verbosity setting for the log messages to screen,
          default is 'critical' messages only.
          Note: independent of verbose setting, all messages always go 
          to the logfile if it is not turned off.
    :type verbose: integer from 0-6, 0=nothing to screen, 6=everything to screen
    
    :param noLogFile: A boolean to make it so no log file is created
    :type noLogFile: Python boolean (True/False)
    """
    
    log=gemLog.getGeminiLog(logName=logName, verbose=verbose, noLogFile=noLogFile)

    log.status('**STARTING** the biasCorrect function')
    
    if not isinstance(adIns,list):
        adIns=[adIns]
    
    if (adIns!=None) and (outNames!=None):
        if isinstance(outNames,list):
            if len(adIns)!= len(outNames):
                if postpend==None:
                   raise ('Then length of the inputs, '+str(len(adIns))+
                       ', did not match the length of the outputs, '+
                       str(len(outNames))+
                       ' AND no value of "postpend" was passed in')
    
    try:
        if adIns!=None:
            # Set up counter for looping through outNames list
            count=0
            
            # Creating empty list of ad's to be returned that will be filled below
            if isinstance(adIns,list) and (len(adIns)>1):
                adOuts = []
                
            # loading and bringing the pyraf related modules into the name-space
            pyraf, gemini, yes, no = pyrafLoader()
                
            # Performing work in a loop, so that different biases may be
            # used for each input as gireduce only allows one bias input per run.
            for ad in adIns:
                
                # To clean up log and screen if multiple inputs
                log.fullinfo('+'*50, category='format')    
                
                if isinstance(outNames,list) and (len(outNames)>1):
                    outName = outNames[count]
                elif isinstance(outNames,list) and (len(outNames)==1):
                    outName = outNames[0]
                else:
                    outName = None
                
                # Determining if gireduce should propigate the VAR and DQ frames, if 'AUTO' was chosen 
                if fl_vardq=='AUTO':
                    if isinstance(adIns,list):
                        if adIns[0].countExts('VAR')==adIns[0].countExts('DQ')==adIns[0].countExts('SCI'):
                            fl_vardq=yes
                        else:
                            fl_vardq=no
                    else:
                        if adIns.countExts('VAR')==adIns.countExts('DQ')==adIns.countExts('SCI'):
                            fl_vardq=yes
                        else:
                            fl_vardq=no
                else:
                    if fl_vardq:
                        fl_vardq=yes
                    elif fl_vardq==False:
                        fl_vardq=no
                
                # Preparing input files, lists, parameters... for input to 
                # the CL script
                clm=gemt.CLManager(adIns=ad, outNames=outName, postpend=postpend,  
                                   funcName='biasCorrect', logName=logName, 
                                   verbose=verbose, noLogFile=noLogFile)
                
                # Check the status of the CLManager object, True=continue, False= issue warning
                if clm.status:               
                    
                    # Setting up the processedBias correctly
                    if (isinstance(biases,list)) and (len(biases)>1):
                        processedBias = biases[count]
                    elif (isinstance(biases,list)) and (len(biases)==1):
                        # Not sure if I need this check, but can't hurt
                        processedBias = biases[0]
                    else:
                        processedBias = biases
                        
                    # Parameters set by the gemt.CLManager or the definition of the function 
                    clPrimParams = {
                      'inimages'    :clm.inputsAsStr(),
                      'gp_outpref'  :clm.uniquePrefix(),
                      # This returns a unique/temp log file for IRAF 
                      'logfile'     :clm.logfile(),     
                      'fl_bias'     :yes,
                      # Possibly add this to the params file so the user can override
                      # this input file
                      'bias'        :processedBias,   
                      # This is actually in the default dict but wanted to show it again  
                      'Stdout'      :gemt.IrafStdout(verbose=verbose), 
                      # This is actually in the default dict but wanted to show it again
                      'Stderr'      :gemt.IrafStdout(verbose=verbose), 
                      # This is actually in the default dict but wanted to show it again
                      'verbose'     :yes                
                                  }
                        
                    # Parameters from the Parameter file adjustable by the user
                    clSoftcodedParams = {
                       # pyrafBoolean converts the python booleans to pyraf ones
                       'fl_trim'    :gemt.pyrafBoolean('fl_trim'),
                       'outpref'    :postpend,
                       'fl_over'    :gemt.pyrafBoolean('fl_over'),
                       'fl_vardq'   :gemt.pyrafBoolean('fl_vardq')
                                       }
                    # Grabbing the default params dict and updating it 
                    # with the two above dicts
                    clParamsDict = CLDefaultParamsDict('gireduce', verbose=verbose)
                    clParamsDict.update(clPrimParams)
                    clParamsDict.update(clSoftcodedParams)
                
                    # Logging the parameters that were not defaults
                    log.fullinfo('\nParameters set automatically:', 
                                 category='parameters')
                    # Loop through the parameters in the clPrimParams dictionary
                    # and log them
                    gemt.logDictParams(clPrimParams, verbose=verbose)
                    
                    log.fullinfo('\nParameters adjustable by the user:', 
                                 category='parameters')
                    # Loop through the parameters in the clSoftcodedParams 
                    # dictionary and log them
                    gemt.logDictParams(clSoftcodedParams)
                    
                    log.debug('calling the gireduce CL script for inputs '+
                                                            clm.inputsAsStr())
                
                    gemini.gmos.gireduce(**clParamsDict)
            
                    if gemini.gmos.gireduce.status:
                        log.critical('gireduce failed for inputs '+
                                     clm.inputsAsStr())
                        raise ('gireduce failed')
                    else:
                        log.status('Exited the gireduce CL script successfully')
                        
                    # Renaming CL outputs and loading them back into memory 
                    # and cleaning up the intermediate temp files written to disk
                    adOut = clm.finishCL()
                    
                    # There is only one at this point so no need to perform a loop
                    # CLmanager outputs a list always, so take the 0th
                    adOut = adOut[0]
                    
                    # Varifying gireduce was actually ran on the file
                    # then logging file names of successfully reduced files
                    if adOut.phuGetKeyValue('GIREDUCE'): 
                        log.fullinfo('\nFile '+clm.preCLNames()[0]+
                                     ' was bias subracted successfully')
                        log.fullinfo('New file name is: '+adOut.filename)
      
                    # Updating the GEM-TLM (automatic) and BIASCORR time stamps in 
                    # the PHU
                    adOut.historyMark(key='BIASCORR', stomp=False)  
                    
                    # Reseting the value set by gireduce to just the filename
                    # for clarity
                    adOut.phuSetKeyValue('BIASIM', os.path.basename(processedBias)) 
                    
                    # Updating log with new GEM-TLM value and BIASIM header keys
                    log.fullinfo('*'*50, category='header')
                    log.fullinfo('File = '+adOut.filename, category='header')
                    log.fullinfo('~'*50, category='header')
                    log.fullinfo('PHU keywords updated/added:\n', 'header')
                    log.fullinfo('GEM-TLM = '+adOut.phuGetKeyValue('GEM-TLM'), 
                                 category='header')
                    log.fullinfo('BIASCORR = '+adOut.phuGetKeyValue('BIASCORR'), 
                                 category='header')
                    log.fullinfo('BIASIM = '+adOut.phuGetKeyValue('BIASIM')+'\n', 
                                 category='header')
                    
                    if (isinstance(adIns,list)) and (len(adIns)>1):
                        adOuts.append(adOut)
                    else:
                        adOuts = adOut
               
                    count = count+1
                    
                else:
                    log.critical('One of the inputs has not been prepared,\
                    the combine function can only work on prepared data.')
                    raise('One of the inputs was not prepared')
                
            log.warning('The CL script gireduce REPLACED the previously '+
                        'calculated DQ frames')
        
        else:
            log.critical('The parameter "adIns" must not be None')
            raise('The parameter "adIns" must not be None')
        
        log.status('**FINISHED** the biasCorrect function')
        
        # Return the outputs (list or single, matching adIns)
        return adOuts
    except:
        raise #('An error occurred while trying to run biasCorrect')    
                
                
                
def mosaic_detectors(adIns, fl_paste=False, interp_function='linear', fl_vardq='AUTO', 
                outNames=None, postpend=None, logName='', verbose=1, noLogFile=False):
    """
    This function will mosaic the SCI frames of the input images, 
    along with the VAR and DQ frames if they exist.  
    
    WARNING: The gmosaic script used here replaces the previously 
    calculated DQ frames with its own versions.  This may be corrected 
    in the future by replacing the use of the gmosaic
    with a Python routine to do the frame mosaicing.
    
    NOTE: The inputs to this function MUST be prepared. 

    A string representing the name of the log file to write all log messages to
    can be defined, or a default of 'gemini.log' will be used.  If the file
    all ready exists in the directory you are working in, then this file will 
    have the log messages during this function added to the end of it.
    
    :param adIns: Astrodata inputs to mosaic the extensions of
    :type adIns: Astrodata objects, either a single or a list of objects
    
    :param fl_paste: Paste images instead of mosaic?
    :type fl_paste: Python boolean (True/False)
    
    :param interp_function: type of interpolation algorithm to use for between the chip gaps.
    :type interp_function: string, options: 'linear', 'nearest', 'poly3', 
                           'poly5', 'spine3', 'sinc'.
    
    :param fl_vardq: Also mosaic VAR and DQ frames?
    :type fl_vardq: Python boolean (True/False), OR string 'AUTO' to do 
                    it automatically if there are VAR and DQ frames in the inputs.
                    NOTE: 'AUTO' uses the first input to determine if VAR and DQ frames exist, 
                    so, if the first does, then the rest MUST also have them as well.
    
    :param outNames: filenames of output(s)
    :type outNames: String, either a single or a list of strings of same length as adIns.
    
    :param postpend: string to postpend on the end of the input filenames 
                    (or outNames if not None) for the output filenames.
    :type postpend: string
    
    :param logName: Name of the log file, default is 'gemini.log'
    :type logName: string
    
    :param verbose: verbosity setting for the log messages to screen,
                    default is 'critical' messages only.
                    Note: independent of verbose setting, all messages always go 
                    to the logfile if it is not turned off.
    :type verbose: integer from 0-6, 0=nothing to screen, 6=everything to screen
    
    :param noLogFile: A boolean to make it so no log file is created
    :type noLogFile: Python boolean (True/False)
    """
    
    log=gemLog.getGeminiLog(logName=logName, verbose=verbose, noLogFile=noLogFile)

    log.status('**STARTING** the mosaicDetectors function')
    
    if not isinstance(adIns,list):
        adIns=[adIns]
    
    if (adIns!=None) and (outNames!=None):
        if isinstance(outNames,list):
            if len(adIns)!= len(outNames):
                if postpend==None:
                   raise ('Then length of the inputs, '+str(len(adIns))+
                       ', did not match the length of the outputs, '+
                       str(len(outNames))+
                       ' AND no value of "postpend" was passed in')
    
    try:
        if adIns!=None: 
            # loading and bringing the pyraf related modules into the name-space
            pyraf, gemini, yes, no = pyrafLoader()  
                
            # Determining if gmosaic should propigate the VAR and DQ frames, if 'AUTO' was chosen 
            if fl_vardq=='AUTO':
                if isinstance(adIns,list):
                    if adIns[0].countExts('VAR')==adIns[0].countExts('DQ')==adIns[0].countExts('SCI'):
                        fl_vardq=yes
                    else:
                        fl_vardq=no
                else:
                    if adIns.countExts('VAR')==adIns.countExts('DQ')==adIns.countExts('SCI'):
                        fl_vardq=yes
                    else:
                        fl_vardq=no
            else:
                if fl_vardq:
                    fl_vardq=yes
                elif fl_vardq==False:
                    fl_vardq=no
            
            # To clean up log and screen if multiple inputs
            log.fullinfo('+'*50, category='format')    
            
            # Preparing input files, lists, parameters... for input to 
            # the CL script
            clm=gemt.CLManager(adIns=adIns, outNames=outNames, postpend=postpend, 
                               funcName='mosaicDetectors', logName=logName, 
                               verbose=verbose, noLogFile=noLogFile)
            
            # Check the status of the CLManager object, True=continue, False= issue warning
            if clm.status: 
                
                # Parameters set by the gemt.CLManager or the definition of the prim 
                clPrimParams = {
                  # Retrieving the inputs as a string of filenames
                  'inimages'    :clm.inputsAsStr(),
                  # Setting the value of FL_vardq set above
                  'fl_vardq'    :fl_vardq,
                  # This returns a unique/temp log file for IRAF 
                  'logfile'     :clm.logfile(),
                  # This is actually in the default dict but wanted to show it again     
                  'Stdout'      :gemt.IrafStdout(verbose=verbose), 
                  # This is actually in the default dict but wanted to show it again
                  'Stderr'      :gemt.IrafStdout(verbose=verbose), 
                  # This is actually in the default dict but wanted to show it again
                  'verbose'     :yes                
                              }
                # Parameters from the Parameter file adjustable by the user
                clSoftcodedParams = {
                  # pyrafBoolean converts the python booleans to pyraf ones
                  'fl_paste'    :gemt.pyrafBoolean(fl_paste),
                  'outpref'     :postpend,
                  'geointer'    :interp_function,
                                  }
                # Grabbing the default params dict and updating it with 
                # the two above dicts
                clParamsDict = CLDefaultParamsDict('gmosaic', verbose=verbose)
                clParamsDict.update(clPrimParams)
                clParamsDict.update(clSoftcodedParams)      
                    
                # Logging the parameters that were not defaults
                log.fullinfo('\nParameters set automatically:', 
                             category='parameters')
                # Loop through the parameters in the clPrimParams dictionary
                # and log them
                gemt.logDictParams(clPrimParams, verbose=verbose)
                
                log.fullinfo('\nParameters adjustable by the user:', 
                             category='parameters')
                # Loop through the parameters in the clSoftcodedParams 
                # dictionary and log them
                gemt.logDictParams(clSoftcodedParams)
                
                log.debug('calling the gmosaic CL script for inputs '+
                                                        clm.inputsAsStr())
            
                gemini.gmos.gmosaic(**clParamsDict)
        
                if gemini.gmos.gmosaic.status:
                    log.critical('gireduce failed for inputs '+
                                 clm.inputsAsStr())
                    raise ('gmosaic failed')
                else:
                    log.status('Exited the gmosaic CL script successfully')    
                    
                    
                # Renaming CL outputs and loading them back into memory 
                # and cleaning up the intermediate temp files written to disk
                adOuts = clm.finishCL()    
                    
                # Wrap up logging
                i=0
                for ad in adOuts:
                    log.fullinfo('-'*50, category='header')
                    
                    # Varifying gireduce was actually ran on the file
                    # then logging file names of successfully reduced files
                    if ad.phuGetKeyValue('GMOSAIC'): 
                        log.fullinfo('\nFile '+clm.preCLNames()[i]+\
                                     ' mosaiced successfully')
                        log.fullinfo('New file name is: '+ad.filename)
                    i=i+1
                    # Updating GEM-TLM (automatic) and MOSAIC time stamps to the PHU
                    ad.historyMark(key='MOSAIC', stomp=False)  
                    
                    # Updating logger with new GEM-TLM value
                    log.fullinfo('*'*50, category='header')
                    log.fullinfo('File = '+ad.filename, category='header')
                    log.fullinfo('~'*50, category='header')
                    log.fullinfo('PHU keywords updated/added:\n', category='header')
                    log.fullinfo('GEM-TLM = '+ad.phuGetKeyValue('GEM-TLM'), 
                                 category='header')
                    log.fullinfo('MOSAIC = '+ad.phuGetKeyValue('MOSAIC')+'\n', 
                                 category='header')    
                
            else:
                    log.critical('One of the inputs has not been prepared,\
                    the mosaicDetectors function can only work on prepared data.')
                    raise('One of the inputs was not prepared')
                
        else:
            log.critical('The parameter "adIns" must not be None')
            raise('The parameter "adIns" must not be None')
        
        log.status('**FINISHED** the mosaicDetectors function')
        
        # Return the outputs (list or single, matching adIns)
        return adOuts
    except:
        raise ('An error occurred while trying to run mosaicDetectors') 
                
                
def normalize_flat(adIns, fl_trim=False, fl_over=False,fl_vardq='AUTO', 
                outNames=None, postpend=None, logName='', verbose=1, noLogFile=False):
    """
    This function will combine the input flats (adIns) and then normalize them 
    using the CL script giflat.
    
    WARNING: The giflat script used here replaces the previously 
    calculated DQ frames with its own versions.  This may be corrected 
    in the future by replacing the use of the giflat
    with a Python routine to do the flat normalizing.
    
    NOTE: The inputs to this function MUST be prepared. 

    A string representing the name of the log file to write all log messages to
    can be defined, or a default of 'gemini.log' will be used.  If the file
    all ready exists in the directory you are working in, then this file will 
    have the log messages during this function added to the end of it.
    
    :param adIns: Astrodata input flat(s) to be combined and normalized
    :type adIns: Astrodata objects, either a single or a list of objects
    
    :param fl_trim: Trim the overscan region from the frames?
    :type fl_trim: Python boolean (True/False)
    
    :param fl_over: Subtract the overscan level from the frames?
    :type fl_over: Python boolean (True/False)
    
    :param fl_vardq: Create variance and data quality frames?
    :type fl_vardq: Python boolean (True/False), OR string 'AUTO' to do 
                    it automatically if there are VAR and DQ frames in the inputs.
                    NOTE: 'AUTO' uses the first input to determine if VAR and DQ frames exist, 
                    so, if the first does, then the rest MUST also have them as well.
    
        
    :param outNames: filenames of output(s)
    :type outNames: String, either a single or a list of strings of same length as adIns.
    
    :param postpend:
            string to postpend on the end of the input filenames 
            (or outNames if not None) for the output filenames.
    :type postpend: string
    
    :param logName: Name of the log file, default is 'gemini.log'
    :type logName: string
    
    :param verbose: verbosity setting for the log messages to screen,
                    default is 'critical' messages only.
                    Note: independent of verbose setting, all messages always go 
                    to the logfile if it is not turned off.
    :type verbose: integer from 0-6, 0=nothing to screen, 6=everything to screen
    
    :param noLogFile: A boolean to make it so no log file is created
    :type noLogFile: Python boolean (True/False)
    """
    
    log=gemLog.getGeminiLog(logName=logName, verbose=verbose, noLogFile=noLogFile)

    log.status('**STARTING** the normalizeFlat function')
    
    if not isinstance(adIns,list):
        adIns=[adIns]
    
    if (adIns!=None) and (outNames!=None):
        if isinstance(outNames,list):
            if len(adIns)!= len(outNames):
                if postpend==None:
                   raise ('Then length of the inputs, '+str(len(adIns))+
                       ', did not match the length of the outputs, '+
                       str(len(outNames))+
                       ' AND no value of "postpend" was passed in')
    
    try:
        if adIns!=None: 
            # loading and bringing the pyraf related modules into the name-space
            pyraf, gemini, yes, no = pyrafLoader()  
                
            # Determining if gmosaic should propigate the VAR and DQ frames, if 'AUTO' was chosen 
            if fl_vardq=='AUTO':
                if isinstance(adIns,list):
                    if adIns[0].countExts('VAR')==adIns[0].countExts('DQ')==adIns[0].countExts('SCI'):
                        fl_vardq=yes
                    else:
                        fl_vardq=no
                else:
                    if adIns.countExts('VAR')==adIns.countExts('DQ')==adIns.countExts('SCI'):
                        fl_vardq=yes
                    else:
                        fl_vardq=no
            else:
                if fl_vardq:
                    fl_vardq=yes
                elif fl_vardq==False:
                    fl_vardq=no
            
            # To clean up log and screen if multiple inputs
            log.fullinfo('+'*50, category='format')    
            
            # Preparing input files, lists, parameters... for input to 
            # the CL script
            clm=gemt.CLManager(adIns=adIns, outNames=outNames, postpend=postpend, 
                               funcName='normalizeFlat', logName=logName, 
                               verbose=verbose, noLogFile=noLogFile)
            
            # Check the status of the CLManager object, True=continue, False= issue warning
            if clm.status:                 
                
                # Creating a dictionary of the parameters set by the gemt.CLManager 
                # or the definition of the function 
                clPrimParams = {
                  'inflats'     :clm.inputList(),
                  # Maybe allow the user to override this in the future
                  'outflat'     :clm.combineOutname(), 
                  # This returns a unique/temp log file for IRAF  
                  'logfile'     :clm.logfile(),         
                  # This is actually in the default dict but wanted to show it again
                  'Stdout'      :gemt.IrafStdout(verbose=verbose),   
                  # This is actually in the default dict but wanted to show it again  
                  'Stderr'      :gemt.IrafStdout(verbose=verbose), 
                  # This is actually in the default dict but wanted to show it again    
                  'verbose'     :yes                    
                              }
                # Creating a dictionary of the parameters from the function call 
                # adjustable by the user
                clSoftcodedParams = {
                   'fl_vardq'   :fl_vardq,
                   'fl_over'    :gemt.pyrafBoolean(fl_over),
                   'fl_trim'    :gemt.pyrafBoolean(fl_trim)
                                   }
                # Grabbing the default params dict and updating it 
                # with the two above dicts
                clParamsDict = CLDefaultParamsDict('giflat', verbose=verbose)
                clParamsDict.update(clPrimParams)
                clParamsDict.update(clSoftcodedParams)
                
                # Logging the parameters that were not defaults
                log.fullinfo('\nParameters set automatically:', 
                             category='parameters')
                # Loop through the parameters in the clPrimParams dictionary
                # and log them
                gemt.logDictParams(clPrimParams, verbose=verbose)
                
                log.fullinfo('\nParameters adjustable by the user:', 
                             category='parameters')
                # Loop through the parameters in the clSoftcodedParams 
                # dictionary and log them
                gemt.logDictParams(clSoftcodedParams)
                
                log.debug('Calling the giflat CL script for inputs list '+
                      clm.inputList())
            
                gemini.giflat(**clParamsDict)
                
                if gemini.giflat.status:
                    log.critical('giflat failed for inputs '+
                                 clm.inputsAsStr())
                    raise ('giflat failed')
                else:
                    log.status('Exited the giflat CL script successfully')
                
                # Renaming CL outputs and loading them back into memory 
                # and cleaning up the intermediate temp files written to disk
                adOuts = clm.finishCL(combine=True) 
            
                # There is only one at this point so no need to perform a loop
                # CLmanager outputs a list always, so take the 0th
                adOut = adOuts[0]
                
                # Adding GEM-TLM (automatic) and GIFLAT time stamps to the PHU
                adOut.historyMark(key='GIFLAT', stomp=False)
                
                # Updating log with new GEM-TLM and GIFLAT time stamps
                log.fullinfo('*'*50, category='header')
                log.fullinfo('File = '+adOut.filename, category='header')
                log.fullinfo('~'*50, category='header')
                log.fullinfo('PHU keywords updated/added:\n', 'header')
                log.fullinfo('GEM-TLM = '+adOut.phuGetKeyValue('GEM-TLM'), 
                             category='header')
                log.fullinfo('GIFLAT = '+adOut.phuGetKeyValue('GIFLAT'), 
                             category='header')
                log.fullinfo('-'*50, category='header')     
                
            else:
                log.critical('One of the inputs has not been prepared,\
                the normalizeFlat function can only work on prepared data.')
                raise('One of the inputs was not prepared')
                
        else:
            log.critical('The parameter "adIns" must not be None')
            raise('The parameter "adIns" must not be None')
        
        log.status('**FINISHED** the normalizeFlat function')
        
        # Return the outputs (list or single, matching adIns)
        return adOuts
    except:
        raise ('An error occurred while trying to run normalizeFlat') 
                
                       
                
                
                
                
                
                

