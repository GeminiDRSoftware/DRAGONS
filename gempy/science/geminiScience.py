
#Author: Kyle Mede, January 2011
#For now, this module is to hold the code which performs the actual work of the 
#primitives that is considered generic enough to be at the 'gemini' level of
#the hierarchy tree.

import os

import pyfits as pf
import numpy as np
from copy import deepcopy
import time
from datetime import datetime
from astrodata.adutils import gemLog
from astrodata.AstroData import AstroData
from astrodata.adutils.gemutil import pyrafLoader
from astrodata.Errors import ScienceError
from gempy.instruments import geminiTools  as gemt
from gempy.instruments.geminiCLParDicts import CLDefaultParamsDict

def add_bpm(adInputs=None, BPMs=None, matchSize=False, outNames=None, suffix=None, logName='', 
                                                    logLevel=1, noLogFile=False):
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
          
    :param adInputs: Astrodata inputs to be converted to Electron pixel units
    :type adInputs: Astrodata objects, either a single or a list of objects
    
    :param BPMs: The BPM(s) to be added to the input(s).
    :type BPMs: 
       AstroData objects in a list, or a single instance.
       Note: If there is multiple inputs and one BPM provided, then the
       same BPM will be applied to all inputs; else the BPMs list  
       must match the length of the inputs.
           
    :param matchSize: A flag to use zeros and the key 'DETSEC' of the 'SCI'
                      extensions to match the size of the BPM arrays to those of
                      for the 'SCI' data arrays.
    :type matchSize: Python boolean (True/False). Default: False.
                      
    :param outNames: filenames of output(s)
    :type outNames: String, either a single or a list of strings of same length as adInputs.
    
    :param suffix: 
         string to add on the end of the input filenames 
         (or outNames if not None) for the output filenames.
    :type suffix: string
    
    :param logName: Name of the log file, default is 'gemini.log'
    :type logName: string
    
    :param logLevel: 
          verbosity setting for the log messages to screen,
          default is 'critical' messages only.
          Note: independent of logLevel setting, all messages always go 
          to the logfile if it is not turned off.
    :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to screen.
                    OR the message level as a string (ie. 'critical', 'status', 
                    'fullinfo'...)
    
    :param noLogFile: A boolean to make it so no log file is created
    :type noLogFile: Python boolean (True/False)
    """
    
    log=gemLog.getGeminiLog(logName=logName, logLevel=logLevel, noLogFile=noLogFile)
        
    log.status('**STARTING** the add_bpm function')
    
    if not isinstance(adInputs,list):
        adInputs=[adInputs]
    
    if (adInputs!=None) and (outNames!=None):
        if isinstance(outNames,list):
            if len(adInputs)!= len(outNames):
                if suffix==None:
                   raise ScienceError('Then length of the inputs, '+str(len(adInputs))+
                       ', did not match the length of the outputs, '+
                       str(len(outNames))+
                       ' AND no value of "suffix" was passed in')
        if isInstance(outNames,str) and len(adInputs)>1:
            if suffix==None:
                   raise ScienceError('Then length of the inputs, '+str(len(adInputs))+
                       ', did not match the length of the outputs, '+
                       str(len(outNames))+
                       ' AND no value of "suffix" was passed in')
                   
    if BPMs==None:
        raise ScienceError('There must be at least one BPM provided, the "BPMs" parameter must not be None.')
                   
    try:
        # Ensure there are inputs to work on and BPMs to add to the inputs
        if (adInputs!=None) and (BPMs!=None):
            # Set up counter for looping through outNames/BPMs lists
            count=0
            
            # Creating empty list of ad's to be returned that will be filled below
            if len(adInputs)>1:
                adOutputs=[]
            
            # Do the work on each ad in the inputs
            for ad in adInputs:
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
                
                # Updating the file name with the suffix for this
                # function and then reporting the new file 
                if suffix!=None:
                    log.debug('Calling gemt.fileNameUpdater on '+adOut.filename)
                    if outNames!=None:
                        adOut.filename = gemt.fileNameUpdater(adIn=adOut, 
                                                              infilename=outNames[count],
                                                          suffix=suffix, 
                                                          strip=False, logLevel=logLevel)
                    else:
                        adOut.filename = gemt.fileNameUpdater(adIn=adOut, 
                                                          suffix=suffix, 
                                                          strip=False, logLevel=logLevel)
                elif suffix==None:
                    if outNames!=None:
                        if len(outNames)>1: 
                            adOut.filename = outNames[count]
                        else:
                            adOut.filename = outNames
                    else:
                        raise ScienceError('outNames and suffix parameters can not BOTH\
                                                                    be None')
                        
                log.status('File name updated to '+adOut.filename)
            
                if (isinstance(adInputs,list)) and (len(adInputs)>1):
                    adOutputs.append(adOut)
                else:
                    adOutputs = adOut

                count=count+1
        else:
            raise ScienceError('The parameter "adInputs" must not be None')
        
        log.status('**FINISHED** the add_bpm function')
        # Return the outputs (list or single, matching adInputs)
        return adOutputs
    except:
        raise ScienceError('An error occurred while trying to run add_bpm')
    
def add_dq(adInputs, fl_nonlinear=True, fl_saturated=True,outNames=None, suffix=None, 
                                    logName='', logLevel=1, noLogFile=False):
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
    
    :param adInputs: Astrodata inputs to have DQ extensions added to
    :type adInputs: Astrodata objects, either a single or a list of objects
    
    :param fl_nonlinear: Flag to turn checking for nonlinear pixels on/off
    :type fl_nonLinear: Python boolean (True/False), default is True
    
    :param fl_saturated: Flag to turn checking for saturated pixels on/off
    :type fl_saturated: Python boolean (True/False), default is True
    
    :param outNames: filenames of output(s)
    :type outNames: String, either a single or a list of strings of same length as adInputs.
    
    :param suffix: 
       string to add on the end of the input filenames 
       (or outNames if not None) for the output filenames.
    :type suffix: string
    
    :param logName: Name of the log file, default is 'gemini.log'
    :type logName: string
    
    :param logLevel: 
         verbosity setting for the log messages to screen,
         default is 'critical' messages only.
         Note: independent of logLevel setting, all messages always go 
         to the logfile if it is not turned off.
    :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to screen.
                    OR the message level as a string (ie. 'critical', 'status', 
                    'fullinfo'...)
    
    :param noLogFile: A boolean to make it so no log file is created
    :type noLogFile: Python boolean (True/False)
    """
    
    log=gemLog.getGeminiLog(logName=logName, logLevel=logLevel, noLogFile=noLogFile)
        
    log.status('**STARTING** the add_dq function')
    
    if not isinstance(adInputs,list):
        adInputs=[adInputs]
    
    if (adInputs!=None) and (outNames!=None):
        if isinstance(outNames,list):
            if len(adInputs)!= len(outNames):
                if suffix==None:
                   raise ScienceError('Then length of the inputs, '+str(len(adInputs))+
                       ', did not match the length of the outputs, '+
                       str(len(outNames))+
                       ' AND no value of "suffix" was passed in')
        if isInstance(outNames,str) and len(adInputs)>1:
            if suffix==None:
                   raise ScienceError('Then length of the inputs, '+str(len(adInputs))+
                       ', did not match the length of the outputs, '+
                       str(len(outNames))+
                       ' AND no value of "suffix" was passed in')
    
    try:
        if adInputs!=None:
            # Set up counter for looping through outNames list
            count=0
            
            # Creating empty list of ad's to be returned that will be filled below
            if len(adInputs)>1:
                adOutputs=[]
            
            # Loop through the inputs to perform the non-linear and saturated
            # pixel searches of the SCI frames to update the BPM frames into
            # full DQ frames. 
            for ad in adInputs:                
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
                
                # Updating the file name with the suffix for this
                # function and then reporting the new file 
                if suffix!=None:
                    log.debug('Calling gemt.fileNameUpdater on '+adOut.filename)
                    if outNames!=None:
                        adOut.filename = gemt.fileNameUpdater(adIn=adOut, 
                                                              infilename=outNames[count],
                                                          suffix=suffix, 
                                                          strip=False, logLevel=logLevel)
                    else:
                        adOut.filename = gemt.fileNameUpdater(adIn=adOut, 
                                                          suffix=suffix, 
                                                          strip=False, logLevel=logLevel)
                elif suffix==None:
                    if outNames!=None:
                        if len(outNames)>1: 
                            adOut.filename = outNames[count]
                        else:
                            adOut.filename = outNames
                    else:
                        raise ScienceError('outNames and suffix parameters can not BOTH\
                                                                    be None')
                        
                log.status('File name updated to '+adOut.filename)
            
                if (isinstance(adInputs,list)) and (len(adInputs)>1):
                    adOutputs.append(adOut)
                else:
                    adOutputs = adOut

                count=count+1
        else:
            raise ScienceError('The parameter "adInputs" must not be None')
        
        log.status('**FINISHED** the add_dq function')
        # Return the outputs (list or single, matching adInputs)
        return adOutputs
    except:
        raise ScienceError('An error occurred while trying to run add_dq')

def add_var(adInputs, outNames=None, suffix=None, logName='', logLevel=1, 
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
    
    :param adInputs: Astrodata inputs to have DQ extensions added to
    :type adInputs: Astrodata objects, either a single or a list of objects
    
    :param outNames: filenames of output(s)
    :type outNames: String, either a single or a list of strings of same length as adInputs.
    
    :param suffix: 
        string to add on the end of the input filenames 
        (or outNames if not None) for the output filenames.
    :type suffix: string
    
    :param logName: Name of the log file, default is 'gemini.log'
    :type logName: string
    
    :param logLevel: 
        verbosity setting for the log messages to screen,
        default is 'critical' messages only.
        Note: independent of logLevel setting, all messages always go 
        to the logfile if it is not turned off.
    :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to screen.
                    OR the message level as a string (ie. 'critical', 'status', 
                    'fullinfo'...)
    
    :param noLogFile: A boolean to make it so no log file is created
    :type noLogFile: Python boolean (True/False)
    """
    
    log=gemLog.getGeminiLog(logName=logName, logLevel=logLevel, noLogFile=noLogFile)
        
    log.status('**STARTING** the add_var function')
    
    if not isinstance(adInputs,list):
        adInputs=[adInputs]
    
    if (adInputs!=None) and (outNames!=None):
        if isinstance(outNames,list):
            if len(adInputs)!= len(outNames):
                if suffix==None:
                   raise ScienceError('Then length of the inputs, '+str(len(adInputs))+
                       ', did not match the length of the outputs, '+
                       str(len(outNames))+
                       ' AND no value of "suffix" was passed in')
        if isInstance(outNames,str) and len(adInputs)>1:
            if suffix==None:
                   raise ScienceError('Then length of the inputs, '+str(len(adInputs))+
                       ', did not match the length of the outputs, '+
                       str(len(outNames))+
                       ' AND no value of "suffix" was passed in')
    
    try:
        if adInputs!=None:
            # Set up counter for looping through outNames list
            count=0
            
            # Creating empty list of ad's to be returned that will be filled below
            if len(adInputs)>1:
                adOutputs=[]
            
            # Loop through the inputs to perform the non-linear and saturated
            # pixel searches of the SCI frames to update the BPM frames into
            # full DQ frames. 
            for ad in adInputs:   
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
                
                # Updating the file name with the suffix for this
                # function and then reporting the new file 
                if suffix!=None:
                    log.debug('Calling gemt.fileNameUpdater on '+adOut.filename)
                    if outNames!=None:
                        adOut.filename = gemt.fileNameUpdater(adIn=adOut, 
                                                              infilename=outNames[count],
                                                          suffix=suffix, 
                                                          strip=False, logLevel=logLevel)
                    else:
                        adOut.filename = gemt.fileNameUpdater(adIn=adOut, 
                                                          suffix=suffix, 
                                                          strip=False, logLevel=logLevel)
                elif suffix==None:
                    if outNames!=None:
                        if len(outNames)>1: 
                            adOut.filename = outNames[count]
                        else:
                            adOut.filename = outNames
                    else:
                        raise ScienceError('outNames and suffix parameters can not BOTH\
                                                                    be None')
                        
                log.status('File name updated to '+adOut.filename)
            
                if (isinstance(adInputs,list)) and (len(adInputs)>1):
                    adOutputs.append(adOut)
                else:
                    adOutputs = adOut

                count=count+1
        else:
            raise ScienceError('The parameter "adInputs" must not be None')
        
        log.status('**FINISHED** the add_var function')
        # Return the outputs (list or single, matching adInputs)
        return adOutputs
    except:
        raise ScienceError('An error occurred while trying to run add_var')

def adu_to_electrons(adInputs=None, outNames=None, suffix=None, logName='', 
                                                    logLevel=1, noLogFile=False):
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
          
    :param adInputs: Astrodata inputs to be converted to Electron pixel units
    :type adInputs: Astrodata objects, either a single or a list of objects
    
    :param outNames: filenames of output(s)
    :type outNames: 
        String, either a single or a list of strings of same length as adInputs.
    
    :param suffix: 
        string to add on the end of the input filenames 
        (or outNames if not None) for the output filenames.
    :type suffix: string
    
    :param logName: Name of the log file, default is 'gemini.log'
    :type logName: string
    
    :param logLevel: 
         verbosity setting for the log messages to screen,
         default is 'critical' messages only.
         Note: independent of logLevel setting, all messages always go 
         to the logfile if it is not turned off.
    :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to screen.
                    OR the message level as a string (ie. 'critical', 'status', 
                    'fullinfo'...)
    
    :param noLogFile: A boolean to make it so no log file is created
    :type noLogFile: Python boolean (True/False)
    """
    
    log=gemLog.getGeminiLog(logName=logName, logLevel=logLevel, noLogFile=noLogFile)
        
    log.status('**STARTING** the adu_to_electrons function')
    
    if not isinstance(adInputs,list):
        adInputs=[adInputs]
    
    if (adInputs!=None) and (outNames!=None):
        if isinstance(outNames,list):
            if len(adInputs)!= len(outNames):
                if suffix==None:
                   raise ScienceError('Then length of the inputs, '+str(len(adInputs))+
                       ', did not match the length of the outputs, '+
                       str(len(outNames))+
                       ' AND no value of "suffix" was passed in')
        if isInstance(outNames,str) and len(adInputs)>1:
            if suffix==None:
                   raise ScienceError('Then length of the inputs, '+str(len(adInputs))+
                       ', did not match the length of the outputs, '+
                       str(len(outNames))+
                       ' AND no value of "suffix" was passed in')
    
    try:
        if adInputs!=None:
            # Set up counter for looping through outNames list
            count=0
            
            # Creating empty list of ad's to be returned that will be filled below
            if len(adInputs)>1:
                adOutputs=[]
            
            # Do the work on each ad in the inputs
            for ad in adInputs:
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
                
                # Updating the file name with the suffix for this
                # function and then reporting the new file 
                if suffix!=None:
                    log.debug('Calling gemt.fileNameUpdater on '+adOut.filename)
                    if outNames!=None:
                        adOut.filename = gemt.fileNameUpdater(adIn=adOut, 
                                                              infilename=outNames[count],
                                                          suffix=suffix, 
                                                          strip=False, logLevel=logLevel)
                    else:
                        adOut.filename = gemt.fileNameUpdater(adIn=adOut, 
                                                          suffix=suffix, 
                                                          strip=False, logLevel=logLevel)
                elif suffix==None:
                    if outNames!=None:
                        if len(outNames)>1: 
                            adOut.filename = outNames[count]
                        else:
                            adOut.filename = outNames
                    else:
                        raise ScienceError('outNames and suffix parameters can not BOTH\
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
                
                if (isinstance(adInputs,list)) and (len(adInputs)>1):
                    adOutputs.append(adOut)
                else:
                    adOutputs = adOut

                count=count+1
        else:
            raise ScienceError('The parameter "adInputs" must not be None')
        log.status('**FINISHED** the adu_to_electrons function')
        # Return the outputs (list or single, matching adInputs)
        return adOutputs
    except:
        raise ScienceError('An error occurred while trying to run adu_to_electrons')

def bias_correct(adInputs, biases=None,fl_vardq='AUTO', fl_trim=False, fl_over=False, 
                outNames=None, suffix=None, logName='', logLevel=1, noLogFile=False):
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
    
    :param adInputs: Astrodata inputs to be bias subtracted
    :type adInputs: Astrodata objects, either a single or a list of objects
    
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
    :type outNames: String, either a single or a list of strings of same length as adInputs.
    
    :param suffix: 
           string to add on the end of the input filenames 
           (or outNames if not None) for the output filenames.
    :type suffix: string
    
    :param logName: Name of the log file, default is 'gemini.log'
    :type logName: string
    
    :param logLevel: 
          verbosity setting for the log messages to screen,
          default is 'critical' messages only.
          Note: independent of logLevel setting, all messages always go 
          to the logfile if it is not turned off.
    :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to screen.
                    OR the message level as a string (ie. 'critical', 'status', 
                    'fullinfo'...)
    
    :param noLogFile: A boolean to make it so no log file is created
    :type noLogFile: Python boolean (True/False)
    """
    
    log=gemLog.getGeminiLog(logName=logName, logLevel=logLevel, noLogFile=noLogFile)

    log.status('**STARTING** the bias_correct function')
    
    if not isinstance(adInputs,list):
        adInputs=[adInputs]
    
    if (adInputs!=None) and (outNames!=None):
        if isinstance(outNames,list):
            if len(adInputs)!= len(outNames):
                if suffix==None:
                   raise ScienceError('Then length of the inputs, '+str(len(adInputs))+
                       ', did not match the length of the outputs, '+
                       str(len(outNames))+
                       ' AND no value of "suffix" was passed in')
        if isInstance(outNames,str) and len(adInputs)>1:
            if suffix==None:
                   raise ScienceError('Then length of the inputs, '+str(len(adInputs))+
                       ', did not match the length of the outputs, '+
                       str(len(outNames))+
                       ' AND no value of "suffix" was passed in')
    
    try:
        if adInputs!=None:
            # Set up counter for looping through outNames list
            count=0
            
            # Creating empty list of ad's to be returned that will be filled below
            if isinstance(adInputs,list) and (len(adInputs)>1):
                adOutputs = []
                
            # loading and bringing the pyraf related modules into the name-space
            pyraf, gemini, yes, no = pyrafLoader()
                
            # Performing work in a loop, so that different biases may be
            # used for each input as gireduce only allows one bias input per run.
            for ad in adInputs:
                
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
                    if isinstance(adInputs,list):
                        if adInputs[0].countExts('VAR')==adInputs[0].countExts('DQ')==adInputs[0].countExts('SCI'):
                            fl_vardq=yes
                        else:
                            fl_vardq=no
                    else:
                        if adInputs.countExts('VAR')==adInputs.countExts('DQ')==adInputs.countExts('SCI'):
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
                clm=gemt.CLManager(imageIns=ad, imageOutsNames=outName, 
                                   suffix=suffix,  funcName='biasCorrect', 
                                   logName=logName, logLevel=logLevel, 
                                   noLogFile=noLogFile)
                
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
                      'inimages'    :clm.imageInsFiles(type='string'),
                      'gp_outpref'  :clm.prefix,
                      'outimages'   :clm.imageOutsFiles(type='string'),
                      # This returns a unique/temp log file for IRAF 
                      'logfile'     :clm.templog.name,     
                      'fl_bias'     :yes,
                      # Possibly add this to the params file so the user can override
                      # this input file
                      'bias'        :processedBias,   
                      # This is actually in the default dict but wanted to show it again  
                      'Stdout'      :gemt.IrafStdout(logLevel=logLevel), 
                      # This is actually in the default dict but wanted to show it again
                      'Stderr'      :gemt.IrafStdout(logLevel=logLevel), 
                      # This is actually in the default dict but wanted to show it again
                      'verbose'     :yes                
                                  }
                        
                    # Parameters from the Parameter file adjustable by the user
                    clSoftcodedParams = {
                       # pyrafBoolean converts the python booleans to pyraf ones
                       'fl_trim'    :gemt.pyrafBoolean('fl_trim'),
                       'outpref'    :suffix,
                       'fl_over'    :gemt.pyrafBoolean('fl_over'),
                       'fl_vardq'   :gemt.pyrafBoolean('fl_vardq')
                                       }
                    # Grabbing the default params dict and updating it 
                    # with the two above dicts
                    clParamsDict = CLDefaultParamsDict('gireduce', logLevel=logLevel)
                    clParamsDict.update(clPrimParams)
                    clParamsDict.update(clSoftcodedParams)
                
                    # Logging the parameters that were not defaults
                    log.fullinfo('\nParameters set automatically:', 
                                 category='parameters')
                    # Loop through the parameters in the clPrimParams dictionary
                    # and log them
                    gemt.logDictParams(clPrimParams, logLevel=logLevel)
                    
                    log.fullinfo('\nParameters adjustable by the user:', 
                                 category='parameters')
                    # Loop through the parameters in the clSoftcodedParams 
                    # dictionary and log them
                    gemt.logDictParams(clSoftcodedParams,logLevel=logLevel)
                    
                    log.debug('calling the gireduce CL script for inputs '+
                                                            clm.imageInsFiles(type='string'))
                
                    gemini.gmos.gireduce(**clParamsDict)
            
                    if gemini.gmos.gireduce.status:
                        log.critical('gireduce failed for inputs '+
                                     clm.imageInsFiles(type='string'))
                        raise ScienceError('gireduce failed')
                    else:
                        log.status('Exited the gireduce CL script successfully')
                        
                    # Renaming CL outputs and loading them back into memory 
                    # and cleaning up the intermediate temp files written to disk
                    # refOuts and arrayOuts are None here
                    imageOuts, refOuts, arrayOuts = clm.finishCL() 
                    
                    # Renaming for symmetry
                    adOutputs=imageOuts
                    
                    # There is only one at this point so no need to perform a loop
                    # CLmanager outputs a list always, so take the 0th
                    adOut = adOutputs[0]
                    
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
                    
                    if (isinstance(adInputs,list)) and (len(adInputs)>1):
                        adOutputs.append(adOut)
                    else:
                        adOutputs = adOut
               
                    count = count+1
                    
                else:
                    log.critical('One of the inputs has not been prepared,\
                    the combine function can only work on prepared data.')
                    raise ScienceError('One of the inputs was not prepared')
                
            log.warning('The CL script gireduce REPLACED the previously '+
                        'calculated DQ frames')
        
        else:
            log.critical('The parameter "adInputs" must not be None')
            raise ScienceError('The parameter "adInputs" must not be None')
        
        log.status('**FINISHED** the bias_correct function')
        
        # Return the outputs (list or single, matching adInputs)
        return adOutputs
    except:
        raise ScienceError('An error occurred while trying to run bias_correct')     
    
def combine(adInputs, fl_vardq=True, fl_dqprop=True, method='average', 
            outNames=None, suffix=None, logName='', logLevel=1, noLogFile=False):
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
    
    :param adInputs: Astrodata inputs to be combined
    :type adInputs: Astrodata objects, either a single or a list of objects
    
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
                    as adInputs.
    
    :param suffix: string to add on the end of the input filenames 
                    (or outNames if not None) for the output filenames.
    :type suffix: string
    
    :param logName: Name of the log file, default is 'gemini.log'
    :type logName: string
    
    :param logLevel: verbosity setting for the log messages to screen,
                    default is 'critical' messages only.
                    Note: independent of logLevel setting, all messages always go 
                    to the logfile if it is not turned off.
    :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to screen.
                    OR the message level as a string (ie. 'critical', 'status', 
                    'fullinfo'...)
    
    :param noLogFile: A boolean to make it so no log file is created
    :type noLogFile: Python boolean (True/False)
    """
    
    log = gemLog.getGeminiLog(logName=logName, logLevel=logLevel, 
                              noLogFile=noLogFile)

    log.status('**STARTING** the combine function')
    
    if not isinstance(adInputs,list):
        adInputs=[adInputs]
    
    if (adInputs!=None) and (outNames!=None):
        if isinstance(outNames,list):
            if len(adInputs)!= len(outNames):
                if suffix==None:
                   raise ScienceError('Then length of the inputs, '+str(len(adInputs))+
                       ', did not match the length of the outputs, '+
                       str(len(outNames))+
                       ' AND no value of "suffix" was passed in')
        if isInstance(outNames,str) and len(adInputs)>1:
            if suffix==None:
                   raise ScienceError('Then length of the inputs, '+str(len(adInputs))+
                       ', did not match the length of the outputs, '+
                       str(len(outNames))+
                       ' AND no value of "suffix" was passed in')
    
    try:
        if adInputs!=None:
            # Set up counter for looping through outNames list
            count=0
            
            # Ensuring there is more than one input to combine
            if (isinstance(adInputs,list)) and (len(adInputs)>1):
                
                # loading and bringing the pyraf related modules into the name-space
                pyraf, gemini, yes, no = pyrafLoader()
                
                # Determining if gireduce should propigate the VAR and DQ frames, if 'AUTO' was chosen 
                if fl_vardq=='AUTO':
                    if isinstance(adInputs,list):
                        if adInputs[0].countExts('VAR')==adInputs[0].countExts('DQ')==adInputs[0].countExts('SCI'):
                            fl_vardq=yes
                        else:
                            fl_vardq=no
                    else:
                        if adInputs.countExts('VAR')==adInputs.countExts('DQ')==adInputs.countExts('SCI'):
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
                clm=gemt.CLManager(imageIns=adInputs, imageOutsNames=outNames, 
                                   suffix=suffix, funcName='combine', 
                                   combinedImages=True, logName=logName,  
                                   logLevel=logLevel, noLogFile=noLogFile)
                
                # Check the status of the CLManager object, True=continue, False= issue warning
                if clm.status:
                
                    # Creating a dictionary of the parameters set by the CLManager  
                    # or the definition of the primitive 
                    clPrimParams = {
                        # Retrieving the inputs as a list from the CLManager
                        'input'       :clm.imageInsFiles(type='listFile'),
                        # Maybe allow the user to override this in the future. 
                        'output'      :clm.imageOutsFiles(type='string'), 
                        # This returns a unique/temp log file for IRAF  
                        'logfile'     :clm.templog.name,  
                        # This is actually in the default dict but wanted to 
                        # show it again       
                        'Stdout'      :gemt.IrafStdout(logLevel=logLevel), 
                        # This is actually in the default dict but wanted to 
                        # show it again    
                        'Stderr'      :gemt.IrafStdout(logLevel=logLevel),
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
                    clParamsDict = CLDefaultParamsDict('gemcombine', logLevel=logLevel)
                    clParamsDict.update(clPrimParams)
                    clParamsDict.update(clSoftcodedParams)
                    
                    # Logging the parameters that were not defaults
                    log.fullinfo('\nParameters set automatically:', category='parameters')
                    # Loop through the parameters in the clPrimParams dictionary
                    # and log them
                    gemt.logDictParams(clPrimParams, logLevel=logLevel)
                    
                    log.fullinfo('\nParameters adjustable by the user:', category='parameters')
                    # Loop through the parameters in the clSoftcodedParams dictionary
                    # and log them
                    gemt.logDictParams(clSoftcodedParams,logLevel=logLevel)
                    
                    log.debug('Calling the gemcombine CL script for input list '+
                              clm.imageInsFiles(type='listFile'))
                    
                    gemini.gemcombine(**clParamsDict)
                    
                    if gemini.gemcombine.status:
                        log.critical('gemcombine failed for inputs '+
                                     clm.imageInsFiles(type='string'))
                        raise ScienceError('gemcombine failed')
                    else:
                        log.status('Exited the gemcombine CL script successfully')
                    
                    
                    # Renaming CL outputs and loading them back into memory 
                    # and cleaning up the intermediate temp files written to disk
                    # refOuts and arrayOuts are None here
                    imageOuts, refOuts, arrayOuts = clm.finishCL() 
                
                    # Renaming for symmetry
                    adOutputs=imageOuts
                
                    # There is only one at this point so no need to perform a 
                    # loop CLmanager outputs a list always, so take the 0th
                    adOut = adOutputs[0]
                    
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
                    raise ScienceError('One of the inputs was not prepared')
        else:
            log.critical('The parameter "adInputs" must not be None')
            raise ScienceError('The parameter "adInputs" must not be None')
        
        log.status('**FINISHED** the combine function')
        
        # Return the outputs (list or single, matching adInputs)
        return adOut
    except:
        raise ScienceError('An error occurred while trying to run combine')
                
def flat_correct(adInputs, flats=None, outNames=None, suffix=None, logName='', logLevel=1, 
                                                            noLogFile=False):
    """
    This function performs a flat correction by dividing the inputs by  
    processed flats, similar to the way gireduce would perform this operation
    but written in pure python in the arith toolbox.
    
    A string representing the name of the log file to write all log messages to
    can be defined, or a default of 'gemini.log' will be used.  If the file
    all ready exists in the directory you are working in, then this file will 
    have the log messages during this function added to the end of it.
    
    :param adInputs: Astrodata inputs to have DQ extensions added to
    :type adInputs: Astrodata objects, either a single or a list of objects
    
    :param flats: The flat(s) to divide the input(s) by.
    :type flats: AstroData objects in a list, or a single instance.
                Note: If there is multiple inputs and one flat provided, then the
                same flat will be applied to all inputs; else the flats   
                list must match the length of the inputs.
    
    :param outNames: filenames of output(s)
    :type outNames: String, either a single or a list of strings of same length as adInputs.
    
    :param suffix: string to add on the end of the input filenames 
                    (or outNames if not None) for the output filenames.
    :type suffix: string
    
    :param logName: Name of the log file, default is 'gemini.log'
    :type logName: string
    
    :param logLevel: verbosity setting for the log messages to screen,
                    default is 'critical' messages only.
                    Note: independent of logLevel setting, all messages always go 
                    to the logfile if it is not turned off.
    :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to screen.
                    OR the message level as a string (ie. 'critical', 'status', 
                    'fullinfo'...)
    
    :param noLogFile: A boolean to make it so no log file is created
    :type noLogFile: Python boolean (True/False)
    """
    
    log=gemLog.getGeminiLog(logName=logName, logLevel=logLevel, noLogFile=noLogFile)
        
    log.status('**STARTING** the flat_correct function')
    
    if not isinstance(adInputs,list):
        adInputs=[adInputs]
    
    if (adInputs!=None) and (outNames!=None):
        if isinstance(outNames,list):
            if len(adInputs)!= len(outNames):
                if suffix==None:
                   raise ScienceError('Then length of the inputs, '+str(len(adInputs))+
                       ', did not match the length of the outputs, '+
                       str(len(outNames))+
                       ' AND no value of "suffix" was passed in')
        if isInstance(outNames,str) and len(adInputs)>1:
            if suffix==None:
                   raise ScienceError('Then length of the inputs, '+str(len(adInputs))+
                       ', did not match the length of the outputs, '+
                       str(len(outNames))+
                       ' AND no value of "suffix" was passed in')
    
    if flats==None:
        raise ScienceError('There must be at least one processed flat provided, the "flats" parameter must not be None.')
    
    try:
        if adInputs!=None:
            # Set up counter for looping through outNames list
            count=0
            
            # Creating empty list of ad's to be returned that will be filled below
            if len(adInputs)>1:
                adOutputs=[]
            
            # Loop through the inputs to perform the non-linear and saturated
            # pixel searches of the SCI frames to update the BPM frames into
            # full DQ frames. 
            for ad in adInputs:                   
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
                
                # Updating the file name with the suffix for this
                # function and then reporting the new file 
                if suffix!=None:
                    log.debug('Calling gemt.fileNameUpdater on '+adOut.filename)
                    if outNames!=None:
                        adOut.filename = gemt.fileNameUpdater(adIn=adOut, 
                                                              infilename=outNames[count],
                                                          suffix=suffix, 
                                                          strip=False, logLevel=logLevel)
                    else:
                        adOut.filename = gemt.fileNameUpdater(adIn=adOut, 
                                                          suffix=suffix, 
                                                          strip=False, logLevel=logLevel)
                elif suffix==None:
                    if outNames!=None:
                        if len(outNames)>1: 
                            adOut.filename = outNames[count]
                        else:
                            adOut.filename = outNames
                    else:
                        raise ScienceError('outNames and suffix parameters can not BOTH\
                                                                    be None')
                        
                log.status('File name updated to '+adOut.filename)
            
                if (isinstance(adInputs,list)) and (len(adInputs)>1):
                    adOutputs.append(adOut)
                else:
                    adOutputs = adOut

                count=count+1
        else:
            raise ScienceError('The parameter "adInputs" must not be None')
        
        log.status('**FINISHED** the flat_correct function')
        
        # Return the outputs (list or single, matching adInputs)
        return adOutputs
    except:
        raise ScienceError('An error occurred while trying to run flat_correct')
                
def measure_iq(adInputs, function='both', display=True, mosaic=True, qa=True,
               keepDats=False, logName='', logLevel=1, noLogFile=False):
    """
    This function will detect the sources in the input images and fit
    both Gaussian and Moffat models to their profiles and calculate the 
    Image Quality and seeing from this.
    
    Since the resultant parameters are formatted into one nice string and 
    normally recorded in a logger message, the returned dictionary of these 
    parameters may be ignored. BUT, if the user wishes to completely shut off
    the logging, then the returned dictionary of the results can be useful.
    The dictionary's format is:
    {adIn1.filename:formatted results string for adIn1, 
    adIn2.filename:formatted results string for adIn2,...}
    
    There are also .dat files that result from this function written to the 
    current working directory under the names 'measure_iq'+adIn.filename+'.dat'.
    ex: input filename 'N20100311S0090.fits', 
    .dat filename 'measure_iqN20100311S0090.dat'
    
    NOTE:
    A string representing the name of the log file to write all log messages to
    can be defined, or a default of 'gemini.log' will be used.  If the file
    all ready exists in the directory you are working in, then this file will 
    have the log messages during this function added to the end of it.
    
    :param adInputs: Astrodata inputs to have their image quality measured
    :type adInputs: Astrodata objects, either a single or a list of objects
    
    :param function: Function for centroid fitting
    :type function: string, can be: 'moffat','gauss' or 'both'; 
                    Default 'both'
                    
    :param display: Flag to turn on displaying the fitting to ds9
    :type display: Python boolean (True/False)
                   Default: True
    
    :param mosaic: Flag to indicate the images have been mosaic'd 
                   (ie only 1 'SCI' extension in images)
    :type mosaic: Python boolean (True/False)
                  default: True
    :param qa: flag to use a grid of sub-windows for detecting the sources in 
               the image frames, rather than the entire frame all at once.
    :type qa: Python boolean (True/False)
              default: True
    
    :param keepDats: flag to keep the .dat files that provide detailed results found
                     while measuring the input's image quality.
    :type keepDats: Python boolean (True/False)
                    default: False
    
    :param outNames: filenames of output(s)
    :type outNames: String, either a single or a list of strings of same length
                    as adInputs.
    
    :param suffix: string to add on the end of the input filenames 
                    (or outNames if not None) for the output filenames.
    :type suffix: string
    
    :param logName: Name of the log file, default is 'gemini.log'
    :type logName: string
    
    :param logLevel: verbosity setting for the log messages to screen,
                     default is 'critical' messages only.
                     Note: independent of logLevel setting, all messages always go 
                     to the logfile if it is not turned off.
    :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to screen.
                    OR the message level as a string (ie. 'critical', 'status', 
                    'fullinfo'...)
    
    :param noLogFile: A boolean to make it so no log file is created
    :type noLogFile: Python boolean (True/False)
    """
    
    if logName!='':
        log=gemLog.getGeminiLog(logName=logName, logLevel=logLevel, 
                                noLogFile=noLogFile)
    else:
        # Use default logName 'gemini.log'
        log=gemLog.getGeminiLog(logLevel=logLevel, noLogFile=noLogFile)
        
    log.status('**STARTING** the measure_iq function')
    
    if not isinstance(adInputs,list):
        adInputs=[adInputs]
    
    try:
        if adInputs!=None:
            # Importing getiq module to perform the source detection and IQ
            # measurements of the inputs
            from iqtool.iq import getiq
            
            # Initializing a total time sum variable for logging purposes 
            total_IQ_time = 0
            
            # Creating dictionary for output strings to be returned in
            outDict = {}
            
            # Loop through the inputs to perform the non-linear and saturated
            # pixel searches of the SCI frames to update the BPM frames into
            # full DQ frames. 
            for ad in adInputs:                     
                # Writing the input to disk under a temp name in the current 
                # working directory for getiq to use to be deleted after getiq
                tmpWriteName = 'measure_iq'+os.path.basename(ad.filename)
                log.fullinfo('The inputs to measureIQ must be in the'+
                             ' current working directory for it to work '+\
                             'correctly, so writting it temperarily to file '+
                             tmpWriteName)
                ad.write(tmpWriteName, rename=False)
                
                # Start time for measuring IQ of current file
                st = time.time()
                
                log.debug('Calling getiq.gemiq for input '+ad.filename)
                
                # Calling the gemiq function to detect the sources and then
                # measure the IQ of the current image 
                iqdata = getiq.gemiq(tmpWriteName, function=function, 
                                      verbose=True, display=display, 
                                      mosaic=mosaic, qa=qa)
                
                # End time for measuring IQ of current file
                et = time.time()
                total_IQ_time = total_IQ_time + (et - st)
                # Logging the amount of time spent measuring the IQ 
                log.debug('MeasureIQ time: '+repr(et - st), category='IQ')
                log.fullinfo('~'*45, category='format')
                
                # If input was writen to temp file on disk, delete it
                if os.path.exists(tmpWriteName):
                    os.remove(tmpWriteName)
                
                # Deleting the .dat file from disk if requested
                if not keepDats:
                    datName = os.path.splitext(name)[0]+'.dat'
                    os.remove(datName)
                    
                # iqdata is list of tuples with image quality metrics
                # (ellMean, ellSig, fwhmMean, fwhmSig)
                # First check if it is empty (ie. gemiq failed in someway)
                if len(iqdata) == 0:
                    log.warning('Problem Measuring IQ Statistics, '+
                                'none reported')
                # If it all worked, then format the output and log it
                else:
                    # Formatting this output for printing or logging                
                    fnStr = 'Filename:'.ljust(19)+ad.filename
                    emStr = 'Ellipticity Mean:'.ljust(19)+str(iqdata[0][0])
                    esStr = 'Ellipticity Sigma:'.ljust(19)+str(iqdata[0][1])
                    fmStr = 'FWHM Mean:'.ljust(19)+str(iqdata[0][2])
                    fsStr = 'FWHM Sigma:'.ljust(19)+str(iqdata[0][3])
                    sStr = 'Seeing:'.ljust(19)+str(iqdata[0][2])
                    psStr = 'PixelScale:'.ljust(19)+str(ad.pixel_scale())
                    vStr = 'VERSION:'.ljust(19)+'None' #$$$$$ made on ln12 of ReductionsObjectRequest.py, always 'None' it seems.
                    tStr = 'TIMESTAMP:'.ljust(19)+str(datetime.now())
                    # Create final formated string
                    finalStr = '-'*45+'\n'+fnStr+'\n'+emStr+'\n'+esStr+'\n'\
                                    +fmStr+'\n'+fsStr+'\n'+sStr+'\n'+psStr+\
                                    '\n'+vStr+'\n'+tStr+'\n'+'-'*45
                    # Log final string
                    log.stdinfo(finalStr, category='IQ')
                    
                    # appending formated string to the output dictionary
                    outDict[ad.filename] = finalStr
                    
            # Logging the total amount of time spent measuring the IQ of all
            # the inputs
            log.debug('Total measureIQ time: '+repr(total_IQ_time), 
                        category='IQ')
            
            #returning complete dictionary for use by the user if desired
            return outDict
        else:
            raise ScienceError('The parameter "adInputs" must not be None')
    except:
        raise ScienceError('An error occurred while trying to run measure_iq')                              
                
def mosaic_detectors(adInputs, fl_paste=False, interp_function='linear', fl_vardq='AUTO', 
                outNames=None, suffix=None, logName='', logLevel=1, noLogFile=False):
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
    
    :param adInputs: Astrodata inputs to mosaic the extensions of
    :type adInputs: Astrodata objects, either a single or a list of objects
    
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
    :type outNames: String, either a single or a list of strings of same length as adInputs.
    
    :param suffix: string to add on the end of the input filenames 
                    (or outNames if not None) for the output filenames.
    :type suffix: string
    
    :param logName: Name of the log file, default is 'gemini.log'
    :type logName: string
    
    :param logLevel: verbosity setting for the log messages to screen,
                    default is 'critical' messages only.
                    Note: independent of logLevel setting, all messages always go 
                    to the logfile if it is not turned off.
    :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to screen.
                    OR the message level as a string (ie. 'critical', 'status', 
                    'fullinfo'...)
    
    :param noLogFile: A boolean to make it so no log file is created
    :type noLogFile: Python boolean (True/False)
    """
    
    log=gemLog.getGeminiLog(logName=logName, logLevel=logLevel, noLogFile=noLogFile)

    log.status('**STARTING** the mosaic_detectors function')
    
    if not isinstance(adInputs,list):
        adInputs=[adInputs]
    
    if (adInputs!=None) and (outNames!=None):
        if isinstance(outNames,list):
            if len(adInputs)!= len(outNames):
                if suffix==None:
                   raise ScienceError('Then length of the inputs, '+str(len(adInputs))+
                       ', did not match the length of the outputs, '+
                       str(len(outNames))+
                       ' AND no value of "suffix" was passed in')
        if isInstance(outNames,str) and len(adInputs)>1:
            if suffix==None:
                   raise ScienceError('Then length of the inputs, '+str(len(adInputs))+
                       ', did not match the length of the outputs, '+
                       str(len(outNames))+
                       ' AND no value of "suffix" was passed in')
    
    try:
        if adInputs!=None: 
            # loading and bringing the pyraf related modules into the name-space
            pyraf, gemini, yes, no = pyrafLoader()  
                
            # Determining if gmosaic should propigate the VAR and DQ frames, if 'AUTO' was chosen 
            if fl_vardq=='AUTO':
                if isinstance(adInputs,list):
                    if adInputs[0].countExts('VAR')==adInputs[0].countExts('DQ')==adInputs[0].countExts('SCI'):
                        fl_vardq=yes
                    else:
                        fl_vardq=no
                else:
                    if adInputs.countExts('VAR')==adInputs.countExts('DQ')==adInputs.countExts('SCI'):
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
            clm=gemt.CLManager(imageIns=adInputs, imageOutsNames=outNames, suffix=suffix, 
                               funcName='mosaicDetectors', logName=logName, 
                               logLevel=logLevel, noLogFile=noLogFile)
            
            # Check the status of the CLManager object, True=continue, False= issue warning
            if clm.status: 
                
                # Parameters set by the gemt.CLManager or the definition of the prim 
                clPrimParams = {
                  # Retrieving the inputs as a string of filenames
                  'inimages'    :clm.imageInsFiles(type='string'),
                  'outimages'   :clm.imageOutsFiles(type='string'),
                  # Setting the value of FL_vardq set above
                  'fl_vardq'    :fl_vardq,
                  # This returns a unique/temp log file for IRAF 
                  'logfile'     :clm.templog.name,
                  # This is actually in the default dict but wanted to show it again     
                  'Stdout'      :gemt.IrafStdout(logLevel=logLevel), 
                  # This is actually in the default dict but wanted to show it again
                  'Stderr'      :gemt.IrafStdout(logLevel=logLevel), 
                  # This is actually in the default dict but wanted to show it again
                  'verbose'     :yes                
                              }
                # Parameters from the Parameter file adjustable by the user
                clSoftcodedParams = {
                  # pyrafBoolean converts the python booleans to pyraf ones
                  'fl_paste'    :gemt.pyrafBoolean(fl_paste),
                  'outpref'     :suffix,
                  'geointer'    :interp_function,
                                  }
                # Grabbing the default params dict and updating it with 
                # the two above dicts
                clParamsDict = CLDefaultParamsDict('gmosaic', logLevel=logLevel)
                clParamsDict.update(clPrimParams)
                clParamsDict.update(clSoftcodedParams)      
                    
                # Logging the parameters that were not defaults
                log.fullinfo('\nParameters set automatically:', 
                             category='parameters')
                # Loop through the parameters in the clPrimParams dictionary
                # and log them
                gemt.logDictParams(clPrimParams, logLevel=logLevel)
                
                log.fullinfo('\nParameters adjustable by the user:', 
                             category='parameters')
                # Loop through the parameters in the clSoftcodedParams 
                # dictionary and log them
                gemt.logDictParams(clSoftcodedParams,logLevel=logLevel)
                
                log.debug('calling the gmosaic CL script for inputs '+
                                                        clm.imageInsFiles(type='string'))
            
                gemini.gmos.gmosaic(**clParamsDict)
        
                if gemini.gmos.gmosaic.status:
                    log.critical('gireduce failed for inputs '+
                                 clm.imageInsFiles(type='string'))
                    raise ScienceError('gmosaic failed')
                else:
                    log.status('Exited the gmosaic CL script successfully')    
                    
                    
                # Renaming CL outputs and loading them back into memory 
                # and cleaning up the intermediate temp files written to disk
                # refOuts and arrayOuts are None here
                imageOuts, refOuts, arrayOuts = clm.finishCL()   
                
                # Renaming for symmetry
                adOutputs=imageOuts
                    
                # Wrap up logging
                i=0
                for ad in adOutputs:
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
                    raise ScienceError('One of the inputs was not prepared')
                
        else:
            log.critical('The parameter "adInputs" must not be None')
            raise ScienceError('The parameter "adInputs" must not be None')
        
        log.status('**FINISHED** the mosaic_detectors function')
        
        # Return the outputs (list or single, matching adInputs)
        return adOutputs
    except:
        raise ScienceError('An error occurred while trying to run mosaic_detectors') 
                
def normalize_flat(adInputs, fl_trim=False, fl_over=False,fl_vardq='AUTO', 
                outNames=None, suffix=None, logName='', logLevel=1, noLogFile=False):
    """
    This function will combine the input flats (adInputs) and then normalize them 
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
    
    :param adInputs: Astrodata input flat(s) to be combined and normalized
    :type adInputs: Astrodata objects, either a single or a list of objects
    
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
    :type outNames: String, either a single or a list of strings of same length as adInputs.
    
    :param suffix:
            string to add on the end of the input filenames 
            (or outNames if not None) for the output filenames.
    :type suffix: string
    
    :param logName: Name of the log file, default is 'gemini.log'
    :type logName: string
    
    :param logLevel: verbosity setting for the log messages to screen,
                    default is 'critical' messages only.
                    Note: independent of logLevel setting, all messages always go 
                    to the logfile if it is not turned off.
    :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to screen.
                    OR the message level as a string (ie. 'critical', 'status', 
                    'fullinfo'...)
    
    :param noLogFile: A boolean to make it so no log file is created
    :type noLogFile: Python boolean (True/False)
    """
    
    log=gemLog.getGeminiLog(logName=logName, logLevel=logLevel, noLogFile=noLogFile)

    log.status('**STARTING** the normalize_flat function')
    
    if not isinstance(adInputs,list):
        adInputs=[adInputs]
    
    if (adInputs!=None) and (outNames!=None):
        if isinstance(outNames,list):
            if len(adInputs)!= len(outNames):
                if suffix==None:
                   raise ScienceError('Then length of the inputs, '+str(len(adInputs))+
                       ', did not match the length of the outputs, '+
                       str(len(outNames))+
                       ' AND no value of "suffix" was passed in')
        if isInstance(outNames,str) and len(adInputs)>1:
            if suffix==None:
                   raise ScienceError('Then length of the inputs, '+str(len(adInputs))+
                       ', did not match the length of the outputs, '+
                       str(len(outNames))+
                       ' AND no value of "suffix" was passed in')
    
    try:
        if adInputs!=None: 
            # loading and bringing the pyraf related modules into the name-space
            pyraf, gemini, yes, no = pyrafLoader()  
                
            # Determining if gmosaic should propigate the VAR and DQ frames, if 'AUTO' was chosen 
            if fl_vardq=='AUTO':
                if isinstance(adInputs,list):
                    if adInputs[0].countExts('VAR')==adInputs[0].countExts('DQ')==adInputs[0].countExts('SCI'):
                        fl_vardq=yes
                    else:
                        fl_vardq=no
                else:
                    if adInputs.countExts('VAR')==adInputs.countExts('DQ')==adInputs.countExts('SCI'):
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
            clm=gemt.CLManager(imageIns=adInputs, imageOutsNames=outNames, suffix=suffix, 
                               funcName='normalizeFlat', logName=logName, 
                               combinedImages=True, logLevel=logLevel, 
                               noLogFile=noLogFile)
            
            # Check the status of the CLManager object, True=continue, False= issue warning
            if clm.status:                 
                
                # Creating a dictionary of the parameters set by the gemt.CLManager 
                # or the definition of the function 
                clPrimParams = {
                  'inflats'     :clm.imageInsFiles(type='listFile'),
                  # Maybe allow the user to override this in the future
                  'outflat'     :clm.imageOutsFiles(type='string'), 
                  # This returns a unique/temp log file for IRAF  
                  'logfile'     :clm.templog.name,         
                  # This is actually in the default dict but wanted to show it again
                  'Stdout'      :gemt.IrafStdout(logLevel=logLevel),   
                  # This is actually in the default dict but wanted to show it again  
                  'Stderr'      :gemt.IrafStdout(logLevel=logLevel), 
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
                clParamsDict = CLDefaultParamsDict('giflat', logLevel=logLevel)
                clParamsDict.update(clPrimParams)
                clParamsDict.update(clSoftcodedParams)
                
                # Logging the parameters that were not defaults
                log.fullinfo('\nParameters set automatically:', 
                             category='parameters')
                # Loop through the parameters in the clPrimParams dictionary
                # and log them
                gemt.logDictParams(clPrimParams, logLevel=logLevel)
                
                log.fullinfo('\nParameters adjustable by the user:', 
                             category='parameters')
                # Loop through the parameters in the clSoftcodedParams 
                # dictionary and log them
                gemt.logDictParams(clSoftcodedParams,logLevel=logLevel)
                
                log.debug('Calling the giflat CL script for inputs list '+
                      clm.imageInsFiles(type='listFile'))
            
                gemini.giflat(**clParamsDict)
                
                if gemini.giflat.status:
                    log.critical('giflat failed for inputs '+
                                 clm.imageInsFiles(type='string'))
                    raise ScienceError('giflat failed')
                else:
                    log.status('Exited the giflat CL script successfully')
                
                # Renaming CL outputs and loading them back into memory 
                # and cleaning up the intermediate temp files written to disk
                # refOuts and arrayOuts are None here
                imageOuts, refOuts, arrayOuts = clm.finishCL()
            
                # Renaming for symmetry
                adOutputs=imageOuts
                    
                # There is only one at this point so no need to perform a loop
                # CLmanager outputs a list always, so take the 0th
                adOut = adOutputs[0]
                
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
                raise ScienceError('One of the inputs was not prepared')
                
        else:
            log.critical('The parameter "adInputs" must not be None')
            raise ScienceError('The parameter "adInputs" must not be None')
        
        log.status('**FINISHED** the normalize_flat function')
        
        # Return the outputs (list or single, matching adInputs)
        return adOutputs
    except:
        raise ScienceError('An error occurred while trying to run normalize_flat')    
    
def overscan_trim(adInputs, outNames=None, suffix=None, logName='', logLevel=1, 
                                                            noLogFile=False):
    """
    This function uses AstroData to trim the overscan region 
    from the input images and update their headers.
    
    A string representing the name of the log file to write all log messages to
    can be defined, or a default of 'gemini.log' will be used.  If the file
    all ready exists in the directory you are working in, then this file will 
    have the log messages during this function added to the end of it.
    
    :param adInputs: Astrodata inputs to have DQ extensions added to
    :type adInputs: Astrodata objects, either a single or a list of objects
    
    :param outNames: filenames of output(s)
    :type outNames: String, either a single or a list of strings of same length
                    as adInputs.
    
    :param suffix: string to add on the end of the input filenames 
                    (or outNames if not None) for the output filenames.
    :type suffix: string
    
    :param logName: Name of the log file, default is 'gemini.log'
    :type logName: string
    
    :param logLevel: verbosity setting for the log messages to screen,
                    default is 'critical' messages only.
                    Note: independent of logLevel setting, all messages always go 
                    to the logfile if it is not turned off.
    :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to screen.
                    OR the message level as a string (ie. 'critical', 'status', 
                    'fullinfo'...)
    
    :param noLogFile: A boolean to make it so no log file is created
    :type noLogFile: Python boolean (True/False)
    """
    
    if logName!='':
        log=gemLog.getGeminiLog(logName=logName, logLevel=logLevel, 
                                noLogFile=noLogFile)
    else:
        # Use default logName 'gemini.log'
        log=gemLog.getGeminiLog(logLevel=logLevel, noLogFile=noLogFile)
        
    log.status('**STARTING** the overscan_trim function')
    
    if not isinstance(adInputs,list):
        adInputs=[adInputs]
    
    if (adInputs!=None) and (outNames!=None):
        if isinstance(outNames,list):
            if len(adInputs)!= len(outNames):
                if suffix==None:
                   raise ScienceError('Then length of the inputs, '+str(len(adInputs))+
                       ', did not match the length of the outputs, '+
                       str(len(outNames))+
                       ' AND no value of "suffix" was passed in')
        if isInstance(outNames,str) and len(adInputs)>1:
            if suffix==None:
                   raise ScienceError('Then length of the inputs, '+str(len(adInputs))+
                       ', did not match the length of the outputs, '+
                       str(len(outNames))+
                       ' AND no value of "suffix" was passed in')
    
    try:
        if adInputs!=None:
            # Set up counter for looping through outNames list
            count=0
            
            # Creating empty list of ad's to be returned that will be filled below
            if len(adInputs)>1:
                adOutputs=[]
            
            # Loop through the inputs to perform the non-linear and saturated
            # pixel searches of the SCI frames to update the BPM frames into
            # full DQ frames. 
            for ad in adInputs:  
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
                
                # Updating the file name with the suffix for this
                # function and then reporting the new file 
                if suffix!=None:
                    log.debug('Calling gemt.fileNameUpdater on '+adOut.filename)
                    if outNames!=None:
                        adOut.filename = gemt.fileNameUpdater(adIn=adOut, 
                                                              infilename=outNames[count],
                                                          suffix=suffix, 
                                                          strip=False, logLevel=logLevel)
                    else:
                        adOut.filename = gemt.fileNameUpdater(adIn=adOut, 
                                                          suffix=suffix, 
                                                          strip=False, logLevel=logLevel)
                elif suffix==None:
                    if outNames!=None:
                        if len(outNames)>1: 
                            adOut.filename = outNames[count]
                        else:
                            adOut.filename = outNames
                    else:
                        raise ScienceError('outNames and suffix parameters can not BOTH\
                                                                    be None')
                        
                log.status('File name updated to '+adOut.filename)
            
                if (isinstance(adInputs,list)) and (len(adInputs)>1):
                    adOutputs.append(adOut)
                else:
                    adOutputs = adOut

                count=count+1
        else:
            raise ScienceError('The parameter "adInputs" must not be None')
        
        log.status('**FINISHED** the overscan_trim function')
        
        # Return the outputs (list or single, matching adInputs)
        return adOutputs
    except:
        raise ScienceError('An error occurred while trying to run overscan_trim')




                
                
                
                
                

