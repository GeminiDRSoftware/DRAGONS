# Author: Kyle Mede, February 2011
# For now, this module is to hold the code which performs the actual work of the 
# primitives that is considered generic enough to be at the 'GMOS' level of
# the hierarchy tree, but too specific for the 'gemini' level above this.

import os

import pyfits as pf
import numpy as np
from copy import deepcopy

from astrodata.adutils import gemLog
from astrodata.AstroData import AstroData
from gempy.instruments import geminiTools  as gemt
from astrodata.adutils.gemutil import pyrafLoader
from gempy.instruments.geminiCLParDicts import CLDefaultParamsDict


def overscanSubtract(adIns, fl_trim=False, fl_vardq='AUTO', 
            biassec='[1:25,1:2304],[1:32,1:2304],[1025:1056,1:2304]',
            outNames=None, postpend=None, logName='', verbose=1, noLogFile=False):
    """
    This function uses the CL script gireduce to subtract the overscan 
    from the input images.
    
    WARNING: 
        The gireduce script used here replaces the previously 
        calculated DQ frames with its own versions.  This may be corrected 
        in the future by replacing the use of the gireduce
        with a Python routine to do the overscan subtraction.

    note
        The inputs to this function MUST be prepared.

    String representing the name of the log file to write all log messages to
    can be defined, or a default of 'gemini.log' will be used.  If the file
    all ready exists in the directory you are working in, then this file will 
    have the log messages during this function added to the end of it.

    FOR FUTURE
        This function has many GMOS dependencies that would be great to work out
        so that this could be made a more general function (say at the Gemini level).
        In the future the parameters can be looked into and the CL script can be 
        upgraded to handle things like row based overscan calculations/fitting/modeling... 
        vs the column based used right now, add the model, nbiascontam, ... params to the 
        functions inputs so the user can choose them for themselves.

    :param noLogFile: A boolean to make it so no log file is created
    :type noLogFile: Python boolean (True/False)
    
    :param fl_trim: Trim the overscan region from the frames?
    :type fl_trim: Python boolean (True/False)
    
    :param fl_vardq: Create variance and data quality frames?
    :type fl_vardq: 
        Python boolean (True/False), OR string 'AUTO' to do 
        it automatically if there are VAR and DQ frames in the inputs.
        NOTE: 'AUTO' uses the first input to determine if VAR and DQ frames exist, 
        so, if the first does, then the rest MUST also have them as well.
        
    :param LogFile: A boolean to make it so no log file is created
    :type LogFile: Python boolean (True/False)

    :param biassec: biassec parameter of format '[#:#,#:#],[#:#,#:#],[#:#,#:#]'
    :type biassec: string. default: '[1:25,1:2304],[1:32,1:2304],[1025:1056,1:2304]' is ideal for 2x2 GMOS data.
    
    :param outNames: filenames of output(s)
    :type outNames: String, either a single or a list of strings of same length as adIns.
    
    :param postpend: string to postpend on the end of the input filenames (or outNames if not None) for the output filenames.
    :type postpend: string
    
    :param logName: Name of the log file, default is 'gemini.log'
    :type logName: string
    
    :param verbose: 
         verbosity setting for the log messages to screen,
         default is 'critical' messages only.
         Note: independent of verbose setting, all messages always go 
         to the logfile if it is not turned off.
    :type verbose: integer from 0-6, 0=nothing to screen, 6=everything to screen

    """

    log=gemLog.getGeminiLog(logName=logName, verbose=verbose, noLogFile=noLogFile)

    log.status('**STARTING** the overscanSubtract function')
    
    if (adIns!=None) and (outNames!=None):
        if isinstance(adIns,list) and isinstance(outNames,list):
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
                               funcName='overscanSubtract', logName=logName,  
                                   verbose=verbose, noLogFile=noLogFile)
            
            # Check the status of the CLManager object, True=continue, False= issue warning
            if clm.status:                     
                
                # Parameters set by the gemt.CLManager or the definition 
                # of the primitive 
                clPrimParams = {
                  'inimages'    :clm.inputsAsStr(),
                  'gp_outpref'  :clm.uniquePrefix(),
                  # This returns a unique/temp log file for IRAF
                  'logfile'     :clm.logfile(),      
                  'fl_over'     :yes, 
                  # This is actually in the default dict but wanted to show it again
                  'Stdout'      :gemt.IrafStdout(verbose=verbose), 
                  # This is actually in the default dict but wanted to show it again
                  'Stderr'      :gemt.IrafStdout(verbose=verbose), 
                  # This is actually in the default dict but wanted to show it again
                  'verbose'     :yes                
                              }
                
                # Taking care of the biasec->nbiascontam param
                if not biassec == '':
                    nbiascontam = clm.nbiascontam(biassec=biassec)
                    log.fullinfo('nbiascontam parameter was updated to = '+
                                 str(nbiascontam))
                else: 
                    # Do not try to calculate it, just use default value of 4.
                    nbiascontam = 4
                
                # Parameters from the Parameter file that are adjustable by the user
                clSoftcodedParams = {
                   # pyrafBoolean converts the python booleans to pyraf ones
                   'fl_trim'    :gemt.pyrafBoolean(fl_trim),
                   'outpref'    :postpend,
                   'fl_vardq'   :fl_vardq,
                   'nbiascontam':nbiascontam
                                   }
                # Grabbing the default params dict and updating it with 
                # the two above dicts
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
                
                log.debug('Calling the gireduce CL script for inputs '+
                      clm.inputsAsStr())
            
                gemini.gmos.gireduce(**clParamsDict)
                
                if gemini.gmos.gireduce.status:
                    log.critical('gireduce failed for inputs '+
                                 clm.inputsAsStr())
                    raise ('gireduce failed')
                else:
                    log.status('Exited the gireduce CL script successfully')
                
                # Renaming CL outputs and loading them back into memory, and 
                # cleaning up the intermediate tmp files written to disk
                adOuts = clm.finishCL()
                
                # Wrap up logging
                i=0
                for adOut in adOuts:
                    # Verifying gireduce was actually ran on the file
                    if adOut.phuGetKeyValue('GIREDUCE'): 
                        # If gireduce was ran, then log the changes to the files 
                        # it made
                        log.fullinfo('\nFile '+clm.preCLNames()[i]+
                                     ' had its overscan subracted successfully')
                        log.fullinfo('New file name is: '+adOut.filename)
                    i = i+1
                    # Updating GEM-TLM and OVERSUB time stamps in the PHU
                    adOut.historyMark(key='OVERSUB', stomp=False)  
                    
                    # Updating logger with new GEM-TLM time stamp value
                    log.fullinfo('************************************************'
                                 , category='header')
                    log.fullinfo('File = '+adOut.filename, category='header')
                    log.fullinfo('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
                                 , category='header')
                    log.fullinfo('PHU keywords updated/added:\n', 'header')
                    log.fullinfo('GEM-TLM = '+adOut.phuGetKeyValue('GEM-TLM'), 
                                  category='header')
                    log.fullinfo('OVERSUB = '+adOut.phuGetKeyValue('OVERSUB')+'\n', 
                                  category='header')
                
                
            else:
                log.critical('One of the inputs has not been prepared,\
                the overscanSubtract function can only work on prepared data.')
                raise('One of the inputs was not prepared')
                
        else:
            log.critical('The parameter "adIns" must not be None')
            raise('The parameter "adIns" must not be None')
        
        log.status('**FINISHED** the overscanSubtract function')
        
        # Return the outputs (list or single, matching adIns)
        return adOuts
    except:
        raise #('An error occurred while trying to run overscanSubtract') 
                
                
                
                
                
