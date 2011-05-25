# This module contains user level functions related to the preprocessing of
# the input dataset with a bias frame

import os, sys
import numpy as np
from copy import deepcopy
from astrodata import Errors
from astrodata.adutils import varutil
from astrodata.adutils.gemutil import pyrafLoader
from astrodata.AstroData import AstroData
from gempy import geminiTools as gt
from gempy import managers as man
from gempy import string
from gempy.geminiCLParDicts import CLDefaultParamsDict

def overscan_subtract_gmos(adInputs, fl_trim=False, fl_vardq='AUTO', 
            biassec='[1:25,1:2304],[1:32,1:2304],[1025:1056,1:2304]',
            outNames=None, suffix=None):
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

    Either a 'main' type logger object, if it exists, or a null logger 
    (ie, no log file, no messages to screen) will be retrieved/created in the 
    ScienceFunctionManager and used within this function.

    FOR FUTURE
    This function has many GMOS dependencies that would be great to work out
    so that this could be made a more general function (say at the Gemini level)
    .  In the future the parameters can be looked into and the CL script can be 
    upgraded to handle things like row based overscan calculations/fitting/
    modeling... vs the column based used right now, add the model, nbiascontam,
    ... params to the functions inputs so the user can choose them for 
    themselves.

    :param adInputs: Astrodata inputs to be converted to Electron pixel units
    :type adInputs: Astrodata objects, either a single or a list of objects
    
    :param fl_trim: Trim the overscan region from the frames?
    :type fl_trim: Python boolean (True/False)
    
    :param fl_vardq: Create variance and data quality frames?
    :type fl_vardq: 
        Python boolean (True/False), OR string 'AUTO' to do 
        it automatically if there are VAR and DQ frames in the inputs.
        NOTE: 'AUTO' uses the first input to determine if VAR and DQ frames  
        exist, so, if the first does, then the rest MUST also have them as well.

    :param biassec: biassec parameter of format 
                    '[x1:x2,y1:y2],[x1:x2,y1:y2],[x1:x2,y1:y2]'
    :type biassec: string. 
                   default: '[1:25,1:2304],[1:32,1:2304],[1025:1056,1:2304]' 
                   is ideal for 2x2 GMOS data.
    
    :param outNames: filenames of output(s)
    :type outNames: String, either a single or a list of strings of same length 
                    as adInputs.
    
    :param suffix: string to postpend on the end of the input filenames 
                   (or outNames if not None) for the output filenames.
    :type suffix: string

    """

    # Instantiate ScienceFunctionManager object
    sfm = man.ScienceFunctionManager(adInputs, outNames, suffix, 
                                      funcName='overscan_subtract_gmos') 
    # Perform start up checks of the inputs, prep/check of outnames, and get log
    adInputs, outNames, log = sfm.startUp()
    
    try: 
        # loading and bringing the pyraf related modules into the name-space
        pyraf, gemini, yes, no = pyrafLoader() 
        ###################################
        ##################################
        pyraf.iraf.task(gireduce='/home/kmede/workspace/gemini_python/test_data/gireduce.cl')
        #gireduce(params)
        ########################
        ############################## 
        # Creating empty list of ad's to be returned that will be filled below
        adOutputs=[]
                
        # Converting input True/False to yes/no or detecting fl_vardq value
        # if 'AUTO' chosen with autoVardq in the ScienceFunctionManager
        fl_vardq = sfm.autoVardq(fl_vardq)
        
        # To clean up log and screen if multiple inputs
        log.fullinfo('+'*50, category='format')                                 
            
        # Preparing input files, lists, parameters... for input to 
        # the CL script
        clm=man.CLManager(imageIns=adInputs, imageOutsNames=outNames,  
                           suffix=suffix, funcName='overscanSubtract',   
                           log=log)
        
        # Check the status of the CLManager object, True=continue, False= issue warning
        if clm.status:                     
            # Parameters set by the man.CLManager or the definition 
            # of the primitive 
            clPrimParams = {
              'inimages'    :clm.imageInsFiles(type='string'),
              'gp_outpref'  :clm.prefix,
              'outimages'   :clm.imageOutsFiles(type='string'),
              # This returns a unique/temp log file for IRAF
              'logfile'     :clm.templog.name,      
              'fl_over'     :yes, 
                          }
            
            # Taking care of the biasec->nbiascontam param
            if not biassec == '':
                nbiascontam = clm.nbiascontam(adInputs, biassec)
                log.fullinfo('nbiascontam parameter was updated to = '+
                             str(nbiascontam))
            else: 
                # Do not try to calculate it, just use default value of 4.
                log.fullinfo('Using default nbiascontam parameter = 4')
                nbiascontam = 4
            
            # Parameters from the Parameter file that are adjustable by the user
            clSoftcodedParams = {
               # pyrafBoolean converts the python booleans to pyraf ones
               'fl_trim'    :gt.pyrafBoolean(fl_trim),
               'outpref'    :suffix,
               'fl_vardq'   :fl_vardq,
               'nbiascontam':nbiascontam
                               }
            # Grabbing the default params dict and updating it with 
            # the two above dicts
            clParamsDict = CLDefaultParamsDict('gireduce')
            clParamsDict.update(clPrimParams)
            clParamsDict.update(clSoftcodedParams)
            
            # Logging the parameters that were not defaults
            log.fullinfo('\nParameters set automatically:', 
                         category='parameters')
            # Loop through the parameters in the clPrimParams dictionary
            # and log them
            gt.logDictParams(clPrimParams)
            
            log.fullinfo('\nParameters adjustable by the user:', 
                         category='parameters')
            # Loop through the parameters in the clSoftcodedParams 
            # dictionary and log them
            gt.logDictParams(clSoftcodedParams)
            
            log.debug('Calling the gireduce CL script for inputs '+
                  clm.imageInsFiles(type='string'))
        
            #gemini.gmos.gireduce(**clParamsDict)
            pyraf.iraf.gireduce(**clParamsDict)
            if gemini.gmos.gireduce.status:
                raise Errors.ScienceError('gireduce failed for inputs '+
                             clm.imageInsFiles(type='string'))
            else:
                log.status('Exited the gireduce CL script successfully')
            
            # Renaming CL outputs and loading them back into memory, and 
            # cleaning up the intermediate tmp files written to disk
            # refOuts and arrayOuts are None here
            imageOuts, refOuts, arrayOuts = clm.finishCL() 
            
            # Renaming for symmetry
            adOutputs=imageOuts
            
            # Wrap up logging
            i=0
            for adOut in adOutputs:
                # Verifying gireduce was actually ran on the file
                if adOut.phu_get_key_value('GIREDUCE'): 
                    # If gireduce was ran, then log the changes to the files 
                    # it made
                    log.fullinfo('\nFile '+clm.preCLimageNames()[i]+
                                 ' had its overscan subracted successfully')
                    log.fullinfo('New file name is: '+adOut.filename)
                i = i+1
                # Updating GEM-TLM and OVERSUB time stamps in the PHU
                adOut.history_mark(key='OVERSUB', stomp=False)  
                
                # Updating GEM-TLM (automatic) and BIASCORR time stamps to the PHU
                # and updating logger with updated/added time stamps
                sfm.mark_history(adOutputs=adOut, historyMarkKey='OVERSUB')
        else:
            raise Errors.ScienceError('One of the inputs has not been prepared, the '+
            'overscan_subtract_gmos function can only work on prepared data.')
        
        log.status('**FINISHED** the overscan_subtract_gmos function')
        
        # Return the outputs list, even if there is only one output
        return adOutputs
    except:
        # logging the exact message from the actual exception that was raised
        # in the try block. Then raising a general ScienceError with message.
        log.critical(repr(sys.exc_info()[1]))
        raise 

def overscan_subtract_gmosNEW(adInputs, fl_vardq='AUTO', biassec='',
            numContamCol=None, outNames=None, suffix=None):
    """
    ######### make this take a nbiascontam param as well as a biassec direct...####
    #################################################################################
    This function uses the CL script colbias to calculate and subtract the  
    overscan from the input images.

    note
    The inputs to this function MUST be prepared.

    Either a 'main' type logger object, if it exists, or a null logger 
    (ie, no log file, no messages to screen) will be retrieved/created in the 
    ScienceFunctionManager and used within this function.

    FOR FUTURE
    .  In the future the parameters can be looked into and an upgraded pure 
    Python version can be made to handle things like row based overscan 
    calculations/fitting/modeling... vs the column based used right now, add the
    model, nbiascontam,... params to the functions inputs so the user can choose
    them for themselves.

    :param adInputs: Astrodata inputs to be converted to Electron pixel units
    :type adInputs: Astrodata objects, either a single or a list of objects
    
    :param fl_trim: Trim the overscan region from the frames?
    :type fl_trim: Python boolean (True/False)
    
    :param fl_vardq: Create variance and data quality frames?
    :type fl_vardq: 
        Python boolean (True/False), OR string 'AUTO' to do 
        it automatically if there are VAR and DQ frames in the inputs.
        NOTE: 'AUTO' uses the first input to determine if VAR and DQ frames  
        exist, so, if the first does, then the rest MUST also have them as well.

    :param biassec: biassec parameter of format 
                    '[x1:x2,y1:y2],[x1:x2,y1:y2],[x1:x2,y1:y2]'
    :type biassec: string. If empty string, then header BIASSEC vals are used.
                   Ex. '[2:25,1:2304],[2:25,1:2304],[1032:1055,1:2304]' 
                   is ideal for 2x2 GMOS data.
    
    :param numContamCol: The number of contaminating columns from the data 
                         section. ie. the desired gap size in pixels between
                         the bias section and data section.
    :type numContamCol: int.  typical values are between 4-10. 
                        Default of 4 is used if numContamCol and biassec are 
                        not defined.
                            
    :param outNames: filenames of output(s)
    :type outNames: String, either a single or a list of strings of same length 
                    as adInputs.
    
    :param suffix: string to postpend on the end of the input filenames 
                   (or outNames if not None) for the output filenames.
    :type suffix: string

    """

    # Instantiate ScienceFunctionManager object
    sfm = man.ScienceFunctionManager(adInputs, outNames, suffix, 
                                      funcName='overscan_subtract_gmos') 
    # Perform start up checks of the inputs, prep/check of outnames, and get log
    adInputs, outNames, log = sfm.startUp()
    
    try: 
        # loading and bringing the pyraf related modules into the name-space
        pyraf, gemini, yes, no = pyrafLoader() 
         
        # Changing the standard output so the excess prints while loading IRAF
        # packages does not get displayed
        import StringIO
        SAVEOUT = sys.stdout
        capture = StringIO.StringIO()
        sys.stdout = capture
        from pyraf.iraf import noao
        from pyraf.iraf import imred
        from pyraf.iraf import bias
        noao()
        imred()
        bias()
        # Returning stdout back to normal so prints show on the screen
        sys.stdout = SAVEOUT
        
        # Converting input True/False to yes/no or detecting fl_vardq value
        # if 'AUTO' chosen with autoVardq in the ScienceFunctionManager
        fl_vardq = sfm.autoVardq(fl_vardq) 
         
        # Creating empty list of ad's to be returned that will be filled below
        adOutputs=[]
        
        count = 0
        for ad in adInputs:
            # Preparing input files, lists, parameters... for input to 
            # the CL script
            clm=man.CLManager(imageIns=ad, imageOutsNames=outNames[count],  
                               suffix=suffix, funcName='overscanSubtract',   
                               log=log)
            # Making a deepcopy of the input to work on
            # (ie. a truly new+different object that is a complete copy of the input)
            adOut = deepcopy(ad)
            
            # Getting the names of the temp disk file versions of the inputs
            infilename = clm.imageInsFiles(type='list')[count]       
            # Getting the file names which the CL script will be writting its
            # outputs to.        
            outfilename = clm.imageOutsFiles(type='list')[count]                     
            
            # loop through the SCI extensions in the deepcopied AD and 
            # conduct the bias subtraction plus header updates to them
            for sciExtIn in adOut['SCI']:
                extVerIn = sciExtIn.extver()
                
                # Taking care of the biasec->nbiascontam param
                if not biassec == '':
                    nbiascontam = clm.nbiascontam(adInputs, biassec)
                    log.fullinfo('nbiascontam parameter was updated to = '+
                                 str(nbiascontam))
                ######## make it handle biassec argument of this function#######
                ########## so in here would be if biassec!='', and another section for if is not nbiascontam:...#######
                biassecStr = sciExtIn.get_key_value('BIASSEC')    ########### convert this to use overscan_section() descriptor when exists, but with pretty=True
                
                ######### make a func or mode nbiascontam to handle this
                biassecList = string.sectionStrToIntList(biassecStr) #####
                bsL=biassecList
                if  biassecList[3]<50:
                    #ie biassec on left of chip
                    print '### bias is on left side of chip # '+str(extVerIn)
                    bsLtrimmed = [bsL[0],bsL[1],bsL[2]+1,bsL[3]-7]
                else:
                    #ie biassec on right of chip
                    print '### bias is on right side of chip # '+str(extVerIn)
                    bsLtrimmed = [bsL[0],bsL[1],bsL[2]+7,bsL[3]-1]
                bsLt = bsLtrimmed    
                # converting the 0-based non-inclusive to 1-based inclusive 
                # string for use by colbias.
                biassecStrTrimmed='['+str(bsLt[2]+1)+':'+str(bsLt[3])+','+str(bsLt[0]+1)+':'+str(bsLt[1])+']'
                
                print biassecStr        
                print repr(sciExtIn.data.shape)
                print repr(bsLtrimmed)
                
                # make versions of the input and output filenames for colbias
                colbiasInputfile = infilename+'[SCI,'+str(extVerIn)+']'
                colbiasOutputfile = outfilename+'[SCI,'+str(extVerIn)+',append]'
                
                # delete the previous temp log for colbias if it exists
                if os.path.exists('tmpoverscanlog'):
                    os.remove('tmpoverscanlog')
                
                # fill out colbias input parameter dictionary
                ### maybe add this to geminiCLParDicts.py??
                colbiasParamDict = {'input'     :colbiasInputfile,
                                    'output'    :colbiasOutputfile,
                                    'bias'      :biassecStrTrimmed,
                                    'trim'      :"[]",
                                    'median'    :no,
                                    'interactive':no,
                                    'function'  :'chebyshev',
                                    'order'     :1,
                                    'low_reject':3.,
                                    'high_reject':3.,
                                    'niterate'  :2,
                                    'logfile'   :'tmpoverscanlog',
                                    'graphics'  :'stdgraph',
                                    'cursor'    :'',
                                    'mode'      :'al'                                        
                                    }
                
                # Loop through the parameters in the colbiasParamDict 
                # dictionary and log them
                gt.logDictParams(colbiasParamDict)
                
                log.debug('Calling colbias')
                noao.imred.bias.colbias(**colbiasParamDict)
                
                log.status('colbias finished subtracting the overscan')
                
                # scan through the temp log to find the RMS value calculated
                for line in open('tmpoverscanlog').readlines():
                    if line.find('RMS')>0:
                        rmsStr = line.split(' ')[-1][0:-1]
                # calculate the mean of the overscan region defined by 
                # trimmed biassec
                overscanMean = sciExtIn.data[bsLt[0]:bsLt[1],bsLt[2]:bsLt[3]].mean()
                
                sciExtIn.set_key_value('OVERRMS',rmsStr,"Overscan RMS value from colbias")
                sciExtIn.set_key_value("OVERSCAN", overscanMean, "Overscan mean value")
                log.stdinfo('RMS in the overscan region found to be '+sciExtIn.get_key_value('OVERRMS'))
                log.stdinfo('mean of the overscan region found to be '+str(sciExtIn.get_key_value('OVERSCAN')))
                
                # add or update the VAR frames if requested.
                ## We don't need to update the DQ frams as they are un-effected
                ## by overscan subtraction.
                if fl_vardq==yes:
                    if adOut['VAR']:
                        # update the current variance with 
                        # varOut=vanIn + (RMS of overscan region)
                        log.status('updating variance plane')
                        adOut['VAR',extVerIn].data = np.add(adOut['VAR',extVerIn].data,float(rmsStr)*float(rmsStr))
                    else:
                        log.status('creating new variance plane')
                        initialVar = gt.calculateInitialVarianceArray(sciExtIn)
                        varheader = gt.createInitialVarianceHeader(
                                                                extver=extVerIn,
                                                                shape=initialVar.shape)
                        # Turning individual variance header and data 
                        # into one astrodata instance
                        varAD = AstroData(header=varheader, data=initialVar)
                        adOut.append(varAD)
                        # update the current variance with 
                        # varOut=vanIn + (RMS of overscan region)
                        adOut['VAR',extVerIn].data = np.add(adOut['VAR',extVerIn].data,float(rmsStr)*float(rmsStr))     
                
            # Renaming CL outputs and loading them back into memory, and 
            # cleaning up the intermediate tmp files written to disk
            # refOuts and arrayOuts are None here
            imageOuts, refOuts, arrayOuts = clm.finishCL() 
            
            # loop to extract SCI extension data from the colbias ouputs.
            # This is because colbias doesn't correctly re-create the MEF 
            # structure of the input, only the single extensions, and it plays  
            # with the header keys in naughty ways. bad colbias bad :-P
            for sciExtOut in imageOuts[0]['SCI']:
                extVerOut = sciExtOut.extver()
                log.fullinfo('copying colbias output SCI data frame '+
                             str(extVerOut)+' to adOut SCI data')
                adOut['SCI',extVerOut].data = sciExtOut.data
                
            # adding the final output ad to the outputs list
            adOutputs.append(adOut)
        
        log.status('**FINISHED** the overscan_subtract_gmos function')
        
        # Return the outputs list, even if there is only one output
        return adOutputs
    except:
        # logging the exact message from the actual exception that was raised
        # in the try block. Then raising a general ScienceError with message.
        log.critical(repr(sys.exc_info()[1]))
        raise 

def overscan_trim(adInputs, outNames=None, suffix=None):
    """
    This function uses AstroData to trim the overscan region 
    from the input images and update their headers.
    
    NOTE: The inputs to this function MUST be prepared. 
    
    Either a 'main' type logger object, if it exists, or a null logger 
    (ie, no log file, no messages to screen) will be retrieved/created in the 
    ScienceFunctionManager and used within this function.
    
    :param adInputs: Astrodata inputs to have DQ extensions added to
    :type adInputs: Astrodata objects, either a single or a list of objects
    
    :param outNames: filenames of output(s)
    :type outNames: String, either a single or a list of strings of same length
                    as adInputs.
    
    :param suffix: string to add on the end of the input filenames 
                    (or outNames if not None) for the output filenames.
    :type suffix: string
    
    """
    
    # Instantiate ScienceFunctionManager object
    sfm = man.ScienceFunctionManager(adInputs, outNames, suffix,
                                      funcName='overscan_trim') 
    # Perform start up checks of the inputs, prep/check of outnames, and get log
    adInputs, outNames, log = sfm.startUp()
    
    try:
        # Set up counter for looping through outNames list
        count=0
        
        # Creating empty list of ad's to be returned that will be filled below
        adOutputs=[]
        
        # Loop through the inputs to perform the non-linear and saturated
        # pixel searches of the SCI frames to update the BPM frames into
        # full DQ frames. 
        for ad in adInputs:  
            # Making a deepcopy of the input to work on
            # (ie. a truly new+different object that is a complete copy of the input)
            adOut = deepcopy(ad)
                             
            # To clean up log and screen if multiple inputs
            log.fullinfo('+'*50, category='format')    
            
            for sciExt in adOut['SCI']:
                # Getting the data section 
                # as a direct string from header
                datasecStr = str(sciExt.data_section(pretty=True))
                # int list of form [y1, y2, x1, x2] 0-based and non-inclusive
                datsecList = sciExt.data_section().as_pytype()
                dsl = datsecList
                
                # Updating logger with the section being kept
                log.stdinfo('\nfor '+adOut.filename+' extension '+
                            str(sciExt.extver())+
                            ', keeping the data from the section '+
                            datasecStr,'science')
                # Trimming the data section from input SCI array
                # and making it the new SCI data
                # NOTE: first elements of arrays in python are inclusive
                #       while last ones are exclusive, thus a 1 must be 
                #       added for the final element to be included.
                sciExt.data=sciExt.data[dsl[0]:dsl[1],dsl[2]:dsl[3]]
                # Updating header keys to match new dimensions
                sciExt.header['NAXIS1'] = dsl[3]-dsl[2]
                sciExt.header['NAXIS2'] = dsl[1]-dsl[0]
                newDataSecStr = '[1:'+str(dsl[3]-dsl[2])+',1:'+\
                                str(dsl[1]-dsl[0])+']' 
                sciExt.header['DATASEC']=newDataSecStr
                sciExt.header.update('TRIMSEC', datasecStr, 
                                   'Data section prior to trimming')
                # Updating logger with updated/added keywords to each SCI frame
                log.fullinfo('*'*50, category='header')
                log.fullinfo('File = '+adOut.filename, category='header')
                log.fullinfo('~'*50, category='header')
                log.fullinfo('SCI extension number '+str(sciExt.extver())+
                             ' keywords updated/added:\n', 'header')
                log.fullinfo('NAXIS1= '+str(sciExt.get_key_value('NAXIS1')),
                            category='header')
                log.fullinfo('NAXIS2= '+str(sciExt.get_key_value('NAXIS2')),
                             category='header')
                log.fullinfo('DATASEC= '+newDataSecStr, category='header')
                log.fullinfo('TRIMSEC= '+datasecStr, category='header')
                    
            # Updating GEM-TLM (automatic) and BIASCORR time stamps to the PHU
            # and updating logger with updated/added time stamps
            sfm.mark_history(adOutputs=adOut, historyMarkKey='OVERTRIM')       
            
            # Setting 'TRIMMED' to 'yes' in the PHU and updating the log
            adOut.phu_set_key_value('TRIMMED','yes','Overscan section trimmed')
            log.fullinfo('Another PHU keywords added:\n', 'header')
            log.fullinfo('TRIMMED = '+adOut.phu_get_key_value('TRIMMED')+'\n', 
                         category='header')
            
            # Appending to output list
            adOutputs.append(adOut)

            count = count+1
        
        log.status('**FINISHED** the overscan_trim function')
        
        # Return the outputs list, even if there is only one output
        return adOutputs
    except:
        # logging the exact message from the actual exception that was raised
        # in the try block. Then raising a general ScienceError with message.
        log.critical(repr(sys.exc_info()[1]))
        raise 
                    
def subtract_bias(adInputs, biases=None ,fl_vardq='AUTO', fl_trim=False, 
                fl_over=False, outNames=None, suffix=None):
    """
    This function will subtract the biases from the inputs using the 
    CL script gireduce.
    
    WARNING: The gireduce script used here replaces the previously 
    calculated DQ frames with its own versions.  This may be corrected 
    in the future by replacing the use of the gireduce
    with a Python routine to do the bias subtraction.
    
    NOTE: The inputs to this function MUST be prepared. 

    Either a 'main' type logger object, if it exists, or a null logger 
    (ie, no log file, no messages to screen) will be retrieved/created in the 
    ScienceFunctionManager and used within this function.
    
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
    :type outNames: String, either a single or a list of strings of same length 
                    as adInputs.
    
    :param suffix: string to add on the end of the input filenames 
                   (or outNames if not None) for the output filenames.
    :type suffix: string
    
    """
    # Instantiate ScienceFunctionManager object
    sfm = man.ScienceFunctionManager(adInputs, outNames, suffix, 
                                      funcName='subtract_bias')
    # Perform start up checks of the inputs, prep/check of outnames, and get log
    adInputs, outNames, log = sfm.startUp()
     
    # casting biases into a list if not one all ready for later indexing
    if not isinstance(biases, list):
        biases = [biases]
    
    # checking the inputs have matching filters, binning and SCI shapes.
    gt.checkInputsMatch(adInsA=biases, adInsB=adInputs) 
        
    try:
        # Set up counter for looping through outNames list
        count=0
        
        # Creating empty list of ad's to be returned that will be filled below
        adOutputs=[]
            
        # loading and bringing the pyraf related modules into the name-space
        pyraf, gemini, yes, no = pyrafLoader()
            
        # Performing work in a loop, so that different biases may be
        # used for each input as gireduce only allows one bias input per run.
        for ad in adInputs:
            
            # To clean up log and screen if multiple inputs
            log.fullinfo('+'*50, category='format')    
            
            # Converting input True/False to yes/no or detecting fl_vardq value
            # if 'AUTO' chosen with autoVardq in the ScienceFunctionManager
            fl_vardq = sfm.autoVardq(fl_vardq)
            
            # Getting the right dark for this input
            if len(biases)>1:
                bias = biases[count]
            else:
                bias = biases[0]
                
            # Preparing input files, lists, parameters... for input to 
            # the CL script
            clm = man.CLManager(imageIns=ad, imageOutsNames=outNames[count], 
                               refIns=bias, suffix=suffix,  
                               funcName='biasCorrect', log=log)
            
            # Check the status of the CLManager object, True=continue, False= issue warning
            if clm.status:               
                    
                # Parameters set by the man.CLManager or the definition of the function 
                clPrimParams = {
                  'inimages'    :clm.imageInsFiles(type='string'),
                  'gp_outpref'  :clm.prefix,
                  'outimages'   :clm.imageOutsFiles(type='string'),
                  # This returns a unique/temp log file for IRAF 
                  'logfile'     :clm.templog.name,     
                  'fl_bias'     :yes,
                  # Possibly add this to the params file so the user can override
                  # this input file
                  'bias'        :clm.refInsFiles(type='string'),     
                              }
                    
                # Parameters from the Parameter file adjustable by the user
                clSoftcodedParams = {
                   # pyrafBoolean converts the python booleans to pyraf ones
                   'fl_trim'    :gt.pyrafBoolean(fl_trim),
                   'outpref'    :suffix,
                   'fl_over'    :gt.pyrafBoolean(fl_over),
                   'fl_vardq'   :gt.pyrafBoolean(fl_vardq)
                                   }
                # Grabbing the default params dict and updating it 
                # with the two above dicts
                clParamsDict = CLDefaultParamsDict('gireduce')
                clParamsDict.update(clPrimParams)
                clParamsDict.update(clSoftcodedParams)
            
                # Logging the parameters that were not defaults
                log.fullinfo('\nParameters set automatically:', 
                             category='parameters')
                # Loop through the parameters in the clPrimParams dictionary
                # and log them
                gt.logDictParams(clPrimParams)
                
                log.fullinfo('\nParameters adjustable by the user:', 
                             category='parameters')
                # Loop through the parameters in the clSoftcodedParams 
                # dictionary and log them
                gt.logDictParams(clSoftcodedParams)
                
                log.debug('calling the gireduce CL script for inputs '+
                                        clm.imageInsFiles(type='string'))
            
                gemini.gmos.gireduce(**clParamsDict)
        
                if gemini.gmos.gireduce.status:
                    raise Errors.ScienceError('gireduce failed for inputs '+
                                 clm.imageInsFiles(type='string'))
                else:
                    log.status('Exited the gireduce CL script successfully')
                    
                # Renaming CL outputs and loading them back into memory 
                # and cleaning up the intermediate temp files written to disk
                # refOuts and arrayOuts are None here
                imageOuts, refOuts, arrayOuts = clm.finishCL() 
                
                # There is only one at this point so no need to perform a loop
                # CLmanager outputs a list always, so take the 0th
                adOut = imageOuts[0]
                
                # Varifying gireduce was actually ran on the file
                # then logging file names of successfully reduced files
                if adOut.phu_get_key_value('GIREDUCE'): 
                    log.fullinfo('\nFile '+clm.preCLimageNames()[0]+
                                 ' was bias subracted successfully')
                    log.fullinfo('New file name is: '+adOut.filename)
  
                # Updating GEM-TLM (automatic) and BIASCORR time stamps to the PHU
                # and updating logger with updated/added time stamps
                sfm.mark_history(adOutputs=adOut, historyMarkKey='BIASCORR')

                # Reseting the value set by gireduce to just the filename
                # for clarity
                adOut.phu_set_key_value('BIASIM', 
                                     os.path.basename(bias.filename)) 
                
                # Updating log with new BIASIM header key
                log.fullinfo('Another PHU keywords added:\n', 'header')
                log.fullinfo('BIASIM = '+adOut.phu_get_key_value('BIASIM')+'\n', 
                             category='header')
           
                # Appending to output list
                adOutputs.append(adOut)

                count = count+1
                
            else:
                raise Errors.ScienceError('One of the inputs has not been prepared,'+
                'the combine function can only work on prepared data.')
            
        log.warning('The CL script gireduce REPLACED the previously '+
                    'calculated DQ frames')
        
        log.status('**FINISHED** the subtract_bias function')
        
        # Return the outputs list, even if there is only one output
        return adOutputs
    except:
        # logging the exact message from the actual exception that was raised
        # in the try block. Then raising a general ScienceError with message.
        log.critical(repr(sys.exc_info()[1]))
        raise 
    
def subtract_biasNEW(adInputs, biases=None, fl_vardq='AUTO', outNames=None, suffix=None):
    """
    This function will subtract the SCI of the input biases from each SCI frame 
    of the inputs.  New VAR frames will be calculated and appended to the 
    output.  If both the bias and input have DQ frames, they will be 
    propogated using a bitwise_or addition. 
    
    NOTE: Any pre-existing VAR frames will be replaced with new ones after the 
    subtraction is complete.
    
    This is all conducted in pure Python through the arith "toolbox" of 
    astrodata. The varutil module is used to create the output VAR frames.
       
    Either a 'main' type logger object, if it exists, or a null logger 
    (ie, no log file, no messages to screen) will be retrieved/created in the 
    ScienceFunctionManager and used within this function.
    
    :param adInputs: Astrodata input science data
    :type adInputs: Astrodata objects, either a single or a list of objects
    
    :param biases: The bias(es) to be added to the input(s).
    :type biases: AstroData objects in a list, or a single instance.
                  Note: If there are multiple inputs and one bias provided, 
                  then the same bias will be applied to all inputs; else the 
                  biases list must match the length of the inputs.
    
    :param outNames: filenames of output(s)
    :type outNames: String, either a single or a list of strings of same length 
                    as adInputs.
    
    :param suffix: string to add on the end of the input filenames 
                   (or outNames if not None) for the output filenames.
    :type suffix: string
    """
    # Instantiate ScienceFunctionManager object
    sfm = man.ScienceFunctionManager(adInputs, outNames, suffix,
                                                    funcName='subtract_bias') 
    # Perform start up checks of the inputs, prep/check of outnames, and get log
    adInputs, outNames, log = sfm.startUp()
    
    # casting darks into a list if not one all ready for later indexing
    if not isinstance(biases, list):
        biases = [biases]
    
    # checking the inputs have matching filters, binning and SCI shapes.
    gt.checkInputsMatch(adInsA=biases, adInsB=adInputs)
    
    try:
        # Set up counter for looping through outNames list
        count=0
        
        # Creating empty list of ad's to be returned that will be filled below
        adOutputs=[]
        
        # Loop through the inputs 
        for ad in adInputs:  
            if ad.phu_get_key_value('BIASIM'):
                # bias image has all ready been subtracted, so don't do it again
                adOut = ad
            else:
                # Getting the right bias for this input
                if len(biases)>1:
                    bias = biases[count]
                else:
                    bias = biases[0]
               
                # sub each bias SCI  from each input SCI and handle the updates to 
                # the DQ and VAR frames.
                # the sub function of the arith toolbox performs a deepcopy so
                # it doesn't need to be done here. 
                adOut = ad.sub(bias)
            
                # adding name of bias image used for subtraction to PHU
                adOut.phu_set_key_value('BIASIM',os.path.basename(bias.filename), 'bias image subtracted')
                
            # adding or replacing current VAR's with new ones
            for sciExt in adOut['SCI']:
                sciExtVer = sciExt.extver()
                # Using the toolbox function calculateInitialVarianceArray
                # to conduct the actual calculation following:
                # var = (read noise/gain)**2 + max(data,0.0)/gain
                varArray = varutil.calculateInitialVarianceArray(sciExt)
                 
                # Creating the variance frame's header and updating it     
                varHeader = varutil.createInitialVarianceHeader(
                                                        extver=sciExtVer,
                                                        shape=varArray.shape)
                
                # append as new extension or replace data and header of current
                if adOut['VAR',sciExtVer]:
                    # VAR extensions exists, so replace its header and data
                    adOut['VAR',sciExtVer].data = varArray
                    adOut['VAR',sciExtVer].header = varHeader
                    log.status('Previous VAR frame version '+str(sciExtVer)+
                               ', had its data and header elements replaced')
                else:
                    # extension doens't exist so make it and append it
                    
                    # Turning individual variance header and data 
                    # into one astrodata instance
                    varAD = AstroData(header=varHeader, data=varArray)
            
                    # Appending variance astrodata instance onto input one
                    log.debug('Appending new VAR HDU onto the file '
                                 +adOut.filename)
                    adOut.append(varAD)
                    log.status('Appending VAR frame '+str(sciExtVer)+
                               ' complete for '+adOut.filename)
            
            # renaming the output ad filename
            adOut.filename = outNames[count]
                    
            adOut.info() ############       
            
            log.status('File name updated to '+adOut.filename+'\n')
            
            # Updating GEM-TLM (automatic) and SUBBIAS time stamps to the PHU
            # and updating logger with updated/added time stamps
            sfm.mark_history(adOutputs=adOut, historyMarkKey='SUBBIAS')
        
            # Appending to output list
            adOutputs.append(adOut)
    
            count=count+1
                
        log.status('**FINISHED** the subtract_bias function')
        # Return the outputs (list or single, matching adInputs)
        return adOutputs
    except:
        # logging the exact message from the actual exception that was raised
        # in the try block. Then raising a general ScienceError with message.
        log.critical(repr(sys.exc_info()[1]))
        raise 

