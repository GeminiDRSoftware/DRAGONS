import sys, StringIO, os

from astrodata.adutils import gemLog
from astrodata import Descriptors
from astrodata.data import AstroData
from gempy.instruments import geminiTools as gemt
from gempy.instruments import gmosTools as gmost
from primitives_GEMINI import GEMINIPrimitives, pyrafLoader
import numpy as np
import pyfits as pf
import shutil
from astrodata.ConfigSpace import lookupPath

log=gemLog.getGeminiLog()

class GMOSException:
    """ This is the general exception the classes and functions in the
    Structures.py module raise.
    
    """
    def __init__(self, message='Exception Raised in Recipe System'):
        """This constructor takes a message to print to the user."""
        self.message = message
    def __str__(self):
        """This str conversion member returns the message given 
        by the user (or the default message)
        when the exception is not caught."""
        return self.message

class GMOSPrimitives(GEMINIPrimitives):
    """ 
    This is the class of all primitives for the GMOS level of the type 
    hierarchy tree.  It inherits all the primitives to the level above
    , 'GEMINIPrimitives'.
    
    """
    astrotype = 'GMOS'
    
    def init(self, rc):
        GEMINIPrimitives.init(self, rc)
        return rc
     
    def addBPM(self,rc):
        """
        This primitive is used by the general addDQ primitive of 
        primitives_GEMINI to add the appropriate BPM (Bad Pixel Mask)
        to the inputs.  This function will add the BPM as frames matching
        that of the SCI frames and ensure the BPM's data array is the same 
        size as that of the SCI data array.
        
        Using this approach, rather than appending the BPM in the addDQ allows
        for specialized BPM processing to be done in the instrument specific
        primitive sets where it belongs.
        
        """
        
        try:
            log.status('*STARTING* to add the BPM frame(s) to the input data')
            
            #$$$$$$$$$$$$$ TO BE callibration search, correct when ready $$$$$$$
            BPM_11 = AstroData(lookupPath('Gemini/GMOS/GMOS_BPM_11.fits'))
            BPM_22 = AstroData(lookupPath('Gemini/GMOS/GMOS_BPM_22.fits'))
            #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            
            for ad in rc.getInputs(style='AD'):
                # Check if BPM extensions all ready exist for this file
                if not  ad['BPM']:
                    for sciExt in ad['SCI']:
                        #$$ edit when callibration service works for BPMs $$$$$$
                        if sciExt.getKeyValue('CCDSUM') == '1 1':
                            BPMArrayIn = BPM_11[('DQ', sciExt.extver())].data
                            BPMfilename = 'GMOS_BPM_11.fits'
                        elif sciExt.getKeyValue('CCDSUM') == '2 2':
                            BPMArrayIn = BPM_22[('DQ', sciExt.extver())].data
                            BPMfilename = 'GMOS_BPM_22.fits'
                        else:
                            BPMArrayIn = np.zeros(sciExt.data.shape, 
                                                  dtype=np.int16)
                            log.error('CCDSUM is not 1x1 or 2x2, using'+
                                      ' zeros array for BPM')
                            BPMfilename = 'None'
                        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$   
                        
                        # logging the BPM file being used for this SCI extension
                        log.fullinfo('SCI extension number '+
                                     str(sciExt.extver())+', of file '+
                                     ad.filename+ ' is matched to BPM file '+
                                     BPMfilename)
                        
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
                        
                        # Using renameExt to correctly set the EXTVer and 
                        # EXTNAME values in the header   
                        bpmAD.renameExt('BPM', ver=sciExt.extver())
                        
                        # Appending BPM astrodata instance to the input one
                        log.debug('Appending new BPM HDU onto the file '+ 
                                  ad.filename)
                        ad.append(bpmAD)
                        log.status('Appending BPM complete for '+ ad.filename)
                        
               # If BPM frames exist, send a critical message to the logger
                else:
                    log.critical('BPM frames all ready exist for '+ad.filename+
                                 ', so addBPM will add new ones') 

                # Updating GEM-TLM (automatic) and ADDBPM time stamps to the PHU
                ad.historyMark(key='ADDBPM', stomp=False) 
                # Updating logger with updated/added time stamps
                log.fullinfo('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
                             , category='header')
                log.fullinfo('PHU keywords updated/added:\n', category='header')
                log.fullinfo('GEM-TLM = '+ad.phuGetKeyValue('GEM-TLM'),
                             category='header')
                log.fullinfo('ADDBPM = '+ad.phuGetKeyValue('ADDBPM'), 
                             category='header')
                log.fullinfo('------------------------------------------------'
                             , category='header')
                
                # Reporting the updated file to the reduction context
                rc.reportOutput(ad)   
                
            log.status('*FINISHED* adding the BPM to the inputs') 
        except:
            log.critical('Problem processing one of '+rc.inputsAsStr())
            raise  
            
        yield rc       
        
    def biasCorrect(self, rc):
        """
        This primitive will subtract the biases from the inputs using the 
        CL script gireduce.
        
        WARNING: The gireduce script used here replaces the previously 
        calculated DQ frames with its own versions.  This may be corrected 
        in the future by replacing the use of the gireduce
        with a Python routine to do the bias subtraction.
        
        """
        # Loading and bringing the pyraf related modules into the name-space
        pyraf, gemini, yes, no = pyrafLoader(rc)
        
        try:
            log.status('*STARTING* to subtract the bias from the inputs')
            
            # Writing input files to disk with prefixes onto their file 
            # names so they can be deleted later easily 
            clm = gemt.CLManager(rc)
            #clm.LogCurParams()
            
            # Getting the bias file for the first file of the inputs and 
            # assuming it is the same for all the inputs. This should be 
            # corrected in the future to be more intelligent and get the 
            # correct bias for each input individually if they are not 
            # all the same. Then gireduce can be called in a loop with 
            # one flat and one bias, this will work well with the CLManager
            # as that was how i wrote this prim originally.
            ad = rc.getInputs(style='AD')[0]
            processedBias = rc.getCal(ad,'bias')
            log.status('Using bias '+processedBias+' to correct the inputs')
            
            log.critical('prim_Gmos204: '+rc.inputsAsStr())
            
            # Parameters set by the gemt.CLManager or the definition of the prim 
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
              'Stdout'      :gemt.IrafStdout(), 
              # This is actually in the default dict but wanted to show it again
              'Stderr'      :gemt.IrafStdout(), 
              # This is actually in the default dict but wanted to show it again
              'verbose'     :yes                
                          }
            # Parameters from the Parameter file adjustable by the user
            clSoftcodedParams = {
               # pyrafBoolean converts the python booleans to pyraf ones
               'fl_trim'    :gemt.pyrafBoolean(rc['fl_trim']),
               'outpref'    :rc['postpend'],
               'fl_over'    :gemt.pyrafBoolean(rc['fl_over']),
               'fl_vardq'   :gemt.pyrafBoolean(rc['fl_vardq'])
                               }
            # Grabbing the default params dict and updating it 
            # with the two above dicts
            clParamsDict = CLDefaultParamsDict('gireduce')
            clParamsDict.update(clPrimParams)
            clParamsDict.update(clSoftcodedParams)
            
            # Logging the values in the soft and prim parameter dictionaries
            log.fullinfo('\nParameters set by the CLManager or dictated by '+
                         'the definition of the primitive:\n', 
                         category='parameters')
            gemt.LogDictParams(clPrimParams)
            log.fullinfo('\nUser adjustable parameters in the parameters '+
                         'file:\n', category='parameters')
            gemt.LogDictParams(clSoftcodedParams)
            
            log.debug('calling the gireduce CL script for inputs '+
                      clm.inputsAsStr())

            gemini.gmos.gireduce(**clParamsDict)
            
            if gemini.gmos.gireduce.status:
                 log.critical('gireduce failed for '+rc.inputsAsStr())
                 raise GMOSException('gireduce failed')
            else:
                 log.status('Exited the gireduce CL script successfully')
            
            # Renaming CL outputs and loading them back into memory, and 
            # cleaning up the intermediate tmp files written to disk
            clm.finishCL()
            
            # Wrap up logging
            i=0
            for ad in rc.getOutputs(style='AD'):
                # Varifying gireduce was actually ran on the file
                # then logging file names of successfully reduced files
                if ad.phuGetKeyValue('GIREDUCE'): 
                    log.fullinfo('File '+clm.preCLNames()[i]+
                                 ' was bias subracted successfully')
                    log.fullinfo('New file name is: '+ad.filename)
                i=i+1
                # Updating the GEM-TLM (automatic) and BIASCORR time stamps in 
                # the PHU
                ad.historyMark(key='BIASCORR', stomp=False)  
                
                # Reseting the value set by gireduce to just the filename
                # for clarity
                ad.phuSetKeyValue('BIASIM', os.path.basename(processedBias)) 
                
                # Updating log with new GEM-TLM value and BIASIM header keys
                log.fullinfo('************************************************'
                             , category='header')
                log.fullinfo('File = '+ad.filename, category='header')
                log.fullinfo('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
                             , category='header')
                log.fullinfo('PHU keywords updated/added:\n', 'header')
                log.fullinfo('GEM-TLM = '+ad.phuGetKeyValue('GEM-TLM'), 
                             category='header')
                log.fullinfo('BIASCORR = '+ad.phuGetKeyValue('BIASCORR'), 
                             category='header')
                log.fullinfo('BIASIM = '+os.path.basename(processedBias)+'\n', 
                             category='header')
                
            log.warning('The CL script gireduce REPLACED the previously '+
                        'calculated DQ frames')
            
            log.status('*FINISHED* subtracting the bias from the input flats')
        except:
            log.critical('Problem processing one of '+rc.inputsAsStr())
            raise 
            
        yield rc

    def display(self, rc):
        """ This is a primitive for displaying GMOS data.
            It utilizes the IRAF routine gdisplay and requires DS9 to be running
            before this primitive is called.
        """
        from astrodata.adutils.future import gemDisplay
        pyraf, gemini, yes, no = pyrafLoader(rc)
        pyraf.iraf.set(stdimage='imtgmos')
        ds = gemDisplay.getDisplayService()
        for i in range(0, len(rc.inputs)):   
            inputRecord = rc.inputs[i]
            try:
                gemini.gmos.gdisplay( inputRecord.filename, i+1, fl_imexam=pyraf.iraf.no,
                                Stdout = rc.getIrafStdout(), Stderr = rc.getIrafStderr() )
            except:
                # This exception should allow for a smooth exiting if there is an 
                # error with gdisplay, most likely due to DS9 not running yet
                GMOSException('ERROR occurred while trying to display '+str(inputRecord.filename)
                                    +' ensure that DS9 is running and try again')
                
            # this version had the display id conversion code which we'll need to redo
            # code above just uses the loop index as frame number
            #gemini.gmos.gdisplay( inputRecord.filename, ds.displayID2frame(rq.disID), fl_imexam=iraf.no,
            #    Stdout = coi.getIrafStdout(), Stderr = coi.getIrafStderr() )
            
        yield rc

    def flatCorrect(self,rc):
        """
        This primitive performs a flat correction by dividing the inputs by a 
        processed flat similar to the way gireduce would perform this operation
        but written in pure python.  
        It is currently assumed that the same flat file may be applied to all
        input images.
        
        """
        try:
            log.status('*STARTING* to flat correct the inputs')
            
            # Retrieving the appropriate flat for the first of the inputs
            adOne = rc.getInputs(style='AD')[0]
            processedFlat = AstroData(rc.getCal(adOne,'flat'))
            
            for ad in rc.getInputs(style='AD'):
                log.status('Input flat file being used for flat correction '
                           +processedFlat.filename)
                log.debug('Calling ad.div on '+ad.filename)
                
                adOut = ad.div(processedFlat)
                log.status('ad.div successfully flat corrected '+ad.filename)
                
                # Updating GEM-TLM (automatic) and FLATCORR time stamps to the 
                # PHU
                adOut.historyMark(key='FLATCORR', stomp=False)
                
                # Updating the file name with the postpend/outsuffix for this
                # primitive and then reporting the new file to the reduction 
                # context
                log.debug('Calling gemt.fileNameUpdater on '+ad.filename)
                adOut.filename = gemt.fileNameUpdater(ad.filename, 
                                                      postpend=rc['postpend'], 
                                                      strip=False)
                log.status('File name updated to '+ad.filename)
                rc.reportOutput(adOut)   
                
                # Updating logger with new GEM-TLM value
                log.fullinfo('************************************************'
                             , category='header')
                log.fullinfo('File = '+adOut.filename, category='header')
                log.fullinfo('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
                             , category='header')
                log.fullinfo('PHU keywords updated/added:\n', 'header')
                log.fullinfo('GEM-TLM = '+ad.phuGetKeyValue('GEM-TLM'), 
                             category='header')
                log.fullinfo('FLATCORR = '+ad.phuGetKeyValue('FLATCORR'), 
                             category='header')
                log.fullinfo('------------------------------------------------'
                             , category='header')    

            log.status('*FINISHED* flat correcting the inputs')  
        except:
            log.critical('Problem processing one of '+rc.inputsAsStr())
            raise  
        yield rc
        
    def getProcessedBias(self,rc):
        """
        A primitive to search and return the appropriate calibration bias from
        a server for the given inputs.
        
        """
        rc.rqCal('bias', rc.getInputs(style='AD'))
        yield rc
        
    def getProcessedFlat(self,rc):
        """
        A primitive to search and return the appropriate calibration flat from
        a server for the given inputs.
        
        """
        rc.rqCal('flat', rc.getInputs(style='AD'))
        yield rc

    def localGetProcessedBias(self,rc):
        """
        A prim that works with the calibration system (MAYBE), but as it isn't 
        written yet this simply copies the bias file from the stored processed 
        bias directory and reports its name to the reduction context. 
        This is the basic form that the calibration system will work as well 
        but with proper checking for what the correct bias file would be rather 
        than my oversimplified checking the bining alone.
        
        """
        try:
            packagePath = sys.argv[0].split('gemini_python')[0]
            calPath = 'gemini_python/test_data/test_cal_files/processed_biases/'
            
            for ad in rc.getInputs(style='AD'):
                if ad.extGetKeyValue(1,'CCDSUM') == '1 1':
                    log.error('NO 1x1 PROCESSED BIAS YET TO USE')
                    raise 'error'
                elif ad.extGetKeyValue(1,'CCDSUM') == '2 2':
                    biasfilename = 'N20020214S022_preparedBias.fits'
                    if not os.path.exists(os.path.join('.reducecache/'+
                                                       'storedcals/retrievd'+
                                                       'biases', biasfilename)):
                        shutil.copy(packagePath+calPath+biasfilename, 
                                    '.reducecache/storedcals/retrievedbiases')
                    rc.addCal(ad,'bias', 
                              os.path.join('.reducecache/storedcals/retrieve'+
                                           'dbiases',biasfilename))
                else:
                    log.error('CCDSUM is not 1x1 or 2x2 for the input flat!!')
           
        except:
            log.critical('Problem preparing one of '+rc.inputsAsStr())
            raise
        yield rc
   
    def localGetProcessedFlat(self,rc):
        """
        A prim that works with the calibration system (MAYBE), but as it 
        isn't written yet this simply copies the bias file from the stored 
        processed bias directory and reports its name to the reduction 
        context. this is the basic form that the calibration system will work 
        as well but with proper checking for what the correct bias file would 
        be rather than my oversimplified checking
        the binning alone.
        
        """
        try:
            packagePath=sys.argv[0].split('gemini_python')[0]
            calPath='gemini_python/test_data/test_cal_files/processed_flats/'
            
            for ad in rc.getInputs(style='AD'):
                if ad.extGetKeyValue(1,'CCDSUM') == '1 1':
                    log.error('NO 1x1 PROCESSED BIAS YET TO USE')
                    raise 'error'
                elif ad.extGetKeyValue(1,'CCDSUM') == '2 2':
                    flatfilename = 'N20020211S156_preparedFlat.fits'
                    if not os.path.exists(os.path.join('.reducecache/storedca'+
                                                       'ls/retrievedflats', 
                                                       flatfilename)):
                        shutil.copy(packagePath+calPath+flatfilename, 
                                    '.reducecache/storedcals/retrievedflats')
                    rc.addCal(ad,'flat', os.path.join('.reducecache/storedca'+
                                                      'ls/retrievedflats', 
                                                      flatfilename))
                else:
                    log.error('CCDSUM is not 1x1 or 2x2 for the input image!!')
           
        except:
            log.critical('Problem retrieving one of '+rc.inputsAsStr())
            raise
        
        yield rc

    def mosaicDetectors(self,rc):
        """
        This primitive will mosaic the SCI frames of the input images, 
        along with the VAR and DQ frames if they exist.  
        
        """
        # loading and bringing the pyraf related modules into the name-space
        pyraf, gemini, yes, no = pyrafLoader(rc)
        
        try:
            log.status('*STARTING* to mosaic the input images SCI extensions'+
                       ' together')
            # Writing input files to disk with prefixes onto their file names so 
            # they can be deleted later easily 
            clm = gemt.CLManager(rc)
            #clm.LogCurParams() 
            
            # Determining if gmosaic should propigate the VAR and DQ frames 
            ad=rc.getInputs(style='AD')[0]
            if ad.countExts('VAR')==ad.countExts('DQ')==ad.countExts('SCI'):
                fl_vardq=yes
            else:
                fl_vardq=no
                
            # Parameters set by the gemt.CLManager or the definition of the prim 
            clPrimParams = {
              # Retrieving the inputs as a string of filenames
              'inimages'    :clm.inputsAsStr(),
              # Setting the value of FL_vardq set above
              'fl_vardq'    :fl_vardq,
              # This returns a unique/temp log file for IRAF 
              'logfile'     :clm.logfile(),
              # This is actually in the default dict but wanted to show it again     
              'Stdout'      :gemt.IrafStdout(), 
              # This is actually in the default dict but wanted to show it again
              'Stderr'      :gemt.IrafStdout(), 
              # This is actually in the default dict but wanted to show it again
              'verbose'     :yes                
                          }
            # Parameters from the Parameter file adjustable by the user
            clSoftcodedParams = {
              # pyrafBoolean converts the python booleans to pyraf ones
              'fl_paste'    :gemt.pyrafBoolean(rc['fl_paste']),
              'outpref'     :rc['postpend'],
              'outimages'   :rc['outimages'],
              'geointer'    :rc['interp_function'],
                              }
            # Grabbing the default params dict and updating it with 
            # the two above dicts
            clParamsDict = CLDefaultParamsDict('gmosaic')
            clParamsDict.update(clPrimParams)
            clParamsDict.update(clSoftcodedParams)
            
            # Logging the values in the soft and prim parameter dictionaries
            log.fullinfo('\nParameters set by the CLManager or dictated by '+
                         'the definition of the primitive:\n', 
                         category='parameters')
            gemt.LogDictParams(clPrimParams)
            log.fullinfo('\nUser adjustable parameters in the parameters '+
                         'file:\n', category='parameters')
            gemt.LogDictParams(clSoftcodedParams)
            
            log.debug('calling the gmosaic CL script for inputs '+\
                      clm.inputsAsStr())
            
            gemini.gmos.gmosaic(**clParamsDict)
            
            if gemini.gmos.gmosaic.status:
                log.critical('gmosaic failed for '+rc.inputsAsStr())
                raise GMOSException('gmosaic failed')
            else:
                log.fullinfo('exited the gmosaic CL script successfully')
            
            # Renaming CL outputs and loading them back into memory, and  
            # cleaning up the intermediate tmp files written to disk
            clm.finishCL()
            # Wrap up logging
            i=0
            for ad in rc.getOutputs(style='AD'):
                # Varifying gireduce was actually ran on the file
                # then logging file names of successfully reduced files
                if ad.phuGetKeyValue('GMOSAIC'): 
                    log.fullinfo('file '+clm.preCLNames()[i]+\
                                 ' mosaiced successfully')
                    log.fullinfo('New file name is: '+ad.filename)
                i=i+1
                # Updating GEM-TLM (automatic) and MOSAIC time stamps to the PHU
                ad.historyMark(key='MOSAIC', stomp=False)  
                
                # Updating logger with new GEM-TLM value
                log.fullinfo('************************************************'
                             , category='header')
                log.fullinfo('File = '+ad.filename, category='header')
                log.fullinfo('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
                             , category='header')
                log.fullinfo('PHU keywords updated/added:\n', category='header')
                log.fullinfo('GEM-TLM = '+ad.phuGetKeyValue('GEM-TLM'), 
                             category='header')
                log.fullinfo('MOSAIC = '+ad.phuGetKeyValue('MOSAIC'), 
                             category='header')
                
            log.status('*FINISHED* mosaicing the input images')
        except:
            log.critical('Problem processing one of '+rc.inputsAsStr())
            raise   
        yield rc

    def normalizeFlat(self, rc):
        """
        This primitive will combine the input flats and then normalize them 
        using the CL script giflat.
        
        Warning: giflat calculates its own DQ frames and thus replaces the 
        previously produced ones in calculateDQ. This may be fixed in the 
        future by replacing giflat with a Python equivilent with more 
        appropriate options for the recipe system.
        
        """
        # Loading and bringing the pyraf related modules into the name-space
        pyraf, gemini, yes, no = pyrafLoader(rc)
        
        try:
            
            log.status('*STARTING* to combine and normalize the input flats')
            # Writing input files to disk with prefixes onto their file names 
            # so they can be deleted later easily 
            clm = gemt.CLManager(rc)
            #clm.LogCurParams()
            #log.critical('prim_Gmos575: '+rc.inputsAsStr())
            # Creating a dictionary of the parameters set by the gemt.CLManager 
            # or the definition of the prim 
            clPrimParams = {
              'inflats'     :clm.inputList(),
              # Maybe allow the user to override this in the future
              'outflat'     :clm.combineOutname(), 
              # This returns a unique/temp log file for IRAF  
              'logfile'     :clm.logfile(),         
              # This is actually in the default dict but wanted to show it again
              'Stdout'      :gemt.IrafStdout(),   
              # This is actually in the default dict but wanted to show it again  
              'Stderr'      :gemt.IrafStdout(), 
              # This is actually in the default dict but wanted to show it again    
              'verbose'     :yes                    
                          }
            # Creating a dictionary of the parameters from the Parameter file 
            # adjustable by the user
            clSoftcodedParams = {
               'fl_bias'    :rc['fl_bias'],
               'fl_vardq'   :rc['fl_vardq'],
               'fl_over'    :rc['fl_over'],
               'fl_trim'    :rc['fl_trim']
                               }
            # Grabbing the default params dict and updating it 
            # with the two above dicts
            clParamsDict = CLDefaultParamsDict('giflat')
            clParamsDict.update(clPrimParams)
            clParamsDict.update(clSoftcodedParams)
            
            # Logging the values in the soft and prim parameter dictionaries
            log.fullinfo('\nParameters set by the CLManager or dictated by '+
                         'the definition of the primitive:\n', 
                         category='parameters')
            gemt.LogDictParams(clPrimParams)
            log.fullinfo('\nUser adjustable parameters in the parameters '+
                         'file:\n', category='parameters')
            gemt.LogDictParams(clSoftcodedParams)
            
            log.debug('Calling the giflat CL script for inputs list '+
                      clm.inputList())
            
            gemini.giflat(**clParamsDict)
            
            if gemini.giflat.status:
                log.critical('giflat failed for '+rc.inputsAsStr())
                raise GMOSException('giflat failed')
            else:
                log.status('Exited the giflat CL script successfully')
                
            # Renaming CL outputs and loading them back into memory, and 
            # cleaning up the intermediate tmp files written to disk
            clm.finishCL(combine=True) 
            
            # There is only one after above combination, so no need to perform a loop
            ad = rc.getOutputs(style='AD')[0] 
            
            # Adding GEM-TLM (automatic) and GIFLAT time stamps to the PHU
            ad.historyMark(key='GIFLAT', stomp=False)
            
            # Updating log with new GEM-TLM and GIFLAT time stamps
            log.fullinfo('****************************************************'
                         , category='header')
            log.fullinfo('File = '+ad.filename, category='header')
            log.fullinfo('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
                         , category='header')
            log.fullinfo('PHU keywords updated/added:\n', 'header')
            log.fullinfo('GEM-TLM = '+ad.phuGetKeyValue('GEM-TLM'), 
                         category='header')
            log.fullinfo('GIFLAT = '+ad.phuGetKeyValue('GIFLAT'), 
                         category='header')
            log.fullinfo('----------------------------------------------------'
                         , category='header')       
                
            log.status('*FINISHED* combining and normalizing the input flats')
        except:
            log.critical('Problem processing one of '+rc.inputsAsStr())
            raise 
            
        yield rc

    def overscanSubtract(self,rc):
        """
        This primitive uses the CL script gireduce to subtract the overscan 
        from the input images.
        
        """
        # Loading and bringing the pyraf related modules into the name-space
        pyraf, gemini, yes, no = pyrafLoader(rc)
        
        try:
            log.status('*STARTING* to subtract the overscan from the inputs')
            # Writing input files to disk with prefixes onto their file 
            # names so they can be deleted later easily 
            clm = gemt.CLManager(rc)
            #clm.LogCurParams()
            
            # Parameters set by the gemt.CLManager or the definition 
            # of the primitive 
            clPrimParams = {
              'inimages'    :clm.inputsAsStr(),
              'gp_outpref'  :clm.uniquePrefix(),
              # This returns a unique/temp log file for IRAF
              'logfile'     :clm.logfile(),      
              'fl_over'     :yes, 
              # This is actually in the default dict but wanted to show it again
              'Stdout'      :gemt.IrafStdout(), 
              # This is actually in the default dict but wanted to show it again
              'Stderr'      :gemt.IrafStdout(), 
              # This is actually in the default dict but wanted to show it again
              'verbose'     :yes                
                          }
            # Parameters from the Parameter file that are adjustable by the user
            clSoftcodedParams = {
               # pyrafBoolean converts the python booleans to pyraf ones
               'fl_trim'    :gemt.pyrafBoolean(rc['fl_trim']),
               'outpref'    :rc['postpend'],
               'fl_vardq'   :gemt.pyrafBoolean(rc['fl_vardq'])
                               }
            # Grabbing the default params dict and updating it with 
            # the two above dicts
            clParamsDict = CLDefaultParamsDict('gireduce')
            clParamsDict.update(clPrimParams)
            clParamsDict.update(clSoftcodedParams)
            
            # Logging the values in the soft and prim parameter dictionaries
            log.fullinfo('\nParameters set by the CLManager or dictated by '+
                         'the definition of the primitive:\n', 
                         category='parameters')
            gemt.LogDictParams(clPrimParams)
            log.fullinfo('\nUser adjustable parameters in the parameters '+
                         'file:\n', category='parameters')
            gemt.LogDictParams(clSoftcodedParams)
            
            # Taking care of the biasec->nbiascontam param
            if not rc['biassec'] == '':
                nbiascontam = clm.nbiascontam()
                clParamsDict.update({'nbiascontam':nbiascontam})
                log.fullinfo('nbiascontam parameter was updated to = '+
                             str(clParamsDict['nbiascontam']),'params')

            log.debug('Calling the gireduce CL script for inputs '+
                      clm.inputsAsStr())
            
            gemini.gmos.gireduce(**clParamsDict)

            if gemini.gmos.gireduce.status:
                log.critical('gireduce failed for '+rc.inputsAsStr()) 
                raise GMOSException('gireduce failed')
            else:
                log.status('Exited the gireduce CL script successfully')
         
            # Renaming CL outputs and loading them back into memory, and 
            # cleaning up the intermediate tmp files written to disk
            clm.finishCL()
            # Wrap up logging
            i=0
            for ad in rc.getOutputs(style='AD'):
                # Verifying gireduce was actually ran on the file
                if ad.phuGetKeyValue('GIREDUCE'): 
                    # If gireduce was ran, then log the changes to the files 
                    # it made
                    log.fullinfo('File '+clm.preCLNames()[i]+
                                 ' had its overscan subracted successfully')
                    log.fullinfo('New file name is: '+ad.filename)
                i = i+1
                # Updating GEM-TLM and OVERSUB time stamps in the PHU
                ad.historyMark(key='OVERSUB', stomp=False)  
                
                # Updating logger with new GEM-TLM time stamp value
                log.fullinfo('************************************************'
                             , category='header')
                log.fullinfo('File = '+ad.filename, category='header')
                log.fullinfo('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
                             , category='header')
                log.fullinfo('PHU keywords updated/added:\n', 'header')
                log.fullinfo('GEM-TLM = '+ad.phuGetKeyValue('GEM-TLM'), 
                              category='header')
                log.fullinfo('OVERSUB = '+ad.phuGetKeyValue('OVERSUB')+'\n', 
                              category='header')
            
            log.status('*FINISHED* subtracting the overscan from the '+
                       'input data')
        except:
            log.critical('Problem processing one of '+rc.inputsAsStr())
            raise 
        
        yield rc    

    def overscanTrim(self,rc):
        """
        This primitive uses AstroData to trim the overscan region 
        from the input images and update their headers.
        
        """
        try:
            log.status('*STARTING* to trim the overscan region from the input data')
            
            for ad in rc.getInputs(style='AD'):
                for sciExt in ad['SCI']:
                    # Converting data section string to an integer list
                    datasecStr=sciExt.data_section()
                    datasecList=gemt.secStrToIntList(datasecStr) 
                    dsl=datasecList
                    # Updating logger with the section being kept
                    log.stdinfo('\nfor '+ad.filename+' extension '+
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
                    log.fullinfo('********************************************'
                                 , category='header')
                    log.fullinfo('File = '+ad.filename, category='header')
                    log.fullinfo('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
                                 , category='header')
                    log.fullinfo('SCI extension number '+str(sciExt.extver())+
                                 ' keywords updated/added:\n', 'header')
                    log.fullinfo('NAXIS1= '+str(sciExt.header['NAXIS1']),
                                category='header')
                    log.fullinfo('NAXIS2= '+str(sciExt.header['NAXIS2']),
                                 category='header')
                    log.fullinfo('DATASEC= '+newDataSecStr, category='header')
                    log.fullinfo('TRIMSEC= '+datasecStr, category='header')
                    
                ad.phuSetKeyValue('TRIMMED','yes','Overscan section trimmed')    
                # Updating the GEM-TLM value and reporting the output to the RC    
                ad.historyMark(key='OVERTRIM', stomp=False)
                
                # Updating the file name with the postpend/outsuffix for this
                # primitive and then reporting the new file to the reduction 
                # context
                log.debug('calling gemt.fileNameUpdater on '+ad.filename)
                ad.filename = gemt.fileNameUpdater(ad.filename, 
                                                   postpend=rc['postpend'], 
                                                   strip=False)
                log.status('File name updated to '+ad.filename)
                rc.reportOutput(ad)
                
                # Updating logger with updated/added keywords to the PHU
                log.fullinfo('************************************************'
                             , category='header')
                log.fullinfo('file = '+ad.filename, category='header')
                log.fullinfo('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
                             , category='header')
                log.fullinfo('PHU keywords updated/added:\n', 'header')
                log.fullinfo('GEM-TLM = '+ad.phuGetKeyValue('GEM-TLM'), 
                             category='header') 
                log.fullinfo('OVERTRIM = '+ad.phuGetKeyValue('OVERTRIM')+'\n', 
                             category='header') 
                
            log.status('*FINISHED* trimming the overscan region from the input data')
        except:
            log.critical('Problem processing one of '+rc.inputsAsStr())
            raise 
        
        yield rc
         
    def standardizeInstrumentHeaders(self,rc):
        """
        This primitive is called by standardizeHeaders to makes the changes and 
        additions to the headers of the input files that are instrument 
        specific.
        
        """
        try:                                           
            for ad in rc.getInputs(style='AD'): 
                log.debug('Calling gmost.stdInstHdrs for '+ad.filename) 
                gmost.stdInstHdrs(ad) 
                log.status('Completed standardizing instrument headers for '+
                           ad.filename)
                    
        except:
            log.critical('Problem preparing one of '+rc.inputsAsStr())
            raise 
        
        yield rc 
   
    def storeProcessedBias(self,rc):
        """
        This should be a primitive that interacts with the calibration system 
        (MAYBE) but that isn't up and running yet. Thus, this will just strip 
        the extra postfixes to create the 'final' name for the 
        makeProcessedBias outputs and write them to disk in a storedcals folder.
        
        """
        try:  
            log.status('*STARTING* to store the processed bias by writing '+
                       'it to disk')
            for ad in rc.getInputs(style='AD'):
                # Updating the file name with the postpend/outsuffix for this
                # primitive and then reporting the new file to the reduction 
                # context
                log.debug('Calling gemt.fileNameUpdater on '+ad.filename)
                ad.filename = gemt.fileNameUpdater(ad.filename, 
                                                   postpend='_preparedbias', 
                                                   strip=True)
                log.status('File name updated to '+ad.filename)
                
                # Adding a GBIAS time stamp to the PHU
                ad.historyMark(key='GBIAS', 
                              comment='fake key to trick CL that GBIAS was ran')
                
                log.fullinfo('File written to = '+rc['storedbiases']+'/'+
                             ad.filename)
                ad.write(os.path.join(rc['storedbiases'],ad.filename), 
                         clobber=rc['clob'])
                
            log.status('*FINISHED* storing the processed bias on disk')
        except:
            log.critical('Problem storing one of '+rc.inputsAsStr())
            raise 
        yield rc
   
    def storeProcessedFlat(self,rc):
        """
        This should be a primitive that interacts with the calibration 
        system (MAYBE) but that isn't up and running yet. Thus, this will 
        just strip the extra postfixes to create the 'final' name for the 
        makeProcessedFlat outputs and write them to disk in a storedcals folder.
        
        """
        try:   
            log.status('*STARTING* to store the processed flat by writing it to disk')
            for ad in rc.getInputs(style='AD'):
                # Updating the file name with the postpend/outsuffix for this
                # primitive and then reporting the new file to the reduction 
                # context
                log.debug('Calling gemt.fileNameUpdater on '+ad.filename)
                ad.filename = gemt.fileNameUpdater(ad.filename, 
                                                   postpend='_preparedflat', 
                                                   strip=True)
                log.status('File name updated to '+ad.filename)
                
                log.fullinfo('File written to = '+rc['storedflats']+'/'
                             +ad.filename)
                ad.write(os.path.join(rc['storedflats'],ad.filename),
                         clobber=rc['clob'])
                
            log.status('*FINISHED* storing the processed flat on disk')
        except:
            log.critical('Problem storing one of '+rc.inputsAsStr())
            raise 
        yield rc
    
    def validateInstrumentData(self,rc):
        """
        This primitive is called by validateData to validate the instrument 
        specific data checks for all input files.
        
        """
        try:
            for ad in rc.getInputs(style='AD'):
                log.debug('Calling gmost.valInstData for '+ad.filename)
                gmost.valInstData(ad)
                log.status('Completed validating instrument data for '+
                           ad.filename)
                
        except:
            log.critical('Problem preparing one of '+rc.inputsAsStr())
            raise 
        
        yield rc

def CLDefaultParamsDict(CLscript):
    """
    A function to return a dictionary full of all the 
    default parameters for each CL script used so far in the Recipe System.
    
    """
    # Loading and bringing the pyraf related modules into the name-space
    pyraf, gemini, yes, no = pyrafLoader()
    
    # Ensuring that if a invalide CLscript was requested, that a critical
    # log message be made and exception raised.
    if (CLscript != 'gireduce') and (CLscript != 'giflat') and \
    (CLscript != 'gmosaic') :
        log.critical('The CLscript '+CLscript+' does not have a default'+
                     ' dictionary')
        raise GEMINIException('The CLscript '+CLscript+
                              ' does not have a default'+' dictionary')
    
    if CLscript == 'gireduce':
        defaultParams = {
            'inimages'   :'',                # Input GMOS images 
            'outpref'    :'DEFAULT',         # Prefix for output images
            'outimages'  :'',                # Output images
            'fl_over'    :no,                # Subtract overscan level
            'fl_trim'    :no,                # Trim off the overscan section
            'fl_bias'    :no,                # Subtract bias image
            'fl_dark'    :no,                # Subtract (scaled) dark image
            'fl_flat'    :no,                # Do flat field correction?
            'fl_vardq'   :no,                # Create variance and data quality frames
            'fl_addmdf'  :no,                # Add Mask Definition File? (LONGSLIT/MOS/IFU modes)
            'bias'       :'',                # Bias image name
            'dark'       :'',                # Dark image name
            'flat1'      :'',                # Flatfield image 1
            'flat2'      :'',                # Flatfield image 2
            'flat3'      :'',                # Flatfield image 3
            'flat4'      :'',                # Flatfield image 4
            'key_exptime':'EXPTIME',         # Header keyword of exposure time
            'key_biassec':'BIASSEC',         # Header keyword for bias section
            'key_datasec':'DATASEC',         # Header keyword for data section
            'rawpath'    :'',                # GPREPARE: Path for input raw images
            'gp_outpref' :'g',               # GPREPARE: Prefix for output images
            'sci_ext'    :'SCI',             # Name of science extension
            'var_ext'    :'VAR',             # Name of variance extension
            'dq_ext'     :'DQ',              # Name of data quality extension
            'key_mdf'    :'MASKNAME',        # Header keyword for the Mask Definition File
            'mdffile'    :'',                # MDF file to use if keyword not found
            'mdfdir'     :'',                # MDF database directory
            'bpm'        :'',                # Bad pixel mask
            #'giandb'     :'default',        # Database with gain data
            'sat'        :65000,             # Saturation level in raw images [ADU]
            'key_nodcount':'NODCOUNT',       # Header keyword with number of nod cycles
            'key_nodpix' :'NODPIX',          # Header keyword with shuffle distance
            'key_filter' :'FILTER2',         # Header keyword of filter
            'key_ron'    :'RDNOISE',         # Header keyword for readout noise
            'key_gain'   :'GAIN',            # Header keyword for gain (e-/ADU)
            'ron'        :3.5,               # Readout noise in electrons
            'gain'       :2.2,               # Gain in e-/ADU
            'fl_mult'    :no, #$$$$$$$$$     # Multiply by gains to get output in electrons
            'fl_inter'   :no,                # Interactive overscan fitting?
            'median'     :no,                # Use median instead of average in column bias?
            'function'   :'chebyshev',       # Overscan fitting function
            'nbiascontam':4, #$$$$$$$        # Number of columns removed from overscan region
            'biasrows'   :'default',         # Rows to use for overscan region
            'order'      :1,                 # Order of overscan fitting function
            'low_reject' :3.0,               # Low sigma rejection factor in overscan fit
            'high_reject':3.0,               # High sigma rejection factor in overscan fit
            'niterate'   :2,                 # Number of rejection iterations in overscan fit
            'logfile'    :'',                # Logfile
            'verbose'    :yes,               # Verbose?
            'status'     :0,                 # Exit status (0=good)
            'Stdout'     :gemt.IrafStdout(),
            'Stderr'     :gemt.IrafStdout()
                           }
    if CLscript == 'giflat':
        defaultParams = { 
            'inflats'    :'',            # Input flat field images
            'outflat'    :'',            # Output flat field image
            'normsec'    :'default',     # Image section to get the normalization.
            'fl_scale'   :yes,           # Scale the flat images before combining?
            'sctype'     :'mean',        # Type of statistics to compute for scaling
            'statsec'    :'default',     # Image section for relative intensity scaling
            'key_gain'   :'GAIN',        # Header keyword for gain (e-/ADU)
            'fl_stamp'   :no,            # Input is stamp image
            'sci_ext'    :'SCI',         # Name of science extension
            'var_ext'    :'VAR',         # Name of variance extension
            'dq_ext'     :'DQ',          # Name of data quality extension
            'fl_vardq'   :no,            # Create variance and data quality frames?
            'sat'        :65000,         # Saturation level in raw images (ADU)
            'verbose'    :yes,           # Verbose output?
            'logfile'    :'',            # Name of logfile
            'status'     :0,             # Exit status (0=good)
            'combine'    :'average',     # Type of combine operation
            'reject'     :'avsigclip',   # Type of rejection in flat average
            'lthreshold' :'INDEF',       # Lower threshold when combining
            'hthreshold' :'INDEF',       # Upper threshold when combining
            'nlow'       :0,             # minmax: Number of low pixels to reject
            'nhigh'      :1,             # minmax: Number of high pixels to reject
            'nkeep'      :1,             # avsigclip: Minimum to keep (pos) or maximum to reject (neg)
            'mclip'      :yes,           # avsigclip: Use median in clipping algorithm?
            'lsigma'     :3.0,           # avsigclip: Lower sigma clipping factor
            'hsigma'     :3.0,           # avsigclip: Upper sigma clipping factor
            'sigscale'   :0.1,           # avsigclip: Tolerance for clipping scaling corrections
            'grow'       :0.0,           # minmax or avsigclip: Radius (pixels) for neighbor rejection
            'gp_outpref' :'g',           # Gprepare prefix for output images
            'rawpath'    :'',            # GPREPARE: Path for input raw images
            'key_ron'    :'RDNOISE',     # Header keyword for readout noise
            'key_datasec':'DATASEC',     # Header keyword for data section
            #'giandb'     :'default',    # Database with gain data
            'bpm'        :'',            # Bad pixel mask
            'gi_outpref' :'r',           # Gireduce prefix for output images
            'bias'       :'',            # Bias calibration image
            'fl_over'    :no,            # Subtract overscan level?
            'fl_trim'    :no,            # Trim images?
            'fl_bias'    :no,            # Bias-subtract images?
            'fl_inter'   :no,            # Interactive overscan fitting?
            'nbiascontam':4, #$$$$$$$    # Number of columns removed from overscan region
            'biasrows'   :'default',     # Rows to use for overscan region
            'key_biassec':'BIASSEC',     # Header keyword for overscan image section
            'median'     :no,            # Use median instead of average in column bias?
            'function'   :'chebyshev',   # Overscan fitting function.
            'order'      :1,             # Order of overscan fitting function.
            'low_reject' :3.0,           # Low sigma rejection factor.
            'high_reject':3.0,           # High sigma rejection factor.
            'niterate'   :2,             # Number of rejection iterations.
            'Stdout'      :gemt.IrafStdout(),
            'Stderr'      :gemt.IrafStdout()
                       }      
    if CLscript == 'gmosaic':
        defaultParams = { 
            'inimages'   :'',                     # Input GMOS images 
            'outimages'  :'',                     # Output images
            'outpref'    :'DEFAULT',              # Prefix for output images
            'fl_paste'   :no,                     # Paste images instead of mosaic
            'fl_vardq'   :no,                     # Propagate the variance and data quality planes
            'fl_fixpix'  :no,                     # Interpolate across chip gaps
            'fl_clean'   :yes ,                   # Clean imaging data outside imaging field
            'geointer'   :'linear',               # Interpolant to use with geotran
            'gap'        :'default',              # Gap between the CCDs in unbinned pixels
            'bpmfile'    :'gmos$data/chipgaps.dat',   # Info on location of chip gaps ## HUH??? Why is variable called 'bpmfile' if it for chip gaps??
            'statsec'    :'default',              # Statistics section for cleaning
            'obsmode'    :'IMAGE',                # Value of key_obsmode for imaging data
            'sci_ext'    :'SCI',                  # Science extension(s) to mosaic, use '' for raw data
            'var_ext'    :'VAR',                  # Variance extension(s) to mosaic
            'dq_ext'     :'DQ',                   # Data quality extension(s) to mosaic
            'mdf_ext'    :'MDF',                  # Mask definition file extension name
            'key_detsec' :'DETSEC',               # Header keyword for detector section
            'key_datsec' :'DATASEC',              # Header keyword for data section
            'key_ccdsum' :'CCDSUM',               # Header keyword for CCD binning
            'key_obsmode':'OBSMODE',              # Header keyword for observing mode
            'logfile'    :'',                     # Logfile
            'fl_real'    :no,                     # Convert file to real before transforming
            'verbose'    :yes,                    # Verbose
            'status'     :0,                      # Exit status (0=good)
            'Stdout'     :gemt.IrafStdout(),
            'Stderr'     :gemt.IrafStdout()
                       }
    return defaultParams    
    #$$$$$$$$$$$$$$$$$$$$$$$ END OF KYLES NEW STUFF $$$$$$$$$$$$$$$$$$$$$$$$$$
