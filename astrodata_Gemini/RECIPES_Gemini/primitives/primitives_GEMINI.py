import os, sys, re
from sets import Set

import time
from astrodata.ReductionObjects import PrimitiveSet
from astrodata.adutils import gemLog
from astrodata import IDFactory
from gempy.instruments import geminiTools  as gemt
import numpy as np
import pyfits as pf
from datetime import datetime
import shutil

log = gemLog.getGeminiLog()

def pyrafLoader(rc=None):
    """
    This function is to load the modules needed by primitives that use pyraf. 
    It will also ensure there are no additional prints to the console when 
    loading the Gemini pyraf package.
    The loaded modules are returned in the order of:
    (pyraf, gemini, iraf.yes, iraf.no)
    to be added to the name-space of the primitive this function is called from.
    eg. (pyraf, gemini, yes, no)=pyrafLoader(rc)
    
    """
    import pyraf
    from pyraf import iraf
    from iraf import gemini
    from iraf import gmos
    import StringIO
    
    # Changing the standard output so the excess prints while loading IRAF
    # packages does not get displayed
    SAVEOUT = sys.stdout
    capture = StringIO.StringIO()
    sys.stdout = capture
    
    # Setting the IRAF versions of True and False
    yes = iraf.yes
    no = iraf.no
    
    # This will load the gemini pyraf package    
    gemini() 
    gmos()    
    
    return (pyraf, gemini, iraf.yes, iraf.no)

class GEMINIException:
    """ This is the general exception the classes and functions in the
    Structures.py module raise.
    
    """
    def __init__(self, message='Exception Raised in Recipe System'):
        """This constructor takes a message to print to the user."""
        self.message = message
    def __str__(self):
        """This str conversion member returns the message given by the 
        user (or the default message)
        when the exception is not caught."""
        return self.message

class GEMINIPrimitives(PrimitiveSet):
    """ 
    This is the class of all primitives for the GEMINI astrotype of 
    the hierarchy tree.  It inherits all the primitives to the level above
    , 'PrimitiveSet'.
    
    """
    astrotype = 'GEMINI'
    
    def init(self, rc):
        return 
    init.pt_hide = True
    
    def addDQ(self,rc):
        """
        This primitive will create a numpy array for the data quality 
        of each SCI frame of the input data. This will then have a 
        header created and append to the input using AstroData as a DQ 
        frame. The value of a pixel will be the sum of the following: 
        (0=good, 1=bad pixel (found in bad pixel mask), 
        2=value is non linear, 4=pixel is saturated)
        
        """
    def clearCalCache(self, rc):
        # print 'pG61:', rc.calindfile
        rc.persistCalIndex(rc.calindfile, newindex={})
        scals = rc['storedcals']
        if scals:
            if os.path.exists(scals):
                shutil.rmtree(scals)
            cachedict = rc['cachedict']
            for cachename in cachedict:
                cachedir = cachedict[cachename]
                if not os.path.exists(cachedir):                        
                    os.mkdir(cachedir)                
        yield rc
        
    def display(self, rc):
        try:
            rc.rqDisplay(displayID=rc["displayID"])           
        except:
            log.critical('Problem displaying output')
            raise 
        yield rc
 
    def setContext(self, rc):
        rc.update(rc.localparms)
        yield rc

    def showParameters(self, rc):
        rcparams = rc.paramNames()
        if (rc['show']):
            toshows = rc['show'].split(':')
            for toshow in toshows:
                if toshow in rcparams:
                    log.fullinfo(toshow+' = '+repr(rc[toshow]), \
                                 category='parameters')
                else:
                    log.fullinfo(toshow+' is not set', category='parameters')
        else:
            for param in rcparams:
                log.fullinfo(param+' = '+repr(rc[param]), category='parameters')
        
        # print 'all',repr(rc.parmDictByTag('showParams', 'all'))
        # print 'iraf',repr(rc.parmDictByTag('showParams', 'iraf'))
        # print 'test',repr(rc.parmDictByTag('showParams', 'test'))
        # print 'sdf',repr(rc.parmDictByTag('showParams', 'sdf'))

        # print repr(dir(rc.ro.primDict[rc.ro.curPrimType][0]))
        yield rc  
            
    def sleep(self, rc):
        if rc['duration']:
            dur = float(rc['duration'])
        else:
            dur = 5.
        log.status('Sleeping for %f seconds' % dur)
        time.sleep(dur)
        yield rc
                      
    def showInputs(self, rc):
        log.fullinfo('Inputs:',category='inputs')
        for inf in rc.inputs:
            log.fullinfo('  '+inf.filename, category='inputs')  
        yield rc  
    showFiles = showInputs
    
    def showCals(self, rc):
        if str(rc['showcals']).lower() == 'all':
            num = 0
            # print 'pG256: showcals=all', repr (rc.calibrations)
            for calkey in rc.calibrations:
                num += 1
                log.fullinfo(rc.calibrations[calkey], category='calibrations')
            if (num == 0):
                log.warning('There are no calibrations in the cache.')
        else:
            for adr in rc.inputs:
                sid = IDFactory.generateAstroDataID(adr.ad)
                num = 0
                for calkey in rc.calibrations:
                    if sid in calkey :
                        num += 1
                        log.fullinfo(rc.calibrations[calkey], \
                                     category='calibrations')
            if (num == 0):
                log.warning('There are no calibrations in the cache.')
        yield rc
    ptusage_showCals='Used to show calibrations currently in cache for inputs.'

    def showStackable(self, rc):
        sidset = set()
        for inp in rc.inputs:
            sidset.add(IDFactory.generateStackableID(inp.ad))
        for sid in sidset:
            stacklist = rc.getStack(sid) #.filelist
            log.status('Stack for stack id=%s' % sid)
            for f in stacklist:
                log.status('   '+os.path.basename(f))
        yield rc
                 
    def time(self, rc):
        cur = datetime.now()
        
        elap = ''
        if rc['lastTime'] and not rc['start']:
            td = cur - rc['lastTime']
            elap = ' (%s)' %str(td)
        log.fullinfo('Time:'+' '+str(datetime.now())+' '+elap)
        
        rc.update({'lastTime':cur})
        yield rc

    def getStackable(self, rc):
        try:
            # @@REFERENCE IMAGE @@NOTE: to pick which stackable list to get
            stackid = IDFactory.generateStackableID(rc.inputs[0].ad)
            log.fullinfo('getting stack '+stackid,'stack')
            rc.rqStackGet()
            yield rc
            stack = rc.getStack(stackid) #.filelist
            #print 'prim_G366: ',repr(stack)
            rc.reportOutput(stack)
        except:
            log.critical('Problem getting stack '+stackid, 'stack')
            raise 
        yield rc      
 
    def setStackable(self, rc):
        """
        This primitive will update the lists of files to be stacked
        that have the same observationID with the current inputs.
        This file is cached between calls to reduce, thus allowing
        for one-file-at-a-time processing.
        """
        try:
            stackid = IDFactory.generateStackableID(rc.inputs[0].ad)
            log.fullinfo('updating stack '+stackid+' with '+rc.inputsAsStr(), \
                         category='stack')
            rc.rqStackUpdate()
            # Writing the files in the stack to disk if not all ready there
            for ad in rc.getInputs(style='AD'):
                if not os.path.exists(ad.filename):
                    log.fullinfo('temporarily writing '+ad.filename+\
                                 ' to disk', category='stack')
                    ad.write(ad.filename)
        except:
            log.critical('Problem preparing stack for files '+rc.inputsAsStr(),\
                         category='stack')
            raise
        yield rc
    
    def validateData(self,rc):
        """
        This primitive will ensure the data is not corrupted or in an odd 
        format that will affect later steps in the reduction process.  
        It will call a function to take care of the general Gemini issues 
        and then one for the instrument specific ones. If there are issues 
        with the data, the flag 'repair' can be used to turn on the feature to 
        repair it or not (eg. validateData(repair=True))
        (this feature is not coded yet).
        """
        
        try:
            if rc['repair'] is True:
               # This should repair the file if it is broken, but this function
               # isn't coded yet and would require some sort of flag set while 
               # checking the data to tell this to perform the corrections
               log.critical('Sorry, but the repair feature of validateData' +\
                            ' is not available yet')
               pass
            
            # Calling addBPM primitive to add the appropriate Bad Pixel Mask
            # to the inputs which will then be updated below to create data 
            # quality frames from these new BPM extensions in the inputs.
            log.debug('Calling addBPM primitive for '+rc.inputsAsStr())
            rc.run('addBPM')
            log.status('Returned from the addBPM primitive successfully')
              
            # Loop through the inputs to perform the non-linear and saturated
            # pixel searches of the SCI frames to update the BPM frames into
            # full DQ frames.       
            for ad in rc.getInputs(style='AD'):
                # Check if DQ extensions all ready exist for this file
                if not ad['DQ']:
                    for sciExt in ad['SCI']: 
                        # Retrieving BPM extension 
                        bpmAD = ad[('BPM',sciExt.extver())]
                        
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
    
                        if (linear is not None) and \
                        (rc['fl_nonlinear'] is True): 
                            log.debug('Performing an np.where to find '+
                                      'non-linear pixels for extension '+
                                      str(sciExt.extver())+' of '+ad.filename)
                            nonLinArray = np.where(sciExt.data>linear,2,0)
                            log.status('Done calculating array of non-linear'+
                                       ' pixels')
                        if (saturated is not None) and \
                        (rc['fl_saturated'] is True):
                            log.debug('Performing an np.where to find '+
                                      'saturated pixels for extension '+
                                      str(sciExt.extver())+' of '+ad.filename)
                            saturatedArray = np.where(sciExt.data>saturated,4,0)
                            log.status('Done calculating array of saturated'+
                                       ' pixels') 
                        
                        # Creating one DQ array from the three
                        dqArray=np.add(BPMArray, nonLinArray, 
                                       saturatedArray) 
                        # Updating data array for the BPM array to be the 
                        # newly calculated DQ array
                        ad[('BPM',sciExt.extver())].data = dqArray
                        
                        # Renaming the extension to DQ from BPM
                        dqheader.update('EXTNAME', 'DQ', 'Extension Name')
                        
                        
                        # Using renameExt to correctly set the EXTVer and 
                        # EXTNAME values in the header   
                        bpmAD.renameExt('DQ', ver=sciExt.extver(), force=True)

                        # Logging that the name of the BPM extension was changed
                        log.fullinfo('BPM Extension '+str(sciExt.extver())+
                                     ' of'+ad.filename+' had its EXTVER '+
                                     'changed to '+
                                     ad[('DQ',sciExt.extver())].header['EXTNAME'])
                        
                # If DQ frames exist, send a critical message to the logger
                else:
                    log.critical('DQ frames all ready exist for '+ad.filename+
                                 ', so addDQ will not calculate new ones')
                    
                # Adding GEM-TLM (automatic) and ADDVARDQ time stamp to the PHU
                ad.historyMark(key='ADDDQ', stomp=False) 
                # updating logger with updated/added time stamps
                log.fullinfo('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
                             ,'header')
                log.fullinfo('PHU keywords updated/added:\n', 'header')
                log.fullinfo('GEM-TLM = '+ad.phuGetKeyValue('GEM-TLM'),'header')
                log.fullinfo('ADDDQ = '+ad.phuGetKeyValue('ADDDQ'),'header')
                log.fullinfo('------------------------------------------------'
                             , 'header')
                
                # Updating the file name with the postpend/outsuffix for this
                # primitive and then reporting the new file to the reduction 
                # context
                log.debug('Calling gemt.fileNameUpdater on '+ad.filename)
                ad.filename = gemt.fileNameUpdater(ad.filename, 
                                                   postpend=rc['outsuffix'], 
                                                   strip=False)
                log.status('File name updated to '+ad.filename)
                rc.reportOutput(ad)        
                
            log.status('*FINISHED* adding the DQ frame(s) to the input data')
        except:
            log.critical('Problem adding the DQ to one of '+rc.inputsAsStr())
            raise 
        yield rc
    
    def addVAR(self,rc):
        """
        This primitive uses numpy to calculate the variance of each SCI frame
        in the input files and appends it as a VAR frame using AstroData.
        
        The calculation will follow the formula:
        variance = (read noise/gain)2 + max(data,0.0)/gain
        
        """
        try:
            log.fullinfo('*STARTING* to add the VAR frame(s) to the input data')
            
            for ad in rc.getInputs(style='AD'):
                #print ad.info()
                # Check if there VAR frames all ready exist
                if ad['VAR']:
                    # If VAR frames don't exist, loop through the SCI extensions 
                    # and calculate a corresponding VAR frame for it, then 
                    # append it
                    for sciExt in ad['SCI']:
                        # var = (read noise/gain)2 + max(data,0.0)/gain
                        
                        # Retrieving necessary values (read noise, gain)
                        readNoise = sciExt.read_noise()
                        gain = sciExt.gain()
                        # Creating (read noise/gain) constant
                        rnOverG = readNoise/gain
                        # Convert negative numbers (if they exist) to zeros
                        maxArray = np.where(sciExt.data>0.0,0,sciExt.data)
                        # Creating max(data,0.0)/gain array
                        maxOverGain = np.divide(maxArray,gain)
                        # Putting it all together
                        varArray = np.add(maxOverGain,rnOverG*rnOverG)
                         
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
                                     +ad.filename)
                        ad.append(varAD)
                        log.status('appending VAR complete for '+ad.filename)
                        
                # If VAR frames all ready existed, 
                # make a critical message in the logger
                else:
                    log.critical('VAR frames all ready exist for '+ad.filename+
                                 ', so addVAR will not calculate new ones')
                
                # Adding GEM-TLM(automatic) and ADDVAR time stamps to the PHU     
                ad.historyMark(key='ADDVAR', stomp=False)    
                
                log.fullinfo('************************************************'
                             ,'header')
                log.fullinfo('file = '+ad.filename, 'header')
                log.fullinfo('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
                             ,'header')
                log.fullinfo('PHU keywords updated/added:\n', 'header')
                log.fullinfo('GEM-TLM = '+ad.phuGetKeyValue('GEM-TLM'), \
                             'header')
                log.fullinfo('ADDVAR = '+ad.phuGetKeyValue('ADDVAR'), 'header')
                log.fullinfo('------------------------------------------------'
                             ,'header')
                
                # Updating the file name with the postpend/outsuffix for this
                # primitive and then reporting the new file to the reduction 
                # context
                log.debug('Calling gemt.fileNameUpdater on '+ad.filename)
                ad.filename = gemt.fileNameUpdater(ad.filename, 
                                                   postpend=rc['outsuffix'], 
                                                   strip=False)
                log.status('File name updated to '+ad.filename)
                rc.reportOutput(ad)        
                
            log.status('*FINISHED* adding the VAR frame(s) to the input data')
        except:
            log.critical('Problem adding the VAR to one of '+rc.inputsAsStr())
            raise 
        yield rc 
    
    def ADUtoElectrons(self,rc):
        """
        This primitive will convert the inputs from having pixel 
        units of ADU to electrons.
        
        """
        try:
            log.status('*STARTING* to convert the pixel values from '+
                       'ADU to electrons')
            for ad in rc.getInputs(style='AD'):
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
                    ext.SetKeyValue('GAINORIG', gainorig, 
                                       'Gain prior to unit conversion (e-/ADU)')
                    ext.SetKeyValue('GAIN', 1.0, 'Physical units is electrons') 
                    ext.SetKeyValue('BUNIT','electrons' , 'Physical units')
                    # Logging the changes to the header keys
                    log.fullinfo('SCI extension number '+
                                 str(ext.extver())+
                                 ' keywords updated/added:\n', 'header')
                    log.fullinfo('GAINORIG = '+str(gainorig),'header' )
                    log.fullinfo('GAIN = '+str(1.0),'header' )
                    log.fullinfo('BUNIT = '+'electrons','header' )
                    log.fullinfo('--------------------------------------------'
                                 ,'header')
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
                    ext.SetKeyValue('BUNIT','electrons squared' , 
                                       'Physical units')
                    # Logging the changes to the VAR extensions header keys
                    log.fullinfo('VAR extension number '+
                                 str(ext.extver())+\
                                 ' keywords updated/added:\n', 'header')
                    log.fullinfo('BUNIT = '+'electrons squared','header' )
                    log.fullinfo('--------------------------------------------'
                                 ,'header')
                
                # Adding GEM-TLM (automatic) and ADU2ELEC time stamps to PHU
                adOut.historyMark('ADU2ELEC', stomp=False)
                
                # Updating logger with time stamps
                log.fullinfo('************************************************'
                             ,'header')
                log.fullinfo('ile = '+adOut.filename, 'header')
                log.fullinfo('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
                             ,'header')
                log.fullinfo('PHU keywords updated/added:\n', 'header')
                log.fullinfo('GEM-TLM = '+ad.phuGetKeyValue('GEM-TLM'), 
                             'header')
                log.fullinfo('ADU2ELEC = '+ad.phuGetKeyValue('ADU2ELEC'), 
                             'header')
                log.fullinfo('------------------------------------------------'
                             , 'header')
                
                # Updating the file name with the postpend/outsuffix for this
                # primitive and then reporting the new file to the reduction 
                # context.
                log.debug('Calling gemt.fileNameUpdater on '+adOut.filename)
                adOut.filename=gemt.fileNameUpdater(adOut.filename, 
                                                    postpend=rc['outpref'], 
                                                    strip=False)
                log.status('File name updated to '+ad.filename)
                rc.reportOutput(adOut)   
                
            log.status('*FINISHED* converting the pixel units to electrons')
        except:
            log.critical('Problem converting the pixel units of one of '+
                         rc.inputsAsStr())
            raise
        yield rc
            
    def combine(self,rc):
        """
        This primitive will average and combine the SCI extensions of the 
        inputs. It takes all the inputs and creates a list of them and 
        then combines each of their SCI extensions together to create 
        average combination file. New VAR frames are made from these 
        combined SCI frames and the DQ frames are propagated through 
        to the final file.
        
        """
        # Loading and bringing the pyraf related modules into the name-space
        pyraf, gemini, yes, no = pyrafLoader(rc)
        
        try:
            if len(rc.getInputs())>1:
                log.status('*STARTING* combine the images of the input data')
                
                # Preparing input files, lists, parameters... for input to 
                # the CL script
                clm=gemt.CLManager(rc)
                clm.LogCurParams()
                
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
                    'Stdout'      :gemt.IrafStdout(), 
                    # This is actually in the default dict but wanted to 
                    # show it again    
                    'Stderr'      :gemt.IrafStdout(),
                    # This is actually in the default dict but wanted to 
                    # show it again     
                    'verbose'     :yes                    
                              }
                # Creating a dictionary of the parameters from the Parameter 
                # file adjustable by the user
                clSoftcodedParams = {
                    'fl_vardq'      :gemt.pyrafBoolean(rc['fl_vardq']),
                    # pyrafBoolean converts the python booleans to pyraf ones
                    'fl_dqprop'     :gemt.pyrafBoolean(rc['fl_dqprop']),
                    'combine'       :rc['method'],
                    'reject'        :'none'
                                    }
                # Grabbing the default parameters dictionary and updating 
                # it with the two above dictionaries
                clParamsDict = CLDefaultParamsDict('gemcombine')
                clParamsDict.update(clPrimParams)
                clParamsDict.update(clSoftcodedParams)
                 
                log.debug('Calling the gemcombine CL script for input list '+
                          clm.inputList())
                
                gemini.gemcombine(**clParamsDict)
                
                if gemini.gemcombine.status:
                    log.critical('gemcombine failed for inputs '+
                                 rc.inputsAsStr())
                    raise GEMINIException('gemcombine failed')
                else:
                    log.status('Exited the gemcombine CL script successfully')
                    
                # Renaming CL outputs and loading them back into memory 
                # and cleaning up the intermediate temp files written to disk
                clm.finishCL(combine=True) 
                #clm.rmStackFiles() #$$$$$$$$$ DON'T do this if 
                #^ Intermediate outputs are wanted!!!!
                
                # There is only one at this point so no need to perform a loop
                ad = rc.getOutputs(style='AD')[0] 
                
                # Adding a GEM-TLM (automatic) and COMBINE time stamps 
                # to the PHU
                ad.historyMark(key='COMBINE',stomp=False)
                # Updating logger with updated/added time stamps
                log.fullinfo('************************************************'
                             ,'header')
                log.fullinfo('file = '+ad.filename, 'header')
                log.fullinfo('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
                             , 'header')
                log.fullinfo('PHU keywords updated/added:\n', 'header')
                log.fullinfo('GEM-TLM = '+ad.phuGetKeyValue('GEM-TLM'), 
                             'header')
                log.fullinfo('COMBINE = '+ad.phuGetKeyValue('COMBINE'), 
                             'header')
                log.fullinfo('------------------------------------------------'
                             , 'header')    
                
                log.status('*FINISHED* combining the images of the input data')
        except:
            log.critical('There was a problem combining '+rc.inputsAsStr())
            raise 
        yield rc

    def crashReduce(self, rc):
        raise 'Crashing'
        yield rc

    def writeOutputs(self,rc):
        """
        A primitive that may be called by a recipe at any stage to
        write the outputs to disk.
        If postpend is set during the call to writeOutputs, any previous 
        postpends will be striped and replaced by the one provided.
        examples: 
        writeOutputs(postpend= '_string'), writeOutputs(prepend= '_string') 
        or if you have a full file name in mind for a SINGLE file being 
        ran through Reduce you may use writeOutputs(outfilename='name.fits').
        
        """
        try:
            log.status('*STARTING* to write the outputs')
            
            # Logging current values of postpend and prepend
            log.status('postpend = '+str(rc['postpend']))
            log.status('prepend = '+str(rc['prepend']))
            
            for ad in rc.getInputs(style='AD'):
                # If the value of 'postpend' was set, then set the file name 
                # to be written to disk to be postpended by it
                if rc['postpend']:
                    log.debug('calling gemt.fileNameUpdater on '+ad.filename)
                    ad.filename = gemt.fileNameUpdater(ad.filename, 
                                        postpend=rc['postpend'], strip=True)
                    log.status('File name updated to '+ad.filename)
                    outfilename = os.path.basename(ad.filename)
                    
                # If the value of 'prepend' was set, then set the file name 
                # to be written to disk to be prepended by it
                elif rc['prepend']:
                    infilename = os.path.basename(ad.filename)
                    outfilename = rc['prepend']+infilename
                    
                # If the 'outfilename' was set, set the file name of the file 
                # file to be written to this
                elif rc['outfilename']:
                    # Check that there is not more than one file to be written
                    # to this file name, if so throw exception
                    if len(rc.getInputs(style='AD'))>1:
                        log.critical('More than one file was requested to be'+
                                     'written to the same name '+
                                     rc['outfilename'])
                        raise GEMINIException('More than one file was '+
                                     'requested to be written to the same'+
                                     'name'+rc['outfilename'])
                    else:
                        outfilename = rc['outfilename']   
                # If no changes to file names are requested then write inputs
                # to their current file names
                else:
                    outfilename = os.path.basename(ad.filename) 
                    log.status('not changing the file name to be written'+
                    ' from its current name') 
                    
                # Finally, write the file to the name that was decided 
                # upon above
                log.status('writing to file = '+outfilename)      
                ad.write(filename=outfilename, clobber=rc['clobber'])     
                #^ AstroData checks if the output exists and raises an exception
                #rc.reportOutput(ad)
            
            log.status('*FINISHED* writing the outputs')   
        except:
            log.critical('There was a problem writing one of '+rc.inputsAsStr())
            raise 
        yield rc   
         
def CLDefaultParamsDict(CLscript):
    """
    A function to return a dictionary full of all the default parameters 
    for each CL script used so far in the Recipe System.
    
    """
    # loading and bringing the pyraf related modules into the name-space
    pyraf, gemini, yes, no = pyrafLoader()
    
    # Ensuring that if a invalide CLscript was requested, that a critical
    # log message be made and exception raised.
    if CLscript != 'gemcombine':
        log.critical('The CLscript '+CLscript+' does not have a default'+
                     ' dictionary')
        raise GEMINIException('The CLscript '+CLscript+
                              ' does not have a default'+' dictionary')
        
    if CLscript == 'gemcombine':
        defaultParams = {
            'input'      :'',            # Input MEF images
            'output'     :'',            # Output MEF image
            'title'      :'DEFAULT',     # Title for output SCI plane
            'combine'    :'average',     # Combination operation
            'reject'     :'avsigclip',   # Rejection algorithm
            'offsets'    :'none',        # Input image offsets
            'masktype'   :'none',        # Mask type
            'maskvalue'  :0.0,           # Mask value
            'scale'      :'none',        # Image scaling
            'zero'       :'none',        # Image zeropoint offset
            'weight'     :'none',        # Image weights
            'statsec'    :'[*,*]',       # Statistics section
            'expname'    :'EXPTIME',     # Exposure time header keyword
            'lthreshold' :'INDEF',       # Lower threshold
            'hthreshold' :'INDEF',       # Upper threshold
            'nlow'       :1,             # minmax: Number of low pixels to reject
            'nhigh'      :1,             # minmax: Number of high pixels to reject
            'nkeep'      :1,             # Minimum to keep or maximum to reject
            'mclip'      :yes,           # Use median in sigma clipping algorithms?
            'lsigma'     :3.0,           # Lower sigma clipping factor
            'hsigma'     :3.0,           # Upper sigma clipping factor
            'key_ron'    :'RDNOISE',     # Keyword for readout noise in e-
            'key_gain'   :'GAIN',        # Keyword for gain in electrons/ADU
            'ron'        :0.0,           # Readout noise rms in electrons
            'gain'       :1.0,           # Gain in e-/ADU
            'snoise'     :'0.0',         # ccdclip: Sensitivity noise (electrons
            'sigscale'   :0.1,           # Tolerance for sigma clipping scaling correction                                
            'pclip'      :-0.5,          # pclip: Percentile clipping parameter
            'grow'       :0.0,           # Radius (pixels) for neighbor rejection
            'bpmfile'    :'',            # Name of bad pixel mask file or image.
            'nrejfile'   :'',            # Name of rejected pixel count image.
            'sci_ext'    :'SCI',         # Name(s) or number(s) of science extension
            'var_ext'    :'VAR',         # Name(s) or number(s) of variance extension
            'dq_ext'     :'DQ',          # Name(s) or number(s) of data quality extension
            'fl_vardq'   :no,            # Make variance and data quality planes?
            'logfile'    :'',            # Log file
            'fl_dqprop'  :no,            # Propagate all DQ values?
            'verbose'    :yes,           # Verbose output?
            'status'     :0,             # Exit status (0=good)
            'Stdout'     :gemt.IrafStdout(),
            'Stderr'     :gemt.IrafStdout()
                       }
        return defaultParams                                  

    
    
