import os, sys, re
from sets import Set

import time
from astrodata.ReductionObjects import PrimitiveSet
from astrodata.adutils import gemLog
from astrodata import IDFactory
from gempy.instruments import geminiTools as gemt
from gempy.science import geminiScience
from datetime import datetime
import shutil
from primitives_GENERAL import GENERALPrimitives
from astrodata.adutils.gemutil import pyrafLoader

#log = gemLog.getGeminiLog()

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

class GEMINIPrimitives(GENERALPrimitives):
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
        
        
        :param suffix: Value to be post pended onto each input name(s) to 
                         create the output name(s).
        :type suffix: string
        
        :param fl_nonlinear: Flag to turn checking for nonlinear pixels on/off
        :type fl_nonLinear: Python boolean (True/False), default is True
    
        :param fl_saturated: Flag to turn checking for saturated pixels on/off
        :type fl_saturated: Python boolean (True/False), default is True
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: int. 
                        This value can be set for each primitive individually 
                        in a recipe only (ie. not in the parameter file). 
                        If no value is specified during the recipe, the value 
                        set during the call to reduce or its default (2) will 
                        be used.
        """
        log = gemLog.getGeminiLog(logLevel=int(rc['logLevel']))
        try: 
            log.status('*STARTING* to add the DQ frame(s) to the input data')
            
            # Calling addBPM primitive to add the appropriate Bad Pixel Mask
            # to the inputs which will then be updated below to create data 
            # quality frames from these new BPM extensions in the inputs.
            log.debug('Calling addBPM primitive for '+rc.inputsAsStr())
            rc.run('addBPM')
            log.status('Returned from the addBPM primitive successfully')
                
            # Calling geminiScience toolbox function ADUtoElectons to do the work
            # of converting the pixels, updating headers and logging.
            log.debug('Calling geminiScience.addDQ')

            adOuts = geminiScience.add_dq(adIns=rc.getInputs(style='AD'), 
                                         fl_nonlinear=rc['fl_nonlinear'], 
                                         fl_saturated=rc['fl_saturated'], 
                                         suffix=rc['suffix'], logLevel=int(rc['logLevel']))    
           
            log.status('geminiScience.addDQ completed successfully')
            
            # Reporting the outputs to the reduction context
            rc.reportOutput(adOuts)          
                
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
        
        :param suffix: Value to be post pended onto each input name(s) to 
                         create the output name(s).
        :type suffix: string
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: int. 
                        This value can be set for each primitive individually 
                        in a recipe only (ie. not in the parameter file). 
                        If no value is specified during the recipe, the value 
                        set during the call to reduce or its default (2) will 
                        be used.
        """
        log = gemLog.getGeminiLog(logLevel=int(rc['logLevel']))
        try:
            log.fullinfo('*STARTING* to add the VAR frame(s) to the input data')
            
            
            # Calling geminiScience toolbox function ADUtoElectons to do the work
            # of converting the pixels, updating headers and logging.
            log.debug('Calling geminiScience.addVAR')
            
            adOuts = geminiScience.add_var(adIns=rc.getInputs(style='AD'), 
                                         suffix=rc['suffix'], 
                                         logLevel=int(rc['logLevel']))    
           
            log.status('geminiScience.addVAR completed successfully')
            
            # Reporting the outputs to the reduction context
            rc.reportOutput(adOuts)            
                
            log.status('*FINISHED* adding the VAR frame(s) to the input data')
        except:
            log.critical('Problem adding the VAR to one of '+rc.inputsAsStr())
            raise 
        yield rc 
    
    def adu_to_electrons(self,rc):
        """
        This primitive will convert the inputs from having pixel 
        units of ADU to electrons.
        
        :param suffix: Value to be post pended onto each input name(s) to 
                         create the output name(s).
        :type suffix: string
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: int. 
                        This value can be set for each primitive individually 
                        in a recipe only (ie. not in the parameter file). 
                        If no value is specified during the recipe, the value 
                        set during the call to reduce or its default (2) will 
                        be used.
        """
        log = gemLog.getGeminiLog(logLevel=int(rc['logLevel']))
        try:
            log.status('*STARTING* to convert the pixel values from '+
                       'ADU to electrons')
            # Calling geminiScience toolbox function ADUtoElectons to do the work
            # of converting the pixels, updating headers and logging.
            log.debug('Calling geminiScience.ADUtoElectrons')
            
            adOuts = geminiScience.ADUtoElectrons(adIns=rc.getInputs(style='AD'), 
                                                  suffix=rc['suffix'], 
                                                  logLevel=int(rc['logLevel']))    
           
            log.status('geminiScience.ADUtoElectrons completed successfully')
            
            # Reporting the outputs to the reduction context
            rc.reportOutput(adOuts)   
            
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
        
        :param suffix: Value to be post pended onto each input name(s) to 
                         create the output name(s).
        :type suffix: string
        
        :param fl_vardq: Create variance and data quality frames?
        :type fl_vardq: Python boolean (True/False), OR string 'AUTO' to do 
                        it automatically if there are VAR and DQ frames in the inputs.
                        NOTE: 'AUTO' uses the first input to determine if VAR 
                        and DQ frames exist, so, if the first does, then the 
                        rest MUST also have them as well.
    
        :param fl_dqprop: propogate the current DQ values?
        :type fl_dqprop: Python boolean (True/False)
    
        :param method: type of combining method to use.
        :type method: string, options: 'average', 'median'.
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: int. 
                        This value can be set for each primitive individually 
                        in a recipe only (ie. not in the parameter file). 
                        If no value is specified during the recipe, the value 
                        set during the call to reduce or its default (2) will 
                        be used.
        """        
        log = gemLog.getGeminiLog(logLevel=int(rc['logLevel']))
        try:
            if len(rc.getInputs())>1:
                log.status('*STARTING* combine the images of the input data')
                
                # Calling geminiScience toolbox function combine to do the work
                # of converting the pixels, updating headers and logging.
                log.debug('Calling geminiScience.combine')
                
                adOut = geminiScience.combine(adIns=rc.getInputs(style='AD'), 
                                              fl_vardq=rc['fl_vardq'], fl_dqprop=rc['fl_dqprop'], 
                                              method=rc['method'], suffix=rc['suffix'], 
                                              logLevel=int(rc['logLevel'])) 
                
                log.status('geminiScience.combine completed successfully')   
                
            else:
                log.status('makeFringeFrame was called with only one input, '+\
                           'so it just passed the inputs through without doing'+\
                           ' anything to them.')
            # Reporting the updated files to the reduction context
            rc.reportOutput(adOuts)
            
            log.status('*FINISHED* combining the images of the input data')
        except:
            log.critical('There was a problem combining '+rc.inputsAsStr())
            raise 
        yield rc

    def crashReduce(self, rc):
        raise 'Crashing'
        yield rc
        
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
        """
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: int. 
                        This value can be set for each primitive individually 
                        in a recipe only (ie. not in the parameter file). 
                        If no value is specified during the recipe, the value 
                        set during the call to reduce or its default (2) will 
                        be used.
        """
        log = gemLog.getGeminiLog(logLevel=int(rc['logLevel']))
        try:
            rc.rqDisplay(displayID=rc['displayID'])           
        except:
            log.critical('Problem displaying output')
            raise 
        yield rc
        
    def flatCorrect(self,rc):
        """
        This primitive performs a flat correction by dividing the inputs by a 
        processed flat similar to the way gireduce would perform this operation
        but written in pure Python in the arith toolbox.
          
        It is currently assumed that the same flat file will be applied to all
        input images.
        
        :param suffix: Value to be post pended onto each input name(s) to 
                         create the output name(s).
        :type suffix: string
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: int. 
                        This value can be set for each primitive individually 
                        in a recipe only (ie. not in the parameter file). 
                        If no value is specified during the recipe, the value 
                        set during the call to reduce or its default (2) will 
                        be used.
        """
        log = gemLog.getGeminiLog(logLevel=int(rc['logLevel']))
        try:
            log.status('*STARTING* to flat correct the inputs')
            
            # Retrieving the appropriate flat for the first of the inputs
            adOne = rc.getInputs(style='AD')[0]
            processedFlat = AstroData(rc.getCal(adOne,'flat'))
            
            # Taking care of the case where there was no, or an invalid flat 
            if processedFlat.countExts('SCI')==0:
                raise GEMINIException('Invalid processed flat retrieved')               
            
            log.debug('Calling geminiScience.flatCorrect function')
            
            adOuts = geminiScience.flat_correct(adIns=rc.getInputs(style='AD'),     
                                         flats=processedFlat, suffix=rc['suffix'], 
                                         logLevel=int(rc['logLevel']))           
            
            log.status('geminiScience.flatCorrect completed successfully')
              
            # Reporting the updated files to the reduction context
            rc.reportOutput(adOuts)   

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
    
    def getStackable(self, rc):
        """
        This primitive will check the files in the stack lists are on disk,
        and then update the inputs list to include all members of the stack 
        for stacking.
        
        :param purpose: 
        :type purpose: string, either: '' for regular image stacking, 
                       or 'fringe' for fringe stacking.
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: int. 
                        This value can be set for each primitive individually 
                        in a recipe only (ie. not in the parameter file). 
                        If no value is specified during the recipe, the value 
                        set during the call to reduce or its default (2) will 
                        be used.
        """
        log = gemLog.getGeminiLog(logLevel=int(rc['logLevel']))
        sidset = set()
        purpose=rc["purpose"]
        if purpose==None:
            purpose = ""
        try:
            for inp in rc.inputs:
                sidset.add(purpose+IDFactory.generateStackableID(inp.ad))
            for sid in sidset:
                stacklist = rc.getStack(sid) #.filelist
                log.fullinfo('Stack for stack id=%s' % sid)
                for f in stacklist:
                    rc.reportOutput(f)
                    log.fullinfo('   '+os.path.basename(f))
            yield rc
        except:
            log.critical('Problem getting stack '+sid, category='stack')

            raise 
        yield rc
    
    def measureIQ(self,rc):
        """
        This primitive will detect the sources in the input images and fit
        both Gaussian and Moffat models to their profiles and calculate the 
        Image Quality and seeing from this.
        
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
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: int. 
                        This value can be set for each primitive individually 
                        in a recipe only (ie. not in the parameter file). 
                        If no value is specified during the recipe, the value 
                        set during the call to reduce or its default (2) will 
                        be used.
        """
        #@@FIXME: Detecting sources is done here as well. This 
        # should eventually be split up into
        # separate primitives, i.e. detectSources and measureIQ.
        
        log = gemLog.getGeminiLog(logLevel=int(rc['logLevel']))
        try:
            log.status('*STARTING* to detect the sources'+
                       ' and measure the IQ of the inputs')
            
            log.debug('Calling geminiScience.measure_iq function')
            
            geminiScience.measure_iq(adIns=rc.getInputs(style='AD'),
                                     function=rc['function'],
                                     display=rc['display'],
                                     mosaic=rc['mosaic'], qa=rc['qa'],
                                     logLevel=int(rc['logLevel']))
            
            log.status('geminiScience.measure_iq completed successfully')
              
            # Reporting the original files through to the reduction context
            rc.reportOutput(rc.getInputs(style='AD'))           
            
            log.status('*FINISHED* measuring the IQ of the inputs')
        except:
            log.critical('There was a problem measuring the IQ of '+rc.inputsAsStr())
            raise 
        yield rc
 
    def pause(self, rc):
        rc.requestPause()
        yield rc
 
    def setContext(self, rc):
        rc.update(rc.localparms)
        yield rc   
       
    def setStackable(self, rc):
        """
        This primitive will update the lists of files to be stacked
        that have the same observationID with the current inputs.
        This file is cached between calls to reduce, thus allowing
        for one-file-at-a-time processing.
        
        :param purpose: 
        :type purpose: string, either: '' for regular image stacking, 
                       or 'fringe' for fringe stacking.
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: int. 
                        This value can be set for each primitive individually 
                        in a recipe only (ie. not in the parameter file). 
                        If no value is specified during the recipe, the value 
                        set during the call to reduce or its default (2) will 
                        be used.
        """
        log = gemLog.getGeminiLog(logLevel=int(rc['logLevel']))
        try:
            log.status('*STARTING* to update/create the stack')
            # Requesting for the reduction context to perform an update
            # to the stack cache file (or create it) with the current inputs.
            purpose = rc["purpose"]
            if purpose == None:
                purpose = ""
                
            rc.rqStackUpdate(purpose= purpose)
            # Writing the files in the stack to disk if not all ready there
            for ad in rc.getInputs(style='AD'):
                if not os.path.exists(ad.filename):
                    log.fullinfo('writing '+ad.filename+\
                                 ' to disk', category='stack')
                    ad.write(ad.filename)
                    
            log.status('*FINISHED* updating/creating the stack')
        except:
            log.critical('Problem writing stack for files '+rc.inputsAsStr(),
                         category='stack')
            raise
        yield rc
    
    def showCals(self, rc):
        """
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: int. 
                        This value can be set for each primitive individually 
                        in a recipe only (ie. not in the parameter file). 
                        If no value is specified during the recipe, the value 
                        set during the call to reduce or its default (2) will 
                        be used.
        """
        log = gemLog.getGeminiLog(logLevel=int(rc['logLevel']))
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
                        log.fullinfo(rc.calibrations[calkey], 
                                     category='calibrations')
            if (num == 0):
                log.warning('There are no calibrations in the cache.')
        yield rc
    ptusage_showCals='Used to show calibrations currently in cache for inputs.'

    def showInputs(self, rc):
        """
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: int. 
                        This value can be set for each primitive individually 
                        in a recipe only (ie. not in the parameter file). 
                        If no value is specified during the recipe, the value 
                        set during the call to reduce or its default (2) will 
                        be used.
        """
        log = gemLog.getGeminiLog(logLevel=int(rc['logLevel']))
        log.fullinfo('Inputs:',category='inputs')
        for inf in rc.inputs:
            log.fullinfo('  '+inf.filename, category='inputs')  
        yield rc  
    showFiles = showInputs

    def showParameters(self, rc):
        """
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: int. 
                        This value can be set for each primitive individually 
                        in a recipe only (ie. not in the parameter file). 
                        If no value is specified during the recipe, the value 
                        set during the call to reduce or its default (2) will 
                        be used.
        """
        log = gemLog.getGeminiLog(logLevel=int(rc['logLevel']))
        rcparams = rc.paramNames()
        if (rc['show']):
            toshows = rc['show'].split(':')
            for toshow in toshows:
                if toshow in rcparams:
                    log.fullinfo(toshow+' = '+repr(rc[toshow]), 
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
         
    def showStackable(self, rc):
        """
        This primitive will log the list of files in the stacking list matching
        the current inputs and 'purpose' value.  
        
        :param purpose: 
        :type purpose: string, either: '' for regular image stacking, 
                       or 'fringe' for fringe stacking.
                       
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: int. 
                        This value can be set for each primitive individually 
                        in a recipe only (ie. not in the parameter file). 
                        If no value is specified during the recipe, the value 
                        set during the call to reduce or its default (2) will 
                        be used.
        """
        log = gemLog.getGeminiLog(logLevel=int(rc['logLevel']))
        sidset = set()
        purpose = rc["purpose"]
        if purpose == None:
            purpose = ""
        # print "pG710"
        if purpose == "all":
            allsids = rc.getStackIDs()
            # print "pG713:", repr(allsids)
            for sid in allsids:
                sidset.add(sid)
        else:   
            for inp in rc.inputs:
                sidset.add(purpose+IDFactory.generateStackableID(inp.ad))
        for sid in sidset:
            stacklist = rc.getStack(sid) #.filelist
            log.status('Stack for stack id=%s' % sid)
            if len(stacklist)>0:
                for f in stacklist:
                    log.status('    '+os.path.basename(f))
            else:
                log.status("    no datasets in list")
        yield rc
            
    def sleep(self, rc):
        log = gemLog.getGeminiLog(logLevel=int(rc['logLevel']))
        if rc['duration']:
            dur = float(rc['duration'])
        else:
            dur = 5.
        log.status('Sleeping for %f seconds' % dur)
        time.sleep(dur)
        yield rc
             
    def standardizeHeaders(self, rc):
        """
        This primitive updates and adds the important header keywords
        for the input MEFs. First the general headers for Gemini will 
        be update/created, followed by those that are instrument specific.
        
        :param suffix: Value to be post pended onto each input name(s) to 
                         create the output name(s).
        :type suffix: string
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: int. 
                        This value can be set for each primitive individually 
                        in a recipe only (ie. not in the parameter file). 
                        If no value is specified during the recipe, the value 
                        set during the call to reduce or its default (2) will 
                        be used.
        """
        log = gemLog.getGeminiLog(logLevel=int(rc['logLevel']))
        try:   
            log.status('*STARTING* to standardize the headers')
            log.status('Standardizing observatory general headers')            
            for ad in rc.getInputs(style='AD'):
                log.debug('calling gemt.stdObsHdrs for '+ad.filename)
                gemt.stdObsHdrs(ad, logLevel=int(rc['logLevel']))
                log.status('Completed standardizing the headers for '+
                           ad.filename)
   
            log.status('Observatory headers fixed')
            log.debug('Calling standardizeInstrumentHeaders primitive')
            log.status('Standardizing instrument specific headers')
            
            # Calling standarizeInstrumentHeaders primitive
            rc.run('standardizeInstrumentHeaders(logLevel='+str(rc['logLevel'])+')') 
            log.status('Instrument specific headers fixed')
            
            # Updating the file name with the suffix/outsuffix  and timestamps 
            # for this primitive and then reporting the new file to the 
            # reduction context 
            for ad in rc.getInputs(style='AD'):
                # Adding a GEM-TLM (automatic) and STDHDRS time stamps 
                # to the PHU
                ad.historyMark(key='STDHDRS',stomp=False)
                log.debug('Calling gemt.fileNameUpdater on '+ad.filename)
                ad.filename = gemt.fileNameUpdater(adIn=ad, 
                                                   suffix=rc['suffix'], 
                                                   strip=False, logLevel= int(rc['logLevel']))
                log.status('File name updated to '+ad.filename)
                # Updating logger with updated/added time stamps
                log.fullinfo('************************************************'
                             ,category='header')
                log.fullinfo('file = '+ad.filename, category='header')
                log.fullinfo('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
                             , 'header')
                log.fullinfo('PHU keywords updated/added:\n', category='header')
                log.fullinfo('GEM-TLM = '+ad.phuGetKeyValue('GEM-TLM'), 
                             category='header')
                log.fullinfo('STDHDRS = '+ad.phuGetKeyValue('STDHDRS'), 
                             category='header')
                log.fullinfo('------------------------------------------------'
                             , category='header')    
                rc.reportOutput(ad)
                
            log.status('*FINISHED* standardizing the headers')
        except:
            log.critical('Problem preparing one of '+rc.inputsAsStr())
            raise 
        yield rc
                                 
    def standardizeStructure(self,rc):
        """
        This primitive ensures the MEF structure is ready for further 
        processing, through adding the MDF if necessary and the needed 
        keywords to the headers.  First the MEF's will be checked for the 
        general Gemini structure requirements and then the instrument specific
        ones if needed. If the data requires a MDF to be attached, use the 
        'addMDF' flag to make this happen 
        (eg. standardizeStructure(addMDF=True)).
        
        :param suffix: Value to be post pended onto each input name(s) to 
                         create the output name(s).
        :type suffix: string
        
        :param addMDF: A flag to turn on/off appending the appropriate MDF 
                       file to the inputs.
        :type addMDF: Python boolean (True/False)
                      default: True
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: int. 
                        This value can be set for each primitive individually 
                        in a recipe only (ie. not in the parameter file). 
                        If no value is specified during the recipe, the value 
                        set during the call to reduce or its default (2) will 
                        be used.
        """
        log = gemLog.getGeminiLog(logLevel=int(rc['logLevel']))
        try:
            log.status('*STARTING* to standardize the structure of input data')
            
            #$$$$ MAYBE SET THIS TO FALSE IF GMOS_IMAGE AND TRUE IF GMOS_SPEC?$$$$$$$
            # Add the MDF if not set to false
            if rc['addMDF'] is True:
                log.debug('Calling attachMDF primitive')
                # Calling the attachMDF primitive
                rc.run('attachMDF(logLevel='+str(rc['logLevel'])+')')
                log.status('Successfully returned to '+
                           'standardizeStructure from the attachMDF primitive')

            for ad in rc.getInputs(style='AD'):
                log.debug('Calling gemt.stdObsStruct on '+ad.filename)
                gemt.stdObsStruct(ad)
                log.status('Completed standardizing the structure for '+
                           ad.filename)
                
                # Adding a GEM-TLM (automatic) and STDSTRUC time stamps 
                # to the PHU
                ad.historyMark(key='STDSTRUC',stomp=False)
                # Updating the file name with the suffix for this   
                # primitive and then reporting the new file to the reduction 
                # context
                log.debug('Calling gemt.fileNameUpdater on '+ad.filename)
                ad.filename = gemt.fileNameUpdater(adIn=ad, 
                                                   suffix=rc['suffix'], 
                                                   strip=False, logLevel= int(rc['logLevel']))
                log.status('File name updated to '+ad.filename)
                # Updating logger with updated/added time stamps
                log.fullinfo('************************************************'
                             ,category='header')
                log.fullinfo('file = '+ad.filename, category='header')
                log.fullinfo('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
                             , category='header')
                log.fullinfo('PHU keywords updated/added:\n', category='header')
                log.fullinfo('GEM-TLM = '+ad.phuGetKeyValue('GEM-TLM'), 
                             category='header')
                log.fullinfo('STDSTRUC = '+ad.phuGetKeyValue('STDSTRUC'), 
                             category='header')
                log.fullinfo('------------------------------------------------'
                             , category='header')  
                rc.reportOutput(ad)
   
            log.status('*FINISHED* standardizing the structure of input data')
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
        
        :param clob: Write over any previous file with the same name that
                     all ready exists?
        :type clob: Python boolean (True/False)
                    default: False
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: int. 
                        This value can be set for each primitive individually 
                        in a recipe only (ie. not in the parameter file). 
                        If no value is specified during the recipe, the value 
                        set during the call to reduce or its default (2) will 
                        be used.
        """
        log = gemLog.getGeminiLog(logLevel=int(rc['logLevel']))
        try:  
            log.status('*STARTING* to store the processed bias by writing '+
                       'it to disk')
            for ad in rc.getInputs(style='AD'):
                # Updating the file name with the suffix for this
                # primitive and then reporting the new file to the reduction 
                # context
                log.debug('Calling gemt.fileNameUpdater on '+ad.filename)
                ad.filename = gemt.fileNameUpdater(adIn=ad, 
                                                   suffix='_preparedbias', 
                                                   strip=True, 
                                                   logLevel= int(rc['logLevel']))
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
        
        :param clob: Write over any previous file with the same name that
                     all ready exists?
        :type clob: Python boolean (True/False)
                    default: False
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: int. 
                        This value can be set for each primitive individually 
                        in a recipe only (ie. not in the parameter file). 
                        If no value is specified during the recipe, the value 
                        set during the call to reduce or its default (2) will 
                        be used.
        """
        log = gemLog.getGeminiLog(logLevel=int(rc['logLevel']))
        try:   
            log.status('*STARTING* to store the processed flat by writing it to disk')
            for ad in rc.getInputs(style='AD'):
                # Updating the file name with the suffix for this
                # primitive and then reporting the new file to the reduction 
                # context
                log.debug('Calling gemt.fileNameUpdater on '+ad.filename)
                ad.filename = gemt.fileNameUpdater(adIn=ad, 
                                                   suffix='_preparedflat', 
                                                   strip=True, 
                                                   logLevel= int(rc['logLevel']))
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
        
    def time(self, rc):
        log = gemLog.getGeminiLog(logLevel=int(rc['logLevel']))
        cur = datetime.now()
        
        elap = ''
        if rc['lastTime'] and not rc['start']:
            td = cur - rc['lastTime']
            elap = ' (%s)' %str(td)
        log.fullinfo('Time:'+' '+str(datetime.now())+' '+elap)
        
        rc.update({'lastTime':cur})
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
        
        :param suffix: Value to be post pended onto each input name(s) to 
                         create the output name(s).
        :type suffix: string
        
        :param repair: A flag to turn on/off repairing the data if there is a
                       problem with it. 
                       Note: this feature does not work yet.
        :type repair: Python boolean (True/False)
                      default: True
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: int. 
                        This value can be set for each primitive individually 
                        in a recipe only (ie. not in the parameter file). 
                        If no value is specified during the recipe, the value 
                        set during the call to reduce or its default (2) will 
                        be used.
        """
        log = gemLog.getGeminiLog(logLevel=int(rc['logLevel']))
        try:           
            if rc['repair'] is True:
               # This should repair the file if it is broken, but this function
               # isn't coded yet and would require some sort of flag set while 
               # checking the data to tell this to perform the corrections
               log.critical('Sorry, but the repair feature of validateData' +
                            ' is not available yet')
            
            log.status('*STARTING* to validate the input data')
            
            log.debug('Calling validateInstrumentData primitive')
            # Calling the validateInstrumentData primitive 
            rc.run('validateInstrumentData(logLevel='+str(rc['logLevel'])+')')
            log.status('Successfully returned to validateData'+
                       ' from the validateInstrumentData primitive') 
            
            # Updating the file name with the suffix  and timestamps 
            # for this primitive and then reporting the new file to the 
            # reduction context 
            for ad in rc.getInputs(style='AD'):
                # Adding a GEM-TLM (automatic) and VALDATA time stamps 
                # to the PHU
                ad.historyMark(key='VALDATA',stomp=False)
                log.debug('calling gemt.gemt.fileNameUpdater on '+ad.filename)        
                ad.filename = gemt.fileNameUpdater(adIn=ad, 
                                                   suffix='_validated', 
                                                   strip=False, logLevel= int(rc['logLevel']))                
                log.status('File name updated to '+ad.filename)
                # Updating logger with updated/added time stamps
                log.fullinfo('************************************************'
                             ,'header')
                log.fullinfo('File = '+ad.filename, category='header')
                log.fullinfo('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
                             , category='header')
                log.fullinfo('PHU keywords updated/added:\n', category='header')
                log.fullinfo('GEM-TLM = '+ad.phuGetKeyValue('GEM-TLM'), 
                              category='header')
                log.fullinfo('VALDATA = '+ad.phuGetKeyValue('VALDATA'), 
                             category='header')
                log.fullinfo('------------------------------------------------'
                             , category='header')  
                rc.reportOutput(ad) 
                        
            log.status('*FINISHED* validating input data')                
        except:
            log.critical('Problem preparing one of  '+rc.inputsAsStr())
            raise 
        yield rc

    def writeOutputs(self,rc):
        """
        A primitive that may be called by a recipe at any stage to
        write the outputs to disk.
        If suffix is set during the call to writeOutputs, any previous 
        suffixs will be striped and replaced by the one provided.
        examples: 
        writeOutputs(suffix= '_string'), writeOutputs(prefix= '_string') 
        or if you have a full file name in mind for a SINGLE file being 
        ran through Reduce you may use writeOutputs(outfilename='name.fits').
        
        :param strip: Strip the previously suffixed strings off file name?
        :type strip: Python boolean (True/False)
                     default: False
        
        :param clobber: Write over any previous file with the same name that
                        all ready exists?
        :type clobber: Python boolean (True/False)
                       default: False
        
        :param suffix: Value to be post pended onto each input name(s) to 
                         create the output name(s).
        :type suffix: string
        
        :param prefix: Value to be post pended onto each input name(s) to 
                         create the output name(s).
        :type prefix: string
        
        :param outfilename: The full filename you wish the file to be written to.
                            Note: this only works if there is ONLY ONE file in the inputs.
        :type outfilename: string
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: int. 
                        This value can be set for each primitive individually 
                        in a recipe only (ie. not in the parameter file). 
                        If no value is specified during the recipe, the value 
                        set during the call to reduce or its default (2) will 
                        be used.
        """
        log = gemLog.getGeminiLog(logLevel=int(rc['logLevel']))
        try:
            log.status('*STARTING* to write the outputs')
            
            # Logging current values of suffix and prefix
            log.status('suffix = '+str(rc['suffix']))
            log.status('prefix = '+str(rc['prefix']))
            log.status('strip = '+str(rc['strip']))
            
            if rc['suffix'] and rc['prefix']:
                log.critical('The input will have '+rc['prefix']+' pre pended'+
                             ' and '+rc['suffix']+' post pended onto it')
                
            for ad in rc.getInputs(style='AD'):
                # If the value of 'suffix' was set, then set the file name 
                # to be written to disk to be postpended by it
                if rc['suffix']:
                    log.debug('calling gemt.fileNameUpdater on '+ad.filename)
                    ad.filename = gemt.fileNameUpdater(adIn=ad, 
                                        suffix=rc['suffix'], 
                                        strip=rc['strip'], logLevel= int(rc['logLevel']))
                    log.status('File name updated to '+ad.filename)
                    outfilename = os.path.basename(ad.filename)
                    
                # If the value of 'prefix' was set, then set the file name 
                # to be written to disk to be pre pended by it
                if rc['prefix']:
                    infilename = os.path.basename(ad.filename)
                    outfilename = rc['prefix']+infilename
                    
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
         