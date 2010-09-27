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

def pyrafLoader(rc = None):
    '''
    This function is to load the modules needed by primitives that use pyraf. 
    It will also ensure there are no additional prints to the console when loading
    the Gemini pyraf package.
    The loaded modules are returned in the order of:
    (pyraf, gemini, iraf.yes, iraf.no)
    to be added to the name-space of the primitive this function is called from.
    eg. (pyraf, gemini, yes, no)=pyrafLoader(rc)
    '''
    import pyraf
    from pyraf import iraf
    from iraf import gemini
    import StringIO
    
    SAVEOUT = sys.stdout
    capture = StringIO.StringIO()
    sys.stdout = capture
    
    yes = iraf.yes
    no = iraf.no
        
    gemini() # this will load the gemini pyraf package
        
    return (pyraf,gemini,iraf.yes,iraf.no)

class GEMINIException:
    ''' This is the general exception the classes and functions in the
    Structures.py module raise.
    '''
    def __init__(self, msg='Exception Raised in Recipe System'):
        '''This constructor takes a message to print to the user.'''
        self.message = msg
    def __str__(self):
        '''This str conversion member returns the message given by the 
        user (or the default message)
        when the exception is not caught.'''
        return self.message

class GEMINIPrimitives(PrimitiveSet):
    ''' 
    This is the class of all primitives for the GEMINI astrotype of 
    the hierarchy tree.  It inherits all the primitives to the level above
    , 'PrimitiveSet'.
    '''
    astrotype = 'GEMINI'
    
    def init(self, rc):
        return 
    init.pt_hide = True
    
    def pause(self, rc):
        rc.requestPause()
        yield rc 

    def crashReduce(self, rc):
        raise 'Crashing'
        yield rc
        
    def clearCalCache(self, rc):
        # print 'pG61:', rc.calindfile
        rc.persistCalIndex(rc.calindfile, newindex = {})
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
            rc.rqDisplay(displayID=rc['displayID'])           
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
                    log.fullinfo(toshow+' = '+repr(rc[toshow]), category='parameters')
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
            log.fullinfo('  '+inf.filename,category='inputs')  
        yield rc  
    showFiles = showInputs
    
    def showCals(self, rc):
        if str(rc['showcals']).lower() == 'all':
            num = 0
            # print 'pG256: showcals=all', repr (rc.calibrations)
            for calkey in rc.calibrations:
                num += 1
                log.fullinfo(rc.calibrations[calkey],category='calibrations')
            if (num == 0):
                log.warning( 'There are no calibrations in the cache.')
        else:
            for adr in rc.inputs:
                sid = IDFactory.generateAstroDataID(adr.ad)
                num = 0
                for calkey in rc.calibrations:
                    if sid in calkey :
                        num += 1
                        log.fullinfo(rc.calibrations[calkey],category='calibrations')
            if (num == 0):
                log.warning('There are no calibrations in the cache.')
        yield rc
    ptusage_showCals='Used to show calibrations currently in cache for inputs.'

    def showStackable(self, rc):
        sidset = set()
        for inp in rc.inputs:
            sidset.add( IDFactory.generateStackableID( inp.ad ))
        for sid in sidset:
            stacklist = rc.getStack(sid).filelist
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
        '''
        This primitive will check the files in the stack lists are on disk
        , if not write them to disk and then update the inputs list to include
        all members of the stack for stacking.
        '''
        try:
            # @@REFERENCE IMAGE @@NOTE: to pick which stackable list to get
            stackid = IDFactory.generateStackableID(rc.inputs[0].ad)
            log.fullinfo('getting stack '+stackid,'stack')
            rc.rqStackGet()
            yield rc
            stack = rc.getStack(stackid).filelist
            #print 'prim_G366: ',repr(stack)
            rc.reportOutput(stack)
        except:
            log.critical('Problem getting stack '+stackid, 'stack')
            raise 
        yield rc      
 
    def setStackable(self, rc):
        '''
        This primitive will update the lists of files to be stacked
        that have the same observationID with the current inputs.
        This file is cached between calls to reduce, thus allowing
        for one-file-at-a-time processing.
        '''
        try:
            stackid = IDFactory.generateStackableID(rc.inputs[0].ad)
            log.fullinfo('updating stack '+stackid+' with '+rc.inputsAsStr(), category='stack')
            rc.rqStackUpdate()
            # writing the files in the stack to disk if not all ready there
            for ad in rc.getInputs(style='AD'):
                if not os.path.exists(ad.filename):
                    log.fullinfo('temporarily writing '+ad.filename+\
                                 ' to disk', category='stack')
                    ad.write(ad.filename)
        except:
            log.critical('Problem preparing stack for files '+rc.inputsAsStr(),category='stack')
            raise
        yield rc
    
    def validateData(self,rc):
        '''
        This primitive will ensure the data is not corrupted or in an odd 
        format that will affect later steps in the reduction process.  
        It will call a function to take care of the general Gemini issues 
        and then one for the instrument specific ones. If there are issues 
        with the data, the flag 'repair' can be used to turn on the feature to 
        repair it or not (eg. validateData(repair=True))
        (this feature is not coded yet).
        '''
        
        try:
            if rc['repair']==True:
               # this should repair the file if it is broken, but this function
               # isn't coded yet and would require some sort of flag set while 
               # checking the data to tell this to perform the corrections
               log.critical('Sorry, but the repair feature of validateData is not available yet')
               pass
            
            log.status('*STARTING* to validate the input data')
            log.debug('calling validateInstrumentData primitive')
            
            # calling the validateInstrumentData primitive 
            rc.run('validateInstrumentData') 
            
            # updating the filenames in the RC
            for ad in rc.getInputs(style='AD'):
                log.debug('calling gemt.gemt.fileNameUpdater on '+ad.filename)        
                ad.filename=gemt.fileNameUpdater(ad.filename,postpend='_validated',strip=False)
                log.status('gemt.fileNameUpdater updated the file name to '+ad.filename)
                rc.reportOutput(ad) 
                        
            log.status('*FINISHED* validating input data')                
        except:
            log.critical('Problem preparing one of these inputs '+rc.inputsAsStr())
            raise 
        yield rc

    def standardizeStructure(self,rc):
        '''
        This primitive ensures the MEF structure is ready for further 
        processing, through adding the MDF if necessary and the needed 
        keywords to the headers.  First the MEF's will be checked for the 
        general Gemini structure requirements and then the instrument specific
        ones if needed. If the data requires a MDF to be attached, use the 
        'addMDF' flag to make this happen 
        (eg. standardizeStructure(addMDF=True)).
        '''
        
        try:
            log.status('*STARTING* to standardize the structure of input data')
            
            # add the MDF if not set to false
            if rc['addMDF']==True:
                log.debug('calling attachMDF primitive')
                
                # calling the attachMDF primitive
                rc.run('attachMDF')

            for ad in rc.getInputs(style='AD'):
                log.debug('calling gemt.stdObsStruct on '+ad.filename)
                gemt.stdObsStruct(ad)
                log.status('gemt.stdObsStruct completed standardizing the structure for '+ad.filename)
                
                # updating the filenames in the RC
                log.debug('calling gemt.fileNameUpdater on '+ad.filename)
                ad.filename=gemt.fileNameUpdater(ad.filename,postpend=rc['outsuffix'], strip=False)
                log.status('gemt.fileNameUpdater updated the file name to '+ad.filename)
                rc.reportOutput(ad)
            
            log.status('*FINISHED* standardizing the structure of input data')
        except:
            log.critical('Problem preparing one of these inputs '+rc.inputsAsStr())
            raise 
        yield rc
        
    def standardizeHeaders(self,rc):
        '''
        This primitive updates and adds the important header keywords
        for the input MEFs. First the general headers for Gemini will 
        be update/created, followed by those that are instrument specific.
        '''
        
        try:   
            log.status('*STARTING* to standardize the headers')
            log.status('standardizing observatory general headers')            
            for ad in rc.getInputs(style='AD'):
                log.debug('calling gemt.stdObsHdrs for '+ad.filename)
                gemt.stdObsHdrs(ad)
                log.status('gemt.stdObsHdrs completed standardizing the headers for '+ad.filename)
   
            log.status('observatory headers fixed')
            log.debug('calling standardizeInstrumentHeaders primitive')
            log.status('standardizing instrument specific headers')
            
            # calling standarizeInstrumentHeaders primitive
            rc.run('standardizeInstrumentHeaders') 
            log.status('instrument specific headers fixed')
            
            # updating the filenames in the RC 
            for ad in rc.getInputs(style='AD'):
                log.debug('calling gemt.fileNameUpdater on '+ad.filename)
                ad.filename=gemt.fileNameUpdater(ad.filename, postpend=rc['outsuffix'], strip=True)
                log.status('gemt.fileNameUpdater updated the file name to '+ad.filename)
                rc.reportOutput(ad)
                
            log.status('*FINISHED* standardizing the headers')
        except:
            log.critical('Problem preparing one of these inputs '+rc.inputsAsStr())
            raise 
        yield rc 

    def calculateVAR(self,rc):
        '''
        This primitive uses numpy to calculate the variance of each SCI frame
        in the input files and appends it as a VAR frame using AstroData.
        
        The calculation will follow the formula:
        variance = (read noise/gain)2 + max(data,0.0)/gain
        '''
        try:
            log.fullinfo('*STARTING* to add the VAR frame(s) to the input data')
            
            for ad in rc.getInputs(style='AD'):
                print ad.info()
                for sciExt in ad['SCI']:
                    # var = (read noise/gain)2 + max(data,0.0)/gain
                    # equation preparation
                    readNoise=sciExt.read_noise()
                    gain=sciExt.gain()
                    # creating (read noise/gain) constant
                    rnOverG=readNoise/gain
                    # convert negative numbers (if they exist) to zeros
                    maxArray=np.where(sciExt.data>0.0,0,sciExt.data)
                    # creating max(data,0.0)/gain array
                    maxOverGain=np.divide(maxArray,gain)
                    # put it all together
                    varArray=np.add(maxOverGain,rnOverG*rnOverG)
                     
                    # creating the variance frame's header       
                    varheader = pf.Header()
                    varheader.update('NAXIS', 2)
                    varheader.update('PCOUNT', 0, 'required keyword; must = 0 ')
                    varheader.update('GCOUNT', 1, 'required keyword; must = 1')
                    # varHDU.renameExt('VAR', sciExt.extver())
                    varheader.update('EXTNAME', 'VAR', 'Extension Name')
                    varheader.update('EXTVER', sciExt.extver(), 'Extension Version')
                    varheader.update('BITPIX', 32, 'number of bits per data pixel')
                    
                    # turning individual variance header and data into one astrodata instance
                    varAD = AstroData( header = varheader, data = varArray )
                    
                    # appending variance astrodata instance onto input one
                    log.fullinfo('appending new HDU onto the file '+ ad.filename)
                    ad.append(varAD)
                    log.fullinfo('appending complete for '+ad.filename)
                    
                    ## updating logger with updated/added keywords
                    log.fullinfo('****************************************************','header')
                    log.fullinfo('file = '+ad.filename,'header')
                    log.fullinfo('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~','header')
                    log.fullinfo('VAR extension number '+str(sciExt.extver())+' keywords updated/added:\n', 'header')
                    log.fullinfo('BITPIX= '+str(32),'header' )
                    log.fullinfo('NAXIS= '+str(2),'header' )
                    log.fullinfo('EXTNAME= '+'VAR','header' )
                    log.fullinfo('EXTVER= '+str(sciExt.extver()),'header' )
                    log.fullinfo('---------------------------------------------------','header')
                
                ut =  ad.historyMark()
                ad.historyMark(key='ADDVARDQ',stomp=False)    
                
                log.fullinfo('****************************************************','header')
                log.fullinfo('file = '+ad.filename,'header')
                log.fullinfo('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~','header')
                log.fullinfo('PHU keywords updated/added:\n', 'header')
                log.fullinfo('GEM-TLM = '+str(ut),'header' )
                log.fullinfo('ADDVARDQ = '+str(ut),'header' )
                log.fullinfo('---------------------------------------------------','header')
                
                # check if there filename all ready has the suffix 
                # '_vardq', if not add it
                if not re.search(rc['outsuffix'],ad.filename): # this is printing a 'None' on the screen, fix that!!!
                    log.debug('calling gemt.fileNameUpdater on '+ad.filename)
                    ad.filename=gemt.fileNameUpdater(ad.filename, postpend=rc['outsuffix'], strip=False)
                    log.status('gemt.fileNameUpdater updated the file name to '+ad.filename)
                rc.reportOutput(ad)        
                
            log.status('*FINISHED* adding the VAR frame(s) to the input data')
        except:
            log.critical('Problem adding the VARDQ to one of '+rc.inputsAsStr())
            raise 
        yield rc 

    def calculateDQ(self,rc):
        '''
        This primitive will create a numpy array for the data quality 
        of each SCI frame of the input data. This will then have a 
        header created and append to the input using AstroData as a DQ 
        frame. The value of a pixel will be the sum of the following: 
        (0=good, 1=bad pixel (found in bad pixel mask), 
        2=value is in non linear region, 4=pixel is saturated)
        '''
        try:
            log.status('*STARTING* to add the DQ frame(s) to the input data')
            
            #$$$$$$$$$$$$$ GMOS IMAGE specific block, consider moving $$$$$$$$$
            packagePath=sys.argv[0].split('gemini_python')[0]
            calPath='gemini_python/test_data/test_cal_files/GMOS_BPM_files/'
            BPM_11=AstroData(packagePath+calPath+'GMOS_BPM_11.fits')
            BPM_22=AstroData(packagePath+calPath+'GMOS_BPM_22.fits')
            #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            
            for ad in rc.getInputs(style='AD'):
                for sciExt in ad['SCI']:
                    #$$ GMOS IMAGE specific block, consider moving $$$$$$$$$$$
                    if sciExt.getKeyValue('CCDSUM')=='1 1':
                        BPMArray=BPM_11['DQ'][sciExt.extver()-1].data
                        BPMfilename = 'GMOS_BPM_11.fits'
                    elif sciExt.getKeyValue('CCDSUM')=='2 2':
                        BPMArray=BPM_22['DQ'][sciExt.extver()-1].data
                        BPMfilename = 'GMOS_BPM_22.fits'
                    else:
                        BPMArray=np.zeros(sciExt.data.shape,dtype=np.int16)
                        log.error('CCDSUM is not 1x1 or 2x2, using zeros array for BPM')
                        BPMfilename='None'
                    # ensuring BPM array is in binary format
                    BPMArray=np.where(BPMArray>=1,1,0)
                    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$    
                    
                    # getting the data section from the header and converting
                    # to an integer list
                    datasecStr=sciExt.data_section()
                    datasecList=gemt.secStrToIntList(datasecStr) 
                    dsl=datasecList
                    
                    # preparing the non linear and saturated pixel arrays
                    # and their respective constants
                    nonLinArray=np.zeros(sciExt.data.shape,dtype=np.int16)
                    saturatedArray=np.zeros(sciExt.data.shape,dtype=np.int16)
                    linear=sciExt.non_linear_level()
                    saturated=sciExt.saturation_level()

                    if (linear!=None) and (rc['fl_nonlinear']==True): 
                        log.debug('performing a np.where to find non-linear pixels'+\
                                  ' for extension '+sciExt.extver()+' of '+ad.filename)
                        nonLinArray=np.where(sciExt.data>linear,2,0)
                        log.status('finished calculating array of non-linear pixels')
                    if (saturated!=None) and (rc['fl_saturated']==True):
                        log.debug('performing a np.where to find saturated pixels'+\
                                  ' for extension '+sciExt.extver()+' of '+ad.filename)
                        saturatedArray=np.where(sciExt.data>saturated,4,0)
                        log.status('finished calculating array of saturated pixels')
                    
                    # BPM file has had its overscan region trimmed all ready, 
                    # so must trim the overscan section from the nonLin and 
                    # saturated arrays to match
                    nonLinArrayTrimmed = nonLinArray[dsl[2]-1:dsl[3],dsl[0]-1:dsl[1]]
                    saturatedArrayTrimmed = saturatedArray[dsl[2]-1:dsl[3],dsl[0]-1:dsl[1]]  
                    
                    # creating one DQ array from the three
                    dqArray=np.add(BPMArray,nonLinArrayTrimmed,saturatedArrayTrimmed) 
                    
                    # creating a header for the dq array and updating it
                    dqheader = pf.Header()
                    dqheader.update('BITPIX', 16, 'number of bits per data pixel')
                    dqheader.update('NAXIS', 2)
                    dqheader.update('PCOUNT', 0, 'required keyword; must = 0 ')
                    dqheader.update('GCOUNT', 1, 'required keyword; must = 1')
                    dqheader.update('BUNIT', 'bit', 'Physical units')
                    dqheader.update('BPMFILE', BPMfilename, 'Name of input Bad Pixel Mask file')
                    dqheader.update('EXTNAME', 'DQ', 'Extension Name')
                    dqheader.update('EXTVER', sciExt.extver(), 'Extension Version')
                    
                    # creating an astrodata instance from the dq array and header
                    dqAD = AstroData( header = dqheader, data = dqArray )
                    
                    # appending data quality astrodata instance to the input one
                    log.fullinfo('appending new HDU onto the file '+ ad.filename)
                    ad.append(dqAD)
                    log.fullinfo('appending complete for '+ ad.filename)
                    
                    # updating logger with updated/added keywords for this extension
                    log.fullinfo('****************************************************','header')
                    log.fullinfo('file = '+ad.filename,'header')
                    log.fullinfo('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~','header')
                    log.fullinfo('DQ extension number '+str(sciExt.extver())+\
                                 ' keywords updated/added:\n', 'header')
                    log.fullinfo('BITPIX= '+str(16),'header' )
                    log.fullinfo('NAXIS= '+str(2),'header' )
                    log.fullinfo('BUNIT= '+'bit','header' )
                    log.fullinfo('BPMFILE= '+BPMfilename,'header' )
                    log.fullinfo('EXTNAME= '+'DQ','header' )
                    log.fullinfo('EXTVER= '+str(sciExt.extver()),'header' )
                    log.fullinfo('---------------------------------------------------','header')
                
                # adding a GEM-TLM and ADDVARDQ time stamp 
                ut = ad.historyMark() 
                ad.historyMark(key='ADDVARDQ',stomp=False) 
                # updating logger with updated/added time stamps
                log.fullinfo('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~','header')
                log.fullinfo('PHU keywords updated/added:\n', 'header')
                log.fullinfo('GEM-TLM = '+str(ut),'header' )
                log.fullinfo('ADDVARDQ = '+str(ut),'header' )
                log.fullinfo('---------------------------------------------------','header')
                
                # check if the filename all ready has the suffix 
                # '_vardq', if not add it
                if not re.search(rc['outsuffix'],ad.filename): # this is printing a 'None' on the screen, fix that!!!
                    log.debug('calling gemt.fileNameUpdater on '+ad.filename)
                    ad.filename=gemt.fileNameUpdater(ad.filename, postpend=rc['outsuffix'], strip=False)
                    log.status('output of addDQ will have the filename: '+ad.filename)
                rc.reportOutput(ad)        
            
            log.status('*FINISHED* adding the DQ frame(s) to the input data')
        except:
            log.critical('Problem adding the VARDQ to one of '+rc.inputsAsStr())
            raise 
        yield rc 
            
    def combine(self,rc):
        '''
        This primitive will average and combine the SCI extensions of the 
        inputs. It takes all the inputs and creates a list of them and 
        then combines each of their SCI extensions together to create 
        average combination file. New VAR frames are made from these 
        combined SCI frames and the DQ frames are propagated through 
        to the final file.
        '''
        # loading and bringing the pyraf related modules into the name-space
        pyraf,gemini,yes,no = pyrafLoader(rc)
        
        try:
            if len(rc.getInputs())>1:
                log.status('*STARTING* combine the images of the input data')
                
                # preparing input files, lists, parameters... for input to 
                # the CL script
                clm=gemt.CLManager(rc)
                clm.LogCurParams()
                
                # params set by the CLManager or the definition of the prim 
                clPrimParams={
                    'input'       :clm.inputList(),
                    'output'      :clm.combineOutname(),  # maybe allow the user to override this in the future. 
                    'Stdout'      :IrafStdout(),          # this is actually in the default dict but wanted to show it again
                    'Stderr'      :IrafStdout(),          # this is actually in the default dict but wanted to show it again
                    'logfile'     :'TEMP.log',            # this is actually in the default dict but wanted to show it again
                    'verbose'     :yes                    # this is actually in the default dict but wanted to show it again
                              }
                # params from the Parameter file adjustable by the user
                clSoftcodedParams={
                    'fl_vardq'      :rc['fl_vardq'],
                    'fl_dqprop'     :pyrafBoolean(rc['fl_dqprop']),
                    'combine'       :rc['method'],
                    'reject'        :'none'
                                    }
                # grabbing the default params dict and updating it with the two above dicts
                clParamsDict=CLDefaultParamsDict('gemcombine')
                clParamsDict.update(clPrimParams)
                clParamsDict.update(clSoftcodedParams)
                 
                log.debug('calling the gemcombine CL script for input list '+clm.inputList())
                
                gemini.gemcombine(**clParamsDict)
                
                if gemini.gemcombine.status:
                    log.critical('gemcombine failed for inputs '+rc.inputsAsStr())
                    raise GEMINIException('gemcombine failed')
                else:
                    log.status('exited the gemcombine CL script successfully')
                    
                # renaming CL outputs and loading them back into memory 
                # and cleaning up the intermediate tmp files written to disk
                clm.finishCL(combine=True) 
                os.remove(clPrimParams['logfile'])
                #clm.rmStackFiles() #$$$$$$$$$ DON'T do this if 
                #^ intermediate outputs are wanted!!!!
                
                ad = rc.getOutputs(style='AD')[0] # there is only one at this point so no need to perform a loop
                
                # adding a GEM-TLM and GBIAS time stamps to the PHU
                ut = ad.historyMark()
                ad.historyMark(key='GBIAS',stomp=False)
                # updating logger with updated/added time stamps
                log.fullinfo('****************************************************','header')
                log.fullinfo('file = '+ad.filename,'header')
                log.fullinfo('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~','header')
                log.fullinfo('PHU keywords updated/added:\n', 'header')
                log.fullinfo('GEM-TLM = '+str(ut),'header' )
                log.fullinfo('GBIAS = '+str(ut),'header' )
                log.fullinfo('---------------------------------------------------','header')    
                
                log.status('*FINISHED* combining the images of the input data')
        except:
            log.critical('Problem combining the images for inputs: '+rc.inputsAsStr())
            raise 
        yield rc   

    def measureIQ(self,rc):
        '''
        '''
        #@@FIXME: Detecting sources is done here as well. This 
        # should eventually be split up into
        # separate primitives, i.e. detectSources and measureIQ.
        try:
            log.status('*STARTING* to detect the sources and measure the IQ of the inputs')
            for ad in rc.getInputs(style='AD'):
                if not os.path.dirname(ad.filename)=='':
                    log.critical('The inputs to measureIQ must be in the pwd for it to work correctly')
                    raise GEMINIException('inputs to measureIQ were not in pwd')
                print ad.info()    
               # if 'GEMINI_NORTH' in inp.ad.getTypes():
               #     observ = 'gemini-north'
               # elif 'GEMINI_SOUTH' in inp.ad.getTypes():
               #     observ = 'gemini-south'
               # else:
               #     observ = 'gemini-north'
                
                st = time.time()
                from iqtool.iq import getiq
                iqdata = getiq.gemiq( ad.filename, function='moffat', display=True, mosaic=True, qa=True)
                et = time.time()
                print 'MeasureIQ time:', (et - st)
                # iqdata is list of tuples with image quality metrics
                # (ellMean, ellSig, fwhmMean, fwhmSig)
                if len(iqdata) == 0:
                    print 'WARNING: Problem Measuring IQ Statistics, none reported'
                else:
                    rc.rqIQ( ad, *iqdata[0] )
            
            log.status('*FINISHED* measuring the IQ of the inputs')
        except:
            log.critical('Problem combining the images.')
            raise 
        yield rc  
    
    def ADUtoElectrons(self,rc):
        '''
        This primitive will convert the inputs from having pixel 
        units of ADU to electrons.
        '''
        try:
            log.status('*STARTING* to convert the pixel values from ADU to electrons')
            for ad in rc.getInputs(style='AD'):
                log.fullinfo('calling ad.mult on '+ad.filename)
                
                adOut = ad.mult(ad['SCI'].gain(asDict=True))  
                  
                log.status('ad.mult completed converting the pixel units to electrons')              

                
                # updating SCI headers
                for ext in adOut['SCI']:
                    gainorig=ext.gain()
                    ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'GAINORIG', gainorig, 'Gain prior to unit conversion (e-/ADU)')
                    ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'GAIN', 1.0, 'Gain (e-/ADU)') # =1 by definition
                    ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'BUNIT','electrons' , 'Physical units')
                    
                    log.fullinfo('SCI extension number '+str(ext.header['EXTVER'])+' keywords updated/added:\n', 'header')
                    log.fullinfo('GAINORIG = '+str(gainorig),'header' )
                    log.fullinfo('GAIN = '+str(1.0),'header' )
                    log.fullinfo('BUNIT = '+'electrons','header' )
                    log.fullinfo('---------------------------------------------------','header')
                # updating VAR headers if they exist (not updating any 
                # DQ headers as no changes were made to them here)  
                if adOut.countExts('VAR')==adOut.countExts('SCI'):
                    for ext in adOut['VAR']:
                        gainorig=adOut.extGetKeyValue(('SCI',ext.extver()),'GAINORIG')
                        
                        ext.extSetKeyValue(('VAR',int(ext.header['EXTVER'])),'GAINORIG', gainorig, 'Gain prior to unit conversion (e-/ADU)')
                        ext.extSetKeyValue(('VAR',int(ext.header['EXTVER'])),'GAIN', gainorig*gainorig, 'Gain (e-/ADU)')
                        ext.extSetKeyValue(('VAR',int(ext.header['EXTVER'])),'BUNIT','electrons squared' , 'Physical units')
                        
                        log.fullinfo('VAR extension number '+str(ext.header['EXTVER'])+' keywords updated/added:\n', 'header')
                        log.fullinfo('GAINORIG = '+str(gainorig),'header' )
                        log.fullinfo('GAIN = '+str(gainorig*gainorig),'header' )
                        log.fullinfo('BUNIT = '+'electrons squared','header' )
                        log.fullinfo('---------------------------------------------------','header')
                
                # adding GEM-TLM and ADU2ELEC time stamps to PHU
                ut = adOut.historyMark()
                adOut.historyMark('ADU2ELEC',stomp=False)
                
                # updating logger with time stamps
                log.fullinfo('****************************************************','header')
                log.fullinfo('file = '+adOut.filename,'header')
                log.fullinfo('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~','header')
                log.fullinfo('PHU keywords updated/added:\n', 'header')
                log.fullinfo('GEM-TLM = '+str(ut),'header' )
                log.fullinfo('ADU2ELEC = '+str(ut),'header' )
                log.fullinfo('---------------------------------------------------','header')
                
                log.debug('calling gemt.fileNameUpdater on '+adOut.filename)
                adOut.filename=gemt.fileNameUpdater(adOut.filename,postpend=rc['outpref'], strip=False)
                log.status('gemt.fileNameUpdater updated the file name to '+adOut.filename)
                rc.reportOutput(adOut)   
                
            log.status('*FINISHED* converting the pixel units to electrons')
        except:
            log.critical('Problem converting the pixel units of one of '+rc.inputsAsStr())
            raise
        yield rc                    

    def writeOutputs(self,rc, clob = False):
        '''
        A primitive that may be called by a recipe at any stage to
        write the outputs to disk.
        If postpend is set during the call to writeOutputs, any previous 
        postpends will be striped and replaced by the one provided.
        examples: 
        writeOutputs(postpend= '_string'), writeOutputs(prepend= '_string') 
        or if you have a full file name in mind for a SINGLE file being 
        ran through Reduce you may use writeOutputs(outfilename='name.fits').
        '''
        try:
            log.status('*STARTING* to write the outputs')
            log.status('postpend = '+str(rc['postpend']))
            log.status('prepend = '+str(rc['prepend']))
            
            for ad in rc.getInputs(style='AD'):
                if rc['postpend']:
                    log.debug('calling gemt.fileNameUpdater on '+ad.filename)
                    ad.filename=gemt.fileNameUpdater(ad.filename, \
                                        postpend=rc['postpend'], strip=True)
                    log.status('gemt.fileNameUpdater updated the file name to '+ad.filename)
                    outfilename=os.path.basename(ad.filename)
                elif rc['prepend']:
                    infilename=os.path.basename(ad.filename)
                    outfilename=rc['prepend']+infilename
                elif rc['outfilename']:
                    outfilename=rc['outfilename']   
                else:
                    outfilename=os.path.basename(ad.filename) 
                    log.status('not changing the file name to be written'+\
                    ' from its current name') 
                log.status('writing to file = '+outfilename)      
                ad.write(filename=outfilename,clobber=clob)     
                #^ AstroData checks if the output exists and raises an exception
                #rc.reportOutput(ad)
            
            # clearing the value of 'postpend' and 'prepend' in the RC so 
            # they don't persist to the next writeOutputs call and screw it up
            rc['postpend']=None
            rc['prepend']=None
            log.status('*FINISHED* writing the outputs')   
        except:
            log.critical('Problem writing one of '+rc.inputsAsStr())
            raise 
        yield rc   
         
def CLDefaultParamsDict(CLscript):
    '''
    A function to return a dictionary full of all the default parameters 
    for each CL script used so far in the Recipe System.
    '''
    # loading and bringing the pyraf related modules into the name-space
    pyraf,gemini,yes,no = pyrafLoader()
    
    if CLscript=='gemcombine':
        defaultParams={
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
            'Stdout'     :IrafStdout(),
            'Stderr'     :IrafStdout()
                       }
        return defaultParams                                  
#$$$$$$$$$$$$$$$$$$$$$$$ END OF KYLES NEW STUFF $$$$$$$$$$$$$$$$$$$$$$$$$$
    
    
