from time import sleep
import time
from astrodata.ReductionObjects import PrimitiveSet
from astrodata.adutils import filesystem
from astrodata.adutils import gemLog
from astrodata import IDFactory
import os,sys, re
from sets import Set
from gempy.instruments.gemini import *
import numpy as np
import pyfits
import pyraf
from datetime import datetime
import shutil
log=gemLog.getGeminiLog()
yes = pyraf.iraf.yes
no = pyraf.iraf.no
if True:

    from pyraf.iraf import tables, stsdas, images
    from pyraf.iraf import gemini
    import pyraf

    gemini()

stepduration = 1.

class GEMINIException:
    """ This is the general exception the classes and functions in the
    Structures.py module raise.
    """
    def __init__(self, msg="Exception Raised in Recipe System"):
        """This constructor takes a message to print to the user."""
        self.message = msg
    def __str__(self):
        """This str conversion member returns the message given by the user (or the default message)
        when the exception is not caught."""
        return self.message

class GEMINIPrimitives(PrimitiveSet):
    astrotype = "GEMINI"
    
    def init(self, rc):
        return 
    init.pt_hide = True
    
    def pause(self, rc):
        rc.requestPause()
        yield rc
 
    def exit(self, rc):
        print "calling sys.exit()"
        sys.exit()
    exit.pt_usage = "Used to exit the recipe."
    ptusage_exit = "Must exit recipe."    
 #------------------------------------------------------------------------------ 
    def crashReduce(self, rc):
        raise 'Crashing'
        yield rc
       
 #------------------------------------------------------------------------------ 
    def clearCalCache(self, rc):
        print "pG61:", rc.calindfile
        rc.persistCalIndex(rc.calindfile, newindex = {})
        scals = rc["storedcals"]
        if scals:
            if os.path.exists(scals):
                shutil.rmtree(scals)
            cachedict = rc["cachedict"]
            for cachename in cachedict:
                cachedir = cachedict[cachename]
                if not os.path.exists(cachedir):                        
                    os.mkdir(cachedir)                
        yield rc
        
    def display(self, rc):
        try:
            rc.rqDisplay(displayID=rc["displayID"])           
        except:
            print "Problem displaying output"
            raise 
        yield rc
        
#------------------------------------------------------------------------------ 
    def displayStructure(self, rc):
        print "displayStructure"
        for i in range(0,5):
            print "\tds ",i
            sleep(stepduration)
            yield rc
            
#------------------------------------------------------------------------------ 
    def gem_produce_bias(self, rc):
        print "gem_produce_bias step called"
        # rc.update({"bias" :rc.calibrations[(rc.inputs[0], "bias")]})
        yield rc   
        
#------------------------------------------------------------------------------ 
    def gem_produce_im_flat(self, rc):
        print "gem_produce_imflat step called"
        # rc.update({"flat" :rc.calibrations[(rc.inputs[0], "flat")]})
        yield rc

#------------------------------------------------------------------------------ 
    def getProcessedBias(self, rc):
        try:
            print "getting bias"
            rc.rqCal( "bias" )
        except:
            print "Problem getting bias"
            raise 
        yield rc
        if rc.calFilename("bias") == None:
            print "${RED}can't find bias for inputs\ngetProcessedBias fail is fatal${NORMAL}"
            print "${RED}${REVERSE}STOPPING RECIPE${NORMAL}"
            rc.finish()
        yield rc
#------------------------------------------------------------------------------                 
    def getProcessedFlat(self, rc):
        try:
            print "getting flat"
            rc.rqCal( "twilight" )
        except:
            print "Problem getting flat"
            raise 
        
        yield rc 
        
        if rc.calFilename("bias") == None:
            print "${RED}can't find bias for inputs${NORMAL}"
            rc.finish()
        yield rc
        
#------------------------------------------------------------------------------ 
    def getStackable(self, rc):
        try:
            print "getting stack"
            rc.rqStackGet()
            yield rc
            # @@REFERENCE IMAGE @@NOTE: to pick which stackable list to get
            stackid = IDFactory.generateStackableID( rc.inputs[0].ad )
            stack = rc.getStack(stackid).filelist
            print 'prim_g126: ',repr(stack)
            rc.reportOutput(stack)
        except:
            print "Problem getting stack"
            raise 

        yield rc      
                
#------------------------------------------------------------------------------ 
    def logFilename (self, rc):
        print "logFilename"
        for i in range(0,5):
            print "\tlogFilename",i
            sleep(stepduration)
            yield rc

#------------------------------------------------------------------------------ 
    def measureIQ(self, rc):
        try:
            #@@FIXME: Detecting sources is done here as well. This should eventually be split up into
            # separate primitives, i.e. detectSources and measureIQ.
            print "measuring iq"
            '''
            image, outFile='default', function='both', verbose=True,\
            residuals=False, display=True, \
            interactive=False, rawpath='.', prefix='auto', \
            observatory='gemini-north', clip=True, \
            sigma=2.3, pymark=True, niters=4, boxSize=2., debug=False):
            '''
            for inp in rc.inputs:
                if 'GEMINI_NORTH' in inp.ad.getTypes():
                    observ = 'gemini-north'
                elif 'GEMINI_SOUTH' in inp.ad.getTypes():
                    observ = 'gemini-south'
                else:
                    observ = 'gemini-north'
                st = time.time()
                from iqtool.iq import getiq
                iqdata = getiq.gemiq( inp.filename, function='moffat', display=False, mosaic=True, qa=True)
                et = time.time()
                print 'MeasureIQ time:', (et - st)
                # iqdata is list of tuples with image quality metrics
                # (ellMean, ellSig, fwhmMean, fwhmSig)
                if len(iqdata) == 0:
                    print "WARNING: Problem Measuring IQ Statistics, none reported"
                else:
                    rc.rqIQ( inp.ad, *iqdata[0] )
            
        except:
            print 'Problem measuring IQ'
            raise 
        
        yield rc

#------------------------------------------------------------------------------ 
    def printParameters(self, rc):
        print "printing parameters"
        print rc.paramsummary()
        yield rc              
        
#------------------------------------------------------------------------------ 
    def printStackable(self, rc):
        ID = IDFactory.generateStackableID(rc.inputs, "1_0")
        ls = rc.getStack(ID)
        print "STACKABLE"
        print "ID:", ID
        if ls is None:
            print "No Stackable list created for this input."
        else:
            for item in ls.filelist:
                print "\t", item
        yield rc

#------------------------------------------------------------------------------ 
    def setContext(self, rc):
        rc.update(rc.localparms)
        yield rc
#------------------------------------------------------------------------------ 
    def showParams(self, rc):
        rcparams = rc.paramNames()
        if (rc["show"]):
            toshows = rc["show"].split(":")
            for toshow in toshows:
                if toshow in rcparams:
                    print toshow+" = "+repr(rc[toshow])
                else:
                    print toshow+" is not set"
        else:
            for param in rcparams:
                print param+" = "+repr(rc[param])
        
        # print "all",repr(rc.parmDictByTag("showParams", "all"))
        # print "iraf",repr(rc.parmDictByTag("showParams", "iraf"))
        # print "test",repr(rc.parmDictByTag("showParams", "test"))
        # print "sdf",repr(rc.parmDictByTag("showParams", "sdf"))

        # print repr(dir(rc.ro.primDict[rc.ro.curPrimType][0]))
        yield rc
            
            
            
                      
#------------------------------------------------------------------------------ 
    def dsetStackable(self, rc):
        try:
            print "updating stackable with input"
            rc.rqStackUpdate()
        except:
            print "Problem stacking input"
            raise

        yield rc

    def showInputs(self, rc):
        print "Inputs:"
        for inf in rc.inputs:
            print "  ", inf.filename   
        yield rc  
    showFiles = showInputs
    
    def showCals(self, rc):
        if str(rc["showcals"]).lower() == "all":
            num = 0
            # print "pG256: showcals=all", repr (rc.calibrations)
            for calkey in rc.calibrations:
                num += 1
                print rc.calibrations[calkey]
            if (num == 0):
                print "There are no calibrations in the cache."
        else:
            for adr in rc.inputs:
                sid = IDFactory.generateAstroDataID(adr.ad)
                num = 0
                for calkey in rc.calibrations:
                    if sid in calkey :
                        num += 1
                        print rc.calibrations[calkey]
            if (num == 0):
                print "There are no calibrations in the cache."
        yield rc
    ptusage_showCals = "Used to show calibrations currently in cache for inputs."
#------------------------------------------------------------------------------ 
    def showStackable(self, rc):
        sidset = set()
        for inp in rc.inputs:
            sidset.add( IDFactory.generateStackableID( inp.ad ))
        for sid in sidset:
            stacklist = rc.getStack(sid).filelist
            
            print "Stack for stack id=%s" % sid
            for f in stacklist:
                print "   "+os.path.basename(f)
        
        yield rc
                 
        
            
#------------------------------------------------------------------------------ 
    def summarize(self, rc):
        print "done with task"
        for i in range(0,5):
            sleep(stepduration)
            yield rc  
#------------------------------------------------------------------------------ 
    def time(self, rc):
        cur = datetime.now()
        
        elap = ""
        if rc["lastTime"] and not rc["start"]:
            td = cur - rc["lastTime"]
            elap = " (%s)" %str(td)
        print "Time:", str(datetime.now()), elap
        
        rc.update({"lastTime":cur})
        yield rc
        
  
#-------------------------------------------------------------------
#$$$$$$$$$$$$$$$$$$$$$$$$ NEW VERSIONS OF ABOVE PRIMS BY KYLE $$$$$$$$$$$$$

    def getStackable(self, rc):
        try:
            log.fullinfo("getting stack",'Clprep')
            rc.rqStackGet()
            yield rc
            # @@REFERENCE IMAGE @@NOTE: to pick which stackable list to get
            stackid = IDFactory.generateStackableID( rc.inputs[0].ad )
            stack = rc.getStack(stackid).filelist
            #print 'prim_G295: ',repr(stack)
            rc.reportOutput(stack)
        except:
            log.critical("Problem getting stack", 'critical')
            raise 

        yield rc      
 #---------------------------------------------------------------------------     
 
    def setStackable(self, rc):
        try:
            log.fullinfo("updating stackable with input", 'CLprep')
            rc.rqStackUpdate()
            # writing the files in the stack to disk if not all ready there
            for ad in rc.getInputs(style="AD"):
                if not os.path.exists(ad.filename):
                    log.fullinfo('temporarily writing '+ad.filename+' to disk', 'CLprep')
                    ad.write(ad.filename)
        except:
            log.critical("Problem stacking input",'critical')
            raise

        yield rc

#$$$$$$$$$$$$$$$$$$$$ NEW STUFF BY KYLE FOR: PREPARE $$$$$$$$$$$$$$$$$$$$$
    '''
    These primitives are now functioning and can be used, BUT are not set up to run with the current demo system.
    commenting has been added to hopefully assist those reading the code.
    Excluding validateWCS, all the primitives for 'prepare' are complete (as far as we know of at the moment that is)
    and so I am moving onto working on the primitives following 'prepare'.
    '''

#----------------------------------------------------------------------------    
    def validateData(self,rc):
        '''
        This primitive will ensure the data is not corrupted or in an odd format that will effect later steps
        in the reduction process.  It will call a function to take care of the general Gemini issues and then 
        one for the instrument specific ones. If there are issues with the data, the flag 'repair' can be used to 
        turn the feature to repair it or not (eg. validateData(repair=True)).
        '''
        
        try:
            if rc["repair"]==True:
               #this should repair the file if it is broken, but this function isn't coded yet and would require
               #some sort of flag set while checking the data to tell this to perform the corrections
               pass
           
            writeInt = rc['writeInt'] #current way we are passing a boolean around to cue the writing of intermediate files, later this will be done in Reduce
            
            log.status('*STARTING* to validate the input data','status')
            log.debug('calling validateInstrumentData', 'status')
            rc.run("validateInstrumentData")
            
            # updating the filenames in the RC
            for ad in rc.getInputs(style="AD"):
                log.debug('calling fileNameUpdater','status')        
                ad.filename=fileNameUpdater(ad.filename, postpend='_validated', strip=False)
                rc.reportOutput(ad) 
                        
            log.status('*FINISHED* validating input data','status')
            
            if writeInt:    
                # writing outputs of this primitive for debugging
                log.status('writing the outputs of validateData to disk','status')
                rc.run('writeOutputs')
                log.status('writing complete','status')
                
        except:
            log.critical("Problem preparing the image.",'critical')
            raise GEMINIException
        
        yield rc

#----------------------------------------------------------------------
    def standardizeStructure(self,rc):
        '''
        This primitive ensures the MEF structure is ready for further processing, through 
        adding the MDF if necessary and the needed keywords to the headers.  First the 
        MEF's will be checked for the general Gemini structure requirements and then the 
        instrument specific ones. If the data requires a MDF to be attached, use the 
        'addMDF' flag to make this happen (eg. standardizeStructure(addMDF=False)).
        '''
        
        try:
            writeInt = rc['writeInt']
            
            # add the MDF if not set to false
            if rc["addMDF"]==True:
                log.debug('calling attachMDF','status')
                rc.run("attachMDF")
             
            log.status('*STARTING* to standardize the structure of input data','status')
            
            for ad in rc.getInputs(style="AD"):
                log.debug('calling stdObsStruct', 'status')
                stdObsStruct(ad)
                # updating the filenames in the RC
                log.debug('calling fileNameUpdater','status')
                ad.filename=fileNameUpdater(ad.filename, postpend=rc['outsuffix'], strip=False)
                rc.reportOutput(ad)
            
            log.status('*FINISHED* standardizing the structure of input data','status')
                
            if writeInt:
                log.status('writing the outputs of standardizeStructure to disk','status')
                rc.run('writeOutputs')
                log.status('writing complete','status')
 
        except:
            log.critical("Problem preparing the image.",'critical')
            raise GEMINIException
                     
        yield rc
        

#-------------------------------------------------------------------
    def standardizeHeaders(self,rc):
        '''
        This primitive updates and adds the important header keywords for the input MEFs. 
        First the general headers for Gemini will be update/created, followed by those
        that are instrument specific.
        '''
        
        try:   
            writeInt = rc['writeInt']
            
            log.status('*STARTING* to standardize the headers','status')
            log.status('standardizing observatory general headers','status')            
            for ad in rc.getInputs(style="AD"):
                log.debug('calling stdObsHdrs','status')
                stdObsHdrs(ad)
   
            log.status("observatory headers fixed",'status')
            log.debug('calling standardizeInstrumentHeaders','status')
            log.status('standardizing instrument specific headers','status')
            rc.run("standardizeInstrumentHeaders") 
            log.status("instrument specific headers fixed",'status')
            
            # updating the filenames in the RC #$$$$$$$$$$ this is temperarily commented out, uncomment when below brick is put into validateWCS
            # for ad in rc.getInputs(style="AD"):
            #     ad.filename=fileNameUpdater(ad.filename,postpend='_Hdrs', strip=False)
            # rc.reportOutput(ad)
                
            # updating the filenames in the RC $$$$ TEMPERARILY HERE TILL validateWCS IS WRITTEN AND THIS WILL THEN GO THERE as it will be the final prim of prepare
            for ad in rc.getInputs(style="AD"):
                log.debug('calling fileNameUpdater','status')
                ad.filename=fileNameUpdater(ad.filename, postpend=rc['outsuffix'], strip=True)
                rc.reportOutput(ad)
            log.status('*FINISHED* standardizing the headers','status')
              
            # writing output file of prepare make this require the 'writeint' flag or something when validateWCS is written
            log.status('writing the outputs of prepare to disk','status')
            rc.run('writeOutputs')
            log.status('writing complete','status')
                
        except:
            log.critical("Problem preparing the image.",'critical',)
            raise GEMINIException
        
        yield rc 
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ Prepare primitives end here $$$$$$$$$$$$$$$$$$$$$$$$$$$$
                
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ primitives following Prepare below $$$$$$$$$$$$$$$$$$$$ 
    def calculateVAR(self,rc):
        '''
        This primitive uses numpy to calculate the variance of each SCI frame in the input files and 
        appends it as a VAR frame using AstroData.
        '''
        try:
            log.fullinfo('*STARTING* to add the VAR frame(s) to the input data', 'fullinfo')
            log.critical('CURRENTLY VARIENCE IS NOT BEING CALCULATED, JUST ADDING A ZEROS ARRAY!!!!', 'critical')
            
            for ad in rc.getInputs(style='AD'):
                for sciExt in ad['SCI']:
                    varArray=np.zeros(sciExt.data.shape,dtype=np.float32)
                
                    varheader = pyfits.Header()
                    varheader.update('NAXIS', 2)
                    varheader.update('PCOUNT', 0, 'required keyword; must = 0 ')
                    varheader.update('GCOUNT', 1, 'required keyword; must = 1')
                    # varHDU.renameExt("VAR", sciExt.extver())
                    varheader.update('EXTNAME', 'VAR', 'Extension Name')
                    varheader.update('EXTVER', sciExt.extver(), 'Extension Version')
                    varheader.update('BITPIX', 32, 'number of bits per data pixel')
                    
                    varAD = AstroData( header = varheader, data = varArray )
                
                    log.fullinfo('varHDU created and data added, now updating the header keys','status')
                    log.fullinfo('appending new HDU onto the file','status')
                    ad.append(varAD)
                    log.fullinfo('appending complete','status')
                    
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
                ad.historyMark(key="ADDVARDQ",stomp=False)    
                log.fullinfo('****************************************************','header')
                log.fullinfo('file = '+ad.filename,'header')
                log.fullinfo('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~','header')
                log.fullinfo('PHU keywords updated/added:\n', 'header')
                log.fullinfo('GEM-TLM = '+str(ut),'header' )
                log.fullinfo('ADDVARDQ = '+str(ut),'header' )
                log.fullinfo('---------------------------------------------------','header')
                
                print ad.info()
                
                ## check if there filename all ready has the suffix '_vardq', if not add it
                if not re.search(rc['outsuffix'],ad.filename): #%%%% this is printing a 'None' on the screen, fix that!!!
                    log.debug('calling fileNameUpdater','status')
                    ad.filename=fileNameUpdater(ad.filename, postpend=rc['outsuffix'], strip=False)
                rc.reportOutput(ad)        
                
            log.fullinfo('*FINISHED* adding the VAR frame(s) to the input data', 'fullinfo')
     
        except:
            log.critical("Problem adding the VARDQ to the image.",'critical',)
            raise GEMINIException
        
        yield rc 
      
#--------------------------------------------------------------------------

    def calculateDQ(self,rc):
        '''
        This primitive will create a numpy array for the data quality of each SCI frame of the input data.
        This will then have a header created and be append to the input using AstroData as a DQ frame.
        The value of a pixel will be the sum of the following: 
        (0=good, 1=bad pixel (found in bad pixel mask), 2= value is in non linear region, 4=pixel is saturated)
        '''
        try:
            log.status('*STARTING* to add the DQ frame(s) to the input data', 'status')
            #log.critical('CURRENTLY NO BPM FILE LOADING, JUST ADDING A ZEROS ARRAY!!!!', 'critical')
            
            packagePath=sys.argv[0].split('gemini_python')[0]
            calPath='gemini_python/test_data/test_cal_files/GMOS_BPM_files/'
            
            #$$$$$$$$$$$$$ this block is GMOS IMAGE specific, consider moving or something $$$$$$$$$$$$
            BPM_11=AstroData(packagePath+calPath+'GMOS_BPM_11.fits')
            BPM_22=AstroData(packagePath+calPath+'GMOS_BPM_22.fits')
            #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            
            for ad in rc.getInputs(style='AD'):
                
                for sciExt in ad['SCI']:
                    
                    #$$$$$$$$$$$$$ this block is GMOS IMAGE specific, consider moving or something $$$$$$$$$$$$
                    if sciExt.getKeyValue('CCDSUM')=='1 1':
                        BPMArray=BPM_11['DQ'][sciExt.extver()-1].data
                        BPMfilename = 'GMOS_BPM_11.fits'
                    elif sciExt.getKeyValue('CCDSUM')=='2 2':
                        BPMArray=BPM_22['DQ'][sciExt.extver()-1].data
                        BPMfilename = 'GMOS_BPM_22.fits'
                    else:
                        BPMArray=np.zeros(sciExt.data.shape,dtype=np.int16)
                        log.error('CCDSUM is not 1x1 or 2x2, using zeros array for BPM', 'error')
                        BPMfilename='None'
                    BPMArray=np.where(BPMArray>=1,1,0)
                    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$    
                    
                    datasecStr=sciExt.data_section()
                    datasecList=secStrToIntList(datasecStr) 
                    dsl=datasecList
                    
                    nonLinArray=np.zeros(sciExt.data.shape,dtype=np.int16)
                    saturatedArray=np.zeros(sciExt.data.shape,dtype=np.int16)
                    linear=sciExt.non_linear_level()
                    saturated=sciExt.saturation_level()

                    if (linear!=None) and (rc['fl_nonlinear']==True): 
                        log.fullinfo('performing a np.where to find non-linear pixels','status')
                        nonLinArray=np.where(sciExt.data>linear,2,0)
                    if (saturated!=None) and (rc['fl_saturated']==True):
                        log.fullinfo('performing a np.where to find saturated pixels','status')
                        saturatedArray=np.where(sciExt.data>saturated,4,0)
                    
                    # BPM file has had its overscan region trimmed all ready, so must trim the overscan section from the nonLin and saturated arrays to match
                    nonLinArrayTrimmed = nonLinArray[dsl[2]-1:dsl[3],dsl[0]-1:dsl[1]]
                    saturatedArrayTrimmed = saturatedArray[dsl[2]-1:dsl[3],dsl[0]-1:dsl[1]]  
                     
                    dqArray=np.add(BPMArray,nonLinArrayTrimmed,saturatedArrayTrimmed)
                    
                    dqheader = pyfits.Header()

                    dqheader.update('BITPIX', 16, 'number of bits per data pixel')
                    dqheader.update('NAXIS', 2)
                    dqheader.update('PCOUNT', 0, 'required keyword; must = 0 ')
                    dqheader.update('GCOUNT', 1, 'required keyword; must = 1')
                    dqheader.update('BUNIT', 'bit', 'Physical units')
                    dqheader.update('BPMFILE', BPMfilename, 'Name of input Bad Pixel Mask file')
                    dqheader.update('EXTNAME', 'DQ', 'Extension Name')
                    dqheader.update('EXTVER', sciExt.extver(), 'Extension Version')
                    
                    dqAD = AstroData( header = dqheader, data = dqArray )
                    
                    log.fullinfo('appending new HDU onto the file','status')
                    ad.append(dqAD)
                    log.fullinfo('appending complete','status')
                    
                    ## updating logger with updated/added keywords
                    log.fullinfo('****************************************************','header')
                    log.fullinfo('file = '+ad.filename,'header')
                    log.fullinfo('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~','header')
                    log.fullinfo('DQ extension number '+str(sciExt.extver())+' keywords updated/added:\n', 'header')
                    log.fullinfo('BITPIX= '+str(16),'header' )
                    log.fullinfo('NAXIS= '+str(2),'header' )
                    log.fullinfo('BUNIT= '+'bit','header' )
                    log.fullinfo('BPMFILE= '+BPMfilename,'header' )
                    log.fullinfo('EXTNAME= '+'DQ','header' )
                    log.fullinfo('EXTVER= '+str(sciExt.extver()),'header' )
                    log.fullinfo('---------------------------------------------------','header')
                
                ut = ad.historyMark() 
                ad.historyMark(key="ADDVARDQ",stomp=False) 
                log.fullinfo('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~','header')
                log.fullinfo('PHU keywords updated/added:\n', 'header')
                log.fullinfo('GEM-TLM = '+str(ut),'header' )
                log.fullinfo('ADDVARDQ = '+str(ut),'header' )
                log.fullinfo('---------------------------------------------------','header')
                    
                #print ad.info()
                
                ## check if there filename all ready has the suffix '_vardq', if not add it
                if not re.search(rc['outsuffix'],ad.filename): #%%%% this is printing a 'None' on the screen, fix that!!!
                    log.debug('calling fileNameUpdater','status')
                    ad.filename=fileNameUpdater(ad.filename, postpend=rc['outsuffix'], strip=False)
                    log.status('output of addDQ will have the filename: '+ad.filename,'status')
                rc.reportOutput(ad)        
            
            log.status('*FINISHED* adding the DQ frame(s) to the input data', 'status')

        except:
            log.critical("Problem adding the VARDQ to the image.",'critical',)
            raise GEMINIException
        
        yield rc 
        
  #--------------------------------------------------------------------------      
    def combine(self,rc):
        '''
        This primitive will average and combine the SCI extensions of the inputs. 
        It takes all the inputs and creates a list of them and then combines each
        of their SCI extensions together to create average combination file.
        New VAR frames are made from these combined SCI frames and the DQ frames
        are propagated through to the final file.
        
        '''

        try:
            if len(rc.getInputs())>1:
                log.status('*STARTING* combine the images of the input data', 'status')
                #preparing input files, lists, parameters... for input to the CL script
                clm=CLManager(rc)
                clm.LogCurParams()
                
                # params set by the CLManager or the definition of the prim 
                clPrimParams={
                              'input'       :clm.inputList(),
                              'output'      :clm.combineOutname(), # maybe allow the user to override this in the future. 
                              'Stdout'      :IrafStdout(), # this is actually in the default dict but wanted to show it again
                              'Stderr'      :IrafStdout(), # this is actually in the default dict but wanted to show it again
                              'logfile'     :'TEMP.log', # this is actually in the default dict but wanted to show it again
                              'verbose'     :yes # this is actually in the default dict but wanted to show it again
                              }
                
                # params from the Parameter file adjustable by the user
                clSoftcodedParams={
                                    'fl_vardq'      :rc["fl_vardq"],
                                    'fl_dqprop'     :pyrafBoolean(rc['fl_dqprop']),
                                    'combine'       :rc['method'],
                                    'reject'        :"none"
                                    }
                 
                # grabbing the default params dict and updating it with the two above dicts
                clParamsDict=CLDefaultParamsDict('gemcombine')
                clParamsDict.update(clPrimParams)
                clParamsDict.update(clSoftcodedParams)
                 
                log.fullinfo('calling the gemcombine CL script', 'status')
                
                gemini.gemcombine(**clParamsDict)
                
                #gemini.gemcombine(clm.inputList(),  output=clm.combineOutname(),combine=rc['method'], reject="none",\
                #                  fl_vardq=pyrafBoolean(rc['fl_vardq']), fl_dqprop=pyrafBoolean(rc['fl_dqprop']),\
                #                   Stdout = IrafStdout(), Stderr = IrafStdout(),\
                #                   logfile='temp.log',verbose=pyrafBoolean(True))
                
                if gemini.gemcombine.status:
                    log.critical('gemcombine failed','critical')
                    raise 
                else:
                    log.fullinfo('exited the gemcombine CL script successfully', 'status')
                    
                # renaming CL outputs and loading them back into memory and cleaning up the intermediate tmp files written to disk
                clm.finishCL(combine=True) 
                os.remove(clPrimParams['logfile'])
                #clm.rmStackFiles() #$$$$$$$$$ DON"T do this if intermediate outputs are wanted!!!!
                ad = rc.getOutputs(style='AD')[0] #there is only one at this point so no need to perform a loop
                
                ut = ad.historyMark()
                ad.historyMark(key='GBIAS',stomp=False)
                
                log.fullinfo('****************************************************','header')
                log.fullinfo('file = '+ad.filename,'header')
                log.fullinfo('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~','header')
                log.fullinfo('PHU keywords updated/added:\n', 'header')
                log.fullinfo('GEM-TLM = '+str(ut),'header' )
                log.fullinfo('GBIAS = '+str(ut),'header' )
                log.fullinfo('---------------------------------------------------','header')    
                
                
                log.status('*FINISHED* combining the images of the input data', 'status')
        
        except:
            log.critical("Problem combining the images.",'critical',)
            raise GEMINIException
        
        yield rc   
        
    #------------------------------------------------------------------------
    
    def ADUtoElectrons(self,rc):
        '''
        This primitive will convert the inputs from pixel units of ADU to electrons
        '''
        try:
            log.status('*STARTING* to convert the pixel values from ADU to electrons','status')
            for ad in rc.getInputs(style='AD'):
                adOut = ad.mult(ad['SCI'].gain(asDict=True))  
                ut = adOut.historyMark()
                adOut.historyMark('ADU2ELEC',stomp=False)
                adOut.filename=fileNameUpdater(ad.filename,postpend=rc["outpref"], strip=False)
                #print adOut.info()
                log.fullinfo('****************************************************','header')
                log.fullinfo('file = '+adOut.filename,'header')
                log.fullinfo('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~','header')
                log.fullinfo('PHU keywords updated/added:\n', 'header')
                log.fullinfo('GEM-TLM = '+str(ut),'header' )
                log.fullinfo('ADU2ELEC = '+str(ut),'header' )
                log.fullinfo('---------------------------------------------------','header')
                
                # updating SCI headers
                for ext in adOut["SCI"]:
                    gainorig=ext.gain()
                    ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'GAINORIG', gainorig, 'Gain prior to unit conversion (e-/ADU)')
                    ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'GAIN', 1.0, "Gain (e-/ADU)")
                    ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'BUNIT','electrons' , 'Physical units')
                    
                    log.fullinfo('SCI extension number '+str(ext.header['EXTVER'])+' keywords updated/added:\n', 'header')
                    log.fullinfo('GAINORIG = '+str(gainorig),'header' )
                    log.fullinfo('GAIN = '+str(1.0),'header' )
                    log.fullinfo('BUNIT = '+'electrons','header' )
                    log.fullinfo('---------------------------------------------------','header')
                # updating VAR headers if they exist    
                if adOut.countExts('VAR')==adOut.countExts('SCI'):
                    for ext in adOut["VAR"]:
                        gainorig=adOut.extGetKeyValue(('SCI',ext.extver()),'GAINORIG')
                        
                        ext.extSetKeyValue(('VAR',int(ext.header['EXTVER'])),'GAINORIG', gainorig, 'Gain prior to unit conversion (e-/ADU)')
                        ext.extSetKeyValue(('VAR',int(ext.header['EXTVER'])),'GAIN', gainorig*gainorig, "Gain (e-/ADU)")
                        ext.extSetKeyValue(('VAR',int(ext.header['EXTVER'])),'BUNIT','electrons squared' , 'Physical units')
                        
                        log.fullinfo('VAR extension number '+str(ext.header['EXTVER'])+' keywords updated/added:\n', 'header')
                        log.fullinfo('GAINORIG = '+str(gainorig),'header' )
                        log.fullinfo('GAIN = '+str(gainorig*gainorig),'header' )
                        log.fullinfo('BUNIT = '+'electrons squared','header' )
                        log.fullinfo('---------------------------------------------------','header')
                rc.reportOutput(adOut)   
                
            log.status('*FINISHED* converting the pixels units to electrons','status')
        except:
            log.critical("Problem converting the pixel units of the images.",'critical',)
            raise GEMINIException
        
        yield rc         
   #--------------------------------------------------------------------------                

    def writeOutputs(self,rc, clob = False):
        '''
        a primitive that may be called by a recipe at any stage for if the user would like files to be written to disk
        at specific stages of the recipe, compared to that of it writing the outputs of each primitive with the --writeInt flag of 
        Reduce.  An example call in this case would be : writeOutputs(postpend= '_string'), writeOutputs(prepend= '_string') or if you 
        have a full file name in mind for a SINGLE file being ran through Reduce you may use writeOutputs(outfilename='name.fits').
        '''
        try:
            log.status('*STARTING* to write the outputs','status')
            log.status('postpend = '+str(rc["postpend"]),'status')
            log.status('prepend = '+str(rc["prepend"]),'status')
            for ad in rc.getInputs(style="AD"):
                if rc["postpend"]:
                    log.debug('calling fileNameUpdater','status')
                    ad.filename=fileNameUpdater(ad.filename, postpend=rc["postpend"], strip=True)
                    outfilename=os.path.basename(ad.filename)
                elif rc["prepend"]:
                    infilename=os.path.basename(ad.filename)
                    outfilename=rc['prepend']+infilename
                elif rc["outfilename"]:
                    outfilename=rc["outfilename"]   
                else:
                    outfilename=os.path.basename(ad.filename) 
                    log.status('not changing the file name to be written from its current name','status') 
                log.status('writing to file = '+outfilename,'status')      
                ad.write(filename=outfilename,clobber=clob)     #AstroData checks if the output exists and raises and exception
                #rc.reportOutput(ad)
            
            # clearing the value of 'postpend' and 'prepend' in the RC so they don't persist to the next writeOutputs call and screw it up
            rc["postpend"]=None
            rc['prepend']=None
            log.status('*FINISHED* writting the outputs','status')   
        except:
            log.critical("Problem writing the image.",'critical')
            raise GEMINIException
        
        yield rc 
# TEMP prim for testing gain values of inputs #######################################################
    def gotGain(self,rc):
        for ad in rc.getInputs(style='AD'):
            print ad.info()
            for sci in ad:
                print ad.filename, ' extension ', str(sci.extname()),str(sci.extver())
                try:
                    print 'GAIN = ',str(sci.getKeyValue('GAIN'))
                except:
                    print 'no GAIN value'
                try:
                    print 'GAINORIG = ',sci.getKeyValue('GAINORIG')        
                except:
                    print 'no GAINORIG value'
        yield rc
# end of temp test prim ###################################################################   
         
def CLDefaultParamsDict(CLscript):
    '''
    A function to return a dictionary full of all the default parameters for each CL script used so far in the Recipe System.
    '''
    if CLscript=='gemcombine':
        defaultParams={
                       'input'      :'',            #Input MEF images
                       'output'     :"",            #Output MEF image
                       'title'      :'DEFAULT',     #Title for output SCI plane
                       'combine'    :"average",     #Combination operation
                       'reject'     :"avsigclip",   #Rejection algorithm
                       'offsets'    :"none",        #Input image offsets
                       'masktype'   :"none",        #Mask type
                       'maskvalue'  :0.0,           #Mask value
                       'scale'      :"none",        #Image scaling
                       'zero'       :"none",        #Image zeropoint offset
                       'weight'     :"none",        #Image weights
                       'statsec'    :"[*,*]",       #Statistics section
                       'expname'    :"EXPTIME",     #Exposure time header keyword
                       'lthreshold' :'INDEF',       #Lower threshold
                       'hthreshold' :'INDEF',       #Upper threshold
                       'nlow'       :1,             #minmax: Number of low pixels to reject
                       'nhigh'      :1,             #minmax: Number of high pixels to reject
                       'nkeep'      :1,             #Minimum to keep or maximum to reject
                       'mclip'      :yes,           #Use median in sigma clipping algorithms?
                       'lsigma'     :3.0,           #Lower sigma clipping factor
                       'hsigma'     :3.0,           #Upper sigma clipping factor
                       'key_ron'    :"RDNOISE",     #Keyword for readout noise in e-
                       'key_gain'   :"GAIN",        #Keyword for gain in electrons/ADU
                       'ron'        :0.0,           #Readout noise rms in electrons
                       'gain'       :1.0,           #Gain in e-/ADU
                       'snoise'     :"0.0",         #ccdclip: Sensitivity noise (electrons
                       'sigscale'   :0.1,           #Tolerance for sigma clipping scaling correction                                
                       'pclip'      :-0.5,          #pclip: Percentile clipping parameter
                       'grow'       :0.0,           #Radius (pixels) for neighbor rejection
                       'bpmfile'    :'',            #Name of bad pixel mask file or image.
                       'nrejfile'   :'',            #Name of rejected pixel count image.
                       'sci_ext'    :'SCI',         #Name(s) or number(s) of science extension
                       'var_ext'    :'VAR',         #Name(s) or number(s) of variance extension
                       'dq_ext'     :'DQ',          #Name(s) or number(s) of data quality extension
                       'fl_vardq'   :no,            #Make variance and data quality planes?
                       'logfile'    :'',            #Log file
                       'fl_dqprop'  :no,            #Propagate all DQ values?
                       'verbose'    :yes,           #Verbose output?
                       'status'     :0,             #Exit status (0=good)
                       'Stdout'     :IrafStdout(),
                       'Stderr'     :IrafStdout()
                       }
        return defaultParams                  
                       
                       
                                       
#$$$$$$$$$$$$$$$$$$$$$$$ END OF KYLES NEW STUFF $$$$$$$$$$$$$$$$$$$$$$$$$$
    
    
