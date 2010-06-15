from time import sleep
import time
from astrodata.ReductionObjects import PrimitiveSet
from astrodata.adutils import filesystem
from astrodata.adutils.future import gemLog
from astrodata import IDFactory
import os,sys
from sets import Set
from iqtool.iq import getiq
from gempy.instruments.gemini import *

from datetime import datetime
log=gemLog.getGeminiLog()
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
        yield rc
            
            
            
                      
#------------------------------------------------------------------------------ 
    def setStackable(self, rc):
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
        for adr in rc.inputs:
            sid = IDFactory.generateAstroDataID(adr.ad)
            for calkey in rc.calibrations:
                if sid in calkey:
                    print rc.calibrations[calkey]
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
#$$$$$$$$$$$$$$$$$$$$ NEW STUFF BY KYLE FOR: PREPARE $$$$$$$$$$$$$$$$$$$$$
    '''
    all the stuff in here is very much a work in progress and I will not be fully
    commenting it for others while developing it, sorry.
    '''

#----------------------------------------------------------------------------    
    def validateData(self,rc):
        try:
            # setting the input 'repair' to a boolean from its current string type
            repair = rc["repair"]
            if repair == None:
                repair = True
            else:
                repair = ((rc["repair"]).lower() == "true")
            
            # checking if there is a default debugLevel, if not set to 1
            if not rc['debugLevel']:
                rc['debugLevel']=1
                print 'prim_G296: using the default debugLevel=1'
            debugLevel=int(rc['debugLevel'])
            
             # inserting the input file's headers to log for debugging (makes log HUGE)    
            for ad in rc.getInputs(style="AD"):
                log.debug('******** input headers to prepare **************')
                log.debug('################### Headers for file: '+ad.filename+' ##################')
                for ext in range(len(ad)+1):    
                        log.debug(ad.getHeaders()[ext-1]) #this will loop to print the PHU and then each of the following pixel extensions
                        log.debug('---------------------------------------------------------------------------------')
                
            log.info('prim_G307: validating input data','status')
            
            rc.run("validateInstrumentData")
            
            # updating the filenames in the RC
            for ad in rc.getInputs(style="AD"):        
                fileNameUpdater(ad,debugLevel, postpend='_validated', strip=False)
                # printing the updated headers
                for ext in range(len(ad)+1):    
                    log.debug(ad.getHeaders()[ext-1]) #this will loop to print the PHU and then each of the following pixel extensions
                rc.reportOutput(ad) 
                    
            log.info('prim_G319: input data validated, off to a good start','status')
            
            if debugLevel>=5:    
                # writing outputs of this primitive for debugging
                log.info('prim_G323: writing the outputs of standardizeStructure to disk','status')
                rc.run('writeOutputs')
                log.info('prim_G325: writing complete','status')
            # inserting the primitive output file's headers to log for debugging (makes log HUGE)
            for ad in rc.getInputs(style="AD"):
                log.debug('******** output headers of validateData **************')
                log.debug(('################### Headers for file: '+ad.filename+' ##################'))
                for ext in range(len(ad)+1):    
                        log.debug(ad.getHeaders()[ext-1]) #this will loop to print the PHU and then each of the following pixel extensions
                        log.debug('---------------------------------------------------------------------------------')
                
        except:
            log.critical("Problem preparing the image.",'critical')
            raise 
        
        yield rc

#----------------------------------------------------------------------
    def standardizeStructure(self,rc):
        try:
            # setting the input 'repair' to a boolean from its current string type
            addMDF = rc["addMDF"]
            if addMDF == None:
                addMDF = True
            else:
                addMDF = ((rc["addMDF"]).lower() == "true")
                
            debugLevel=int(rc['debugLevel'])
            
            # add the MDF if not set to false
            if addMDF:
                rc.run("attachMDF")
                
            for ad in rc.getInputs(style="AD"):
                stdObsStruct(ad)
                # updating the filenames in the RC
                fileNameUpdater(ad,debugLevel, postpend='_struct', strip=False)
                rc.reportOutput(ad)
                
            for ext in range(len(ad)+1):    
                log.debug(ad.getHeaders()[ext-1]) #this will loop to print the PHU and then each of the following pixel extensions
                
             
            if debugLevel>=5:     
                # writing outputs of this primitive for debugging
                log.info('prim_G368: writing the outputs of standardizeStructure to disk','status')
                rc.run('writeOutputs')
                log.info('prim_G370: writing complete','status')
            # inserting the primitive output file's headers to log for debugging (makes log HUGE)
            for ad in rc.getInputs(style="AD"):
                log.debug('******** output headers of standardizeStructure **************')
                log.debug('################### Headers for file: '+ad.filename+' ##################')
                for ext in range(len(ad)+1):    
                        log.debug(ad.getHeaders()[ext-1]) #this will loop to print the PHU and then each of the following pixel extensions
                        log.debug('---------------------------------------------------------------------------------')
                            
        except:
            log.critical("Problem preparing the image.",'critical')
            raise
                     
        yield rc
        

#-------------------------------------------------------------------
    def standardizeHeaders(self,rc):
        try:            
            for ad in rc.getInputs(style="AD"):
                log.info('prim_G390: callgDing stdObsHdrs','status')   
                stdObsHdrs(ad)
                
            debugLevel=int(rc['debugLevel'])
                 
            log.debug("prim_G304: printing the updated headers")
            for ext in range(len(ad)+1):
                log.debug('--------------------------------------------------------------')    
                log.debug(ad.getHeaders()[ext-1]) #this will loop to print the PHU and then each of the following pixel extensions
                  
            log.info("Prim_G400: observatory headers fixed",'status')
            log.info('prim_G401: calling standardizeInstrumentHeaders','status')
            rc.run("standardizeInstrumentHeaders") 
            log.info('prim_G403: instrument headers fixed','status')

            # updating the filenames in the RC #$$$$$$$$$$ this is temperarily commented out, uncomment when below brick is put into validateWCS
            # for ad in rc.getInputs(style="AD"):
            #     fileNameUpdater(ad, debugLevel,postpend='_Hdrs', strip=False)
            # rc.reportOutput(ad)
                
            # updating the filenames in the RC $$$$ TEMPERARILY HERE TILL validateWCS IS WRITEN AND THIS WILL THEN GO THERE
            for ad in rc.getInputs(style="AD"):
                fileNameUpdater(ad,debugLevel, postpend='_prepared', strip=True)
                rc.reportOutput(ad)
              
            # writing output file of prepare
            log.info('prim_G416: writing the outputs of prepare to disk','status')
            rc.run('writeOutputs')
            log.info('prim_G418: writing complete','status')
            # inserting the primitive output file's headers to log for debugging (makes log HUGE)
            for ad in rc.getInputs(style="AD"):
                log.debug('******** output headers of standardizeStructure **************')
                log.debug(('################### Headers for file: '+ad.filename+' ##################'))
                for ext in range(len(ad)+1):    
                        log.debug(ad.getHeaders()[ext-1]) #this will loop to print the PHU and then each of the following pixel extensions
                        log.debug('---------------------------------------------------------------------------------')
                
        except:
            log.critical("Problem preparing the image.",'critical')
            raise 
        
        yield rc 
        
#--------------------------------------------------------------------------

    def writeOutputs(self,rc):
        try:
            
            log.info('prim_G438: postpend = '+str(rc["postpend"]),'status')
            for ad in rc.getInputs(style="AD"):
                if rc["postpend"]:
                    fileNameUpdater(ad,debugLevel, rc["postpend"], strip=False)
                    outfilename=os.path.basename(ad.filename)
                elif rc["outfilename"]:
                    outfilename=rc["outfilename"]   
                else:
                    outfilename=os.path.basename(ad.filename) 
                    log.info('prim_G447: not changing the file name to be written from its current name','status') 
                    log.info('prim_G448:  writing to file'+outfilename,'status')      
                ad.write(fname=outfilename)     #AstroData checks if the output exists and raises and exception
                rc.reportOutput(ad)
                
        except:
            log.critical("Problem writing the image.",'critical')
            raise 
        
        yield rc 
             
                
#$$$$$$$$$$$$$$$$$$$$$$$ END OF KYLES NEW STUFF $$$$$$$$$$$$$$$$$$$$$$$$$$


    
