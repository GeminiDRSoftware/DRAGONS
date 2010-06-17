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
            writeInt = rc['writeInt']
            
             ## inserting the input file's headers to log for debugging (makes log HUGE)    
            #for ad in rc.getInputs(style="AD"):
                #log.debug('******** input headers to prepare **************','debug')
                #log.debug('########## Headers for file: '+ad.filename+' ########','debug')
                #for ext in range(len(ad)+1):    
                        #log.debug(ad.getHeaders()[ext-1],'debug') #this will loop to print the PHU and then each of the following pixel extensions
                        #log.debug('--------------------------------------------------------------------------','debug')
              
            log.status('STARTING to validate the input data','status')
            log.debug('calling validateInstrumentData', 'status')
            rc.run("validateInstrumentData")
            
            # updating the filenames in the RC
            for ad in rc.getInputs(style="AD"):
                log.debug('calling fileNameUpdater','status')        
                fileNameUpdater(ad, postpend='_validated', strip=False)
                # printing the updated headers
                #for ext in range(len(ad)+1):    
                    #log.debug(ad.getHeaders()[ext-1],'debug') #this will loop to print the PHU and then each of the following pixel extensions
                rc.reportOutput(ad) 
                        
            log.status('FINISHED validating input data','status')
            
            if writeInt:    
                # writing outputs of this primitive for debugging
                log.status('writing the outputs of validateData to disk','status')
                rc.run('writeOutputs')
                log.status('writing complete','status')
            # inserting the primitive output file's headers to log for debugging (makes log HUGE)
            #for ad in rc.getInputs(style="AD"):
                #log.debug('******** output headers of validateData **************','debug')
                #log.debug('######### Headers for file: '+ad.filename+' #########','debug')
                #for ext in range(len(ad)+1):    
                        #log.debug(ad.getHeaders()[ext-1],'debug') #this will loop to print the PHU and then each of the following pixel extensions
                        #log.debug('---------------------------------------------------------------------------','debug')
                
        except:
            log.critical("Problem preparing the image.",'critical')
            raise 
        
        yield rc

#----------------------------------------------------------------------
    def standardizeStructure(self,rc):
        try:
            writeInt = rc['writeInt']
            
            # setting the input 'repair' to a boolean from its current string type
            addMDF = rc["addMDF"]
            if addMDF == None:
                addMDF = True
            else:
                addMDF = ((rc["addMDF"]).lower() == "true")
            
            # add the MDF if not set to false
            if addMDF:
                log.debug('calling attachMDF','status')
                rc.run("attachMDF")
             
            log.status('STARTING to standardize the structure of input data','status')
            
                
            for ad in rc.getInputs(style="AD"):
                log.debug('calling stdObsStruct', 'status')
                stdObsStruct(ad)
                # updating the filenames in the RC
                log.debug('calling fileNameUpdater','status')
                fileNameUpdater(ad, postpend='_struct', strip=False)
                rc.reportOutput(ad)
            
            log.status('FINISHED standardizing the structure of input data','status')
                
            if writeInt:
                log.status('writing the outputs of standardizeStructure to disk','status')
                rc.run('writeOutputs')
                log.status('writing complete','status')
            # inserting the primitive output file's headers to log for debugging (makes log HUGE)
            #for ad in rc.getInputs(style="AD"):
                #log.debug('******** output headers of standardizeStructure **************','debug')
                #log.debug('########## Headers for file: '+ad.filename+' ########','debug')
                #for ext in range(len(ad)+1):    
                        #log.debug(ad.getHeaders()[ext-1],'debug') #this will loop to print the PHU and then each of the following pixel extensions
                        #log.debug('---------------------------------------------------------------------------','debug')
                            
        except:
            log.critical("Problem preparing the image.",'critical')
            raise
                     
        yield rc
        

#-------------------------------------------------------------------
    def standardizeHeaders(self,rc):
        try:   
            writeInt = rc['writeInt']
            
            log.status('STARTING to standardize the headers','status')
            log.status('standardizing observatory general headers','status')            
            for ad in rc.getInputs(style="AD"):
                log.debug('calling stdObsHdrs','status')
                stdObsHdrs(ad)
                 
            #log.debug("printing the updated headers",'debug')
            #for ext in range(len(ad)+1):
                #log.debug('--------------------------------------------------------------','debug')    
                #log.debug(ad.getHeaders()[ext-1],'debug') #this will loop to print the PHU and then each of the following pixel extensions
                  
            log.status("observatory headers fixed",'status')
            log.debug('calling standardizeInstrumentHeaders','status')
            log.status('standardizing instrument specific headers','status')
            rc.run("standardizeInstrumentHeaders") 
            log.status("instrument specific headers fixed",'status')
            
            # updating the filenames in the RC #$$$$$$$$$$ this is temperarily commented out, uncomment when below brick is put into validateWCS
            # for ad in rc.getInputs(style="AD"):
            #     fileNameUpdater(ad,postpend='_Hdrs', strip=False)
            # rc.reportOutput(ad)
                
            # updating the filenames in the RC $$$$ TEMPERARILY HERE TILL validateWCS IS WRITEN AND THIS WILL THEN GO THERE
            for ad in rc.getInputs(style="AD"):
                log.debug('calling fileNameUpdater','status')
                fileNameUpdater(ad, postpend='_prepared', strip=True)
                rc.reportOutput(ad)
                
            log.status('FINISHED standardizing the headers','status')
              
            # writing output file of prepare
            log.status('writing the outputs of prepare to disk','status')
            rc.run('writeOutputs')
            log.status('writing complete','status')
            # inserting the primitive output file's headers to log for debugging (makes log HUGE)
            #for ad in rc.getInputs(style="AD"):
                #log.debug('******** output headers of standardizeStructure **************','debug')
                #log.debug('######### Headers for file: '+ad.filename+' ########','debug')
                #for ext in range(len(ad)+1):    
                        #log.debug(ad.getHeaders()[ext-1],'debug') #this will loop to print the PHU and then each of the following pixel extensions
                        #log.debug('---------------------------------------------------------------------------','debug')
                
        except:
            log.critical("Problem preparing the image.",'critical',)
            raise 
        
        yield rc 
        
#--------------------------------------------------------------------------

    def writeOutputs(self,rc):
        try:
            log.status('postpend = '+str(rc["postpend"]),'status')
            for ad in rc.getInputs(style="AD"):
                if rc["postpend"]:
                    log.debug('calling fileNameUpdater','status')
                    fileNameUpdater(ad, rc["postpend"], strip=False)
                    outfilename=os.path.basename(ad.filename)
                elif rc["outfilename"]:
                    outfilename=rc["outfilename"]   
                else:
                    outfilename=os.path.basename(ad.filename) 
                    log.status('not changing the file name to be written from its current name','status') 
                    log.status('writing to file = '+outfilename,'status')      
                ad.write(fname=outfilename)     #AstroData checks if the output exists and raises and exception
                rc.reportOutput(ad)
                
        except:
            log.critical("Problem writing the image.",'critical')
            raise 
        
        yield rc 
             
                
#$$$$$$$$$$$$$$$$$$$$$$$ END OF KYLES NEW STUFF $$$$$$$$$$$$$$$$$$$$$$$$$$


    
