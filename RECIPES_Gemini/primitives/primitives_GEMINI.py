from time import sleep
from ReductionObjects import ReductionObject
from utils import filesystem
import IDFactory

from pyraf.iraf import tables, stsdas, images
from pyraf.iraf import gemini
import pyraf

gemini()


stepduration = 1.
class GEMINIPrimitives(ReductionObject):
    
    # primitives
    def init(self, co):
        ReductionObject.init(self, co)
        return co
    
    def logFilename (self, co):
        print "logFilename"
        for i in range(0,5):
            print "\tlogFilename",i
            sleep(stepduration)
            yield co
    
    def displayStructure(self, co):
        print "displayStructure"
        for i in range(0,5):
            print "\tds ",i
            sleep(stepduration)
            yield co
        
    def summarize(self, co):
        print "done with task"
        for i in range(0,5):
            sleep(stepduration)
            yield co        
    
    def gem_produce_im_flat(self, co):
        print "gem_produce_imflat step called"
        co.update({"flat" :co.calibrations[(co.inputs[0], "flat")]})
        yield co
    
    def gem_produce_bias(self, co):
        print "gem_produce_bias step called"
        co.update({"bias" :co.calibrations[(co.inputs[0], "bias")]})
        yield co    

    def getProcessedBias(self, co):
        try:
            print "getting bias"
            co.rqCal( "bias" )
        except:
            print "problem getting bias"
            raise
        yield co
    
    def getProcessedFlat(self, co):
        try:
            print "getting flat"
            #co.rqCal( "flat" )
        except:
            print "problem getting flat"
            raise
        yield co    
        
        
    def setStackable(self, co):
        try:
            print "updating stackable with input"
            co.rqStackUpdate()
        except:
            print "problem stacking input"
            raise
        yield co
        
    def getStackable(self, co):
        try:
            print "getting stack"
            co.rqStackGet()
        except:
            print "problem getting stack"
            raise
        yield co
        
    def printStackable(self, co):
        ID = IDFactory.generateStackableID(co.inputs, "1_0")
        ls = co.getStack(ID)
        print "STACKABLE"
        print "ID:", ID
        for item in ls.filelist:
            print "\t", item
        yield co
        
    def printParameters(self, co):
        print "printing parameters"
        list_of_keys = co.keys()
        list_of_keys.sort()
        print "-"*40
        for param in list_of_keys:
            print co[param]
            print "-"*40
        yield co
                
    def display(self, co):
        try:
            print "displaying output"
            co.rqDisplay()           
        except:
            print "problem displaying output"
            raise
        yield co
    
    def mosaic(self, co):
       try:
          print "producing image mosaic"
          mstr = 'flatdiv_'+co.inputsAsStr()          
          gemini.gmosaic( mstr  )
       except:
          print "Problem producing image mosaic"         
          raise
       yield co
       
    def averageCombine(self, co):
        try:
            # @@TODO: need to include parameter options here
            print "Combining and averaging" 
            filesystem.deleteFile('inlist')
            #print 'now check inlist dude:', co.makeInlistFile()           
            gemini.gemcombine( co.makeInlistFile(),  "tstgemcombine1.fits",\
               combine="median", reject="none" )
            
        except:
            print "Problem combining and averaging"
            raise 
        yield co
        
        
        
