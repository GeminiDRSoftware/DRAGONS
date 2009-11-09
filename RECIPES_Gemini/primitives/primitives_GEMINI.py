from time import sleep
from ReductionObjects import ReductionObject
from utils import filesystem
import IDFactory
import os

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
            co.rqCal( "twilight" )
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
        if ls is None:
            print "No Stackable list created for this input."
        else:
            for item in ls.filelist:
                print "\t", item
        yield co
        
    def printParameters(self, co):
        print "printing parameters"
        print co.paramsummary()
        yield co
                
    def display(self, co):
        try:
            print "displaying output"
            co.rqDisplay()           
        except:
            print "problem displaying output"
            raise
        yield co
    
    def mosaicChips(self, co):
       try:
          print "producing image mosaic"
          #mstr = 'flatdiv_'+co.inputsAsStr()          
          gemini.gmosaic( co.inputsAsStr(), outpref="mo_" )
          co.reportOutput(co.prependNames("mo_", currentDir = True))
       except:
          print "Problem producing image mosaic"         
          raise
       yield co
       
    def averageCombine(self, co):
        try:
            # @@TODO: need to include parameter options here
            print "Combining and averaging" 
            filesystem.deleteFile('inlist')
            
            templist = []
            
            for inp in co.inputs:
                 templist.append( IDFactory.generateStackableID( inp.ad ) )
                 
            templist = list( set(templist) ) # Removes duplicates.
            
            for stackID in templist:
                #@@FIXME: There are a lot of big issues in here. First, we need to backup the
                # previous average combined file, not delete. Backup is needed if something goes wrong.
                # Second, the pathnames in here have too many assumptions. (i.e.) It is assumed all the
                # stackable images are in the same spot which may not be the case.
                
                stacklist = co.getStack( stackID ).filelist
                print "STACKLIST:", stacklist
                
                if len( stacklist ) > 1:
                    stackname = "avgcomb_" + os.path.basename(stacklist[0])
                    filesystem.deleteFile( stackname )
                    gemini.gemcombine( co.makeInlistFile(stackID),  output=stackname,
                       combine="average", reject="none" )
                    co.reportOutput(stackname)
                else:
                    print "'%s' was not combined because there is only one image." %( stacklist[0] )
            else:
                print "There are not enough images to combine."
        except:
            print "Problem combining and averaging"
            raise 
        yield co
        
    def printHeaders(self, co):
        try:
            print "writing out headers"
            co.printHeaders()
        except:
            raise "Problem printing out headers"
        yield co
        
