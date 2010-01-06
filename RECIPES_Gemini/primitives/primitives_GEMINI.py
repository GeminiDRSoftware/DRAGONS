from time import sleep
import time
from astrodata.ReductionObjects import ReductionObject
from utils import filesystem
from astrodata import IDFactory
import os

import IQTool
from IQTool.iq import getiq

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


class GEMINIPrimitives(ReductionObject):
    
    def init(self, rc):
        ReductionObject.init(self, rc)
        return rc
        
    def pause(self, rc):
        rc.requestPause()
        yield rc
#------------------------------------------------------------------------------ 
    def logFilename (self, rc):
        print "logFilename"
        for i in range(0,5):
            print "\tlogFilename",i
            sleep(stepduration)
            yield rc
#------------------------------------------------------------------------------ 
    def displayStructure(self, rc):
        print "displayStructure"
        for i in range(0,5):
            print "\tds ",i
            sleep(stepduration)
            yield rc
#------------------------------------------------------------------------------ 
    def summarize(self, rc):
        print "done with task"
        for i in range(0,5):
            sleep(stepduration)
            yield rc        
#------------------------------------------------------------------------------ 
    def gem_produce_im_flat(self, rc):
        print "gem_produce_imflat step called"
        # rc.update({"flat" :rc.calibrations[(rc.inputs[0], "flat")]})
        yield rc
#------------------------------------------------------------------------------ 
    def gem_produce_bias(self, rc):
        print "gem_produce_bias step called"
        # rc.update({"bias" :rc.calibrations[(rc.inputs[0], "bias")]})
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
#------------------------------------------------------------------------------ 
    def getProcessedFlat(self, rc):
        try:
            print "getting flat"
            rc.rqCal( "twilight" )
        except:
            print "Problem getting flat"
            raise 
        
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
    def printParameters(self, rc):
        print "printing parameters"
        print rc.paramsummary()
        yield rc
#------------------------------------------------------------------------------ 
    def display(self, rc):
        try:
            print "displaying output"
            print "rc.localparms:", repr(rc.localparms)
            print 'rc["fnarp"]', repr(rc["fnarp"])
            print 'rc["thing"]', repr(rc["thing"])
            rc.rqDisplay()           
        except:
            print "Problem displaying output"
            raise 

        yield rc
#------------------------------------------------------------------------------ 
    def mosaicChips(self, rc):
       try:
          print "producing image mosaic"
          #mstr = 'flatdiv_'+rc.inputsAsStr()          
          gemini.gmosaic( rc.inputsAsStr(), outpref="mo_",
            Stdout = rc.getIrafStdout(), Stderr = rc.getIrafStderr() )
          
          
          rc.reportOutput(rc.prependNames("mo_", currentDir = True))
       except:
          print "Problem producing image mosaic"
          raise 

       yield rc
#------------------------------------------------------------------------------ 
    def averageCombine(self, rc):
        try:
            # @@TODO: need to include parameter options here
            print "Combining and averaging" 
            filesystem.deleteFile('inlist')
            
            templist = []
            
            for inp in rc.inputs:
                 templist.append( IDFactory.generateStackableID( inp.ad ) )
                 
            templist = list( set(templist) ) # Removes duplicates.
            
            for stackID in templist:
                #@@FIXME: There are a lot of big issues in here. First, we need to backup the
                # previous average combined file, not delete. Backup is needed if something goes wrong.
                # Second, the pathnames in here have too many assumptions. (i.e.) It is assumed all the
                # stackable images are in the same spot which may not be the case.
                
                stacklist = rc.getStack( stackID ).filelist
                #print "pG147: STACKLIST:", stacklist

                
                if len( stacklist ) > 1:
                    stackname = "avgcomb_" + os.path.basename(stacklist[0])
                    filesystem.deleteFile( stackname )
                    gemini.gemcombine( rc.makeInlistFile(stackID),  output=stackname,
                        combine="average", reject="none", 
                        Stdout = rc.getIrafStdout(), Stderr = rc.getIrafStderr())
                    rc.reportOutput(stackname)
                else:
                    print "'%s' was not combined because there is only one image." %( stacklist[0] )
        except:
            print "Problem combining and averaging"
            raise

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
    def crashReduce(self, rc):
        raise 'Crashing'
        yield rc
