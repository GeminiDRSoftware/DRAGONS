#from Reductionobjects import Reductionobject
from primitives_GEMINI import GEMINIPrimitives
# All GEMINI IRAF task wrappers.
import time
from astrodata.adutils import filesystem
from astrodata import IDFactory
from astrodata import Descriptors
from astrodata.data import AstroData

from pyraf.iraf import tables, stsdas, images
from pyraf.iraf import gemini
import pyraf
import iqtool
from iqtool.iq import getiq
from preparelib.prepareTK import PrepareTK
from preparelib.wcsTK import WCSTK


import pyfits
import numdisplay
import string

yes = pyraf.iraf.yes
no = pyraf.iraf.no

# NOTE, the sys.stdout stuff is to shut up gemini and gmos startup... some primitives
# don't execute pyraf code and so do not need to print this interactive 
# package init display (it shows something akin to the dir(gmos)
import sys, StringIO, os
SAVEOUT = sys.stdout
capture = StringIO.StringIO()
sys.stdout = capture
gemini()
gemini.gmos()
sys.stdout = SAVEOUT

class GMOS_SPECTException:
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


class GMOS_SPECTPrimitives(GEMINIPrimitives):
    astrotype = "GMOS_SPECT"
    
    def init(self, rc):
        
        if "iraf" in rc and "adata" in rc["iraf"]:
            pyraf.iraf.set (adata=rc["iraf"]['adata'])  
        else:
            # @@REFERENCEIMAGE: used to set adata path for primitives
            if len(rc.inputs) > 0:
                (root, name) = os.path.split(rc.inputs[0].filename)
                pyraf.iraf.set (adata=root)
                if "iraf" not in rc:
                    rc.update({"iraf":{}})
                if "adata" not in rc["iraf"]:
                    rc["iraf"].update({"adata":root}) 
        
        GEMINIPrimitives.init(self, rc)
        return rc

    
    def display(self, rc):
        from adutils.future import gemDisplay
        import pyraf
        from pyraf import iraf
        iraf.set(stdimage='imtgmos')
        ds = gemDisplay.getDisplayService()
        for i in range(0, len(rc.inputs)):   
            inputRecord = rc.inputs[i]
            gemini.gmos.gdisplay( inputRecord.filename, i+1, fl_imexam=iraf.no,
                Stdout = rc.getIrafStdout(), Stderr = rc.getIrafStderr() )
            # this version had the display id conversion code which we'll need to redo
            # code above just uses the loop index as frame number
            #gemini.gmos.gdisplay( inputRecord.filename, ds.displayID2frame(rq.disID), fl_imexam=iraf.no,
            #    Stdout = coi.getIrafStdout(), Stderr = coi.getIrafStderr() )
        yield rc
    
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   
    #$$$$$$$$$$$$$$$$$$$$ NEW STUFF BY KYLE FOR: PREPARE $$$$$$$$$$$$$$$$$$$$$
    '''
    all the stuff in here is very much a work in progress and I will not be fully
    commenting it for others while developing it, sorry.
    '''
    
    
    def validateData(self,rc):
        #to be written
        print "prim_G_I507: nothing in here yet"
        yield rc
    def standardizeHeaders(self,rc):
        #to be written
        print "prim_G_I511: nothing in here yet"
        yield rc
    def validateWCS(self,rc):
        #to be written
        print "prim_G_I515: nothing in here yet"
        yield rc

    def standardizeHeaders(self,rc):
        try:
            print 'prepare step 1'
            ptk = PrepareTK()
            #print rc.inputsAsStr(strippath = True)
            #print rc['iraf']['adata']
            #print rc.getIrafStdout()
            #print rc.getIrafStderr()
            #outnamerc = rc.prependNames("g", currentDir = True)
            #print 'outnamerc: ', outnamerc
            
            #outs=os.path.basename(outnamerc[0][0])
            #print outs
            
            #ad = AstroData()
            #rc.reportOutput([ad,ad,rc.inputs[0].filename, rc.inputs[0].ad])
            
            #for inp in rc.inputs:
            #    print "input id:", id(inp.ad)
            #for inp in rc.outputs["standard"]:
            #    print "output id:", id(inp.ad)    
            
            for ad in rc.getInputs(style="AD"):
                infilename = ad.filename
                print 'prim_G_S133 :', infilename
                print 'prim_G_S134: ', os.path.abspath(infilename) #absolute path of input file
                print 'prim_G_S135: ', os.path.dirname(infilename) #reletive directory of input file without /
                adnew = ad
                ad.filename = 'g'+os.path.basename(ad.filename)
                outfilename = ad.filename
                #print 'prim_G_I531 :', outfilename
                ptk.fixHeader(ad,fullPrint=False)
                print "Prim_G_S140: ", 'header fixed'
            
            
            # ptk.fixHeader(ins,outs)
            
            #outfilerc = rc.reportOutput(outnameptk)
            #print 'outfilerc: ', outf
            
        except:
            print "Problem preparing the image."
            raise 
        
        yield rc 
    
    def attachMDF(self,rc):
        try:
            print 'prepare step 2'
            
            for ad in rc.getInputs(style ='AD'):
                infilename = ad.filename
                print 'prim_G_S162:', infilename
                #print 'prim_G_I531: ', os.path.abspath(infilename) #absolute path of input file
                #print 'prim_G_I531: ', os.path.dirname(infilename) #reletive directory of input file without /
                
                pathname = 'kyles_test_images/' #$$$$ HARDCODED FOR NOW, TILL FIX COMES FROM CRAIG
                maskname = ad.phuGetKeyValue("MASKNAME")
                print "Pim_G_S170: maskname = ", maskname
                inMDFname = 'kyles_test_images/'+maskname +'.fits'
                print 'Prim_G_S172: input MDF file = ', inMDFname
                admdf = AstroData(inMDFname)
                admdf.extSetKeyValue(len(admdf)-1,'EXTNAME', 'MDF',"Extension name" )
                
                print admdf[0].getHeader()
                print admdf.info()
                #admdf[0].setKeyValue("EXTNAME","MDF")
                #admdf[0].setKeyValue("EXTVER",1)
                #print 'prim_G_S172: ', os.path.dirname(infilename) #reletive directory of input file without /
                ad.append(moredata=admdf)
                print ad.info()
                
                ad.extSetKeyValue(len(ad)-1,'EXTNAME', 'MDF',"Extension name" )
                ad.extSetKeyValue(len(ad)-1,'EXTVER', 1,"Extension version" )
                #print ad[3].getHeader()
                #addMDF(ad,mdf,fullPrint=True)
                print ad.info()
                #print len(ad)
                print 'prim_G_S177: finished adding the MDF'
        except:
            print "Problem preparing the image."
            raise 
        
        yield rc
        
        
    #$$$$$$$$$$$$$$$$$$$$$$$ END OF KYLES NEW STUFF $$$$$$$$$$$$$$$$$$$$$$$$$$
        