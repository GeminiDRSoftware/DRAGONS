#from Reductionobjects import Reductionobject
from primitives_GEMINI import GEMINIPrimitives
# All GEMINI IRAF task wrappers.

from pyraf.iraf import tables, stsdas, images
from pyraf.iraf import gemini
import pyraf


import pyfits
import numdisplay
import string

yes = pyraf.iraf.yes
no = pyraf.iraf.no

# NOTE, the sys.stdout stuff is to shut up gemini and gmos startup... some primitives
# don't execute pyraf code and so do not need to print this interactive 
# package init display (it shows something akin to the dir(gmos)
import sys, StringIO
SAVEOUT = sys.stdout
capture = StringIO.StringIO()
sys.stdout = capture
gemini()
gemini.gmos()
sys.stdout = SAVEOUT

class GMOS_IMAGEPrimitives(GEMINIPrimitives):
    def init(self, co):
        pyraf.iraf.set (adata=co["adata"].value)  
        GEMINIPrimitives.init(self, co)
        return co

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def flatCreate(self, co):

        # FLAT made with giflat
        try:
            print 'combining and normalizing best 20 twilight flats'
            gemini.giflat(co.inputsAsStr(), outflat=co["outflat"],
                bias=co.calName("REDUCED_BIAS"),rawpath=co["caldir"],
                fl_over=co["fl_over"], fl_trim=co["fl_trim"], 
                fl_vardq=co["fl_vardq"])
        except:
            print "problem combining imaging flats with giflat"
            raise
        yield co

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def biasCreate(self, co):
        # Things done to the bias image before we subtract it:
        # overscan subtract
        # overscan trim
        # average combine images

        # BIAS made for all GMOS modes (imaging, spectroscopy, IFU) we need to
        # consider a generic task. using gbias (IRAF generic task)
        try:
            print "combining biases to create master bias"
            gemini.gbias(co.inputsAsStr(), outbias=co["outbias"],
                rawpath=co["caldir"], fl_trim=co["fl_trim"], 
                fl_over=co["fl_over"], fl_vardq=co["fl_vardq"])
        except:
            print "problem combining biases with gbias"
            raise SystemExit

        yield co 

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def prepare(self, co):
        try:
            print "Updating keywords PIXSCALE, NEXTEND, OBSMODE, GEM-TLM, GPREPARE"
            print "Updating GAIN keyword by calling GGAIN"
            gemini.gmos.gprepare(co.inputsAsStr(strippath = True), rawpath="adata$")
            co.reportOutput(co.prependNames("g", currentDir = True))
            
        except:
            print "Problem in GPREPARE"
            raise 
        yield co 

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def overscanSub(self, co):
        try:
            print "Determining overscan subtraction region using nbiascontam"
            print "parameter and BIASSEC header keyword"
            print "Subtracting overscan bias levels using colbias"
            gemini.gmos.gireduce(co.inputsAsStr(strippath=True), fl_over=pyraf.iraf.yes,fl_trim=no, fl_bias=no, \
                fl_flat=no, outpref="oversub_")
            co.reportOutput(co.prependNames("oversub_", currentDir = True))
        except:
            print "Problem subtracting overscan bias"
            print "Problem in GIREDUCE"
            raise
        yield co

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def overscanTrim(self, co):
        try:
            print "Determining overscan region using BIASSEC header keyword"
            print "Trimming off overscan"
            gemini.gmos.gireduce(co.inputsAsStr(), fl_over=no,fl_trim=yes, 
            fl_bias=no, fl_flat=no, outpref="trim_")
            co.reportOutput(co.prependNames("trim_"))
        except:
            print "Problem trimming off overscan region"
            print "Problem in GIREDUCE"
            raise 
        yield co

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def biasSub(self, co):
        # not really sure we need to use gireduce here. I think we could easily make a
        # more generic bias sub task
        try:
            print "Subtracting off bias "+co.calFilename("bias")+" from "+co.inputsAsStr()
            gemini.gmos.gireduce(co.inputsAsStr(), fl_over=no,
                fl_trim=no, fl_bias=yes,bias=co.calFilename("bias"),
                fl_flat=no, outpref="biassub_") # this flag was removed?,fl_mult=no)
            co.reportOutput(co.prependNames("biassub_"))
        except:
            print "Problem subtracting bias"
            print "Problem in GIREDUCE"
            raise
            
        yield co

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def flatField(self, co):
        try:
            print "dividing "+co.calFilename("flat")+" from "+co.inputsAsStr()
            gemini.gmos.gireduce(co.inputsAsStr(), fl_over=no,fl_trim=no,
                fl_bias=no, flat1=co.calFilename("flat"), fl_flat=yes, outpref="flatdiv_")    
        except:
            print "Problem dividing by normalized flat"
            print "Problem in GIREDUCE"
            raise
        yield co

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def fringeCreate(self, co):
        try:
            print "creating fringe frame"
            gemini.gifringe(co.inputsAsStr(), "fringe")
        except:
            print "Problem creating fringe from "+co.inputsAsStr()
            print "Problem in GIFRINGE"
            raise
        yield co
        
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def fringeSubtract(self, co):
        try:
            print "subtracting fringe frame"
            gemini.girmfringe(co.inputsAsStr(), co["fringe"])
        except:
            print "Problem subtracting fringe from "+co.inputsAsStr()
            print "Problem in GIRMFRINGE"
            raise
        yield co

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def findshiftsAndCombine(self, co):
       try:
          print "shifting and combining images"
          # @@TODO hardcoded parmeters and ***imcoadd.dat may need to move from 
          # imcoadd_data/test4 to test_data dir before running
          gemini.imcoadd(co.stack_inputsAsStr(),fwhm=5, threshold=100,\
                fl_over=yes, fl_avg=yes)
       except:
          print "Problem shifting and combining images "
          print "Problem in IMCOADD"
          raise
       yield co
       
    
    
    