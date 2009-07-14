#!/bin/env python
import astrodata
from datetime import datetime

a = datetime.now()

from RecipeManager import ContextObject
from RecipeManager import RecipeLibrary
from GeminiData import GeminiData
from optparse import OptionParser
import sys, os, glob
import time

b = datetime.now()



############################################################
# this script was developed to exercise the GeminiDataType class
# but now serves a general purpose in addition to that and as
# a demo for GeminiData... see options documentation.

# parsing the command line
parser = OptionParser()
# parser.add_option("-r", "--reduce", dest="twdir", default =".",
#        help="Recursively walk given directory and put type information to stdout.")
parser.add_option("-r", "--reduce", dest="bReduce", action="store_true",
                  help="Select the appropriate recipe(s) and execute them on the given file(S)")
parser.add_option("-m", "--monitor", dest="bMonitor", action="store_true",
                  default = False,
                  help= "Open TkInter Window to Monitor Progress of" + \
                  "execution. " + \
                  "Note: One window is opened for each recipe which " + \
                  "will run")
(options,  args) = parser.parse_args()

useTK =  options.bMonitor

# some built in arguments for now during testing to save long command lines

# ------
#$Id: recipeman.py,v 1.8 2008/08/05 03:28:06 callen Exp $
from tkMonitor import *

infile   = "./recipedata/N20020606S0141.fits"
biasfile = "./recipedata/N20020507S0045_bias.fits"
flatfile = "./recipedata/N20020606S0149_flat.fits"
adatadir = "./recipedata/"

if (False): # this code will delete all *.fits in the current directory, was convienient for testing at one point
            # kept around for a bit as it may make a reappearance as a command line flag driven behavior.
    try:
        files = glob.glob("*.fits")
        for fil in files:
            os.remove(fil)

    except TypeError:
        # this happens if glob returned 0 files, no previous outputs
        raise

# end of arguments kluge (hard coded filenames)
generate_pycallgraphs = False
if (generate_pycallgraphs):
    import pycallgraph
    pycallgraph.start_trace()
    
gd = GeminiData(infile)

# start the Gemini Specific class code

rl = RecipeLibrary()
ro = rl.retrieveReductionObject(astrotype="GMOS_OBJECT_RAW") # can be done by filename

reclist = rl.getApplicableRecipes(infile)
recdict = rl.getApplicableRecipes(infile, collate=True)

types = gd.getTypes()
print "\nProcessing dataset %s\n\tRecipes found by type:" % infile
for typ in recdict.keys():
    recs = recdict[typ]
    print "\ttype %s" % typ
    for rec in recs:
        print "\t\t%s" % rec

bReportHistory = False
cwlist = []
if (useTK):
    cw = TkRecipeControl(recipes = reclist)
    cw.start()

for rec in reclist:

    try:
        # create fresh context object
        # @@TODO:possible: see if deepcopy can do this better 
        co = ContextObject() 
        # add input file
        co.addInput(infile)
        # add biases
        co.addCal(infile, "bias", biasfile)
        co.addCal(infile, "flat", flatfile)
        co.update({"adata":adatadir})
        if (useTK):
            while cw.bReady == False:
                # this is hopefully not really needed
                # did it to give the tk thread a chance to get running
                time.sleep(.1)
            cw.newControlWindow(rec,co)

        # @@TODO:evaluate use of init for each recipe vs. for all recipes
        ro.init(co)
        print "running recipe '%s'" % rec
        rl.loadAndBindRecipe(ro,rec, file=infile)
        if (useTK):
            cw.running(rec)
        for coi in ro.substeps(rec, co):
            coi.processCmdReq()
            while (coi.paused):
                time.sleep(.100)
            if co.finished:
                break
            pass
    
    except KeyboardInterrupt:
        co.isFinished(True)
        cw.quit()
        print "Ctrl-C Exit"
        sys.exit(0)
    except:
        print "CONTEXT AFTER FATAL ERROR"
        print "--------------------------"
        if (bReportHistory):
            co.reportHistory()
            rl.reportHistory()
        co.isFinished(True)
        raise

    if (bReportHistory):

        print "CONTEXT HISTORY"
        print "---------------"

        co.reportHistory()
        rl.reportHistory()
        
    co.isFinished(True)

if useTK:
    cw.done()
    raw_input("Press Enter to Close Monitor Windows:")
    cw.quit()
    

if (generate_pycallgraphs):
    pycallgraph.make_dot_graph("recipman-callgraph.png")

# print co.reportHistory()
# main()
    



