#!/usr/bin/env python
import astrodata
import terminal
from terminal import TerminalController
from astrodata.AstroData import AstroData
from datetime import datetime
term = TerminalController()
a = datetime.now()

from RecipeManager import ReductionContext
from RecipeManager import RecipeLibrary
from GeminiData import GeminiData
from optparse import OptionParser
from StackKeeper import StackKeeper
from CalibrationRequestEvent import CalibrationRequestEvent
from StackableEvents import UpdateStackableEvent, GetStackableEvent

from LocalCalibrationService import CalibrationService

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
parser.add_option("-r", "--recipe", dest="recipename", default=None,
                  help="Specify which recipe to run by name.")

parser.add_option("-m", "--monitor", dest="bMonitor", action="store_true",
                  default = False,
                  help= "Open TkInter Window to Monitor Progress of" + \
                  "execution. " + \
                  "Note: One window is opened for each recipe which " + \
                  "will run")
(options,  args) = parser.parse_args()

useTK =  options.bMonitor
# ------
#$Id: recipeman.py,v 1.8 2008/08/05 03:28:06 callen Exp $
from tkMonitor import *

# start color printing filter for xgtermc
term = TerminalController()
REALSTDOUT = sys.stdout
sys.stdout = terminal.ColorStdout(REALSTDOUT, term)

print "${NORMAL}"
try:
    infile   = args[0] # "./recipedata/N20020606S0141.fits"
except IndexError:
    print "${RED}NO INPUT FILE${NORMAL}"
    sys.exit(1)

adatadir = "./recipedata/"

generate_pycallgraphs = False
if (generate_pycallgraphs):
    import pycallgraph
    pycallgraph.start_trace()
    
gd = AstroData(infile)
# start the Gemini Specific class code

# get RecipeLibrary
rl = RecipeLibrary()

# get ReductionObject for this dataset
#ro = rl.retrieveReductionObject(astrotype="GMOS_IMAGE") # can be done by filename
ro = rl.retrieveReductionObject(infile) # can be done by filename

if options.recipename == None:
    reclist = rl.getApplicableRecipes(infile)
    recdict = rl.getApplicableRecipes(infile, collate=True)
else:
    #force recipe
    reclist = [options.recipename]
    recdict = {"all": [options.recipename]}

types = gd.getTypes()

# Local Calibration Service Setup
cs = CalibrationService()


title = "  Processing dataset: %s  " % infile
tl = len(title)
tb = " " * tl
print "${REVERSE}" + tb
print title
print tb + "${NORMAL}"

if options.recipename == None:
    #print ("\n${UNDERLINE}Recipe(s) found by dataset type:${NORMAL}")
    print "\nRecipe(s) found by dataset type:"
else:
    #print ("\n${UNDERLINE}A recipe was specified:${NORMAL}")
    print "\nA recipe was specified:"
    
for typ in recdict.keys():
    recs = recdict[typ]
    print "  for type: %s" % typ
    for rec in recs:
        print "    %s" % rec

print 

bReportHistory = False
cwlist = []
if (useTK):
    cw = TkRecipeControl(recipes = reclist)
    cw.start()
    

for rec in reclist:

    try:
        # create fresh context object
        # @@TODO:possible: see if deepcopy can do this better 
        co = ReductionContext()
        # restore cache
        if not os.path.exists(".reducecache"):
            os.mkdir(".reducecache")
        calindfile = "./.reducecache/calindex.pkl"
        co.restoreCalIndex(calindfile)
        stkindfile = "./.reducecache/stkindex.pkl"
        co.restoreStkIndex( stkindfile )
        
        
        # add input file
        co.addInput(infile)
        co.update({"adata":adatadir})
        if (useTK):
            while cw.bReady == False:
                # this is hopefully not really needed
                # did it to give the tk thread a chance to get running
                time.sleep(.1)
            cw.newControlWindow(rec,co)
            cw.mainWindow.protocol("WM_DELETE_WINDOW", co.finish) 


        # @@TODO:evaluate use of init for each recipe vs. for all recipes
        ro.init(co)
        print term.render("${GREEN}running recipe: '%s'${NORMAL}\n") % rec
        rl.loadAndBindRecipe(ro,rec, file=infile)
        if (useTK):
            cw.running(rec)
            
        ################
        # CONTROL LOOP #
        ################
        for coi in ro.substeps(rec, co):
            coi.processCmdReq()
            while (coi.paused):
                time.sleep(.100)
            if co.finished:
                break
            
            #process calibration requests
            for rq in coi.rorqs:
                if type(rq) == CalibrationRequestEvent:
                    fn = rq.filename
                    typ = rq.caltype
                    calname = coi.getCal(fn, typ)
                    if calname == None:
                        # Do the calibration search
                        calname = cs.search( rq )
                        if calname == None:
                            print "No suitable calibration for '" + fn + "'."
                        elif len( calname ) >= 1:
                            # Not sure if this is where the one returned calibration is chosen, or if
                            # that is done in the calibration service, etc.
                            calname = calname[0]
                        coi.addCal(fn, typ, calname)
                        coi.persistCalIndex( calindfile )
                elif type(rq) == UpdateStackableEvent:
                    coi.stackAppend(rq.stkID, rq.stkList)
                    coi.persistStkIndex( stkindfile )
                elif type(rq) == GetStackableEvent:
                    pass
                    # Don't actually do anything, because this primitive allows the control system to
                    #  retrieve the list from another resource, but reduce lets ReductionContext keep the
                    # cache.
                    #print "RD172: GET STACKABLE REQS:", rq
            # CLEAR THE REQUEST LEAGUE
            coi.clearRqs()
        
    
    except KeyboardInterrupt:
        co.isFinished(True)
        if (useTK):
            cw.quit()
        co.persistCalIndex(calindfile)
        print "Ctrl-C Exit"
        sys.exit(0)
    except:
        print "CONTEXT AFTER FATAL ERROR"
        print "--------------------------"
        raise
        co.persistCalIndex(calindfile)
        if (bReportHistory):
            co.reportHistory()
            rl.reportHistory()
        co.isFinished(True)
        raise
    co.persistCalIndex(calindfile)

    if (bReportHistory):

        print "CONTEXT HISTORY"
        print "---------------"

        co.reportHistory()
        rl.reportHistory()
        
    co.isFinished(True)

if useTK:
    try:
        cw.done()
        cw.mainWindow.after_cancel(cw.pcqid)
        if cw.killed == True:
            raw_input("Press Enter to Close Monitor Windows:")
        # After ID print cw.pcqid
        cw.mainWindow.quit()
    except:
        raise
        cw.quit()    

if (generate_pycallgraphs):
    pycallgraph.make_dot_graph("recipman-callgraph.png")

from time import sleep
while (False):
    for th in threading.enumerate():
        print str(th)
    sleep(5.)
# print co.reportHistory()
# main()
    



