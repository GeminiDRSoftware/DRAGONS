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
from ReductionObjectRequests import CalibrationRequest, UpdateStackableRequest, \
        GetStackableRequest, DisplayRequest


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
version = '1_0'
parser.set_description( 
"""The standalone recipe processor from Gemini. Created by Craig Allen (callen@gemini.edu)."""
 )
parser.set_usage( parser.get_usage()[:-1] + " file.fits\n" )
# Testing
parser.add_option("-r", "--recipe", dest="recipename", default=None,
                  help="Specify which recipe to run by name.")

parser.add_option("-m", "--monitor", dest="bMonitor", action="store_true",
                  default = False,
                  help= "Open TkInter Window to Monitor Progress of" + \
                  "execution. " + \
                  "Note: One window is opened for each recipe which " + \
                  "will run")
parser.add_option("--addcal", dest="add_cal", default=None, type="string",
                  help="'--caltype' Must be specified as well when using this! " + \
                  "Provide the filename for a calibration. This is will overwrite " + \
                  "an existing calibration if in the index. An example of what " + \
                  "this would look like: \n" + \
                  "reduce --addcal=N2009_bias.fits --caltype=bias N20091002S0219.fits" )
parser.add_option("--remcal", dest="rem_cal", default=False, action="store_true",
                  help="'--caltype' Must be specified as well when using this! " + \
                  "This will remove the calibration for that file from cache. By making --caltype " + \
                  "'all', all the associated calibrations for that file will be removed. An " + \
                  "example of what this would look like: \n" + \
                  "reduce --remcal --caltype=bias N20091002S0219.fits" )
parser.add_option("--clrcal", dest="clr_cal", default=False, action="store_true",
                  help="Remove all calibrations.")
parser.add_option("--caltype", dest="cal_type", default=None, type="string",
                  help="Works in conjunction with '--addcal'. Ignored otherwise. " + \
                  "This should be the type of calibration in lowercase and one word. " + \
                  "For example: 'bias', 'twilight'.")
(options,  args) = parser.parse_args()


useTK =  options.bMonitor
# ------
#$Id: recipeman.py,v 1.8 2008/08/05 03:28:06 callen Exp $
from tkMonitor import *

# start color printing filter for xgtermc
term = TerminalController()
REALSTDOUT = sys.stdout
sys.stdout = terminal.ColorStdout(REALSTDOUT, term)


adatadir = "./recipedata/"
calindfile = "./.reducecache/calindex.pkl"
stkindfile = "./.reducecache/stkindex.pkl"



def command_line():
    '''
    This function is just here so that all the command line oriented parsing is one common location.
    Hopefully, this makes things look a little cleaner.
    '''
    infile = None
    if options.clr_cal:
        clrFile = None
        
        co = ReductionContext()
        co.restoreCalIndex(calindfile)
        co.calibrations = {}
        co.persistCalIndex( calindfile )
        print "Entire calibration cache cleared."
        sys.exit(0)
    
    print "${NORMAL}"
    try:
        if len( args ) == 0:
            raise IndexError
        infile   = args # "./recipedata/N20020606S0141.fits"
        #print "ARGS:", args
    except IndexError:
        
        print "${RED}NO INPUT FILE${NORMAL}"
        parser.print_help()
        sys.exit(1)
    
    for inf in infile:
        if not os.access( inf, os.R_OK ):
            print "'" + inf + "' does not exist or cannot be accessed."
            sys.exit(1)
    
    if options.add_cal != None:
        if options.cal_type == None:
            print "Reduce requires a calibration type. Use --cal-type. For more " + \
            "information use '-h' or '--help'."
            sys.exit(1)
        elif not os.access( options.add_cal, os.R_OK ):
            print "'" + options.add_cal + "' does not exist or cannot be accessed."
            sys.exit(1)
        
        # @@TODO: Perhaps need a list of valid calibration types.
        # @@TODO: Need testing if passed in calibration type is valid.
        
        co = ReductionContext()
        co.restoreCalIndex(calindfile)
        for arg in infile:
            co.addCal( arg, options.cal_type, os.path.abspath(options.add_cal) )
        co.persistCalIndex( calindfile )
        print "'" + options.add_cal + "' was successfully added for '" + str(infile) + "'."
        sys.exit(0)
        
    elif options.rem_cal:
        if options.cal_type == None:
            print "Reduce requires a calibration type. Use --cal-type. For more " + \
            "information use '-h' or '--help'."
            sys.exit(1)
        
        # @@TODO: Perhaps need a list of valid calibration types.
        # @@TODO: Need testing if passed in calibration type is valid.
        
        co = ReductionContext()
        co.restoreCalIndex(calindfile)
        if options.cal_type == 'all':
            for key in co.calibrations.keys():
                for arg in infile:
                    if os.path.abspath(arg) in key:
                        co.calibrations.pop( key )
                        
            print "All calibrations for " + str(infile) + "' were removed."
        else:
            for arg in infile:
                try:
                    co.calibrations.pop( (os.path.abspath(arg), options.cal_type) )
                except:
                    print arg + ' had no ' + options.cal_type
            print "'" + options.cal_type + "' was removed from '" + str(infile) + "'."
        co.persistCalIndex( calindfile )
        sys.exit(0)
    
    return infile
    

infiles = command_line()

generate_pycallgraphs = False
if (generate_pycallgraphs):
    import pycallgraph
    pycallgraph.start_trace()
    
for infile in infiles:
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
            
            co.restoreCalIndex(calindfile)
            
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
                    if type(rq) == CalibrationRequest:
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
                    elif type(rq) == UpdateStackableRequest:
                        coi.stackAppend(rq.stkID, rq.stkList)
                        coi.persistStkIndex( stkindfile )
                    elif type(rq) == GetStackableRequest:
                        pass
                        # Don't actually do anything, because this primitive allows the control system to
                        #  retrieve the list from another resource, but reduce lets ReductionContext keep the
                        # cache.
                        #print "RD172: GET STACKABLE REQS:", rq
                    elif type(rq) == DisplayRequest:
                        print 'you made it!'
                        print rq
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
