#!/usr/bin/env python
#import time
#ost = time.time()
#------------------------------------------------------------------------------ 
import commands
from datetime import datetime
import glob
from optparse import OptionParser
import os
#st = time.time()
try:
    import pyraf
    from pyraf import iraf
except:
    print "didn't get pyraf"
#et = time.time()
#print 'IRAF TIME', (et-st)
import subprocess
import sys
import time
#------------------------------------------------------------------------------ 
a = datetime.now()

import astrodata
from astrodata.AstroData import AstroData
from astrodata.RecipeManager import ReductionContext
from astrodata.RecipeManager import RecipeLibrary
from astrodata.StackKeeper import StackKeeper
from astrodata.ReductionObjectRequests import CalibrationRequest, UpdateStackableRequest, \
        GetStackableRequest, DisplayRequest, ImageQualityRequest
from astrodata import gdpgutil
from astrodata.LocalCalibrationService import CalibrationService
from utils.future import gemDisplay
from utils import paramutil
from utils.gemutil import gemdate
from utils import terminal
from utils.terminal import TerminalController, ProgressBar 
#------------------------------------------------------------------------------ 
#oet = time.time()
#print 'TIME:', (oet -ost)
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
                  "This will remove the calibration for that file from cache. An " + \
                  "example of what this would look like: \n" + \
                  "reduce --remcal --caltype=bias N20091002S0219.fits" )
parser.add_option("--clrcal", dest="clr_cal", default=False, action="store_true",
                  help="Remove all calibrations.")
parser.add_option("--caltype", dest="cal_type", default=None, type="string",
                  help="Works in conjunction with '--addcal'. Ignored otherwise. " + \
                  "This should be the type of calibration in lowercase and one word. " + \
                  "For example: 'bias', 'twilight'.")
parser.add_option("--showcolors", dest="show_colors", default=False, action = "store_true",
                    help="""For debugging any color output problems, show
s what colors
                    reduce thinks are available based on the terminal setting.""")
##@@FIXME: This next option should not be put into the package
parser.add_option("-x", "--rtf-mode", dest="rtf", default=False, action="store_true",
                  help="Only used for rtf.")
parser.add_option("-i", "--intelligence", dest='intelligence', default=False, action="store_true",
                  help="Give the system some intelligence to perform operations faster and smoother.")
(options,  args) = parser.parse_args()

useTK =  options.bMonitor
# ------
#$Id: recipeman.py,v 1.8 2008/08/05 03:28:06 callen Exp $
from astrodata.tkMonitor import *

# start color printing filter for xgtermc
REALSTDOUT = sys.stdout
REALSTDERR = sys.stderr
filteredstdout = terminal.FilteredStdout()
filteredstdout.addFilter( terminal.ColorFilter())
irafstdout = terminal.IrafStdout(fout = filteredstdout)
sys.stdout = filteredstdout
# sys.stderr = terminal.ColorStdout(REALSTDERR, term)
adatadir = "./recipedata/"
calindfile = "./.reducecache/calindex.pkl"
stkindfile = "./.reducecache/stkindex.pkl"



def command_line():
    '''
    This function is just here so that all the command line oriented parsing is one common location.
    Hopefully, this makes things look a little cleaner.
    '''
    
    if  options.show_colors:
        print dir(term)
        sys.exit(0)
    infile = None
    if options.clr_cal:
        clrFile = None
        
        co = ReductionContext()
        co.restoreCalIndex(calindfile)
        co.calibrations = {}
        co.persistCalIndex( calindfile )
        print "Entire calibration cache cleared."
        sys.exit(0)
    
    print "${NORMAL}",
    try:
        if len( args ) == 0:
            raise IndexError
        infile   = args
    except IndexError:
        print "${RED}NO INPUT FILE${NORMAL}"
        parser.print_help()
        sys.exit(1)
    
    input_files = []
    for inf in infile:
        #"""
        tmpInp = paramutil.checkImageParam( inf )
        if tmpInp == None:
            raise "The input had "+ str(inf)+" cannot be loaded."
        input_files.extend( tmpInp )
        
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

            co.addCal( AstroData(arg), options.cal_type, os.path.abspath(options.add_cal) )
        co.persistCalIndex( calindfile )
        print "'" + options.add_cal + "' was successfully added for '" + str(input_files) + "'."
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
        for arg in infile:
            try:
                co.rmCal( arg, options.cal_type )
            except:
                print arg + ' had no ' + options.cal_type
        print "'" + options.cal_type + "' was removed from '" + str(input_files) + "'."
        co.persistCalIndex( calindfile )
        sys.exit(0)
    
    return input_files
    

infiles = command_line()

generate_pycallgraphs = False
if (generate_pycallgraphs):
    import pycallgraph
    pycallgraph.start_trace()

if options.intelligence:
    typeIndex = gdpgutil.clusterTypes( infiles )
    # If there was super intelligence, it would determine ordering. For now, it will 
    # run recipes in simple ordering, (i.e. the order values() is returned in).
    infiles = typeIndex.values()
else:
    ##@FIXME: This is pretty stupid
    testla = []
    for infile in infiles:
        testla.append( [AstroData(infile)] )
    infiles = testla


#===============================================================================
# Local PRS Components
#===============================================================================
# Local Calibration Service Setup
cs = CalibrationService()

# Local Display Service Setup
ds = gemDisplay.getDisplayService()


frameForDisplay = 1 
i = 1
numFiles = len(infiles)
for infile in infiles: #for dealing with multiple files.
    print "${BOLD}Starting File #%d, %s" % (i, infile[0].filename)
    currentImageNum = i
    i += 1
    # get RecipeLibrary
    rl = RecipeLibrary()
    
    # get ReductionObject for this dataset
    #ro = rl.retrieveReductionObject(astrotype="GMOS_IMAGE") # can be done by filename
    ro = rl.retrieveReductionObject(infile[0]) # can be done by filename #**
    
    if options.recipename == None:
        reclist = rl.getApplicableRecipes(infile[0]) #**
        recdict = rl.getApplicableRecipes(infile[0], collate=True) #**
    else:
        #force recipe
        reclist = [options.recipename]
        recdict = {"all": [options.recipename]}
    
    types = infile[0].getTypes()
    
    
    
    infilenames = []
    for infs in infile:
        if type(infs) == AstroData:
            infilenames.append( infs.filename )
        else:
            for inf in infs:
                infilenames.append( inf.filename )
        
    title = "  Processing dataset: %s  " % str(infilenames) #**
    tl = len(title)
    tb = " " * tl
    print "${REVERSE}" + tb
    print "${REVERSE}" + title
    print "${REVERSE}" + tb + "${NORMAL}",
    
    if options.recipename == None:
        print "\nRecipe(s) found by dataset type:"
    else:
        print "\nA recipe was specified:"
        
    for typ in recdict.keys():
        recs = recdict[typ]
        print "  for type: %s" % typ
        for rec in recs:
            print "    %s" % rec
    
    
    bReportHistory = False
    cwlist = []
    if (useTK and currentImageNum == 1):
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
            
            # add input files
            co.addInput(infile)
            co.setIrafStdout(irafstdout)
            co.setIrafStderr(irafstdout)
            
            rl.retrieveParameters(infile[0], co, rec)
            if (useTK):
                while cw.bReady == False:
                    # this is hopefully not really needed
                    # did it to give the tk thread a chance to get running
                    time.sleep(.1)
                cw.newControlWindow(rec,co)
                cw.mainWindow.protocol("WM_DELETE_WINDOW", co.finish) 
    
    
            # @@TODO:evaluate use of init for each recipe vs. for all recipes
            ro.init(co)
            print "running recipe: '%s'\n" % rec
            rl.loadAndBindRecipe(ro,rec, dataset=infile[0])
            if (useTK):
                cw.running(rec)
            
            controlLoopCounter = 1
            ################
            # CONTROL LOOP #
            ################
            #print str(dir(TerminalController))
            primfilter = terminal.PrimitiveFilter()
            filteredstdout.addFilter(primfilter)
            frameForDisplay = 1
            print "Created Filtered Output"
            # not this only works because we install a stdout filter right away with this
            # member function
            if (True): # try:
                for coi in ro.substeps(rec, co):
                    # filteredstdout.removeFilter(primfilter)
                    print "${NORMAL}",
                    coi.processCmdReq()
                    while (coi.paused):
                        time.sleep(.100)
                    if co.finished:
                        break

                    #process calibration requests
                    for rq in coi.rorqs:
                        rqTyp = type(rq)
                        msg = '${BOLD}REDUCE:${NORMAL}\n'
                        msg += '-'*30+'\n'
                        if rqTyp == CalibrationRequest:
                            fn = rq.filename
                            typ = rq.caltype
                            calname = coi.getCal(fn, typ)

                            if calname == None:
                                # Do the calibration search
                                calname = cs.search( rq )
                                if calname == None:
                                    raise "No suitable calibration for '" + str(fn) + "'."
                                elif len( calname ) >= 1:
                                    # Not sure if this is where the one returned calibration is chosen, or if
                                    # that is done in the calibration service, etc.
                                    calname = calname[0]
                                
                                
                                msg += 'A suitable %s found:\n' %(str(typ))
                                coi.addCal(fn, typ, calname)
                                coi.persistCalIndex( calindfile )
                            else:
                                msg += '%s already stored.\n' %(str(typ))
                                msg += 'Using:\n'
                            
                            #msg += '${RED}%s${NORMAL} at ${BLUE}%s${NORMAL}' %( str(os.path.basename(calname)), 
                            #                                                   str(os.path.dirname(calname)) )
                            msg += '${BLUE}%s%s${RED}%s${NORMAL}' %( os.path.dirname(calname), os.path.sep, os.path.basename(calname))
                                                            
                            #print msg
                            #print '-'*30
                                                 
                        elif rqTyp == UpdateStackableRequest:
                            coi.stackAppend(rq.stkID, rq.stkList)
                            coi.persistStkIndex( stkindfile )
                        elif rqTyp == GetStackableRequest:
                            pass
                            # Don't actually do anything, because this primitive allows the control system to
                            #  retrieve the list from another resource, but reduce lets ReductionContext keep the
                            # cache.
                            #print "RD172: GET STACKABLE REQS:", rq
                        elif rqTyp == DisplayRequest:
                            from pyraf.iraf import gemini
                            gemini()
                            gemini.gmos()
                            if ds.ds9 is None:
                                ds.setupDS9()
                            
                            ##@@FIXME: This os.system way, is very kluged and should be changed.
                            if   (commands.getstatusoutput('ps -ef | grep -v grep | grep ds9' )[0] > 0) \
                                 and (commands.getstatusoutput('ps -eA > .tmp; grep -q ds9 .tmp')[0] > 0):
                                print "CANNOT DISPLAY: No ds9 running."
                            else:
                                iraf.set(stdimage='imtgmos')
                                for tmpImage in rq.disList:
                                    if type(tmpImage) != str:
                                        #print "RED329:", tmpImage.filename
                                        tmpImage = tmpImage.filename

                                    # tmpImage should be a string at this point.
                                    #print "RED332:", type(tmpImage), tmpImage
                                    try:
                                        gemini.gmos.gdisplay( tmpImage, frameForDisplay, fl_imexam=iraf.no,
                                            Stdout = coi.getIrafStdout(), Stderr = coi.getIrafStderr() )
#                                        ds.display( tmpImage )
#                                        print ds.ds9.frames()
                                        frameForDisplay += 1    
                                    except:
                                        print "CANNOT DISPLAY"
                                        raise 
                        elif rqTyp == ImageQualityRequest:
                            #print 'RED394:'
                            filteredstdout.write(str(rq)+"\n", forceprefix = ("${NORMAL}${RED}","IQ reported to reduce: ", "${NORMAL}"))
                            #@@FIXME: All of this is kluge and will not remotely reflect how the 
                            # RecipeProcessor will deal with ImageQualityRequests.
                            if True:
                                #@@FIXME: Kluge to get this to work.
                                dispFrame = 0
                                if frameForDisplay > 0:
                                    dispFrame = frameForDisplay - 1

                                st = time.time()
                                if (useTK):
                                    iqlog = "%s: %s = %s\n"
                                    ell    = iqlog % (gemdate(timestamp=rq.timestamp),"mean ellipticity", rq.ellMean)
                                    seeing = iqlog % (gemdate(timestamp=rq.timestamp),"seeing", rq.fwhmMean)
                                    print ell
                                    print seeing
                                    timestr = gemdate(timestamp = rq.timestamp)
                                    
                                    cw.iqLog(co.inputs[0].filename, '', timestr)
                                    cw.iqLog("mean ellipticity", str(rq.ellMean), timestr)
                                    cw.iqLog("seeing", str(rq.fwhmMean)  , timestr)
                                    cw.iqLog('', '-'*14, timestr)
                                elif ds.ds9 is not None:
                                    dispText = 'fwhm=%s\nelli=%s\n' %( str(rq.fwhmMean), str(rq.ellMean) )
                                    ds.markText( 0, 2200, dispText )
                                
                                
                                else:    
                                # this was a kludge to mark the image with the metric 
                                # The following is annoying IRAF file methodology.
                                    tmpFilename = 'tmpfile.tmp'
                                    tmpFile = open( tmpFilename, 'w' )
                                    coords = '100 2100 fwhm=%(fwhm)s\n100 2050 elli=%(ell)s\n' %{'fwhm':str(rq.fwhmMean),
                                                                                     'ell':str(rq.ellMean)}
                                    tmpFile.write( coords )
                                    tmpFile.close()
                                    iraf.tvmark( frame=dispFrame,coords=tmpFilename,
                                    pointsize=0, color=204, label=pyraf.iraf.yes )
                                et = time.time()
                                #print 'RED422:', (et - st)


                    coi.clearRqs()      


                    #dump the reduction context object 
                    if options.rtf:
                        results = open( "test.result", "a" )
                        #results.write( "\t\t\t<< CONTROL LOOP " + str(controlLoopCounter" >>\n")
                        #print "\t\t\t<< CONTROL LOOP ", controlLoopCounter," >>\n"
                        #print "#" * 80
                        #controlLoopCounter += 1
                        results.write( str( coi ) )
                        results.close()
                        #print "#" * 80
                        #print "\t\t\t<< END CONTROL LOOP ", controlLoopCounter - 1," >>\n"
                        # CLEAR THE REQUEST LEAGUE
                    if primfilter == None:
                        raise "holy hell what's going on?"
                    # filteredstdout.addFilter(primfilter)
                # filteredstdout.removeFilter(primfilter)
        except KeyboardInterrupt:
            co.isFinished(True)
            if (useTK):
                cw.quit()
            co.persistCalIndex(calindfile)
            print "Ctrl-C Exit"
            sys.exit(0)
        except astrodata.ReductionObjects.ReductionExcept, e:
            print "${RED}FATAL:" + str(e) + "${NORMAL}"
            sys.exit()
        except:
            print "CONTEXT AFTER FATAL ERROR"
            print "--------------------------"
            co.persistCalIndex(calindfile)
            if (bReportHistory):
                co.reportHistory()
                rl.reportHistory()
            co.isFinished(True)
            if (useTK):
                cw.killed = True
                cw.quit()
            co.persistCalIndex(calindfile)
            raise
        co.persistCalIndex(calindfile)
    
        if (bReportHistory):
    
            print "CONTEXT HISTORY"
            print "---------------"
    
            co.reportHistory()
            rl.reportHistory()
            
        co.isFinished(True)
    
    if useTK and currentImageNum == numFiles:
        try:
            cw.done()
            cw.mainWindow.after_cancel(cw.pcqid)
            if True: #cw.killed == True:
                raw_input("Press Enter to Close Monitor Windows:")
            # After ID print cw.pcqid
            cw.mainWindow.quit()
        except:
            cw.mainWindow.quit()    
            raise
            
    if (generate_pycallgraphs):
        pycallgraph.make_dot_graph("recipman-callgraph.png")
    
    from time import sleep
    while (False):
        for th in threading.enumerate():
            print str(th)
        sleep(5.)
    # print co.reportHistory()
    # main()
    # don't leave the terminal in another color/mode, that's rude
    print "${NORMAL}"
