#!/usr/bin/env python
#import time
#ost = time.time()
# ---PROFILER START to profile imports
#import hotshot
#importprof = hotshot.Profile("hotshot_edi_stats")

#------------------------------------------------------------------------------ 
from adutils import terminal
from adutils.terminal import TerminalController, ProgressBar 
import sys
# start color printing filter for xgtermc
REALSTDOUT = sys.stdout
REALSTDERR = sys.stderr
filteredstdout = terminal.FilteredStdout()
filteredstdout.addFilter( terminal.ColorFilter())
irafstdout = terminal.IrafStdout(fout = filteredstdout)
sys.stdout = filteredstdout
# sys.stderr = terminal.ColorStdout(REALSTDERR, term)
import commands
from datetime import datetime
import glob
from optparse import OptionParser
import os
#st = time.time()
if True:
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
import re
#------------------------------------------------------------------------------ 
a = datetime.now()

import astrodata
from astrodata import RecipeManager
from astrodata.AstroData import AstroData
from astrodata.AstroDataType import getClassificationLibrary
from astrodata.RecipeManager import ReductionContext
from astrodata.RecipeManager import RecipeLibrary
from astrodata.StackKeeper import StackKeeper
from astrodata.ReductionObjectRequests import CalibrationRequest,\
        UpdateStackableRequest, GetStackableRequest, DisplayRequest,\
        ImageQualityRequest
from astrodata import gdpgutil
from astrodata.LocalCalibrationService import CalibrationService
from adutils.future import gemDisplay
from adutils import paramutil
from adutils.gemutil import gemdate
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
parser.add_option("-p", "--param", dest="userparam", default = None,
                    help="""Set a parameter from the command line.\
The form '-p paramname=val' sets the param in the reduction
context such that all primitives will 'see' it.  The 
form '-p ASTROTYPE:primitivename:paramname=val' sets the
parameter such that it applies only when
the current reduction type (type of current reference image)
is 'ASTROTYPE' and the primitive is 'primitivename'.
Multiple settings can appear separated by commas, but
no whitespace in the setting, i.e. 'param=val,param2=val2',
not 'param=val, param2=val2'.""")
parser.add_option("-f", "--paramfile", dest = "paramfile", default = None,
                    help="Specify a parameter file.")
parser.add_option("-t", "--astrotype", dest = "astrotype", default = None,
                    help="To run a recipe based on astrotype, either to override the default type of the file, or to start a recipe without initial input (i.e. which begin with primitives that acquire dta).")
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
parser.add_option("--force-width", dest = "forceWidth", default=None,
                  help="Use to force width of terminal for output purposes instead of using actual terminal width.")
parser.add_option("--force-height", dest = "forceHeight", default=None,
                  help="Use to force height of terminal for output purposes instead of using actual temrinal height.")
parser.add_option("--addprimset", dest = "primsetname", default = None,
                  help="Use to add user supplied primitives to the reduction object.")                  
(options,  args) = parser.parse_args()

useTK =  options.bMonitor
# ------
#$Id: recipeman.py,v 1.8 2008/08/05 03:28:06 callen Exp $
from astrodata.tkMonitor import *

adatadir = "./recipedata/"
calindfile = "./.reducecache/calindex.pkl"
stkindfile = "./.reducecache/stkindex.pkl"

terminal.forceWidth = options.forceWidth
terminal.forceHeight = options.forceHeight

def abortBadParamfile(lines):
    for i in range(0,len(lines)):
        print "  %03d:%s" % (i, lines[i]),
    print "  %03d:<<stopped parsing due to error>>" % (i+1)
    sys.exit(1)

def command_line():
    '''
    This function is just here so that all the command line oriented parsing is one common location.
    Hopefully, this makes things look a little cleaner.
    '''
    
    if  options.show_colors:
        print dir(filteredstdout.term)
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
        if len( args ) == 0 and options.astrotype == None:
            raise IndexError
        infile   = args
    except IndexError:
        print "${RED}NO INPUT FILE${NORMAL}"
        parser.print_help()
        sys.exit(1)
    
    input_files = []
    for inf in infile:
        #"""
        # checkImageParam allows the argument to be an @list, turns it
        # into a list of filenames as otherwise expected from the command line
        tmpInp = paramutil.checkImageParam( inf )
        if tmpInp == None:
            raise "The input had "+ str(inf)+" cannot be loaded."
        # extend the list of input files with contents of @ list
        input_files.extend( tmpInp )

    # print "r161:", input_files
        
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
        
    # parameters from command line and/or parameter file
    clups = []
    clgparms = {}
    pfups = []
    pfgparms = {}
    
    if options.userparam:
        ups = []
        gparms = {}
        allupstr = options.userparam
        allparams = allupstr.split(",")
        for upstr in allparams:
            tmp = upstr.split("=")
            spec = tmp[0].strip()
            # @@TODO: check and convert to correct type
            val = tmp[1].strip()
            
            if ":" in spec:
                typ,prim,param = spec.split(":")
                up = RecipeManager.UserParam(typ, prim, param, val)
                ups.append(up)
            else:
                gparms.update({spec:val})
        # command line ups and gparms
        clups = ups
        clgparms = gparms
        
    if options.paramfile:
        ups = []
        gparms = {}
        pfile = file(options.paramfile)
        astrotype = None
        primname = None
        cl = getClassificationLibrary()
        
        i = 0
        lines = []
        for line in pfile:
            i += 1
            oline = line
            lines.append(oline)
            # strip comments
            line = re.sub("#.*?$", "", line)
            line = line.strip()
            
            if len(line)>0:
                if "]" in line:
                    # then line is a header
                    name = re.sub("[\[\]]", "", line)
                    name = name.strip()
                    if len(name)== 0:
                        astrotype = None
                        primname = None
                    elif cl.isNameOfType(name):
                        astrotype = name
                    else:
                        primname = name
                else:
                    # not a section
                    keyval = line.split("=")
                    if len(keyval)<2:
                        print "${RED}Badly formatted parameter file (%s)${NORMAL}" \
                              "\n  Line #%d: %s""" % (options.paramfile, i, oline)
                        abortBadParamfile(lines)
                        sys.exit(1)
                    key = keyval[0].strip()
                    val = keyval[1].strip()
                    if val[0] == "'" or val[0] == '"':
                        val = val[1:]
                    if val[-1] == "'" or val[-1] == '"':
                        val = val[0:-1]
                    if primname and not astrotype:
                        print "${RED}Badly formatted parameter file (%s)${NORMAL}" \
                              '\n  The primitive name is set to "%s", but the astrotype is not set' \
                              "\n  Line #%d: %s" % (options.paramfile, primname, i, oline[:-1])
                        
                        abortBadParamfile(lines)
                    if not primname and astrotype:
                        print "${RED}Badly formatted parameter file (%s)${NORMAL}" \
                              '\n  The astrotype is set to "%s", but the primitive name is not set' \
                              "\n  Line #%d: %s" % (options.paramfile, astrotype, i, oline)
                        abortBadParamfile(lines)
                    if not primname and not astrotype:
                        gparms.update({key:val})
                    else:
                        up = RecipeManager.UserParam(astrotype, primname, key, val)
                        ups.append(up)
                        
        # parameter file ups and gparms                                
        pfups = ups
        pfgparms = gparms
    fups = RecipeManager.UserParams()
    for up in clups:
        fups.addUserParam(up)
    for up in pfups:
        fups.addUserParam(up)
    options.userParams = fups
    options.globalParams = {}
    options.globalParams.update(clgparms)
    options.globalParams.update(pfgparms)
       
            
    return input_files
    

# called once per substep (every yeild in any primitive when struck)
# registered with the reduction object
def commandClause(ro, coi):
    print "${NORMAL}",
    coi.processCmdReq()
    while (coi.paused):
        time.sleep(.100)
    if co.finished:
        return

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
                    break; # ignore
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
            print "${RED}DISPLAY REQUESTS NOT SUPPORTED AT THIS TIME.\nCall instrument specific display function in display primitive.${NORMAL}"
            raise "DISPLAY REQUESTS NOT SUPPORTED AT THIS TIME. Call instrument specific display function in display primitive."
            from pyraf import iraf
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
                    #print "RED
                    try:
                        # print "r420:", rq.disID, ds.displayID2frame(rq.disID)
                        raise "CANNOT DO DISPLAY REQUESTS AT THIS TIME. Call instrument specific display function in display primitive."
#                        gemini.gmos.gdisplay( tmpImage, ds.displayID2frame(rq.disID), fl_imexam=iraf.no,
#                            Stdout = coi.getIrafStdout(), Stderr = coi.getIrafStderr() )
#                                        ds.display( tmpImage )
#                                        print ds.ds9.frames()  
                    except:
                        print "CANNOT DISPLAY"
                        raise 
        elif rqTyp == ImageQualityRequest:
            #print 'RED394:'
            filteredstdout.write(str(rq)+"\n", forceprefix = ("${NORMAL}${RED}","IQ reported: ", "${NORMAL}"))
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


######################
######################
######################
# END MODULE FUNCTIONS
# START SCRIPT
######################
######################
######################

# get RecipeLibrary
rl = RecipeLibrary()

allinputs = command_line()

generate_pycallgraphs = False
if (generate_pycallgraphs):
    import pycallgraph
    pycallgraph.start_trace()

if options.intelligence:
    typeIndex = gdpgutil.clusterTypes( allinputs )
    # If there was super intelligence, it would determine ordering. For now, it will 
    # run recipes in simple ordering, (i.e. the order values() is returned in).
    allinputs = typeIndex.values()
else:
    nl = []
    for inp in allinputs:
        nl.append(AstroData(inp))
        
    allinputs = [nl]
    

#===============================================================================
# Local PRS Components
#===============================================================================
# Local Calibration Service Setup
cs = CalibrationService()

# Local Display Service Setup
ds = gemDisplay.getDisplayService()

numReductions = len(allinputs)
i = 1
for infiles in allinputs: #for dealing with multiple files.
    #print "r232: profiling end"
    #prof.close()
    #raise "over"
    
    print "${BOLD}Starting Reduction #%d of %d${NORMAL}" % (i, numReductions)
    for infile in infiles:
        print "    %s" % (infile.filename)
    currentReductionNum = i
    i += 1
    
    # get ReductionObject for this dataset
    #ro = rl.retrieveReductionObject(astrotype="GMOS_IMAGE") 
    # can be done by filename
    #@@REFERENCEIMAGE: used to retrieve/build correct reduction object
    if (options.astrotype == None):
        ro = rl.retrieveReductionObject(infile[0]) 
    else:
        ro = rl.retrieveReductionObject(astrotype = options.astrotype)
    
    # add command clause
    ro.registerCommandClause(commandClause)
        
    if options.recipename == None:
        if options.astrotype == None:
            reclist = rl.getApplicableRecipes(infiles[0]) #**
            recdict = rl.getApplicableRecipes(infiles[0], collate=True) #**
        else:
            reclist = rl.getApplicableRecipes(astrotype = options.astrotype)
            recdict = rl.getApplicableRecipes(astrotype = options.astrotype, collate = True)
    else:
        #force recipe
        reclist = [options.recipename]
        recdict = {"all": [options.recipename]}
    
    # @@REFERENCEIMAGE
    # first file in group is used as reference
    # for the types that are used to load the recipe and primitives
    
    if (options.astrotype == None):
        types = infiles[0].getTypes()
    else:
        types = [options.astrotype]
            
    infilenames = []
    for infs in infiles:
        if type(infs) == AstroData:
            infilenames.append( infs.filename )
        else:
            # I don't think this can happen now
            # where the input files are still strings at this point
            infilenames.append( infs )
            raise "not expected to happen"
       
    numi = len(infilenames) 

    if numi < 1:
        title = "  No Datasets  "
    elif numi == 1:        
        title = "  Processing dataset: %s  " % (str(infilenames[0])) #**
    else:
        title = "  Processing datasets:"
        for infiln in infilenames:
            title += "\n    %s" % infiln
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
    if (useTK and currentReductionNum == 1):
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
            co.addInput(infiles)
            co.setIrafStdout(irafstdout)
            co.setIrafStderr(irafstdout)
            
            # odl way rl.retrieveParameters(infile[0], co, rec)
            if hasattr(options, "userParams"):
                co.userParams = options.userParams
            if hasattr(options, "globalParams"):
                for pkey in options.globalParams.keys():
                    co.update({pkey:options.globalParams[pkey]})
            # print "r352:", repr(co.userParams.userParamDict)
            if (useTK):
                while cw.bReady == False:
                    # this is hopefully not really needed
                    # did it to give the tk thread a chance to get running
                    time.sleep(.1)
                cw.newControlWindow(rec,co)
                cw.mainWindow.protocol("WM_DELETE_WINDOW", co.finish) 
    
    
            # @@TODO:evaluate use of init for each recipe vs. for all recipes
            ro.init(co)
            if options.primsetname != None:
                dr = os.path.abspath(os.path.dirname(options.primsetname))
                # print "r349:", dr
                sys.path.append(dr)
                # print "r351:", sys.path
                
                exec ("import "+ os.path.basename(options.primsetname)[:-3] + " as newmodule")
                userPrimSet = newmodule.userPrimSet
                
                userPrimSet.astrotype = ro.curPrimType
                ro.addPrimSet(userPrimSet)
                
                
            print "running recipe: '%s'\n" % rec
            if (os.path.exists(rec)):
                if "recipe." not in rec:
                    raise "Recipe files must be named 'recipe.RECIPENAME'"
                else:
                    rname = re.sub("recipe.", "", os.path.basename(rec))
                rf = open(rec)
                rsrc = rf.read()
                prec = rl.composeRecipe(rname, rsrc)
                rfunc = rl.compileRecipe(rname, prec)
                ro = rl.bindRecipe(ro, rname, rfunc)
                rec = rname
            else:
                if options.astrotype:
                    rl.loadAndBindRecipe(ro, rec, astrotype=options.astrotype)
                else:
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
            #######
            #######
            #######
            #######
            ####### COMMAND LOOP
            #######
            #######
            #######
            # not this only works because we install a stdout filter right away with this
            # member function
            if (True): # try:
                ro.run(rec, co)
                #for coi in ro.substeps(rec, co):
                #    ro.executeCommandClause()
                    # filteredstdout.addFilter(primfilter)
                # filteredstdout.removeFilter(primfilter)
            #######
            #######
            #######
            #######
            #######
            #######
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
    
    if useTK and currentReductionNum == numReductions:
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
