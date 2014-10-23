#!/usr/bin/env python
#import time
#ost = time.time()
# ---PROFILER START to profile imports
#import hotshot
#importprof = hotshot.Profile("hotshot_edi_stats")
#------------------------------------------------------------------------------ 
try:
    #print "reduce IN BRANCH"
    from astrodata.adutils import logutils
    from optparse import OptionParser
    
    
    version = '1_0'

    # parsing the command line
    parser = OptionParser()
    parser.set_description("_"*11 + "Gemini Observatory Recipe System Processor"
                           " (v_1.0 2011)" + "_"*10 + " " + "_"*19 +\
                           "Written by GDPSG" + "_"*20)
    parser.set_usage( parser.get_usage()[:-1] + " file.fits\n" )

    parser.add_option("-c", "--paramfile", dest="paramfile", default=None,
                      help="specify parameter file")
    parser.add_option("-i", "--intelligence", dest='intelligence', default=False,
                      action="store_true", help="Endow recipe system with "
                      "intelligence to perform operations faster and smoother")
    parser.add_option("-m", "--monitor", dest="bMonitor", action="store_true",
                      default=False,
                      help="Open TkInter window to monitor progress of"
                      "execution (NOTE: One window will open per recipe run)")
    parser.add_option("-p", "--param", dest="userparam", default=None,
                      help="Set a parameter from the command line. The form '-p' "
                      "paramname=val' sets the parameter in the reduction context "
                      "such that all primitives will 'see' it.  The form: '-p "
                      "ASTROTYPE:primitivename:paramname=val', sets the parameter "
                      "such that it applies only when the current reduction type "
                      "(type of current reference image) is 'ASTROTYPE' and the "
                      "primitive is 'primitivename'. Multiple settings can appear "
                      "separated by commas, but no whitespace in the setting (i.e."
                      "'param=val,param2=val2', not 'param=val, param2=val2')")
    parser.add_option("--context", dest="running_contexts", default=None,
                        help="provides general 'context name' for primitives"
                        "sensitive to context.")
    parser.add_option("-r", "--recipe", dest="recipename", default=None,
                      help="specify which recipe to run by name")
    parser.add_option("-t", "--astrotype", dest="astrotype", default=None,
                      help="Run a recipe based on astrotype (either overrides the"
                      " default type, or begins without initial input (ex. "
                      "recipes that begin with primitives that acquire data))")
    ##@@FIXME: This next option should not be put into the package
    parser.add_option("-x", "--rtf-mode", dest="rtf", default=False, 
                      action="store_true", help="only used for rtf")
    parser.add_option("--throw_descriptor_exceptions", dest = "throwDescriptorExceptions", default=False,
                        action = "store_true", help="debug mode, throws exceptions when Descriptors fail")
    
    #parser.add_option("--addcal", dest="add_cal", default=None, type="string",
    #                  help="Add calibration. NOTE: won't work unless "
    #                  "'--caltype' is set AND will overwrite any existing "
    #                  "calibration in the index. (ex. reduce --addcal=N2"
    #                  "009_bias.fits --caltype=bias N20091002S0219.fits)")
    #parser.add_option("--clrcal", dest="clr_cal", default=False, 
    #                  action="store_true", help="remove all calibrations.")
    #parser.add_option("--remcal", dest="rem_cal", default=False, 
    #                  action="store_true", help="Remove calibration (of target)"
    #                  "from cache. NOTE: will not work unless --caltype is set."
    #                  " (ex. reduce --remcal --caltype=bias N20091002S0219.fits)")
    #parser.add_option("--caltype", dest="cal_type", default=None, type="string",
    #                  help="Calibration type. NOTE: only works with '--addcal' or "
    #                  "'--remcal' AND accepts only lowercase one word (ex. 'bias', "
    #                  "'twilight')")
    
    parser.add_option("--addprimset", dest="primsetname", default = None,
                      help="add user supplied primitives to reduction")
    parser.add_option("--calmgr", dest="cal_mgr", default=None, type="string",
                      help="calibration manager url (overides lookup table)")
    parser.add_option("--force-height", dest="forceHeight", default=None,
                      help="force height of terminal output")
    parser.add_option("--force-width", dest="forceWidth", default=None,
                      help="force width of terminal output")
    parser.add_option("--invoked", dest="invoked", default=False, 
                      action="store_true", help="tell user reduce invoked by adcc")
    parser.add_option("--logmode", dest="logmode", default="standard", 
                      type="string", help="Set logging mode (standard, "
                      "console, quiet, debug, null)")
    parser.add_option("--logfile", dest="logfile", default="reduce.log", 
                      type='string', help="name of log (default = 'reduce.log')") 
    parser.add_option("--loglevel", dest="loglevel", default="stdinfo", 
                      type="string", help="Set the verbose level for console "
                      "logging; (critical, error, warning, status, stdinfo, "
                      "fullinfo, debug)")
    parser.add_option("--showcolors", dest="show_colors", default=False, 
                      action="store_true", help="Shows available colors based "
                      "on terminal setting (used for debugging color issues)")
    # @@DEPRECATED remove --usercal flag below, was old name
    parser.add_option("--override_cal", "--usercal", dest="user_cals", default=None, type="string",
                      help="Add calibration to User Calibration Service of this format:"
                            "'--usercal=CALTYPE_1:CALFILEPATH_1,...,CALTYPE_N:CALFILEPATH_N', "
                            "EXAMPLE: --override_cal=processed_arc:/home/nzarate/wcal/gsN20011112S064_arc.fits ")
    parser.add_option("--writeInt", dest='writeInt', default=False, 
                      action="store_true", help="write intermediate outputs"
                      " (UNDER CONSTRUCTION)")   
    parser.add_option("--suffix", dest='suffix', default=None,
                        help="Suffix to add to filenames written at end of reduction.")    

    (options,  args) = parser.parse_args()
    
    # Configure logging, then instantiate the log
    if options.recipename == "USER":
        options.loglevel = "fullinfo"
    #if options.invoked:
    #    options.loglevel = "fullinfo"
    logutils.config(mode=options.logmode, console_lvl=options.loglevel, \
                     file_name=options.logfile)
    log = logutils.get_logger(__name__)

    import os
    import sys
    import traceback
    import commands
    import glob
    import subprocess
    import sys
    import time
    import re

    from datetime import datetime
    
    _show_times = False
    if _show_times:
        start_time = datetime.now()
        print "start time:%s" % start_time
    
    
    from astrodata.adutils import terminal
    from astrodata.adutils.terminal import TerminalController, ProgressBar 
    # start color printing filter for xgtermc
    REALSTDOUT = sys.stdout
    REALSTDERR = sys.stderr
    #filteredstdout = terminal.FilteredStdout()
    #filteredstdout.addFilter( terminal.ColorFilter())
    irafstdout = terminal.IrafStdout() #fout = filteredstdout)
    #sys.stdout = filteredstdout
    # sys.stderr = terminal.ColorStdout(REALSTDERR, term)
    #st = time.time()
    if False:
        try:
            import pyraf
            from pyraf import iraf
        except:
            print "reduce: didn't find pyraf"
    #et = time.time()
    #print 'IRAF TIME', (et-st)
    a = datetime.now()
    import astrodata
    from astrodata import RecipeManager
    from astrodata import Errors
    from astrodata.AstroData import AstroData
    from astrodata.AstroDataType import get_classification_library
    from astrodata.RecipeManager import ReductionContext
    from astrodata.RecipeManager import RecipeLibrary
    from astrodata.RecipeManager import RecipeError
    from astrodata.StackKeeper import StackKeeper
    from astrodata.ReductionObjectRequests import CalibrationRequest,\
            UpdateStackableRequest, GetStackableRequest, DisplayRequest,\
            ImageQualityRequest
    from astrodata import gdpgutil
    # from astrodata.LocalCalibrationService import CalibrationService
    # from astrodata.adutils.future import gemDisplay
    from astrodata.adutils import paramutil
    from astrodata.adutils.gemutil import gemdate
    from astrodata import Proxies
   
    from astrodata import Lookups
    #oet = time.time()
    #print 'TIME:', (oet -ost)
    b = datetime.now()

    from astrodata.usercalibrationservice import user_cal_service
    if options.user_cals:
        user_cals = options.user_cals.split(",")
        for user_cal in user_cals:
            ucary = user_cal.split(":")
            if len(ucary)>1:
                caltype = ucary[0]
                calname = ucary[1]
                user_cal_service.add_calibration(caltype, calname)
    # GLOBAL/CONSTANTS (could be exported to config file)
    cachedirs = [".reducecache",
                 "calibrations",
                 "calibrations/storedcals",
                 "calibrations/retrievedcals",
                 #".reducecache/storedcals/storedbiases",
                 #".reducecache/storedcals/storeddarks",
                 #".reducecache/storedcals/storedflats",
                 #".reducecache/storedcals/storedfringes",
                 #".reducecache/storedcals/retrievedbiases",
                 #".reducecache/storedcals/retrieveddarks",
                 #".reducecache/storedcals/retrievedflats",
                 #".reducecache/storedcals/retrievedfringes",
                 ]
    CALDIR = "calibrations/storedcals"
    # constructed below             
    cachedict = {} 
    for cachedir in cachedirs:
        if not os.path.exists(cachedir):                        
            os.mkdir(cachedir)
        cachename = os.path.basename(cachedir)
        if cachename[0] == ".":
            cachename = cachename[1:]
        cachedict.update({cachename:cachedir})

    # Testing
    import pyfits as pf

    useTK =  options.bMonitor
    # ------
    #$Id: recipeman.py,v 1.8 2008/08/05 03:28:06 callen Exp $
    from astrodata.tkMonitor import *

    adatadir = "./recipedata/"
    calindfile = "./.reducecache/calindex.pkl"
    stkindfile = "./.reducecache/stkindex.pkl"

    terminal.forceWidth = options.forceWidth
    terminal.forceHeight = options.forceHeight

    # do debug modes first
    if options.throwDescriptorExceptions:
        from astrodata.debugmodes import set_descriptor_throw
        set_descriptor_throw(True)
        
    if options.invoked:
        opener = "reduce started in adcc mode (--invoked)"
        log.fullinfo("."*len(opener))
        log.fullinfo(opener)
        log.fullinfo("."*len(opener))
        sys.stdout.flush()


    def abortBadParamfile(lines):
        for i in range(0,len(lines)):
            log.error("  %03d:%s" % (i, lines[i]))
        log.error("  %03d:<<stopped parsing due to error>>" % (i+1))
        sys.exit(1)

    def command_line():
        """
        This function is just here so that all the command line oriented parsing is one common location.
        Hopefully, this makes things look a little cleaner.
        """

        # this is done first because command line options can be set in the 
        # config file
        if options.paramfile:
            ups = []
            gparms = {}
            pfile = file(options.paramfile)
            astrotype = None
            primname = None
            cl = get_classification_library()

            i = 0
            lines = []
            for line in pfile:
                i += 1
                oline = line
                lines.append(oline)
                # strip comments
                line = re.sub("#.*?$", "", line)
                line = line.strip()

                # see if they are command options
                if len(line)>2 and line[:2] == "--":
                    #then it's an option

                    if "=" not in line:
                        opt = line

                        option = parser.get_option(opt)
                        # note, it would do to not assume flags mean a true dest value
                        val = "True"
                    else:
                        opt,val = line.split("=")

                    # print "r204:",opt, val
                    opt = opt.strip()
                    val = val.strip()

                    if opt == "--files":
                        files = val.split()
                        args.extend(files)
                    elif parser.has_option(opt):
                        option = parser.get_option(opt)
                        exec("options.%s=val" % str(option.dest))

                elif len(line)>0:
                    if "]" in line:
                        # then line is a header
                        name = re.sub("[\[\]]", "", line)
                        name = name.strip()
                        if len(name)== 0:
                            astrotype = None
                            primname = None
                        elif cl.is_name_of_type(name):
                            astrotype = name
                        else:
                            primname = name
                    else:
                        # not a section
                        keyval = line.split("=")
                        if len(keyval)<2:
                            log.error("$Badly formatted parameter file (%s)" \
                                  "\n  Line #%d: %s""" % (options.paramfile, i, oline))
                            abortBadParamfile(lines)
                            sys.exit(1)
                        key = keyval[0].strip()
                        val = keyval[1].strip()
                        if val[0] == "'" or val[0] == '"':
                            val = val[1:]
                        if val[-1] == "'" or val[-1] == '"':
                            val = val[0:-1]
                        if primname and not astrotype:
                            log.error("Badly formatted parameter file (%s)" \
                                  '\n  The primitive name is set to "%s", but the astrotype is not set' \
                                  "\n  Line #%d: %s" % (options.paramfile, primname, i, oline[:-1]))

                            abortBadParamfile(lines)
                        if not primname and astrotype:
                            log.error("Badly formatted parameter file (%s)" \
                                  '\n  The astrotype is set to "%s", but the primitive name is not set' \
                                  "\n  Line #%d: %s" % (options.paramfile, astrotype, i, oline))
                            abortBadParamfile(lines)
                        if not primname and not astrotype:
                            gparms.update({key:val})
                        else:
                            up = RecipeManager.UserParam(astrotype, primname, key, val)
                            ups.append(up)

            # parameter file ups and gparms                                
            pfups = ups
            pfgparms = gparms

        if  options.show_colors:
            print dir(filteredstdout.term)
            sys.exit(0)
        infile = None

        #if options.clr_cal:
        #    clrFile = None
        #
        #    co = ReductionContext()
        #    co.restore_cal_index(calindfile)
        #    co.calibrations = {}
        #    #co.persist_cal_index( calindfile )
        #    log.status("Calibration cache index cleared")
        #    import shutil

        #    if os.path.exists(CALDIR):
        #        shutil.rmtree(CALDIR)
        #    log.status("Calibration directory removed")
        #
        #    sys.exit(0)

        try:
            if len( args ) == 0 and options.astrotype == None:
                raise IndexError()
            infile   = args
        except IndexError:
            log.info("When no filename is given the astrotype must be specified"
                        "(-t or --astrotype). This is needed in order to load"
                        "correct recipes and primitive sets.")
            log.error("NO INPUT FILE or ASTROTYPE specified")
            log.info("type 'reduce -h' for usage information")
            sys.exit(1)


        input_files = []
        badList = []
        for inf in infile:

            #"""
            # checkImageParam allows the argument to be an @list, turns it
            # into a list of filenames as otherwise expected from the command line
            tmpInp = paramutil.checkImageParam(inf)
            if tmpInp == None:
                badList.append(inf)
            else:
                # extend the list of input files with contents of @ list
                input_files.extend(tmpInp)

        if len(badList) > 0:
            err = "\n\t".join(badList)
            log.error("Some files not found or can't be loaded:\n\t"+err)
            log.error("Exiting due to missing datasets.")
            if len(input_files ) > 0:
                found = "\n\t".join(input_files)
                log.info("These datasets were found and loaded:\n\t"+found)
            sys.exit(1)

        # print "r161:", input_files

        # OBSOLETE CALIBRATION override
        #if options.add_cal != None:
#            if options.cal_type == None:
#                print "Reduce requires a calibration type. Use --caltype. For more " + \
#                "information use '-h' or '--help'."
#                sys.exit(1)
#            elif not os.access(options.add_cal, os.R_OK):
#                print "'" + options.add_cal + "' does not exist or cannot be accessed."
#                sys.exit(1)
#
#            # @@TODO: Perhaps need a list of valid calibration types.
#            # @@TODO: Need testing if passed in calibration type is valid.
#
#            co = ReductionContext()
#
#            #@@CHECK WARNING WHY?
#            co.restore_cal_index(calindfile)
#
#            for arg in infile:
#                co.add_cal(AstroData(arg), options.cal_type, os.path.abspath(options.add_cal))
#
#            #co.persist_cal_index(calindfile)
#            print "'" + options.add_cal + "' was successfully added for '" + str(input_files) + "'."
#            if options.recipename == None:
#                sys.exit(1)
#
#        elif options.rem_cal:
#            if options.cal_type == None:
#                print "Reduce requires a calibration type. Use --caltype. For more " + \
#                "information use '-h' or '--help'."
#                sys.exit(1)
#
#            # @@TODO: Perhaps need a list of valid calibration types.
#            # @@TODO: Need testing if passed in calibration type is valid.
#
#            co = ReductionContext()
#            co.restore_cal_index(calindfile)
#            for arg in infile:
#                try:
#                    co.rm_cal( arg, options.cal_type )
#                except:
#                    print arg + ' had no ' + options.cal_type
#            print "'" + options.cal_type + "' was removed from '" + str(input_files) + "'."
#            #co.persist_cal_index(calindfile)
#            sys.exit(0)

        # parameters from command line and/or parameter file
        clups = []
        clgparms = {}
        pfups = []
        pfgparms = {}

        if options.userparam:
            # print "r451: user params", options.userparam
            ups = []
            gparms = {}
            allupstr = options.userparam
            allparams = allupstr.split(",")
            # print "r456:", repr(allparams)
            for upstr in allparams:
                # print "r458:", upstr
                tmp = upstr.split("=")
                spec = tmp[0].strip()
                # @@TODO: check and convert to correct type
                val = tmp[1].strip()
                
                if ":" in spec:
                    typ,prim,param = spec.split(":")
                    up = RecipeManager.UserParam(typ, prim, param, val)
                    ups.append(up)
                else:
                    up = RecipeManager.UserParam(None, None, spec, val)
                    ups.append(up)
                    gparms.update({spec:val})
            # command line ups and gparms
            clups = ups
            clgparms = gparms
        # print "r473:", repr(clgparms)
        fups = RecipeManager.UserParams()
        for up in clups:
            #print "r473:", up
            fups.add_user_param(up)
        for up in pfups:
            #print "r476:", up
            fups.add_user_param(up)
        options.user_params = fups
        options.globalParams = {}
        options.globalParams.update(clgparms)
        options.globalParams.update(pfgparms)

        return input_files

    #
    # START ADCC
    #
    from astrodata import Proxies
    # I think it best to start the adcc always, since it wants the reduceServer I prefer not
    # to provide to every component that wants to use the adcc as an active-library
    pradcc = datetime.now()
    if (_show_times):
        print "from start to start_adcc: %s" % (pradcc-start_time)
    
    pprox = Proxies.PRSProxy.get_adcc(check_once = True)
    if not pprox:
        adccpid = Proxies.start_adcc()
    
    afadcc = datetime.now()
    if (_show_times):
        print "time to execute start_adcc: %s %s %s" % (afadcc-pradcc,pradcc, afadcc)
    
    # launch xmlrpc interface for control and communication
    reduceServer = Proxies.ReduceServer()
    prs = Proxies.PRSProxy.get_adcc(reduce_server=reduceServer)

    usePRS = True

    
    
    
    
    # print "r395: usePRS=", usePRS

    # called once per substep (every yeild in any primitive when struck)
    # registered with the reduction object
    # !!!! we import this from ReductionObjects.py now
    from astrodata.ReductionObjects import command_clause

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

    try:
        allinputs = command_line()
    except:
        reduceServer.finished=True
        sys.stdout.flush()
        sys.stderr.flush()
        raise

    generate_pycallgraphs = False
    if (generate_pycallgraphs):
        import pycallgraph
        pycallgraph.start_trace()

    if options.intelligence:
        typeIndex = gdpgutil.cluster_by_groupid( allinputs )
        # If there was super intelligence, it would determine ordering. For now, it will 
        # run recipes in simple ordering, (i.e. the order values() is returned in).
        allinputs = typeIndex.values()
        #print "r507:", repr(allinputs)
    else:
        nl = []
        for inp in allinputs:
            try:
                ad = AstroData(inp)
                # renaming files to output (current) directory
                # helps to ensure we don't overwrite raw data (i.e. on auto-write at end)
                # todo: formalize these rules more
                ad.filename = os.path.basename(ad.filename)
                ad.mode = "readonly"

                nl.append(ad)
                ad = None #danger of accidentally using this!
            except:
                # note: should we raise an exception here?
                err = "Can't Load Dataset: %s" % inp
                log.warning(err)
                raise

        # note: this clause might be best placed elsewhere (earlier)
        if len(nl) == 0:
            log.warning("No files...")
            allinputs = [None]
        else:
            allinputs = [nl]


    #===============================================================================
    # Local PRS Components
    #===============================================================================
    # Local Calibration Service Setup
    # cs = CalibrationService() # is this used anymore, don't think so...

    # Local Display Service Setup
    # ds = gemDisplay.getDisplayService()

    numReductions = len(allinputs)
    i = 1
    log.info("About to process %d lists of datasets."% len(allinputs))
    #print "r554", repr(allinputs)
    for infiles in allinputs: #for dealing with multiple sets of files.
        #print "r232: profiling end"
        #prof.close()
        #raise "over"
        try:
            log.info("Starting Reduction on set #%d of %d" % (i, numReductions))
            if infiles:
                for infile in infiles:
                    log.info("    %s" % (infile.filename))
            currentReductionNum = i
            i += 1

            # get ReductionObject for this dataset
            #ro = rl.retrieve_reduction_object(astrotype="GMOS_IMAGE") 
            # can be done by filename
            #@@REFERENCEIMAGE: used to retrieve/build correct reduction object
            try:
                if (options.astrotype == None):
                    ro = rl.retrieve_reduction_object(infiles[0]) 
                else:
                    ro = rl.retrieve_reduction_object(astrotype = options.astrotype)
            except:
                reduceServer.finished=True
                try:
                    prs.unregister()
                except:
                    log.warning("Trouble unregistering from adcc shared services.")
                raise

            # add command clause
            if ro:
                ro.register_command_clause(command_clause)
            else:
                log.error("Unable to get ReductionObject for type %s" % options.astrotype)
                break
            if options.recipename == None:
                if options.astrotype == None:
                    reclist = rl.get_applicable_recipes(infiles[0]) #**
                    recdict = rl.get_applicable_recipes(infiles[0], collate=True) #**
                else:
                    reclist = rl.get_applicable_recipes(astrotype = options.astrotype,
                                                        prune=True)
                    recdict = rl.get_applicable_recipes(astrotype = options.astrotype,
                                                        prune=True, 
                                                        collate = True)
                #print "r599:",repr(reclist), repr(recdict)
            else:
                #force recipe
                reclist = [options.recipename]
                recdict = {"all": [options.recipename]}

            # @@REFERENCEIMAGE
            # first file in group is used as reference
            # for the types that are used to load the recipe and primitives

            if (options.astrotype == None):
                types = infiles[0].types
            else:
                types = [options.astrotype]

            infilenames = []
            if infiles:
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
            log.stdinfo(tb)
            log.stdinfo(title)
            log.stdinfo(tb)
            
            
            if options.recipename == None:
                if len(recdict) == 0:
                    msg = "No recipes found for types: "+repr(types)
                    #log.error(msg)
                    raise Errors.RecipeNotFoundError(msg)
                else:
                    log.info("Recipe(s) found by dataset type:")
            else:
                log.info("A recipe was specified:")

            for typ in recdict.keys():
                recs = recdict[typ]
                log.info("  for type: %s" % typ)
                for rec in recs:
                    log.info("    %s" % rec)

            bReportHistory = False
            cwlist = []
            if (useTK and currentReductionNum == 1):
                cw = TkRecipeControl(recipes = reclist)
                cw.start()

            if "USER" in reclist:
                interactiveMode = True
                import readline
                readline.set_history_length(100)
            else:
                interactiveMode = False

            # counts user given command for interactive mode
            cmdnum = 0 # @@INTERACTIVE
            co = None
            while True: # THIS IS A LOOP FOR INTERACTIVE USE! @@INTERACTIVE
                for rec in reclist:
                    if rec == "USER":
                        try:
                            rec = raw_input("reduce: ")
                            rec = rec.strip()
                            if rec == "exit":
                                interactiveMode = False
                                break
                            if rec.strip() == "":
                                continue
                            cmdnum += 1
                            rawrec = True
                            if rec == "reset":
                                co = None
                                continue
                        except:
                            interactiveMode = False
                            break
                    else:
                        rawrec = False

                    try:
                        if co == None or not interactiveMode:
                            #then we want to keep the 
                            # create fresh context object
                            # @@TODO:possible: see if deepcopy can do this better 
                            co = ReductionContext()
                            #set context(s)
                            if options.running_contexts:
                                cxs = options.running_contexts.split(":")
                            else:
                                cxs = []
                            co.setContext(cxs)
                            if options.rtf:
                                co.update({"rtf":True})
                            #print "r739:stack index file", stkindfile
                            # @@NAME: stackIndexFile, location for persistent stack list cache
                            co.set_cache_file("stackIndexFile", stkindfile)
                            co.ro = ro
                            # @@DOC: put cachedirs in context
                            for cachename in cachedict:
                                co.update({cachename:cachedict[cachename]})
                            co.update({"cachedict":cachedict})
                            # rc.["storedcals"] will be the proper directory

                            # co.restore_cal_index(calindfile)
                            # old local stack stuff co.restore_stk_index( stkindfile )

                            # add input files
                            if infiles:
                                #co.add_input(infiles)
                                co.populate_stream(infiles)
                            co.set_iraf_stdout(irafstdout)
                            co.set_iraf_stderr(irafstdout)

                           # odl way rl.retrieve_parameters(infile[0], co, rec)
                            if hasattr(options, "user_params"):
                                co.user_params = options.user_params
                            if hasattr(options, "globalParams"):
                                for pkey in options.globalParams.keys():
                                    co.update({pkey:options.globalParams[pkey]})

                        # Remove after write int works properly
                        if (options.writeInt == True):       
                                co.update({"writeInt":True})  

                        # Add the log level/name/mode to the global dict
                        co.update({'loglevel':options.loglevel})     
                        co.update({'logfile':options.logfile})       
                        co.update({'logmode':options.logmode})
                        co.update({'logindent':logutils.SW})

                        # Insert calibration url dictionary
                        # if given by command line will overide the lookup
                        calurldict = Lookups.get_lookup_table("Gemini/calurl_dict",
                                                              "calurl_dict")
                        if options.cal_mgr:
                            calmgr_str = options.cal_mgr
                            if calmgr_str[7:12] == 'local':
                                calurldict.update({'LOCALCALMGR' : calmgr_str})
                            else:
                                calurldict.update({'CALMGR' : calmgr_str})
                        co.update({'calurl_dict':calurldict})
                        #print "REDUCE 721", co.report(internal_dict=True)

                        if (useTK):
                            while cw.bReady == False:
                                # this is hopefully not really needed
                                # did it to give the tk thread a chance to get running
                                time.sleep(.1)
                            cw.new_control_window(rec,co)
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
                            ro.add_prim_set(userPrimSet)


                        if rawrec == False:
                            log.info( "running recipe: '%s'\n" % rec)

                        # logic to handle:
                        #  * recipes in config path somewhere
                        #  * filenames
                        #  * which need compiling due to arguments
                        if (os.path.exists(rec)):
                            if "recipe." not in rec:
                                raise "Recipe files must be named 'recipe.RECIPENAME'"
                            else:
                                rname = re.sub("recipe.", "", os.path.basename(rec))
                            rf = open(rec)
                            rsrc = rf.read()
                            prec = rl.compose_recipe(rname, rsrc)
                            rfunc = rl.compile_recipe(rname, prec)
                            ro = rl.bind_recipe(ro, rname, rfunc)
                            rec = rname
                        elif "(" in rec:
                            # print "r819:", rec
                            rsrc = rec
                            rname = "userCommand%d" % cmdnum
                            prec = rl.compose_recipe(rname, rsrc)
                            # log.debug(prec)
                            rfunc = rl.compile_recipe(rname, prec)
                            ro = rl.bind_recipe(ro, rname, rfunc)
                            rec = rname
                        else:
                            if options.astrotype:
                                rl.load_and_bind_recipe(ro, rec, astrotype=options.astrotype)
                            else:
                                rl.load_and_bind_recipe(ro,rec, dataset=infile[0])
                        if (useTK):
                            cw.running(rec)

                        #import meliae.scanner
                        #meliae.scanner.dump_all_objects("memory.prof")
                        #print "r858: memory test"
                        #sys.exit()
                        controlLoopCounter = 1
                        ################
                        # CONTROL LOOP #
                        ################
                        #print str(dir(TerminalController))
                        #@@COLOR primfilter = terminal.PrimitiveFilter()
                        primfilter = None
                        #@@COLOR filteredstdout.addFilter(primfilter)
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
                            #import cProfile
                            #cProfile.run("ro.run(rec, co)", "runout.prof")
                            #for coi in ro.substeps(rec, co):
                            #    ro.execute_command_clause()
                                # filteredstdout.addFilter(primfilter)
                            # filteredstdout.removeFilter(primfilter)
                        #######
                        #######
                        #######
                        #######
                        #######
                        #######
                    except KeyboardInterrupt:
                        co.is_finished(True)
                        if (useTK):
                            cw.quit()
                        #co.persist_cal_index(calindfile)
                        print "Ctrl-C Exit"
                        prs.unregister()
                        raise
                    except astrodata.ReductionObjects.ReductionError, e:
                        log.error("FATAL:" + str(e))
                        break;
                        #prs.unregister()
                        #sys.exit()
                    except:
                        f =  open("context.log", "w")
                        if co:
                            f.write(co.report(showall=True))
                        else:
                            f.write("rc null after exception, no report")
                        f.write(traceback.format_exc())
                        f.close()
                        log.fullinfo("------------------------------------------------")
                        log.fullinfo("Debug information written to context.log. Please")
                        log.fullinfo("provide this log when reporting this problem.")
                        log.fullinfo("------------------------------------------------")

                        if reduceServer:
                            #print "r855:", str(id(Proxies.reduceServer)), repr(Proxies.reduceServer.finished)
                            reduceServer.finished=True
                        #if co: co.persist_cal_index(calindfile)
                        if (bReportHistory):
                            if co: co.report_history()
                            rl.report_history()
                        if co: co.is_finished(True)
                        if (useTK):
                            cw.killed = True
                            cw.quit()
                        # co.persist_cal_index(calindfile)

                        # RAISE THE EXCEPTION AGAIN
                        if interactiveMode != True:
                            # note, I expect this raise to produce
                            # an exit and print of stack to user!
                            # which is why I unregister... interactive mode
                            # does not want to unregister while still
                            # looping
                            prs.unregister()
                            raise
                        else:
                            import traceback
                            traceback.print_exc()
                            print "\n Type 'exit' to exit."

                    
                    #co.persist_cal_index(calindfile)

                    if (bReportHistory):

                        log.error( "CONTEXT HISTORY")
                        log.error( "---------------")

                        co.report_history()
                        rl.report_history()

                    co.is_finished(True)

                    # write outputs
                    outputs = co.get_stream("main")
                    clobber = co["clobber"]
                    print
                    print "@RED L973 - clobber::", type(clobber), clobber
                    print
                    if clobber:
                        clobber = clobber.lower()
                        if clobber == "false":
                            clobber = False
                        else:
                            clobber = True
                    for output in outputs:
                        ad = output.ad
                        name = ad.filename
                        # print "r908:", ad.mode
                        try:
                            ad.write(clobber = clobber, suffix = options.suffix, rename=True)
                            log.stdinfo("Wrote %s in output directory" % ad.filename)
                        except Errors.OutputExists:
                            log.error( "CANNOT WRITE %s, already exists" % ad.filename)
                        except Errors.AstroDataReadonlyError, err:
                            log.warning('%s is in "readonly" mode, will not attempt to write.' % ad.filename)
                        except Errors.AstroDataError, err:
                            log.error("CANNOT WRITE %s: " % ad.filename + err.message)
                        except:
                            log.error("CANNOT WRITE %s, unknown reason" % ad.filename)
                            raise

                if interactiveMode == True:
                    reclist = ["USER"]
                else:
                    # print "r953: breaking from interactive loop"
                    break
                ### end of recipe iteration.
        except KeyboardInterrupt:
            log.error("Interrupted by Keyboard Interrupt")
            break
        except Errors.RecipeNotFoundError, rnf:
            log.error("Recipe not found for " + ",".join([ inp.filename 
                                                            for inp in infiles]))
            log.error(str(rnf))
        except RecipeError, x:
            # print "r995:", str(dir(x))
            traceback.print_exc()
            print "INSTRUCTION MIGHT BE A MISPELLED PRIMITIVE OR RECIPE NAME"
            msg = "name of recipe unknown" 
            if hasattr(x, "name"):
                msg = '"%s" is not a known recipe or primitive name' % x.name
            print "-"*len(msg)
            print msg
            print "-"*len(msg)
                        
        except:
            import traceback
            if infiles:
                log.error("PROBLEM WITH ONE SET OF FILES:\n\t%s \n%s"
                        %(",".join([inp.filename for inp in infiles]),
                             traceback.format_exc()))
            else:
                log.warning("No input files %s" % traceback.format_exc())
except SystemExit:
    log.error("SYSTEM EXIT: see log for more information")
    raise
except:
    import traceback as tb
    log.error("UNHANDLED ERROR, closing down reduce, traceback:\n"
                + tb.format_exc())
    
finally:
    if "reduceServer" not in globals():
        raise
    reduceServer.finished=True
    try:
        prs.unregister()
    except:
        log.warning("Trouble unregistering from adcc shared services.")
        raise
    
