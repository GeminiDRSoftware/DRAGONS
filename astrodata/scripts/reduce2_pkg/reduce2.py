#!/usr/bin/env python
#
#                                                                     QAP Gemini
#
#                                                                     reduce2.py
#                                                               Kenneth Anderson
#                                                                        06-2013
#                                                            kanderso@gemini.edu
# ------------------------------------------------------------------------------

# $Id$
# ------------------------------------------------------------------------------
__version__      = '$Revision$'[11:-3]
__version_date__ = '$Date$'[7:-3]
__author__       = "k.r. anderson, <kanderso@gemini.edu>"
# ------------------------------------------------------------------------------
# reduce2.py -- refactored reduce, cl parsing exported, functionalized.
#               see parseUtils.py
# ------------------------------------------------------------------------------
__version__ = '1_0'

import sys
import os
import re
import traceback
from time import sleep        # use only sleep()
from datetime import datetime
from astrodata.adutils import logutils

# new module imports
import parseUtils

# ---------------------------------- parseUtils ------------------------------ #

parser = parseUtils.buildNewParser(__version__)
args   = parser.parse_args()
try: assert(args.astrotype)
except AssertionError: print "WARN:\tNo astrotype passed."
try: assert(args.files)
except AssertionError: print "WARN:\tNo files specified on command line."

from astrodata.adutils import logutils

# Configure logging, then instantiate the log
if args.recipename == "USER":
    args.loglevel = "fullinfo"

logutils.config(mode=args.logmode,
                console_lvl=args.loglevel,
                file_name=args.logfile)

log = logutils.get_logger(__name__)

try:
    allinputs = parseUtils.command_line(parser,args, log)
    print "POST All parsing ..."
    if args.displayflags:
        parseUtils.show_parser_options(parser,args)
    for inf in allinputs: print "Input fits file:\t",inf
except SystemExit:
    log.error("SYSTEM EXIT: see log for more information")
    raise
# ----------------------------- end parseUtils ------------------------------- #

try:
    _show_times = False
    if _show_times:
        start_time = datetime.now()
        print "start time:%s" % start_time

    import astrodata

    from astrodata import gdpgutil
    from astrodata import Proxies
    from astrodata import Lookups
    from astrodata import Errors
    from astrodata import RecipeManager

    from astrodata.AstroData     import AstroData
    from astrodata.RecipeManager import ReductionContext
    from astrodata.RecipeManager import RecipeLibrary
    from astrodata.RecipeManager import RecipeExcept
    from astrodata.adutils       import terminal 
    from astrodata.usercalibrationservice import user_cal_service

    from caches import cachedirs, stkindfile

    # start color printing filter for xgtermc

    REALSTDOUT = sys.stdout
    REALSTDERR = sys.stderr
    irafstdout = terminal.IrafStdout()     # fout = filteredstdout

    if args.user_cals:
        user_cals = args.user_cals.split(",")
        for user_cal in user_cals:
            ucary = user_cal.split(":")
            if len(ucary)>1:
                caltype = ucary[0]
                calname = ucary[1]
                user_cal_service.add_calibration(caltype, calname)

    cachedict = {} 
    for cachedir in cachedirs:
        if not os.path.exists(cachedir):                        
            os.mkdir(cachedir)
        cachename = os.path.basename(cachedir)
        if cachename[0] == ".":
            cachename = cachename[1:]
        cachedict.update({cachename:cachedir})

    # Testing
    from astrodata.tkMonitor import *
    useTK =  args.bMonitor
    terminal.forceWidth  = args.forceWidth
    terminal.forceHeight = args.forceHeight

    # do debug modes first
    if args.throwDescriptorExceptions:
        from astrodata.debugmodes import set_descriptor_throw
        set_descriptor_throw(True)
        
    if args.invoked:
        opener = "reduce started in adcc mode (--invoked)"
        log.fullinfo("."*len(opener))
        log.fullinfo(opener)
        log.fullinfo("."*len(opener))
        sys.stdout.flush()

    # ------------------------------------------------------------------------ #
    #                                START ADCC
    # ------------------------------------------------------------------------ #
    # Start adcc always, since it wants the reduceServer. Prefer not to provide 
    # to every component that wants to use the adcc as an active-library

    pradcc = datetime.now()
    if (_show_times):
        print "from start to start_adcc: %s" % (pradcc - start_time) 
    
    pprox = Proxies.PRSProxy.get_adcc(check_once=True)
    if not pprox:
        adccpid = Proxies.start_adcc()
    
    afadcc = datetime.now()
    if (_show_times):
        print "time to execute start_adcc: \t%s" % (afadcc - pradcc)
    
    # launch xmlrpc interface for control and communication
    reduceServer = Proxies.ReduceServer()
    prs = Proxies.PRSProxy.get_adcc(reduce_server=reduceServer)
    usePRS = True

    # called once per substep (every yeild in any primitive when struck)
    # registered with the reduction object
    # !!!! we import this from ReductionObjects.py now
    from astrodata.ReductionObjects import command_clause

    # ------------------------------------------------------------------------ #
    #                                 START SCRIPT
    # ------------------------------------------------------------------------ #
    generate_pycallgraphs = False
    if (generate_pycallgraphs):
        import pycallgraph
        pycallgraph.start_trace()

    if args.intelligence:
        typeIndex = gdpgutil.cluster_by_groupid(allinputs)
        # If super intelligence, it would determine ordering. Now, recipes in
        # simple order, (i.e. the order of values()).
        allinputs = typeIndex.values()
    else:
        nl = []
        for inp in allinputs:
            try:
                ad = AstroData(inp)
            except:
                # note: should we raise an exception here?
                err = "Can't Load Dataset: %s" % inp
                log.warning(err)
                continue
            nl.append(ad)
            ad = None #danger of accidentally using this!

        try:
            assert(nl)
            allinputs = [nl]               # Why is allinputs blown away here?
        except AssertionError:
            msg = "No AstroData objects were created."
            log.warning(msg)
            raise IOError, msg

    # ------------------------------------------------------------------------ #
    #                              Local PRS Components
    # ------------------------------------------------------------------------ #
    # get RecipeLibrary

    rl = RecipeLibrary()
    numReductions = len(allinputs)
    i = 1
    log.info("About to process %d lists of datasets."% len(allinputs))

    for infiles in allinputs:    #for dealing with multiple sets of files.
        try:
            log.info("Starting Reduction on set #%d of %d" % (i, numReductions))
            title = "  Processing dataset(s):\n"
            title += "\n".join("\t"+ad.filename for ad in infiles)
            log.stdinfo("\n"+title+"\n")

            currentReductionNum = i
            i += 1

            # @@REFERENCEIMAGE: retrieve/build correct reduction object
            # @@REFERENCEIMAGE
            # first file in group is used as reference
            # for the types that are used to load the recipe and primitives

            try:
                if not args.astrotype:
                    ro = rl.retrieve_reduction_object(infiles[0])
                    types = infiles[0].get_types()
                else:
                    ro = rl.retrieve_reduction_object(astrotype=args.astrotype)
                    types = [args.astrotype]
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
                log.error("Unable to get ReductionObject for type %s" % args.astrotype)
                break

            if not args.recipename:
                if not args.astrotype:
                    reclist = rl.get_applicable_recipes(infiles[0])
                    recdict = rl.get_applicable_recipes(infiles[0], collate=True)
                    if len(recdict) == 0:
                        msg = "No recipes found for types: "+repr(types)
                        #log.error(msg)
                        raise Errors.RecipeNotFoundError(msg)
                else:
                    reclist = rl.get_applicable_recipes(astrotype = args.astrotype,
                                                        prune=True)
                    recdict = rl.get_applicable_recipes(astrotype = args.astrotype,
                                                        prune=True, 
                                                        collate=True)
                    if recdict:
                        log.info("Recipe(s) found by dataset type:")
            else:
                # force user recipe
                reclist = [args.recipename]
                recdict = {"all": [args.recipename]}
                log.info("A recipe was specified:")

            for typ in recdict.keys():
                recs = recdict[typ]
                log.info("  for type: %s" % typ)
                for rec in recs:
                    log.info("    %s" % rec)

            bReportHistory = False

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
                            if args.running_contexts:
                                cxs = args.running_contexts.split(":")
                            else:
                                cxs = []
                            co.setContext(cxs)
                            if args.rtf:
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
                            if hasattr(args, "user_params"):
                                co.user_params = args.user_params
                            if hasattr(args, "globalParams"):
                                for pkey in args.globalParams.keys():
                                    co.update({pkey:args.globalParams[pkey]})

                        # Remove after write int works properly
                        if (args.writeInt == True):       
                                co.update({"writeInt":True})  

                        # Add the log level/name/mode to the global dict
                        co.update({'loglevel':args.loglevel})     
                        co.update({'logfile':args.logfile})       
                        co.update({'logmode':args.logmode})
                        co.update({'logindent':logutils.SW})

                        # Insert calibration url dictionary
                        # if given by command line will overide the lookup
                        calurldict = Lookups.get_lookup_table("Gemini/calurl_dict",
                                                              "calurl_dict")
                        if args.cal_mgr:
                            calmgr_str = args.cal_mgr                        
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

                        if args.primsetname != None:
                            dr = os.path.abspath(os.path.dirname(args.primsetname))
                            # print "r349:", dr
                            sys.path.append(dr)
                            # print "r351:", sys.path

                            exec ("import "+ os.path.basename(args.primsetname)[:-3] + " as newmodule")
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
                            if args.astrotype:
                                rl.load_and_bind_recipe(ro, rec, astrotype=args.astrotype)
                            else:
                                rl.load_and_bind_recipe(ro, rec, dataset=args.files[0])
                        if (useTK):
                            cw.running(rec)

                        controlLoopCounter = 1

                        # ---------------------------------------------------- #
                        # CONTROL LOOP #
                        # ---------------------------------------------------- #
                        primfilter = None
                        frameForDisplay = 1

                        # ---------------------------------------------------- #
                        # COMMAND LOOP
                        # ---------------------------------------------------- #
                        # not this only works because we install a stdout filter 
                        # right away with this

                        if (True):
                            ro.run(rec, co)
                        # ---------------------------------------------------- #

                    except KeyboardInterrupt:
                        co.is_finished(True)
                        if (useTK):
                            cw.quit()
                        #co.persist_cal_index(calindfile)
                        print "Ctrl-C Exit"
                        prs.unregister()
                        raise
                    except astrodata.ReductionObjects.ReductionExcept, e:
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
                            Proxies.reduceServer.finished=True

                        if (bReportHistory):
                            if co:
                                co.report_history()
                            rl.report_history()

                        if co:
                            co.is_finished(True)

                        if (useTK):
                            cw.killed = True
                            cw.quit()

                        # RAISE THE EXCEPTION AGAIN
                        # NOTE: I expect this raise to produce an exit and print
                        # stack to user! which is why I unregister... interactive
                        # mode does not want to unregister while still looping
                        if not interactiveMode:
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
                            ad.write(clobber = clobber, suffix = args.suffix, rename=True)
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
        except RecipeExcept, x:
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
        if False:
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

            while (False):
                from time import sleep
                for th in threading.enumerate():
                    print str(th)
                #sleep(5.)
            # print co.report_history()
            # main()
            # don't leave the terminal in another color/mode, that's rude
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
