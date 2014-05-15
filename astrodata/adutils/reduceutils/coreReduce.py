#
#                                                                  gemini_python
#
#                                                  astrodata.adutils.reduceutils
#                                                                  coreReduce.py
# ------------------------------------------------------------------------------
# $Id$
# ------------------------------------------------------------------------------
__version__      = '$Revision$'[11:-2]
__version_date__ = '$Date$'[7:-2]
# ------------------------------------------------------------------------------
# This will provide Reduce() as a class.
#
# class Reduce
# ------------------------------------------------------------------------------
import os
import re
import sys
import signal
import traceback
from   time import sleep

from astrodata import Proxies
from astrodata import Errors
from astrodata import Lookups

from astrodata.AstroData import AstroData

from astrodata.RecipeManager import RecipeLibrary
from astrodata.RecipeManager import RecipeExcept
from astrodata.RecipeManager import ReductionContext

from astrodata.ReductionObjects import ReductionExcept
from astrodata.ReductionObjects import command_clause

from astrodata.adutils import logutils
from astrodata.adutils.terminal import IrafStdout
from astrodata.usercalibrationservice import user_cal_service

from astrodata.gdpgutil   import cluster_by_groupid
from astrodata.debugmodes import set_descriptor_throw

import parseUtils

from caches import cachedirs
from caches import stkindfile
# ------------------------------------------------------------------------------
# start color printing filter for xgtermc
# useTK      = args.bMonitor
PKG_type   = "Gemini"       # moved out of lookup_table call
irafstdout = IrafStdout()   # fout = filteredstdout
# ------------------------------------------------------------------------------
log = logutils.get_logger(__name__)

# ------------------------------------------------------------------------------
def start_proxy_servers():
    adcc_proc = None
    pprox     = Proxies.PRSProxy.get_adcc(check_once=True)
    if not pprox:
        adcc_proc = Proxies.start_adcc()

    # launch xmlrpc interface for control and communication
    reduceServer = Proxies.ReduceServer()
    prs = Proxies.PRSProxy.get_adcc(reduce_server=reduceServer)
    return (adcc_proc, reduceServer, prs)

# ------------------------------------------------------------------------------
class Reduce(object):
    """
    The Reduce class encapsulates the core processing to be done by reduce.
    The constructor may receive one (1) parameter, which will be an instance
    of a parse_args call on a reduce-defined ArgumentParser object. As with
    all constructors, an instance of this class is returned.

    parameters: <instance>, optional ArgumentParser.parse_args() instance
    return:     <instance>, Reduce instance

    The class provides one (1) public method, runr(), the only call needed to
    run reduce on the supplied argument set.
    """
    def __init__(self, args=None):
        if args is None:
            args = parseUtils.buildParser(__version__).parse_args()

        self.rl  = None

        self.files = args.files
        self.reclist = None
        self.recdict = None
        self.infiles = None

        self.user_params  = None
        self.globalParams = None

        self.astrotype    = args.astrotype
        self.recipename   = args.recipename
        self.primsetname  = args.primsetname

        self.rtf       = args.rtf
        self.cal_mgr   = args.cal_mgr
        self.invoked   = args.invoked
        self.writeInt  = args.writeInt
        self.user_cals = args.user_cals

        self.logfile   = args.logfile
        self.logmode   = args.logmode
        self.loglevel  = args.loglevel
        self.logindent = logutils.SW

        upar, gpar = parseUtils.set_user_params(args.userparam)
        self.user_params  = upar
        self.globalParams = gpar

        self.intelligence = args.intelligence
        self.running_contexts = args.running_contexts
        self.throwDescriptorExceptions = args.throwDescriptorExceptions


    # ----------------------------------------------------------------------
    # configure the run space; execute recipe(s).
    def runr(self, command_clause=command_clause):
        """
        This the one (1) public method on class Reduce. It configures the
        run space for execution of reduce on an argument set supplied
        to the constructor or after a user has set a Reduce instance's
        attributes to appropriate values.

        If no user-specified recipe, i.e. args.recipename or self.recipename
        is None, nominal operation fetches applicable recipes and executes
        those recipes. Caller *may* pass a 'command_clause' function as a
        parameter to this method. But this is ill-advised unless the employment
        of a 'command_clause' like function is well understood.
        
        Nominally, a caller will call this method with no parameter, and
        the ReductionObjects defined command_clause() function will be used.

        parameters: <func>, function*, command_clause, from ReductionObjects
        return:     <void>

        * Presumably, other defined 'command_clause' functions could be used
        for the ReductionObjects command_clause() function. Currently,
        command_clause(), as defined in ReductionObjects.py, handles various
        requests made on the ReductionObject, i.e. the 'ro' instance:

        CalibrationRequest
        UpdateStackableRequest
        GetStackableRequest
        DisplayRequest
        ImageQualityRequest
        """
        red_msg = ("Unable to get ReductionObject for type %s" % self.astrotype)
        rec_msg = ("Recipe exception: Recipe not found")
        xstat   = 0

        def __shutdown_proxy(msg):
            if adcc_proc is None:
                log.stdinfo("ADCC is running externally. No proxies to close")
                return
            if adcc_proc.poll() is None:
                log.stdinfo("Force terminate adcc proxy ...")
                adcc_proc.send_signal(signal.SIGINT)
                adcc_exit = adcc_proc.wait()
            else:
                adcc_exit = adcc_proc.wait()

            log.stdinfo("  adcc terminated on status: %s" % str(adcc_exit))
            log.stdinfo(str(msg))
            return

        # AFTER validation checks, _check_files(), _convert_inputs(),
        # start proxy servers
        self._configure_run_space()
        try:
            valid_inputs = self._check_files()
        except IOError, err:
            xstat = signal.SIGIO
            log.error("IOError raised in __check_files()")
            log.error(str(err))
            return xstat

        allinputs   = self._convert_inputs(valid_inputs)
        nof_ad_sets = len(allinputs)

        try:
            adcc_proc, reduceServer, prs = start_proxy_servers()
        except Errors.ADCCCommunicationError, err:
            xstat = signal.SIGSYS
            log.error("ADCCCommunicationError raised in start_proxy_servers()")
            log.error(str(err))
            return xstat

        i = 0
        for infiles in allinputs:
            i += 1
            log.stdinfo("Starting Reduction on set #%d of %d" % (i, nof_ad_sets))
            title = "  Processing dataset(s):\n"
            title += "\n".join("\t" + ad.filename for ad in infiles)
            log.stdinfo("\n" + title + "\n")

            try:
                self._run_reduce(infiles)
            except KeyboardInterrupt:
                xstat = signal.SIGINT
                reduceServer.finished = True
                prs.registered = False
                log.error("runr() recieved event: SIGINT")
                log.error("Caught Ctrl-C event.")
                log.error("exit code: %d" % xstat)
                break
            except Errors.AstroDataError, err:
                xstat = signal.SIGABRT
                log.error(str(err))
                break                
            except IOError, err:
                xstat = signal.SIGIO
                log.error(str(err))
                break
            except Errors.RecipeNotFoundError, err:
                xstat = signal.SIGIO
                log.error(rec_msg)
                log.error(str(err))
            except RecipeExcept, err:
                xstat = signal.SIGIO
                log.error(rec_msg)
                log.error(str(err))
            except ReductionExcept, err:
                xstat = signal.SIGABRT
                log.error(red_msg)
                break
            except Errors.PrimitiveError, err:
                xstat = signal.SIGABRT
                log.error(err)
                break
            except Exception, err:
                xstat = signal.SIGQUIT
                log.error(str(err))
                log.error("PROBLEM ON PROCESSING ONE SET OF FILES:\n\t%s \n%s"
                          %(",".join([inp.filename for inp in infiles]),
                            traceback.format_exc()))
                break

        msg = "reduce terminated on status: %d" % xstat
        log.stdinfo("Shutting down proxy servers ...")
        reduceServer.finished = True
        if prs.registered:
            prs.unregister()
        sleep(1)
        __shutdown_proxy(msg)
        return xstat

    # ----------------------------- prive --------------------------------------
    def _configure_run_space(self):
        self._signal_invoked()
        self._add_cal_services()
        self.rl = RecipeLibrary()
        set_descriptor_throw(self.throwDescriptorExceptions)
        return

    def _signal_invoked(self):
        opener = "reduce started in adcc mode (--invoked)"
        if self.invoked:
            log.fullinfo("."*len(opener))
            log.fullinfo(opener)
            log.fullinfo("."*len(opener))
            sys.stdout.flush()
        return

    def _add_cal_services(self):
        """
        Add user calibration services to the global user_calibration_service
        namespace.
        
        parameter(s): <void>
        return:       <void>
        """
        # N.B. If a user_cal is passed that does not contain ':', i.e.
        # like CALTYPE:CALFILE, this phrase passes silently. Should it?
        if self.user_cals:
            for user_cal in self.user_cals:
                ucary = user_cal.split(":")
                if len(ucary)>1:
                    caltype = ucary[0]
                    calname = ucary[1]
                    user_cal_service.add_calibration(caltype, calname)
        return

    def _convert_inputs(self, inputs):
        if self.intelligence:
            typeIndex = cluster_by_groupid(inputs)
            # If super intelligence, it would determine ordering. Now, recipes in
            # simple order, (i.e. the order of values()).
            allinputs = typeIndex.values()
        else:
            nl = []
            for inp in inputs:
                try:
                    ad = AstroData(inp)
                    ad.filename = os.path.basename(ad.filename)
                    ad.mode = "readonly"
                except Errors.AstroDataError, err:
                    log.warning("Can't Load Dataset: %s" % inp)
                    log.warning(err)
                    continue
                except ValueError, err:
                    log.warning("Can't Load Dataset: %s" % inp)
                    log.warning(err)
                    continue
                nl.append(ad)
            try:
                assert(nl)
                allinputs = [nl]
            except AssertionError:
                msg = "No AstroData objects were created."
                log.warning(msg)
                raise IOError(msg)
        return allinputs


    def _check_files(self):
        """
        Sanity check on submitted files. Class version of the parseUtils function.
        
        parameters: <void>, other than instance
        return:     <list>, list of 'good' input fits datasets.
        """
        try:
            assert(self.files or self.astrotype)
        except AssertionError:
            log.info("Either file(s) OR an astrotype is required;"
                     "-t or --astrotype.")
            log.error("NO INPUT FILE or ASTROTYPE specified")
            log.stdinfo("type 'reduce -h' for usage information")
            raise IOError("NO INPUT FILE or ASTROTYPE specified")

        input_files = []
        bad_files   = []

        for image in self.files:
            if not os.access(image, os.R_OK):
                log.error('Cannot read file: '+str(image))
                bad_files.append(image)
            else:
                input_files.append(image)
        try:
            assert(bad_files)
            log.stdinfo("Got a badList ... %s" % bad_files)
            err = "\n\t".join(bad_files)
            log.error("Some files not found or cannot be loaded:\n\t%s" % err)
            try:
                assert(input_files)
                found = "\n\t".join(input_files)
                log.stdinfo("These datasets were found and loaded:\n\t%s" % found)
            except AssertionError:
                log.error("Caller passed no valid input files")
                raise IOError("No valid files passed.")
        except AssertionError:
            log.stdinfo("All submitted files appear valid")
        return input_files

    def _set_caches(self):
        cachedict = {}
        for cachedir in cachedirs:
            if not os.path.exists(cachedir):
                os.mkdir(cachedir)
            cachename = os.path.basename(cachedir)
            if cachename[0].startswith("."):
                cachename = cachename[1:]
            cachedict.update({cachename:cachedir})
        return cachedict

    def _run_reduce(self, infiles):
        """
        Run reduce on a passed list of input datasets. These datasets have been
        converted to AstroData objects and grouped by AstroDataType, via 
        _convert_inputs().

        parameters: <list>, list of astrodata instances
        return:     <void>
        """

        # If astrotype is None, first file in group is used as the type
        # reference to load the recipe and primitives, i.e. infiles[0]
        self.infiles  = infiles

        if self.astrotype:
            ro = self.rl.retrieve_reduction_object(astrotype=self.astrotype)
            types = [self.astrotype]
        else:
            ro = self.rl.retrieve_reduction_object(self.infiles[0])
            types = self.infiles[0].get_types()

        # add command clause
        ro.register_command_clause(command_clause)

        if self.recipename:
            self.reclist = [self.recipename]          # force user recipe
            recdict = {"all": [self.recipename]}
            log.info("A recipe was specified:")
        else:
            if self.astrotype:
                self.reclist = self.rl.get_applicable_recipes(astrotype=self.astrotype,
                                                              prune=True)
                recdict = self.rl.get_applicable_recipes(astrotype=self.astrotype,
                                                         prune=True, collate=True)
            else:
                self.reclist = self.rl.get_applicable_recipes(self.infiles[0])
                recdict = self.rl.get_applicable_recipes(self.infiles[0], collate=True)

        if recdict:
            log.info("Recipe(s) found by dataset type:")
            for typ, recs in recdict.items():
                log.info("  for type: %s" % typ)
                [log.info("    %s" % rec) for rec in recs]
        else:
            msg = "No recipes found for types: " + repr(types)
            log.error(msg)
            raise Errors.RecipeNotFoundError(msg)
            
        for recipe in self.reclist:
            self._exec_recipe(recipe, ro)
        
        return

    # exec_recipe
    def _exec_recipe(self, rec, ro):
        co = ReductionContext()
        co = self._configure_context(co, ro)
        ro.init(co)

        if self.primsetname:
            dr = os.path.abspath(os.path.dirname(self.primsetname))
            sys.path.append(dr)
            exec("import " + os.path.basename(self.primsetname)[:-3] + 
                 " as newmodule")
            userPrimSet = newmodule.userPrimSet
            userPrimSet.astrotype = ro.curPrimType
            ro.add_prim_set(userPrimSet)

        log.info( "running recipe: '%s'\n" % rec)

        if (os.path.exists(rec)):
            if rec.startswith("recipe."):
                rname = re.sub("recipe.", "", os.path.basename(rec))
            else:
                raise RecipeExcept("Recipe names must be like 'recipe.RECIPENAME'")

            rsrc  = open(rec).read()
            prec  = self.rl.compose_recipe(rname, rsrc)
            rfunc = self.rl.compile_recipe(rname, prec)
            ro    = self.rl.bind_recipe(ro, rname, rfunc)
            rec   = rname
        elif "(" in rec:
            rsrc = rec
            rname = "userCommand%d" % cmdnum
            prec  = self.rl.compose_recipe(rname, rsrc)
            rfunc = self.rl.compile_recipe(rname, prec)
            ro    = self.rl.bind_recipe(ro, rname, rfunc)
            rec   = rname
        else:
            try:
                if self.astrotype:
                    self.rl.load_and_bind_recipe(ro, rec, astrotype=self.astrotype)
                else:
                    self.rl.load_and_bind_recipe(ro, rec, dataset=self.infiles[0])
            except RecipeExcept, x:
                log.error("INSTRUCTION MAY BE A MISPELLED PRIMITIVE OR RECIPE NAME")
                msg = "name of recipe unknown"
                if hasattr(x, "name"):
                    msg = '"%s" is not a known recipe or primitive name' % x.name
                log.error("-"*len(msg))
                log.error(msg)
                log.error("-"*len(msg))
                raise RecipeExcept(msg)
        # ---------------------------------------------------- #
        # COMMAND LOOP
        # ---------------------------------------------------- #
        # not this only works because we install a stdout filter 
        # right away with this

        try:
            ro.run(rec, co)
        except KeyboardInterrupt:
            print "Caught a KeyboardInterrupt signal"
            print "Shutting down the Context object"
            co.is_finished(True)
            raise KeyboardInterrupt
        except Errors.PrimitiveError, err:
            self._write_context_log(co)
            co.is_finished(True)
            raise Errors.PrimitiveError(err)
        except Errors.InputError, err:
            self._write_context_log(co)
            co.is_finished(True)
            raise IOError(err)
        except Exception:
            self._write_context_log(co)
            co.is_finished(True)
            raise
        return


    def _configure_context(self, co, ro):
        co.ro = ro

        if self.running_contexts:
            cxs = self.running_contexts.split(":")
        else:
            cxs = []
        co.setContext(cxs)

        co.set_iraf_stdout(irafstdout)
        co.set_iraf_stderr(irafstdout)

        # Add the log level/name/mode to context obj.
        co.update({'logfile'  : self.logfile})
        co.update({'logmode'  : self.logmode})
        co.update({'loglevel' : self.loglevel})
        co.update({'logindent': self.logindent})

        co.set_cache_file("stackIndexFile", stkindfile)
        cachedict = self._set_caches()
        [co.update({name:path}) for (name, path) in cachedict.items()]
        co.update({"cachedict":cachedict})

        co.update({'rtf': self.rtf})
        co.update({"writeInt": self.writeInt})

        [co.update({key:val}) for (key, val) in self.globalParams.items()]
        if self.user_params:
            co.user_params = self.user_params

        # add input files to the stream
        if self.infiles:
            co.populate_stream(self.infiles)

        # Insert calibration url dictionary
        # command line overides the lookup
        table_path = os.path.join(PKG_type, 'calurl_dict')
        calurldict = Lookups.get_lookup_table(table_path, 'calurl_dict')

        if self.cal_mgr:
            calmgr_str = self.cal_mgr                        
            if calmgr_str[7:12] == 'local':
                calurldict.update({'LOCALCALMGR' : calmgr_str})
            else:
                calurldict.update({'CALMGR' : calmgr_str})

        co.update({'calurl_dict':calurldict})

        return co

    def _write_context_log(self, co=None, bReportHistory=False):
        """ Write a context report in the event of a non-specific
        exception.

        parameters: <ReductionContext object>, <bool>
        return:     <void>
        """
        co_log =  open("context.log", "w")
        if co:
            co_log.write(co.report(showall=True))
        else:
            co_log.write("rc null after exception, no report")

        co_log.write(traceback.format_exc())
        co_log.close()

        log.fullinfo("------------------------------------------------")
        log.fullinfo("Debug information written to context.log. Please")
        log.fullinfo("provide this log when reporting this problem.")
        log.fullinfo("------------------------------------------------")
        if (bReportHistory):
            if co:
                co.report_history()
            self.rl.report_history()
        if co: 
            co.is_finished(True)
        return
