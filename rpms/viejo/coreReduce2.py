#
#                                                                  gemini_python
#
#                                                                 coreReduce2.py
# ------------------------------------------------------------------------------
# $Id$
# ------------------------------------------------------------------------------
__version__      = '$Revision$'[11:-2]
__version_date__ = '$Date$'[7:-2]
# ------------------------------------------------------------------------------
# Provides reduce functionlity as a class, Reduce
#
# class Reduce
# ------------------------------------------------------------------------------
import os
import re
import sys
import signal
import inspect
import traceback
import imp
from time import sleep

from astrodata import AstroData
from astrodata.utils import Errors
from astrodata.utils import Lookups

from astrodata.utils import logutils
from astrodata.utils.terminal import IrafStdout
from astrodata.utils.gdpgutil import cluster_by_groupid
from astrodata.utils.debugmodes import set_descriptor_throw

from .recipeManager import RecipeLibrary
from .recipeManager import RecipeError
from .reductionObjects import ReductionError
from .reductionObjects import command_clause
from .reductionContext import ReductionContext

from ..adcc.servers import xmlrpc_proxy
from ..cal_service.usercalibrationservice import user_cal_service

import parseUtils

from .caches import cachedirs
from .caches import stkindfile
# ------------------------------------------------------------------------------
PKG_type   = "Gemini"       # moved out of lookup_table call
irafstdout = IrafStdout()   # fout = filteredstdout
# ------------------------------------------------------------------------------
log = logutils.get_logger(__name__)
# ------------------------------------------------------------------------------
def start_proxy_servers():
    adcc_proc = None
    pprox     = xmlrpc_proxy.PRSProxy.get_adcc(check_once=True)
    if not pprox:
        adcc_proc = xmlrpc_proxy.start_adcc()

    # launch xmlrpc interface for control and communication
    reduceServer = xmlrpc_proxy.ReduceServer()
    prs = xmlrpc_proxy.PRSProxy.get_adcc(reduce_server=reduceServer)
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
    def __init__(self, sys_args=None):
        if sys_args:
            args = sys_args
        elif self._confirm_args():
            args = parseUtils.buildParser(__version__).parse_args()
        else:
            args = parseUtils.buildParser(__version__).parse_args([])

        self.files = args.files
        self.infiles = None
        self.user_params  = None
        self.recipename   = args.recipename

        self.cal_mgr   = args.cal_mgr
        self.suffix    = args.suffix
        self.user_cals = args.user_cals

        self.logfile   = args.logfile
        self.logmode   = args.logmode
        self.loglevel  = args.loglevel
        self.logindent = logutils.SW

        self.running_contexts = args.running_contexts

    @property
    def userparam(self):
        return self._userparam

    @userparam.setter
    def userparam(self, uparam):
        self._userparam = uparam
        upar, gpar = parseUtils.set_user_params(self._userparam)
        self.user_params  = upar
        self.globalParams = gpar        
        return
    # The certain values indicated as 'False' in the above member set 
    # have been discontinued and removed from the reduce parser, i.e.
    # they are no longer available on the command line and values for
    # them are not returned by the buildParser() function. Some may
    # be implemented in future development, such as 'intelligence.'
    #
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
        red_msg = ("Unable to get ReductionObject")
        rec_msg = ("Recipe exception: Recipe not found")
        xstat   = 0

        def __shutdown_proxy(msg):
            if adcc_proc is None:
                log.stdinfo(str(msg))
                return

            if adcc_proc.poll() is None:
                adcc_proc.send_signal(signal.SIGINT)
                adcc_exit = adcc_proc.wait()
            else:
                adcc_exit = adcc_proc.wait()

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

        try:
            allinputs   = self._convert_inputs(valid_inputs)
            nof_ad_sets = len(allinputs)
        except IOError, err:
            xstat = signal.SIGIO
            log.error("IOError raised in _convert_inputs()")
            log.error(str(err))
            return xstat

        try:
            adcc_proc, reduceServer, prs = start_proxy_servers()
        except Errors.ADCCCommunicationError, err:
            xstat = signal.SIGSYS
            log.error("ADCCCommunicationError raised: start_proxy_servers()")
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
                self._run_reduce(infiles, prs)
            except KeyboardInterrupt:
                xstat = signal.SIGINT
                reduceServer.finished = True
                prs.registered = False
                log.error("runr() recieved event: SIGINT")
                log.error("Caught Ctrl-C event.")
                log.error("exit code: %d" % xstat)
                break
            except Errors.AstroDataError, err:
                xstat = signal.SIGQUIT
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
            except RecipeError, err:
                xstat = signal.SIGIO
                log.error(rec_msg)
                log.error(str(err))
            except ReductionError, err:
                xstat = signal.SIGILL
                log.error(red_msg)
                break
            except Errors.ReduceError, err:
                xstat = signal.SIGILL
                log.error(red_msg)
                break
            except Errors.PrimitiveError, err:
                xstat = signal.SIGILL
                log.error(str(err))
                break
            except Exception, err:
                xstat = signal.SIGQUIT
                log.error(type(err))
                log.error("PROBLEM ON PROCESSING ONE SET OF FILES:\n\t%s \n%s"
                          %(",".join([inp.filename for inp in infiles]),
                            traceback.format_exc()))
                break

        if xstat != 0:
            msg = "reduce instance aborted."
        else:
            msg = "\nreduce completed successfully."
        reduceServer.finished = True
        if prs.registered:
            log.stdinfo("Unregistering prsproxy ...")
            prs.unregister()
        sleep(1)
        __shutdown_proxy(msg)
        return xstat

    # ----------------------------- prive --------------------------------------
    def _confirm_args(self):
        """
        Confirm that the first executable frame in the call stack is a reduce 
        command line. This asserts that a nominal reduce parser, as returned by 
        buildParser() function, is an equivalent Namespace object to that
        of an 'args' key in the stack's 'f_locals' namespace. If the Namespace
        objects are not equal, reduce is not calling this class.

        :parameters: <void>
        :returns: Value of whether 'reduce' or some other executable is
                  instantiating this class.
        :rtype: <bool>
 
        """
        is_reduce = False
        exe_path = sys.argv[0]
        red_namespace = parseUtils.buildParser(__version__).parse_args([])
        if exe_path:
            cstack = inspect.stack()
            for local, value in cstack[-1][0].f_locals.items():
                if local == 'args':
                    try:
                        assert value.__dict__.keys() == red_namespace.__dict__.keys()
                        is_reduce = True
                    except AssertionError:
                        log.stdinfo("A non-reduce command line was detected.")
                        pass

        return is_reduce

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
            # super intelligence would determine ordering. Now, recipes in
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
                if not len(ad):
                    log.warning("%s contains no extensions." % ad.filename)
                    continue
                nl.append(ad)
            try:
                assert(nl)
                allinputs = [nl]
            except AssertionError:
                msg = "No AstroData objects can be processed."
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
            assert self.files
        except AssertionError:
            log.info("No file(s)")
            log.error("NO INPUT FILE specified")
            log.stdinfo("type 'reduce -h' for usage information")
            raise IOError("NO INPUT FILE specified")

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

    def _run_reduce(self, infiles, adcc):
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
        ro = self.rl.retrieve_reduction_object(self.infiles[0])
        # add command clause
        ro.register_command_clause(command_clause)

        if self.recipename:
            log.stdinfo("\tSpecified RECIPE:: {}\n\n".format(self.recipename))
            recdict = self.rl.get_applicable_recipe(self.infiles[0], 
                                                    recipe=self.recipename)
        else:
            recdict = self.rl.get_applicable_recipe(self.infiles[0])

        if recdict:
            log.info("Recipe(s) found by dataset type:")
            for typ, recs in recdict.items():
                log.info("type: %s" % typ)
        else:
            msg = "No recipe found for {}\n".format(self.infiles[0].filename)
            msg += "On TYPES: {}".format(self.infiles[0].type(prune=True))
            raise Errors.RecipeNotFoundError(msg)

        self._exec_recipe(recdict, ro, adcc)
        return

    # exec_recipe
    def _exec_recipe(self, recdict, ro, adcc):
        rec = recdict[recdict.keys()[0]][0]  # recipe name, 1st tuple element.
        co = ReductionContext()
        co = self._configure_context(co, ro)
        ro.init(co)

        if self.primsetname:
            dr = os.path.abspath(os.path.dirname(self.primsetname))
            modname = os.path.basename(self.primsetname)
            if modname.endswith('.py'):
                modname = modname[:-3]
            newmodule = imp.load_module('newmodule', *imp.find_module(modname, [dr]))
            userPrimSet = newmodule.userPrimSet
            userPrimSet.astrotype = ro.curPrimType
            ro.add_prim_set(userPrimSet)

        log.info( "recipe: '%s'\n" % rec)

        if (os.path.exists(rec)):
            if rec.startswith("recipe."):
                rname = re.sub("recipe.", "", os.path.basename(rec))
            else:
                raise RecipeError("Recipe names must be like 'recipe.RECIPENAME'")

            rsrc  = open(rec).read()
            rfunc = self.rl.compose_recipe(rname, rsrc)
            ro    = self.rl.bind_recipe(ro, rname, rfunc)
            rec   = rname
        elif "(" in rec:
            rsrc = rec
            rname = "userCommand%d" % cmdnum
            rfunc = self.rl.compose_recipe(rname, rsrc)
            ro    = self.rl.bind_recipe(ro, rname, rfunc)
            rec   = rname
        else:
            try:
                self.rl.load_and_bind_recipe(ro, rdict=recdict)
            except RecipeError, x:
                print x
                raise RecipeError(x)

        #raise RecipeError("TEST STOP")
        # ---------------------------------------------------- #
        # COMMAND LOOP
        # ---------------------------------------------------- #
        # not this only works because we install a stdout filter 
        # right away with this
        try:
            status = self._start_status()
            co.report_status(status)
            ml = co.get_metric_list(clear=True)
            ro.run(rec, co, reduce_status=ml)
            status = self._end_status(0)
            co.report_status(status)
            ml = co.get_metric_list(clear=True)
            if ml and adcc:
                adcc.report_qametrics(ml)
        except KeyboardInterrupt:
            log.error("Caught a KeyboardInterrupt signal")
            log.error( "Shutting down ReductionContext")
            status = self._end_status(signal.SIGINT)
            co.report_status(status)
            ml = co.get_metric_list(clear=True)
            if ml and adcc:
                adcc.report_qametrics(ml)
            co.is_finished(True)
            raise KeyboardInterrupt
        except ReductionError, err:
            status = self._end_status(signal.SIGILL)
            co.report_status(status)
            ml = co.get_metric_list(clear=True)
            if ml and adcc:
                adcc.report_qametrics(ml)
            self._write_context_log(co)
            co.is_finished(True)
            raise ReductionError(err)
        except Errors.ReduceError, err:
            status = self._end_status(signal.SIGILL)
            co.report_status(status)
            ml = co.get_metric_list(clear=True)
            if ml and adcc:
                adcc.report_qametrics(ml)
            self._write_context_log(co)
            co.is_finished(True)
            raise Errors.ReduceError(err)
        except Errors.PrimitiveError, err:
            status = self._end_status(signal.SIGILL)
            co.report_status(status)
            ml = co.get_metric_list(clear=True)
            if ml and adcc:
                adcc.report_qametrics(ml)
            self._write_context_log(co)
            co.is_finished(True)
            raise Errors.PrimitiveError(err)
        except Errors.InputError, err:
            status = self._end_status(signal.SIGIO)
            co.report_status(status)
            ml = co.get_metric_list(clear=True)
            if ml and adcc:
                adcc.report_qametrics(ml)
            self._write_context_log(co)
            co.is_finished(True)
            raise IOError(err)
        except Exception:
            status = self._end_status(signal.SIGQUIT)
            co.report_status(status)
            ml = co.get_metric_list(clear=True)
            if ml and adcc:
                adcc.report_qametrics(ml)
            self._write_context_log(co)
            co.is_finished(True)
            raise

        outputs = co.get_stream("main")
        clobber = co["clobber"]
        for output in outputs:
            ad = output.ad
            adname = ad._AstroData__origFilename
            origname = None
            user_name = None
            if self.suffix:
                origname, user_name = self._make_user_name(adname)
            try:
                if origname and user_name:
                    ad.write(filename=origname, clobber=clobber, suffix=self.suffix, 
                             rename=True)
                    log.stdinfo("Wrote %s in output directory" % user_name)
                else:
                    ad.write(filename=origname, clobber=clobber, suffix=self.suffix, 
                             rename=True)
                    log.stdinfo("Wrote %s in output directory" % ad.filename)
            except Errors.OutputExists:
                log.error( "%s exists. Will not write." % ad.filename)
            except Errors.AstroDataReadonlyError, err:
                log.warning('%s is "readonly". Will not write.' % ad.filename)
            except Errors.AstroDataError, err:
                log.error("CANNOT WRITE %s: " % ad.filename + err.message)
            except:
                log.error("CANNOT WRITE %s, unhandled exception." % ad.filename)
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

    def _make_user_name(self, adname):
        """
        Method recieves a passed AstroData filename, adname and the user-supplied
        suffix (command line --suffix flag), in order to strip off primitive
        suffixes and replace any and all of those with the final user suffix.
        The method must *assume* that processing has only appended underscored 
        suffixes to the original filename. This is a reasonable assumption for QAP 
        and astrodata_Gemini; primitives *only* appended suffixes like "_forStack", 
        "_addVAR", etc.. Other (future) packages are *not* guaranteed to do only 
        this.

        This should be considered a superficial, X1-only solution to the problem
        of the user supplied --suffix issue (See Trac #403 for discussion).
        The problem is properly addressed by AstroData and the write() method.
        Currently, AstroData.write() does not and cannot distinguish a
        suffix passed by primitives and other funtion calls  and one that may be 
        passed by reduce from the --suffix command line flag. Discussion in
        Trac #403 generally agrees that the user-supplied --suffix flag
        should *override* any and all other suffixes on the final output 
        file(s).

        :parameter adname: filename to convert with a user supplied suffix,
                           self.suffix.
        :type adname: <str>

        :returns: New filename with user suffix. This should be the original
                  input filename head + suffix + ext.
                  Eg., file_suffix.fits
        :rtype: <str>

        """
        origname = None
        usrname  = None
        adhead, adtail = os.path.splitext(adname)
        for infile in self.files:
            infile_base = os.path.basename(infile)
            inhead, intail = os.path.splitext(infile_base)
            if inhead in adhead:
                origname = infile_base
                usrname  = adname.replace(adhead, inhead + self.suffix)
                break
        return origname, usrname

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

    def _start_status(self):
        ad = self.infiles[0]
        log = self.logfile
        status = {"adinput": ad, "current": "Running", "logfile": log}
        return status

    def _end_status(self, xstat):
        ad = self.infiles[0]
        log = self.logfile
        if xstat:
            status = {"adinput": ad, "current": ".ERROR: {0}".format(xstat), 
                      "logfile": log}
        else:
           status = {"adinput": ad, "current": "Finished", "logfile": log}

        return status
