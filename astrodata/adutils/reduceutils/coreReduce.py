#
#                                                                  gemini_python
#
#                                                                  coreReduce.py
# ------------------------------------------------------------------------------
# $Id$
# ------------------------------------------------------------------------------
__version__      = '$Rev$'[11:-3]
__version_date__ = '$Date$'[7:-3]
# ------------------------------------------------------------------------------
# coreReduce -- provides 
#
# class CoreReduce
# class ProxyInterface
# ------------------------------------------------------------------------------
import os
import re
import sys
import traceback

from astrodata import Errors
from astrodata import Lookups
from astrodata import RecipeManager

from astrodata.RecipeManager import ReductionContext
from astrodata.ReductionObjects import command_clause

from astrodata.adutils import logutils
from astrodata.adutils.terminal import IrafStdout
from astrodata.usercalibrationservice import user_cal_service

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

def set_caches():
    cachedict = {}
    for cachedir in cachedirs:
        if not os.path.exists(cachedir):                        
            os.mkdir(cachedir)
        cachename = os.path.basename(cachedir)
        if cachename[0].startswith("."):
            cachename = cachename[1:]
        cachedict.update({cachename:cachedir})
    return cachedict


def add_service(user_cals):
    """
    Add a user calibration service to the global user_calibration_service
    namespace.
    
    parameter(s): <str>, from args.user_cals
    return:       <void>
    """
    #N.B. If a user_cal is passed that does not contain a ':', i.e.
    # like CALTYPE:CALFILE, this phrase passes silently. Should it?
    if user_cals:
        for user_cal in user_cals:
            ucary = user_cal.split(":")
            if len(ucary)>1:
                caltype = ucary[0]
                calname = ucary[1]
                user_cal_service.add_calibration(caltype, calname)
    return

# ------------------------------------------------------------------------------
class CoreReduce(object):
    """
    The CoreReduce class encapsulates the core processing to be done by reduce.
    This comprises configuring the run space for a dataset list (infiles)
    passed to the constuctor, determining the appropriate recipe(s), and 
    executing the recipe(s) on the configured ReductionObject instance (ro).
    The dataset list (infiles) is a list of AstroData objects.

    parameters: <list>, <inst>, <log>, <RecipeLibrary>, <Proxy>
    return:     <instance>, CoreReduce instance.

    <list>          list of AstroData instances
    <inst>          Namespace or ProxyInterface instance
    <log>           logging.logger object used throughout reduce
    <RecipeLibrary> RecipeLibrary instance, the unfortunately named 'rl'
    <Proxy>         Proxy Server instance on the adcc.

    The class provides one (1) public method, runr(), the only call needed to
    run reduce on the supplied inputs and parameters.
    """
    def __init__(self, infiles, args, rl):
        self.infiles = infiles
        self.reclist = None
        self.recdict = None

        self.astrotype    = args.astrotype
        self.recipename   = args.recipename
        self.primsetname  = args.primsetname
        self.user_params  = None
        self.globalParams = None

        self.rtf       = args.rtf
        self.cal_mgr   = args.cal_mgr
        self.invoked   = args.invoked
        self.writeInt  = args.writeInt
        self.user_cals = args.user_cals

        self.logfile   = args.logfile
        self.logmode   = args.logmode
        self.loglevel  = args.loglevel
        self.logindent = args.logindent

        self.log = log
        self.rl  = rl

        if hasattr(args, "user_params"):
            self.user_params = args.user_params
        if hasattr(args, "globalParams"):
            self.globalParams = args.globalParams

        self.intelligence = args.intelligence
        self.running_contexts = args.running_contexts
        self.throwDescriptorExceptions = args.throwDescriptorExceptions


    # ----------------------------------------------------------------------
    # configure the run space; execute recipe(s).
    def runr(self, command_clause):
        """This method configures the run space for the input astrotypes.
        If no user-specified recipe, this fetches applicable recipes, 
        and executes those recipes. Caller passes a 'command_clause' function,
        nominally this will be command_clause() as defined by in
        ReductionObjects.

        parameters: <func>, function*, command_clause, from ReductionObjects
        return:     <void>

        * Presumably, other defined 'command_clause' functions could be substitued 
        for the ReductionObjects command_clause() function. Currently, 
        command_clause(), as defined in ReductionObjects.py, handles various
        requests made on the ReductionObject, i.e. the 'ro' instance:

        CalibrationRequest
        UpdateStackableRequest
        GetStackableRequest
        DisplayRequest
        ImageQualityRequest
        """
        # add any user calibration overrides ...
        add_service(self.user_cals)

        # if astrotype is unspecified, first file in group is type reference
        # for types used to load the recipe and primitives
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
            self.log.info("A recipe was specified:")
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
            self.log.info("Recipe(s) found by dataset type:")
            for typ, recs in recdict.items():
                self.log.info("  for type: %s" % typ)
                [self.log.info("    %s" % rec) for rec in recs]
        else:
            msg = "No recipes found for types: " + repr(types)
            self.log.error(msg)
            raise Errors.RecipeNotFoundError(msg)
            
        for recipe in self.reclist:
            self._exec_recipe(recipe, ro)
        
        return


    def write_context_log(self, co=None, rl=None, bReportHistory=False):
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
            rl.report_history()
        if co: 
            co.is_finished(True)
        return 

    # ----------------------------- prive --------------------------------------
    # exec_recipe functional

    def _exec_recipe(self, rec, ro):
        co = ReductionContext()
        co = self._configure_context(co, ro)

        #raise StopIteration, "Halting ..."
        #print "exec_recipe() @L313:", co.report(internal_dict=True)

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
                traceback.print_exc()
                print "INSTRUCTION MIGHT BE A MISPELLED PRIMITIVE OR RECIPE NAME"
                msg = "name of recipe unknown" 
                if hasattr(x, "name"):
                    msg = '"%s" is not a known recipe or primitive name' % x.name
                    print "-"*len(msg)
                    print msg
                    print "-"*len(msg)
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
        cachedict = set_caches()
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


class ProxyInterface(object):
    """
    A Proxy class to mimic an ArgumentParser Namespace instance. This allows
    a caller to programmatically spoof command line arguments without using
    the reduce command line interface or the need to instantiate an 
    ArgumentParser object. ProxyInterface provides the same set of arguments
    and defaults defined by the parseUtils.buildParser() function.

    The ProxyInterface class provides the same interface to a reduce 
    command line argument set, namely, argument values are accessed as 
    instance attributes. I.e.

    >>> args.astrotype
    'GMOS_IMAGE'
    >>> args.recipename
    'qareduce'

    The caller instantiates ProxyInterface() object, and then can set argument
    values as needed.

    Eg.,

    >>> args = ProxyInterface()
    >>> print args.recipename
    None
    >>> args.recipename = 'recipe.test_recipe'
    >>> args.astrotype = 'GMOS_SPECT'
    >>> print args.recipename
    recipe.test_recipe
    >>> print args.astrotype
    GMOS_SPECT

    The caller can then pass this args object to the reduce API. 
    (details to follow)
    """
    def __init__(self):
        self.files = []           # input filenames, <list>

        # types, recipes, primitives, contexts
        self.astrotype    = None
        self.recipename   = None
        self.primsetname  = None
        self.intelligence = False
        self.running_contexts =	None

        # calibration, parameters
        self.cal_mgr   = None
        self.user_cals = None
        self.userparam = None

        # files, modes, handles
        self.logfile   = 'reduce.log'
        self.loglevel  = 'stdinfo'
        self.logmode   = 'standard'
        self.logindent = 3
        self.suffix    = None

        self.forceWidth  = None
        self.forceHeight = None

        # runtime flags
        self.rtf      = False
        self.invoked  = False
        self.bMonitor = False
        self.writeInt = False
        self.throwDescriptorExceptions =  False
