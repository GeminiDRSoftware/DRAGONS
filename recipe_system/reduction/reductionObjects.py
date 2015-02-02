#
#                                                                  gemini_python
#
#
#                                                            ReductionObjects.py
# ------------------------------------------------------------------------------
# $Id: reductionObjects.py 5082 2014-12-17 14:49:22Z kanderson $
# ------------------------------------------------------------------------------
__version__      = '$Revision: 5082 $'[11:-2]
__version_date__ = '$Date: 2014-12-17 04:49:22 -1000 (Wed, 17 Dec 2014) $'[7:-2]
# ------------------------------------------------------------------------------
import re 
import os
import gc
import time
import urllib2           # get httperror
import traceback

import IDFactory

from astrodata import AstroData
from astrodata.utils import logutils

from .reductionObjectRequests import DisplayRequest
from .reductionObjectRequests import CalibrationRequest
from .reductionObjectRequests import GetStackableRequest
from .reductionObjectRequests import ImageQualityRequest
from .reductionObjectRequests import UpdateStackableRequest

from ..adcc.servers import xmlrpc_proxy
from ..cal_service.usercalibrationservice import user_cal_service

# ------------------------------------------------------------------------------
log = logutils.get_logger(__name__)

heap_dump = False
if heap_dump:
    #import heapy
    print "HEAPTRACKFILE CREATIONS: heapdump.log\n"*5
    heap_dump_file = open("heapdump.log", "w+")
else:
    heap_dump_file = None

usePRS = True
prs    = None

# !!! @@@ GET THESE SOME CLEANER WAY, HAVE REDUCE/CONTROL SYSTEMS set them!
adatadir = "./recipedata/"
calindfile = "./.reducecache/calindex.pkl"
stkindfile = "./.reducecache/stkindex.pkl"

# ------------------------------------------------------------------------------
def dump_heap(ro, fout):
    msg = ro.curPrimName
    fout.write("ro.curPrimName=%s\n" % ro.curPrimName)
    statusfile = open(os.path.join("/proc", str(os.getpid()), "status"))
    stats = statusfile.read()
    stats = stats.split(os.linesep)
    statdict = {}
    for statline in stats:
        stat = statline.split(":")
        if len(stat)>1:
            statdict.update({stat[0]:stat[1]})
    
    statnames = ["VmSize","VmRSS"]
    statrow = "\n".join(["%10s: %s" %( statname,statdict[statname])
                             for statname in statnames])+"\n"
    
    fout.write(statrow)
    fout.flush()
    return

# ------------------------------------------------------------------------------
class ReductionError(Exception):
    """
    General exception for ReductionObject troubles.
    """

class IterationError(ReductionError):
    pass

# ------------------------------------------------------------------------------    
class ReductionObject(object):
    context     = None
    primDic     = None          # dict of primitive sets keyed by astrodata type
    recipeLib   = None
    curPrimType = None
    curPrimName = None
    funccommand_clause = None
    primstype_order = None
    
    def __init__(self):
        self.primDict= {}
        self.primstype_order = []
    
    def init(self, rc):
        """ This member is purely for overwriting.  Controllers should call this
        before iterating over the steps of the recipe"""
        self.context = rc
        return rc
    
    def execute_command_clause(self, rc):
        if heap_dump_file:
            dump_heap(self, heap_dump_file)
        cmdclause = self.funccommand_clause
        if cmdclause:
            cmdclause(self, rc)
            
    def new_primitive_set(self, primtype = None, btype = "EXTERNAL"):
        a = PrimitiveSet()
        a.btype = "RECIPE"
        a.astrotype = primtype
        return a
     
    def parameter_prop(self, param, prop= "default"):
        if self.curPrimType not in self.primDict:
            return None
        prims = self.primDict[self.curPrimType]
        for prim in prims:
            if self.curPrimName in prim.param_dict:
                if ((param in prim.param_dict[self.curPrimName]) 
                    and 
                    (prop  in prim.param_dict[self.curPrimName][param])):
                    return prim.param_dict[self.curPrimName][param][prop]
        return None

        
    def parm_dict_by_tag(self, prim, tag):
        if self.curPrimType not in self.primDict:
            return {}
        primsets = self.primDict[self.curPrimType]
        retd = {}
        # for each primset assigned this type, check param_dict in order
        for primset in primsets:
            if prim in primset.param_dict:
                params = primset.param_dict[prim]
            else:
                continue
            for pkey in params.keys():
                param = params[pkey]
                
                include = False
                if (tag == "all"):
                    include = True
                elif "tags" in param and tag in param["tags"]:
                    include = True
                if include:    
                    retd.update({pkey:
                                 None if not "default" in param 
                                      else param["default"]})
        return retd
        
    def substeps(self, primname, context):
        self.curPrimName = primname
        prevprimname     = self.curPrimName
        savedLocalparms  = context.localparms
        context.status   = "RUNNING"

        # check to see current primitive set type is correct
        correctPrimType = self.recipeLib.discover_correct_prim_type(context)
        
        # will be NONE if no current inputs, maintain current curPrimType
        if correctPrimType and correctPrimType != self.curPrimType:
            # print "RO98:", repr(correctPrimType), repr(self.curPrimType)
            newprimset = self.recipeLib.retrieve_primitive_set(astrotype=correctPrimType)
            self.add_prim_set(newprimset, add_to_front=True)
            self.curPrimType = correctPrimType
        self.recipeLib.check_and_bind(self, primname, context=context) 
        
        primset = self.get_prim_set(primname)
        if hasattr(primset, primname):
            prim = eval("primset.%s" % primname)
        else:
            msg = "There is no recipe or primitive named \"%s\" in  %s" % \
                (primname, str(repr(self)))
            self.curPrimName = prevprimname
            raise ReductionError(msg)
        
        # set type of prim for logging
        btype = primset.btype
        logstring = "%s: %s" % (btype,primname)
        if context['index'] == None:
            # top-level recipe, set indent=0, add some extra demarcation
            logutils.update_indent(0, context['logmode'])
            context.update({'index':0})
            log.status("="*80)
            if primname.startswith('proxy'):
                log.debug(logstring)
            else:
                log.status(logstring)
                log.status("="*80)
        else:
            if btype=="RECIPE":
                # if it is a proxy recipe, log only at debug level,
                if primname.startswith('proxy'):
                    log.debug(logstring)
                else:
                    # increase the index
                    indx = context['index'] + 1
                    context.update({'index':indx})
                    logutils.update_indent(indx, context['logmode'])
                    log.status(logstring)
                    log.status("=" * len(logstring))
            else:
                indx = context['index'] + 1
                context.update({'index':indx})
                logutils.update_indent(indx, context['logmode'])
                log.status(logstring)
                log.status("-" * len(logstring))
                
        # primset init should perhaps be called ready
        # because it needs to be called each step because though
        # this primset may have been initted, it takes the context
        # which may have changed
        primset.init(context)
        context.parameter_collate(self.curPrimType, primset, primname)
        from recipeManager import SettingFixedParam
        nonStandardStream = None
        if context["stream"] != None:
            # print "RO132: got stream arg", context["stream"]
            nonStandardStream = context.switch_stream(context["stream"])
        
        context.begin(primname)
        
        try:
            for rc in prim(context):
                # @@note: call the command clause callback here
                # @@note2: no, this yields and the command loop act that way
                # @@.....: and it is in run, which caps the yields which must
                # @@.....: call the command clause.
                if rc == None:
                    msg = "Primitive '%s' returned None on rc yield\n" % primname
                    raise ReductionError(msg)
 
                rcmd = rc.pop_return_command()
                if rcmd == "return_from_recipe":
                    rc.terminate_primitive()
                    break
                if rcmd == "terminate_primitive":
                    break
                if rc.is_finished():
                    break
                yield rc
            gc.collect() # @@MEM
        except SettingFixedParam, e:
            print str(e)
        except TypeError,e:
            print 'Recieved TypeError: "%s" during iteration' % e
            msg = "The running primitive, '%s'." % primname
            raise # IterationError(msg)
        except:
            print "%(name)s failed due to an exception." %{'name':primname}
            logutils.update_indent(0, context['logmode'])
            raise
        context.curPrimName = None
        self.curPrimName = prevprimname

        yield context.end(primname)
        
        if nonStandardStream:
            context.restore_stream(from_stream = nonStandardStream)
            
        context.localparms = savedLocalparms
        if context['index'] == None:
            # top-level recipe, add some extra demarcation
            logutils.update_indent(0, context['logmode'])
            context.update({'index':0})
            log.status("="*80)
        else:
            if btype=="RECIPE":
                if not primname.startswith('proxy'):
                    indx = context['index'] - 1
                    context.update({'index':indx})
                    logutils.update_indent(indx, context['logmode'])
            else:
                log.status(".")
                indx = context['index'] - 1
                context.update({'index':indx})
                logutils.update_indent(indx, context['logmode'])

        yield context
        
    def runstep(self, primname, cfgobj):
        """
        runstep(primitiveName, reductionContext)
        """
        #print "RO275:", repr(cfgobj.inputs)
        # this is just a blocking thunk to substeps which executes the 
        # command clause.
        # @@NOTE: substeps does not execute the command clause because it 
        # yields to a caller which either runs/calls it at the top of the loop.
        for cfg in self.substeps(primname, cfgobj):
            ## call command clause
            if cfg.is_finished():
                break

            self.execute_command_clause(cfg)
            if cfg.is_finished():
                break
            pass
        return cfg

    # run is alias for runstep -- Why?
    run = runstep
    
    def register_command_clause(self, function):
        self.funccommand_clause = function
        return
        
    def join_param_dicts(self, newprimset, primsetary):
        # make sure all paramDicts are the same object
        if len(primsetary)>0:
            paramdict0 = primsetary[0].param_dict
        else:
            paramdict0 = newprimset.param_dict
        for primset in primsetary:
            if primset.param_dict != paramdict0:
                raise ReductionError("ParamDict not coherent")
        paramdict0.update(newprimset.param_dict)
        newprimset.param_dict = paramdict0
        return

    def add_prim_set(self,primset, add_to_front = False):
        if type(primset) == list:
            for ps in primset:
                self.add_prim_set(ps, add_to_front = add_to_front)
            return
            
        if primset.astrotype == None:
            raise ReductionError("Primitive Set astrotype is None, fatal error, corrupt configuration")
        if primset.btype == "RECIPE":
            if hasattr(primset,"param_dict") and primset.param_dict != None:
                print repr(primset.param_dict)
                raise ReductionError("Primitive btype=RECIPE should not have a param_dict")
            primset.param_dict = {}

        if not self.primDict.has_key(primset.astrotype):
            if add_to_front:
                self.primstype_order.insert(0,primset.astrotype)
            else:
                self.primstype_order.append(primset.astrotype)
            self.primDict.update({primset.astrotype:[]})

        primset.ro = self
        primsetary = self.primDict[primset.astrotype]
        self.join_param_dicts(primset, primsetary)
        if add_to_front:
            primsetary.insert(0,primset)
        else:
            primsetary.append (primset)
        return
    
    def get_prim_set(self, primname):
        primset = None
        if self.curPrimType != self.primstype_order[0]:
            print "RO355:", self.curPrimType, self.primstype_order
            raise ReductionError("curPrimType does not equal primstype_order[0], unexpected")
        for atype in self.primstype_order:
            primset =  self.get_prim_set_for_type(primname, astrotype = atype)
            if primset:
                break
        return primset

    def get_prim_set_for_type(self, primname, astrotype = None):
        # Get all possible types the primitive could be inherited from,
        # starting from the leaf node and working up the tree
        from astrodata.interface.AstroDataType import get_classification_library
        cl = get_classification_library()
        type_obj = cl.get_type_obj(astrotype)
        if type_obj is None:
            return None
        possible_types = type_obj.get_super_types(append_to=[type_obj])
        possible_types = [t.name for t in possible_types]

        for atype in possible_types:
            # If the primitive set has not been loaded, load it
            if atype not in self.primDict.keys():
                newprimset = self.recipeLib.retrieve_primitive_set(
                    astrotype=atype)
                self.add_prim_set(newprimset)

                # If it's still not there, raise an error
                if atype not in self.primDict.keys():
                    raise ReductionError("Could not add primitive set "\
                                          "for astrotype %s" % atype)

            # Get all the primitive sets for this type
            primsetary = self.primDict[atype]
            for primset in primsetary:
                # Check for the primitive
                if hasattr(primset, primname):
                    return primset
        return None
        
        
class PrimitiveSet(object):
    ro = None
    astrotype = None
    btype = "PRIMITIVE"
    filename = None
    directory = None
    param_dict = None
    def __init__(self):
        pass
        
    def init(self, context):
        return
    pthide_init = True
        
    def acquire_param_dict(self):
        # run through class hierarchy
        wpdict = {} # whole pdict, to return
        # print "RO134:"
        parlist = self.get_parent_modules(type(self),[])
        for parmod in parlist:
            # module names of this module and parents, in order
            # load the param_dict
            exec("import " + parmod)
            filename = eval(parmod +".__file__")
            # @@NAMING CONVENTION RELIANCE
            # make sure it's .py, not .pyc
            filename = re.sub(".pyc", ".py", filename)
            paramfile = re.sub("primitives_", "parameters_", filename)
            # print "RO144:", paramfile
            if os.path.exists(paramfile):
                # load and integrate
                f = file(paramfile)
                d = f.read()
                f.close()
                
                try:
                    pdict = eval(d)
                except:
                    pdict = {}
                    print "WARNING: can't load parameter dict in:", paramfile
                    traceback.format_exc()
                # this loop add all paramaters for all primnames which
                # do not already contain the parameter, giving precedence
                # to parameter dicts for leaf types, while inheriting
                # anything not overridden
                # @@NOTE: default parameters can be set for types which do
                # @@NOTE: .. not themselves have that primitive defined
                for primname in pdict.keys():
                    for param in pdict[primname]:
                        if primname not in wpdict:
                            wpdict.update({primname:{}})
                        if param not in wpdict[primname]:
                            
                            wpdict[primname].update({param:pdict[primname][param]})
        # to make version that returns this instead of set it, just return wpdict
        # but make this function call that one.                            
        self.param_dict = wpdict
        
    def get_parent_modules(self, cls, append_list):
        """This method returns a list of parent modules for primitives
        which can be used to mirror inheritance behavior in parameter
        dictionaries, used to store meta-data about parameters (such as
        their default value for the given primitive).  Note, the list
        is ordered from top level module down. Module name is used because
        consumers of this information use naming conventions to seek
        parameters for a given primtive set."""
        
        if "primitives_" in cls.__module__:
            append_list.append(cls.__module__)
            for bcls in cls.__bases__:
                bcls.get_parent_modules(self, bcls, append_list)
        return append_list


def command_clause(ro, coi):
    global prs
    global log
    
    if log==None:
        log = logutils.get_logger(__name__)
        
    coi.process_cmd_req()
    while (coi.paused):
        time.sleep(.100)
    if coi.finished:
        return
    
    ml = coi.get_metric_list(clear=True)
    prs = xmlrpc_proxy.PRSProxy.get_adcc(check_once=True)
    if ml and prs is not None:
        prs.report_qametrics(ml)
    
    #process  reques
    for rq in coi.rorqs:
        rqTyp = type(rq)
        msg = 'REDUCE:\n'
        msg += '-'*30+'\n'
        if rqTyp == CalibrationRequest:
            sci_ad = rq.ad
            if sci_ad:
                fn=sci_ad.filename
            else:
                fn = rq.filename
            typ = rq.caltype
            calname = None
            ## THIS IS THE CACHE CHECK, DISABLED NOW: calname = coi.get_cal(fn, typ)
            calmd5 = None
            if calname == None:
                # Do the calibration search
                calurl = None
                if usePRS and prs == None:
                    prs = xmlrpc_proxy.PRSProxy.get_adcc()
                    
                if usePRS:
                    try:
                        if user_cal_service:
                            calurl = user_cal_service.get_calibration(caltype = rq.caltype)
                            #print "488: calurl", repr(calurl)
                            #if calname:
                            #    return calname
                            if calurl and len(calurl) and calurl[0] == None:
                                log.warning(calurl[1])
                                calurl = None
                        if calurl == None:
                            # print "RO492", repr(rq)
                            calurl = prs.calibration_search( rq )
                            if calurl is not None:
                                if len(calurl) and calurl[0] == None:
                                    adcc_msg = calurl[1]                                    
                                    calurl = None
                                    log.error("CALIBRATION SERVICE REPORT:\n"*2)
                                    log.error(adcc_msg)
                                    log.error("END CAL SERVICE REPORT:\n"*2)
                                    calurl = None                                
                                else:                            
                                    calurl,calmd5 = calurl
                    except:
                        calurl = None
                        raise
                        
                if calurl == None:
                    log.warning('No '+str(typ)+' calibration file found for '+\
                                str(fn))
                    # this is not fatal because perhaps there isn't a calibration
                    # the system checks both the local and central source
                    # raise RecipeError("CALIBRATION for %s NOT FOUND, FATAL" % fn)
                    #break
                    continue
                log.info("found calibration (url): " + calurl)
                if calurl.startswith("file://"):
                    calfile = calurl[7:]
                    calurl = calfile
                else:
                    calfile = None
                
                #print "RO393:", calurl, calfile
                
                
                msg += 'A suitable %s found:\n' %(str(typ))
                
                useCached = False
                storenames = {"bias":"retrievedbiases",
                              "flat":"retrievedflats"
                              }
                
                if calfile:
                    calfname = os.path.basename(calfile)
                    caldname = os.path.dirname(calfile)
                    # print "ro517:[%s]\n%s (%s)" % (calfile, calfname, caldname)
                elif os.path.exists(calurl):
                    calfname = calurl
                    caldname = None
                else:
                    calfname = os.path.join(coi["retrievedcals"], typ, os.path.basename(calurl))
                    caldname = os.path.dirname(calfname)
                if caldname and not os.path.exists(caldname):
                    os.mkdir(caldname)
                # print "RO400:",calfname
                if os.path.exists(calfname) and caldname:
                    #coi.add_cal(fn, typ, calfname)
                    # check md5
                    ondiskmd5 = IDFactory.generate_md5_file(calfname)
                    if calmd5 == ondiskmd5:
                        log.stdinfo("File %s exists at calibration location, " \
                                "md5 checksums match, using cached copy." % os.path.basename(calfname))
                        useCached = True
                    else:
                        log.stdinfo("File %s exists at calibration location, " \
                                "but md5 checksums DO NOT MATCH, retrieving." % os.path.basename(calfname))
                    
                try:
                    if useCached:
                        ad = AstroData(calfname)
                    else:
                        ad = AstroData(calurl, store=caldname)
                except urllib2.HTTPError, error:
                    ad = None
                    errstr = "Could not retrieve %s" % calurl
                    log.error(errstr)
                    #@@TODO: should this raise? raise ReductionError(errstr)
                if ad:
                    coi.add_cal(sci_ad, typ, ad.filename)
            # adcc handles this now: coi.persist_cal_index()
                calname = calurl
            else:
                msg += '%s already stored.\n' %(str(typ))
                msg += 'Using:\n'

            msg += '%s%s%s' %( os.path.dirname(calname), os.path.sep, os.path.basename(calname))

        elif rqTyp == UpdateStackableRequest:
            coi.stack_append(rq.stk_id, rq.stk_list, stkindfile)
            coi.persist_stk_index( stkindfile )
        elif rqTyp == GetStackableRequest:
            pass

            # Do nothing; this primitive allows the control system to retrieve 
            # the list from another resource, but reduce lets ReductionContext 
            # keep the cache.
        elif rqTyp == DisplayRequest:
            # process display request
            nd = rq.to_nested_dicts()
            if usePRS and prs == None:
                prs = xmlrpc_proxy.PRSProxy.get_adcc()
            prs.display_request(nd)
                
                   
        elif rqTyp == ImageQualityRequest:
            # Logging returned Image Quality statistics
            raise "SHOULD NOT HAPPEN, OBSOLETE, to be removed"
            log.stdinfo(str(rq), category='IQ')
            log.stdinfo('-'*40, category='IQ')
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
                    ell = iqlog % (gemdate(timestamp=rq.timestamp),
                                   "mean ellipticity", rq.ell_mean)
                    seeing = iqlog % (gemdate(timestamp=rq.timestamp),
                                      "seeing", rq.fwhmMean)
                    log.status(ell)
                    log.status(seeing)
                    timestr = gemdate(timestamp = rq.timestamp)

                    cw.iq_log(co.inputs[0].filename, '', timestr)
                    cw.iq_log("mean ellipticity", str(rq.ell_mean), timestr)
                    cw.iq_log("seeing", str(rq.fwhmMean)  , timestr)
                    cw.iq_log('', '-'*14, timestr)
                else:    
                # this was a kludge to mark the image with the metric 
                # The following i)s annoying IRAF file methodology.
                    tmpFilename = 'tmpfile.tmp'
                    tmpFile = open( tmpFilename, 'w' )
                    coords = '100 2100 fwhm=%(fwhm)s\n100 2050 elli=%(ell)s\n' %{'fwhm':str(rq.fwhmMean),
                                                                     'ell':str(rq.ell_mean)}
                    tmpFile.write( coords )
                    tmpFile.close()
                    import pyraf
                    from pyraf import iraf  
                    iraf.tvmark( frame=dispFrame,coords=tmpFilename,
                                 pointsize=0, color=204, label=pyraf.iraf.yes )
                et = time.time()

    # note: will this throw away rq's, should throw exception?  review
    # why do this, better to assert it IS empty than empty it?
    coi.clear_rqs()
    
    #dump the reduction context object 
    if coi['rtf']:
        results = open( "test.result", "a" )
        results.write( str( coi ) )
        results.close()
    return
