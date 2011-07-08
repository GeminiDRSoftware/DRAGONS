import re 
import os
import traceback
from astrodata import AstroData
from astrodata.adutils import gemLog
import inspect

log = None

class ReductionExcept:
    """ This is the general exception the classes and functions in the
    Structures.py module raise.
    """
    def __init__(self, msg="Exception Raised by ReductionObject"):
        """This constructor takes a message to print to the user."""
        self.message = msg
    def __str__(self):
        """This str conversion member returns the message given by the user (or the default message)
        when the exception is not caught."""
        return self.message
        
class IterationError(ReductionExcept):
    pass
    
class ReductionObject(object):

    recipeLib = None
    context = None
    # primDict is a dictionary of primitive sets keyed by astrodata type (a string)
    primDict = None
    curPrimType = None
    curPrimName = None
    funccommand_clause = None
    
    def __init__(self):
        self.primDict= {}
        global log
        if log==None:
                log = gemLog.getGeminiLog()
    
    def init(self, rc):
        """ This member is purely for overwriting.  Controllers should call this
        before iterating over the steps of the recipe"""
        self.context = rc
        return rc
    
    def execute_command_clause(self, rc):
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
        savedLocalparms = context.localparms
        context.status = "RUNNING"
        
        prevprimname = self.curPrimName
        self.curPrimName = primname
        # check to see current primitive set type is correct
        correctPrimType = self.recipeLib.discover_correct_prim_type(context)
        # will be NONE if there are no current inputs, maintain current
        # curPrimType
        if correctPrimType and correctPrimType != self.curPrimType:
            # print "RO98:", repr(correctPrimType), repr(self.curPrimType)
            newprimset  = self.recipeLib.retrieve_primitive_set(astrotype=correctPrimType)
            self.add_prim_set(newprimset)
            self.curPrimType = correctPrimType
        self.recipeLib.check_and_bind(self, primname, context=context) 
        # print "substeps(%s,%s)" % (primname, str(cfgobj))
        primset = self.get_prim_set(primname)
        if hasattr(primset, primname):
            prim = eval("primset.%s" % primname)
        else:
            msg = "There is no recipe or primitive named \"%s\" in  %s" % (primname, str(repr(self)))
            self.curPrimName = prevprimname
            raise ReductionExcept(msg)
        
        # set type of prim for logging
        log.changeIndent(indentLevel=context.indent)
        btype = primset.btype
        logstring = "%s: %s" % (btype,primname)
        if context.indent==0:
            # top-level recipe, add some extra demarcation
            log.status("="*80)
            spacer = "\n"+"="*80
        elif btype=="PRIMITIVE":
            spacer = "\n"+"-"*len(logstring)
        else:
            spacer = "\n"+"="*len(logstring)
        log.status(logstring + spacer)
                
        # primset init should perhaps be called ready
        # because it needs to be called each step because though
        # this primset may have been initted, it takes the context
        # which may have changed
        primset.init(context)
        context.parameter_collate(self.curPrimType, primset, primname)
        from RecipeManager import SettingFixedParam
        nonStandardStream = None
        if context["stream"] != None:
            # print "RO132: got stream arg", context["stream"]
            nonStandardStream = context.switch_stream(context["stream"])
        
        context.begin(primname)
        
        try:
            #2.6 feature
            #if inspect.isgeneratorfunction(prim):
            #    print "it's a primitive"
            #else:
            #    print "it's not a primitive"
            
            for rc in prim(context):
                # @@note: call the command clause callback here
                # @@note2: no, this yields and the command loop act that way
                # @@.....: and it is in run, which caps the yields which must
                # @@.....: call the command clause.\
                if rc == None:
                    raise ReductionExcept(
                            "Primitive '%s' returned None for rc on yield" % primname)
                if rc.is_finished():
                    break
                yield rc
        except SettingFixedParam, e:
            print "${RED}"+str(e)+"${NORMAL}"
        except TypeError,e:
            print 'Recieved TypeError: "%s" during iteration' % e
            msg = "The running primitive, '%s'." % primname
            raise # IterationError(msg)
        except:
            print "%(name)s failed due to an exception." %{'name':primname}
            raise
        context.curPrimName = None
        self.curPrimName = prevprimname
        #print "RO165:", repr(context.outputs)
        yield context.end(primname)
        #print "RO167:", repr(context.outputs)
        
        if nonStandardStream:
            context.restore_stream(from_stream = nonStandardStream)
            
        context.localparms = savedLocalparms
        if context.indent==0:
            # top-level recipe, add some extra demarcation
            log.changeIndent(indentLevel=context.indent)
            log.status("="*80)
        log.status("")
        yield context
        
    def runstep(self, primname, cfgobj):
        """runsetp(primitiveName, reductionContext)"""
        
        # this is just a blocking thunk to substeps which executes the command clause
        # @@NOTE: substeps does not execute the command clause because it yields to
        # @@..... a caller which either runs/calls it at the top of the loop.
        for cfg in self.substeps(primname, cfgobj):
            ## call command clause
            if cfg.is_finished():
                break
            self.execute_command_clause(cfg)
            if cfg.is_finished():
                break
            pass
        
        return cfg
    # run is alias for runstep
    run = runstep
    
    def register_command_clause(self, function):
        self.funccommand_clause = function
        
    def join_param_dicts(self, newprimset, primsetary):
        # make sure all paramDicts are the same object
        if len(primsetary)>0:
            paramdict0 = primsetary[0].param_dict
        else:
            paramdict0 = newprimset.param_dict
        for primset in primsetary:
            if primset.param_dict != paramdict0:
                raise ReductionExcept("ParamDict not coherent")
        paramdict0.update(newprimset.param_dict)
        newprimset.param_dict = paramdict0               
        
    def add_prim_set(self,primset):
        if type(primset) == list:
            for ps in primset:
                self.add_prim_set(ps)
            return
            
        if primset.astrotype == None:
            raise ReductionExcept("Primitive Set astrotype is None, fatal error, corrupt configuration")
        if primset.btype == "RECIPE":
            if hasattr(primset,"param_dict") and primset.param_dict != None:
                print repr(primset.param_dict)
                raise ReductionExcept("Primitive btype=RECIPE should not have a param_dict")
            primset.param_dict = {}
        if not self.primDict.has_key(primset.astrotype):
            self.primDict.update({primset.astrotype:[]})
        primset.ro = self
        primsetary = self.primDict[primset.astrotype]
        self.join_param_dicts(primset, primsetary)
        primsetary.append (primset)
    
    def get_prim_set(self, primname, astrotype = None):
        # print "RO110:", astrotype, self.curPrimType
        primsetary = self.primDict[self.curPrimType]
        # print "RO112:" , primsetary
        for primset in primsetary:
            if hasattr(primset, primname):
                return primset
        return None
        # draise ReductionExcept("No valid primset for type %s, primitive name: %s" % (self.curPrimType, primname)) 
        
        
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

from ReductionContextRecords import CalibrationRecord, StackableRecord, AstroDataRecord, FringeRecord
from astrodata.ReductionObjectRequests import CalibrationRequest,\
        UpdateStackableRequest, GetStackableRequest, DisplayRequest,\
        ImageQualityRequest
import Proxies
usePRS = True
prs = None

# !!! @@@ GET THESE SOME CLEANER WAY, HAVE REDUCE/CONTROL SYSTEMS set them!
adatadir = "./recipedata/"
calindfile = "./.reducecache/calindex.pkl"
stkindfile = "./.reducecache/stkindex.pkl"

def command_clause(ro, coi):
    global prs
    global log
    
    if log==None:
        log = gemLog.getGeminiLog()
        
    coi.process_cmd_req()
    while (coi.paused):
        time.sleep(.100)
    if coi.finished:
        return
    
    #process calibration requests
    for rq in coi.rorqs:
        rqTyp = type(rq)
        msg = 'REDUCE:\n'
        msg += '-'*30+'\n'
        if rqTyp == CalibrationRequest:
            fn = rq.filename
            typ = rq.caltype
            calname = coi.get_cal(fn, typ)
            # print "r399:", "handling calibrations"
            if calname == None:
                # Do the calibration search
                calurl = None
                if usePRS and prs == None:
                    # print "r454: getting prs"
                    prs = Proxies.PRSProxy.get_adcc()
                    
                if usePRS:
                    # print "RO316:", repr(rq)
                    calurl = prs.calibration_search( rq )
                
                # print "r396:", calurl
                if calurl == None:
                    log.critical('No '+str(typ)+' calibration file found for '+\
                                str(fn))
                    # this is not fatal because perhaps there isn't a calibration
                    # the system checks both the local and central source
                    # raise RecipeExcept("CALIBRATION for %s NOT FOUND, FATAL" % fn)
                    break

                msg += 'A suitable %s found:\n' %(str(typ))
                
                storenames = {"bias":"retrievedbiases",
                              "flat":"retrievedflats"
                              }
                calfname = os.path.join(coi[storenames[typ]], os.path.basename(calurl))
                if os.path.exists(calfname):
                    coi.add_cal(fn, typ, calfname)
                else:
                    coi.add_cal(fn, typ, AstroData(calurl, store=coi[storenames[typ]]).filename)
                coi.persist_cal_index()
                calname = calurl
            else:
                msg += '%s already stored.\n' %(str(typ))
                msg += 'Using:\n'

            msg += '%s%s%s' %( os.path.dirname(calname), os.path.sep, os.path.basename(calname))

            #print msg
            #print '-'*30

        elif rqTyp == UpdateStackableRequest:
            coi.stack_append(rq.stk_id, rq.stk_list, stkindfile)
            coi.persist_stk_index( stkindfile )
        elif rqTyp == GetStackableRequest:
            pass
            # Don't actually do anything, because this primitive allows the control system to
            #  retrieve the list from another resource, but reduce lets ReductionContext keep the
            # cache.
            #print "RD172: GET STACKABLE REQS:", rq
        elif rqTyp == DisplayRequest:
            # process display request
            nd = rq.to_nested_dicts()
            #print "r508:", repr(nd)
            if usePRS and prs == None:
                # print "r454: getting prs"
                prs = Proxies.PRSProxy.get_adcc()
            prs.display_request(nd)
                
                   
        elif rqTyp == ImageQualityRequest:
            # Logging returned Image Quality statistics
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
                    ell    = iqlog % (gemdate(timestamp=rq.timestamp),"mean ellipticity", rq.ell_mean)
                    seeing = iqlog % (gemdate(timestamp=rq.timestamp),"seeing", rq.fwhmMean)
                    log.status(ell)
                    log.status(seeing)
                    timestr = gemdate(timestamp = rq.timestamp)

                    cw.iq_log(co.inputs[0].filename, '', timestr)
                    cw.iq_log("mean ellipticity", str(rq.ell_mean), timestr)
                    cw.iq_log("seeing", str(rq.fwhmMean)  , timestr)
                    cw.iq_log('', '-'*14, timestr)
               
               # $$$ next three lines are commented out as the display server
               # $$$ is not in use anymore.
               # elif ds.ds9 is not None:
               #     dispText = 'fwhm=%s\nelli=%s\n' %( str(rq.fwhmMean), str(rq.ell_mean) )
               #     ds.markText( 0, 2200, dispText )


                else:    
                # this was a kludge to mark the image with the metric 
                # The following i)s annoying IRAF file methodology.
                    tmpFilename = 'tmpfile.tmp'
                    tmpFile = open( tmpFilename, 'w' )
                    coords = '100 2100 fwhm=%(fwhm)s\n100 2050 elli=%(ell)s\n' %{'fwhm':str(rq.fwhmMean),
                                                                     'ell':str(rq.ell_mean)}
                    tmpFile.write( coords )
                    tmpFile.close()
                    #print 'r165: importing iraf again'
                    import pyraf
                    from pyraf import iraf  
                    iraf.tvmark( frame=dispFrame,coords=tmpFilename,
                    pointsize=0, color=204, label=pyraf.iraf.yes )
                et = time.time()
                #print 'RED422:', (et - st)

    # note: will this throw away rq's, should throw exception?  review
    # why do this, better to assert it IS empty than empty it!
    # coi.clear_rqs()
    
    #dump the reduction context object 
    if coi['rtf']:
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
#    if primfilter == None:
#        raise "This is an error that should never happen, primfilter = None"
