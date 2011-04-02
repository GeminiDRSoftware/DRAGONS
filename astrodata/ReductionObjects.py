import re 
import os
import traceback
from astrodata import AstroData
from astrodata.adutils import gemLog

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
    FUNCcommandClause = None
    
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
    
    def executeCommandClause(self, rc):
        cmdclause = self.FUNCcommandClause
        if cmdclause:
            cmdclause(self, rc)
            
    def newPrimitiveSet(self, primtype = None, btype = "EXTERNAL"):
        a = PrimitiveSet()
        a.btype = "RECIPE"
        a.astrotype = primtype
        return a
     
    def parameterProp(self, param, prop= "default"):
        if self.curPrimType not in self.primDict:
            return None
        prims = self.primDict[self.curPrimType]
        for prim in prims:
            if self.curPrimName in prim.paramDict:
                if ((param in prim.paramDict[self.curPrimName]) 
                    and 
                    (prop  in prim.paramDict[self.curPrimName][param])):
                    return prim.paramDict[self.curPrimName][param][prop]
        return None
        
    def parmDictByTag(self, prim, tag):
        if self.curPrimType not in self.primDict:
            return {}
        primsets = self.primDict[self.curPrimType]
        retd = {}
        # for each primset assigned this type, check paramDict in order
        for primset in primsets:
            if prim in primset.paramDict:
                params = primset.paramDict[prim]
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
        correctPrimType = self.recipeLib.discoverCorrectPrimType(context)
        # will be NONE if there are no current inputs, maintain current
        # curPrimType
        if correctPrimType and correctPrimType != self.curPrimType:
            print "RO98:", repr(correctPrimType), repr(self.curPrimType)
            newprimset  = self.recipeLib.retrievePrimitiveSet(astrotype=correctPrimType)
            self.addPrimSet(newprimset)
            self.curPrimType = correctPrimType
        self.recipeLib.checkAndBind(self, primname, context=context) 
        # print "substeps(%s,%s)" % (primname, str(cfgobj))
        primset = self.getPrimSet(primname)
        if hasattr(primset, primname):
            prim = eval("primset.%s" % primname)
        else:
            msg = "There is no recipe or primitive named \"%s\" in  %s" % (primname, str(repr(self)))
            self.curPrimName = prevprimname
            raise ReductionExcept(msg)
        
        # set type of prim for logging
        btype = primset.btype
        log.status("STARTING %s: %s" % (btype,primname))
                
        context.begin(primname)
        # primset init should perhaps be called ready
        # because it needs to be called each step because though
        # this primset may have been initted, it takes the context
        # which may have changed
        primset.init(context)
        context.parameterCollate(self.curPrimType, primset, primname)
        from RecipeManager import SettingFixedParam
        try:
            for rc in prim(context):
                # @@note: call the command clause callback here
                # @@note2: no, this yields and the command loop act that way
                # @@.....: and it is in run, which caps the yields which must
                # @@.....: call the command clause.
                if rc.isFinished():
                    break
                yield rc
        except SettingFixedParam, e:
            print "${RED}"+str(e)+"${NORMAL}"
        except TypeError,e:
            print 'Recieved TypeError: "%s" during iteration' % e
            msg = "The running primitive, '%s', probably lacks 'yield rc'." % primname
            raise IterationError(msg)
        except:
            print "%(name)s failed due to an exception." %{'name':primname}
            raise
        context.curPrimName = None
        self.curPrimName = prevprimname
        yield context.end(primname)
        context.localparms = savedLocalparms
        log.status("ENDING %s: %s" % (btype, primname))
        yield context
        
    def runstep(self, primname, cfgobj):
        """runsetp(primitiveName, reductionContext)"""
        
        # this is just a blocking thunk to substeps which executes the command clause
        # @@NOTE: substeps does not execute the command clause because it yields to
        # @@..... a caller which either runs/calls it at the top of the loop.
        for cfg in self.substeps(primname, cfgobj):
            ## call command clause
            if cfg.isFinished():
                break
            self.executeCommandClause(cfg)
            if cfg.isFinished():
                break
            pass
        
        return cfg
    # run is alias for runstep
    run = runstep
    
    def registerCommandClause(self, function):
        self.FUNCcommandClause = function
        
    def joinParamDicts(self, newprimset, primsetary):
        # make sure all paramDicts are the same object
        if len(primsetary)>0:
            paramdict0 = primsetary[0].paramDict
        else:
            paramdict0 = newprimset.paramDict
        for primset in primsetary:
            if primset.paramDict != paramdict0:
                raise ReductionExcept("ParamDict not coherent")
        paramdict0.update(newprimset.paramDict)
        newprimset.paramDict = paramdict0               
        
    def addPrimSet(self,primset):
        if type(primset) == list:
            for ps in primset:
                self.addPrimSet(ps)
            return
            
        if primset.astrotype == None:
            raise ReductionExcept("Primitive Set astrotype is None, fatal error, corrupt configuration")
        if primset.btype == "RECIPE":
            if hasattr(primset,"paramDict") and primset.paramDict != None:
                print repr(primset.paramDict)
                raise ReductionExcept("Primitive btype=RECIPE should not have a paramDict")
            primset.paramDict = {}
        if not self.primDict.has_key(primset.astrotype):
            self.primDict.update({primset.astrotype:[]})
        primset.ro = self
        primsetary = self.primDict[primset.astrotype]
        self.joinParamDicts(primset, primsetary)
        primsetary.append (primset)
    
    def getPrimSet(self, primname, astrotype = None):
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
    paramDict = None
    def __init__(self):
        pass
        
    def init(self, context):
        return
    pthide_init = True
        
    def acquireParamDict(self):
        # run through class hierarchy
        wpdict = {} # whole pdict, to return
        # print "RO134:"
        parlist = self.getParentModules(type(self),[])
        for parmod in parlist:
            # module names of this module and parents, in order
            # load the paramDict
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
        self.paramDict = wpdict
        
    def getParentModules(self, cls, appendList):
        """This method returns a list of parent modules for primitives
        which can be used to mirror inheritance behavior in parameter
        dictionaries, used to store meta-data about parameters (such as
        their default value for the given primitive).  Note, the list
        is ordered from top level module down. Module name is used because
        consumers of this information use naming conventions to seek
        parameters for a given primtive set."""
        
        if "primitives_" in cls.__module__:
            appendList.append(cls.__module__)
            for bcls in cls.__bases__:
                bcls.getParentModules(self, bcls, appendList)
        return appendList

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

def commandClause(ro, coi):
    global prs
    global log
    
    if log==None:
        log = gemLog.getGeminiLog()
        
    coi.processCmdReq()
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
            calname = coi.getCal(fn, typ)
            # print "r399:", "handling calibrations"
            if calname == None:
                # Do the calibration search
                calurl = None
                if usePRS and prs == None:
                    # print "r454: getting prs"
                    prs = Proxies.PRSProxy.getADCC()
                    
                if usePRS:
                    print "RO316:", repr(rq)
                    calurl = prs.calibrationSearch( rq )
                
                # print "r396:", calurl
                if calurl == None:
                    log.critical('No '+str(typ)+' calibration file found for '+\
                                str(fn))
                    return None
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
                    coi.addCal(fn, typ, calfname)
                else:
                    coi.addCal(fn, typ, AstroData(calurl, store=coi[storenames[typ]]).filename)
                coi.persistCalIndex()
                calname = calurl
            else:
                msg += '%s already stored.\n' %(str(typ))
                msg += 'Using:\n'

            msg += '%s%s%s' %( os.path.dirname(calname), os.path.sep, os.path.basename(calname))

            #print msg
            #print '-'*30

        elif rqTyp == UpdateStackableRequest:
            coi.stackAppend(rq.stkID, rq.stkList, stkindfile)
            coi.persistStkIndex( stkindfile )
        elif rqTyp == GetStackableRequest:
            pass
            # Don't actually do anything, because this primitive allows the control system to
            #  retrieve the list from another resource, but reduce lets ReductionContext keep the
            # cache.
            #print "RD172: GET STACKABLE REQS:", rq
        elif rqTyp == DisplayRequest:
            # process display request
            nd = rq.toNestedDicts()
            #print "r508:", repr(nd)
            if usePRS and prs == None:
                # print "r454: getting prs"
                prs = Proxies.PRSProxy.getADCC()
            prs.displayRequest(nd)
                
                   
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
                    ell    = iqlog % (gemdate(timestamp=rq.timestamp),"mean ellipticity", rq.ellMean)
                    seeing = iqlog % (gemdate(timestamp=rq.timestamp),"seeing", rq.fwhmMean)
                    log.status(ell)
                    log.status(seeing)
                    timestr = gemdate(timestamp = rq.timestamp)

                    cw.iqLog(co.inputs[0].filename, '', timestr)
                    cw.iqLog("mean ellipticity", str(rq.ellMean), timestr)
                    cw.iqLog("seeing", str(rq.fwhmMean)  , timestr)
                    cw.iqLog('', '-'*14, timestr)
               
               # $$$ next three lines are commented out as the display server
               # $$$ is not in use anymore.
               # elif ds.ds9 is not None:
               #     dispText = 'fwhm=%s\nelli=%s\n' %( str(rq.fwhmMean), str(rq.ellMean) )
               #     ds.markText( 0, 2200, dispText )


                else:    
                # this was a kludge to mark the image with the metric 
                # The following i)s annoying IRAF file methodology.
                    tmpFilename = 'tmpfile.tmp'
                    tmpFile = open( tmpFilename, 'w' )
                    coords = '100 2100 fwhm=%(fwhm)s\n100 2050 elli=%(ell)s\n' %{'fwhm':str(rq.fwhmMean),
                                                                     'ell':str(rq.ellMean)}
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
    # coi.clearRqs()
    
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
