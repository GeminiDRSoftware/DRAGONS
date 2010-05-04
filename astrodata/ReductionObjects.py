import re 
import os
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
        
class ReductionObject(object):

    recipeLib = None
    context = None
    # primDict is a dictionary of primitive sets keyed by astrodata type (a string)
    primDict = None
    curPrimType = None
    FUNCcommandClause = None
    
    def __init__(self):
        self.primDict= {}
    
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

    def substeps(self, primname, context):
        # check to see current primitive set type is correct
        correctPrimType = self.recipeLib.discoverCorrectPrimType(context)
        # will be NONE if there are no current inputs, maintain current
        # curPrimType
        if correctPrimType and correctPrimType != self.curPrimType:
            newprimset  = self.recipeLib.retrievePrimitiveSet(astrotype=correctPrimType)
            self.addPrimSet(newprimset)
            self.curPrimType = newprimset.astrotype
        self.recipeLib.checkAndBind(self, primname, context=context) 
        # print "substeps(%s,%s)" % (primname, str(cfgobj))
        primset = self.getPrimSet(primname)
        if hasattr(primset, primname):
            prim = eval("primset.%s" % primname)
        else:
            msg = "There is no recipe or primitive named \"%s\" in  %s" % (primname, str(repr(self)))
            raise ReductionExcept(msg)
                
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
        except:
            print "%(name)s failed due to an exception." %{'name':primname}
            raise
        context.curPrimName = None
        yield context.end(primname)
        
    def runstep(self, primname, cfgobj):
        cfgobj.status = "RUNNING"
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
        raise ReductionExcept("No valid primset for type %s, primitive name %s" % (self.curPrimType, primname)) 
        
        
class PrimitiveSet(object):
    ro = None
    astrotype = None
    btype = "PRIMSET"
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
                
                pdict = eval(d)
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
